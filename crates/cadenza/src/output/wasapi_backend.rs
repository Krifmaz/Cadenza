//! WASAPI exclusive-mode output (Windows): the device is taken over, the sample rate is
//! exactly what we ask for and integer samples reach the driver untouched.
//!
//! Event-driven exclusive mode: the device signals an event once per period and expects
//! exactly one period of frames in return, so `write` accumulates bytes and feeds the
//! device one period at a time, blocking on the event. This file is only compiled on
//! Windows (`cfg(windows)` + feature `wasapi`).

use super::{DeviceInfo, Output, OutputError, OutputFormat, STANDARD_RATES};
use std::collections::VecDeque;
use wasapi::{
    AudioClient, AudioRenderClient, Device, DeviceEnumerator, Direction, Handle, SampleType,
    StreamMode, WaveFormat,
};

/// See the module documentation.
pub struct WasapiOutput {
    device_name: String,
    device: Device,
    format: Option<OutputFormat>,
    stream: Option<Stream>,
}

struct Stream {
    client: AudioClient,
    render: AudioRenderClient,
    event: Handle,
    wave: WaveFormat,
    /// Frames per period (= the exclusive-mode buffer size).
    period_frames: usize,
    bytes_per_frame: usize,
    pending: VecDeque<u8>,
    started: bool,
    paused: bool,
}

// COM objects are used from the engine thread only.
unsafe impl Send for WasapiOutput {}

fn err<E: std::fmt::Display>(e: E) -> OutputError {
    OutputError::Backend(e.to_string())
}

fn init_com() {
    // S_FALSE ("already initialised on this thread") is fine; a real failure surfaces later.
    let _ = wasapi::initialize_mta();
}

fn find_device(name: &str) -> Result<Device, OutputError> {
    init_com();
    let enumerator = DeviceEnumerator::new().map_err(err)?;
    if name.is_empty() || name == "default" {
        return enumerator
            .get_default_device(&Direction::Render)
            .map_err(err);
    }
    enumerator
        .get_device_collection(&Direction::Render)
        .map_err(err)?
        .get_device_with_name(name)
        .map_err(|_| OutputError::NoDevice(name.to_string()))
}

/// 16-bit samples travel in 16-bit words; 24-bit in 32-bit words with 24 valid bits.
fn wave_format(format: OutputFormat) -> WaveFormat {
    let (store, valid) = match format.bits {
        16 => (16, 16),
        24 => (32, 24),
        _ => (32, 32),
    };
    WaveFormat::new(
        store,
        valid,
        &SampleType::Int,
        format.rate as usize,
        format.channels as usize,
        None,
    )
}

impl WasapiOutput {
    /// Prepare for `device` (friendly name; empty = default render endpoint).
    pub fn new(device: &str) -> Result<Self, OutputError> {
        let dev = find_device(device)?;
        let device_name = dev
            .get_friendlyname()
            .unwrap_or_else(|_| device.to_string());
        Ok(Self {
            device_name,
            device: dev,
            format: None,
            stream: None,
        })
    }
}

/// Every render endpoint with the exclusive-mode rates it accepts.
pub fn list_devices() -> Vec<DeviceInfo> {
    init_com();
    let Ok(enumerator) = DeviceEnumerator::new() else {
        return Vec::new();
    };
    let default_id = enumerator
        .get_default_device(&Direction::Render)
        .and_then(|d| d.get_id())
        .ok();
    let Ok(collection) = enumerator.get_device_collection(&Direction::Render) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for device in &collection {
        let Ok(device) = device else { continue };
        let Ok(name) = device.get_friendlyname() else {
            continue;
        };
        let Ok(client) = device.get_iaudioclient() else {
            continue;
        };
        let supports = |rate: u32, bits: u32| {
            client
                .is_supported_exclusive_with_quirks(&wave_format(OutputFormat {
                    rate,
                    bits,
                    channels: 2,
                }))
                .is_ok()
        };
        let rates: Vec<u32> = STANDARD_RATES
            .iter()
            .copied()
            .filter(|r| supports(*r, 24) || supports(*r, 16) || supports(*r, 32))
            .collect();
        let bits: Vec<u32> = [16u32, 24, 32]
            .into_iter()
            .filter(|b| supports(44100, *b) || supports(48000, *b))
            .collect();
        out.push(DeviceInfo {
            backend: "wasapi".into(),
            id: name.clone(),
            default: default_id.is_some() && device.get_id().ok() == default_id,
            name,
            rates,
            bits,
            channels: None,
            exclusive: true,
        });
    }
    out
}

impl Output for WasapiOutput {
    fn name(&self) -> String {
        format!("wasapi exclusive · {}", self.device_name)
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        self.close();
        let mut client = self.device.get_iaudioclient().map_err(err)?;
        let wave = client
            .is_supported_exclusive_with_quirks(&wave_format(format))
            .map_err(|_| OutputError::Unsupported {
                device: self.device_name.clone(),
                format,
            })?;
        // Three default periods (~30 ms) keeps USB DACs happy; HDA devices want 128-byte alignment.
        let (default_period, _min) = client.get_device_period().map_err(err)?;
        let period = client
            .calculate_aligned_period_near(3 * default_period, Some(128), &wave)
            .map_err(err)?;
        client
            .initialize_client(
                &wave,
                &Direction::Render,
                &StreamMode::EventsExclusive { period_hns: period },
            )
            .map_err(|e| OutputError::Backend(format!("initialize exclusive stream: {e}")))?;
        let event = client.set_get_eventhandle().map_err(err)?;
        let render = client.get_audiorenderclient().map_err(err)?;
        let period_frames = client.get_buffer_size().map_err(err)? as usize;
        let bytes_per_frame = wave.get_blockalign() as usize;
        self.stream = Some(Stream {
            client,
            render,
            event,
            wave,
            period_frames,
            bytes_per_frame,
            pending: VecDeque::new(),
            started: false,
            paused: false,
        });
        self.format = Some(format);
        Ok(format)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        if !matches!(bits, 16 | 24 | 32) {
            return false;
        }
        let Ok(client) = self.device.get_iaudioclient() else {
            return false;
        };
        client
            .is_supported_exclusive_with_quirks(&wave_format(OutputFormat {
                rate,
                bits,
                channels: 2,
            }))
            .is_ok()
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        let fmt = self
            .format
            .ok_or_else(|| OutputError::Backend("wasapi stream not open".into()))?;
        let s = self.stream.as_mut().unwrap();
        let store = s.wave.get_bitspersample();
        for &v in frames {
            match store {
                16 => s.pending.extend((v as i16).to_le_bytes()),
                // 24 valid bits in a 32-bit container must be left-justified (MSB aligned).
                _ if fmt.bits == 24 => s.pending.extend((v << 8).to_le_bytes()),
                _ => s.pending.extend(v.to_le_bytes()),
            }
        }
        let period_bytes = s.period_frames * s.bytes_per_frame;
        while s.pending.len() >= period_bytes {
            if s.started {
                s.event.wait_for_event(2000).map_err(err)?;
            }
            let chunk: Vec<u8> = s.pending.drain(..period_bytes).collect();
            s.render
                .write_to_device(s.period_frames, &chunk, None)
                .map_err(err)?;
            if !s.started {
                // The buffer is primed; start the clock.
                s.client.start_stream().map_err(err)?;
                s.started = true;
            }
        }
        Ok(())
    }

    fn pause(&mut self, paused: bool) -> Result<(), OutputError> {
        if let Some(s) = &mut self.stream {
            if paused && s.started {
                s.client.stop_stream().map_err(err)?;
            } else if !paused && s.started && s.paused {
                s.client.start_stream().map_err(err)?;
            }
            s.paused = paused;
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), OutputError> {
        if let Some(s) = &mut self.stream {
            if !s.pending.is_empty() {
                let period_bytes = s.period_frames * s.bytes_per_frame;
                s.pending.resize(period_bytes, 0);
                if s.started {
                    s.event.wait_for_event(2000).map_err(err)?;
                }
                let chunk: Vec<u8> = s.pending.drain(..).collect();
                s.render
                    .write_to_device(s.period_frames, &chunk, None)
                    .map_err(err)?;
                if !s.started {
                    s.client.start_stream().map_err(err)?;
                    s.started = true;
                }
            }
            if s.started {
                // Let the last period play out, then stop.
                s.event.wait_for_event(2000).map_err(err)?;
                s.event.wait_for_event(2000).map_err(err)?;
                s.client.stop_stream().map_err(err)?;
                s.client.reset_stream().map_err(err)?;
                s.started = false;
            }
        }
        Ok(())
    }

    fn discard(&mut self) -> Result<(), OutputError> {
        if let Some(s) = &mut self.stream {
            s.pending.clear();
            if s.started {
                s.client.stop_stream().map_err(err)?;
                s.client.reset_stream().map_err(err)?;
                s.started = false;
            }
        }
        Ok(())
    }

    fn close(&mut self) {
        if let Some(s) = self.stream.take() {
            if s.started {
                let _ = s.client.stop_stream();
            }
        }
        self.format = None;
    }

    fn queued_frames(&self) -> usize {
        self.stream
            .as_ref()
            .map(|s| s.pending.len() / s.bytes_per_frame.max(1) + s.period_frames)
            .unwrap_or(0)
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }
}
