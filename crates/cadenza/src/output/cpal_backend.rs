//! Shared-mode output through `cpal` (ALSA/PulseAudio/PipeWire, CoreAudio, WASAPI shared).
//!
//! Frames go into a ring buffer that the device callback drains; `write` blocks while more
//! than ~0.5 s is queued, which is how the engine gets paced by the device clock.
//! Samples are handed to cpal as `f32` (the mixer converts anyway) unless the device offers
//! an integer format matching the requested depth.

use super::{DeviceInfo, DsdCaps, Output, OutputError, OutputFormat, STANDARD_RATES};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, Stream, StreamConfig};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

struct Ring {
    queue: VecDeque<f32>,
    paused: bool,
    /// Samples the callback had to invent because the queue ran dry.
    underruns: u64,
}

/// See the module documentation.
pub struct CpalOutput {
    device: cpal::Device,
    device_name: String,
    format: Option<OutputFormat>,
    stream: Option<Stream>,
    ring: Arc<(Mutex<Ring>, Condvar)>,
    high_water: usize,
}

// cpal's Stream is !Send on some platforms; we only ever touch it from the engine thread
// that created it, and never from the callback, so moving the wrapper is sound.
unsafe impl Send for CpalOutput {}

/// Find a device by name (empty = default output).
fn find_device(name: &str) -> Result<cpal::Device, OutputError> {
    let host = cpal::default_host();
    if name.is_empty() || name == "default" {
        return host
            .default_output_device()
            .ok_or_else(|| OutputError::NoDevice("no default output device".into()));
    }
    host.output_devices()
        .map_err(|e| OutputError::Backend(e.to_string()))?
        .find(|d| d.name().map(|n| n == name).unwrap_or(false))
        .ok_or_else(|| OutputError::NoDevice(name.to_string()))
}

impl CpalOutput {
    /// Prepare to play on `device` (nothing is opened until [`Output::open`]).
    pub fn new(device: &str) -> Result<Self, OutputError> {
        let dev = find_device(device)?;
        let device_name = dev.name().unwrap_or_else(|_| device.to_string());
        Ok(Self {
            device: dev,
            device_name,
            format: None,
            stream: None,
            ring: Arc::new((
                Mutex::new(Ring {
                    queue: VecDeque::new(),
                    paused: false,
                    underruns: 0,
                }),
                Condvar::new(),
            )),
            high_water: 0,
        })
    }

    fn supported(&self, rate: u32) -> Option<cpal::SupportedStreamConfigRange> {
        let mut best: Option<cpal::SupportedStreamConfigRange> = None;
        for cfg in self.device.supported_output_configs().ok()? {
            if cfg.min_sample_rate().0 <= rate && rate <= cfg.max_sample_rate().0 {
                // Prefer f32, then the widest integer format.
                let rank = |f: SampleFormat| match f {
                    SampleFormat::F32 => 3,
                    SampleFormat::I32 => 2,
                    SampleFormat::I16 => 1,
                    _ => 0,
                };
                if best
                    .as_ref()
                    .is_none_or(|b| rank(cfg.sample_format()) > rank(b.sample_format()))
                {
                    best = Some(cfg);
                }
            }
        }
        best
    }
}

/// Every cpal output device with the rates it accepts.
pub fn list_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let default_name = host.default_output_device().and_then(|d| d.name().ok());
    let Ok(devices) = host.output_devices() else {
        return Vec::new();
    };
    devices
        .filter_map(|d| {
            let name = d.name().ok()?;
            let configs: Vec<_> = d.supported_output_configs().ok()?.collect();
            let rates = STANDARD_RATES
                .iter()
                .copied()
                .filter(|r| {
                    configs
                        .iter()
                        .any(|c| c.min_sample_rate().0 <= *r && *r <= c.max_sample_rate().0)
                })
                .collect();
            let channels = configs.iter().map(|c| c.channels() as u32).max();
            Some(DeviceInfo {
                backend: "cpal".into(),
                id: name.clone(),
                default: default_name.as_deref() == Some(name.as_str()),
                name,
                rates,
                bits: vec![16, 24, 32],
                channels,
                exclusive: false,
                // A shared-mode mixer resamples and dithers: DoP markers would not survive.
                dsd: DsdCaps::NONE,
            })
        })
        .collect()
}

impl Output for CpalOutput {
    fn name(&self) -> String {
        format!("cpal · {}", self.device_name)
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        self.close();
        if format.is_dsd() {
            // The OS mixer would resample the DoP frames into noise; refuse loudly instead.
            return Err(OutputError::Unsupported {
                device: self.device_name.clone(),
                format,
            });
        }
        let range = self
            .supported(format.rate)
            .ok_or(OutputError::Unsupported {
                device: self.device_name.clone(),
                format,
            })?;
        let sample_format = range.sample_format();
        let config = StreamConfig {
            channels: format.channels as u16,
            sample_rate: SampleRate(format.rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let ring = Arc::clone(&self.ring);
        let scale = 1.0 / (1u64 << (format.bits - 1)) as f32;
        let err_fn = |e| log::error!("cpal stream error: {e}");
        let stream = match sample_format {
            SampleFormat::F32 => self.device.build_output_stream(
                &config,
                move |out: &mut [f32], _| fill(&ring, out, |v| v),
                err_fn,
                None,
            ),
            SampleFormat::I16 => self.device.build_output_stream(
                &config,
                move |out: &mut [i16], _| {
                    fill(&ring, out, |v| {
                        (v * 32768.0).clamp(-32768.0, 32767.0) as i16
                    })
                },
                err_fn,
                None,
            ),
            SampleFormat::I32 => self.device.build_output_stream(
                &config,
                move |out: &mut [i32], _| {
                    fill(&ring, out, |v| {
                        (v as f64 * 2147483648.0).clamp(-2147483648.0, 2147483647.0) as i32
                    })
                },
                err_fn,
                None,
            ),
            other => {
                return Err(OutputError::Backend(format!(
                    "unsupported cpal sample format {other:?}"
                )))
            }
        }
        .map_err(|e| OutputError::Backend(e.to_string()))?;
        stream
            .play()
            .map_err(|e| OutputError::Backend(e.to_string()))?;
        self.stream = Some(stream);
        self.format = Some(format);
        self.high_water = (format.rate as usize / 2) * format.channels as usize;
        let _ = scale;
        Ok(format)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        matches!(bits, 16 | 24 | 32) && self.supported(rate).is_some()
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        let fmt = self
            .format
            .ok_or_else(|| OutputError::Backend("cpal stream not open".into()))?;
        let scale = 1.0 / (1u64 << (fmt.bits - 1)) as f32;
        let (lock, cv) = &*self.ring;
        let mut ring = lock.lock().unwrap();
        while ring.queue.len() > self.high_water {
            ring = cv.wait_timeout(ring, Duration::from_millis(100)).unwrap().0;
        }
        ring.queue.extend(frames.iter().map(|&s| s as f32 * scale));
        Ok(())
    }

    fn pause(&mut self, paused: bool) -> Result<(), OutputError> {
        self.ring.0.lock().unwrap().paused = paused;
        if let Some(s) = &self.stream {
            let r = if paused {
                s.pause().map_err(|e| e.to_string())
            } else {
                s.play().map_err(|e| e.to_string())
            };
            r.map_err(OutputError::Backend)?;
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), OutputError> {
        let (lock, cv) = &*self.ring;
        let mut ring = lock.lock().unwrap();
        let mut waited = 0;
        while !ring.queue.is_empty() && waited < 50 {
            ring = cv.wait_timeout(ring, Duration::from_millis(100)).unwrap().0;
            waited += 1;
        }
        Ok(())
    }

    fn discard(&mut self) -> Result<(), OutputError> {
        self.ring.0.lock().unwrap().queue.clear();
        Ok(())
    }

    fn close(&mut self) {
        self.stream = None;
        self.format = None;
        let mut ring = self.ring.0.lock().unwrap();
        ring.queue.clear();
        ring.paused = false;
    }

    fn queued_frames(&self) -> usize {
        let ch = self.format.map(|f| f.channels as usize).unwrap_or(2).max(1);
        self.ring.0.lock().unwrap().queue.len() / ch
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }
}

/// The device callback: copy from the ring, converting with `conv`; silence on underrun.
fn fill<T: Copy + Default>(
    ring: &Arc<(Mutex<Ring>, Condvar)>,
    out: &mut [T],
    conv: impl Fn(f32) -> T,
) {
    let (lock, cv) = &**ring;
    let mut r = lock.lock().unwrap();
    if r.paused {
        out.fill(T::default());
        return;
    }
    for o in out.iter_mut() {
        match r.queue.pop_front() {
            Some(v) => *o = conv(v),
            None => {
                r.underruns += 1;
                *o = T::default();
            }
        }
    }
    cv.notify_all();
}
