//! WASAPI exclusive-mode output (Windows): the device is taken over, the sample rate is
//! exactly what we ask for and integer samples reach the driver untouched.
//!
//! Event-driven exclusive mode: the device signals an event once per period and expects
//! exactly one period of frames in return, so `write` accumulates bytes and feeds the
//! device one period at a time, blocking on the event. This file is only compiled on
//! Windows (`cfg(windows)` + feature `wasapi`).
//!
//! **DSD.** DoP is the standard way: 24-bit PCM at `f_dsd / 16`, which any exclusive
//! 24-bit-capable endpoint carries. Windows has no documented native-DSD wave format;
//! a few vendor drivers accept a `WAVEFORMATEXTENSIBLE` whose `SubFormat` is a
//! driver-specific GUID with 32-bit words at `f_dsd / 32`. Because that GUID is not
//! standardised we do not hard-code one: set `CADENZA_WASAPI_DSD_GUID` to the GUID your
//! driver documents (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`) and Cadenza probes the
//! endpoint with it; when the driver accepts it, the device lists `native dsd_u32_le`
//! (Windows is little-endian, so the words are written as `DSD_U32_LE` bytes).
//!
//! **Format negotiation.** Drivers differ in the containers they take (see
//! [`super::wasapi_formats`]): PCM is opened with the first layout of
//! [`exclusive_candidates`] the driver accepts, and [`Output::open`] reports the bit
//! depth actually used so the engine produces matching words. When exclusive mode is
//! refused outright, PCM falls back to an event-driven *shared* stream of 32-bit float
//! with `AUTOCONVERTPCM`, so Windows resamples and mixes it like any other app: not
//! bit-perfect, but the track plays instead of failing with "format not supported".
//! DSD never falls back (the mixer would turn DoP into noise).

use super::wasapi_formats::{encode, exclusive_candidates, SampleLayout};
use super::{DeviceInfo, DsdCaps, Encoding, Output, OutputError, OutputFormat, STANDARD_RATES};
use cadenza_dsp::dsd::packing::words_to_bytes;
use cadenza_dsp::dsd::{DsdPacking, DsdRate, NativeFormat};
use std::collections::VecDeque;
use wasapi::{
    AudioClient, AudioRenderClient, Device, DeviceEnumerator, Direction, Handle, SampleType,
    StreamMode, WaveFormat, GUID,
};

/// Environment variable naming the driver's native-DSD subformat GUID (see module docs).
pub const DSD_GUID_ENV: &str = "CADENZA_WASAPI_DSD_GUID";

/// See the module documentation.
pub struct WasapiOutput {
    device_name: String,
    device: Device,
    format: Option<OutputFormat>,
    stream: Option<Stream>,
    /// Native DSD in use: words are serialised as `DSD_U32_LE` bytes.
    native: bool,
}

/// An initialised WASAPI render stream and the bytes waiting to go into it.
struct Stream {
    /// The audio client the stream was initialised on.
    client: AudioClient,
    /// Hands out the device buffer to fill.
    render: AudioRenderClient,
    /// Signalled by the device whenever buffer space frees up.
    event: Handle,
    /// How each sample is laid out in the device buffer.
    layout: SampleLayout,
    /// `true` for the shared-mode fallback (Windows mixer), `false` for exclusive.
    shared: bool,
    /// Exclusive: frames per period (= the buffer size). Shared: the whole buffer size.
    period_frames: usize,
    /// Bytes per interleaved frame in `layout`.
    bytes_per_frame: usize,
    /// Encoded bytes not yet handed to the device.
    pending: VecDeque<u8>,
    /// The device clock is running (the buffer was primed once).
    started: bool,
    /// Held by [`Output::pause`].
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

/// Parse `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (braces optional).
fn parse_guid(s: &str) -> Option<GUID> {
    let s = s.trim().trim_start_matches('{').trim_end_matches('}');
    let hex: String = s.split('-').collect();
    if hex.len() != 32 || s.split('-').count() != 5 {
        return None;
    }
    u128::from_str_radix(&hex, 16).ok().map(GUID::from_u128)
}

/// The native-DSD subformat GUID from the environment, if the user gave one.
fn dsd_guid() -> Option<GUID> {
    std::env::var(DSD_GUID_ENV)
        .ok()
        .and_then(|s| parse_guid(&s))
}

/// The wave format for `format` in `layout`: integer layouts as given, float as 32-bit
/// IEEE. DoP is 24-bit PCM; native DSD is 32-bit words with the driver's subformat GUID
/// (`None` when no GUID is configured).
fn wave_format(format: OutputFormat, layout: SampleLayout) -> Option<WaveFormat> {
    let (store, valid, kind) = match layout {
        SampleLayout::Int { store, valid } => (store, valid, SampleType::Int),
        SampleLayout::Float32 => (32, 32, SampleType::Float),
    };
    let mut wave = WaveFormat::new(
        store as usize,
        valid as usize,
        &kind,
        format.rate as usize,
        format.channels as usize,
        None,
    );
    if let Encoding::Dsd {
        packing: DsdPacking::Native,
        ..
    } = format.encoding
    {
        wave.wave_fmt.SubFormat = dsd_guid()?;
    }
    Some(wave)
}

/// The exclusive-mode layouts to try for `format`: every container of its bit depth for
/// PCM (24-bit: 32-bit words first, then packed), the one fixed layout for DSD.
fn layouts_for(format: OutputFormat) -> Vec<SampleLayout> {
    if format.is_dsd() {
        return vec![SampleLayout::int_for_bits(format.bits)];
    }
    exclusive_candidates(format.bits)
        .into_iter()
        .filter(|l| l.valid_bits() == Some(format.bits))
        .collect()
}

/// Whether the endpoint takes `format` in exclusive mode in any container of its depth.
fn client_supports(client: &AudioClient, format: OutputFormat) -> bool {
    layouts_for(format).into_iter().any(|l| {
        wave_format(format, l)
            .is_some_and(|w| client.is_supported_exclusive_with_quirks(&w).is_ok())
    })
}

/// Open `format` in exclusive mode in `layout`: `None` when the driver refuses the
/// format, `Some(Err)` when it accepts it but initialising the stream fails (device in
/// use, exclusive mode disabled …).
fn open_exclusive(
    device: &Device,
    format: OutputFormat,
    layout: SampleLayout,
) -> Option<Result<Stream, OutputError>> {
    // A client whose Initialize failed cannot be reused, so every attempt gets a new one.
    let mut client = device.get_iaudioclient().map_err(err).ok()?;
    let wanted = wave_format(format, layout)?;
    let wave = client.is_supported_exclusive_with_quirks(&wanted).ok()?;
    Some((|| {
        // Three default periods (~30 ms) keeps USB DACs happy; HDA devices want 128-byte
        // alignment.
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
        finish_stream(client, &wave, layout, false)
    })())
}

/// Open `format` as a shared-mode float stream the Windows mixer converts
/// (`AUTOCONVERTPCM`: any rate and channel count the engine sends is resampled to the
/// mix format), for endpoints where exclusive mode is not available.
fn open_shared(device: &Device, format: OutputFormat) -> Result<Stream, OutputError> {
    let mut client = device.get_iaudioclient().map_err(err)?;
    let layout = SampleLayout::Float32;
    let wave = wave_format(format, layout)
        .ok_or_else(|| OutputError::Backend("no shared wave format".into()))?;
    let (default_period, _min) = client.get_device_period().map_err(err)?;
    client
        .initialize_client(
            &wave,
            &Direction::Render,
            &StreamMode::EventsShared {
                autoconvert: true,
                // Ten default periods (~100 ms) of headroom against scheduling hiccups.
                buffer_duration_hns: 10 * default_period,
            },
        )
        .map_err(|e| OutputError::Backend(format!("initialize shared stream: {e}")))?;
    finish_stream(client, &wave, layout, true)
}

/// Fetch the event, render client and buffer size of an initialised `client`.
fn finish_stream(
    client: AudioClient,
    wave: &WaveFormat,
    layout: SampleLayout,
    shared: bool,
) -> Result<Stream, OutputError> {
    let event = client.set_get_eventhandle().map_err(err)?;
    let render = client.get_audiorenderclient().map_err(err)?;
    let period_frames = client.get_buffer_size().map_err(err)? as usize;
    Ok(Stream {
        client,
        render,
        event,
        layout,
        shared,
        period_frames,
        bytes_per_frame: wave.get_blockalign() as usize,
        pending: VecDeque::new(),
        started: false,
        paused: false,
    })
}

impl Stream {
    /// Shared mode: move as much of `pending` into the device buffer as fits, waiting for
    /// the device whenever the buffer is full, until less than one frame is left.
    fn feed_shared(&mut self) -> Result<(), OutputError> {
        let bpf = self.bytes_per_frame.max(1);
        while self.pending.len() >= bpf {
            let free = self.client.get_available_space_in_frames().map_err(err)? as usize;
            let n = (self.pending.len() / bpf).min(free);
            if n == 0 {
                // Full: start the clock if this is the first fill, then wait for room.
                self.start_once()?;
                self.event.wait_for_event(2000).map_err(err)?;
                continue;
            }
            self.render
                .write_to_device_from_deque(n, &mut self.pending, None)
                .map_err(err)?;
        }
        Ok(())
    }

    /// Start the device clock unless it is running already.
    fn start_once(&mut self) -> Result<(), OutputError> {
        if !self.started {
            self.client.start_stream().map_err(err)?;
            self.started = true;
        }
        Ok(())
    }
}

/// DSD capabilities of an endpoint: DoP wherever 24-bit PCM reaches `f_dsd/16`, native
/// only with a user-supplied GUID the driver accepts.
fn dsd_caps(client: &AudioClient) -> DsdCaps {
    let mut caps = DsdCaps {
        dop: client_supports(client, OutputFormat::pcm(176_400, 24, 2))
            || client_supports(client, OutputFormat::pcm(192_000, 24, 2)),
        native: Vec::new(),
        max_rate: None,
    };
    let native_possible = dsd_guid().is_some();
    for &r in DsdRate::catalog() {
        for base in [44_100u32, 48_000] {
            let hz = r.hz(base);
            let dop = caps.dop
                && r.dop_possible()
                && client_supports(client, OutputFormat::dsd(hz, DsdPacking::Dop, 2));
            let nat = native_possible
                && client_supports(client, OutputFormat::dsd(hz, DsdPacking::Native, 2));
            if nat && caps.native.is_empty() {
                caps.native.push(NativeFormat::U32Le);
            }
            if dop || nat {
                caps.max_rate = Some(caps.max_rate.map_or(hz, |m: u32| m.max(hz)));
            }
        }
    }
    caps
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
            native: false,
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
        let supports =
            |rate: u32, bits: u32| client_supports(&client, OutputFormat::pcm(rate, bits, 2));
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
            dsd: dsd_caps(&client),
        });
    }
    out
}

impl Output for WasapiOutput {
    /// `wasapi exclusive · Speakers`, or `wasapi shared · …` while the fallback is open.
    fn name(&self) -> String {
        let mode = if self.exclusive() {
            "exclusive"
        } else {
            "shared"
        };
        format!("wasapi {mode} · {}", self.device_name)
    }

    /// Open `format`. PCM tries every exclusive layout of [`exclusive_candidates`] (which
    /// may change the bit depth; the returned format says which) and then the shared
    /// fallback; DSD only opens exactly as asked.
    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        self.close();
        let unsupported = || OutputError::Unsupported {
            device: self.device_name.clone(),
            format,
        };
        let layouts = if format.is_dsd() {
            layouts_for(format)
        } else {
            exclusive_candidates(format.bits)
        };
        // The first exclusive initialisation error, reported if nothing else opens.
        let mut init_err = None;
        let mut opened = None;
        for layout in layouts {
            let fmt = match layout.valid_bits() {
                Some(bits) if !format.is_dsd() => OutputFormat { bits, ..format },
                _ => format,
            };
            match open_exclusive(&self.device, fmt, layout) {
                Some(Ok(stream)) => {
                    opened = Some((stream, fmt));
                    break;
                }
                Some(Err(e)) => {
                    log::warn!("{}: exclusive {fmt} ({layout:?}): {e}", self.device_name);
                    init_err.get_or_insert(e);
                }
                None => log::debug!("{}: exclusive {fmt} ({layout:?}) refused", self.device_name),
            }
        }
        if opened.is_none() && !format.is_dsd() {
            // No exclusive layout worked: let the Windows mixer convert instead of failing.
            match open_shared(&self.device, format) {
                Ok(stream) => {
                    log::warn!(
                        "{}: exclusive mode refused {format}; playing through the Windows \
                         mixer (shared mode, not bit-perfect)",
                        self.device_name
                    );
                    opened = Some((stream, format));
                }
                Err(e) => {
                    log::warn!("{}: shared {format}: {e}", self.device_name);
                    init_err.get_or_insert(e);
                }
            }
        }
        let Some((stream, got)) = opened else {
            return Err(init_err.unwrap_or_else(unsupported));
        };
        if got.bits != format.bits {
            log::info!(
                "{}: {format} refused in exclusive mode; opened {got}",
                self.device_name
            );
        }
        self.stream = Some(stream);
        self.format = Some(got);
        self.native = matches!(
            got.encoding,
            Encoding::Dsd {
                packing: DsdPacking::Native,
                ..
            }
        );
        Ok(got)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        if !matches!(bits, 16 | 24 | 32) {
            return false;
        }
        let Ok(client) = self.device.get_iaudioclient() else {
            return false;
        };
        client_supports(&client, OutputFormat::pcm(rate, bits, 2))
    }

    fn dsd_caps(&self) -> DsdCaps {
        match self.device.get_iaudioclient() {
            Ok(client) => dsd_caps(&client),
            Err(_) => DsdCaps::NONE,
        }
    }

    fn supports_format(&self, format: &OutputFormat) -> bool {
        let Ok(client) = self.device.get_iaudioclient() else {
            return false;
        };
        client_supports(&client, *format)
    }

    /// Exclusive unless the open stream is the shared fallback (before opening: the
    /// backend's intent, exclusive).
    fn exclusive(&self) -> bool {
        !self.stream.as_ref().is_some_and(|s| s.shared)
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        let fmt = self
            .format
            .ok_or_else(|| OutputError::Backend("wasapi stream not open".into()))?;
        let s = self.stream.as_mut().unwrap();
        if self.native {
            s.pending.extend(words_to_bytes(
                frames,
                fmt.channels as usize,
                NativeFormat::U32Le,
            ));
        } else {
            encode(frames, fmt.bits, s.layout, &mut s.pending);
        }
        if s.shared {
            // The shared buffer takes any number of frames; it blocks only when full.
            return s.feed_shared();
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
            if s.shared {
                s.feed_shared()?;
                if s.client.get_current_padding().map_err(err)? > 0 {
                    s.start_once()?;
                }
                // Wait (at most ~2 s per event) until the mixer has taken everything.
                while s.started && s.client.get_current_padding().map_err(err)? > 0 {
                    s.event.wait_for_event(2000).map_err(err)?;
                }
                if s.started {
                    s.client.stop_stream().map_err(err)?;
                    s.client.reset_stream().map_err(err)?;
                    s.started = false;
                }
                return Ok(());
            }
            if !s.pending.is_empty() {
                let period_bytes = s.period_frames * s.bytes_per_frame;
                // DSD silence is 0x69, PCM silence is 0.
                let pad = if self.format.is_some_and(|f| f.is_dsd()) && self.native {
                    0x69
                } else {
                    0
                };
                s.pending.resize(period_bytes, pad);
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
        self.native = false;
    }

    fn queued_frames(&self) -> usize {
        self.stream
            .as_ref()
            .map(|s| {
                // Exclusive: one full period is always in the device. Shared: what the
                // mixer still holds.
                let in_device = if s.shared {
                    s.client.get_current_padding().unwrap_or(0) as usize
                } else {
                    s.period_frames
                };
                s.pending.len() / s.bytes_per_frame.max(1) + in_device
            })
            .unwrap_or(0)
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }
}
