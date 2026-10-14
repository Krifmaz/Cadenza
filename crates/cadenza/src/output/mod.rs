//! Audio outputs behind one trait, so the engine does not care whether frames go to a
//! counter, the OS mixer, a raw ALSA device, a WASAPI exclusive stream, a hogged CoreAudio
//! device or a network endpoint.
//!
//! **Formats.** An [`OutputFormat`] is always "words of `bits` valid bits at `rate` frames
//! per second, `channels` per frame" – that is what every backend physically opens – plus
//! an [`Encoding`] saying what the words mean: PCM, or DSD packed as DoP (24-bit words at
//! `f_dsd / 16`) or as native 32-bit words (`f_dsd / 32`). The pipeline's
//! `device_format()` yields exactly those `(rate, bits)` pairs, and its words are already
//! laid out this way (see `cadenza_dsp::dsd::packing`), so the engine hands the same
//! `&[i32]` slices to [`Output::write`] in every mode; a backend converts DSD words to the
//! byte order its driver wants.

pub mod null;

#[cfg(all(feature = "alsa", target_os = "linux"))]
pub mod alsa_backend;
// The ASIO host is platform independent (tested everywhere against a fake driver); only
// its loader needs Windows.
#[cfg(any(test, all(windows, feature = "asio")))]
pub mod asio;
#[cfg(all(feature = "coreaudio", target_os = "macos"))]
pub mod coreaudio_backend;
#[cfg(feature = "cpal")]
pub mod cpal_backend;
#[cfg(all(feature = "wasapi", windows))]
pub mod wasapi_backend;

use cadenza_dsp::dsd::{DsdPacking, DsdRate, NativeFormat};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Sample rates worth probing a device for (PCM; up to 32× = 1.4112 / 1.536 MHz, plus the
/// DSD64 bit rates as PCM for the few devices that take them as modulator input).
pub const STANDARD_RATES: &[u32] = &[
    44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000, 705600, 768000, 1_411_200,
    1_536_000, 2_822_400, 3_072_000,
];

/// Highest PCM rate hardware takes today (32× 48 kHz); above it only DSD exists.
pub const MAX_PCM_RATE: u32 = 1_536_000;

/// What the words of a stream mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Encoding {
    /// Plain PCM: `bits` valid bits per word.
    #[default]
    Pcm,
    /// A 1-bit stream of `rate` bits per second per channel, packed as `packing`.
    Dsd {
        /// DSD bit rate in Hz (2 822 400 for DSD64 of the 44.1 family …).
        rate: u32,
        /// DoP (24-bit PCM frames with markers) or native 32-bit words.
        packing: DsdPacking,
    },
}

/// The format a stream is opened with: interleaved integer words of `bits` valid bits
/// (16 in `i16`, 24 or 32 in `i32` words) at `rate` frames per second, carrying
/// [`Encoding`] `encoding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputFormat {
    /// Frame rate in Hz (the PCM rate; `f_dsd / 16` for DoP, `f_dsd / 32` for native DSD).
    pub rate: u32,
    /// Valid bits per word: 16, 24 or 32 (24 for DoP, 32 for native DSD).
    pub bits: u32,
    /// Channels per frame.
    pub channels: u32,
    /// PCM or DSD.
    #[serde(default)]
    pub encoding: Encoding,
}

impl OutputFormat {
    /// A PCM format.
    pub fn pcm(rate: u32, bits: u32, channels: u32) -> Self {
        Self {
            rate,
            bits,
            channels,
            encoding: Encoding::Pcm,
        }
    }

    /// A DSD format for `dsd_rate` bits per second, packed as `packing`: DoP is 24-bit
    /// frames at `f/16`, native is 32-bit words at `f/32`.
    pub fn dsd(dsd_rate: u32, packing: DsdPacking, channels: u32) -> Self {
        let (rate, bits) = match packing {
            DsdPacking::Dop => (dsd_rate / 16, 24),
            DsdPacking::Native => (dsd_rate / 32, 32),
        };
        Self {
            rate,
            bits,
            channels,
            encoding: Encoding::Dsd {
                rate: dsd_rate,
                packing,
            },
        }
    }

    /// From the pipeline's `device_format()` pair and, in DSD mode, the DSD bit rate and
    /// packing it produces.
    pub fn from_device_format(
        (rate, bits): (u32, u32),
        channels: u32,
        dsd: Option<(u32, DsdPacking)>,
    ) -> Self {
        match dsd {
            Some((dsd_rate, packing)) => Self::dsd(dsd_rate, packing, channels),
            None => Self::pcm(rate, bits, channels),
        }
    }

    /// `true` for either DSD packing.
    pub fn is_dsd(&self) -> bool {
        matches!(self.encoding, Encoding::Dsd { .. })
    }

    /// The DSD bit rate and packing, in DSD.
    pub fn dsd_info(&self) -> Option<(u32, DsdPacking)> {
        match self.encoding {
            Encoding::Dsd { rate, packing } => Some((rate, packing)),
            Encoding::Pcm => None,
        }
    }

    /// Same physical stream, ignoring the encoding (what a PCM-only backend opens).
    pub fn as_pcm(&self) -> Self {
        Self {
            encoding: Encoding::Pcm,
            ..*self
        }
    }

    /// `DSD256 (DoP)`, `DSD512 native` or `24-bit 352.8 kHz`.
    pub fn label(&self) -> String {
        match self.encoding {
            Encoding::Pcm => format!(
                "{}-bit {} kHz",
                self.bits,
                cadenza_dsp::signal_path::rate_label(self.rate)
            ),
            Encoding::Dsd { rate, packing } => {
                let name = DsdRate::from_hz(rate)
                    .map(|(r, _)| r.label())
                    .unwrap_or_else(|| format!("DSD {:.4} MHz", rate as f64 / 1e6));
                match packing {
                    DsdPacking::Dop => format!("{name} (DoP)"),
                    DsdPacking::Native => format!("{name} native"),
                }
            }
        }
    }
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}ch", self.label(), self.channels)
    }
}

/// What a device can do with DSD.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DsdCaps {
    /// Takes DoP (any exclusive device that plays 24-bit PCM at `f_dsd/16` can carry it;
    /// whether the DAC decodes the markers is up to the DAC).
    pub dop: bool,
    /// Native DSD byte layouts the driver accepts (empty = none).
    pub native: Vec<NativeFormat>,
    /// Highest DSD bit rate (Hz) the device accepts in any packing, when known.
    pub max_rate: Option<u32>,
}

impl DsdCaps {
    /// No DSD at all.
    pub const NONE: DsdCaps = DsdCaps {
        dop: false,
        native: Vec::new(),
        max_rate: None,
    };

    /// DoP only, limited by the PCM rates the device reaches.
    pub fn dop_only(max_rate: Option<u32>) -> Self {
        Self {
            dop: true,
            native: Vec::new(),
            max_rate,
        }
    }

    /// Anything at all?
    pub fn any(&self) -> bool {
        self.dop || !self.native.is_empty()
    }

    /// `DSD512` – the label of the highest rate, for "your device stops at DSDxxx".
    pub fn max_rate_label(&self) -> Option<String> {
        self.max_rate
            .and_then(DsdRate::from_hz)
            .map(|(r, _)| r.label())
    }

    /// `DoP · native DSD_U32_BE/DSD_U8 · up to DSD512` for `cadenza devices`.
    pub fn summary(&self) -> String {
        if !self.any() {
            return "no DSD".into();
        }
        let mut parts = Vec::new();
        if self.dop {
            parts.push("DoP".to_string());
        }
        if !self.native.is_empty() {
            let ids: Vec<&str> = self.native.iter().map(|n| n.id()).collect();
            parts.push(format!("native {}", ids.join("/")));
        }
        if let Some(l) = self.max_rate_label() {
            parts.push(format!("up to {l}"));
        }
        parts.join(" · ")
    }
}

/// Errors from an output.
#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    /// The named device/backend does not exist.
    #[error("no such device: {0}")]
    NoDevice(String),
    /// The device refuses this format.
    #[error("format not supported by {device}: {format}")]
    Unsupported {
        /// Device name.
        device: String,
        /// The format that was requested.
        format: OutputFormat,
    },
    /// The backend was not compiled in.
    #[error("backend '{0}' is not compiled in (enable the cargo feature)")]
    NotCompiled(String),
    /// Anything else the backend reports.
    #[error("{0}")]
    Backend(String),
}

/// One device as shown by `cadenza devices` and `/api/devices`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceInfo {
    /// Backend id (`null`, `cpal`, `alsa`, `wasapi`, `asio`, `coreaudio`, `endpoint`).
    pub backend: String,
    /// The id to put in the configuration.
    pub id: String,
    /// Human readable name.
    pub name: String,
    /// Sample rates the device reports as supported (empty when unknown).
    pub rates: Vec<u32>,
    /// Bit depths the device reports as supported.
    pub bits: Vec<u32>,
    /// Maximum channel count, when known.
    pub channels: Option<u32>,
    /// The backend's default device.
    pub default: bool,
    /// `true` for exclusive / direct backends (bit-perfect capable).
    pub exclusive: bool,
    /// DSD support.
    pub dsd: DsdCaps,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            backend: String::new(),
            id: String::new(),
            name: String::new(),
            rates: Vec::new(),
            bits: Vec::new(),
            channels: None,
            default: false,
            exclusive: false,
            dsd: DsdCaps::NONE,
        }
    }
}

/// An audio sink.
pub trait Output: Send {
    /// `backend · device` name for the signal path.
    fn name(&self) -> String;

    /// Open (or reopen) the stream in `format`. Returns the format actually opened, which
    /// may differ in bit depth when the device needs a different container.
    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError>;

    /// Whether PCM at `rate`/`bits` can be opened (best effort without opening).
    fn supports(&self, rate: u32, bits: u32) -> bool;

    /// What the device does with DSD. The default is nothing; exclusive backends report
    /// DoP (and native layouts where the driver has them).
    fn dsd_caps(&self) -> DsdCaps {
        DsdCaps::NONE
    }

    /// Whether `format` (PCM or DSD) can be opened. The default answers from
    /// [`Output::supports`] and [`Output::dsd_caps`].
    fn supports_format(&self, format: &OutputFormat) -> bool {
        match format.encoding {
            Encoding::Pcm => self.supports(format.rate, format.bits),
            Encoding::Dsd { rate, packing } => {
                let caps = self.dsd_caps();
                let within = caps.max_rate.is_none_or(|m| rate <= m);
                within
                    && match packing {
                        DsdPacking::Dop => caps.dop && self.supports(format.rate, 24),
                        DsdPacking::Native => !caps.native.is_empty(),
                    }
            }
        }
    }

    /// Exclusive / hog / direct access (the device is ours alone, no mixer in between).
    fn exclusive(&self) -> bool {
        false
    }

    /// Queue interleaved words in the open format; blocks while the device is busy, so
    /// the caller is paced by the device clock.
    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError>;

    /// Hold playback (`true`) or resume it.
    fn pause(&mut self, paused: bool) -> Result<(), OutputError>;

    /// Block until everything queued has been played.
    fn drain(&mut self) -> Result<(), OutputError>;

    /// Drop whatever is queued (seek, stop).
    fn discard(&mut self) -> Result<(), OutputError>;

    /// Close the stream.
    fn close(&mut self);

    /// Frames queued but not yet played, for position reporting.
    fn queued_frames(&self) -> usize {
        0
    }

    /// Total frames accepted by `write` since the output was created (diagnostics, tests).
    fn frames_written(&self) -> u64 {
        0
    }

    /// The format in use, if open.
    fn format(&self) -> Option<OutputFormat>;
}

/// Backend ids compiled into this build.
pub fn backends() -> Vec<&'static str> {
    let mut v = vec!["null"];
    #[cfg(feature = "cpal")]
    v.push("cpal");
    #[cfg(all(feature = "alsa", target_os = "linux"))]
    v.push("alsa");
    #[cfg(all(feature = "wasapi", windows))]
    v.push("wasapi");
    #[cfg(all(feature = "asio", windows))]
    v.push("asio");
    #[cfg(all(feature = "coreaudio", target_os = "macos"))]
    v.push("coreaudio");
    v.push("endpoint");
    v
}

/// The backend to use when the configuration does not say: WASAPI exclusive on Windows if
/// compiled, CoreAudio hog mode on macOS if compiled, ALSA on Linux if compiled (and cpal
/// is not), otherwise cpal, otherwise null.
pub fn default_backend() -> &'static str {
    if cfg!(all(feature = "wasapi", windows)) {
        "wasapi"
    } else if cfg!(all(feature = "coreaudio", target_os = "macos")) {
        "coreaudio"
    } else if cfg!(all(
        feature = "alsa",
        target_os = "linux",
        not(feature = "cpal")
    )) {
        "alsa"
    } else if cfg!(feature = "cpal") {
        "cpal"
    } else {
        "null"
    }
}

/// Every device of every compiled backend. Network endpoints found by mDNS so far are
/// listed too (`endpoint` backend); discovery runs in the background, so the first
/// listing after start-up may miss them.
pub fn list_devices() -> Vec<DeviceInfo> {
    let mut out = vec![null::device_info()];
    #[cfg(feature = "cpal")]
    out.extend(cpal_backend::list_devices());
    #[cfg(all(feature = "alsa", target_os = "linux"))]
    out.extend(alsa_backend::list_devices());
    #[cfg(all(feature = "wasapi", windows))]
    out.extend(wasapi_backend::list_devices());
    #[cfg(all(feature = "asio", windows))]
    out.extend(asio::windows::list_devices());
    #[cfg(all(feature = "coreaudio", target_os = "macos"))]
    out.extend(coreaudio_backend::list_devices());
    out.extend(crate::endpoint::client::list_devices());
    out
}

/// Split `endpoint://name` style ids into `(backend, device)`; anything else is returned
/// as `(backend, device)` unchanged.
pub fn split_backend<'a>(backend: &'a str, device: &'a str) -> (&'a str, &'a str) {
    match backend.split_once("://") {
        Some((b, d)) if device.is_empty() => (b, d),
        Some((b, _)) => (b, device),
        None => (backend, device),
    }
}

/// Create an output for `backend` / `device` (empty device = the backend's default).
/// `backend` may also be `endpoint://name`.
pub fn open_backend(backend: &str, device: &str) -> Result<Box<dyn Output>, OutputError> {
    let (backend, device) = split_backend(backend, device);
    match backend {
        "null" => Ok(Box::new(null::NullOutput::new(true))),
        #[cfg(feature = "cpal")]
        "cpal" => Ok(Box::new(cpal_backend::CpalOutput::new(device)?)),
        #[cfg(all(feature = "alsa", target_os = "linux"))]
        "alsa" => Ok(Box::new(alsa_backend::AlsaOutput::new(device)?)),
        #[cfg(all(feature = "wasapi", windows))]
        "wasapi" => Ok(Box::new(wasapi_backend::WasapiOutput::new(device)?)),
        #[cfg(all(feature = "asio", windows))]
        "asio" => Ok(Box::new(asio::windows::open(device)?)),
        #[cfg(all(feature = "coreaudio", target_os = "macos"))]
        "coreaudio" => Ok(Box::new(coreaudio_backend::CoreAudioOutput::new(device)?)),
        "endpoint" => Ok(Box::new(crate::endpoint::client::EndpointOutput::connect(
            device,
        )?)),
        other => Err(OutputError::NotCompiled(other.to_string())),
    }
}

/// Pick the output rate for a source at `source_rate` given the configured preference:
/// `None` (auto) → the highest rate the output supports that is an integer multiple of the
/// source rate's family (44.1 k or 48 k), capped at `max_rate`; `Some(r)` → `r` if the
/// output supports it, otherwise the source rate.
pub fn choose_rate(
    output: &dyn Output,
    source_rate: u32,
    preferred: Option<u32>,
    max_rate: u32,
    bits: u32,
) -> u32 {
    match preferred {
        Some(r) if output.supports(r, bits) => r,
        Some(r) => {
            log::warn!(
                "{} does not support {r} Hz; staying at {source_rate} Hz",
                output.name()
            );
            source_rate
        }
        None => {
            let base = if source_rate % 11025 == 0 {
                44100
            } else if source_rate % 8000 == 0 {
                48000
            } else {
                source_rate
            };
            let mut best = source_rate;
            let mut r = base;
            while r <= max_rate.min(MAX_PCM_RATE) {
                if r >= source_rate && r % source_rate == 0 && output.supports(r, bits) {
                    best = r;
                }
                r *= 2;
            }
            best
        }
    }
}

/// Pick the DSD rate and packing to open for a source of `family_base` (44 100 or 48 000):
/// `wanted` if the output takes it, otherwise the highest lower rate it does; `packing` is
/// kept when possible, and native falls back to DoP (never the other way round, because a
/// device that lists native DSD decodes native). `None` when the device does no DSD at
/// all, so the caller stays with PCM.
pub fn choose_dsd(
    output: &dyn Output,
    family_base: u32,
    wanted: DsdRate,
    packing: DsdPacking,
    channels: u32,
) -> Option<(DsdRate, DsdPacking)> {
    let packings: &[DsdPacking] = match packing {
        DsdPacking::Native => &[DsdPacking::Native, DsdPacking::Dop],
        DsdPacking::Dop => &[DsdPacking::Dop],
    };
    for &p in packings {
        for &r in DsdRate::catalog().iter().rev() {
            if r > wanted || (p == DsdPacking::Dop && !r.dop_possible()) {
                continue;
            }
            if output.supports_format(&OutputFormat::dsd(r.hz(family_base), p, channels)) {
                return Some((r, p));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_rate_picks_the_highest_family_multiple() {
        let out = null::NullOutput::new(false);
        assert_eq!(choose_rate(&out, 44100, None, 768_000, 24), 705_600);
        assert_eq!(choose_rate(&out, 44100, None, 192_000, 24), 176_400);
        assert_eq!(choose_rate(&out, 48000, None, 384_000, 24), 384_000);
        assert_eq!(choose_rate(&out, 96000, None, 192_000, 24), 192_000);
        assert_eq!(choose_rate(&out, 44100, Some(48000), 768_000, 24), 48000);
        assert_eq!(choose_rate(&out, 44100, Some(1234), 768_000, 24), 44100);
        assert_eq!(choose_rate(&out, 44100, None, 44100, 24), 44100);
        // 32× is the PCM ceiling; a higher max_rate does not go to 64×.
        assert_eq!(choose_rate(&out, 48000, None, 10_000_000, 24), 1_536_000);
    }

    #[test]
    fn dsd_formats_have_the_dop_and_native_frame_rates() {
        let dop = OutputFormat::dsd(2_822_400, DsdPacking::Dop, 2);
        assert_eq!((dop.rate, dop.bits), (176_400, 24));
        assert_eq!(dop.label(), "DSD64 (DoP)");
        let native = OutputFormat::dsd(11_289_600, DsdPacking::Native, 2);
        assert_eq!((native.rate, native.bits), (352_800, 32));
        assert_eq!(native.label(), "DSD256 native");
        assert_eq!(native.dsd_info(), Some((11_289_600, DsdPacking::Native)));
        assert!(!native.as_pcm().is_dsd());
        let pcm = OutputFormat::from_device_format((705_600, 24), 2, None);
        assert_eq!(pcm, OutputFormat::pcm(705_600, 24, 2));
        assert_eq!(pcm.to_string(), "24-bit 705.6 kHz 2ch");
        // Old JSON without `encoding` still parses as PCM.
        let f: OutputFormat =
            serde_json::from_str(r#"{"rate":44100,"bits":24,"channels":2}"#).unwrap();
        assert_eq!(f.encoding, Encoding::Pcm);
        let j = serde_json::to_value(dop).unwrap();
        assert_eq!(j["encoding"]["kind"], "dsd");
        assert_eq!(j["encoding"]["packing"], "dop");
    }

    #[test]
    fn choose_dsd_steps_down_to_what_the_device_takes() {
        // The null output: DoP and every native layout, up to DSD1024.
        let out = null::NullOutput::new(false);
        assert_eq!(
            choose_dsd(&out, 44_100, DsdRate::Dsd256, DsdPacking::Dop, 2),
            Some((DsdRate::Dsd256, DsdPacking::Dop))
        );
        // DSD2048 is above the null device's ceiling and DoP cannot carry it anyway.
        assert_eq!(
            choose_dsd(&out, 48_000, DsdRate::Dsd2048, DsdPacking::Native, 2),
            Some((DsdRate::Dsd1024, DsdPacking::Native))
        );
        // A DoP-only device asked for native gets DoP.
        struct DopOnly;
        impl Output for DopOnly {
            fn name(&self) -> String {
                "dop".into()
            }
            fn open(&mut self, f: OutputFormat) -> Result<OutputFormat, OutputError> {
                Ok(f)
            }
            fn supports(&self, rate: u32, bits: u32) -> bool {
                rate <= 384_000 && bits == 24
            }
            fn dsd_caps(&self) -> DsdCaps {
                DsdCaps::dop_only(None)
            }
            fn write(&mut self, _: &[i32]) -> Result<(), OutputError> {
                Ok(())
            }
            fn pause(&mut self, _: bool) -> Result<(), OutputError> {
                Ok(())
            }
            fn drain(&mut self) -> Result<(), OutputError> {
                Ok(())
            }
            fn discard(&mut self) -> Result<(), OutputError> {
                Ok(())
            }
            fn close(&mut self) {}
            fn format(&self) -> Option<OutputFormat> {
                None
            }
        }
        // 384 kHz PCM ceiling → DoP carries at most DSD128 (f/16 = 352.8 k).
        assert_eq!(
            choose_dsd(&DopOnly, 44_100, DsdRate::Dsd512, DsdPacking::Native, 2),
            Some((DsdRate::Dsd128, DsdPacking::Dop))
        );
        assert_eq!(
            DsdCaps::dop_only(Some(5_644_800)).summary(),
            "DoP · up to DSD128"
        );
        assert_eq!(DsdCaps::NONE.summary(), "no DSD");
        assert_eq!(
            split_backend("endpoint://kitchen", ""),
            ("endpoint", "kitchen")
        );
        assert_eq!(split_backend("alsa", "hw:1,0"), ("alsa", "hw:1,0"));
    }
}
