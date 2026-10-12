//! Audio outputs behind one trait, so the engine does not care whether frames go to a
//! counter, the OS mixer, a raw ALSA device or a WASAPI exclusive stream.

pub mod null;

#[cfg(all(feature = "alsa", target_os = "linux"))]
pub mod alsa_backend;
#[cfg(feature = "cpal")]
pub mod cpal_backend;
#[cfg(all(feature = "wasapi", windows))]
pub mod wasapi_backend;

use serde::{Deserialize, Serialize};
use std::fmt;

/// Sample rates worth probing a device for.
pub const STANDARD_RATES: &[u32] = &[
    44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000, 705600, 768000,
];

/// The format a stream is opened with: interleaved integers of `bits` valid bits
/// (16 in `i16`, 24 or 32 in `i32` words) at `rate` Hz.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputFormat {
    /// Sample rate in Hz.
    pub rate: u32,
    /// Valid bits per sample: 16, 24 or 32.
    pub bits: u32,
    /// Channels.
    pub channels: u32,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-bit {} kHz {}ch",
            self.bits,
            cadenza_dsp::signal_path::rate_label(self.rate),
            self.channels
        )
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
pub struct DeviceInfo {
    /// Backend id (`null`, `cpal`, `alsa`, `wasapi`).
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
}

/// An audio sink.
pub trait Output: Send {
    /// `backend · device` name for the signal path.
    fn name(&self) -> String;

    /// Open (or reopen) the stream in `format`. Returns the format actually opened, which
    /// may differ in bit depth when the device needs a different container.
    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError>;

    /// Whether `rate`/`bits` can be opened (best effort without opening).
    fn supports(&self, rate: u32, bits: u32) -> bool;

    /// Queue interleaved frames in the open format; blocks while the device is busy, so
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
    v
}

/// The backend to use when the configuration does not say: WASAPI exclusive on Windows if
/// compiled, ALSA on Linux if compiled, otherwise cpal, otherwise null.
pub fn default_backend() -> &'static str {
    #[cfg(all(feature = "wasapi", windows))]
    {
        return "wasapi";
    }
    #[cfg(all(feature = "alsa", target_os = "linux", not(feature = "cpal")))]
    {
        return "alsa";
    }
    #[cfg(feature = "cpal")]
    {
        return "cpal";
    }
    #[allow(unreachable_code)]
    "null"
}

/// Every device of every compiled backend.
pub fn list_devices() -> Vec<DeviceInfo> {
    let mut out = vec![null::device_info()];
    #[cfg(feature = "cpal")]
    out.extend(cpal_backend::list_devices());
    #[cfg(all(feature = "alsa", target_os = "linux"))]
    out.extend(alsa_backend::list_devices());
    #[cfg(all(feature = "wasapi", windows))]
    out.extend(wasapi_backend::list_devices());
    out
}

/// Create an output for `backend` / `device` (empty device = the backend's default).
pub fn open_backend(backend: &str, device: &str) -> Result<Box<dyn Output>, OutputError> {
    match backend {
        "null" => Ok(Box::new(null::NullOutput::new(true))),
        #[cfg(feature = "cpal")]
        "cpal" => Ok(Box::new(cpal_backend::CpalOutput::new(device)?)),
        #[cfg(all(feature = "alsa", target_os = "linux"))]
        "alsa" => Ok(Box::new(alsa_backend::AlsaOutput::new(device)?)),
        #[cfg(all(feature = "wasapi", windows))]
        "wasapi" => Ok(Box::new(wasapi_backend::WasapiOutput::new(device)?)),
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
            while r <= max_rate {
                if r >= source_rate && r % source_rate == 0 && output.supports(r, bits) {
                    best = r;
                }
                r *= 2;
            }
            best
        }
    }
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
    }
}
