//! Direct ALSA output (`hw:` devices): exclusive access, the exact sample rate and an
//! integer format with no conversion in between, which is the Linux bit-perfect path.
//!
//! `hw:` devices refuse anything they cannot do natively; `plughw:` would silently
//! resample, so the configuration should name `hw:CARD=xxx,DEV=0` (see `cadenza devices`).
//!
//! **DSD.** DoP is 24-bit PCM at `f_dsd / 16` and goes through the S24 path untouched.
//! Native DSD uses ALSA's DSD formats; which the driver has is probed with
//! `snd_pcm_hw_params_test_format`. The ALSA rate for `DSD_U32_*` is `f_dsd / 32` (one
//! 32-bit word per channel per frame) and for `DSD_U8` it is `f_dsd / 8` (one byte per
//! channel per frame). The pipeline's native words are `DSD_U32_BE` as `i32`
//! (`cadenza_dsp::dsd::packing`), so for every layout the words are serialised with
//! `words_to_bytes` and written as raw bytes, which makes the byte order explicit instead
//! of relying on the host's endianness.

use super::{DeviceInfo, DsdCaps, Encoding, Output, OutputError, OutputFormat, STANDARD_RATES};
use alsa::pcm::{Access, Format, HwParams, State, PCM};
use alsa::{Direction, ValueOr};
use cadenza_dsp::dsd::packing::words_to_bytes;
use cadenza_dsp::dsd::{DsdPacking, DsdRate, NativeFormat};

/// See the module documentation.
pub struct AlsaOutput {
    /// ALSA device name, e.g. `hw:1,0`.
    device: String,
    /// The open PCM, if any.
    pcm: Option<PCM>,
    /// Format the PCM was opened with.
    format: Option<OutputFormat>,
    /// ALSA sample format of the open PCM.
    alsa_format: Format,
    /// Native DSD layout in use (None for PCM and DoP).
    native: Option<NativeFormat>,
    /// Size of the device buffer in frames.
    buffer_frames: usize,
}

/// An ALSA error as an output error.
fn err(e: alsa::Error) -> OutputError {
    OutputError::Backend(e.to_string())
}

/// ALSA sample format for a bit depth: S24 lives in 32-bit words, LSB-justified, which is
/// exactly how the quantiser produces 24-bit values.
fn alsa_format(bits: u32) -> Option<Format> {
    match bits {
        16 => Some(Format::s16()),
        24 => Some(Format::s24()),
        32 => Some(Format::s32()),
        _ => None,
    }
}

/// ALSA format for a native DSD layout.
fn native_format(n: NativeFormat) -> Format {
    match n {
        NativeFormat::U32Be => Format::DSDU32BE,
        NativeFormat::U32Le => Format::DSDU32LE,
        NativeFormat::U8 => Format::DSDU8,
    }
}

/// Native layouts in order of preference (U32 moves four bytes per frame; U8 is the
/// fallback for drivers that only have the byte format).
const NATIVE_PREFERENCE: [NativeFormat; 3] =
    [NativeFormat::U32Be, NativeFormat::U32Le, NativeFormat::U8];

/// The ALSA rate and format a native DSD stream of `dsd_rate` opens with.
fn native_params(dsd_rate: u32, n: NativeFormat) -> (u32, Format) {
    match n {
        NativeFormat::U8 => (dsd_rate / 8, Format::DSDU8),
        NativeFormat::U32Be | NativeFormat::U32Le => (dsd_rate / 32, native_format(n)),
    }
}

/// `(rate, format, native layout)` the device must be configured with for `format`.
fn plan(format: OutputFormat, hwp: &HwParams) -> Option<(u32, Format, Option<NativeFormat>)> {
    match format.encoding {
        Encoding::Pcm => Some((format.rate, alsa_format(format.bits)?, None)),
        Encoding::Dsd {
            packing: DsdPacking::Dop,
            ..
        } => Some((format.rate, Format::s24(), None)),
        Encoding::Dsd {
            rate,
            packing: DsdPacking::Native,
        } => NATIVE_PREFERENCE.iter().copied().find_map(|n| {
            let (r, f) = native_params(rate, n);
            (hwp.test_format(f).is_ok() && hwp.test_rate(r).is_ok()).then_some((r, f, Some(n)))
        }),
    }
}

impl AlsaOutput {
    /// Prepare for `device` (`hw:0,0`, `hw:CARD=DAC,DEV=0`; empty = `default`).
    pub fn new(device: &str) -> Result<Self, OutputError> {
        let device = if device.is_empty() {
            "default".to_string()
        } else {
            device.to_string()
        };
        // Open once now so a typo is reported immediately.
        PCM::new(&device, Direction::Playback, false)
            .map_err(|e| OutputError::NoDevice(format!("{device}: {e}")))?;
        Ok(Self {
            device,
            pcm: None,
            format: None,
            alsa_format: Format::s32(),
            native: None,
            buffer_frames: 0,
        })
    }

    /// Set up `pcm` for interleaved `fmt` at `rate` with no resampling; returns the buffer
    /// size in frames.
    fn configure(pcm: &PCM, rate: u32, channels: u32, fmt: Format) -> Result<usize, alsa::Error> {
        let hwp = HwParams::any(pcm)?;
        hwp.set_rate_resample(false)?;
        hwp.set_access(Access::RWInterleaved)?;
        hwp.set_format(fmt)?;
        hwp.set_channels(channels)?;
        hwp.set_rate(rate, ValueOr::Nearest)?;
        if hwp.get_rate()? != rate {
            return Err(alsa::Error::unsupported("exact sample rate"));
        }
        // Half a second of buffer in eight periods: deep enough for a user-space DSP box,
        // shallow enough that pause and seek feel immediate.
        let buffer = hwp.set_buffer_size_near((rate / 2) as alsa::pcm::Frames)?;
        hwp.set_period_size_near(buffer / 8, ValueOr::Nearest)?;
        pcm.hw_params(&hwp)?;
        Ok(buffer as usize)
    }

    /// Run `f` on a fresh set of hardware parameters for the device (probing only).
    fn with_hwp<T>(&self, f: impl FnOnce(&HwParams) -> T) -> Option<T> {
        let pcm = PCM::new(&self.device, Direction::Playback, true).ok()?;
        let hwp = HwParams::any(&pcm).ok()?;
        let _ = hwp.set_rate_resample(false);
        Some(f(&hwp))
    }
}

/// Every playback `hw:` device with the rates and formats it accepts.
pub fn list_devices() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    let Ok(hints) = alsa::device_name::HintIter::new_str(None, "pcm") else {
        return out;
    };
    for hint in hints {
        let Some(name) = hint.name else { continue };
        if !name.starts_with("hw:") || matches!(hint.direction, Some(Direction::Capture)) {
            continue;
        }
        let desc = hint.desc.unwrap_or_default().replace('\n', " · ");
        let (rates, bits, channels, dsd) = probe(&name);
        out.push(DeviceInfo {
            backend: "alsa".into(),
            id: name.clone(),
            name: if desc.is_empty() {
                name
            } else {
                format!("{desc} ({name})")
            },
            rates,
            bits,
            channels,
            default: false,
            exclusive: true,
            dsd,
        });
    }
    out
}

/// DSD capabilities from a device's hardware parameters: DoP wherever 24-bit PCM reaches
/// `f_dsd/16`, native where the driver lists a DSD format, `max_rate` the highest DSD rate
/// of either family that any packing reaches.
fn dsd_caps(hwp: &HwParams) -> DsdCaps {
    let dop_ok = hwp.test_format(Format::s24()).is_ok();
    let native: Vec<NativeFormat> = NATIVE_PREFERENCE
        .iter()
        .copied()
        .filter(|n| hwp.test_format(native_format(*n)).is_ok())
        .collect();
    let mut max_rate = None;
    for &r in DsdRate::catalog() {
        for base in [44_100u32, 48_000] {
            let hz = r.hz(base);
            let dop = dop_ok && r.dop_possible() && hwp.test_rate(hz / 16).is_ok();
            let nat = native.iter().any(|n| {
                let (rate, _) = native_params(hz, *n);
                hwp.test_rate(rate).is_ok()
            });
            if dop || nat {
                max_rate = Some(max_rate.map_or(hz, |m: u32| m.max(hz)));
            }
        }
    }
    DsdCaps {
        dop: dop_ok,
        native,
        max_rate,
    }
}

/// Probe device `name` without opening it for playback: `(rates, bit depths, max
/// channels, DSD formats)`.
fn probe(name: &str) -> (Vec<u32>, Vec<u32>, Option<u32>, DsdCaps) {
    let Ok(pcm) = PCM::new(name, Direction::Playback, true) else {
        return (vec![], vec![], None, DsdCaps::NONE);
    };
    let Ok(hwp) = HwParams::any(&pcm) else {
        return (vec![], vec![], None, DsdCaps::NONE);
    };
    let _ = hwp.set_rate_resample(false);
    let rates = STANDARD_RATES
        .iter()
        .copied()
        .filter(|r| hwp.test_rate(*r).is_ok())
        .collect();
    let bits = [16u32, 24, 32]
        .into_iter()
        .filter(|b| alsa_format(*b).is_some_and(|f| hwp.test_format(f).is_ok()))
        .collect();
    let dsd = dsd_caps(&hwp);
    (rates, bits, hwp.get_channels_max().ok(), dsd)
}

impl Output for AlsaOutput {
    fn name(&self) -> String {
        format!("alsa · {}", self.device)
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        self.close();
        let unsupported = || OutputError::Unsupported {
            device: self.device.clone(),
            format,
        };
        let pcm = PCM::new(&self.device, Direction::Playback, false).map_err(err)?;
        let (rate, fmt, native) = {
            let hwp = HwParams::any(&pcm).map_err(err)?;
            let _ = hwp.set_rate_resample(false);
            plan(format, &hwp).ok_or_else(unsupported)?
        };
        let buffer = Self::configure(&pcm, rate, format.channels, fmt).map_err(|e| {
            log::warn!("alsa {}: {e}", self.device);
            unsupported()
        })?;
        pcm.prepare().map_err(err)?;
        if let Some(n) = native {
            log::info!(
                "alsa {}: native DSD as {} at {rate} Hz",
                self.device,
                n.id()
            );
        }
        self.pcm = Some(pcm);
        self.format = Some(format);
        self.alsa_format = fmt;
        self.native = native;
        self.buffer_frames = buffer;
        Ok(format)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        let Some(fmt) = alsa_format(bits) else {
            return false;
        };
        self.with_hwp(|hwp| hwp.test_rate(rate).is_ok() && hwp.test_format(fmt).is_ok())
            .unwrap_or(false)
    }

    fn dsd_caps(&self) -> DsdCaps {
        self.with_hwp(dsd_caps).unwrap_or(DsdCaps::NONE)
    }

    fn supports_format(&self, format: &OutputFormat) -> bool {
        self.with_hwp(|hwp| match plan(*format, hwp) {
            Some((rate, fmt, _)) => hwp.test_rate(rate).is_ok() && hwp.test_format(fmt).is_ok(),
            None => false,
        })
        .unwrap_or(false)
    }

    fn exclusive(&self) -> bool {
        true
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        let pcm = self
            .pcm
            .as_ref()
            .ok_or_else(|| OutputError::Backend("alsa device not open".into()))?;
        let fmt = self.format.unwrap();
        let ch = fmt.channels as usize;
        match self.native {
            // Native DSD: explicit byte order, raw byte frames. U32 → one frame per word,
            // U8 → four one-byte frames per word.
            Some(n) => {
                let bytes = words_to_bytes(frames, ch, n);
                let frame_bytes = ch * if n == NativeFormat::U8 { 1 } else { 4 };
                let total = bytes.len() / frame_bytes;
                let io = pcm.io_bytes();
                let mut written = 0;
                while written < total {
                    match io.writei(&bytes[written * frame_bytes..]) {
                        Ok(k) => written += k,
                        Err(e) => {
                            log::warn!("alsa write: {e}; recovering");
                            pcm.try_recover(e, true).map_err(err)?;
                        }
                    }
                }
                Ok(())
            }
            None => {
                let mut written = 0;
                let total = frames.len() / ch;
                while written < total {
                    let rest = &frames[written * ch..];
                    let r = if fmt.bits == 16 {
                        let io = pcm.io_i16().map_err(err)?;
                        let buf: Vec<i16> = rest.iter().map(|&v| v as i16).collect();
                        io.writei(&buf)
                    } else {
                        let io = pcm.io_i32().map_err(err)?;
                        io.writei(rest)
                    };
                    match r {
                        Ok(n) => written += n,
                        Err(e) => {
                            // EPIPE = underrun; recover and keep going instead of dying.
                            log::warn!("alsa write: {e}; recovering");
                            pcm.try_recover(e, true).map_err(err)?;
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn pause(&mut self, paused: bool) -> Result<(), OutputError> {
        if let Some(pcm) = &self.pcm {
            if pcm.pause(paused).is_err() {
                // Many USB DACs cannot pause: drop the buffer and re-prepare instead.
                if paused {
                    pcm.drop().map_err(err)?;
                } else if pcm.state() != State::Running {
                    pcm.prepare().map_err(err)?;
                }
            }
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), OutputError> {
        if let Some(pcm) = &self.pcm {
            pcm.drain().map_err(err)?;
            pcm.prepare().map_err(err)?;
        }
        Ok(())
    }

    fn discard(&mut self) -> Result<(), OutputError> {
        if let Some(pcm) = &self.pcm {
            pcm.drop().map_err(err)?;
            pcm.prepare().map_err(err)?;
        }
        Ok(())
    }

    fn close(&mut self) {
        if let Some(pcm) = self.pcm.take() {
            let _ = pcm.drop();
        }
        self.format = None;
        self.native = None;
    }

    fn queued_frames(&self) -> usize {
        let frames = self
            .pcm
            .as_ref()
            .and_then(|p| p.delay().ok())
            .map(|d| d.max(0) as usize)
            .unwrap_or(0);
        // DSD_U8 frames are a quarter of the pipeline's 32-bit word frames.
        if self.native == Some(NativeFormat::U8) {
            frames / 4
        } else {
            frames
        }
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_rates_follow_the_alsa_conventions() {
        // DSD256 (44.1 family): U32 frames at f/32, U8 bytes at f/8.
        let (r, f) = native_params(11_289_600, NativeFormat::U32Be);
        assert_eq!((r, f), (352_800, Format::DSDU32BE));
        let (r, f) = native_params(11_289_600, NativeFormat::U8);
        assert_eq!((r, f), (1_411_200, Format::DSDU8));
        let (r, f) = native_params(3_072_000, NativeFormat::U32Le);
        assert_eq!((r, f), (96_000, Format::DSDU32LE));
    }
}
