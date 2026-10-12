//! Direct ALSA output (`hw:` devices): exclusive access, the exact sample rate and an
//! integer format with no conversion in between, which is the Linux bit-perfect path.
//!
//! `hw:` devices refuse anything they cannot do natively; `plughw:` would silently
//! resample, so the configuration should name `hw:CARD=xxx,DEV=0` (see `cadenza devices`).

use super::{DeviceInfo, Output, OutputError, OutputFormat, STANDARD_RATES};
use alsa::pcm::{Access, Format, HwParams, State, PCM};
use alsa::{Direction, ValueOr};

/// See the module documentation.
pub struct AlsaOutput {
    device: String,
    pcm: Option<PCM>,
    format: Option<OutputFormat>,
    alsa_format: Format,
    buffer_frames: usize,
}

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
            buffer_frames: 0,
        })
    }

    fn configure(pcm: &PCM, format: OutputFormat, fmt: Format) -> Result<usize, alsa::Error> {
        let hwp = HwParams::any(pcm)?;
        hwp.set_rate_resample(false)?;
        hwp.set_access(Access::RWInterleaved)?;
        hwp.set_format(fmt)?;
        hwp.set_channels(format.channels)?;
        hwp.set_rate(format.rate, ValueOr::Nearest)?;
        if hwp.get_rate()? != format.rate {
            return Err(alsa::Error::unsupported("exact sample rate"));
        }
        // Half a second of buffer in eight periods: deep enough for a user-space DSP box,
        // shallow enough that pause and seek feel immediate.
        let buffer = hwp.set_buffer_size_near((format.rate / 2) as alsa::pcm::Frames)?;
        hwp.set_period_size_near(buffer / 8, ValueOr::Nearest)?;
        pcm.hw_params(&hwp)?;
        Ok(buffer as usize)
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
        let (rates, bits, channels) = probe(&name);
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
        });
    }
    out
}

fn probe(name: &str) -> (Vec<u32>, Vec<u32>, Option<u32>) {
    let Ok(pcm) = PCM::new(name, Direction::Playback, true) else {
        return (vec![], vec![], None);
    };
    let Ok(hwp) = HwParams::any(&pcm) else {
        return (vec![], vec![], None);
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
    (rates, bits, hwp.get_channels_max().ok())
}

impl Output for AlsaOutput {
    fn name(&self) -> String {
        format!("alsa · {}", self.device)
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        self.close();
        let fmt = alsa_format(format.bits).ok_or(OutputError::Unsupported {
            device: self.device.clone(),
            format,
        })?;
        let pcm = PCM::new(&self.device, Direction::Playback, false).map_err(err)?;
        let buffer = Self::configure(&pcm, format, fmt).map_err(|e| {
            log::warn!("alsa {}: {e}", self.device);
            OutputError::Unsupported {
                device: self.device.clone(),
                format,
            }
        })?;
        pcm.prepare().map_err(err)?;
        self.pcm = Some(pcm);
        self.format = Some(format);
        self.alsa_format = fmt;
        self.buffer_frames = buffer;
        Ok(format)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        let Some(fmt) = alsa_format(bits) else {
            return false;
        };
        let Ok(pcm) = PCM::new(&self.device, Direction::Playback, true) else {
            return false;
        };
        let Ok(hwp) = HwParams::any(&pcm) else {
            return false;
        };
        let _ = hwp.set_rate_resample(false);
        hwp.test_rate(rate).is_ok() && hwp.test_format(fmt).is_ok()
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        let pcm = self
            .pcm
            .as_ref()
            .ok_or_else(|| OutputError::Backend("alsa device not open".into()))?;
        let fmt = self.format.unwrap();
        let ch = fmt.channels as usize;
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
    }

    fn queued_frames(&self) -> usize {
        self.pcm
            .as_ref()
            .and_then(|p| p.delay().ok())
            .map(|d| d.max(0) as usize)
            .unwrap_or(0)
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }
}
