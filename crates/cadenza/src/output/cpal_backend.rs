//! Shared-mode output through `cpal` (ALSA/PulseAudio/PipeWire, CoreAudio, WASAPI shared).
//!
//! Frames go into a ring buffer that the device callback drains; `write` blocks while more
//! than ~0.5 s is queued, which is how the engine gets paced by the device clock.
//! Samples are handed to cpal as `f32` (the mixer converts anyway) unless the device offers
//! an integer format matching the requested depth.
//!
//! **The mix format.** cpal opens WASAPI in shared mode without `AUTOCONVERTPCM`, so on
//! Windows the only stream it can open is the endpoint's mix format: typically 48 kHz with
//! the speaker layout's channel count. A 44.1 kHz track then has no config at its rate
//! (that was "format not supported by Speaker (Realtek(R) Audio): 24-bit 44.1 kHz 2ch").
//! Instead of failing, [`Output::open`] falls back to the device's default config: it
//! reports the device rate in the returned format, so the engine resamples to it, and
//! maps the engine's channels onto the device's in the callback.

use super::{DeviceInfo, DsdCaps, Output, OutputError, OutputFormat, STANDARD_RATES};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, Stream, StreamConfig};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Samples shared between the writer and cpal's callback.
struct Ring {
    /// Channels per frame in `queue` (what the engine writes).
    src_channels: usize,
    /// Channels per frame the device stream was opened with.
    dev_channels: usize,
    /// Interleaved `f32` samples waiting for the callback.
    queue: VecDeque<f32>,
    /// While set, the callback plays silence and keeps the queue.
    paused: bool,
    /// Samples the callback had to invent because the queue ran dry.
    underruns: u64,
}

/// See the module documentation.
pub struct CpalOutput {
    /// The cpal device.
    device: cpal::Device,
    /// Its name, for the status.
    device_name: String,
    /// Format of the open stream, if any.
    format: Option<OutputFormat>,
    /// The running stream, if open.
    stream: Option<Stream>,
    /// The queue, with a condvar the writer waits on while it is full.
    ring: Arc<(Mutex<Ring>, Condvar)>,
    /// `write` blocks while the queue holds more than this many samples (half a second).
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
                    src_channels: 2,
                    dev_channels: 2,
                    queue: VecDeque::new(),
                    paused: false,
                    underruns: 0,
                }),
                Condvar::new(),
            )),
            high_water: 0,
        })
    }

    /// The device's best config range covering `rate`: `f32` first, then the widest integer
    /// format.
    fn supported(&self, rate: u32) -> Option<cpal::SupportedStreamConfigRange> {
        self.supported_with(rate, None)
    }

    /// Like [`Self::supported`], limited to configs with exactly `channels` channels when
    /// given.
    fn supported_with(
        &self,
        rate: u32,
        channels: Option<u16>,
    ) -> Option<cpal::SupportedStreamConfigRange> {
        let mut best: Option<cpal::SupportedStreamConfigRange> = None;
        for cfg in self.device.supported_output_configs().ok()? {
            let channels_ok = channels.is_none_or(|c| cfg.channels() == c);
            if channels_ok && cfg.min_sample_rate().0 <= rate && rate <= cfg.max_sample_rate().0 {
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
        // The exact rate and channel count if the device takes them, otherwise its own
        // default (mix) format: the engine resamples to that rate and `fill` maps channels.
        let (sample_format, dev_rate, dev_channels) =
            match self.supported_with(format.rate, Some(format.channels as u16)) {
                Some(range) => (range.sample_format(), format.rate, format.channels as u16),
                None => {
                    let def = self.device.default_output_config().map_err(|e| {
                        log::warn!("{}: no default config: {e}", self.device_name);
                        OutputError::Unsupported {
                            device: self.device_name.clone(),
                            format,
                        }
                    })?;
                    log::warn!(
                        "{}: {format} not offered; opening the device's own {} Hz {}ch {:?}",
                        self.device_name,
                        def.sample_rate().0,
                        def.channels(),
                        def.sample_format()
                    );
                    (def.sample_format(), def.sample_rate().0, def.channels())
                }
            };
        let config = StreamConfig {
            channels: dev_channels,
            sample_rate: SampleRate(dev_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        {
            let mut ring = self.ring.0.lock().unwrap();
            ring.src_channels = format.channels.max(1) as usize;
            ring.dev_channels = dev_channels.max(1) as usize;
        }
        // What the engine must produce: the device's rate, the engine's own channels.
        let format = OutputFormat {
            rate: dev_rate,
            ..format
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
/// When the device has other channels than the engine writes, frames are remapped (see
/// [`map_frame`]).
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
    let (src, dev) = (r.src_channels, r.dev_channels);
    if src != dev {
        let mut frame = vec![0f32; src];
        for o in out.chunks_mut(dev) {
            if r.queue.len() < src {
                r.underruns += 1;
                o.fill(T::default());
                continue;
            }
            for v in frame.iter_mut() {
                *v = r.queue.pop_front().unwrap_or(0.0);
            }
            for (c, slot) in o.iter_mut().enumerate() {
                *slot = conv(map_frame(&frame, c));
            }
        }
        cv.notify_all();
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

/// Device channel `c` of an engine frame `frame`: mono goes to the first two channels,
/// otherwise channels map one to one and device channels the engine does not have are
/// silent (stereo on a 5.1/7.1 mix plays on front left/right).
fn map_frame(frame: &[f32], c: usize) -> f32 {
    match frame.len() {
        1 if c < 2 => frame[0],
        _ => frame.get(c).copied().unwrap_or(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::map_frame;

    #[test]
    fn frames_map_onto_other_channel_counts() {
        // Stereo onto 5.1: front left/right, the rest silent.
        let st = [0.25, -0.5];
        let six: Vec<f32> = (0..6).map(|c| map_frame(&st, c)).collect();
        assert_eq!(six, [0.25, -0.5, 0.0, 0.0, 0.0, 0.0]);
        // Mono onto stereo: both sides.
        assert_eq!([map_frame(&[0.1], 0), map_frame(&[0.1], 1)], [0.1, 0.1]);
        // 5.1 onto stereo: front left/right only.
        let surround = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            [map_frame(&surround, 0), map_frame(&surround, 1)],
            [1.0, 2.0]
        );
    }
}
