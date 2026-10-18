//! The null output: counts frames and, optionally, paces itself like a real device so
//! positions advance in real time. Used by the tests and on machines without a DAC.
//!
//! It pretends to be an ideal exclusive device: PCM at any rate up to 32×, DoP and every
//! native DSD layout up to DSD1024 (DSD2048 is refused, so the "your device stops at …"
//! path is exercised too). In DoP it checks the markers of what it is given, so an engine
//! test that writes a DSD stream to it catches a broken packer. With
//! [`NullOutput::capture_to`] (`device = "capture:<file>"`) every sample written is
//! appended to a file as little-endian `i32`, so tests can check what was played.
//!
//! For multi-room tests the device string also takes `ppm=<n>` (the simulated DAC clock
//! runs that much fast or slow, as real ones do) and `timeline=<file>` (each write is
//! recorded with the time its first frame plays, on [`epoch`]), separated by `;`.
//! `unplug=<file>` makes writes fail while that file exists, as an unplugged device's do.

use super::{DeviceInfo, DsdCaps, Encoding, Output, OutputError, OutputFormat, STANDARD_RATES};
use cadenza_dsp::dsd::packing::DOP_MARKERS;
use cadenza_dsp::dsd::{DsdPacking, NativeFormat};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The clock the `timeline=` recordings use (shared by every null output of a process).
pub fn epoch() -> Instant {
    static E: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *E.get_or_init(Instant::now)
}

/// Highest DSD bit rate the null device "accepts" (DSD1024, 48 k family).
pub const MAX_DSD_RATE: u32 = 49_152_000;

/// See the module documentation.
pub struct NullOutput {
    realtime: bool,
    format: Option<OutputFormat>,
    written: Arc<AtomicU64>,
    /// When the frames written so far will have "played", for pacing.
    clock: Option<Instant>,
    paused: bool,
    /// Next expected DoP marker index (DoP only).
    dop_marker: usize,
    /// DoP frames whose marker byte was wrong.
    dop_errors: u64,
    capture: Option<std::io::BufWriter<std::fs::File>>,
    /// Simulated clock error of the "DAC", parts per million.
    ppm: f64,
    /// `timeline=`: per write, the play time of its first frame (ns since [`epoch`]),
    /// the frame count and the words.
    timeline: Option<std::io::BufWriter<std::fs::File>>,
    /// `unplug=`: writes fail while this file exists.
    unplug: Option<std::path::PathBuf>,
    /// `dsd=dop`: a DAC that takes DSD only as DoP.
    dop_only: bool,
}

impl NullOutput {
    /// `realtime` makes `write` sleep so that frames "play" at the sample rate.
    pub fn new(realtime: bool) -> Self {
        Self {
            realtime,
            format: None,
            written: Arc::new(AtomicU64::new(0)),
            clock: None,
            paused: false,
            dop_marker: 0,
            dop_errors: 0,
            capture: None,
            ppm: 0.0,
            timeline: None,
            unplug: None,
            dop_only: false,
        }
    }

    fn real_rate(&self, rate: u32) -> f64 {
        rate as f64 * (1.0 + self.ppm * 1e-6)
    }

    /// Append every sample written to `path` (little-endian `i32`).
    pub fn capture_to(mut self, path: &std::path::Path) -> std::io::Result<Self> {
        self.capture = Some(std::io::BufWriter::new(std::fs::File::create(path)?));
        Ok(self)
    }

    /// A null output for a configured device name: `capture:<file>`, or `;`-separated
    /// `capture=<file>`, `ppm=<n>`, `timeline=<file>`, `unplug=<file>`, `dsd=dop`.
    pub fn for_device(realtime: bool, device: &str) -> Self {
        let mut out = Self::new(realtime);
        if let Some(path) = device.strip_prefix("capture:") {
            return match Self::new(realtime).capture_to(std::path::Path::new(path)) {
                Ok(o) => o,
                Err(e) => {
                    log::warn!("null output capture {path}: {e}");
                    out
                }
            };
        }
        for opt in device.split(';') {
            let Some((k, v)) = opt.split_once('=') else {
                continue;
            };
            let file = || {
                std::fs::File::create(v.trim())
                    .map(std::io::BufWriter::new)
                    .map_err(|e| log::warn!("null output {k}={v}: {e}"))
                    .ok()
            };
            match k.trim() {
                "ppm" => out.ppm = v.trim().parse().unwrap_or(0.0),
                "unplug" => out.unplug = Some(v.trim().into()),
                "dsd" => out.dop_only = v.trim() == "dop",
                "capture" => out.capture = file(),
                "timeline" => {
                    epoch();
                    out.timeline = file();
                }
                _ => {}
            }
        }
        out
    }

    /// A counter shared with the caller: total frames written since creation.
    pub fn counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.written)
    }

    /// Total frames written since creation.
    pub fn frames_written(&self) -> u64 {
        self.written.load(Ordering::Relaxed)
    }

    /// DoP frames seen with a wrong marker byte (0 for a correct stream).
    pub fn dop_marker_errors(&self) -> u64 {
        self.dop_errors
    }
}

/// What the null device claims to do with DSD.
pub fn dsd_caps() -> DsdCaps {
    DsdCaps {
        dop: true,
        native: vec![NativeFormat::U32Be, NativeFormat::U32Le, NativeFormat::U8],
        max_rate: Some(MAX_DSD_RATE),
    }
}

/// The device entry for the listing.
pub fn device_info() -> DeviceInfo {
    DeviceInfo {
        backend: "null".into(),
        id: "null".into(),
        name: "Null output (no device)".into(),
        rates: STANDARD_RATES.to_vec(),
        bits: vec![16, 24, 32],
        channels: None,
        default: false,
        exclusive: true,
        dsd: dsd_caps(),
    }
}

impl Output for NullOutput {
    fn name(&self) -> String {
        "null".into()
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        if !self.supports_format(&format) {
            return Err(OutputError::Unsupported {
                device: "null".into(),
                format,
            });
        }
        self.format = Some(format);
        self.clock = None;
        self.dop_marker = 0;
        self.dop_errors = 0;
        Ok(format)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        (8000..=super::MAX_PCM_RATE).contains(&rate) && matches!(bits, 16 | 24 | 32)
    }

    fn dsd_caps(&self) -> DsdCaps {
        let mut c = dsd_caps();
        if self.dop_only {
            c.native.clear();
        }
        c
    }

    fn exclusive(&self) -> bool {
        true
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        if self.unplug.as_ref().is_some_and(|f| f.exists()) {
            return Err(OutputError::Backend("the device is gone".into()));
        }
        let fmt = self
            .format
            .ok_or_else(|| OutputError::Backend("null output not open".into()))?;
        let ch = fmt.channels.max(1) as usize;
        let n = frames.len() / ch;
        if let Encoding::Dsd {
            packing: DsdPacking::Dop,
            ..
        } = fmt.encoding
        {
            for frame in frames.chunks(ch) {
                let want = DOP_MARKERS[self.dop_marker];
                self.dop_marker ^= 1;
                if frame
                    .iter()
                    .any(|w| ((*w as u32 >> 16) & 0xFF) as u8 != want)
                {
                    self.dop_errors += 1;
                }
            }
        }
        self.written.fetch_add(n as u64, Ordering::Relaxed);
        if let Some(c) = &mut self.capture {
            use std::io::Write;
            for s in frames {
                c.write_all(&s.to_le_bytes())
                    .map_err(|e| OutputError::Backend(e.to_string()))?;
            }
            c.flush().map_err(|e| OutputError::Backend(e.to_string()))?;
        }
        if self.realtime {
            // Pretend to hold a quarter second of buffer: sleep until the queue is below that.
            let dur = Duration::from_secs_f64(n as f64 / self.real_rate(fmt.rate));
            let now = Instant::now();
            let begin = match self.clock {
                Some(c) if c > now => c,
                _ => now,
            };
            let end = begin + dur;
            self.clock = Some(end);
            if let Some(t) = &mut self.timeline {
                use std::io::Write;
                let at = begin.duration_since(epoch()).as_nanos() as u64;
                let mut rec = Vec::with_capacity(12 + frames.len() * 4);
                rec.extend(at.to_le_bytes());
                rec.extend((n as u32).to_le_bytes());
                for w in frames {
                    rec.extend(w.to_le_bytes());
                }
                t.write_all(&rec)
                    .and_then(|_| t.flush())
                    .map_err(|e| OutputError::Backend(e.to_string()))?;
            }
            let ahead = end.saturating_duration_since(now);
            if ahead > Duration::from_millis(250) {
                std::thread::sleep(ahead - Duration::from_millis(250));
            }
        }
        Ok(())
    }

    fn pause(&mut self, paused: bool) -> Result<(), OutputError> {
        self.paused = paused;
        Ok(())
    }

    fn drain(&mut self) -> Result<(), OutputError> {
        if let (true, Some(c)) = (self.realtime, self.clock) {
            let now = Instant::now();
            if c > now {
                std::thread::sleep(c - now);
            }
        }
        self.clock = None;
        Ok(())
    }

    fn discard(&mut self) -> Result<(), OutputError> {
        self.clock = None;
        self.dop_marker = 0;
        Ok(())
    }

    fn close(&mut self) {
        self.format = None;
        self.clock = None;
    }

    fn queued_frames(&self) -> usize {
        match (self.format, self.clock) {
            (Some(f), Some(c)) => {
                (c.saturating_duration_since(Instant::now()).as_secs_f64() * self.real_rate(f.rate))
                    as usize
            }
            _ => 0,
        }
    }

    fn frames_written(&self) -> u64 {
        self.written.load(Ordering::Relaxed)
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cadenza_dsp::dsd::{DsdRate, Packer};

    #[test]
    fn counts_frames() {
        let mut o = NullOutput::new(false);
        o.open(OutputFormat::pcm(48000, 24, 2)).unwrap();
        o.write(&[0; 2000]).unwrap();
        o.write(&[0; 6]).unwrap();
        assert_eq!(o.frames_written(), 1003);
        assert!(o.supports(768_000, 32));
        assert!(o.supports(1_536_000, 32));
        assert!(!o.supports(48000, 20));
        assert!(!o.supports(3_072_000, 24));
    }

    #[test]
    fn accepts_dsd_up_to_1024_and_checks_dop_markers() {
        let mut o = NullOutput::new(false);
        assert!(o.supports_format(&OutputFormat::dsd(
            DsdRate::Dsd1024.hz(44_100),
            DsdPacking::Native,
            2
        )));
        assert!(!o.supports_format(&OutputFormat::dsd(
            DsdRate::Dsd2048.hz(44_100),
            DsdPacking::Native,
            2
        )));
        let f = OutputFormat::dsd(2_822_400, DsdPacking::Dop, 2);
        assert_eq!(o.open(f).unwrap(), f);
        // A correctly packed DoP stream has no marker errors…
        let mut packer = Packer::new(DsdPacking::Dop, 2);
        let bytes = vec![0x69u8; 64];
        let mut words = Vec::new();
        packer.push(&[&bytes, &bytes], &mut words);
        o.write(&words[..16]).unwrap();
        o.write(&words[16..]).unwrap();
        assert_eq!(o.frames_written(), 32);
        assert_eq!(o.dop_marker_errors(), 0);
        // …and plain PCM written in DoP mode is caught.
        o.write(&[0i32; 8]).unwrap();
        assert_eq!(o.dop_marker_errors(), 4);
        assert_eq!(
            device_info().dsd.max_rate_label().as_deref(),
            Some("DSD1024")
        );
    }
}
