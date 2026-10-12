//! The null output: counts frames and, optionally, paces itself like a real device so
//! positions advance in real time. Used by the tests and on machines without a DAC.

use super::{DeviceInfo, Output, OutputError, OutputFormat, STANDARD_RATES};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// See the module documentation.
pub struct NullOutput {
    realtime: bool,
    format: Option<OutputFormat>,
    written: Arc<AtomicU64>,
    /// When the frames written so far will have "played", for pacing.
    clock: Option<Instant>,
    paused: bool,
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
        }
    }

    /// A counter shared with the caller: total frames written since creation.
    pub fn counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.written)
    }

    /// Total frames written since creation.
    pub fn frames_written(&self) -> u64 {
        self.written.load(Ordering::Relaxed)
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
    }
}

impl Output for NullOutput {
    fn name(&self) -> String {
        "null".into()
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        self.format = Some(format);
        self.clock = None;
        Ok(format)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        (8000..=1_536_000).contains(&rate) && matches!(bits, 16 | 24 | 32)
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        let fmt = self
            .format
            .ok_or_else(|| OutputError::Backend("null output not open".into()))?;
        let n = frames.len() / fmt.channels.max(1) as usize;
        self.written.fetch_add(n as u64, Ordering::Relaxed);
        if self.realtime {
            // Pretend to hold a quarter second of buffer: sleep until the queue is below that.
            let dur = Duration::from_secs_f64(n as f64 / fmt.rate as f64);
            let now = Instant::now();
            let end = match self.clock {
                Some(c) if c > now => c + dur,
                _ => now + dur,
            };
            self.clock = Some(end);
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
        Ok(())
    }

    fn close(&mut self) {
        self.format = None;
        self.clock = None;
    }

    fn queued_frames(&self) -> usize {
        match (self.format, self.clock) {
            (Some(f), Some(c)) => {
                (c.saturating_duration_since(Instant::now()).as_secs_f64() * f.rate as f64) as usize
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

    #[test]
    fn counts_frames() {
        let mut o = NullOutput::new(false);
        o.open(OutputFormat {
            rate: 48000,
            bits: 24,
            channels: 2,
        })
        .unwrap();
        o.write(&[0; 2000]).unwrap();
        o.write(&[0; 6]).unwrap();
        assert_eq!(o.frames_written(), 1003);
        assert!(o.supports(768_000, 32));
        assert!(!o.supports(48000, 20));
    }
}
