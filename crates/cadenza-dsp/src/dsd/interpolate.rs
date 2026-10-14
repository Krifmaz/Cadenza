//! The short last stage of the `staged` oversampling path: an integer ×L polyphase
//! interpolator (L = 2 … 16) with a Kaiser windowed sinc.
//!
//! The long FIR before it has already band-limited the signal to the source Nyquist, so
//! this stage only has to keep the passband flat up to there and remove the images at
//! `f_in ± passband`; with a transition band that wide a few hundred taps give 150 dB.

use super::DsdError;
use crate::fir::{kaiser_beta, windowed_sinc};

/// Stopband attenuation of the short stage, dB.
pub const ATTENUATION_DB: f64 = 150.0;

/// Streaming ×L interpolator for `channels` planar channels.
#[derive(Clone, Debug)]
pub struct ShortInterpolator {
    in_rate: u32,
    l: usize,
    taps: usize,
    passband_hz: f64,
    /// `phases[p][k] = h[k·L + p] · L`.
    phases: Vec<Vec<f64>>,
    /// Per channel history of input samples, newest first (length = taps per phase).
    hist: Vec<Vec<f64>>,
}

impl ShortInterpolator {
    /// Interpolate from `in_rate` by `l`, keeping everything below `passband_hz` (the
    /// source Nyquist; clamped to `0.45 × in_rate`).
    pub fn new(in_rate: u32, l: usize, passband_hz: f64, channels: usize) -> Result<Self, DsdError> {
        if !(2..=16).contains(&l) {
            return Err(DsdError::Invalid(format!("short stage factor {l} must be 2…16")));
        }
        if in_rate == 0 || channels == 0 {
            return Err(DsdError::Invalid("rate or channel count is zero".into()));
        }
        let out_rate = in_rate as f64 * l as f64;
        let pass = passband_hz.clamp(1000.0, 0.45 * in_rate as f64);
        let stop = in_rate as f64 - pass;
        let cutoff = 0.5 * (pass + stop) / out_rate;
        let transition = (stop - pass) / out_rate;
        let mut taps = ((ATTENUATION_DB - 8.0) / (2.285 * 2.0 * std::f64::consts::PI * transition)).ceil()
            as usize
            + 1;
        taps = taps.div_ceil(l).max(8) * l;
        let h = windowed_sinc(taps, cutoff, kaiser_beta(ATTENUATION_DB));
        let per_phase = taps / l;
        let phases = (0..l)
            .map(|p| (0..per_phase).map(|k| h[k * l + p] * l as f64).collect())
            .collect();
        Ok(Self {
            in_rate,
            l,
            taps,
            passband_hz: pass,
            phases,
            hist: vec![vec![0.0; per_phase]; channels],
        })
    }

    /// Interpolation factor.
    pub fn factor(&self) -> usize {
        self.l
    }
    /// Filter length.
    pub fn taps(&self) -> usize {
        self.taps
    }
    /// Output rate in Hz.
    pub fn out_rate(&self) -> u32 {
        self.in_rate * self.l as u32
    }
    /// Passband edge actually used, Hz.
    pub fn passband_hz(&self) -> f64 {
        self.passband_hz
    }
    /// Group delay in output samples.
    pub fn delay_samples(&self) -> usize {
        (self.taps - 1) / 2
    }
    /// Channels.
    pub fn channels(&self) -> usize {
        self.hist.len()
    }

    /// Interpolate one channel, appending `input.len() × L` samples to `out`.
    pub fn process_channel(&mut self, channel: usize, input: &[f64], out: &mut Vec<f64>) {
        let hist = &mut self.hist[channel];
        let n = hist.len();
        out.reserve(input.len() * self.l);
        for &x in input {
            for k in (1..n).rev() {
                hist[k] = hist[k - 1];
            }
            hist[0] = x;
            for phase in &self.phases {
                let mut acc = 0.0;
                for (c, h) in phase.iter().zip(hist.iter()) {
                    acc += c * h;
                }
                out.push(acc);
            }
        }
    }

    /// Forget the history.
    pub fn reset(&mut self) {
        for h in &mut self.hist {
            h.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::magnitude_db;
    use std::f64::consts::PI;

    #[test]
    fn images_are_suppressed_and_the_passband_is_flat() {
        let in_rate = 352_800;
        let it = ShortInterpolator::new(in_rate, 8, 22_050.0, 1).unwrap();
        assert_eq!(it.out_rate(), 2_822_400);
        assert_eq!(it.taps() % 8, 0);
        // Impulse response → frequency response.
        let mut it2 = it.clone();
        let mut h = Vec::new();
        it2.process_channel(0, &[1.0], &mut h);
        it2.process_channel(0, &vec![0.0; it.taps() / 8], &mut h);
        let nfft = 1 << 16;
        let db = magnitude_db(&h, nfft);
        let bin = |hz: f64| (hz / 2_822_400.0 * nfft as f64).round() as usize;
        let gain0 = db[0];
        for (b, d) in db.iter().enumerate().take(bin(22_050.0) + 1) {
            assert!((d - gain0).abs() < 1e-4, "passband bin {b}: {}", d - gain0);
        }
        for (b, d) in db.iter().enumerate().take(nfft / 2 + 1).skip(bin(352_800.0 - 22_050.0)) {
            assert!(d - gain0 < -145.0, "stopband bin {b}: {}", d - gain0);
        }
    }

    #[test]
    fn a_tone_keeps_its_amplitude() {
        let in_rate = 352_800;
        let mut it = ShortInterpolator::new(in_rate, 8, 22_050.0, 2).unwrap();
        // 4410 Hz: exactly 80 input samples per cycle, so RMS windows are whole cycles.
        let x: Vec<f64> = (0..4000)
            .map(|i| 0.5 * (2.0 * PI * 4410.0 * i as f64 / in_rate as f64).sin())
            .collect();
        let mut y = Vec::new();
        it.process_channel(1, &x, &mut y);
        assert_eq!(y.len(), 32_000);
        let mid = &y[8000..8000 + 640 * 25];
        let rms = (mid.iter().map(|v| v * v).sum::<f64>() / mid.len() as f64).sqrt();
        assert!((20.0 * (rms * 2f64.sqrt() / 0.5).log10()).abs() < 1e-3);
    }

    #[test]
    fn factor_is_validated() {
        assert!(ShortInterpolator::new(352_800, 1, 22_050.0, 1).is_err());
        assert!(ShortInterpolator::new(352_800, 32, 22_050.0, 1).is_err());
        assert!(ShortInterpolator::new(352_800, 16, 22_050.0, 1).is_ok());
    }
}
