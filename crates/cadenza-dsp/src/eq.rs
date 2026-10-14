//! Parametric equaliser: up to 16 cascaded biquads with the Audio-EQ-Cookbook formulas
//! (Robert Bristow-Johnson), computed and run in `f64`, transposed direct form II.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Maximum number of bands a single [`Equalizer`] runs.
pub const MAX_BANDS: usize = 16;

/// Filter shape of one band.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BandKind {
    /// Bell: `gain_db` at `freq`, width from `q`.
    Peaking,
    /// Shelf below `freq`.
    LowShelf,
    /// Shelf above `freq`.
    HighShelf,
    /// 2nd-order low-pass at `freq` (`gain_db` ignored).
    LowPass,
    /// 2nd-order high-pass at `freq` (`gain_db` ignored).
    HighPass,
    /// Notch at `freq` (`gain_db` ignored).
    Notch,
}

/// One EQ band.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Band {
    /// Shape.
    pub kind: BandKind,
    /// Centre / corner frequency in Hz.
    pub freq: f64,
    /// Gain in dB (peaking and shelves).
    #[serde(default)]
    pub gain_db: f64,
    /// Quality factor (bandwidth). `0.7071` is a Butterworth pass filter.
    #[serde(default = "default_q")]
    pub q: f64,
    /// Bands can be kept in the list but switched off.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_q() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

impl Band {
    /// A peaking band.
    pub fn peaking(freq: f64, gain_db: f64, q: f64) -> Self {
        Self {
            kind: BandKind::Peaking,
            freq,
            gain_db,
            q,
            enabled: true,
        }
    }
}

/// Normalised biquad coefficients (`a0 = 1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coefficients {
    /// Feed-forward.
    pub b0: f64,
    /// Feed-forward.
    pub b1: f64,
    /// Feed-forward.
    pub b2: f64,
    /// Feedback.
    pub a1: f64,
    /// Feedback.
    pub a2: f64,
}

impl Coefficients {
    /// Cookbook coefficients for `band` at `rate` Hz.
    pub fn design(band: &Band, rate: f64) -> Self {
        let f = band.freq.clamp(1.0, rate * 0.499);
        let q = band.q.max(0.01);
        let a = 10f64.powf(band.gain_db / 40.0);
        let w0 = 2.0 * PI * f / rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let (b0, b1, b2, a0, a1, a2) = match band.kind {
            BandKind::Peaking => (
                1.0 + alpha * a,
                -2.0 * cos,
                1.0 - alpha * a,
                1.0 + alpha / a,
                -2.0 * cos,
                1.0 - alpha / a,
            ),
            BandKind::LowShelf => {
                let s = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) - (a - 1.0) * cos + s),
                    2.0 * a * ((a - 1.0) - (a + 1.0) * cos),
                    a * ((a + 1.0) - (a - 1.0) * cos - s),
                    (a + 1.0) + (a - 1.0) * cos + s,
                    -2.0 * ((a - 1.0) + (a + 1.0) * cos),
                    (a + 1.0) + (a - 1.0) * cos - s,
                )
            }
            BandKind::HighShelf => {
                let s = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) + (a - 1.0) * cos + s),
                    -2.0 * a * ((a - 1.0) + (a + 1.0) * cos),
                    a * ((a + 1.0) + (a - 1.0) * cos - s),
                    (a + 1.0) - (a - 1.0) * cos + s,
                    2.0 * ((a - 1.0) - (a + 1.0) * cos),
                    (a + 1.0) - (a - 1.0) * cos - s,
                )
            }
            BandKind::LowPass => (
                (1.0 - cos) / 2.0,
                1.0 - cos,
                (1.0 - cos) / 2.0,
                1.0 + alpha,
                -2.0 * cos,
                1.0 - alpha,
            ),
            BandKind::HighPass => (
                (1.0 + cos) / 2.0,
                -(1.0 + cos),
                (1.0 + cos) / 2.0,
                1.0 + alpha,
                -2.0 * cos,
                1.0 - alpha,
            ),
            BandKind::Notch => (1.0, -2.0 * cos, 1.0, 1.0 + alpha, -2.0 * cos, 1.0 - alpha),
        };
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Magnitude response in dB at `freq` Hz for a filter running at `rate`.
    pub fn response_db(&self, freq: f64, rate: f64) -> f64 {
        let w = 2.0 * PI * freq / rate;
        let z1 = rustfft::num_complex::Complex::new(w.cos(), -w.sin());
        let z2 = z1 * z1;
        let num = self.b0 + self.b1 * z1 + self.b2 * z2;
        let den = 1.0 + self.a1 * z1 + self.a2 * z2;
        20.0 * (num / den).norm().log10()
    }
}

/// One biquad with per-channel state.
#[derive(Clone, Debug)]
struct Biquad {
    c: Coefficients,
    /// `(z1, z2)` per channel.
    state: Vec<(f64, f64)>,
}

impl Biquad {
    #[inline]
    fn tick(&mut self, ch: usize, x: f64) -> f64 {
        let (z1, z2) = self.state[ch];
        let y = self.c.b0 * x + z1;
        let mut z1 = self.c.b1 * x - self.c.a1 * y + z2;
        let mut z2 = self.c.b2 * x - self.c.a2 * y;
        // Denormal guard (see `matrix::crossover::Section::tick`): no subnormal states
        // during digital silence, so the per-sample cost stays constant.
        if z1.abs() < crate::matrix::DENORMAL_FLOOR {
            z1 = 0.0;
        }
        if z2.abs() < crate::matrix::DENORMAL_FLOOR {
            z2 = 0.0;
        }
        self.state[ch] = (z1, z2);
        y
    }
}

/// A cascade of enabled bands for one stream format.
#[derive(Clone, Debug)]
pub struct Equalizer {
    rate: f64,
    channels: usize,
    bands: Vec<Band>,
    stages: Vec<Biquad>,
}

impl Equalizer {
    /// Build the cascade for `bands` (only enabled ones become stages; at most [`MAX_BANDS`]).
    pub fn new(rate: u32, channels: usize, bands: &[Band]) -> Self {
        let rate = rate as f64;
        let bands: Vec<Band> = bands.iter().copied().take(MAX_BANDS).collect();
        let stages = bands
            .iter()
            .filter(|b| b.enabled)
            .map(|b| Biquad {
                c: Coefficients::design(b, rate),
                state: vec![(0.0, 0.0); channels],
            })
            .collect();
        Self {
            rate,
            channels,
            bands,
            stages,
        }
    }

    /// The bands (enabled or not).
    pub fn bands(&self) -> &[Band] {
        &self.bands
    }

    /// Number of active biquads.
    pub fn active_stages(&self) -> usize {
        self.stages.len()
    }

    /// `true` when no band changes the signal.
    pub fn is_identity(&self) -> bool {
        self.stages.is_empty()
    }

    /// Filter interleaved samples in place.
    pub fn process(&mut self, interleaved: &mut [f64]) {
        if self.stages.is_empty() {
            return;
        }
        let ch = self.channels;
        for frame in interleaved.chunks_mut(ch) {
            for (c, v) in frame.iter_mut().enumerate() {
                let mut x = *v;
                for s in &mut self.stages {
                    x = s.tick(c, x);
                }
                *v = x;
            }
        }
    }

    /// Combined magnitude response in dB at `freq`.
    pub fn response_db(&self, freq: f64) -> f64 {
        self.stages
            .iter()
            .map(|s| s.c.response_db(freq, self.rate))
            .sum()
    }

    /// Clear the filter memory.
    pub fn reset(&mut self) {
        for s in &mut self.stages {
            s.state.iter_mut().for_each(|z| *z = (0.0, 0.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measure_gain(eq: &mut Equalizer, freq: f64, rate: f64) -> f64 {
        let n = (rate * 2.0) as usize;
        let mut x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / rate).sin())
            .collect();
        eq.process(&mut x);
        let tail = &x[n / 2..];
        let rms = (tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64).sqrt();
        20.0 * (rms * 2f64.sqrt()).log10()
    }

    #[test]
    fn peaking_band_has_its_gain_at_the_centre_and_none_far_away() {
        let band = Band::peaking(1000.0, 6.0, 1.0);
        let mut eq = Equalizer::new(48000, 1, &[band]);
        assert!((eq.response_db(1000.0) - 6.0).abs() < 1e-9);
        assert!((measure_gain(&mut eq, 1000.0, 48000.0) - 6.0).abs() < 0.01);
        assert!(eq.response_db(20.0).abs() < 0.05);
        assert!(eq.response_db(20000.0).abs() < 0.1);
    }

    #[test]
    fn shelves_pass_and_notch_behave() {
        let rate = 44100.0;
        let low = Equalizer::new(
            44100,
            2,
            &[Band {
                kind: BandKind::LowShelf,
                freq: 200.0,
                gain_db: -4.0,
                q: std::f64::consts::FRAC_1_SQRT_2,
                enabled: true,
            }],
        );
        assert!((low.response_db(10.0) + 4.0).abs() < 0.05);
        assert!(low.response_db(10000.0).abs() < 0.05);
        let high = Equalizer::new(
            44100,
            2,
            &[Band {
                kind: BandKind::HighShelf,
                freq: 5000.0,
                gain_db: 3.0,
                q: std::f64::consts::FRAC_1_SQRT_2,
                enabled: true,
            }],
        );
        assert!((high.response_db(20000.0) - 3.0).abs() < 0.1);
        let lp = Equalizer::new(
            44100,
            1,
            &[Band {
                kind: BandKind::LowPass,
                freq: 1000.0,
                gain_db: 0.0,
                q: std::f64::consts::FRAC_1_SQRT_2,
                enabled: true,
            }],
        );
        assert!(
            (lp.response_db(1000.0) + 3.01).abs() < 0.05,
            "{}",
            lp.response_db(1000.0)
        );
        assert!(lp.response_db(10000.0) < -35.0);
        let hp = Equalizer::new(
            44100,
            1,
            &[Band {
                kind: BandKind::HighPass,
                freq: 1000.0,
                gain_db: 0.0,
                q: std::f64::consts::FRAC_1_SQRT_2,
                enabled: true,
            }],
        );
        assert!(hp.response_db(50.0) < -50.0);
        let notch = Equalizer::new(
            44100,
            1,
            &[Band {
                kind: BandKind::Notch,
                freq: 60.0,
                gain_db: 0.0,
                q: 10.0,
                enabled: true,
            }],
        );
        assert!(notch.response_db(60.0) < -60.0);
        assert!(notch.response_db(1000.0).abs() < 0.05);
        let _ = rate;
    }

    #[test]
    fn disabled_bands_are_identity_and_limit_is_enforced() {
        let mut band = Band::peaking(1000.0, 6.0, 1.0);
        band.enabled = false;
        let eq = Equalizer::new(48000, 2, &[band]);
        assert!(eq.is_identity());
        let many: Vec<Band> = (0..20)
            .map(|i| Band::peaking(100.0 * (i + 1) as f64, 1.0, 1.0))
            .collect();
        let eq = Equalizer::new(48000, 2, &many);
        assert_eq!(eq.active_stages(), MAX_BANDS);
    }
}
