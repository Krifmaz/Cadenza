//! Quantisation to 16/24/32-bit integers with dither and noise shaping.
//!
//! The quantiser is an error-feedback structure:
//!
//! ```text
//! v[n] = x[n] − Σ_k c_k · e[n−k]        (shaped feedback of past errors)
//! y[n] = round(v[n] + d[n])              (d = dither noise, in LSBs)
//! e[n] = y[n] − v[n]                     (this sample's quantisation error)
//! ```
//!
//! so the output is `Y = X + E·(1 − Σ c_k z^-k)`: the error spectrum is shaped by the
//! noise transfer function `1 − Σ c_k z^-k`. The coefficient sets are stored exactly as
//! published (Lipshitz / Vanderkooy / Wannamaker 1991, Wannamaker 1992, and the Gesemann
//! sets as distributed with SoX); they assume a 44.1 kHz (or 48 kHz) output rate.

use crate::rng::Rng;
use serde::{Deserialize, Serialize};

/// Dither and noise-shaping choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DitherKind {
    /// Plain rounding.
    None,
    /// Rectangular PDF, ±0.5 LSB.
    Rectangular,
    /// Triangular PDF, ±1 LSB (the sum of two rectangular draws): decorrelates both the
    /// error and its power from the signal.
    #[default]
    Tpdf,
    /// TPDF plus a 2nd-order `(1 − z^-1)²` error-feedback high-pass.
    ErrorFeedback2,
    /// TPDF plus Lipshitz, Vanderkooy & Wannamaker's 5-coefficient F-weighted shaper (44.1 kHz).
    Lipshitz,
    /// TPDF plus Wannamaker's 9-coefficient F-weighted shaper (44.1 kHz).
    Wannamaker9,
    /// TPDF plus Wannamaker's 3-coefficient F-weighted shaper (44.1 kHz).
    Wannamaker3,
    /// TPDF plus Gesemann's 8-coefficient shaper for 44.1 kHz.
    Gesemann441,
    /// TPDF plus Gesemann's 8-coefficient shaper for 48 kHz.
    Gesemann48,
}

/// Published noise-shaping coefficient sets, in the `1 − Σ c_k z^-k` convention.
pub mod coefficients {
    /// `(1 − z^-1)²`.
    pub const ERROR_FEEDBACK_2: &[f64] = &[2.0, -1.0];
    /// Lipshitz, Vanderkooy, Wannamaker, "Minimally audible noise shaping", JAES 1991 (44.1 kHz).
    pub const LIPSHITZ_44: &[f64] = &[2.033, -2.165, 1.959, -1.590, 0.6149];
    /// Wannamaker, "Psychoacoustically optimal noise shaping", JAES 1992, 9 taps, F-weighted (44.1 kHz).
    pub const WANNAMAKER_9_44: &[f64] = &[
        2.412, -3.370, 3.937, -4.174, 3.353, -2.205, 1.281, -0.569, 0.0847,
    ];
    /// Wannamaker 1992, 3 taps, F-weighted (44.1 kHz).
    pub const WANNAMAKER_3_44: &[f64] = &[1.623, -0.982, 0.109];
    /// Sebastian Gesemann's 8-coefficient shaper for 44.1 kHz (as distributed with SoX).
    pub const GESEMANN_44: &[f64] = &[
        2.2061, -0.4706, -0.2534, -0.6214, 1.0587, 0.0676, -0.6054, -0.2738,
    ];
    /// Sebastian Gesemann's 8-coefficient shaper for 48 kHz (as distributed with SoX).
    pub const GESEMANN_48: &[f64] = &[
        2.2374, -0.7339, -0.1251, -0.6033, 0.903, 0.0116, -0.5853, -0.2571,
    ];
}

impl DitherKind {
    /// Every kind, in display order.
    pub fn catalog() -> &'static [DitherKind] {
        &[
            DitherKind::None,
            DitherKind::Rectangular,
            DitherKind::Tpdf,
            DitherKind::ErrorFeedback2,
            DitherKind::Lipshitz,
            DitherKind::Wannamaker9,
            DitherKind::Wannamaker3,
            DitherKind::Gesemann441,
            DitherKind::Gesemann48,
        ]
    }

    /// Stable id used in configuration and APIs.
    pub fn id(self) -> &'static str {
        match self {
            DitherKind::None => "none",
            DitherKind::Rectangular => "rectangular",
            DitherKind::Tpdf => "tpdf",
            DitherKind::ErrorFeedback2 => "ef2",
            DitherKind::Lipshitz => "lipshitz",
            DitherKind::Wannamaker9 => "wannamaker9",
            DitherKind::Wannamaker3 => "wannamaker3",
            DitherKind::Gesemann441 => "gesemann441",
            DitherKind::Gesemann48 => "gesemann48",
        }
    }

    /// Human readable label.
    pub fn label(self) -> &'static str {
        match self {
            DitherKind::None => "None (round)",
            DitherKind::Rectangular => "Rectangular",
            DitherKind::Tpdf => "TPDF",
            DitherKind::ErrorFeedback2 => "2nd-order error feedback",
            DitherKind::Lipshitz => "Lipshitz 5-coef F-weighted (44.1k)",
            DitherKind::Wannamaker9 => "Wannamaker 9-coef F-weighted (44.1k)",
            DitherKind::Wannamaker3 => "Wannamaker 3-coef F-weighted (44.1k)",
            DitherKind::Gesemann441 => "Gesemann 8-coef (44.1k)",
            DitherKind::Gesemann48 => "Gesemann 8-coef (48k)",
        }
    }

    /// Parse an id (case-insensitive; a few aliases accepted).
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        Self::catalog()
            .iter()
            .copied()
            .find(|k| k.id() == s)
            .or(match s.as_str() {
                "off" | "round" => Some(DitherKind::None),
                "rect" | "rpdf" => Some(DitherKind::Rectangular),
                "triangular" => Some(DitherKind::Tpdf),
                "shaped" | "fweighted" | "f-weighted" => Some(DitherKind::Wannamaker9),
                _ => None,
            })
    }

    /// The noise-shaping feedback coefficients, if this kind shapes.
    pub fn coefficients(self) -> &'static [f64] {
        match self {
            DitherKind::None | DitherKind::Rectangular | DitherKind::Tpdf => &[],
            DitherKind::ErrorFeedback2 => coefficients::ERROR_FEEDBACK_2,
            DitherKind::Lipshitz => coefficients::LIPSHITZ_44,
            DitherKind::Wannamaker9 => coefficients::WANNAMAKER_9_44,
            DitherKind::Wannamaker3 => coefficients::WANNAMAKER_3_44,
            DitherKind::Gesemann441 => coefficients::GESEMANN_44,
            DitherKind::Gesemann48 => coefficients::GESEMANN_48,
        }
    }

    /// The sample rate a psychoacoustic shaper was designed for (its curve is only
    /// meaningful there); `None` for rate-independent kinds.
    pub fn design_rate(self) -> Option<u32> {
        match self {
            DitherKind::Lipshitz
            | DitherKind::Wannamaker9
            | DitherKind::Wannamaker3
            | DitherKind::Gesemann441 => Some(44100),
            DitherKind::Gesemann48 => Some(48000),
            _ => None,
        }
    }

    /// Whether this kind adds dither noise (everything except plain rounding).
    pub fn adds_noise(self) -> bool {
        self != DitherKind::None
    }
}

/// Converts `f64` samples in `[-1, 1)` to integers of `bits` bits, per channel.
#[derive(Clone, Debug)]
pub struct Quantizer {
    bits: u32,
    kind: DitherKind,
    channels: usize,
    scale: f64,
    min: f64,
    max: f64,
    coefs: Vec<f64>,
    /// Per channel ring buffer of past errors (newest at `pos`).
    errors: Vec<Vec<f64>>,
    pos: usize,
    rng: Rng,
}

impl Quantizer {
    /// A quantiser to `bits` (16, 24 or 32) for `channels` interleaved channels.
    pub fn new(bits: u32, kind: DitherKind, channels: usize) -> Self {
        assert!(
            matches!(bits, 8 | 16 | 20 | 24 | 32),
            "unsupported bit depth {bits}"
        );
        let scale = (1u64 << (bits - 1)) as f64;
        let coefs = kind.coefficients().to_vec();
        Self {
            bits,
            kind,
            channels: channels.max(1),
            scale,
            min: -scale,
            max: scale - 1.0,
            errors: vec![vec![0.0; coefs.len().max(1)]; channels.max(1)],
            coefs,
            pos: 0,
            rng: Rng::default(),
        }
    }

    /// Output bit depth.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// The dither kind in use.
    pub fn kind(&self) -> DitherKind {
        self.kind
    }

    /// Forget the error history.
    pub fn reset(&mut self) {
        for e in &mut self.errors {
            e.fill(0.0);
        }
        self.pos = 0;
    }

    #[inline]
    fn dither(&mut self) -> f64 {
        match self.kind {
            DitherKind::None => 0.0,
            DitherKind::Rectangular => self.rng.centred(),
            _ => self.rng.centred() + self.rng.centred(),
        }
    }

    /// Quantise interleaved `input`, appending integers (in the native range of `bits`,
    /// e.g. −32768…32767 for 16) to `output`.
    pub fn process(&mut self, input: &[f64], output: &mut Vec<i32>) {
        output.reserve(input.len());
        let ch = self.channels;
        let n = self.coefs.len();
        for frame in input.chunks(ch) {
            for (c, &x) in frame.iter().enumerate() {
                let v = x * self.scale;
                let shaped = if n == 0 {
                    v
                } else {
                    let hist = &self.errors[c];
                    let mut fb = 0.0;
                    for (k, coef) in self.coefs.iter().enumerate() {
                        fb += coef * hist[(self.pos + n - k) % n];
                    }
                    v - fb
                };
                let d = self.dither();
                let q = (shaped + d).round().clamp(self.min, self.max);
                if n > 0 {
                    // Feed back the error against the un-dithered, shaped value; clipping
                    // leaks into the loop too, which is what keeps it from winding up.
                    let next = (self.pos + 1) % n;
                    self.errors[c][next] = q - shaped;
                    if c == ch - 1 {
                        self.pos = next;
                    }
                }
                output.push(q as i64 as i32);
            }
        }
    }

    /// Quantise and convert straight back to `f64` in `[-1, 1)`: handy for measurements.
    pub fn process_to_f64(&mut self, input: &[f64]) -> Vec<f64> {
        let mut ints = Vec::with_capacity(input.len());
        self.process(input, &mut ints);
        ints.iter().map(|&v| v as f64 / self.scale).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::magnitude_db;
    use std::f64::consts::PI;

    fn quiet_sine(amp_lsb: f64, bits: u32, n: usize) -> Vec<f64> {
        let scale = (1u64 << (bits - 1)) as f64;
        (0..n)
            .map(|i| amp_lsb / scale * (2.0 * PI * 1000.0 * i as f64 / 44100.0).sin())
            .collect()
    }

    #[test]
    fn rounding_a_signal_below_half_an_lsb_is_silence_but_tpdf_is_not() {
        let x = quiet_sine(0.3, 16, 20_000);
        let mut plain = Vec::new();
        Quantizer::new(16, DitherKind::None, 1).process(&x, &mut plain);
        assert!(
            plain.iter().all(|&v| v == 0),
            "undithered output is a dead zero"
        );

        let mut tpdf = Vec::new();
        Quantizer::new(16, DitherKind::Tpdf, 1).process(&x, &mut tpdf);
        let mean = tpdf.iter().map(|&v| v as f64).sum::<f64>() / tpdf.len() as f64;
        let var = tpdf.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / tpdf.len() as f64;
        assert!(var > 0.1, "TPDF output variance {var}");
        assert!(mean.abs() < 0.05, "TPDF mean {mean}");
        // The dither keeps the tone: correlate with the input.
        let scale = 32768.0;
        let corr: f64 = tpdf
            .iter()
            .zip(&x)
            .map(|(&q, &s)| q as f64 / scale * s)
            .sum();
        let energy: f64 = x.iter().map(|s| s * s).sum();
        assert!(
            (corr / energy - 1.0).abs() < 0.2,
            "tone preserved through dither: {}",
            corr / energy
        );
    }

    #[test]
    fn bit_exact_for_values_already_on_the_grid() {
        let scale = 32768.0;
        let ints = [-32768i32, -1, 0, 1, 12345, 32767];
        let x: Vec<f64> = ints.iter().map(|&i| i as f64 / scale).collect();
        let mut out = Vec::new();
        Quantizer::new(16, DitherKind::None, 2).process(&x, &mut out);
        assert_eq!(out, ints);
        let mut out32 = Vec::new();
        Quantizer::new(32, DitherKind::None, 1).process(&[-1.0, 0.5, 0.999_999_999_9], &mut out32);
        assert_eq!(out32[0], i32::MIN);
        assert_eq!(out32[1], 1 << 30);
    }

    fn in_band_noise_db(kind: DitherKind) -> f64 {
        let n = 1 << 16;
        let x = quiet_sine(50.0, 16, n);
        let mut q = Quantizer::new(16, kind, 1);
        let y = q.process_to_f64(&x);
        let err: Vec<f64> = y.iter().zip(&x).map(|(a, b)| a - b).collect();
        let db = magnitude_db(&err, n);
        let hz_per_bin = 44100.0 / n as f64;
        let (lo, hi) = (
            (100.0 / hz_per_bin) as usize,
            (4000.0 / hz_per_bin) as usize,
        );
        let power: f64 = db[lo..hi].iter().map(|d| 10f64.powf(d / 10.0)).sum();
        10.0 * power.log10()
    }

    #[test]
    fn f_weighted_shapers_lower_the_in_band_noise() {
        let flat = in_band_noise_db(DitherKind::Tpdf);
        for kind in [
            DitherKind::Lipshitz,
            DitherKind::Wannamaker9,
            DitherKind::Wannamaker3,
        ] {
            let shaped = in_band_noise_db(kind);
            assert!(
                shaped < flat - 6.0,
                "{kind:?}: in-band {shaped:.1} dB vs flat {flat:.1} dB"
            );
        }
        // Gesemann's curve is gentler below 4 kHz (it optimises a different weighting),
        // but it still moves noise out of the band.
        for kind in [
            DitherKind::Gesemann441,
            DitherKind::Gesemann48,
            DitherKind::ErrorFeedback2,
        ] {
            let shaped = in_band_noise_db(kind);
            assert!(
                shaped < flat,
                "{kind:?}: in-band {shaped:.1} dB vs flat {flat:.1} dB"
            );
        }
    }

    #[test]
    fn error_feedback_stays_bounded_at_full_scale() {
        let x: Vec<f64> = (0..50_000)
            .map(|i| 0.999 * (2.0 * PI * 50.0 * i as f64 / 44100.0).sin())
            .collect();
        let mut q = Quantizer::new(16, DitherKind::Wannamaker9, 1);
        let y = q.process_to_f64(&x);
        let err = y
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 64.0 / 32768.0, "error {err}");
    }

    #[test]
    fn ids_round_trip() {
        for k in DitherKind::catalog() {
            assert_eq!(DitherKind::parse(k.id()), Some(*k));
        }
        assert_eq!(DitherKind::parse("TPDF"), Some(DitherKind::Tpdf));
        assert_eq!(DitherKind::parse("nope"), None);
    }
}
