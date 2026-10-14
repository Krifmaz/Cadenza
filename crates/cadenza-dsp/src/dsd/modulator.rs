//! 1-bit sigma-delta modulators (`asdm5`, `asdm7`, `dsd5v2`, `dsd7v2`).
//!
//! The loop is the error-feedback realisation of the designed NTF:
//!
//! ```text
//! v[n] = x[n] − w[n]                     w = H(z)·e,  H = 1 − NTF = (A − B) / A
//! y[n] = sign(v[n] + d[n])               1-bit quantiser (d = optional dither)
//! e[n] = y[n] − v[n]                     quantisation error
//! ```
//!
//! so `Y = X + NTF·E` with a signal transfer function of exactly 1. `(A − B)` has no
//! `z⁰` term, so the feedback only ever uses past errors and the loop is realisable;
//! the filter runs as a direct form in `f64` (the digital equivalent of the CIFB/CRFB
//! integrator chain – in software there is no component matching to worry about, and the
//! direct form keeps the NTF exactly as designed).
//!
//! **Design point.** Lee's `H∞ = 1.5` gives the lowest noise but a 1-bit loop of order 5
//! or 7 is then only stable to about 0.55 full scale (measured here on a 1 kHz tone and
//! on DC, the worst case). Cadenza uses `H∞ = 1.35` for the 5th-order loops (≈ 112 dB
//! SNR in 20 kHz at DSD64 for a −6 dBFS tone, stable to 0.72) and `H∞ = 1.28` for the
//! 7th-order loops (≈ 117 dB, stable to 0.72) so that the default `dsd_gain_db = -3`
//! (0 dBFS → 0.708) sits inside the stable range at every DSD rate.
//!
//! **Stability guard.** Above its stable limit the loop filter state winds up and the
//! output turns into a stuck or chaotic pattern. Two guards keep it well behaved at any
//! input level up to and beyond −0.3 dBFS:
//!
//! 1. an input clipper at the modulator's stable limit ([`SigmaDelta::input_limit`]),
//! 2. a loop-state reset the moment the feedback value leaves its normal range
//!    (|w| > 2.5; it stays around 1 in normal operation) – the digital equivalent of
//!    clipping the integrators, which drops the loop back into a stable state within a
//!    few samples instead of letting it ring on.
//!
//! Each engagement is counted so the signal path can report it. With the default gain
//! none of the guards engages for any PCM signal within full scale.
//!
//! The `v2` variants add adaptive TPDF dither at the quantiser input: generous at low
//! level (where idle tones live) and fading to nothing as the signal approaches the
//! stability limit, so the dither never costs headroom.

use super::ntf::Ntf;
use super::DsdModulator;
use crate::rng::Rng;
use std::sync::Arc;

/// Signal band the NTF zeros are spread over, Hz.
pub const NTF_BAND_HZ: f64 = 20_000.0;
/// Feedback value beyond which the loop is considered overloaded and is reset
/// (quantiser-input units; normal operation stays around ±1).
const OVERLOAD: f64 = 2.5;
/// Largest dither amplitude for the `v2` quantisers (peak, quantiser-input units).
const DITHER_MAX: f64 = 0.35;
/// Envelope level at which the adaptive dither has faded to zero.
const DITHER_FADE_LEVEL: f64 = 0.6;

/// One channel of 1-bit modulation.
#[derive(Clone, Debug)]
pub struct SigmaDelta {
    kind: DsdModulator,
    ntf: Arc<Ntf>,
    rate: u32,
    /// `(A − B)[1..=N]`.
    c: Vec<f64>,
    /// `A[1..=N]`.
    a: Vec<f64>,
    /// Past errors, newest first.
    e_hist: Vec<f64>,
    /// Past feedback values, newest first.
    w_hist: Vec<f64>,
    input_limit: f64,
    clips: u64,
    resets: u64,
    dither: Option<Dither>,
    /// Bits not yet emitted as a whole byte (MSB first).
    pending: u8,
    pending_bits: u8,
    samples: u64,
}

#[derive(Clone, Debug)]
struct Dither {
    rng: Rng,
    env: f64,
    decay: f64,
}

impl SigmaDelta {
    /// A modulator of `kind` running at `rate` Hz (the DSD rate).
    pub fn new(kind: DsdModulator, rate: u32) -> Self {
        let ntf = Arc::new(Ntf::for_rate(kind.order(), rate, NTF_BAND_HZ, kind.h_inf()));
        Self::with_ntf(kind, rate, ntf)
    }

    /// A modulator around an already designed NTF (shared between channels).
    pub fn with_ntf(kind: DsdModulator, rate: u32, ntf: Arc<Ntf>) -> Self {
        let n = ntf.order;
        let dither = kind.dithered().then(|| Dither {
            rng: Rng::new(0xD5D0_0000_0000_0001 ^ rate as u64),
            env: 0.0,
            // ~2 ms release at the modulator rate.
            decay: (-1.0 / (0.002 * rate as f64)).exp(),
        });
        Self {
            kind,
            c: ntf.feedback_numerator(),
            a: ntf.a[1..=n].to_vec(),
            e_hist: vec![0.0; n],
            w_hist: vec![0.0; n],
            input_limit: kind.input_limit(),
            clips: 0,
            resets: 0,
            dither,
            pending: 0,
            pending_bits: 0,
            samples: 0,
            ntf,
            rate,
        }
    }

    /// Which modulator this is.
    pub fn kind(&self) -> DsdModulator {
        self.kind
    }
    /// Modulator (DSD) rate in Hz.
    pub fn rate(&self) -> u32 {
        self.rate
    }
    /// The designed noise transfer function.
    pub fn ntf(&self) -> &Ntf {
        &self.ntf
    }
    /// Input magnitude above which the clipper guard engages (full scale = 1).
    pub fn input_limit(&self) -> f64 {
        self.input_limit
    }
    /// Samples the input clipper limited.
    pub fn clips(&self) -> u64 {
        self.clips
    }
    /// Loop-state resets after an overload.
    pub fn resets(&self) -> u64 {
        self.resets
    }
    /// All guard engagements together (what the signal path shows).
    pub fn guard_engagements(&self) -> u64 {
        self.clips + self.resets
    }
    /// Samples modulated so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }
    /// Bits produced but not yet emitted as a byte.
    pub fn pending_bits(&self) -> u8 {
        self.pending_bits
    }

    /// Forget the loop state (not the counters).
    pub fn reset(&mut self) {
        self.e_hist.fill(0.0);
        self.w_hist.fill(0.0);
        self.pending = 0;
        self.pending_bits = 0;
        if let Some(d) = &mut self.dither {
            d.env = 0.0;
        }
    }

    /// Modulate one sample; `true` is a positive (`1`) bit.
    #[inline]
    pub fn step(&mut self, x: f64) -> bool {
        self.samples += 1;
        let x = if x > self.input_limit {
            self.clips += 1;
            self.input_limit
        } else if x < -self.input_limit {
            self.clips += 1;
            -self.input_limit
        } else if x.is_nan() {
            0.0
        } else {
            x
        };
        let n = self.c.len();
        let mut w = 0.0;
        for k in 0..n {
            w += self.c[k] * self.e_hist[k] - self.a[k] * self.w_hist[k];
        }
        if w.abs() > OVERLOAD || w.is_nan() {
            // Overload (or a NaN that must never propagate): back to a clean state.
            self.resets += 1;
            self.e_hist.fill(0.0);
            self.w_hist.fill(0.0);
            w = 0.0;
        }
        let v = x - w;
        let d = match &mut self.dither {
            Some(dt) => {
                let ax = x.abs();
                dt.env = if ax > dt.env { ax } else { dt.env * dt.decay };
                let amp = DITHER_MAX * (1.0 - (dt.env / DITHER_FADE_LEVEL).min(1.0));
                amp * (dt.rng.centred() + dt.rng.centred())
            }
            None => 0.0,
        };
        let y = if v + d >= 0.0 { 1.0 } else { -1.0 };
        let e = y - v;
        // Shift the histories (N ≤ 8, cheaper than a ring buffer).
        for k in (1..n).rev() {
            self.e_hist[k] = self.e_hist[k - 1];
            self.w_hist[k] = self.w_hist[k - 1];
        }
        self.e_hist[0] = e;
        self.w_hist[0] = w;
        y > 0.0
    }

    /// Modulate `input` (full scale ±1, any length), appending whole bytes (8 bits, oldest
    /// in the MSB) to `out`. A partial byte is kept until the next call.
    pub fn process(&mut self, input: &[f64], out: &mut Vec<u8>) {
        out.reserve(input.len() / 8 + 1);
        for &x in input {
            let bit = self.step(x) as u8;
            self.pending = (self.pending << 1) | bit;
            self.pending_bits += 1;
            if self.pending_bits == 8 {
                out.push(self.pending);
                self.pending = 0;
                self.pending_bits = 0;
            }
        }
    }

    /// Modulate silence until the pending byte is complete, appending it (if any).
    pub fn complete_byte(&mut self, out: &mut Vec<u8>) {
        while self.pending_bits != 0 {
            self.process(&[0.0], out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsd::measure::{coherent_tone_hz, tone_snr_db};
    use crate::dsd::DsdRate;
    use std::f64::consts::PI;

    fn tone(amp: f64, hz: f64, rate: u32, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * hz * i as f64 / rate as f64).sin())
            .collect()
    }

    /// Modulate a tone and measure SNR in the 20 kHz band at DSD64.
    fn snr_at_dsd64(kind: DsdModulator, amp: f64) -> f64 {
        let rate = DsdRate::Dsd64.hz(44_100);
        let pcm_rate = rate / 8;
        let nfft = 1 << 17;
        let hz = coherent_tone_hz(1000.0, pcm_rate, nfft);
        let mut m = SigmaDelta::new(kind, rate);
        let n = (nfft + 4096) * 8;
        let mut bits = Vec::new();
        m.process(&tone(amp, hz, rate, n), &mut bits);
        tone_snr_db(&bits, rate, hz, 20_000.0, nfft).0
    }

    #[test]
    fn asdm5_minus_6_dbfs_tone_has_110_db_snr_at_dsd64() {
        let snr = snr_at_dsd64(DsdModulator::Asdm5, 0.5);
        assert!(snr >= 110.0, "asdm5 SNR {snr:.1} dB");
    }

    #[test]
    fn asdm7_minus_6_dbfs_tone_has_110_db_snr_at_dsd64() {
        let snr = snr_at_dsd64(DsdModulator::Asdm7, 0.5);
        assert!(snr >= 115.0, "asdm7 SNR {snr:.1} dB");
    }

    #[test]
    fn dithered_variants_keep_the_snr() {
        for kind in [DsdModulator::Dsd5v2, DsdModulator::Dsd7v2] {
            let snr = snr_at_dsd64(kind, 0.5);
            assert!(snr >= 108.0, "{} SNR {snr:.1} dB", kind.id());
        }
    }

    #[test]
    fn dither_suppresses_idle_tones_at_low_level() {
        // A tiny DC offset: the undithered loop locks into a periodic pattern with
        // discrete in-band tones; the dithered one spreads them into noise.
        let rate = DsdRate::Dsd64.hz(44_100);
        let nfft = 1 << 16;
        let n = (nfft + 4096) * 8;
        let peak_over_mean = |kind: DsdModulator| {
            let mut m = SigmaDelta::new(kind, rate);
            let mut bits = Vec::new();
            m.process(&vec![0.00123; n], &mut bits);
            crate::dsd::measure::spectrum_peak_to_mean_db(&bits, rate, 20_000.0, nfft)
        };
        let plain = peak_over_mean(DsdModulator::Asdm5);
        let dithered = peak_over_mean(DsdModulator::Dsd5v2);
        assert!(
            dithered < plain,
            "dithered idle spectrum peak/mean {dithered:.1} dB vs plain {plain:.1} dB"
        );
    }

    #[test]
    fn stays_stable_and_recovers_at_minus_0_3_dbfs() {
        let rate = DsdRate::Dsd64.hz(44_100);
        for kind in DsdModulator::catalog() {
            let mut m = SigmaDelta::new(*kind, rate);
            let loud = tone(0.966, 1000.0, rate, rate as usize / 4); // 250 ms at −0.3 dBFS
            let mut bits = Vec::new();
            m.process(&loud, &mut bits);
            // Bounded, finite, never stuck: every 1 ms window has both polarities.
            let window = rate as usize / 8000;
            for chunk in bits[bits.len() / 2..].chunks(window) {
                let ones: u32 = chunk.iter().map(|b| b.count_ones()).sum();
                assert!(ones > 0 && ones < chunk.len() as u32 * 8, "{}: stuck output", kind.id());
            }
            assert!(m.w_hist.iter().all(|w| w.is_finite()));
            assert!(m.clips() > 0, "{}: the clipper must be what keeps −0.3 dBFS stable", kind.id());
            // Afterwards a −6 dBFS tone still gets its full SNR: the loop recovered.
            let pcm_rate = rate / 8;
            let nfft = 1 << 16;
            let hz = coherent_tone_hz(1000.0, pcm_rate, nfft);
            let mut bits = Vec::new();
            m.process(&tone(0.5, hz, rate, (nfft + 4096) * 8), &mut bits);
            let snr = tone_snr_db(&bits, rate, hz, 20_000.0, nfft).0;
            assert!(snr > 105.0, "{} after overload: SNR {snr:.1} dB", kind.id());
        }
    }

    #[test]
    fn stable_at_the_input_limit_on_dc_and_tone_at_every_rate() {
        // DC is the worst case for a 1-bit loop; the clipper threshold must sit inside the
        // loop's own stable range at every DSD rate, with the default −3 dB gain
        // (0 dBFS → 0.708) below the threshold.
        for kind in DsdModulator::catalog() {
            assert!(kind.input_limit() > 0.708, "{}: default gain must not clip", kind.id());
            for rate in [DsdRate::Dsd64, DsdRate::Dsd256, DsdRate::Dsd1024] {
                let hz = rate.hz(44_100);
                let mut m = SigmaDelta::new(*kind, hz);
                let mut bits = Vec::new();
                m.process(&vec![kind.input_limit(); 200_000], &mut bits);
                m.process(&vec![-kind.input_limit(); 200_000], &mut bits);
                m.process(&tone(kind.input_limit(), 1000.0, hz, 400_000), &mut bits);
                assert_eq!(
                    m.guard_engagements(),
                    0,
                    "{} at {}: {} resets at the input limit",
                    kind.id(),
                    rate.label(),
                    m.resets()
                );
            }
        }
    }

    #[test]
    fn guards_are_counted_and_default_gain_avoids_them() {
        let rate = DsdRate::Dsd64.hz(44_100);
        let mut m = SigmaDelta::new(DsdModulator::Asdm7, rate);
        let mut bits = Vec::new();
        m.process(&tone(0.708, 1000.0, rate, rate as usize / 10), &mut bits);
        assert_eq!(m.guard_engagements(), 0, "0 dBFS − 3 dB must not engage any guard");
        m.process(&tone(0.999, 1000.0, rate, rate as usize / 10), &mut bits);
        assert!(m.clips() > 0);
        // The reset guard catches what the clipper lets through (here: forced overload).
        let mut m = SigmaDelta::new(DsdModulator::Asdm5, rate);
        m.input_limit = 10.0;
        m.process(&vec![0.95; 50_000], &mut bits);
        assert!(m.resets() > 0);
        assert!(m.w_hist.iter().all(|w| w.abs() <= OVERLOAD + 3.0));
    }

    #[test]
    fn bytes_are_msb_first_and_partial_bytes_are_kept() {
        let mut m = SigmaDelta::new(DsdModulator::Asdm5, 2_822_400);
        let mut out = Vec::new();
        m.process(&[0.0; 13], &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(m.pending_bits(), 5);
        m.complete_byte(&mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(m.pending_bits(), 0);
        // Silence modulates to a roughly 50 % duty pattern.
        let mut m = SigmaDelta::new(DsdModulator::Asdm5, 2_822_400);
        let mut out = Vec::new();
        m.process(&vec![0.0; 80_000], &mut out);
        let ones: u32 = out.iter().map(|b| b.count_ones()).sum();
        assert!((ones as f64 / 80_000.0 - 0.5).abs() < 0.01);
    }
}
