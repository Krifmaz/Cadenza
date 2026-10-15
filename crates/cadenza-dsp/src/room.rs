//! Room correction: measure a loudspeaker in its room and design the correction filter.
//!
//! 1. **Sweep** ([`Sweep`]): an exponential sine sweep (Farina) is played and recorded
//!    with a measurement microphone. Dividing the recording's spectrum by the sweep's
//!    (regularised outside the swept band) turns it back into the room's impulse response
//!    ([`Sweep::impulse`]); harmonic distortion lands at negative times and is cut away.
//! 2. **Response** ([`response`]): the impulse's magnitude, smoothed over a fraction of
//!    an octave, on a logarithmic frequency grid – what the plots show.
//! 3. **Correction** ([`design`]): the inverse of the smoothed response towards a target
//!    (flat or tilted), limited in boost and cut, applied only inside a band (by default
//!    below 500 Hz, where the room's modes dominate and a correction holds over the whole
//!    listening area) and faded out at its edges, as a minimum-phase FIR (no pre-ringing,
//!    no added latency to speak of).

use crate::fft::{next_pow2, RealFft};
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// An exponential sine sweep.
#[derive(Clone, Debug)]
pub struct Sweep {
    /// Sample rate.
    pub rate: u32,
    /// Start and end frequencies, Hz.
    pub f1: f64,
    /// End frequency.
    pub f2: f64,
    /// The sweep (peak 1.0).
    pub signal: Vec<f64>,
}

impl Sweep {
    /// A sweep from `f1` to `f2` Hz lasting `seconds` at `rate`.
    pub fn new(rate: u32, f1: f64, f2: f64, seconds: f64) -> Self {
        let fs = rate as f64;
        let n = (seconds * fs).round() as usize;
        let t_len = n as f64 / fs;
        let k = (f2 / f1).ln();
        let fade_in = (0.05 * fs) as usize;
        let fade_out = (0.01 * fs) as usize;
        let mut signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * f1 * t_len / k * ((t * k / t_len).exp() - 1.0)).sin()
            })
            .collect();
        for (i, v) in signal.iter_mut().enumerate().take(fade_in) {
            *v *= 0.5 - 0.5 * (PI * i as f64 / fade_in as f64).cos();
        }
        for i in 0..fade_out.min(n) {
            signal[n - 1 - i] *= 0.5 - 0.5 * (PI * i as f64 / fade_out as f64).cos();
        }
        Self {
            rate,
            f1,
            f2,
            signal,
        }
    }

    /// The recording divided by the sweep in the frequency domain: the impulse response
    /// at time 0 onwards, distortion at the end (negative times).
    fn deconvolve(&self, recording: &[f64]) -> Vec<f64> {
        let nfft = next_pow2(recording.len() + self.signal.len());
        let mut fft = RealFft::new(nfft);
        let mut r = recording.to_vec();
        r.resize(nfft, 0.0);
        let mut x = self.signal.clone();
        x.resize(nfft, 0.0);
        let (mut rs, mut xs) = (fft.spectrum(), fft.spectrum());
        fft.forward(&mut r, &mut rs);
        fft.forward(&mut x, &mut xs);
        let df = self.rate as f64 / nfft as f64;
        let peak = xs.iter().map(|c| c.norm_sqr()).fold(0.0, f64::max);
        for (k, (rk, xk)) in rs.iter_mut().zip(&xs).enumerate() {
            let f = k as f64 * df;
            // Inside the sweep a plain division; outside it regularised, rising smoothly
            // over an octave (a hard edge would ring for longer than the window keeps).
            let out = if f < self.f1 {
                (self.f1 / f.max(1e-9)).log2().min(1.0)
            } else if f > self.f2 {
                (f / self.f2).log2().min(1.0)
            } else {
                0.0
            };
            let w = 0.5 - 0.5 * (PI * out).cos();
            let eps = peak * (1e-10 + 1e-1 * w);
            *rk = *rk * xk.conj() / (xk.norm_sqr() + eps);
        }
        let mut out = vec![0.0; nfft];
        fft.inverse(&mut rs, &mut out);
        out
    }

    /// Recover the impulse response from a recording made while the sweep played. The
    /// result starts `pre_ms` before the direct sound and lasts `length_ms`; its start is
    /// faded in over `pre_ms` and its end over the last tenth.
    pub fn impulse(&self, recording: &[f64], pre_ms: f64, length_ms: f64) -> Vec<f64> {
        let full = self.deconvolve(recording);
        // The linear response is at non-negative times: search the first part only (the
        // distortion products wrap round to the end of the buffer).
        let search = recording.len().min(full.len());
        let (peak, _) = full[..search]
            .iter()
            .enumerate()
            .fold((0usize, 0.0f64), |acc, (i, v)| {
                if v.abs() > acc.1 {
                    (i, v.abs())
                } else {
                    acc
                }
            });
        let fs = self.rate as f64;
        let pre = (pre_ms * 1e-3 * fs) as usize;
        let len = (length_ms * 1e-3 * fs) as usize;
        let from = peak.saturating_sub(pre);
        let mut ir: Vec<f64> = full[from..(from + len).min(full.len())].to_vec();
        let lead = peak - from;
        for (i, v) in ir.iter_mut().enumerate().take(lead) {
            *v *= 0.5 - 0.5 * (PI * i as f64 / lead as f64).cos();
        }
        let tail = ir.len() / 10;
        let n = ir.len();
        for i in 0..tail {
            ir[n - 1 - i] *= 0.5 - 0.5 * (PI * i as f64 / tail as f64).cos();
        }
        ir
    }
}

/// Linear convolution through the FFT.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let n = a.len() + b.len() - 1;
    let nfft = next_pow2(n);
    let mut fft = RealFft::new(nfft);
    let mut pa = a.to_vec();
    pa.resize(nfft, 0.0);
    let mut pb = b.to_vec();
    pb.resize(nfft, 0.0);
    let mut sa = fft.spectrum();
    let mut sb = fft.spectrum();
    fft.forward(&mut pa, &mut sa);
    fft.forward(&mut pb, &mut sb);
    for (x, y) in sa.iter_mut().zip(&sb) {
        *x *= y;
    }
    let mut out = vec![0.0; nfft];
    fft.inverse(&mut sa, &mut out);
    out.truncate(n);
    out
}

/// |H(f)| of an FIR (direct evaluation).
#[cfg(test)]
fn gain_at(h: &[f64], fs: f64, f: f64) -> f64 {
    let w = 2.0 * PI * f / fs;
    let (re, im) = h.iter().enumerate().fold((0.0, 0.0), |(re, im), (k, v)| {
        (re + v * (w * k as f64).cos(), im - v * (w * k as f64).sin())
    });
    (re * re + im * im).sqrt()
}

/// Power spectrum of `h` zero-padded to at least `min_len` (a power of two): bins
/// `0..=nfft/2`, and the frequency step.
fn power_spectrum(h: &[f64], fs: f64, min_len: usize) -> (Vec<f64>, f64) {
    let nfft = next_pow2(h.len().max(min_len));
    let mut fft = RealFft::new(nfft);
    let mut x = h.to_vec();
    x.resize(nfft, 0.0);
    let mut s = fft.spectrum();
    fft.forward(&mut x, &mut s);
    (s.iter().map(|c| c.norm_sqr()).collect(), fs / nfft as f64)
}

/// `points` frequencies spaced logarithmically from `f_lo` to `f_hi`.
pub fn log_grid(f_lo: f64, f_hi: f64, points: usize) -> Vec<f64> {
    (0..points)
        .map(|i| f_lo * (f_hi / f_lo).powf(i as f64 / (points - 1).max(1) as f64))
        .collect()
}

/// Power averaged over `±1/(2·fraction)` octave around each frequency of `grid`, in dB.
fn smooth_db(power: &[f64], df: f64, grid: &[f64], fraction: f64) -> Vec<f64> {
    let half = 2f64.powf(0.5 / fraction);
    grid.iter()
        .map(|f| {
            let lo = ((f / half / df).floor() as usize).max(1);
            let hi = ((f * half / df).ceil() as usize)
                .min(power.len() - 1)
                .max(lo);
            let p = power[lo..=hi].iter().sum::<f64>() / (hi - lo + 1) as f64;
            10.0 * p.max(1e-30).log10()
        })
        .collect()
}

/// The impulse's response at the `grid` frequencies, smoothed over 1/`fraction` octave
/// (6 for a detailed plot, 3 for an overview), in dB.
pub fn response(ir: &[f64], rate: u32, grid: &[f64], fraction: f64) -> Vec<f64> {
    let (p, df) = power_spectrum(ir, rate as f64, 1 << 16);
    smooth_db(&p, df, grid, fraction)
}

/// How to design the correction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DesignOptions {
    /// Correct from here …
    pub low_hz: f64,
    /// … to here (500 Hz: the modal region, where a correction holds for the whole room).
    pub high_hz: f64,
    /// Most boost, dB (a dip is usually a cancellation that boosting cannot fill).
    pub max_boost_db: f64,
    /// Most cut, dB.
    pub max_cut_db: f64,
    /// Target tilt, dB per octave above 1 kHz (0 = flat; −0.5 or so is a gentle "house
    /// curve" that sounds natural in most rooms).
    pub tilt_db_per_octave: f64,
    /// Smoothing of the measured response before inverting, fraction of an octave.
    pub smoothing: f64,
    /// FIR length.
    pub taps: usize,
}

impl Default for DesignOptions {
    fn default() -> Self {
        Self {
            low_hz: 20.0,
            high_hz: 500.0,
            max_boost_db: 6.0,
            max_cut_db: 15.0,
            tilt_db_per_octave: 0.0,
            smoothing: 6.0,
            taps: 16384,
        }
    }
}

/// The target curve at `f`, dB.
pub fn target_db(f: f64, o: &DesignOptions) -> f64 {
    if f > 1000.0 {
        o.tilt_db_per_octave * (f / 1000.0).log2()
    } else {
        0.0
    }
}

/// A minimum-phase FIR of `taps` whose magnitude follows `gain_db(f)` (evaluated on an
/// FFT grid at `fs`), by the real cepstrum.
pub fn min_phase(taps: usize, fs: f64, gain_db: impl Fn(f64) -> f64) -> Vec<f64> {
    let nfft = next_pow2(taps * 4);
    let bins = nfft / 2 + 1;
    let df = fs / nfft as f64;
    let mut fft = RealFft::new(nfft);
    // log|H| → cepstrum.
    let mut logmag: Vec<Complex<f64>> = (0..bins)
        .map(|k| Complex::new(gain_db(k as f64 * df) / 20.0 * std::f64::consts::LN_10, 0.0))
        .collect();
    let mut cep = vec![0.0; nfft];
    fft.inverse(&mut logmag, &mut cep);
    // Fold: keep c[0] and c[n/2], double the causal half, drop the anti-causal half.
    for (i, c) in cep.iter_mut().enumerate() {
        if i > 0 && i < nfft / 2 {
            *c *= 2.0;
        } else if i > nfft / 2 {
            *c = 0.0;
        }
    }
    let mut spec = fft.spectrum();
    fft.forward(&mut cep, &mut spec);
    for c in &mut spec {
        *c = c.exp();
    }
    let mut h = vec![0.0; nfft];
    fft.inverse(&mut spec, &mut h);
    h.truncate(taps);
    // Fade the last tenth (the tail is tiny for a smooth magnitude).
    let tail = taps / 10;
    for i in 0..tail {
        h[taps - 1 - i] *= 0.5 - 0.5 * (PI * i as f64 / tail as f64).cos();
    }
    h
}

/// The measured response's level in the 200 Hz – 2 kHz band, dB (the reference the
/// correction keeps).
pub fn reference_level(ir: &[f64], rate: u32) -> f64 {
    let grid = log_grid(200.0, 2000.0, 64);
    let r = response(ir, rate, &grid, 3.0);
    r.iter().sum::<f64>() / r.len() as f64
}

/// The correction gain (dB) at `f` for a measurement whose smoothed response is
/// `measured(f)` (dB, already relative to [`reference_level`]).
pub fn correction_db(f: f64, measured: f64, o: &DesignOptions) -> f64 {
    // Fade the correction in and out over half an octave at each band edge.
    let edge = |x: f64| -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            0.5 - 0.5 * (PI * x).cos()
        }
    };
    let w = edge((f / o.low_hz).log2() * 2.0 + 1.0) * edge((o.high_hz / f).log2() * 2.0 + 1.0);
    if w == 0.0 {
        return 0.0;
    }
    let mut g = (target_db(f, o) - measured).clamp(-o.max_cut_db, o.max_boost_db);
    // Below the speaker's own roll-off there is nothing to boost.
    if measured < -12.0 && f < 120.0 {
        g = g.min(0.0);
    }
    g * w
}

/// A designed correction.
#[derive(Clone, Debug)]
pub struct Correction {
    /// The FIR (minimum phase), at the measurement rate.
    pub fir: Vec<f64>,
    /// The measured response's reference level (dB) that maps to 0 dB.
    pub reference_db: f64,
}

/// Design the correction for a measured impulse `ir` at `rate`.
pub fn design(ir: &[f64], rate: u32, o: &DesignOptions) -> Correction {
    let fs = rate as f64;
    let reference_db = reference_level(ir, rate);
    let (p, df) = power_spectrum(ir, fs, 1 << 16);
    // The smoothed response on a fine log grid, interpolated for the FFT bins.
    let grid = log_grid(10.0, fs / 2.0 * 0.999, 2000);
    let meas: Vec<f64> = smooth_db(&p, df, &grid, o.smoothing)
        .into_iter()
        .map(|v| v - reference_db)
        .collect();
    let at = |f: f64| -> f64 {
        if f <= grid[0] {
            return meas[0];
        }
        let x =
            (f / grid[0]).ln() / (grid[grid.len() - 1] / grid[0]).ln() * (grid.len() - 1) as f64;
        let i = (x.floor() as usize).min(grid.len() - 2);
        let a = x - i as f64;
        meas[i] * (1.0 - a) + meas[i + 1] * a
    };
    let fir = min_phase(o.taps, fs, |f| {
        if f <= 0.0 {
            0.0
        } else {
            correction_db(f, at(f), o)
        }
    });
    Correction { fir, reference_db }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq::{Band, BandKind};

    /// A "room": a delay, a 60 Hz mode (+8 dB), a 120 Hz dip (−6 dB) and a gentle treble
    /// roll-off, as an impulse response at 48 kHz.
    fn room(rate: u32) -> Vec<f64> {
        let mut x = vec![0.0; rate as usize];
        x[240] = 0.5; // 5 ms of flight time
        let mut eq = crate::eq::Equalizer::new(
            rate,
            1,
            &[
                Band {
                    kind: BandKind::Peaking,
                    freq: 60.0,
                    gain_db: 8.0,
                    q: 5.0,
                    enabled: true,
                },
                Band {
                    kind: BandKind::Peaking,
                    freq: 120.0,
                    gain_db: -6.0,
                    q: 4.0,
                    enabled: true,
                },
                Band {
                    kind: BandKind::HighShelf,
                    freq: 8000.0,
                    gain_db: -3.0,
                    q: 0.7,
                    enabled: true,
                },
            ],
        );
        eq.process(&mut x);
        x
    }

    #[test]
    fn sweep_recovers_the_impulse_response() {
        let rate = 48000;
        let sweep = Sweep::new(rate, 10.0, 22000.0, 3.0);
        let h = room(rate);
        // Play: half a second of silence, the sweep, a second for the tail; a little noise.
        let mut played = vec![0.0; rate as usize / 2];
        played.extend(&sweep.signal);
        played.extend(vec![0.0; rate as usize]);
        let mut rec = convolve(&played, &h);
        let mut seed = 7u64;
        for v in &mut rec {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v += ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e-4;
        }
        let ir = sweep.impulse(&rec, 5.0, 500.0);
        // Same response as the true room, within 0.5 dB from 30 Hz to 15 kHz.
        let grid = log_grid(30.0, 15000.0, 120);
        let want = response(&h, rate, &grid, 6.0);
        let got = response(&ir, rate, &grid, 6.0);
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 0.5, "{worst:.2} dB");
    }

    #[test]
    fn correction_flattens_the_modes_within_limits() {
        let rate = 48000;
        let h = room(rate);
        let o = DesignOptions::default();
        let c = design(&h, rate, &o);
        assert_eq!(c.fir.len(), o.taps);
        let corrected = convolve(&h, &c.fir);
        let grid = log_grid(30.0, 400.0, 80);
        let before = response(&h, rate, &grid, 6.0);
        let after = response(&corrected, rate, &grid, 6.0);
        let dev = |r: &[f64]| {
            r.iter()
                .map(|v| (v - c.reference_db).abs())
                .fold(0.0, f64::max)
        };
        assert!(
            dev(&before) > 6.0,
            "the room is uneven: {:.1}",
            dev(&before)
        );
        // Within 1.5 dB of flat over the corrected band.
        assert!(dev(&after) < 1.5, "after: {:.2} dB", dev(&after));
        // Above the band the correction leaves the speaker alone.
        let hi = log_grid(2000.0, 10000.0, 20);
        let (b, a) = (
            response(&h, rate, &hi, 6.0),
            response(&corrected, rate, &hi, 6.0),
        );
        let change = b
            .iter()
            .zip(&a)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(change < 0.3, "{change:.2} dB above the band");
        // Never more than the allowed boost.
        let fir_gain = response(&c.fir, rate, &log_grid(20.0, 20000.0, 200), 12.0);
        assert!(fir_gain.iter().all(|g| *g < o.max_boost_db + 0.5));
    }

    #[test]
    fn min_phase_is_causal_and_follows_the_magnitude() {
        let fs = 48000.0;
        let h = min_phase(4096, fs, |f| if f < 1000.0 { 6.0 } else { 0.0 });
        // Energy front-loaded (minimum phase).
        let e: f64 = h.iter().map(|v| v * v).sum();
        let e_head: f64 = h[..256].iter().map(|v| v * v).sum();
        assert!(e_head / e > 0.95);
        let g = |f| 20.0 * gain_at(&h, fs, f).log10();
        assert!((g(100.0) - 6.0).abs() < 0.3, "{}", g(100.0));
        assert!(g(10000.0).abs() < 0.3, "{}", g(10000.0));
    }
}
