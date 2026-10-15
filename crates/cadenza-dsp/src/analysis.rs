//! Measuring what came back: THD+N of a test tone, and a bit-exact comparison of a played
//! pattern with its recording (loopback verification).

use crate::fft::RealFft;
use serde::Serialize;
use std::f64::consts::PI;

/// The 7-term Blackman–Harris window: sidelobes below −180 dB, so a −140 dB residual is not
/// swamped by the fundamental's leakage.
fn blackman_harris_7(n: usize) -> Vec<f64> {
    const A: [f64; 7] = [
        0.271_051_400_693_424_2,
        -0.433_297_939_234_485_6,
        0.218_122_999_543_110_6,
        -0.065_925_446_388_030_9,
        0.010_811_742_098_372_3,
        -0.000_776_584_825_226_3,
        0.000_013_887_217_352_3,
    ];
    (0..n)
        .map(|i| {
            let t = 2.0 * PI * i as f64 / n as f64;
            A.iter()
                .enumerate()
                .map(|(k, a)| a * (k as f64 * t).cos())
                .sum()
        })
        .collect()
}

/// Half-width (bins) of the window's main lobe, plus a margin.
const LOBE: usize = 9;

/// A THD+N measurement of a sine.
#[derive(Clone, Debug, Serialize)]
pub struct ThdN {
    /// Measured frequency of the fundamental (Hz).
    pub frequency: f64,
    /// Its level (dBFS, sine-referenced).
    pub level_db: f64,
    /// Everything but the fundamental in 20 Hz – 20 kHz, relative to it (dB).
    pub thd_n_db: f64,
    /// The same in percent.
    pub thd_n_percent: f64,
    /// Harmonics 2 … 10 (those below 20 kHz), relative to the fundamental (dB).
    pub thd_db: f64,
    /// THD+N without the harmonics (dB).
    pub noise_db: f64,
    /// Each harmonic's level relative to the fundamental (dB), from the 2nd.
    pub harmonics_db: Vec<f64>,
}

/// Measure THD+N of `x` (one channel at `rate`) around the expected frequency `f0`.
pub fn thd_n(x: &[f64], rate: f64, f0: f64) -> Option<ThdN> {
    if x.len() < 1 << 12 {
        return None;
    }
    let n = 1usize << x.len().ilog2().min(20);
    let x = &x[(x.len() - n) / 2..][..n];
    let w = blackman_harris_7(n);
    let mut fft = RealFft::new(n);
    let mut buf: Vec<f64> = x.iter().zip(&w).map(|(v, w)| v * w).collect();
    let mut spec = fft.spectrum();
    fft.forward(&mut buf, &mut spec);
    let p: Vec<f64> = spec.iter().map(|c| c.norm_sqr()).collect();
    let hz = rate / n as f64;
    let bin = |f: f64| ((f / hz).round() as usize).min(p.len() - 1);
    // The fundamental: the strongest bin within ±5 % of f0.
    let (lo, hi) = (bin(f0 * 0.95), bin(f0 * 1.05));
    let peak = (lo..=hi).max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap())?;
    let band = |c: usize| c.saturating_sub(LOBE)..(c + LOBE + 1).min(p.len());
    let power = |c: usize| p[band(c)].iter().sum::<f64>();
    let fund = power(peak);
    if fund <= 0.0 {
        return None;
    }
    // Frequency from the power-weighted centre of the lobe.
    let r = band(peak);
    let frequency =
        r.clone().map(|i| i as f64 * p[i]).sum::<f64>() / r.map(|i| p[i]).sum::<f64>() * hz;
    let (b20, b20k) = (bin(20.0), bin(20_000f64.min(rate / 2.0)));
    let total: f64 = p[b20..=b20k].iter().sum();
    let fund_lobe = band(peak);
    let in_band_fund: f64 = p[fund_lobe.start.max(b20)..fund_lobe.end.min(b20k + 1)]
        .iter()
        .sum();
    let rest = (total - in_band_fund).max(1e-300);
    let mut harmonics_db = Vec::new();
    let mut harm = 0.0;
    for k in 2..=10 {
        let f = frequency * k as f64;
        if f > 20_000f64.min(rate / 2.0) - LOBE as f64 * hz {
            break;
        }
        let ph = power(bin(f));
        harm += ph;
        harmonics_db.push(10.0 * (ph / fund).max(1e-30).log10());
    }
    // A full-scale sine windowed and summed this way: its power is the window's
    // incoherent gain times n²/4 for amplitude 1.
    let gain: f64 = w.iter().map(|v| v * v).sum::<f64>() * n as f64 / 4.0;
    let amp = (fund / gain).sqrt();
    let db = |v: f64| 10.0 * v.max(1e-30).log10();
    Some(ThdN {
        frequency,
        level_db: 20.0 * amp.max(1e-15).log10(),
        thd_n_db: db(rest / fund),
        thd_n_percent: 100.0 * (rest / fund).sqrt(),
        thd_db: db(harm / fund),
        noise_db: db((rest - harm).max(1e-300) / fund),
        harmonics_db,
    })
}

/// How a recording compares with what was played, sample for sample.
#[derive(Clone, Debug, Default, Serialize)]
pub struct BitCompare {
    /// Where in the recording the pattern starts (samples), if it was found.
    pub offset: Option<usize>,
    /// Samples compared.
    pub compared: usize,
    /// Samples that differed.
    pub mismatches: usize,
    /// Index (in the pattern) of the first difference.
    pub first_mismatch: Option<usize>,
    /// Every compared sample matched (and enough of them were compared).
    pub bit_perfect: bool,
}

/// Look for `sent` (integers) in `got` (integers at the same scale) and compare. The start
/// is found by an exact match of the first `key` samples (the pattern is pseudo-random, so
/// a match there is no accident); everything after is compared.
pub fn bit_compare(sent: &[i32], got: &[i32], key: usize) -> BitCompare {
    let key = key.min(sent.len());
    if key == 0 || got.len() < key {
        return BitCompare::default();
    }
    let offset = (0..=got.len() - key).find(|&o| got[o..o + key] == sent[..key]);
    let Some(offset) = offset else {
        return BitCompare::default();
    };
    let compared = sent.len().min(got.len() - offset);
    let diffs: Vec<usize> = (0..compared)
        .filter(|&i| got[offset + i] != sent[i])
        .collect();
    BitCompare {
        offset: Some(offset),
        compared,
        mismatches: diffs.len(),
        first_mismatch: diffs.first().copied(),
        bit_perfect: diffs.is_empty() && compared == sent.len(),
    }
}

/// A pseudo-random pattern of `n` integers of `bits` bits (full range, reproducible).
pub fn pattern(n: usize, bits: u32, seed: u64) -> Vec<i32> {
    let mut s = seed | 1;
    let half = 1i64 << (bits - 1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as i64 % (2 * half) - half) as i32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: f64, f: f64, amp: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * f * i as f64 / rate).sin())
            .collect()
    }

    #[test]
    fn a_pure_sine_measures_clean_and_at_its_level() {
        let x = sine(48_000.0, 997.0, 0.5, 1 << 17);
        let m = thd_n(&x, 48_000.0, 1000.0).unwrap();
        assert!((m.frequency - 997.0).abs() < 0.5, "{}", m.frequency);
        assert!(
            (m.level_db - 20.0 * 0.5f64.log10()).abs() < 0.05,
            "{}",
            m.level_db
        );
        assert!(m.thd_n_db < -150.0, "{}", m.thd_n_db);
    }

    #[test]
    fn known_distortion_and_noise_are_measured() {
        let rate = 48_000.0;
        let a = 0.5;
        let k = 0.02;
        // y = x + k x³: the 3rd harmonic is k a³ / 4, so THD = k a² / 4.
        let mut s = 7u64;
        let y: Vec<f64> = sine(rate, 997.0, a, 1 << 17)
            .into_iter()
            .map(|x| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2e-5;
                x + k * x * x * x + noise
            })
            .collect();
        let m = thd_n(&y, rate, 1000.0).unwrap();
        let want = 20.0 * (k * a * a / 4.0 / (1.0 + 3.0 * k * a * a / 4.0)).log10();
        assert!((m.thd_db - want).abs() < 0.2, "THD {} vs {want}", m.thd_db);
        assert!((m.harmonics_db[1] - want).abs() < 0.2);
        // Uniform noise ±1e-5 over 20 Hz – 20 kHz of 24 kHz, relative to the sine's power.
        let noise =
            10.0 * ((2e-5f64).powi(2) / 12.0 * (19_980.0 / 24_000.0) / (a * a / 2.0)).log10();
        assert!(
            (m.noise_db - noise).abs() < 1.0,
            "noise {} vs {noise}",
            m.noise_db
        );
        assert!(m.thd_n_db >= m.thd_db && m.thd_n_db >= m.noise_db);
    }

    #[test]
    fn bit_compare_finds_the_offset_and_the_differences() {
        let sent = pattern(10_000, 24, 1);
        assert!(sent.iter().all(|&v| (-(1 << 23)..(1 << 23)).contains(&v)));
        let mut got = vec![0i32; 123];
        got.extend(&sent);
        got.extend([0; 50]);
        let c = bit_compare(&sent, &got, 32);
        assert_eq!((c.offset, c.compared, c.mismatches), (Some(123), 10_000, 0));
        assert!(c.bit_perfect);
        got[123 + 5000] ^= 1;
        let c = bit_compare(&sent, &got, 32);
        assert_eq!((c.mismatches, c.first_mismatch), (1, Some(5000)));
        assert!(!c.bit_perfect);
        // A gain change anywhere: not found at all.
        let halved: Vec<i32> = got.iter().map(|v| v / 2).collect();
        assert_eq!(bit_compare(&sent, &halved, 32).offset, None);
    }
}
