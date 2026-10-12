//! Long FIR low-pass design: Kaiser windowed sinc, minimum-phase conversion through the
//! real cepstrum, and the named filter catalogue (`sinc-l-64k`, `sinc-ma-1m`, …).
//!
//! Everything is computed in `f64`. The designs are deterministic, so they can be cached
//! on disk (see [`crate::cache`]); a 4M-tap minimum-phase design takes a few seconds.

use crate::fft::{next_pow2, RealFft};
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Shortest filter the designer accepts.
pub const MIN_TAPS: usize = 1 << 8;
/// Longest filter the designer accepts (2^22 taps, plus one so the length is odd).
pub const MAX_TAPS: usize = (1 << 22) + 1;

/// Phase behaviour of a filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Symmetric impulse response: no phase distortion, pre- and post-ringing, latency of
    /// half the filter length.
    Linear,
    /// Energy pushed to the front of the impulse (cepstral conversion): no pre-ringing,
    /// low latency, frequency-dependent delay.
    Minimum,
}

impl Phase {
    /// Lower-case name used in ids and the signal path.
    pub fn name(self) -> &'static str {
        match self {
            Phase::Linear => "linear",
            Phase::Minimum => "minimum",
        }
    }
}

/// A complete description of one low-pass design.
///
/// `cutoff` is normalised to the sample rate the filter runs at (cycles per sample, so
/// Nyquist is `0.5`) and marks the centre of the transition band, i.e. the -6 dB point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirSpec {
    /// Number of coefficients. Always made odd so the linear-phase delay is a whole sample.
    pub taps: usize,
    /// Normalised cutoff (centre of the transition band).
    pub cutoff: f64,
    /// Kaiser design stopband attenuation in dB (also bounds the passband ripple).
    pub attenuation_db: f64,
    /// Linear or minimum phase.
    pub phase: Phase,
}

impl FirSpec {
    /// A design from a tap count: the transition band follows from `taps` and `attenuation_db`.
    pub fn from_taps(taps: usize, cutoff: f64, attenuation_db: f64, phase: Phase) -> Self {
        let taps = (taps.clamp(MIN_TAPS, MAX_TAPS)) | 1;
        assert!(
            cutoff > 0.0 && cutoff < 0.5,
            "cutoff must be in (0, 0.5), got {cutoff}"
        );
        Self {
            taps,
            cutoff,
            attenuation_db: attenuation_db.clamp(30.0, 300.0),
            phase,
        }
    }

    /// A design from a transition width (normalised): the tap count follows from the
    /// Kaiser estimate `N ≈ (A − 8) / (2.285 · Δω) + 1` and is clamped to the legal range.
    pub fn from_transition(
        cutoff: f64,
        transition: f64,
        attenuation_db: f64,
        phase: Phase,
    ) -> Self {
        let a = attenuation_db.clamp(30.0, 300.0);
        let delta_omega = 2.0 * PI * transition.max(1e-9);
        let taps = ((a - 8.0) / (2.285 * delta_omega)).ceil() as usize + 1;
        Self::from_taps(taps, cutoff, a, phase)
    }

    /// Kaiser window shape parameter for the requested attenuation.
    pub fn kaiser_beta(&self) -> f64 {
        kaiser_beta(self.attenuation_db)
    }

    /// Width of the transition band (normalised frequency) this design achieves.
    pub fn transition_width(&self) -> f64 {
        (self.attenuation_db - 8.0) / (2.285 * 2.0 * PI * (self.taps as f64 - 1.0))
    }

    /// Where the passband ends (normalised): the ripple stays within the Kaiser bound below this.
    pub fn passband_edge(&self) -> f64 {
        self.cutoff - self.transition_width() / 2.0
    }

    /// Where the stopband starts (normalised): attenuation is at least `attenuation_db` above this.
    pub fn stopband_edge(&self) -> f64 {
        self.cutoff + self.transition_width() / 2.0
    }

    /// Group delay in samples: exact for linear phase, approximately zero for minimum phase.
    pub fn group_delay(&self) -> usize {
        match self.phase {
            Phase::Linear => (self.taps - 1) / 2,
            Phase::Minimum => 0,
        }
    }

    /// A file-system safe, human readable key that identifies this exact design.
    pub fn cache_key(&self) -> String {
        format!(
            "sinc-{}-{}-c{}-a{}",
            self.phase.name(),
            self.taps,
            format!("{:.12e}", self.cutoff).replace(['+', '-'], "_"),
            self.attenuation_db as i64
        )
    }

    /// Compute the coefficients (unity DC gain). Not cached; see [`crate::cache`].
    pub fn design(&self) -> Vec<f64> {
        let linear = windowed_sinc(self.taps, self.cutoff, self.kaiser_beta());
        match self.phase {
            Phase::Linear => linear,
            // Everything below the floor is treated as "zero" by the cepstrum: the true
            // nulls of the window design go to -inf dB and would otherwise dominate the log.
            Phase::Minimum => minimum_phase(&linear, self.attenuation_db + 60.0),
        }
    }
}

/// Kaiser `β` for a stopband attenuation `a` in dB (Kaiser's empirical formula).
pub fn kaiser_beta(a: f64) -> f64 {
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

/// Modified Bessel function of the first kind, order zero, by its power series.
///
/// The series converges for every `x`; for the `β ≲ 35` used here it needs well under a
/// hundred terms, and the terms are computed by recurrence so there is no factorial overflow.
pub fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    loop {
        term *= (half / k) * (half / k);
        sum += term;
        if term < sum * 1e-17 {
            return sum;
        }
        k += 1.0;
    }
}

/// Kaiser window of `n` points.
pub fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    let denom = bessel_i0(beta);
    let m = (n as f64 - 1.0) / 2.0;
    let mut w = vec![0.0; n];
    // The window is symmetric, so compute one half and mirror it.
    for i in 0..n.div_ceil(2) {
        let r = (i as f64 - m) / m;
        let arg = (1.0 - r * r).max(0.0).sqrt();
        let v = bessel_i0(beta * arg) / denom;
        w[i] = v;
        w[n - 1 - i] = v;
    }
    w
}

/// Linear-phase Kaiser windowed sinc low-pass with unity DC gain.
pub fn windowed_sinc(taps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let w = kaiser_window(taps, beta);
    let m = (taps as f64 - 1.0) / 2.0;
    let mut h = vec![0.0; taps];
    for i in 0..taps.div_ceil(2) {
        let x = i as f64 - m;
        let s = if x == 0.0 {
            2.0 * cutoff
        } else {
            (2.0 * PI * cutoff * x).sin() / (PI * x)
        };
        let v = s * w[i];
        h[i] = v;
        h[taps - 1 - i] = v;
    }
    let sum: f64 = h.iter().sum();
    for v in &mut h {
        *v /= sum;
    }
    h
}

/// Convert any FIR into its minimum-phase equivalent (same magnitude response) with the
/// homomorphic / real-cepstrum method:
///
/// 1. `C = log |H|` on a dense grid (4× zero padding keeps cepstral aliasing small),
/// 2. real cepstrum `c = IFFT(C)`, folded so only the causal part remains
///    (`c[0]`, `2·c[n]` for `0 < n < N/2`, `c[N/2]`),
/// 3. `h_min = IFFT(exp(FFT(c_folded)))`, truncated to the original length.
///
/// `floor_db` is the level below the peak at which the magnitude is clamped before taking
/// the logarithm; it bounds how deep the result's stopband nulls can be.
pub fn minimum_phase(h: &[f64], floor_db: f64) -> Vec<f64> {
    let n = h.len();
    // Memory for a 4M-tap design at 4× padding would be several hundred MB, so pad less there.
    let pad = if n > (1 << 20) { 2 } else { 4 };
    let m = next_pow2(n) * pad;
    let mut fft = RealFft::new(m);

    let mut buf = vec![0.0; m];
    buf[..n].copy_from_slice(h);
    let mut spec = fft.spectrum();
    fft.forward(&mut buf, &mut spec);

    let peak = spec
        .iter()
        .map(|c| c.norm())
        .fold(0.0, f64::max)
        .max(1e-300);
    let floor = peak * 10f64.powf(-floor_db / 20.0);
    let mut log_mag: Vec<Complex<f64>> = spec
        .iter()
        .map(|c| Complex::new(c.norm().max(floor).ln(), 0.0))
        .collect();

    let mut ceps = vec![0.0; m];
    fft.inverse(&mut log_mag, &mut ceps);
    for v in ceps.iter_mut().take(m / 2).skip(1) {
        *v *= 2.0;
    }
    for v in ceps.iter_mut().skip(m / 2 + 1) {
        *v = 0.0;
    }

    let mut cspec = fft.spectrum();
    fft.forward(&mut ceps, &mut cspec);
    for c in cspec.iter_mut() {
        *c = c.exp();
    }
    let mut out = vec![0.0; m];
    fft.inverse(&mut cspec, &mut out);
    out.truncate(n);

    // Truncation and the floor move the DC gain by a hair; put it back exactly.
    let target: f64 = h.iter().sum();
    let got: f64 = out.iter().sum();
    if got.abs() > 1e-300 {
        let k = target / got;
        for v in &mut out {
            *v *= k;
        }
    }
    out
}

/// A user-facing filter choice: the shape of the filter independent of the sample rates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FilterKind {
    /// Linear or minimum phase.
    pub phase: Phase,
    /// Apodizing designs place the cutoff slightly below the source Nyquist so the
    /// (pre-)ringing of the recording's own anti-alias filter is attenuated.
    pub apodizing: bool,
    /// Nominal length; the designer makes it odd.
    pub taps: usize,
}

/// How far below the source Nyquist the apodizing variants put their cutoff
/// (0.95 → 20.95 kHz for 44.1 kHz material, like HQPlayer's `-2s` family).
pub const APODIZING_CUTOFF: f64 = 0.95;

/// Default stopband attenuation for the catalogue designs.
pub const DEFAULT_ATTENUATION_DB: f64 = 150.0;

impl FilterKind {
    /// The catalogue id, e.g. `sinc-l-64k`, `sinc-ma-1m`.
    pub fn id(&self) -> String {
        let p = match (self.phase, self.apodizing) {
            (Phase::Linear, false) => "l",
            (Phase::Minimum, false) => "m",
            (Phase::Linear, true) => "la",
            (Phase::Minimum, true) => "ma",
        };
        format!("sinc-{p}-{}", taps_label(self.taps))
    }

    /// Human readable label, e.g. `sinc-L 64k taps · linear phase`.
    pub fn label(&self) -> String {
        format!(
            "sinc-{}{} {} taps · {} phase{}",
            if self.phase == Phase::Linear {
                "L"
            } else {
                "M"
            },
            if self.apodizing { "a" } else { "" },
            taps_label(self.taps),
            self.phase.name(),
            if self.apodizing { " · apodizing" } else { "" }
        )
    }

    /// Parse an id produced by [`FilterKind::id`]; the tap count may be plain (`65536`)
    /// or use `k`/`m` suffixes (`64k`, `1m`).
    pub fn parse(id: &str) -> Option<Self> {
        let rest = id.trim().to_ascii_lowercase();
        let rest = rest.strip_prefix("sinc-")?;
        let (p, taps) = rest.split_once('-')?;
        let (phase, apodizing) = match p {
            "l" => (Phase::Linear, false),
            "m" => (Phase::Minimum, false),
            "la" => (Phase::Linear, true),
            "ma" => (Phase::Minimum, true),
            _ => return None,
        };
        let taps = parse_taps(taps)?;
        if !(MIN_TAPS..=MAX_TAPS).contains(&taps) {
            return None;
        }
        Some(Self {
            phase,
            apodizing,
            taps,
        })
    }

    /// The design for an integer upsampling ratio `l` (the filter runs at the output rate,
    /// so the source Nyquist sits at `1 / (2 l)`), with the given attenuation.
    pub fn spec_for_ratio(&self, l: usize, m: usize, attenuation_db: f64) -> FirSpec {
        // For a rational L/M converter the filter runs at L × input rate and must keep
        // everything below the lower of the two Nyquist frequencies.
        let nyquist = 0.5 / l.max(m) as f64;
        let cutoff = if self.apodizing {
            nyquist * APODIZING_CUTOFF
        } else {
            nyquist
        };
        FirSpec::from_taps(self.taps, cutoff, attenuation_db, self.phase)
    }

    /// Every filter offered by default, in display order.
    pub fn catalog() -> Vec<FilterKind> {
        let mut out = Vec::new();
        for &taps in &[1 << 8, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22] {
            for &(phase, apodizing) in &[
                (Phase::Linear, false),
                (Phase::Minimum, false),
                (Phase::Linear, true),
                (Phase::Minimum, true),
            ] {
                out.push(FilterKind {
                    phase,
                    apodizing,
                    taps,
                });
            }
        }
        out
    }
}

impl Default for FilterKind {
    fn default() -> Self {
        FilterKind {
            phase: Phase::Linear,
            apodizing: false,
            taps: 1 << 16,
        }
    }
}

/// `65536 → "64k"`, `1048576 → "1m"`, `300 → "300"`.
pub fn taps_label(taps: usize) -> String {
    let t = taps & !1; // drop the "+1" the designer adds
    if t >= 1 << 20 && t % (1 << 20) == 0 {
        format!("{}m", t >> 20)
    } else if t >= 1024 && t % 1024 == 0 {
        format!("{}k", t >> 10)
    } else {
        taps.to_string()
    }
}

fn parse_taps(s: &str) -> Option<usize> {
    if let Some(n) = s.strip_suffix('m') {
        n.parse::<usize>().ok().map(|n| n << 20)
    } else if let Some(n) = s.strip_suffix('k') {
        n.parse::<usize>().ok().map(|n| n << 10)
    } else {
        s.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::magnitude_db;

    /// Measure passband ripple and stopband attenuation of `h` for `spec` on an FFT grid.
    fn measure(h: &[f64], spec: &FirSpec, nfft: usize) -> (f64, f64) {
        let db = magnitude_db(h, nfft);
        let nfft = db.len() * 2 - 2;
        let bin = |f: f64| (f * nfft as f64).round() as usize;
        let pass_end = bin(spec.passband_edge());
        let stop_start = bin(spec.stopband_edge());
        let ripple = db[..=pass_end].iter().map(|d| d.abs()).fold(0.0, f64::max);
        let stop = db[stop_start..].iter().cloned().fold(f64::MIN, f64::max);
        (ripple, -stop)
    }

    #[test]
    fn bessel_matches_known_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008).abs() < 1e-12);
        assert!((bessel_i0(5.0) - 27.239_871_823_604_44).abs() < 1e-9);
    }

    #[test]
    fn kaiser_2_16_taps_meets_the_design_band() {
        let spec = FirSpec::from_taps(1 << 16, 0.125, 150.0, Phase::Linear);
        assert_eq!(spec.taps, (1 << 16) + 1);
        let h = spec.design();
        assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let (ripple, atten) = measure(&h, &spec, 1 << 19);
        assert!(ripple < 1e-4, "passband ripple {ripple} dB");
        assert!(atten >= 120.0, "stopband attenuation {atten} dB");
    }

    #[test]
    fn transition_width_estimate_round_trips() {
        let spec = FirSpec::from_transition(0.25, 0.001, 120.0, Phase::Linear);
        let got = spec.transition_width();
        assert!((got - 0.001).abs() / 0.001 < 0.01, "transition {got}");
    }

    #[test]
    fn minimum_phase_front_loads_energy_and_keeps_magnitude() {
        let spec = FirSpec::from_taps(1 << 12, 0.125, 120.0, Phase::Minimum);
        let lin = FirSpec {
            phase: Phase::Linear,
            ..spec.clone()
        }
        .design();
        let min = spec.design();
        assert_eq!(min.len(), lin.len());
        let energy = |h: &[f64]| h.iter().map(|v| v * v).sum::<f64>();
        let tenth = min.len() / 10;
        let front = energy(&min[..tenth]) / energy(&min);
        let front_lin = energy(&lin[..tenth]) / energy(&lin);
        assert!(front > 0.9, "min-phase front energy {front}");
        assert!(front_lin < 0.01, "linear-phase front energy {front_lin}");

        let (ripple, atten) = measure(&min, &spec, 1 << 16);
        assert!(ripple < 0.05, "min-phase passband ripple {ripple} dB");
        assert!(atten >= 100.0, "min-phase stopband {atten} dB");
    }

    #[test]
    fn ids_round_trip_and_catalog_is_consistent() {
        for k in FilterKind::catalog() {
            assert_eq!(FilterKind::parse(&k.id()), Some(k), "{}", k.id());
        }
        assert_eq!(FilterKind::parse("sinc-ma-1m").unwrap().taps, 1 << 20);
        assert_eq!(FilterKind::parse("sinc-l-65536").unwrap().taps, 65536);
        assert!(FilterKind::parse("sinc-x-64k").is_none());
        assert!(FilterKind::parse("sinc-l-7").is_none());
        let spec = FilterKind::default().spec_for_ratio(4, 1, 150.0);
        assert!((spec.cutoff - 0.125).abs() < 1e-15);
        let apod = FilterKind {
            apodizing: true,
            ..Default::default()
        }
        .spec_for_ratio(4, 1, 150.0);
        assert!(apod.cutoff < spec.cutoff);
    }
}
