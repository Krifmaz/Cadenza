//! Crossover filters for active speakers: Butterworth (6…48 dB/oct), Linkwitz–Riley
//! (12/24/36/48 dB/oct, the low- and high-pass sum to an all-pass, i.e. flat magnitude) and
//! a linear-phase FIR option built on the Kaiser windowed sinc of [`crate::fir`], where the
//! high-pass is the exact complement of the low-pass (`δ − LP`) so every band of a complete
//! set sums to a pure delay.
//!
//! The IIR cascades are run as transposed direct form II sections in `f64`; the FIR
//! filters are run by the matrix stage through the partitioned convolver.

use crate::eq::{Band, BandKind, Coefficients};
use crate::fir::{kaiser_beta, windowed_sinc};
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Stopband attenuation the linear-phase FIR crossovers are designed for, in dB.
pub const FIR_ATTENUATION_DB: f64 = 100.0;
/// Smallest FIR crossover (taps); keeps very high crossover points sane.
pub const FIR_MIN_TAPS: usize = 63;
/// Largest FIR crossover (taps); a 20 Hz crossover at 768 kHz would otherwise explode.
pub const FIR_MAX_TAPS: usize = 1 << 20;

/// Filter family of one crossover.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CrossoverKind {
    /// Butterworth of order `slope_db / 6` (1…8). The low/high pair sums with a +3 dB bump
    /// at the crossover point for even orders.
    Butterworth,
    /// Linkwitz–Riley: two cascaded Butterworth filters of half the order (`slope_db` =
    /// 12, 24, 36 or 48). Low + high is an all-pass, so the acoustic sum is flat. For
    /// LR12 and LR36 the high-pass is inverted internally (as a real install wires the
    /// tweeter) so that the sum stays flat.
    #[default]
    LinkwitzRiley,
    /// Linear-phase FIR (Kaiser windowed sinc, ≈ 100 dB stopband, half-octave transition);
    /// `slope_db` is ignored. Adds the FIR group delay; channels without an FIR are
    /// delayed to match.
    LinearPhase,
}

impl CrossoverKind {
    /// Stable id used in config.toml and the API.
    pub fn id(self) -> &'static str {
        match self {
            CrossoverKind::Butterworth => "butterworth",
            CrossoverKind::LinkwitzRiley => "linkwitz-riley",
            CrossoverKind::LinearPhase => "linear-phase",
        }
    }

    /// Short display label (`BW`, `LR`, `FIR`).
    pub fn short(self) -> &'static str {
        match self {
            CrossoverKind::Butterworth => "BW",
            CrossoverKind::LinkwitzRiley => "LR",
            CrossoverKind::LinearPhase => "FIR",
        }
    }

    /// Parse an id or a common alias (`lr`, `bw`, `fir`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "butterworth" | "bw" => Some(CrossoverKind::Butterworth),
            "linkwitz-riley" | "linkwitz_riley" | "lr" => Some(CrossoverKind::LinkwitzRiley),
            "linear-phase" | "linear_phase" | "fir" => Some(CrossoverKind::LinearPhase),
            _ => None,
        }
    }

    /// Slopes (dB/oct) this family can be built with.
    pub fn slopes(self) -> &'static [u32] {
        match self {
            CrossoverKind::Butterworth => &[6, 12, 18, 24, 30, 36, 42, 48],
            CrossoverKind::LinkwitzRiley => &[12, 24, 36, 48],
            CrossoverKind::LinearPhase => &[],
        }
    }

    /// Every family, for the UI.
    pub fn all() -> &'static [CrossoverKind] {
        &[
            CrossoverKind::Butterworth,
            CrossoverKind::LinkwitzRiley,
            CrossoverKind::LinearPhase,
        ]
    }
}

/// One crossover filter (a low-pass or a high-pass, decided by where it is used).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrossoverFilter {
    /// Family.
    #[serde(default)]
    pub kind: CrossoverKind,
    /// Crossover frequency in Hz.
    pub freq: f64,
    /// Slope in dB per octave (ignored for `linear-phase`).
    #[serde(default = "default_slope")]
    pub slope_db: u32,
}

fn default_slope() -> u32 {
    24
}

impl CrossoverFilter {
    /// Any family at `freq`.
    pub fn new(kind: CrossoverKind, freq: f64, slope_db: u32) -> Self {
        Self {
            kind,
            freq,
            slope_db,
        }
    }

    /// Linkwitz–Riley 24 dB/oct at `freq` (the usual choice).
    pub fn lr24(freq: f64) -> Self {
        Self::new(CrossoverKind::LinkwitzRiley, freq, 24)
    }

    /// Linear-phase FIR at `freq`.
    pub fn linear_phase(freq: f64) -> Self {
        Self::new(CrossoverKind::LinearPhase, freq, 0)
    }

    /// Check frequency and slope for a stream at `rate` Hz.
    pub fn validate(&self, rate: f64) -> Result<(), String> {
        if !(self.freq > 0.0 && self.freq < rate / 2.0) {
            return Err(format!(
                "crossover frequency {} Hz is outside (0, {}) Hz",
                self.freq,
                rate / 2.0
            ));
        }
        if self.kind != CrossoverKind::LinearPhase && !self.kind.slopes().contains(&self.slope_db) {
            return Err(format!(
                "{} slope {} dB/oct is not one of {:?}",
                self.kind.id(),
                self.slope_db,
                self.kind.slopes()
            ));
        }
        Ok(())
    }

    /// `LR24 @ 2 kHz`, `BW12 @ 80 Hz`, `FIR @ 300 Hz`.
    pub fn label(&self) -> String {
        match self.kind {
            CrossoverKind::LinearPhase => format!("FIR @ {}", freq_label(self.freq)),
            k => format!("{}{} @ {}", k.short(), self.slope_db, freq_label(self.freq)),
        }
    }

    /// Taps of the linear-phase FIR for this filter at `rate` Hz (odd, so the group delay
    /// is a whole number of samples).
    pub fn fir_taps(&self, rate: f64) -> usize {
        fir_taps_for(self.freq, rate)
    }
}

/// `80 Hz`, `2 kHz`, `2.5 kHz`.
pub fn freq_label(freq: f64) -> String {
    if freq >= 1000.0 {
        let k = freq / 1000.0;
        if (k - k.round()).abs() < 1e-9 {
            format!("{} kHz", k.round() as u32)
        } else {
            let t = format!("{k:.2}");
            format!("{} kHz", t.trim_end_matches('0').trim_end_matches('.'))
        }
    } else if (freq - freq.round()).abs() < 1e-9 {
        format!("{} Hz", freq.round() as u32)
    } else {
        format!("{freq:.1} Hz")
    }
}

/// Low-pass or high-pass side of a crossover, or their sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// Keep what is below the crossover frequency.
    Low,
    /// Keep what is above it.
    High,
    /// `LP + HP`: for Linkwitz–Riley (and odd-order Butterworth) an all-pass with the phase
    /// of the crossover. A band that does not take part in a crossover point runs through
    /// its all-pass so a 3-way (or more) set still sums flat.
    All,
}

/// Kaiser windowed-sinc tap count for a half-octave transition around `freq`, odd.
pub fn fir_taps_for(freq: f64, rate: f64) -> usize {
    let tw = 0.5 * freq / rate;
    let n = ((FIR_ATTENUATION_DB - 8.0) / (2.285 * 2.0 * PI * tw)).ceil() as usize;
    let n = n.clamp(FIR_MIN_TAPS, FIR_MAX_TAPS);
    n | 1
}

/// Linear-phase low-pass of `taps` (odd) coefficients at `freq` Hz, unity DC gain.
pub fn fir_lowpass(taps: usize, freq: f64, rate: f64) -> Vec<f64> {
    windowed_sinc(taps, freq / rate, kaiser_beta(FIR_ATTENUATION_DB))
}

/// A unit impulse of `taps` coefficients centred like the low-passes (the "all-pass" in
/// the complementary set: `HP = δ − LP`).
pub fn fir_delta(taps: usize) -> Vec<f64> {
    let mut h = vec![0.0; taps];
    h[(taps - 1) / 2] = 1.0;
    h
}

/// The complementary band between an optional high-pass at `hp` and an optional low-pass
/// at `lp`: `LP(lp) − LP(hp)` with `LP(none) = δ` on the low-pass side and `0` on the
/// high-pass side. All filters of a set designed with the same `taps` sum to `δ`.
pub fn fir_band(taps: usize, hp: Option<f64>, lp: Option<f64>, rate: f64) -> Vec<f64> {
    let mut h = match lp {
        Some(f) => fir_lowpass(taps, f, rate),
        None => fir_delta(taps),
    };
    if let Some(f) = hp {
        for (a, b) in h.iter_mut().zip(fir_lowpass(taps, f, rate)) {
            *a -= b;
        }
    }
    h
}

/// Filter-state magnitude below which the state is flushed to zero (see [`Section::tick`]).
pub(crate) const DENORMAL_FLOOR: f64 = 1e-290;

/// One first- or second-order section (transposed direct form II).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Section {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Section {
    fn from_coefficients(c: Coefficients) -> Self {
        Self {
            b0: c.b0,
            b1: c.b1,
            b2: c.b2,
            a1: c.a1,
            a2: c.a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Bilinear first-order low- or high-pass at `freq`.
    fn first_order(pass: Pass, freq: f64, rate: f64) -> Self {
        let k = (PI * freq / rate).tan();
        let a1 = (k - 1.0) / (k + 1.0);
        let (b0, b1) = match pass {
            Pass::Low => (k / (1.0 + k), k / (1.0 + k)),
            Pass::High => (1.0 / (1.0 + k), -1.0 / (1.0 + k)),
            Pass::All => unreachable!("all-pass is built from both sides"),
        };
        Self {
            b0,
            b1,
            b2: 0.0,
            a1,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn tick(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        // Denormal guard: during digital silence the states decay towards f64 subnormals,
        // where x86 cores slow down 10–100× per multiply; Rust does not set FTZ/DAZ. Far
        // below −5000 dBFS, so inaudible by any measure.
        if self.z1.abs() < DENORMAL_FLOOR {
            self.z1 = 0.0;
        }
        if self.z2.abs() < DENORMAL_FLOOR {
            self.z2 = 0.0;
        }
        y
    }

    fn response(&self, freq: f64, rate: f64) -> Complex<f64> {
        let w = 2.0 * PI * freq / rate;
        let z1 = Complex::new(w.cos(), -w.sin());
        let z2 = z1 * z1;
        (self.b0 + self.b1 * z1 + self.b2 * z2) / (1.0 + self.a1 * z1 + self.a2 * z2)
    }
}

/// Butterworth sections of order `order` (1…8) at `freq`.
fn butterworth_sections(order: usize, pass: Pass, freq: f64, rate: f64) -> Vec<Section> {
    let mut sections = Vec::with_capacity(order.div_ceil(2));
    let kind = match pass {
        Pass::Low => BandKind::LowPass,
        Pass::High => BandKind::HighPass,
        Pass::All => unreachable!("all-pass is built from both sides"),
    };
    for k in 1..=order / 2 {
        let q = 1.0 / (2.0 * ((2 * k - 1) as f64 * PI / (2.0 * order as f64)).sin());
        let band = Band {
            kind,
            freq,
            gain_db: 0.0,
            q,
            enabled: true,
            slope_db: 12,
        };
        sections.push(Section::from_coefficients(Coefficients::design(
            &band, rate,
        )));
    }
    if order % 2 == 1 {
        sections.push(Section::first_order(pass, freq, rate));
    }
    sections
}

/// An IIR crossover (one side) for one channel.
#[derive(Clone, Debug, PartialEq)]
pub struct IirCrossover {
    sections: Vec<Section>,
    /// `-1.0` when this side is inverted so that LR12/LR36 sum flat.
    sign: f64,
    /// `Pass::All`: the high-pass branch summed with `sections` (the low-pass).
    all: Option<(Vec<Section>, f64)>,
    pass: Pass,
}

impl IirCrossover {
    /// Build the `pass` side of `filter` for a stream at `rate` Hz. Linear-phase filters
    /// are not IIR: this builds a Linkwitz–Riley 24 for them (the matrix stage uses
    /// [`fir_band`] instead).
    pub fn new(filter: &CrossoverFilter, pass: Pass, rate: f64) -> Self {
        if pass == Pass::All {
            let lp = Self::new(filter, Pass::Low, rate);
            let hp = Self::new(filter, Pass::High, rate);
            return Self {
                sections: lp.sections,
                sign: lp.sign,
                all: Some((hp.sections, hp.sign)),
                pass,
            };
        }
        let (sections, sign) = match filter.kind {
            CrossoverKind::Butterworth => {
                let order = (filter.slope_db / 6).clamp(1, 8) as usize;
                (butterworth_sections(order, pass, filter.freq, rate), 1.0)
            }
            CrossoverKind::LinkwitzRiley | CrossoverKind::LinearPhase => {
                let half = if filter.kind == CrossoverKind::LinearPhase {
                    2
                } else {
                    (filter.slope_db / 12).clamp(1, 4) as usize
                };
                let mut s = butterworth_sections(half, pass, filter.freq, rate);
                s.extend(butterworth_sections(half, pass, filter.freq, rate));
                let sign = if pass == Pass::High && half % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                (s, sign)
            }
        };
        Self {
            sections,
            sign,
            all: None,
            pass,
        }
    }

    /// Which side this is.
    pub fn pass(&self) -> Pass {
        self.pass
    }

    /// Number of biquad/first-order sections (both branches for `Pass::All`).
    pub fn sections(&self) -> usize {
        self.sections.len() + self.all.as_ref().map_or(0, |(s, _)| s.len())
    }

    /// `true` when this side is polarity-inverted (odd-order Linkwitz–Riley high-pass).
    pub fn inverted(&self) -> bool {
        self.sign < 0.0
    }

    /// Filter one sample.
    #[inline]
    pub fn tick(&mut self, x: f64) -> f64 {
        let mut v = x;
        for s in &mut self.sections {
            v = s.tick(v);
        }
        let mut y = v * self.sign;
        if let Some((hp, sign)) = &mut self.all {
            let mut w = x;
            for s in hp.iter_mut() {
                w = s.tick(w);
            }
            y += w * *sign;
        }
        y
    }

    /// Filter a mono buffer in place.
    pub fn process(&mut self, buf: &mut [f64]) {
        for v in buf {
            *v = self.tick(*v);
        }
    }

    /// Complex frequency response at `freq` Hz.
    pub fn response(&self, freq: f64, rate: f64) -> Complex<f64> {
        let branch = |sections: &[Section], sign: f64| {
            sections
                .iter()
                .map(|s| s.response(freq, rate))
                .fold(Complex::new(sign, 0.0), |acc, r| acc * r)
        };
        let mut r = branch(&self.sections, self.sign);
        if let Some((hp, sign)) = &self.all {
            r += branch(hp, *sign);
        }
        r
    }

    /// Magnitude response in dB at `freq` Hz.
    pub fn response_db(&self, freq: f64, rate: f64) -> f64 {
        20.0 * self.response(freq, rate).norm().log10()
    }

    /// Clear the filter memory.
    pub fn reset(&mut self) {
        let hp = self.all.iter_mut().flat_map(|(s, _)| s.iter_mut());
        for s in self.sections.iter_mut().chain(hp) {
            s.z1 = 0.0;
            s.z2 = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum_db(filter: CrossoverFilter, freq: f64, rate: f64) -> f64 {
        let lp = IirCrossover::new(&filter, Pass::Low, rate);
        let hp = IirCrossover::new(&filter, Pass::High, rate);
        20.0 * (lp.response(freq, rate) + hp.response(freq, rate))
            .norm()
            .log10()
    }

    #[test]
    fn linkwitz_riley_pairs_sum_flat_at_every_slope() {
        let rate = 48000.0;
        for slope in [12, 24, 36, 48] {
            let f = CrossoverFilter::new(CrossoverKind::LinkwitzRiley, 2000.0, slope);
            f.validate(rate).unwrap();
            for probe in [20.0, 200.0, 1000.0, 2000.0, 4000.0, 10000.0, 20000.0] {
                let s = sum_db(f, probe, rate);
                assert!(s.abs() < 0.01, "LR{slope} sum at {probe} Hz = {s} dB");
            }
            // Each side is −6 dB at the crossover point.
            let lp = IirCrossover::new(&f, Pass::Low, rate);
            assert!((lp.response_db(2000.0, rate) + 6.02).abs() < 0.05);
            // The all-pass really is one.
            let mut all = IirCrossover::new(&f, Pass::All, rate);
            for probe in [20.0, 2000.0, 20000.0] {
                assert!(all.response_db(probe, rate).abs() < 1e-9);
            }
            assert_eq!(all.sections(), 2 * lp.sections());
            let y: Vec<f64> = (0..200)
                .map(|i| all.tick(if i == 0 { 1.0 } else { 0.0 }))
                .collect();
            assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-6);
            all.reset();
        }
    }

    #[test]
    fn butterworth_has_the_right_slope_and_corner() {
        // High rate so the bilinear warping at 8 kHz is negligible.
        let rate = 384000.0;
        for slope in [6u32, 12, 18, 24, 48] {
            let f = CrossoverFilter::new(CrossoverKind::Butterworth, 1000.0, slope);
            let lp = IirCrossover::new(&f, Pass::Low, rate);
            assert!(
                (lp.response_db(1000.0, rate) + 3.01).abs() < 0.05,
                "{slope}"
            );
            let a = lp.response_db(4000.0, rate);
            let b = lp.response_db(8000.0, rate);
            assert!(
                ((a - b) - slope as f64).abs() < 0.3,
                "BW{slope}: {} dB/oct",
                a - b
            );
            let hp = IirCrossover::new(&f, Pass::High, rate);
            assert!((hp.response_db(1000.0, rate) + 3.01).abs() < 0.05);
            // 20 Hz is 5.64 octaves below the corner.
            assert!(hp.response_db(20.0, rate) < -(slope as f64 * 5.64) + 0.5);
        }
        // BW2 low + high: the two sides are 180° apart at the crossover, so the in-phase sum
        // nulls there and the inverted one peaks +3 dB. The reason Linkwitz–Riley exists.
        let f = CrossoverFilter::new(CrossoverKind::Butterworth, 1000.0, 12);
        assert!(sum_db(f, 1000.0, rate) < -40.0);
        let lp = IirCrossover::new(&f, Pass::Low, rate);
        let hp = IirCrossover::new(&f, Pass::High, rate);
        let diff = (lp.response(1000.0, rate) - hp.response(1000.0, rate)).norm();
        assert!((20.0 * diff.log10() - 3.01).abs() < 0.05);
    }

    #[test]
    fn time_domain_matches_the_analytic_response() {
        let rate = 44100.0;
        let f = CrossoverFilter::lr24(500.0);
        let mut lp = IirCrossover::new(&f, Pass::Low, rate);
        let n = 44100;
        let probe = 500.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * probe * i as f64 / rate).sin())
            .collect();
        let y: Vec<f64> = x.iter().map(|&v| lp.tick(v)).collect();
        let tail = &y[n / 2..];
        let rms = (tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64).sqrt();
        let db = 20.0 * (rms * 2f64.sqrt()).log10();
        assert!((db - lp.response_db(probe, rate)).abs() < 0.02, "{db}");
        lp.reset();
        assert_eq!(lp.tick(0.0), 0.0);
    }

    #[test]
    fn fir_bands_are_exactly_complementary() {
        let rate = 48000.0;
        let taps = fir_taps_for(300.0, rate).max(fir_taps_for(3000.0, rate));
        assert_eq!(taps % 2, 1);
        let low = fir_band(taps, None, Some(300.0), rate);
        let mid = fir_band(taps, Some(300.0), Some(3000.0), rate);
        let high = fir_band(taps, Some(3000.0), None, rate);
        let delta = fir_delta(taps);
        for i in 0..taps {
            let s = low[i] + mid[i] + high[i];
            assert!((s - delta[i]).abs() < 1e-12, "tap {i}: {s}");
        }
        // The low-pass really is a low-pass: ≥ 90 dB down an octave above.
        let mag = crate::fft::magnitude_db(&low, 1 << 16);
        let bin = |f: f64| (f / rate * (1 << 16) as f64).round() as usize;
        assert!(mag[bin(600.0)] < -90.0, "{}", mag[bin(600.0)]);
        assert!(mag[bin(100.0)].abs() < 0.01);
    }

    #[test]
    fn parsing_labels_and_validation() {
        assert_eq!(
            CrossoverKind::parse("lr"),
            Some(CrossoverKind::LinkwitzRiley)
        );
        assert_eq!(
            CrossoverKind::parse("FIR"),
            Some(CrossoverKind::LinearPhase)
        );
        assert_eq!(CrossoverKind::parse("bessel"), None);
        assert_eq!(CrossoverFilter::lr24(2000.0).label(), "LR24 @ 2 kHz");
        assert_eq!(
            CrossoverFilter::new(CrossoverKind::Butterworth, 80.0, 12).label(),
            "BW12 @ 80 Hz"
        );
        assert_eq!(
            CrossoverFilter::linear_phase(2500.0).label(),
            "FIR @ 2.5 kHz"
        );
        assert!(
            CrossoverFilter::new(CrossoverKind::LinkwitzRiley, 100.0, 18)
                .validate(48000.0)
                .is_err()
        );
        assert!(CrossoverFilter::lr24(30000.0).validate(48000.0).is_err());
        assert!(CrossoverFilter::lr24(80.0).validate(48000.0).is_ok());
        let json = serde_json::to_string(&CrossoverFilter::lr24(80.0)).unwrap();
        assert!(json.contains("\"linkwitz-riley\""));
        let back: CrossoverFilter = serde_json::from_str(r#"{"freq": 120.0}"#).unwrap();
        assert_eq!(back, CrossoverFilter::lr24(120.0));
    }
}
