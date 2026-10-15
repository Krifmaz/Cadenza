//! The filter designer: upsampling filters made to measure from a transition width, a
//! stopband depth, a cutoff, a window and a phase, and the response plots the web UI shows
//! for any filter (designed or from the catalogue).
//!
//! A design has an id like the catalogue's, so it travels through the same `filter`
//! setting: `custom-kaiser-l-c100-t5-s150` is a Kaiser linear-phase filter with its −6 dB
//! point at 100 % of the source Nyquist, a transition band 5 % of the source Nyquist wide
//! and a 150 dB stopband. The tap count follows from those at every ratio, so the filter
//! has the same shape in Hz whatever the output rate.

use crate::fft::{next_pow2, RealFft};
use crate::fir::{
    gaussian_alpha, FilterInfo, FilterKind, FirSpec, Phase, Window, COSINE3_ATTENUATION_DB,
    MAX_TAPS, MIN_TAPS,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Id prefix of designed filters.
pub const PREFIX: &str = "custom-";

/// Every window the designer offers.
pub const WINDOWS: [Window; 4] = [
    Window::Kaiser,
    Window::DolphChebyshev,
    Window::Gaussian,
    Window::Cosine3,
];

/// A filter made to measure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilterDesign {
    /// Window shape.
    pub window: Window,
    /// Linear or minimum phase.
    pub phase: Phase,
    /// The −6 dB point as a fraction of the source Nyquist (0.8 … 1.0; below 1 is
    /// "apodizing").
    pub cutoff: f64,
    /// Width of the transition band as a fraction of the source Nyquist (0.0001 … 0.5).
    pub transition: f64,
    /// Stopband attenuation in dB (60 … 250; the 3-term cosine window always gives 71.5).
    pub stopband_db: f64,
}

impl Default for FilterDesign {
    fn default() -> Self {
        Self {
            window: Window::Kaiser,
            phase: Phase::Linear,
            cutoff: 1.0,
            transition: 0.05,
            stopband_db: 150.0,
        }
    }
}

/// Shortest number form for ids: `5`, `0.5`, `99.25`.
fn num(x: f64) -> String {
    let s = format!("{:.4}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

impl FilterDesign {
    /// Check the ranges; returns the reason when one is off.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.8..=1.0).contains(&self.cutoff) {
            return Err("cutoff must be 80–100 % of the source Nyquist".into());
        }
        if !(0.0001..=0.5).contains(&self.transition) {
            return Err("transition must be 0.01–50 % of the source Nyquist".into());
        }
        if !(60.0..=250.0).contains(&self.stopband_db) {
            return Err(
                "stopband must be 60–250 dB (deeper is below what 64-bit coefficients can hold)"
                    .into(),
            );
        }
        Ok(())
    }

    /// The id, e.g. `custom-kaiser-l-c100-t5-s150`.
    pub fn id(&self) -> String {
        format!(
            "{PREFIX}{}-{}-c{}-t{}-s{}",
            self.window.name(),
            if self.phase == Phase::Linear {
                "l"
            } else {
                "m"
            },
            num(self.cutoff * 100.0),
            num(self.transition * 100.0),
            num(self.stopband_db)
        )
    }

    /// Parse an id made by [`FilterDesign::id`] (ranges checked).
    pub fn parse(id: &str) -> Option<Self> {
        let id = id.trim().to_ascii_lowercase();
        let rest = id.strip_prefix(PREFIX)?;
        let parts: Vec<&str> = rest.split('-').collect();
        if parts.len() < 5 {
            return None;
        }
        let n = parts.len();
        let field = |s: &str, p: char| s.strip_prefix(p)?.parse::<f64>().ok();
        let window_name = parts[..n - 4].join("-");
        let window = WINDOWS.into_iter().find(|w| w.name() == window_name)?;
        let phase = match parts[n - 4] {
            "l" => Phase::Linear,
            "m" => Phase::Minimum,
            _ => return None,
        };
        let d = Self {
            window,
            phase,
            cutoff: field(parts[n - 3], 'c')? / 100.0,
            transition: field(parts[n - 2], 't')? / 100.0,
            stopband_db: field(parts[n - 1], 's')?,
        };
        d.validate().ok()?;
        Some(d)
    }

    /// Human readable label.
    pub fn label(&self) -> String {
        format!(
            "custom {} · {} phase · {} % transition · {} dB{}",
            self.window.name(),
            self.phase.name(),
            num(self.transition * 100.0),
            num(self.effective_stopband_db()),
            if self.cutoff < 1.0 {
                format!(" · cutoff {} %", num(self.cutoff * 100.0))
            } else {
                String::new()
            }
        )
    }

    fn effective_stopband_db(&self) -> f64 {
        match self.window {
            Window::Cosine3 => COSINE3_ATTENUATION_DB,
            _ => self.stopband_db,
        }
    }

    /// The tap count giving a transition `tw` (normalised to the filter rate) with this
    /// window and stopband; inverts [`FirSpec::transition_width`].
    fn taps_for(&self, tw: f64) -> usize {
        let a = self.effective_stopband_db();
        let n = match self.window {
            // Kaiser's estimate is optimistic for deep stopbands (measured: up to 17 dB short
            // at 250 dB right at the edge); lengthen by up to 30 % (at 250 dB) so the stopband is met.
            Window::Kaiser => {
                (a - 8.0) / (2.285 * 2.0 * PI * tw) * (1.0 + (a - 100.0).max(0.0) / 500.0)
            }
            Window::DolphChebyshev => {
                let r: f64 = 10f64.powf(a / 20.0);
                2.0 * (r + (r * r - 1.0).sqrt()).ln() / (PI * tw)
            }
            Window::Gaussian => {
                2.0 * gaussian_alpha(a) * (a * std::f64::consts::LN_10 / 10.0).sqrt() / (PI * tw)
            }
            Window::Cosine3 => 6.0 / tw,
        };
        (n.ceil().min(MAX_TAPS as f64) as usize + 1).clamp(MIN_TAPS, MAX_TAPS)
    }

    /// The design for a stage converting at `l/m`, where `band` is the fraction of the
    /// stage's input Nyquist that carries signal (1 for the first stage; the second stage
    /// of a cascade only has to keep the source band, so it is never apodizing and its
    /// transition sits above that band).
    pub fn spec_for_stage(&self, l: usize, m: usize, band: f64) -> FirSpec {
        let nyquist = 0.5 / l.max(m).max(1) as f64;
        let (cutoff, transition) = if band < 1.0 {
            (1.0, self.transition.min(2.0 * (1.0 - band) * 0.9))
        } else {
            (self.cutoff, self.transition)
        };
        FirSpec::with_window(
            self.taps_for(transition * nyquist),
            cutoff * nyquist,
            self.stopband_db,
            self.phase,
            self.window,
        )
    }

    /// The design for a single-step `l/m` conversion.
    pub fn spec_for_ratio(&self, l: usize, m: usize) -> FirSpec {
        self.spec_for_stage(l, m, 1.0)
    }

    /// Describe this design for a conversion `in_rate → out_rate`, like the catalogue.
    pub fn describe(&self, in_rate: u32, out_rate: u32) -> FilterInfo {
        let (spec, filter_rate) = Filter::Designed(self.clone()).primary(in_rate, out_rate, 0.0);
        let g = gcd(in_rate.max(1), out_rate.max(1));
        let m = (in_rate / g).max(1) as usize;
        let passband = in_rate.min(out_rate) as f64 / 2.0;
        FilterInfo {
            id: self.id(),
            label: self.label(),
            family: "custom".into(),
            description: "Designed to measure: transition, stopband, cutoff, window and phase"
                .into(),
            phase: self.phase.name().into(),
            apodizing: self.cutoff < 1.0,
            taps: spec.taps,
            taps_at_rate: spec.taps,
            window: spec.window.name().into(),
            in_rate,
            out_rate,
            cutoff_hz: spec.cutoff * filter_rate,
            // What was asked for (and is met, see the tests), not Kaiser's estimate.
            transition_hz: self.transition * passband,
            transition_percent: 100.0 * self.transition,
            stopband_db: spec.attenuation_db,
            latency_samples: spec.group_delay() / m,
            latency_seconds: spec.group_delay() as f64 / filter_rate,
            bytes: spec.taps * std::mem::size_of::<f64>(),
        }
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Is `id` a filter Cadenza can run (catalogue or designed)?
pub fn is_filter_id(id: &str) -> bool {
    Filter::parse(id).is_some()
}

/// Either kind of filter.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    /// A catalogue filter (`sinc-l-64k`, …).
    Catalog(FilterKind),
    /// A designed filter (`custom-…`).
    Designed(FilterDesign),
}

impl Filter {
    /// Parse a catalogue or designed id.
    pub fn parse(id: &str) -> Option<Self> {
        FilterKind::parse(id)
            .map(Filter::Catalog)
            .or_else(|| FilterDesign::parse(id).map(Filter::Designed))
    }

    /// The id.
    pub fn id(&self) -> String {
        match self {
            Filter::Catalog(k) => k.id(),
            Filter::Designed(d) => d.id(),
        }
    }

    /// Describe for a conversion (`attenuation_db` applies to catalogue filters).
    pub fn describe(&self, in_rate: u32, out_rate: u32, attenuation_db: f64) -> FilterInfo {
        match self {
            Filter::Catalog(k) => k.describe(in_rate, out_rate, attenuation_db),
            Filter::Designed(d) => d.describe(in_rate, out_rate),
        }
    }

    /// Where the passband ends and the stopband begins (Hz) for `in_rate → out_rate`: what
    /// a design was asked for, or what a catalogue filter is specified to.
    pub fn edges(&self, in_rate: u32, out_rate: u32, attenuation_db: f64) -> (f64, f64) {
        match self {
            Filter::Designed(d) => {
                let nyq = in_rate.min(out_rate) as f64 / 2.0;
                (
                    (d.cutoff - d.transition / 2.0) * nyq,
                    (d.cutoff + d.transition / 2.0) * nyq,
                )
            }
            Filter::Catalog(_) => {
                let (spec, rate) = self.primary(in_rate, out_rate, attenuation_db);
                (spec.passband_edge() * rate, spec.stopband_edge() * rate)
            }
        }
    }

    /// The filter that shapes the passband for `in_rate → out_rate` (the first stage's, as
    /// the resampler runs it) and the rate it runs at.
    pub fn primary(&self, in_rate: u32, out_rate: u32, attenuation_db: f64) -> (FirSpec, f64) {
        let spec = |l: usize, m: usize, band: f64| match self {
            Filter::Catalog(k) => {
                let k = if band < 1.0 {
                    FilterKind {
                        apodizing: false,
                        ..*k
                    }
                } else {
                    *k
                };
                k.spec_for_ratio(l, m, attenuation_db)
            }
            Filter::Designed(d) => d.spec_for_stage(l, m, band),
        };
        let (a, b, s) = crate::resampler::plan(in_rate, out_rate, &spec)
            .into_iter()
            .next()
            .expect("a plan has a stage");
        let g = gcd(a, b);
        (s, a as f64 * (b / g) as f64)
    }
}

/// Longest filter [`response`] plots (its FFT needs 4 × the next power of two in points).
pub const MAX_PLOT_TAPS: usize = (1 << 21) + 1;

/// What the web UI plots for a filter.
#[derive(Clone, Debug, Serialize)]
pub struct Response {
    /// The rate the filter runs at (Hz).
    pub filter_rate: f64,
    /// Source Nyquist (Hz).
    pub nyquist: f64,
    /// Taps.
    pub taps: usize,
    /// `[Hz, dB]` from 0 to twice the source Nyquist, peak per point (normalised to the DC
    /// gain).
    pub transition: Vec<[f64; 2]>,
    /// `[Hz, dB]` from 0 to the filter's Nyquist, peak per point.
    pub full: Vec<[f64; 2]>,
    /// `[Hz, dB]` across the passband (0 to the passband edge), the deviation from flat.
    pub passband: Vec<[f64; 2]>,
    /// `[ms, value]` around the impulse's peak (normalised to the peak).
    pub impulse: Vec<[f64; 2]>,
    /// Measured passband ripple, peak to peak (dB).
    pub ripple_db: f64,
    /// Measured stopband: the highest level above the stopband edge (dB, negative).
    pub stopband_db: f64,
    /// Passband edge (Hz).
    pub passband_hz: f64,
    /// Stopband edge (Hz).
    pub stopband_hz: f64,
}

/// Compute the plots for coefficients `h` running at `filter_rate`, where the source
/// Nyquist is `nyquist` Hz; ripple and stopband are measured below `passband_hz` and above
/// `stopband_hz` (see [`Filter::edges`]).
pub fn response(
    h: &[f64],
    filter_rate: f64,
    nyquist: f64,
    passband_hz: f64,
    stopband_hz: f64,
) -> Response {
    let n = (next_pow2(h.len()) * 4).max(1 << 16);
    let mut fft = RealFft::new(n);
    let mut buf = vec![0.0; n];
    buf[..h.len()].copy_from_slice(h);
    let mut spec_c = fft.spectrum();
    fft.forward(&mut buf, &mut spec_c);
    drop(buf);
    let dc = spec_c[0].norm().max(1e-300);
    let db: Vec<f64> = spec_c
        .iter()
        .map(|c| (20.0 * (c.norm() / dc).max(1e-20).log10()).max(-400.0))
        .collect();
    drop(spec_c);
    let bin_hz = filter_rate / n as f64;
    let bin = |hz: f64| ((hz / bin_hz).round() as usize).min(db.len() - 1);
    // Peak per plotted point, so a narrow sidelobe never falls between points.
    let peak = |from: f64, to: f64, points: usize| -> Vec<[f64; 2]> {
        (0..points)
            .map(|i| {
                let a = from + (to - from) * i as f64 / points as f64;
                let b = from + (to - from) * (i + 1) as f64 / points as f64;
                let (ia, ib) = (bin(a), bin(b).max(bin(a) + 1).min(db.len()));
                let v = db[ia..ib].iter().cloned().fold(f64::MIN, f64::max);
                [round(a, 1), round(v, 3)]
            })
            .collect()
    };
    let pass = &db[..=bin(passband_hz).max(1)];
    let (lo, hi) = pass
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let stop = db[((stopband_hz / bin_hz).ceil() as usize).min(db.len() - 1)..]
        .iter()
        .cloned()
        .fold(f64::MIN, f64::max);
    let passband: Vec<[f64; 2]> = (0..400)
        .map(|i| {
            let f = passband_hz * i as f64 / 400.0;
            [round(f, 1), db[bin(f)]]
        })
        .collect();
    // The impulse: from 16 source periods before the peak to 48 after, at most 2000 points.
    let l = (filter_rate / (2.0 * nyquist)).round().max(1.0) as usize;
    let p = h.iter().enumerate().fold(
        (0, 0.0),
        |(bi, bv), (i, &v)| {
            if v.abs() > bv {
                (i, v.abs())
            } else {
                (bi, bv)
            }
        },
    );
    let (from, to) = (p.0.saturating_sub(16 * l), (p.0 + 48 * l).min(h.len()));
    let step = ((to - from) / 2000).max(1);
    let norm = if p.1 > 0.0 { h[p.0] } else { 1.0 };
    let impulse = (from..to)
        .step_by(step)
        .map(|i| {
            [
                round((i as f64 - p.0 as f64) / filter_rate * 1000.0, 5),
                h[i] / norm,
            ]
        })
        .collect();
    Response {
        filter_rate,
        nyquist,
        taps: h.len(),
        transition: peak(0.0, (2.0 * nyquist).min(filter_rate / 2.0), 800),
        full: peak(0.0, filter_rate / 2.0, 800),
        passband,
        impulse,
        ripple_db: hi - lo,
        stopband_db: stop,
        passband_hz,
        stopband_hz,
    }
}

fn round(x: f64, digits: i32) -> f64 {
    let s = 10f64.powi(digits);
    (x * s).round() / s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resampler::Resampler;
    use crate::CoefficientCache;

    #[test]
    fn ids_round_trip_and_reject_nonsense() {
        let d = FilterDesign {
            window: Window::DolphChebyshev,
            phase: Phase::Minimum,
            cutoff: 0.95,
            transition: 0.025,
            stopband_db: 180.0,
        };
        assert_eq!(d.id(), "custom-dolph-chebyshev-m-c95-t2.5-s180");
        assert_eq!(FilterDesign::parse(&d.id()), Some(d));
        assert_eq!(
            FilterDesign::parse("custom-kaiser-l-c100-t5-s150"),
            Some(FilterDesign::default())
        );
        for bad in [
            "custom-kaiser-l-c100-t5",
            "custom-hann-l-c100-t5-s150",
            "custom-kaiser-x-c100-t5-s150",
            "custom-kaiser-l-c120-t5-s150",
            "custom-kaiser-l-c100-t0-s150",
            "custom-kaiser-l-c100-t5-s20",
            "custom-kaiser-l-c100-t5-s300",
            "sinc-l-64k",
        ] {
            assert_eq!(FilterDesign::parse(bad), None, "{bad}");
        }
        assert!(is_filter_id("sinc-l-64k"));
        assert!(is_filter_id("custom-gaussian-l-c100-t10-s160"));
        assert!(!is_filter_id("custom-gaussian"));
    }

    /// The designed filter meets what was asked: transition width, stopband, ripple, at
    /// two ratios, for every window and both phases.
    #[test]
    fn designs_meet_their_specification() {
        let cache = CoefficientCache::disabled();
        for (window, phase) in WINDOWS
            .into_iter()
            .flat_map(|w| [(w, Phase::Linear), (w, Phase::Minimum)])
        {
            for (in_rate, out_rate) in [(44_100, 176_400), (48_000, 384_000)] {
                let d = FilterDesign {
                    window,
                    phase,
                    transition: 0.1,
                    stopband_db: 120.0,
                    ..Default::default()
                };
                let (spec, rate) = Filter::Designed(d.clone()).primary(in_rate, out_rate, 0.0);
                let nyq = in_rate as f64 / 2.0;
                // The transition asked for, within a few percent.
                let tw = spec.transition_width() * rate;
                assert!(
                    (tw / (0.1 * nyq) - 1.0).abs() < 0.05,
                    "{window:?} {in_rate}: {tw} Hz"
                );
                let h = cache.get_or_design(&spec);
                let f = Filter::Designed(d.clone());
                let (p, st) = f.edges(in_rate, out_rate, 0.0);
                let r = response(&h, rate, nyq, p, st);
                // Within 1 dB of what was asked, measured from the asked-for edge.
                let want = if window == Window::Cosine3 {
                    70.0
                } else {
                    119.0
                };
                assert!(
                    r.stopband_db < -want,
                    "{window:?} {in_rate}: stopband {:.1}",
                    r.stopband_db
                );
                assert!(r.ripple_db < 0.01, "{window:?}: ripple {}", r.ripple_db);
                // −6 dB at the cutoff.
                let at = r
                    .transition
                    .iter()
                    .min_by(|a, b| (a[0] - nyq).abs().partial_cmp(&(b[0] - nyq).abs()).unwrap())
                    .unwrap();
                assert!((at[1] + 6.0).abs() < 1.0, "{window:?}: {at:?}");
            }
        }
    }

    /// A narrower transition costs proportionally more taps, and the shape in Hz is the
    /// same at every ratio.
    #[test]
    fn taps_scale_with_the_transition_and_the_ratio() {
        let a = FilterDesign::default().describe(44_100, 352_800);
        let b = FilterDesign {
            transition: 0.025,
            ..Default::default()
        }
        .describe(44_100, 352_800);
        let ratio = b.taps as f64 / a.taps as f64;
        assert!((ratio - 2.0).abs() < 0.01, "{ratio}");
        let c = FilterDesign::default().describe(44_100, 705_600);
        assert!((c.transition_hz - a.transition_hz).abs() / a.transition_hz < 0.01);
        assert!((c.taps as f64 / a.taps as f64 - 2.0).abs() < 0.01);
    }

    /// A designed filter runs in the resampler, through a cascade too, and keeps a tone.
    #[test]
    fn a_designed_filter_resamples() {
        let cache = CoefficientCache::disabled();
        let d = FilterDesign {
            transition: 0.1,
            stopband_db: 140.0,
            phase: Phase::Minimum,
            ..Default::default()
        };
        for (a, b) in [(44_100, 176_400), (44_100, 192_000)] {
            let mut r = Resampler::with_design(a, b, 1, &d, &cache, None).unwrap();
            let x: Vec<f64> = (0..a as usize)
                .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f64 / a as f64).sin())
                .collect();
            let mut y = r.push(&x);
            y.extend(r.flush());
            let tail = &y[y.len() / 2..];
            let peak = tail.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!((peak - 0.5).abs() < 0.01, "{a}->{b}: {peak}");
        }
    }

    #[test]
    fn catalogue_filters_plot_too() {
        let f = Filter::parse("sinc-l-16k").unwrap();
        let (spec, rate) = f.primary(44_100, 176_400, 150.0);
        let h = CoefficientCache::disabled().get_or_design(&spec);
        let (p, st) = f.edges(44_100, 176_400, 150.0);
        let r = response(&h, rate, 22_050.0, p, st);
        assert_eq!(r.taps, spec.taps);
        assert!(r.stopband_db < -140.0, "{}", r.stopband_db);
        assert_eq!(r.transition.len(), 800);
        assert!(r.impulse.iter().any(|p| p[0] == 0.0 && p[1] == 1.0));
    }
}
