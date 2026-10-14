//! Long FIR low-pass design and the named filter catalogue.
//!
//! Families (the `<family>-<phase>-<len>` id scheme, e.g. `sinc-l-64k`, `closed-form-m-1m`,
//! `poly-sinc-gauss-la-long`, `sinc-xtr-l-16m`, `sinc-mx-l-1m`, `poly-sinc-ext3-l-8k`):
//!
//! * `sinc` – Kaiser windowed sinc; the attenuation picks the window shape.
//! * `closed-form` – sinc with an analytically chosen Dolph–Chebyshev window: the stopband is
//!   exactly equiripple at the requested attenuation, so the transition is as narrow as a
//!   window design can make it for that length.
//! * `poly-sinc-gauss` – Gaussian windowed sinc (`short`, `long`, `xla`): no sidelobes at all,
//!   a smooth, monotonic transition and the least ringing of the catalogue.
//! * `sinc-xtr` – extremely long Kaiser designs (2^23 and 2^24 taps, ≥ 200 dB stopband).
//! * `sinc-mx` – a 1M-tap design *matched to 16×*: at any ratio the length is scaled so the
//!   transition band in Hz is the one a 1M-tap filter has at ×16 (like HQPlayer's sinc-M).
//! * `poly-sinc-ext3` – a short sinc with Nuttall's optimised 3-term cosine window.
//!
//! Every family comes in linear and minimum phase, plain and apodizing. Everything is
//! computed in `f64` (also on disk, see [`crate::cache`]); the minimum-phase conversion is
//! the real-cepstrum method of [`minimum_phase`].

use crate::fft::{next_pow2, RealFft};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Shortest filter the designer accepts.
pub const MIN_TAPS: usize = 1 << 8;
/// Longest filter the designer accepts (2^24 taps, plus one so the length is odd).
///
/// A 16M-tap design is 128 MB of `f64` on disk and in memory; its minimum-phase
/// conversion needs a 32M-point FFT (about 1.5 GB of scratch while designing).
pub const MAX_TAPS: usize = (1 << 24) + 1;

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

/// The window shape a sinc design is multiplied with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Window {
    /// Kaiser (parameter from the attenuation; Kaiser's empirical transition formula).
    #[default]
    Kaiser,
    /// Dolph–Chebyshev: equiripple sidelobes exactly at the attenuation (closed-form).
    DolphChebyshev,
    /// Gaussian, truncated at the attenuation level: sidelobe-free, smoothest transition.
    Gaussian,
    /// Nuttall's minimum-sidelobe 3-term cosine window (−71.5 dB sidelobes).
    Cosine3,
}

impl Window {
    /// Lower-case name used in cache keys and reports.
    pub fn name(self) -> &'static str {
        match self {
            Window::Kaiser => "kaiser",
            Window::DolphChebyshev => "dolph-chebyshev",
            Window::Gaussian => "gaussian",
            Window::Cosine3 => "cosine3",
        }
    }
}

/// Peak sidelobe level of [`Window::Cosine3`] in dB (the stopband it can deliver).
pub const COSINE3_ATTENUATION_DB: f64 = 71.48;

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
    /// Design stopband attenuation in dB (also bounds the passband ripple).
    pub attenuation_db: f64,
    /// Linear or minimum phase.
    pub phase: Phase,
    /// Window shape.
    #[serde(default)]
    pub window: Window,
}

impl FirSpec {
    /// A Kaiser design from a tap count: the transition band follows from `taps` and `attenuation_db`.
    pub fn from_taps(taps: usize, cutoff: f64, attenuation_db: f64, phase: Phase) -> Self {
        Self::with_window(taps, cutoff, attenuation_db, phase, Window::Kaiser)
    }

    /// A design with an explicit window.
    pub fn with_window(
        taps: usize,
        cutoff: f64,
        attenuation_db: f64,
        phase: Phase,
        window: Window,
    ) -> Self {
        let taps = (taps.clamp(MIN_TAPS, MAX_TAPS)) | 1;
        assert!(
            cutoff > 0.0 && cutoff < 0.5,
            "cutoff must be in (0, 0.5), got {cutoff}"
        );
        let attenuation_db = match window {
            // The 3-term window has a fixed sidelobe level; the parameter is informational.
            Window::Cosine3 => COSINE3_ATTENUATION_DB,
            _ => attenuation_db.clamp(30.0, 300.0),
        };
        Self {
            taps,
            cutoff,
            attenuation_db,
            phase,
            window,
        }
    }

    /// A Kaiser design from a transition width (normalised): the tap count follows from the
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

    /// Width of the transition band (normalised frequency) this design achieves: the
    /// main-lobe width of its window.
    pub fn transition_width(&self) -> f64 {
        let n = self.taps as f64 - 1.0;
        match self.window {
            Window::Kaiser => (self.attenuation_db - 8.0) / (2.285 * 2.0 * PI * n),
            // First null of the Dolph–Chebyshev main lobe ≈ acosh(r) / (π N); the sinc's
            // transition spans the whole main lobe.
            Window::DolphChebyshev => {
                let r = 10f64.powf(self.attenuation_db / 20.0);
                2.0 * acosh(r) / (PI * n)
            }
            // The Gaussian's transform is Gaussian; it is `A` dB down at
            // ω = (2α/N)·√(A·ln10/10), and the transition spans ±ω.
            Window::Gaussian => {
                let alpha = gaussian_alpha(self.attenuation_db);
                let omega = (2.0 * alpha / n)
                    * (self.attenuation_db * std::f64::consts::LN_10 / 10.0).sqrt();
                omega / PI
            }
            // Main lobe of a 3-term cosine window: nulls at ±3/N.
            Window::Cosine3 => 6.0 / n,
        }
    }

    /// Where the passband ends (normalised): the ripple stays within the design bound below this.
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
        let family = match self.window {
            Window::Kaiser => "sinc".to_string(),
            w => format!("sinc-{}", w.name()),
        };
        format!(
            "{family}-{}-{}-c{}-a{}",
            self.phase.name(),
            self.taps,
            format!("{:.12e}", self.cutoff).replace(['+', '-'], "_"),
            self.attenuation_db as i64
        )
    }

    /// The window of this design.
    pub fn window_values(&self) -> Vec<f64> {
        match self.window {
            Window::Kaiser => kaiser_window(self.taps, self.kaiser_beta()),
            Window::DolphChebyshev => dolph_chebyshev_window(self.taps, self.attenuation_db),
            Window::Gaussian => gaussian_window(self.taps, gaussian_alpha(self.attenuation_db)),
            Window::Cosine3 => cosine3_window(self.taps),
        }
    }

    /// Compute the coefficients (unity DC gain). Not cached; see [`crate::cache`].
    pub fn design(&self) -> Vec<f64> {
        let linear = windowed_sinc_with(self.taps, self.cutoff, &self.window_values());
        match self.phase {
            Phase::Linear => linear,
            // Everything below the floor is treated as "zero" by the cepstrum: the true
            // nulls of the window design go to -inf dB and would otherwise dominate the log.
            Phase::Minimum => minimum_phase(&linear, self.attenuation_db + 60.0),
        }
    }
}

fn acosh(x: f64) -> f64 {
    if x < 1.0 {
        0.0
    } else {
        x.ln() + (1.0 - 1.0 / (x * x)).sqrt().ln_1p()
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

/// Gaussian window parameter `α` so the truncated edge sits `a` dB below the peak
/// (`exp(−α²/2) = 10^(−a/20)`).
pub fn gaussian_alpha(a: f64) -> f64 {
    (a * std::f64::consts::LN_10 / 10.0).sqrt()
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

/// Evaluate a symmetric window from a function of the normalised position `r ∈ [-1, 1]`
/// (computed in parallel for the long designs and mirrored).
fn symmetric_window(n: usize, f: impl Fn(f64) -> f64 + Sync) -> Vec<f64> {
    let m = (n as f64 - 1.0) / 2.0;
    let half: Vec<f64> = (0..n.div_ceil(2))
        .into_par_iter()
        .map(|i| f((i as f64 - m) / m.max(0.5)))
        .collect();
    let mut w = vec![0.0; n];
    for (i, v) in half.into_iter().enumerate() {
        w[i] = v;
        w[n - 1 - i] = v;
    }
    w
}

/// Kaiser window of `n` points.
pub fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    let denom = bessel_i0(beta);
    symmetric_window(n, |r| {
        let arg = (1.0 - r * r).max(0.0).sqrt();
        bessel_i0(beta * arg) / denom
    })
}

/// Gaussian window of `n` points: `exp(−½ (α r)²)` for `r ∈ [-1, 1]`.
pub fn gaussian_window(n: usize, alpha: f64) -> Vec<f64> {
    symmetric_window(n, |r| (-0.5 * (alpha * r) * (alpha * r)).exp())
}

/// Nuttall's minimum-sidelobe 3-term cosine window (`−71.48 dB` peak sidelobe).
pub fn cosine3_window(n: usize) -> Vec<f64> {
    const A: [f64; 3] = [0.424_380_1, 0.497_340_6, 0.078_279_3];
    symmetric_window(n, |r| {
        // r = -1 is the first sample (angle 0), r = 1 the last (angle 2π).
        let x = PI * (r + 1.0);
        A[0] - A[1] * x.cos() + A[2] * (2.0 * x).cos()
    })
}

/// Dolph–Chebyshev window of `n` points with equiripple sidelobes `attenuation_db` below
/// the main lobe, from its closed-form Chebyshev spectrum through an inverse DFT.
pub fn dolph_chebyshev_window(n: usize, attenuation_db: f64) -> Vec<f64> {
    let order = (n - 1) as f64;
    let r = 10f64.powf(attenuation_db / 20.0);
    // β = cosh(acosh(r) / (N − 1)); computed as 1 + x so the tiny excess over 1 for long
    // windows survives in f64.
    let t = acosh(r) / order;
    // β − 1 = 2 sinh²(t/2): for a 16M-tap window this is ~1e-12 and must not be formed as
    // `cosh(t) − 1`.
    let beta_minus_1 = 2.0 * (t / 2.0).sinh().powi(2);
    // T_{N−1}(β) = cosh((N−1)·acosh β) = cosh(acosh r) = r, exactly.

    // W(k) = T_{N−1}(β cos(πk/N)) / r, real and even. Near the main lobe the argument is
    // within 1e-12 of ±1, so `1 − |x|` is assembled from its two small parts:
    // |x| − 1 = (β − 1)·|cos θ| − 2 sin²(θ/2) (and the mirror image for cos θ < 0).
    let odd_order = (n - 1) % 2 == 1;
    let spectrum: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|k| {
            let theta = PI * k as f64 / n as f64;
            // Work with |cos θ| and restore the sign through T_M(−x) = (−1)^M T_M(x).
            let (c, negative) = {
                let c = theta.cos();
                (c.abs(), c < 0.0)
            };
            let half = if negative {
                (PI - theta) / 2.0
            } else {
                theta / 2.0
            };
            let excess = beta_minus_1 * c - 2.0 * half.sin().powi(2); // |x| − 1
            let v = if excess > 0.0 {
                // |x| > 1: cosh(M·acosh(1 + ε)), acosh(1+ε) = ln1p(ε + √(ε(2+ε))).
                let ac = (excess + (excess * (2.0 + excess)).sqrt()).ln_1p();
                (order * ac).cosh()
            } else {
                // |x| ≤ 1: cos(M·acos(1 − δ)), acos(1−δ) = 2·asin(√(δ/2)).
                let delta = -excess;
                let ac = 2.0 * (delta / 2.0).sqrt().min(1.0).asin();
                (order * ac).cos()
            };
            let v = if negative && odd_order { -v } else { v };
            v / r
        })
        .collect();

    // Inverse DFT of a real, even sequence is real and even: w[m] = Σ_k W(k) cos(2πkm/N).
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(n);
    let mut buf: Vec<Complex<f64>> = spectrum.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft.process(&mut buf);
    // The result is centred on index 0; rotate so the peak sits in the middle.
    let shift = n / 2;
    let mut w: Vec<f64> = (0..n).map(|i| buf[(i + n - shift) % n].re).collect();
    let peak = w.iter().cloned().fold(0.0, f64::max).max(1e-300);
    for v in &mut w {
        *v /= peak;
    }
    // Enforce exact symmetry (rounding in the FFT breaks it at the 1e-16 level).
    for i in 0..n / 2 {
        let a = 0.5 * (w[i] + w[n - 1 - i]);
        w[i] = a;
        w[n - 1 - i] = a;
    }
    w
}

/// Linear-phase Kaiser windowed sinc low-pass with unity DC gain.
pub fn windowed_sinc(taps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    windowed_sinc_with(taps, cutoff, &kaiser_window(taps, beta))
}

/// Linear-phase windowed sinc low-pass with unity DC gain and an explicit window.
pub fn windowed_sinc_with(taps: usize, cutoff: f64, window: &[f64]) -> Vec<f64> {
    assert_eq!(window.len(), taps);
    let m = (taps as f64 - 1.0) / 2.0;
    let half: Vec<f64> = (0..taps.div_ceil(2))
        .into_par_iter()
        .map(|i| {
            let x = i as f64 - m;
            let s = if x == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * x).sin() / (PI * x)
            };
            s * window[i]
        })
        .collect();
    let mut h = vec![0.0; taps];
    for (i, v) in half.into_iter().enumerate() {
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
    // Memory for a 4M-tap design at 4× padding would be several hundred MB, so pad less
    // there; the 8M/16M `sinc-xtr` designs get the bare power of two (still ≥ 2× for 2^k+1).
    let pad = if n > (1 << 22) {
        1
    } else if n > (1 << 20) {
        2
    } else {
        4
    };
    let m = next_pow2(n) * pad;
    let mut fft = RealFft::new(m);

    let mut buf = vec![0.0; m];
    buf[..n].copy_from_slice(h);
    let mut spec = fft.spectrum();
    fft.forward(&mut buf, &mut spec);
    drop(buf);

    let peak = spec
        .iter()
        .map(|c| c.norm())
        .fold(0.0, f64::max)
        .max(1e-300);
    let floor = peak * 10f64.powf(-floor_db / 20.0);
    for c in spec.iter_mut() {
        *c = Complex::new(c.norm().max(floor).ln(), 0.0);
    }

    let mut ceps = vec![0.0; m];
    fft.inverse(&mut spec, &mut ceps);
    for v in ceps.iter_mut().take(m / 2).skip(1) {
        *v *= 2.0;
    }
    for v in ceps.iter_mut().skip(m / 2 + 1) {
        *v = 0.0;
    }

    fft.forward(&mut ceps, &mut spec);
    for c in spec.iter_mut() {
        *c = c.exp();
    }
    let mut out = ceps;
    fft.inverse(&mut spec, &mut out);
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

/// A filter family: the window and length policy behind a catalogue id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    /// Kaiser windowed sinc (`sinc-…`).
    #[default]
    Sinc,
    /// Dolph–Chebyshev windowed sinc (`closed-form-…`).
    ClosedForm,
    /// Gaussian windowed sinc (`poly-sinc-gauss-…`).
    PolySincGauss,
    /// Extremely long Kaiser sinc, 8M/16M taps, ≥ 200 dB (`sinc-xtr-…`).
    SincXtr,
    /// Kaiser sinc whose length scales with the ratio so the transition matches ×16 (`sinc-mx-…`).
    SincMx,
    /// Short sinc with Nuttall's optimised 3-term window (`poly-sinc-ext3-…`).
    PolySincExt3,
}

impl Family {
    /// Every family, longest id prefix first (so parsing `sinc-xtr-…` never matches `sinc`).
    pub const ALL: [Family; 6] = [
        Family::PolySincGauss,
        Family::PolySincExt3,
        Family::ClosedForm,
        Family::SincXtr,
        Family::SincMx,
        Family::Sinc,
    ];

    /// The id prefix.
    pub fn id(self) -> &'static str {
        match self {
            Family::Sinc => "sinc",
            Family::ClosedForm => "closed-form",
            Family::PolySincGauss => "poly-sinc-gauss",
            Family::SincXtr => "sinc-xtr",
            Family::SincMx => "sinc-mx",
            Family::PolySincExt3 => "poly-sinc-ext3",
        }
    }

    /// Display name.
    pub fn label(self) -> &'static str {
        match self {
            Family::Sinc => "sinc",
            Family::ClosedForm => "closed-form",
            Family::PolySincGauss => "poly-sinc-gauss",
            Family::SincXtr => "sinc-xtr",
            Family::SincMx => "sinc-mx",
            Family::PolySincExt3 => "poly-sinc-ext3",
        }
    }

    /// One-line description for the UI.
    pub fn description(self) -> &'static str {
        match self {
            Family::Sinc => "Kaiser windowed sinc; the classic long-FIR upsampler",
            Family::ClosedForm => {
                "Dolph–Chebyshev windowed sinc: equiripple stopband, narrowest transition per tap"
            }
            Family::PolySincGauss => "Gaussian windowed sinc: no sidelobes, least ringing",
            Family::SincXtr => "Extremely long sinc (8M/16M taps), ≥ 200 dB stopband",
            Family::SincMx => "1M-tap sinc matched to ×16: same transition in Hz at every rate",
            Family::PolySincExt3 => "Short sinc with Nuttall's optimised 3-term window",
        }
    }

    /// The window the family designs with.
    pub fn window(self) -> Window {
        match self {
            Family::Sinc | Family::SincXtr | Family::SincMx => Window::Kaiser,
            Family::ClosedForm => Window::DolphChebyshev,
            Family::PolySincGauss => Window::Gaussian,
            Family::PolySincExt3 => Window::Cosine3,
        }
    }

    /// Nominal lengths offered in the catalogue.
    pub fn catalog_taps(self) -> &'static [usize] {
        match self {
            Family::Sinc => &[1 << 8, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22],
            Family::ClosedForm => &[1 << 16, 1 << 20],
            Family::PolySincGauss => &[GAUSS_SHORT, GAUSS_LONG, GAUSS_XLA],
            Family::SincXtr => &[1 << 23, 1 << 24],
            Family::SincMx => &[1 << 20],
            Family::PolySincExt3 => &[1 << 13],
        }
    }

    /// Stopband attenuation the family is designed for (`sinc-xtr` asks for more than the
    /// default; the 3-term window delivers what it can).
    pub fn attenuation_db(self, requested: f64) -> f64 {
        match self {
            Family::SincXtr => requested.max(SINC_XTR_ATTENUATION_DB),
            Family::PolySincExt3 => COSINE3_ATTENUATION_DB,
            _ => requested,
        }
    }

    /// Parse the family prefix of an id; returns the family and the remainder after the `-`.
    fn split_prefix(id: &str) -> Option<(Family, &str)> {
        Family::ALL.iter().find_map(|f| {
            id.strip_prefix(f.id())
                .and_then(|rest| rest.strip_prefix('-'))
                .map(|rest| (*f, rest))
        })
    }
}

/// `poly-sinc-gauss-…-short` length.
pub const GAUSS_SHORT: usize = 1 << 14;
/// `poly-sinc-gauss-…-long` length.
pub const GAUSS_LONG: usize = 1 << 18;
/// `poly-sinc-gauss-…-xla` length.
pub const GAUSS_XLA: usize = 1 << 20;
/// Minimum stopband of the `sinc-xtr` family.
pub const SINC_XTR_ATTENUATION_DB: f64 = 200.0;
/// The ratio `sinc-mx` lengths are matched to.
pub const SINC_MX_REFERENCE_RATIO: usize = 16;

/// A user-facing filter choice: the shape of the filter independent of the sample rates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FilterKind {
    /// Which family (window / length policy).
    #[serde(default)]
    pub family: Family,
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

/// What `/api/filters` and `cadenza filters` show for one catalogue entry at a given ratio.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilterInfo {
    /// Catalogue id.
    pub id: String,
    /// Human readable label.
    pub label: String,
    /// Family id.
    pub family: String,
    /// Family description.
    pub description: String,
    /// `linear` or `minimum`.
    pub phase: String,
    /// Apodizing variant.
    pub apodizing: bool,
    /// Nominal tap count (what the id says).
    pub taps: usize,
    /// Tap count actually designed for `in_rate → out_rate`.
    pub taps_at_rate: usize,
    /// Window name.
    pub window: String,
    /// Input rate the numbers below are for.
    pub in_rate: u32,
    /// Output rate the numbers below are for.
    pub out_rate: u32,
    /// Cutoff (−6 dB point) in Hz.
    pub cutoff_hz: f64,
    /// Width of the transition band in Hz.
    pub transition_hz: f64,
    /// Transition band as a percentage of the source passband (HQPlayer-style figure).
    pub transition_percent: f64,
    /// Designed stopband attenuation in dB.
    pub stopband_db: f64,
    /// Group delay in output samples (0 for minimum phase).
    pub latency_samples: usize,
    /// Group delay in seconds.
    pub latency_seconds: f64,
    /// Coefficient storage (`f64`) in bytes.
    pub bytes: usize,
}

impl FilterKind {
    /// A filter of `family`, `phase`, apodizing or not, with a nominal length.
    pub fn new(family: Family, phase: Phase, apodizing: bool, taps: usize) -> Self {
        Self {
            family,
            phase,
            apodizing,
            taps,
        }
    }

    /// A Kaiser `sinc` filter (the pre-v2 constructor).
    pub fn sinc(phase: Phase, apodizing: bool, taps: usize) -> Self {
        Self::new(Family::Sinc, phase, apodizing, taps)
    }

    fn phase_token(&self) -> &'static str {
        match (self.phase, self.apodizing) {
            (Phase::Linear, false) => "l",
            (Phase::Minimum, false) => "m",
            (Phase::Linear, true) => "la",
            (Phase::Minimum, true) => "ma",
        }
    }

    /// The length token of the id: `64k`, `1m`, or `short`/`long`/`xla` for the Gaussians.
    pub fn length_token(&self) -> String {
        if self.family == Family::PolySincGauss {
            match self.taps {
                GAUSS_SHORT => return "short".into(),
                GAUSS_LONG => return "long".into(),
                GAUSS_XLA => return "xla".into(),
                _ => {}
            }
        }
        taps_label(self.taps)
    }

    /// The catalogue id, e.g. `sinc-l-64k`, `closed-form-ma-1m`, `poly-sinc-gauss-l-long`.
    pub fn id(&self) -> String {
        format!(
            "{}-{}-{}",
            self.family.id(),
            self.phase_token(),
            self.length_token()
        )
    }

    /// Human readable label, e.g. `sinc-L 64k taps · linear phase`.
    pub fn label(&self) -> String {
        let (fam, p) = match self.family {
            Family::Sinc => (
                String::new(),
                if self.phase == Phase::Linear {
                    "sinc-L"
                } else {
                    "sinc-M"
                },
            ),
            f => (f.label().to_string(), ""),
        };
        format!(
            "{fam}{p}{} {} taps · {} phase{}",
            if self.apodizing { "a" } else { "" },
            self.length_token(),
            self.phase.name(),
            if self.apodizing { " · apodizing" } else { "" }
        )
    }

    /// Parse an id produced by [`FilterKind::id`]; the tap count may be plain (`65536`),
    /// use `k`/`m` suffixes (`64k`, `1m`), or be a named length (`short`, `long`, `xla`).
    pub fn parse(id: &str) -> Option<Self> {
        let rest = id.trim().to_ascii_lowercase();
        let (family, rest) = Family::split_prefix(&rest)?;
        let (p, taps) = rest.split_once('-')?;
        let (phase, apodizing) = match p {
            "l" => (Phase::Linear, false),
            "m" => (Phase::Minimum, false),
            "la" => (Phase::Linear, true),
            "ma" => (Phase::Minimum, true),
            _ => return None,
        };
        let taps = match taps {
            "short" => GAUSS_SHORT,
            "long" => GAUSS_LONG,
            "xla" => GAUSS_XLA,
            t => parse_taps(t)?,
        };
        if !(MIN_TAPS..=MAX_TAPS).contains(&taps) {
            return None;
        }
        Some(Self {
            family,
            phase,
            apodizing,
            taps,
        })
    }

    /// The tap count designed for an upsampling ratio `l` (`sinc-mx` scales with the ratio).
    pub fn taps_for_ratio(&self, l: usize) -> usize {
        match self.family {
            Family::SincMx => {
                (self.taps * l.max(1) / SINC_MX_REFERENCE_RATIO).clamp(1 << 16, MAX_TAPS)
            }
            _ => self.taps,
        }
    }

    /// The design for a rational `l/m` conversion (the filter runs at `l ×` input rate, so
    /// the source Nyquist sits at `1 / (2 l)`), with the given attenuation.
    pub fn spec_for_ratio(&self, l: usize, m: usize, attenuation_db: f64) -> FirSpec {
        // For a rational L/M converter the filter runs at L × input rate and must keep
        // everything below the lower of the two Nyquist frequencies.
        let nyquist = 0.5 / l.max(m) as f64;
        let cutoff = if self.apodizing {
            nyquist * APODIZING_CUTOFF
        } else {
            nyquist
        };
        FirSpec::with_window(
            self.taps_for_ratio(l),
            cutoff,
            self.family.attenuation_db(attenuation_db),
            self.phase,
            self.family.window(),
        )
    }

    /// Describe this filter for a conversion `in_rate → out_rate` (taps, transition band,
    /// stopband, latency); `attenuation_db` is the configured stopband.
    pub fn describe(&self, in_rate: u32, out_rate: u32, attenuation_db: f64) -> FilterInfo {
        let g = gcd(in_rate.max(1), out_rate.max(1));
        let (l, m) = ((out_rate / g) as usize, (in_rate / g) as usize);
        let spec = self.spec_for_ratio(l.max(1), m.max(1), attenuation_db);
        let filter_rate = in_rate as f64 * l as f64;
        let passband = in_rate.min(out_rate) as f64 / 2.0;
        let latency_samples = spec.group_delay() / m.max(1);
        FilterInfo {
            id: self.id(),
            label: self.label(),
            family: self.family.id().into(),
            description: self.family.description().into(),
            phase: self.phase.name().into(),
            apodizing: self.apodizing,
            taps: self.taps,
            taps_at_rate: spec.taps,
            window: spec.window.name().into(),
            in_rate,
            out_rate,
            cutoff_hz: spec.cutoff * filter_rate,
            transition_hz: spec.transition_width() * filter_rate,
            transition_percent: 100.0 * spec.transition_width() * filter_rate / passband,
            stopband_db: spec.attenuation_db,
            latency_samples,
            latency_seconds: spec.group_delay() as f64 / filter_rate,
            bytes: spec.taps * std::mem::size_of::<f64>(),
        }
    }

    /// [`FilterKind::describe`] for the reference conversion 44.1 kHz → 705.6 kHz (×16).
    pub fn describe_default(&self) -> FilterInfo {
        self.describe(44_100, 705_600, DEFAULT_ATTENUATION_DB)
    }

    /// Every filter offered by default, in display order.
    pub fn catalog() -> Vec<FilterKind> {
        let mut out = Vec::new();
        for family in [
            Family::Sinc,
            Family::ClosedForm,
            Family::PolySincGauss,
            Family::SincXtr,
            Family::SincMx,
            Family::PolySincExt3,
        ] {
            for &taps in family.catalog_taps() {
                for &(phase, apodizing) in &[
                    (Phase::Linear, false),
                    (Phase::Minimum, false),
                    (Phase::Linear, true),
                    (Phase::Minimum, true),
                ] {
                    out.push(FilterKind {
                        family,
                        phase,
                        apodizing,
                        taps,
                    });
                }
            }
        }
        out
    }

    /// The catalogue described for the reference ×16 conversion.
    pub fn catalog_info() -> Vec<FilterInfo> {
        Self::catalog()
            .iter()
            .map(FilterKind::describe_default)
            .collect()
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl Default for FilterKind {
    fn default() -> Self {
        FilterKind {
            family: Family::Sinc,
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
    fn dolph_chebyshev_window_is_equiripple_at_the_requested_level() {
        for &(n, a) in &[(255usize, 80.0f64), (4097, 120.0), (65537, 150.0)] {
            let w = dolph_chebyshev_window(n, a);
            assert_eq!(w.len(), n);
            assert!((w[n / 2] - 1.0).abs() < 1e-9, "peak {}", w[n / 2]);
            assert!(w.iter().all(|v| *v > -1e-6 && *v <= 1.0 + 1e-9));
            assert!((w[0] - w[n - 1]).abs() < 1e-12);
            // Spectrum: the sidelobes sit at −a dB relative to the main lobe, and the main
            // lobe ends where the closed form says (first null at acosh(r)/(πN)).
            let db = magnitude_db(&w, n * 8);
            let peak = db[0];
            let r = 10f64.powf(a / 20.0);
            let f0 = acosh(r) / (PI * (n as f64 - 1.0));
            let nfft = db.len() * 2 - 2;
            let null_bin = (f0 * nfft as f64).round() as usize;
            let entered = db.iter().position(|d| peak - d >= a).unwrap_or(0);
            assert!(
                entered > 0 && entered <= null_bin + 2,
                "n={n}: sidelobe level reached at bin {entered}, main lobe predicted to end at {null_bin}"
            );
            let worst = db[null_bin * 2..].iter().cloned().fold(f64::MIN, f64::max);
            let side = peak - worst;
            assert!(
                (side - a).abs() < 1.0,
                "n={n}: sidelobes at −{side:.2} dB, wanted −{a}"
            );
        }
    }

    #[test]
    fn every_window_family_meets_its_design_band() {
        for &(window, taps, a, nfft) in &[
            (Window::DolphChebyshev, 1usize << 14, 140.0f64, 1usize << 18),
            (Window::Gaussian, 1 << 14, 140.0, 1 << 18),
            (Window::Cosine3, 1 << 13, 150.0, 1 << 17),
            (Window::Kaiser, 1 << 14, 140.0, 1 << 18),
        ] {
            let spec = FirSpec::with_window(taps, 1.0 / 32.0, a, Phase::Linear, window);
            let h = spec.design();
            assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let (ripple, atten) = measure(&h, &spec, nfft);
            let want = spec.attenuation_db - 3.0;
            assert!(
                atten >= want,
                "{window:?}: stopband {atten:.1} dB < {want:.1} (transition {:.3e})",
                spec.transition_width()
            );
            let max_ripple =
                20.0 * (1.0 + 10f64.powf(-spec.attenuation_db / 20.0)).log10() * 4.0 + 1e-6;
            assert!(
                ripple < max_ripple.max(1e-4),
                "{window:?}: passband ripple {ripple} dB"
            );
        }
    }

    #[test]
    fn transition_width_estimate_round_trips() {
        let spec = FirSpec::from_transition(0.25, 0.001, 120.0, Phase::Linear);
        let got = spec.transition_width();
        assert!((got - 0.001).abs() / 0.001 < 0.01, "transition {got}");
    }

    #[test]
    fn minimum_phase_front_loads_energy_and_keeps_magnitude() {
        for window in [Window::Kaiser, Window::DolphChebyshev, Window::Gaussian] {
            let spec = FirSpec::with_window(1 << 12, 0.125, 120.0, Phase::Minimum, window);
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
            assert!(front > 0.9, "{window:?}: min-phase front energy {front}");
            assert!(
                front_lin < 0.01,
                "{window:?}: linear-phase front energy {front_lin}"
            );

            let (ripple, atten) = measure(&min, &spec, 1 << 16);
            assert!(
                ripple < 0.05,
                "{window:?}: min-phase passband ripple {ripple} dB"
            );
            assert!(atten >= 100.0, "{window:?}: min-phase stopband {atten} dB");
        }
    }

    #[test]
    fn ids_round_trip_and_catalog_is_consistent() {
        let cat = FilterKind::catalog();
        assert_eq!(cat.len(), (7 + 2 + 3 + 2 + 1 + 1) * 4);
        for k in &cat {
            assert_eq!(FilterKind::parse(&k.id()), Some(*k), "{}", k.id());
            let info = k.describe_default();
            assert_eq!(info.id, k.id());
            assert!(
                info.transition_hz > 0.0 && info.stopband_db >= 70.0,
                "{}",
                k.id()
            );
            assert!(info.latency_seconds >= 0.0);
        }
        assert_eq!(FilterKind::parse("sinc-ma-1m").unwrap().taps, 1 << 20);
        assert_eq!(FilterKind::parse("sinc-l-65536").unwrap().taps, 65536);
        assert_eq!(
            FilterKind::parse("sinc-xtr-l-16m").unwrap().family,
            Family::SincXtr
        );
        assert_eq!(FilterKind::parse("sinc-xtr-l-16m").unwrap().taps, 1 << 24);
        assert_eq!(
            FilterKind::parse("poly-sinc-gauss-la-long").unwrap(),
            FilterKind::new(Family::PolySincGauss, Phase::Linear, true, GAUSS_LONG)
        );
        assert_eq!(
            FilterKind::parse("closed-form-m-1m").unwrap().family,
            Family::ClosedForm
        );
        assert_eq!(
            FilterKind::parse("poly-sinc-ext3-l-8k").unwrap().family,
            Family::PolySincExt3
        );
        assert!(FilterKind::parse("sinc-x-64k").is_none());
        assert!(FilterKind::parse("sinc-l-7").is_none());
        assert!(FilterKind::parse("sinc-l-32m").is_none());
        let spec = FilterKind::default().spec_for_ratio(4, 1, 150.0);
        assert!((spec.cutoff - 0.125).abs() < 1e-15);
        let apod = FilterKind {
            apodizing: true,
            ..Default::default()
        }
        .spec_for_ratio(4, 1, 150.0);
        assert!(apod.cutoff < spec.cutoff);
    }

    #[test]
    fn sinc_mx_keeps_the_transition_in_hz_across_ratios() {
        let mx = FilterKind::parse("sinc-mx-l-1m").unwrap();
        let at16 = mx.describe(44_100, 705_600, 150.0);
        let at2 = mx.describe(44_100, 88_200, 150.0);
        let at32 = mx.describe(44_100, 1_411_200, 150.0);
        assert_eq!(at16.taps_at_rate, (1 << 20) + 1);
        assert_eq!(at2.taps_at_rate, (1 << 17) + 1);
        assert_eq!(at32.taps_at_rate, (1 << 21) + 1);
        for info in [&at2, &at32] {
            let rel = (info.transition_hz - at16.transition_hz).abs() / at16.transition_hz;
            assert!(
                rel < 1e-3,
                "transition {} vs {}",
                info.transition_hz,
                at16.transition_hz
            );
        }
    }

    #[test]
    fn sinc_xtr_is_specified_as_promised() {
        let xtr = FilterKind::parse("sinc-xtr-l-16m")
            .unwrap()
            .describe_default();
        assert!(xtr.stopband_db >= 200.0);
        assert!(
            xtr.transition_percent < 0.5,
            "transition {}%",
            xtr.transition_percent
        );
        assert_eq!(xtr.bytes, ((1 << 24) + 1) * 8);
        let spec = FilterKind::parse("sinc-xtr-m-8m")
            .unwrap()
            .spec_for_ratio(16, 1, 150.0);
        assert_eq!(spec.taps, (1 << 23) + 1);
        assert_eq!(spec.attenuation_db, 200.0);
        assert!(spec.cache_key().starts_with("sinc-minimum-8388609-"));
    }

    #[test]
    fn describe_reports_rational_ratios() {
        let info = FilterKind::default().describe(44_100, 48_000, 150.0);
        assert!((info.cutoff_hz - 22_050.0).abs() < 1e-6);
        assert!(info.latency_samples > 0);
        let info = FilterKind::default().describe(44_100, 1_536_000, 150.0);
        assert!((info.cutoff_hz - 22_050.0).abs() < 1e-6);
    }
}
