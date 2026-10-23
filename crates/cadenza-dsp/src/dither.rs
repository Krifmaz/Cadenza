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
//! noise transfer function `1 − Σ c_k z^-k`. Two kinds of coefficient sets exist:
//!
//! * the published ones (Lipshitz / Vanderkooy / Wannamaker 1991, Wannamaker 1992, and the
//!   Gesemann sets as distributed with SoX), stored exactly as published for 44.1 kHz
//!   (Gesemann also for 48 kHz);
//! * the designed ones (`ns5`, `ns9`, `lns15`, `ns-adaptive`): weighted least-squares
//!   designs computed for the actual output rate (any 44.1 or 48 kHz-family rate) from a
//!   hearing-threshold weighting, see [`design_shaper`].
//!
//! Every shaped kind runs with a clip-safe limiter: the error fed back is bounded, so a
//! clipped full-scale passage can never wind the feedback loop up.

use crate::rng::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

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
    /// TPDF plus a 5th-order weighted least-squares shaper designed for the output rate
    /// (44.1 and 48 kHz families).
    Ns5,
    /// TPDF plus a 9th-order F-weighted least-squares shaper designed for the output rate.
    Ns9,
    /// TPDF plus a 15th-order shaper optimised to the Lipshitz/Wannamaker high-order
    /// weighting (steep, deep in-band suppression), designed for the output rate.
    Lns15,
    /// The 9th-order design with dithered error feedback whose step adapts to the signal:
    /// near full scale the dither and the shaped feedback are scaled back smoothly so the
    /// loop stays well inside the clip-safe region (lower idle tones, no limit cycles).
    NsAdaptive,
    /// Gaussian dither (σ = 0.5 LSB), no shaping.
    Gauss,
    /// TPDF plus a 1st-order `(1 − z^-1)` shaper: the gentlest slope out of the band.
    Ns1,
    /// TPDF plus a 4th-order weighted least-squares shaper designed for the output rate.
    Ns4,
    /// Chosen per output from the DAC's noise floor: the gentlest of TPDF, NS1, EF2, NS4,
    /// NS5, NS9 and LNS15 whose A-weighted in-band noise is at least 10 dB under it (see
    /// [`auto_choice`]).
    Auto,
}

/// Published noise-shaping coefficient sets, in the `1 − Σ c_k z^-k` convention.
pub mod coefficients {
    /// `(1 − z^-1)²`.
    pub const ERROR_FEEDBACK_2: &[f64] = &[2.0, -1.0];
    /// `1 − z^-1`.
    pub const FIRST_ORDER: &[f64] = &[1.0];
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

/// Absolute threshold of hearing in dB SPL (Terhardt 1979), `f` in Hz.
///
/// `3.64 f^-0.8 − 6.5 exp(−0.6 (f − 3.3)²) + 1e-3 f^4` with `f` in kHz. This is the curve
/// the designed shapers weight the noise with: noise is pushed where the ear is least
/// sensitive.
pub fn hearing_threshold_db(f_hz: f64) -> f64 {
    let f = (f_hz / 1000.0).max(0.02);
    3.64 * f.powf(-0.8) - 6.5 * (-0.6 * (f - 3.3).powi(2)).exp() + 1e-3 * f.powi(4)
}

/// The weighting a designed shaper minimises against: `10^(−T(f)/10)` with `T` the
/// hearing threshold relative to its minimum, limited to `range_db` of dynamic range
/// (what bounds the shaper's out-of-band gain, and so its stability margin).
pub fn weighting(f_hz: f64, range_db: f64) -> f64 {
    const T_MIN: f64 = -5.5; // minimum of the Terhardt curve, around 3.4 kHz
    let t = (hearing_threshold_db(f_hz) - T_MIN).clamp(0.0, range_db);
    10f64.powf(-t / 10.0)
}

/// Design an `order`-coefficient error-feedback noise shaper for sample rate `rate` by
/// weighted least squares: the `c_k` minimising `Σ_f W(f) |1 − Σ_k c_k e^{−jωk}|²` with
/// `W` = [`weighting`]. The minimiser of a monic polynomial's weighted power is the
/// autocorrelation (linear prediction) solution, which is always minimum phase, so the
/// resulting NTF is stable; `range_db` bounds its out-of-band gain.
pub fn design_shaper(order: usize, rate: u32, range_db: f64) -> Vec<f64> {
    let grid = 8192usize;
    let nyquist = rate as f64 / 2.0;
    // R[d] = Σ_f W cos(ω d), d = 0..order ; normal equations R_{kl} = R[|k−l|], rhs r_k = R[k+1]
    let mut acf = vec![0.0f64; order + 1];
    for i in 0..grid {
        let f = (i as f64 + 0.5) / grid as f64 * nyquist;
        let w = weighting(f, range_db);
        let omega = PI * (i as f64 + 0.5) / grid as f64;
        for (d, a) in acf.iter_mut().enumerate() {
            *a += w * (omega * d as f64).cos();
        }
    }
    // Solve the Toeplitz system by Gaussian elimination with partial pivoting (tiny system,
    // f64 is plenty with the regularising dynamic-range clamp).
    let n = order;
    let mut a = vec![vec![0.0f64; n + 1]; n];
    for k in 0..n {
        for l in 0..n {
            a[k][l] = acf[k.abs_diff(l)];
        }
        a[k][n] = acf[k + 1];
    }
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        a.swap(col, piv);
        let p = a[col][col];
        let pivot_row = a[col].clone();
        for row in a.iter_mut().skip(col + 1) {
            let f = row[col] / p;
            if f != 0.0 {
                for (v, pv) in row.iter_mut().zip(&pivot_row).skip(col) {
                    *v -= f * pv;
                }
            }
        }
    }
    let mut c = vec![0.0f64; n];
    for row in (0..n).rev() {
        let mut s = a[row][n];
        for col in row + 1..n {
            s -= a[row][col] * c[col];
        }
        c[row] = s / a[row][row];
    }
    c
}

/// Magnitude of the noise transfer function `1 − Σ c_k z^-k` at `f_hz` for `rate`, in dB.
pub fn ntf_db(coefs: &[f64], f_hz: f64, rate: u32) -> f64 {
    let omega = 2.0 * PI * f_hz / rate as f64;
    let (mut re, mut im) = (1.0, 0.0);
    for (k, c) in coefs.iter().enumerate() {
        let a = omega * (k as f64 + 1.0);
        re -= c * a.cos();
        im += c * a.sin();
    }
    10.0 * (re * re + im * im).max(1e-300).log10()
}

/// Dynamic range of the weighting for each designed order (bounds the NTF gain).
fn range_db_for(kind: DitherKind) -> f64 {
    match kind {
        DitherKind::Ns4 => 30.0,
        DitherKind::Ns5 => 36.0,
        DitherKind::Ns9 | DitherKind::NsAdaptive => 48.0,
        DitherKind::Lns15 => 60.0,
        _ => 0.0,
    }
}

/// Designed noise-shaping filters by kind and sample rate, computed once per process.
type DesignCache = Mutex<HashMap<(DitherKind, u32), Arc<Vec<f64>>>>;

/// The process-wide [`DesignCache`].
fn design_cache() -> &'static DesignCache {
    static CACHE: OnceLock<DesignCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Error magnitude (in LSB) the feedback loop is allowed to remember: the clip-safe
/// limiter. An unclipped sample's error is at most 0.5 LSB plus the dither (≤ 1 LSB for
/// TPDF); anything larger came from clipping and is cut off here.
pub const ERROR_LIMIT_LSB: f64 = 1.5;

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
            DitherKind::Ns5,
            DitherKind::Ns9,
            DitherKind::Lns15,
            DitherKind::NsAdaptive,
            DitherKind::Gauss,
            DitherKind::Ns1,
            DitherKind::Ns4,
            DitherKind::Auto,
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
            DitherKind::Ns5 => "ns5",
            DitherKind::Ns9 => "ns9",
            DitherKind::Lns15 => "lns15",
            DitherKind::NsAdaptive => "ns-adaptive",
            DitherKind::Gauss => "gauss",
            DitherKind::Ns1 => "ns1",
            DitherKind::Ns4 => "ns4",
            DitherKind::Auto => "auto",
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
            DitherKind::Ns5 => "NS5 · 5th-order weighted LS (44.1k/48k families)",
            DitherKind::Ns9 => "NS9 · 9th-order F-weighted LS (44.1k/48k families)",
            DitherKind::Lns15 => "LNS15 · 15th-order high-order weighting (44.1k/48k families)",
            DitherKind::NsAdaptive => "NS adaptive · 9th-order, adaptive dithered feedback",
            DitherKind::Gauss => "Gaussian (σ 0.5 LSB)",
            DitherKind::Ns1 => "NS1 · 1st-order",
            DitherKind::Ns4 => "NS4 · 4th-order weighted LS (44.1k/48k families)",
            DitherKind::Auto => "Auto · from the DAC's noise floor",
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
                "ns2" => Some(DitherKind::ErrorFeedback2),
                "gauss1" | "gaussian" => Some(DitherKind::Gauss),
                "ns-5" => Some(DitherKind::Ns5),
                "ns-9" => Some(DitherKind::Ns9),
                "lns-15" | "ns15" => Some(DitherKind::Lns15),
                "nsadaptive" | "adaptive" => Some(DitherKind::NsAdaptive),
                _ => None,
            })
    }

    /// Whether this kind's coefficients are designed for the output rate at build time.
    pub fn is_designed(self) -> bool {
        matches!(
            self,
            DitherKind::Ns4
                | DitherKind::Ns5
                | DitherKind::Ns9
                | DitherKind::Lns15
                | DitherKind::NsAdaptive
        )
    }

    /// Order of the designed kinds (number of feedback coefficients).
    pub fn order(self) -> usize {
        match self {
            DitherKind::Ns4 => 4,
            DitherKind::Ns5 => 5,
            DitherKind::Ns9 | DitherKind::NsAdaptive => 9,
            DitherKind::Lns15 => 15,
            other => other.coefficients().len(),
        }
    }

    /// The noise-shaping feedback coefficients, if this kind shapes. Designed kinds return
    /// their 44.1 kHz set; see [`DitherKind::coefficients_for_rate`].
    pub fn coefficients(self) -> &'static [f64] {
        match self {
            DitherKind::None
            | DitherKind::Rectangular
            | DitherKind::Tpdf
            | DitherKind::Gauss
            | DitherKind::Auto => &[],
            DitherKind::ErrorFeedback2 => coefficients::ERROR_FEEDBACK_2,
            DitherKind::Ns1 => coefficients::FIRST_ORDER,
            DitherKind::Ns4 => {
                static C: OnceLock<Vec<f64>> = OnceLock::new();
                C.get_or_init(|| DitherKind::Ns4.coefficients_for_rate(44_100).to_vec())
            }
            DitherKind::Lipshitz => coefficients::LIPSHITZ_44,
            DitherKind::Wannamaker9 => coefficients::WANNAMAKER_9_44,
            DitherKind::Wannamaker3 => coefficients::WANNAMAKER_3_44,
            DitherKind::Gesemann441 => coefficients::GESEMANN_44,
            DitherKind::Gesemann48 => coefficients::GESEMANN_48,
            DitherKind::Ns5 => {
                static C: OnceLock<Vec<f64>> = OnceLock::new();
                C.get_or_init(|| DitherKind::Ns5.coefficients_for_rate(44_100).to_vec())
            }
            DitherKind::Ns9 => {
                static C: OnceLock<Vec<f64>> = OnceLock::new();
                C.get_or_init(|| DitherKind::Ns9.coefficients_for_rate(44_100).to_vec())
            }
            DitherKind::Lns15 => {
                static C: OnceLock<Vec<f64>> = OnceLock::new();
                C.get_or_init(|| DitherKind::Lns15.coefficients_for_rate(44_100).to_vec())
            }
            DitherKind::NsAdaptive => DitherKind::Ns9.coefficients(),
        }
    }

    /// The feedback coefficients for an output rate: designed kinds are computed (once,
    /// then cached in memory) for exactly `rate`; published kinds are fixed.
    pub fn coefficients_for_rate(self, rate: u32) -> Arc<Vec<f64>> {
        if !self.is_designed() {
            return Arc::new(self.coefficients().to_vec());
        }
        let key = (self.design_kind(), rate.max(8000));
        if let Some(c) = design_cache().lock().unwrap().get(&key) {
            return Arc::clone(c);
        }
        let c = Arc::new(design_shaper(self.order(), key.1, range_db_for(self)));
        design_cache().lock().unwrap().insert(key, Arc::clone(&c));
        c
    }

    /// The kind whose shaper is designed: `ns-adaptive` scales the `ns9` shaper at run time.
    fn design_kind(self) -> DitherKind {
        match self {
            DitherKind::NsAdaptive => DitherKind::Ns9,
            k => k,
        }
    }

    /// The sample rate a psychoacoustic shaper was designed for (its curve is only
    /// meaningful there); `None` for rate-independent kinds and for the designed kinds,
    /// which follow the output rate.
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

    /// Variance of the dither this kind adds, in LSB².
    pub fn dither_variance(self) -> f64 {
        match self {
            DitherKind::None => 0.0,
            DitherKind::Rectangular => 1.0 / 12.0,
            DitherKind::Gauss => GAUSS_SIGMA * GAUSS_SIGMA,
            _ => 1.0 / 6.0,
        }
    }
}

/// Standard deviation of the Gaussian dither, in LSB.
pub const GAUSS_SIGMA: f64 = 0.5;

/// The DAC noise floor `auto` assumes when none was measured or entered (dBFS, A-weighted):
/// a good current DAC.
pub const DEFAULT_DAC_FLOOR_DB: f64 = -120.0;

/// How far under the DAC's floor `auto` wants the dither noise (dB).
pub const AUTO_MARGIN_DB: f64 = 10.0;

/// The kinds `auto` picks from, gentlest first.
pub const AUTO_CANDIDATES: [DitherKind; 7] = [
    DitherKind::Tpdf,
    DitherKind::Ns1,
    DitherKind::ErrorFeedback2,
    DitherKind::Ns4,
    DitherKind::Ns5,
    DitherKind::Ns9,
    DitherKind::Lns15,
];

/// A-weighting gain at `f_hz` (IEC 61672), linear (1 at 1 kHz).
pub fn a_weighting(f_hz: f64) -> f64 {
    let f2 = f_hz * f_hz;
    let ra = 12194f64.powi(2) * f2 * f2
        / ((f2 + 20.6f64.powi(2))
            * ((f2 + 107.7f64.powi(2)) * (f2 + 737.9f64.powi(2))).sqrt()
            * (f2 + 12194f64.powi(2)));
    ra * 10f64.powf(2.0 / 20.0)
}

/// Range of the hearing-threshold weighting used to judge audibility (dB).
const AUDIBILITY_RANGE_DB: f64 = 100.0;

/// `∫ W df` over 20 Hz – 20 kHz for a weighting `W` (midpoint rule).
fn band_integral(w: impl Fn(f64) -> f64) -> f64 {
    let steps = 4000;
    let df = (20_000.0 - 20.0) / steps as f64;
    (0..steps)
        .map(|i| w(20.0 + (i as f64 + 0.5) * df) * df)
        .sum()
}

/// How audible the noise a kind leaves at `bits` and `rate` is, in dBFS (relative to a
/// full-scale sine): the quantisation error plus dither, through the noise transfer
/// function, weighted by the hearing threshold (what the shapers are designed against) and
/// scaled so white noise reads the same as A-weighted – comparable with a DAC's A-weighted
/// noise floor. A-weighting alone would rank a psychoacoustic shaper below plain TPDF: it
/// still counts the noise moved to 15–20 kHz, where the ear is far less sensitive at these
/// levels.
pub fn audible_noise_dbfs(kind: DitherKind, bits: u32, rate: u32) -> f64 {
    let coefs = kind.coefficients_for_rate(rate);
    let var = 1.0 / 12.0 + kind.dither_variance();
    let top = 20_000f64.min(rate as f64 / 2.0);
    let w = |f: f64| {
        if f > top {
            0.0
        } else {
            10f64.powf(ntf_db(&coefs, f, rate) / 10.0) * weighting(f, AUDIBILITY_RANGE_DB)
        }
    };
    let calibration = band_integral(|f| a_weighting(f).powi(2))
        / band_integral(|f| weighting(f, AUDIBILITY_RANGE_DB));
    // White error of variance `var` spreads over 0 … rate/2.
    let p = var * band_integral(w) * calibration / (rate as f64 / 2.0);
    let full_scale = (1u64 << (bits - 1)) as f64;
    10.0 * (p / (full_scale * full_scale / 2.0)).log10()
}

/// What `auto` runs for an output of `bits` at `rate` into a DAC whose own noise floor is
/// `dac_floor_db` dBFS (A-weighted): the gentlest candidate whose audible noise is
/// [`AUTO_MARGIN_DB`] under the floor, or the quietest one when none is. Gentler shapers put
/// less noise power out of the band, so they are preferred when they are already inaudible.
pub fn auto_choice(bits: u32, rate: u32, dac_floor_db: f64) -> DitherKind {
    let target = dac_floor_db - AUTO_MARGIN_DB;
    let noise: Vec<(DitherKind, f64)> = AUTO_CANDIDATES
        .iter()
        .map(|&k| (k, audible_noise_dbfs(k, bits, rate)))
        .collect();
    noise
        .iter()
        .find(|(_, n)| *n <= target)
        .or_else(|| noise.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()))
        .map(|(k, _)| *k)
        .unwrap_or(DitherKind::Tpdf)
}

/// Converts `f64` samples in `[-1, 1)` to integers of `bits` bits, per channel.
#[derive(Clone, Debug)]
pub struct Quantizer {
    /// Output bit depth.
    bits: u32,
    /// Dither and shaping actually applied (after `auto` resolved).
    kind: DitherKind,
    /// Output rate the shaper was designed for.
    rate: u32,
    /// Interleaved channel count (at least 1).
    channels: usize,
    /// Full scale in output LSBs: `2^(bits − 1)`.
    scale: f64,
    /// Lowest representable output value in LSBs.
    min: f64,
    /// Highest representable output value in LSBs.
    max: f64,
    /// Error-feedback coefficients of the shaper (empty for plain dither).
    coefs: Arc<Vec<f64>>,
    /// Per channel ring buffer of past errors (newest at `pos`).
    errors: Vec<Vec<f64>>,
    /// Ring index shared by every channel's error history.
    pos: usize,
    /// Per channel adaptive step of `ns-adaptive` (1 = full shaping and dither).
    step: Vec<f64>,
    /// Samples whose error exceeded the clip-safe limit (the output clipped).
    clipped: u64,
    /// Generator for the dither noise.
    rng: Rng,
    /// `Some(dac floor)` when `auto` chose the kind.
    auto: Option<f64>,
    /// The second Box–Muller deviate.
    spare: Option<f64>,
}

/// Level (fraction of full scale) above which `ns-adaptive` starts scaling back.
const ADAPTIVE_KNEE: f64 = 0.9;

impl Quantizer {
    /// A quantiser to `bits` (16, 24 or 32) for `channels` interleaved channels; designed
    /// shapers use their 44.1 kHz set (see [`Quantizer::for_rate`]).
    pub fn new(bits: u32, kind: DitherKind, channels: usize) -> Self {
        Self::for_rate(bits, kind, channels, 44_100)
    }

    /// A quantiser whose designed shaper (`ns4`, `ns5`, `ns9`, `lns15`, `ns-adaptive`) is
    /// computed for the output `rate`; `auto` assumes [`DEFAULT_DAC_FLOOR_DB`].
    pub fn for_rate(bits: u32, kind: DitherKind, channels: usize, rate: u32) -> Self {
        Self::for_dac(bits, kind, channels, rate, DEFAULT_DAC_FLOOR_DB)
    }

    /// [`Quantizer::for_rate`] for a DAC whose noise floor is `dac_floor_db` dBFS
    /// (A-weighted), which `auto` chooses from.
    pub fn for_dac(
        bits: u32,
        kind: DitherKind,
        channels: usize,
        rate: u32,
        dac_floor_db: f64,
    ) -> Self {
        let auto = kind == DitherKind::Auto;
        let kind = if auto {
            auto_choice(bits, rate, dac_floor_db)
        } else {
            kind
        };
        assert!(
            matches!(bits, 8 | 16 | 20 | 24 | 32),
            "unsupported bit depth {bits}"
        );
        let scale = (1u64 << (bits - 1)) as f64;
        let coefs = kind.coefficients_for_rate(rate);
        Self {
            bits,
            kind,
            rate,
            channels: channels.max(1),
            scale,
            min: -scale,
            max: scale - 1.0,
            errors: vec![vec![0.0; coefs.len().max(1)]; channels.max(1)],
            coefs,
            pos: 0,
            step: vec![1.0; channels.max(1)],
            clipped: 0,
            rng: Rng::default(),
            auto: auto.then_some(dac_floor_db),
            spare: None,
        }
    }

    /// When the kind was chosen by `auto`: the DAC floor it was chosen for.
    pub fn auto_floor(&self) -> Option<f64> {
        self.auto
    }

    /// Output bit depth.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// The dither kind in use.
    pub fn kind(&self) -> DitherKind {
        self.kind
    }

    /// The output rate the shaper was designed for.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// The feedback coefficients actually running.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefs
    }

    /// Samples so far whose quantisation error exceeded [`ERROR_LIMIT_LSB`], i.e. that clipped.
    pub fn clipped(&self) -> u64 {
        self.clipped
    }

    /// Forget the error history.
    pub fn reset(&mut self) {
        for e in &mut self.errors {
            e.fill(0.0);
        }
        self.step.fill(1.0);
        self.pos = 0;
    }

    /// One dither sample in LSBs for the current kind (TPDF, rectangular, Gaussian or none).
    #[inline]
    fn dither(&mut self) -> f64 {
        match self.kind {
            DitherKind::None => 0.0,
            DitherKind::Rectangular => self.rng.centred(),
            DitherKind::Gauss => {
                if let Some(v) = self.spare.take() {
                    return v;
                }
                let (u1, u2) = (1.0 - self.rng.uniform(), self.rng.uniform());
                let r = GAUSS_SIGMA * (-2.0 * u1.ln()).sqrt();
                self.spare = Some(r * (2.0 * PI * u2).sin());
                r * (2.0 * PI * u2).cos()
            }
            _ => self.rng.centred() + self.rng.centred(),
        }
    }

    /// Quantise interleaved `input`, appending integers (in the native range of `bits`,
    /// e.g. −32768…32767 for 16) to `output`.
    pub fn process(&mut self, input: &[f64], output: &mut Vec<i32>) {
        output.reserve(input.len());
        let ch = self.channels;
        let n = self.coefs.len();
        let adaptive = self.kind == DitherKind::NsAdaptive;
        for frame in input.chunks(ch) {
            for (c, &x) in frame.iter().enumerate() {
                let v = x * self.scale;
                let step = if adaptive {
                    // Smoothly back off between the knee and full scale; recover slowly.
                    let level = x.abs();
                    let target = if level <= ADAPTIVE_KNEE {
                        1.0
                    } else {
                        (1.0 - (level - ADAPTIVE_KNEE) / (1.0 - ADAPTIVE_KNEE)).clamp(0.0, 1.0)
                    };
                    let s = self.step[c];
                    let s = if target < s {
                        target
                    } else {
                        s + (target - s) * 0.002
                    };
                    self.step[c] = s;
                    s
                } else {
                    1.0
                };
                let shaped = if n == 0 {
                    v
                } else {
                    let hist = &self.errors[c];
                    let mut fb = 0.0;
                    for (k, coef) in self.coefs.iter().enumerate() {
                        fb += coef * hist[(self.pos + n - k) % n];
                    }
                    v - fb * step
                };
                let d = self.dither() * step;
                let q = (shaped + d).round().clamp(self.min, self.max);
                if n > 0 {
                    // Feed back the error against the un-dithered, shaped value. The error
                    // is bounded (clip-safe limiter): a clipped sample's excess never enters
                    // the loop, so the shaper cannot wind up.
                    let next = (self.pos + 1) % n;
                    let e = q - shaped;
                    let limited = e.clamp(-ERROR_LIMIT_LSB, ERROR_LIMIT_LSB);
                    if limited != e {
                        self.clipped += 1;
                    }
                    self.errors[c][next] = limited;
                    if c == ch - 1 {
                        self.pos = next;
                    }
                } else if q != shaped + d && (q == self.min || q == self.max) {
                    self.clipped += 1;
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

    /// Noise power spectrum (dB per bin) of the quantisation error of a quiet tone.
    fn error_spectrum(kind: DitherKind, rate: u32, n: usize) -> Vec<f64> {
        let x = quiet_sine(50.0, 16, n);
        let mut q = Quantizer::for_rate(16, kind, 1, rate);
        let y = q.process_to_f64(&x);
        // Blackman–Harris window: the shaped spectra span 60 dB, and a rectangular window's
        // leakage from the ultrasonic region would fill the in-band notch.
        let err: Vec<f64> = y
            .iter()
            .zip(&x)
            .enumerate()
            .map(|(i, (a, b))| {
                let t = 2.0 * PI * i as f64 / n as f64;
                let w = 0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos()
                    - 0.01168 * (3.0 * t).cos();
                (a - b) * w
            })
            .collect();
        magnitude_db(&err, n)
    }

    fn band_power_db(db: &[f64], rate: u32, lo_hz: f64, hi_hz: f64) -> f64 {
        let n = (db.len() - 1) * 2;
        let hz_per_bin = rate as f64 / n as f64;
        let (lo, hi) = ((lo_hz / hz_per_bin) as usize, (hi_hz / hz_per_bin) as usize);
        let power: f64 = db[lo..hi].iter().map(|d| 10f64.powf(d / 10.0)).sum();
        10.0 * power.log10()
    }

    fn in_band_noise_db(kind: DitherKind) -> f64 {
        band_power_db(&error_spectrum(kind, 44100, 1 << 16), 44100, 100.0, 4000.0)
    }

    #[test]
    fn f_weighted_shapers_lower_the_in_band_noise() {
        let flat = in_band_noise_db(DitherKind::Tpdf);
        for kind in [
            DitherKind::Lipshitz,
            DitherKind::Wannamaker9,
            DitherKind::Wannamaker3,
            DitherKind::Ns5,
            DitherKind::Ns9,
            DitherKind::Lns15,
            DitherKind::NsAdaptive,
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

    /// The designed shapers' measured noise spectrum lies below the weighted target
    /// curve: flat TPDF noise minus the suppression their NTF promises, in every 1 kHz
    /// band up to 16 kHz (the ear-weighted region), at both base rates.
    #[test]
    fn designed_shapers_follow_the_weighted_target_curve() {
        for rate in [44_100u32, 48_000] {
            let n = 1 << 16;
            let flat = error_spectrum(DitherKind::Tpdf, rate, n);
            for kind in [DitherKind::Ns5, DitherKind::Ns9, DitherKind::Lns15] {
                let coefs = kind.coefficients_for_rate(rate);
                assert_eq!(coefs.len(), kind.order());
                let shaped = error_spectrum(kind, rate, n);
                let mut worst_excess = f64::MIN;
                let mut bands_checked = 0;
                let mut lo = 200.0;
                while lo + 1000.0 <= 16_000.0 {
                    let hi = lo + 1000.0;
                    // Power average of the NTF over the band (a notch at the band centre
                    // would otherwise make the target unfairly deep).
                    let target_ntf = {
                        let steps = 64;
                        let p: f64 = (0..steps)
                            .map(|i| lo + (i as f64 + 0.5) / steps as f64 * (hi - lo))
                            .map(|f| 10f64.powf(ntf_db(&coefs, f, rate) / 10.0))
                            .sum::<f64>()
                            / steps as f64;
                        10.0 * p.log10()
                    };
                    let got =
                        band_power_db(&shaped, rate, lo, hi) - band_power_db(&flat, rate, lo, hi);
                    // Allow 2 dB for the finite-length estimate and the dither's own TPDF
                    // spectrum; the target curve is the NTF (what the design promises).
                    worst_excess = worst_excess.max(got - target_ntf);
                    bands_checked += 1;
                    lo = hi;
                }
                assert!(bands_checked >= 15);
                assert!(
                    worst_excess < 2.0,
                    "{kind:?} @ {rate}: noise exceeds the NTF target by {worst_excess:.1} dB"
                );
                // And the target itself is a real improvement where the ear is sensitive.
                let at_3k = ntf_db(&coefs, 3_000.0, rate);
                let floor = match kind {
                    DitherKind::Ns5 => -14.0,
                    DitherKind::Ns9 => -22.0,
                    _ => -27.0,
                };
                assert!(
                    at_3k < floor,
                    "{kind:?} @ {rate}: NTF at 3 kHz is {at_3k:.1} dB"
                );
            }
            // Higher order is strictly better at 3 kHz.
            let a = ntf_db(&DitherKind::Ns5.coefficients_for_rate(rate), 3000.0, rate);
            let b = ntf_db(&DitherKind::Ns9.coefficients_for_rate(rate), 3000.0, rate);
            let c = ntf_db(&DitherKind::Lns15.coefficients_for_rate(rate), 3000.0, rate);
            assert!(a > b && b > c, "{a} {b} {c}");
        }
    }

    #[test]
    fn designed_shapers_stay_stable_at_full_scale_for_ten_seconds() {
        for kind in [
            DitherKind::Ns5,
            DitherKind::Ns9,
            DitherKind::Lns15,
            DitherKind::NsAdaptive,
            DitherKind::Wannamaker9,
        ] {
            for rate in [44_100u32, 48_000] {
                let n = rate as usize * 10;
                // Full-scale: a 1 kHz tone at exactly 0 dBFS plus a DC offset so it clips hard.
                let x: Vec<f64> = (0..n)
                    .map(|i| {
                        (1.0 * (2.0 * PI * 1000.0 * i as f64 / rate as f64).sin() + 0.1)
                            .clamp(-1.0, 1.0)
                    })
                    .collect();
                let mut q = Quantizer::for_rate(16, kind, 1, rate);
                let y = q.process_to_f64(&x);
                let worst = y
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                // The output can only differ from the input by clipping to the grid (the
                // input itself touches +1.0, which the integer range cannot hold) plus the
                // shaper's bounded noise; a wound-up loop would be far above this.
                let gain: f64 = 1.0 + q.coefficients().iter().map(|c| c.abs()).sum::<f64>();
                let bound = (ERROR_LIMIT_LSB * gain + 2.0) / 32768.0 + 1.0 / 32768.0;
                assert!(worst < bound, "{kind:?} @ {rate}: error {worst} ≥ {bound}");
                assert!(y.iter().all(|v| v.is_finite()));
                if kind != DitherKind::NsAdaptive {
                    assert!(
                        q.clipped() > 0,
                        "{kind:?}: the full-scale input must have clipped"
                    );
                }
                // The tail must still be quiet noise around the tone, not a limit cycle.
                let tail = &y[n - 4096..];
                let err: Vec<f64> = tail
                    .iter()
                    .zip(&x[n - 4096..])
                    .map(|(a, b)| a - b)
                    .collect();
                let rms = (err.iter().map(|e| e * e).sum::<f64>() / err.len() as f64).sqrt();
                // Expected shaped noise: 0.5 LSB rms (rounding + TPDF) times the NTF's rms
                // gain; clipping adds some, a limit cycle would add a lot.
                let rms_gain = (1.0 + q.coefficients().iter().map(|c| c * c).sum::<f64>()).sqrt();
                let expected = 0.5 * rms_gain / 32768.0;
                assert!(
                    rms < 3.0 * expected,
                    "{kind:?} @ {rate}: tail error rms {rms} vs expected {expected}"
                );
            }
        }
    }

    #[test]
    fn adaptive_shaper_backs_off_near_full_scale_and_recovers() {
        let rate = 44_100;
        let loud: Vec<f64> = (0..20_000)
            .map(|i| 0.999 * (2.0 * PI * 50.0 * i as f64 / rate as f64).sin())
            .collect();
        let mut q = Quantizer::for_rate(16, DitherKind::NsAdaptive, 1, rate);
        q.process_to_f64(&loud);
        assert!(q.step[0] < 1.0, "step {} did not back off", q.step[0]);
        let quiet = quiet_sine(50.0, 16, 60_000);
        q.process_to_f64(&quiet);
        assert!(q.step[0] > 0.99, "step {} did not recover", q.step[0]);
        // Quiet material is shaped like ns9.
        let a = in_band_noise_db(DitherKind::NsAdaptive);
        let b = in_band_noise_db(DitherKind::Ns9);
        assert!((a - b).abs() < 3.0, "{a} vs {b}");
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
    fn designs_are_cached_and_rate_specific() {
        let a = DitherKind::Ns9.coefficients_for_rate(44_100);
        let b = DitherKind::Ns9.coefficients_for_rate(44_100);
        assert!(Arc::ptr_eq(&a, &b));
        let c = DitherKind::Ns9.coefficients_for_rate(48_000);
        assert_ne!(a.as_slice(), c.as_slice());
        assert_eq!(DitherKind::Ns9.coefficients(), a.as_slice());
        assert_eq!(DitherKind::NsAdaptive.coefficients(), a.as_slice());
        // High output rates design fine too (the whole audio band is "in band").
        let hi = DitherKind::Lns15.coefficients_for_rate(352_800);
        assert_eq!(hi.len(), 15);
        assert!(hi.iter().all(|c| c.is_finite()));
        assert!(ntf_db(&hi, 10_000.0, 352_800) < -20.0);
    }

    #[test]
    fn ids_round_trip() {
        for k in DitherKind::catalog() {
            assert_eq!(DitherKind::parse(k.id()), Some(*k));
        }
        assert_eq!(DitherKind::parse("TPDF"), Some(DitherKind::Tpdf));
        assert_eq!(DitherKind::parse("LNS15"), Some(DitherKind::Lns15));
        assert_eq!(
            DitherKind::parse("ns-adaptive"),
            Some(DitherKind::NsAdaptive)
        );
        assert_eq!(DitherKind::parse("nope"), None);
    }

    #[test]
    fn new_kinds_parse_and_ns1_is_no_longer_an_alias() {
        for (id, k) in [
            ("ns1", DitherKind::Ns1),
            ("ns4", DitherKind::Ns4),
            ("gauss", DitherKind::Gauss),
            ("gauss1", DitherKind::Gauss),
            ("auto", DitherKind::Auto),
            ("ns2", DitherKind::ErrorFeedback2),
        ] {
            assert_eq!(DitherKind::parse(id), Some(k), "{id}");
        }
        assert_eq!(DitherKind::Ns1.coefficients(), &[1.0]);
        for rate in [44_100, 48_000, 96_000, 192_000] {
            let c = DitherKind::Ns4.coefficients_for_rate(rate);
            assert_eq!(c.len(), 4);
            // Unity-ish at DC is not required, but the NTF must dip in the band and rise above it.
            assert!(ntf_db(&c, 3000.0, rate) < -10.0, "{rate}");
            assert!(ntf_db(&c, rate as f64 * 0.45, rate) > 3.0, "{rate}");
        }
    }

    #[test]
    fn gaussian_dither_has_the_stated_spread() {
        let mut q = Quantizer::new(16, DitherKind::Gauss, 1);
        let x = vec![0.25 / 32768.0; 200_000];
        let y = q.process_to_f64(&x);
        let e: Vec<f64> = y.iter().zip(&x).map(|(a, b)| (a - b) * 32768.0).collect();
        let mean = e.iter().sum::<f64>() / e.len() as f64;
        let var = e.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / e.len() as f64;
        let want = 1.0 / 12.0 + GAUSS_SIGMA * GAUSS_SIGMA;
        assert!(
            (var / want - 1.0).abs() < 0.05,
            "variance {var}, want {want}"
        );
        assert!(mean.abs() < 0.01, "mean {mean}");
    }

    /// The audibility model agrees with what the quantiser actually does: the weighted
    /// in-band noise of a run, relative to TPDF, matches the prediction.
    #[test]
    fn predicted_audible_noise_matches_the_quantiser() {
        let rate = 44_100;
        let n = 1 << 18;
        let weighted = |kind: DitherKind| {
            let db = error_spectrum(kind, rate, n);
            let hz = rate as f64 / n as f64;
            let p: f64 = db
                .iter()
                .enumerate()
                .map(|(i, d)| (i as f64 * hz, d))
                .filter(|(f, _)| (20.0..20_000.0).contains(f))
                .map(|(f, d)| 10f64.powf(d / 10.0) * weighting(f, AUDIBILITY_RANGE_DB))
                .sum();
            10.0 * p.log10()
        };
        let base = weighted(DitherKind::Tpdf);
        let base_model = audible_noise_dbfs(DitherKind::Tpdf, 16, rate);
        for kind in [
            DitherKind::Ns1,
            DitherKind::ErrorFeedback2,
            DitherKind::Ns4,
            DitherKind::Ns9,
            DitherKind::Lns15,
        ] {
            let measured = weighted(kind) - base;
            let model = audible_noise_dbfs(kind, 16, rate) - base_model;
            assert!(
                (measured - model).abs() < 1.5,
                "{kind:?}: measured {measured:.1} dB, model {model:.1} dB"
            );
        }
    }

    #[test]
    fn auto_picks_by_the_dac_floor() {
        // 16 bits at 44.1 kHz under a good DAC: the strongest shaping.
        assert_eq!(auto_choice(16, 44_100, -120.0), DitherKind::Lns15);
        // A noisy DAC needs less.
        assert_eq!(auto_choice(16, 44_100, -100.0), DitherKind::Ns9);
        // 24 bits: TPDF is already far under any DAC.
        assert_eq!(auto_choice(24, 44_100, -130.0), DitherKind::Tpdf);
        assert_eq!(auto_choice(24, 384_000, -130.0), DitherKind::Tpdf);
        // At high rates gentle shapers already clear the floor.
        assert_eq!(auto_choice(16, 384_000, -110.0), DitherKind::Ns1);
        // A quieter DAC never gets a gentler shaper.
        for rate in [44_100, 48_000, 96_000, 192_000] {
            let mut last = 0;
            for floor in [-90.0, -100.0, -110.0, -120.0, -130.0] {
                let k = auto_choice(16, rate, floor);
                let i = AUTO_CANDIDATES.iter().position(|c| *c == k).unwrap();
                assert!(i >= last, "{rate} {floor}: {k:?}");
                last = i;
            }
        }
        // The quantiser resolves it and says so.
        let q = Quantizer::for_dac(16, DitherKind::Auto, 2, 48_000, -105.0);
        assert_eq!(q.kind(), auto_choice(16, 48_000, -105.0));
        assert_eq!(q.auto_floor(), Some(-105.0));
        assert_eq!(
            Quantizer::for_rate(16, DitherKind::Tpdf, 2, 48_000).auto_floor(),
            None
        );
    }
}
