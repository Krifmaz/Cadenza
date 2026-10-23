//! 1-bit sigma-delta modulators (`asdm5`, `asdm7`, `dsd5v2`, `dsd7v2`, and the
//! error-correction loops `asdm5ec`, `asdm7ec`, `asdm7ec8`).
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
//! on DC, the worst case). The design point depends on the oversampling ratio
//! ([`DsdModulator::h_inf`]): the 5th-order loops use `H∞ = 1.35` at DSD64 (≈ 110 dB SNR
//! in 20 kHz for a −6 dBFS tone, stable to 0.72) and `H∞ = 1.28` from DSD128 up (where
//! the SNR is ≥ 118 dB anyway and the tamer loop stays clear of the reset guard on
//! sustained full-scale DC and on a full-scale DC step); the 7th-order loops use
//! `H∞ = 1.28` at DSD64 (≈ 117 dB, stable to 0.72) and `H∞ = 1.25` from DSD128 up.
//! Measured over a second each of +0.708 and −0.708 DC (a ±0.708 step between them) and
//! 1 kHz / 20 kHz / 40 Hz tones at that level, at every rate in both families, none of the
//! guards engages (the sweep also showed that the neighbouring values 1.32/1.30 for order 5
//! and 1.28 for order 7 each still trip one reset at some rate, so the chosen values carry
//! a margin), hence the default `dsd_gain_db = -3` (0 dBFS → 0.708) sits inside the stable
//! range at every DSD rate.
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
//!
//! **Error correction (`ec`).** The `ec` modulators run exactly the same NTF loop (same
//! design point, same input limit) but do not decide each bit on its own: a look-ahead
//! trellis ([`Lookahead`]) keeps the `N` best candidate bit sequences over the next `K`
//! bits (an M-algorithm, the pruned form of a Viterbi search over all `2^K` sequences)
//! and commits the oldest bit of the cheapest path. The plain quantiser is the special
//! case `K = 0`: greedy, minimising `|e[n]|` one sample at a time.
//!
//! The path cost is `Σ (e² + W·f²)`, with `f` the quantisation error through a
//! first-order low-pass at 4.5 kHz (unity DC gain). `e²` alone is a poor target: `e` is
//! almost white and its energy is dominated by its out-of-band part, which the NTF lets
//! through anyway – minimising it measurably *lowers* the SNR of a 7th-order loop. The
//! in-band output noise is `|NTF|²·|E|²` inside 20 kHz, so what has to go down is the
//! low-frequency content of `e`, which is what `f` measures; the trellis then works as an
//! extra noise-shaping stage on the quantiser error that a sample-by-sample quantiser
//! cannot provide. `W` (per modulator, referenced to DSD64 and scaled with the rate)
//! trades in-band SNR against headroom on DC; the chosen values keep at least 0.71 FS of
//! DC stable at DSD64 in a 250 ms DC-then-tone sweep (`asdm5ec` 0.71 like `asdm5`,
//! `asdm7ec` 0.73 against `asdm7`'s 0.75) and the default gain guard-free at every rate.
//! Measured at DSD64, −6 dBFS 1 kHz, 20 kHz band: `asdm5` 110.5 dB →
//! `asdm5ec` 116.9 dB, `asdm7` 116.8 dB → `asdm7ec` 127.3 dB, `asdm7ec8` ≈ 128 dB. At
//! DSD128 and above the in-band floor is so low that this instrument's scatter between
//! tone frequencies (±15 dB for the plain loops too) hides the difference.
//!
//! A path whose loop would overload (|w| > 2.5) is dropped instead of being reset, and
//! the reset guard engages only when every path would overload. The trellis state is a
//! transposed direct form II of `H` (so the next feedback value is just the first state
//! element). Output is delayed by `K` samples (under 3 µs at DSD64), and every input
//! sample still produces exactly one bit.

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
    /// Which modulator design this is.
    kind: DsdModulator,
    /// The noise transfer function the loop realises (shared between channels).
    ntf: Arc<Ntf>,
    /// Modulator (DSD) rate in Hz.
    rate: u32,
    /// `(A − B)[1..=N]`.
    c: Vec<f64>,
    /// `A[1..=N]`.
    a: Vec<f64>,
    /// Past errors, newest first.
    e_hist: Vec<f64>,
    /// Past feedback values, newest first.
    w_hist: Vec<f64>,
    /// Input clip level that keeps this loop stable.
    input_limit: f64,
    /// Input samples that had to be clipped to `input_limit`.
    clips: u64,
    /// Loop-state resets after an overload.
    resets: u64,
    /// Level-dependent dither, for the dithered modulator kinds.
    dither: Option<Dither>,
    /// Look-ahead search state, for the error-correction modulator kinds.
    trellis: Option<Box<Trellis>>,
    /// Bits not yet emitted as a whole byte (MSB first).
    pending: u8,
    /// How many bits of `pending` are filled (a byte is emitted at 8).
    pending_bits: u8,
    /// Samples modulated so far.
    samples: u64,
}

/// Dither that fades out with the signal level, so it only decorrelates the idle loop.
#[derive(Clone, Debug)]
struct Dither {
    /// Generator for the dither noise.
    rng: Rng,
    /// Peak-hold envelope of the input level.
    env: f64,
    /// Per-sample release factor of the envelope (about 2 ms).
    decay: f64,
}

/// Look-ahead (trellis) parameters of an error-correction modulator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lookahead {
    /// Decision depth `K`: bits are committed `K` samples after they were first
    /// considered (2…8).
    pub depth: u32,
    /// Paths kept after every step (the `N` best of the `2N` extensions; 2…64).
    pub paths: usize,
    /// Weight of the in-band (low-passed) error in the path cost against the plain
    /// `e²` term (0…10 000). More buys in-band SNR and costs stability near full scale.
    pub weight: f64,
}

impl Lookahead {
    /// Smallest and largest decision depth.
    pub const DEPTH_RANGE: std::ops::RangeInclusive<u32> = 2..=8;
    /// Smallest and largest number of survivor paths.
    pub const PATHS_RANGE: std::ops::RangeInclusive<usize> = 2..=64;

    /// `depth` bits of look-ahead with the default number of survivor paths for it
    /// (as many as the depth), clamped into the valid ranges.
    pub fn new(depth: u32) -> Self {
        Self {
            depth,
            paths: depth as usize,
            weight: DEFAULT_WEIGHT,
        }
        .clamped()
    }

    /// The same parameters forced into the valid ranges.
    pub fn clamped(self) -> Self {
        Self {
            depth: self
                .depth
                .clamp(*Self::DEPTH_RANGE.start(), *Self::DEPTH_RANGE.end()),
            paths: self
                .paths
                .clamp(*Self::PATHS_RANGE.start(), *Self::PATHS_RANGE.end()),
            weight: if self.weight.is_finite() {
                self.weight.clamp(0.0, 10_000.0)
            } else {
                DEFAULT_WEIGHT
            },
        }
    }
}

/// Corner of the first-order low-pass that weights the trellis cost towards the signal
/// band (the error's content below it is what the in-band noise is made of).
const WEIGHT_CORNER_HZ: f64 = 4_500.0;
/// Default weight of the low-passed error against the plain `e²` term (at DSD64; see
/// [`Lookahead::weight`]). Larger values buy more in-band SNR at the cost of stability
/// near full scale.
pub const DEFAULT_WEIGHT: f64 = 100.0;
/// The rate the [`Lookahead::weight`] values refer to (DSD64, 2.8224 MHz).
const WEIGHT_REFERENCE_RATE: f64 = 2_822_400.0;

/// Largest loop order the trellis state holds (the NTF design allows up to 8).
const MAX_ORDER: usize = 8;

/// One candidate path through the trellis.
#[derive(Clone, Copy, Debug)]
struct Path {
    /// Loop filter state, transposed direct form II of `H = (A − B) / A` (strictly
    /// causal, so the next feedback value is simply `z[0]`).
    z: [f64; MAX_ORDER],
    /// The error through the in-band weighting low-pass (unity DC gain).
    f: f64,
    /// Accumulated cost `Σ e² + W·f²` (relative to the best path of the
    /// previous step).
    cost: f64,
    /// Undecided bits, newest in bit 0 (`depth` of them once primed).
    bits: u32,
}

impl Path {
    /// A path with a zero loop state, no cost and no bits (the start, and after a reset).
    const CLEAN: Path = Path {
        z: [0.0; MAX_ORDER],
        f: 0.0,
        cost: 0.0,
        bits: 0,
    };
}

/// The two extensions of one surviving path: the bit the plain quantiser would pick
/// (`near`, the sign of `v`) and the other one (`far`).
#[derive(Clone, Copy, Debug, Default)]
struct Extension {
    /// Index in `cur` of the path this extends.
    parent: usize,
    /// That path's next feedback value.
    w: f64,
    /// Quantiser input `x − w`.
    v: f64,
    /// Cost of extending with the plain quantiser's bit.
    near: f64,
    /// Cost of extending with the other bit.
    far: f64,
}

/// The M-algorithm state of an error-correction modulator.
#[derive(Clone, Debug)]
struct Trellis {
    /// Search size and depth.
    look: Lookahead,
    /// `(A − B)[1..=N]` and `A[1..=N]`, zero-padded to [`MAX_ORDER`].
    c: [f64; MAX_ORDER],
    /// `A[1..=N]`, zero-padded to [`MAX_ORDER`].
    a: [f64; MAX_ORDER],
    /// Survivors, cheapest first (`count` of them are live).
    cur: Vec<Path>,
    /// Survivors being built for the next step (swapped with `cur`).
    next: Vec<Path>,
    /// Number of live paths in `cur`.
    count: usize,
    /// Both extensions of every valid survivor this step.
    ext: Vec<Extension>,
    /// Indices into `ext` sorted by `near` cost.
    near_order: Vec<usize>,
    /// Indices into `ext` of the competitive far extensions, sorted by `far` cost.
    far_order: Vec<usize>,
    /// Extensions since the last reset, saturating at `depth` (bits are committed once
    /// the paths are `depth` bits long).
    fill: u32,
    /// Pole of the in-band weighting low-pass.
    r: f64,
    /// `look.weight` scaled to this rate.
    weight: f64,
}

impl Trellis {
    /// A trellis for `look` around the loop filter `c / a`, weighting in-band error for a
    /// modulator running at `rate`.
    fn new(look: Lookahead, c: &[f64], a: &[f64], rate: u32) -> Self {
        let look = look.clamped();
        let n = look.paths;
        let mut cc = [0.0; MAX_ORDER];
        let mut aa = [0.0; MAX_ORDER];
        cc[..c.len()].copy_from_slice(c);
        aa[..a.len()].copy_from_slice(a);
        Self {
            look,
            c: cc,
            a: aa,
            cur: vec![Path::CLEAN; n],
            next: vec![Path::CLEAN; n],
            count: 1,
            ext: vec![Extension::default(); n],
            near_order: vec![0; n],
            far_order: vec![0; n],
            fill: 0,
            r: 1.0 - 2.0 * std::f64::consts::PI * WEIGHT_CORNER_HZ / rate.max(1) as f64,
            // The low-passed error's power falls with the oversampling ratio; scale the
            // weight with it so the balance between the two terms is the same at every
            // rate (the weights are specified at DSD64).
            weight: look.weight * (rate.max(1) as f64 / WEIGHT_REFERENCE_RATE),
        }
    }

    /// The cost of choosing error `e` after weighting state `f`, and the new state.
    #[inline(always)]
    fn weigh(&self, f: f64, e: f64) -> (f64, f64) {
        let f = self.r * f + (1.0 - self.r) * e;
        (e * e + self.weight * f * f, f)
    }

    /// Back to a single clean path (no bits pending).
    fn reset(&mut self) {
        self.cur[0] = Path::CLEAN;
        self.count = 1;
        self.fill = 0;
    }

    /// Sort `order[..len]` (indices into `ext`) by `key`, stable (insertion sort: the
    /// survivors arrive sorted by cost, so the keys are nearly sorted already).
    #[inline(always)]
    fn sort_by<F: Fn(&Extension) -> f64>(
        order: &mut [usize],
        ext: &[Extension],
        len: usize,
        key: F,
    ) {
        for (i, o) in order[..len].iter_mut().enumerate() {
            *o = i;
        }
        for i in 1..len {
            let cur = order[i];
            let k = key(&ext[cur]);
            let mut j = i;
            while j > 0 && key(&ext[order[j - 1]]) > k {
                order[j] = order[j - 1];
                j -= 1;
            }
            order[j] = cur;
        }
    }

    /// Extend every survivor by both bits for input `x`, keep the `N` cheapest
    /// extensions and – once the paths are `depth` bits long – commit the oldest
    /// undecided bit of the cheapest one (dropping the paths that disagree with it).
    /// Paths whose loop would overload (|w| > 2.5, or NaN) are dropped; if all of them
    /// would, the loop is reset and `resets` counts it.
    fn advance(&mut self, x: f64, resets: &mut u64) -> Option<bool> {
        let depth = self.look.depth;
        let keep = self.look.paths;
        let mut valid;
        loop {
            valid = 0;
            for (i, p) in self.cur[..self.count].iter().enumerate() {
                let w = p.z[0];
                if w.abs() > OVERLOAD || w.is_nan() {
                    continue; // this path would overload the loop (or went NaN)
                }
                let v = x - w;
                let s = if v >= 0.0 { 1.0 } else { -1.0 };
                let (en, ef) = (s - v, -s - v);
                self.ext[valid] = Extension {
                    parent: i,
                    w,
                    v,
                    near: p.cost + self.weigh(p.f, en).0,
                    far: p.cost + self.weigh(p.f, ef).0,
                };
                valid += 1;
            }
            if valid > 0 {
                break;
            }
            // Every path overloads: back to a clean loop, keeping the undecided bits of
            // the cheapest path so the output stays continuous.
            *resets += 1;
            self.cur[0] = Path {
                bits: self.cur[0].bits,
                ..Path::CLEAN
            };
            self.count = 1;
        }
        // The near extensions, sorted; a far extension (always dearer than its own near
        // one, by 4|v|) is only a candidate while it beats the dearest kept near one.
        Self::sort_by(&mut self.near_order, &self.ext, valid, |e| e.near);
        let threshold = if valid >= keep {
            self.ext[self.near_order[keep - 1]].near
        } else {
            f64::INFINITY
        };
        let mut far_count = 0;
        for i in 0..valid {
            if self.ext[i].far < threshold {
                self.far_order[far_count] = i;
                far_count += 1;
            }
        }
        {
            let (order, ext) = (&mut self.far_order[..far_count], &self.ext);
            for i in 1..order.len() {
                let cur = order[i];
                let k = ext[cur].far;
                let mut j = i;
                while j > 0 && ext[order[j - 1]].far > k {
                    order[j] = order[j - 1];
                    j -= 1;
                }
                order[j] = cur;
            }
        }

        let commit = self.fill >= depth;
        let mask = if commit {
            (1u32 << depth) - 1
        } else {
            u32::MAX
        };
        let (c, a) = (&self.c, &self.a);
        let (mut i, mut j, mut out) = (0, 0, 0);
        let mut base = 0.0;
        let mut decided: Option<u32> = None;
        while out < keep && (i < valid || j < far_count) {
            // Merge the two sorted lists; on a tie the plain quantiser's bit wins.
            let take_near = j >= far_count
                || (i < valid
                    && self.ext[self.near_order[i]].near <= self.ext[self.far_order[j]].far);
            let (ext, near) = if take_near {
                i += 1;
                (self.ext[self.near_order[i - 1]], true)
            } else {
                j += 1;
                (self.ext[self.far_order[j - 1]], false)
            };
            let bit = (ext.v >= 0.0) == near;
            let cost = if near { ext.near } else { ext.far };
            let parent = &self.cur[ext.parent];
            let bits = (parent.bits << 1) | bit as u32;
            if out == 0 {
                base = cost;
            }
            if commit {
                let b = (bits >> depth) & 1;
                match decided {
                    None => decided = Some(b),
                    Some(d) if d != b => continue,
                    Some(_) => {}
                }
            }
            // State update for the chosen error (fixed length: the padding beyond the
            // order stays zero because those coefficients are zero).
            let e = if bit { 1.0 } else { -1.0 } - ext.v;
            let mut z = [0.0; MAX_ORDER];
            for k in 0..MAX_ORDER {
                let carry = if k + 1 < MAX_ORDER {
                    parent.z[k + 1]
                } else {
                    0.0
                };
                z[k] = c[k] * e - a[k] * ext.w + carry;
            }
            self.next[out] = Path {
                z,
                f: self.weigh(parent.f, e).1,
                cost: cost - base,
                bits: bits & mask,
            };
            out += 1;
        }
        std::mem::swap(&mut self.cur, &mut self.next);
        self.count = out;
        if !commit {
            self.fill += 1;
        }
        decided.map(|b| b == 1)
    }
}

impl SigmaDelta {
    /// A modulator of `kind` running at `rate` Hz (the DSD rate).
    pub fn new(kind: DsdModulator, rate: u32) -> Self {
        let ntf = Arc::new(Ntf::for_rate(
            kind.order(),
            rate,
            NTF_BAND_HZ,
            kind.h_inf(super::oversampling_ratio(rate)),
        ));
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
        assert!(n <= MAX_ORDER, "loop order {n} exceeds {MAX_ORDER}");
        let c = ntf.feedback_numerator();
        let a = ntf.a[1..=n].to_vec();
        let trellis = kind
            .lookahead()
            .map(|l| Box::new(Trellis::new(l, &c, &a, rate)));
        let mut m = Self {
            kind,
            c,
            a,
            e_hist: vec![0.0; n],
            w_hist: vec![0.0; n],
            input_limit: kind.input_limit(),
            clips: 0,
            resets: 0,
            dither,
            trellis,
            pending: 0,
            pending_bits: 0,
            samples: 0,
            ntf,
            rate,
        };
        m.prime();
        m
    }

    /// Replace the look-ahead parameters of an error-correction modulator (`K` = 2…8,
    /// any number of paths 2…64; out-of-range values are clamped). Turns a plain
    /// modulator into a trellis one as well. Resets the loop state.
    pub fn set_lookahead(&mut self, look: Lookahead) {
        self.trellis = Some(Box::new(Trellis::new(look, &self.c, &self.a, self.rate)));
        self.reset();
    }

    /// The look-ahead parameters (`None` for the plain quantisers).
    pub fn lookahead(&self) -> Option<Lookahead> {
        self.trellis.as_ref().map(|t| t.look)
    }

    /// Output delay in samples (the trellis decision depth; 0 for the plain loops).
    pub fn delay_samples(&self) -> u32 {
        self.lookahead().map_or(0, |l| l.depth)
    }

    /// Fill the trellis with `K` samples of silence so that every later input sample
    /// yields exactly one committed bit (the output is delayed by `K` samples).
    fn prime(&mut self) {
        if let Some(t) = &mut self.trellis {
            t.reset();
            for _ in 0..t.look.depth {
                t.advance(0.0, &mut self.resets);
            }
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
        self.prime();
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
        if let Some(t) = &mut self.trellis {
            return t.advance(x, &mut self.resets).unwrap_or(false);
        }
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
                assert!(
                    ones > 0 && ones < chunk.len() as u32 * 8,
                    "{}: stuck output",
                    kind.id()
                );
            }
            assert!(m.w_hist.iter().all(|w| w.is_finite()));
            assert!(
                m.clips() > 0,
                "{}: the clipper must be what keeps −0.3 dBFS stable",
                kind.id()
            );
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
    fn stable_at_the_default_gain_on_dc_and_tones_at_every_rate() {
        // Sustained DC is the worst case for a 1-bit loop: a second of ±0.708 (0 dBFS at
        // the default −3 dB gain) plus a bass, a mid and a top-of-band tone at that level
        // must engage neither guard at any rate in either family (~100 s in release for
        // the plain loops: the rates up to DSD2048 add up to 4 G modulator samples; the
        // `ec` loops run quarter-second segments, `asdm7ec8` only up to DSD256).
        const LEVEL: f64 = 0.708;
        for kind in DsdModulator::catalog() {
            assert!(
                kind.input_limit() > LEVEL,
                "{}: default gain must not clip",
                kind.id()
            );
            for rate in DsdRate::catalog() {
                if kind.max_rate().is_some_and(|max| *rate > max) {
                    continue;
                }
                for base in [44_100, 48_000] {
                    let hz = rate.hz(base);
                    // The trellis loops are 5–20× slower: a quarter second per segment
                    // (still ≥ 700 k samples at DSD64, far beyond any wind-up time).
                    let second = if kind.error_correction() {
                        hz as usize / 4
                    } else {
                        hz as usize
                    };
                    let mut m = SigmaDelta::new(*kind, hz);
                    let mut bits = Vec::new();
                    m.process(&vec![LEVEL; second], &mut bits);
                    m.process(&vec![-LEVEL; second], &mut bits);
                    for f in [1000.0, 20_000.0, 40.0] {
                        m.process(&tone(LEVEL, f, hz, second / 8), &mut bits);
                    }
                    assert_eq!(
                        m.guard_engagements(),
                        0,
                        "{} at {} ({base} family): {} resets, {} clips at the default gain",
                        kind.id(),
                        rate.label(),
                        m.resets(),
                        m.clips()
                    );
                }
            }
        }
    }

    #[test]
    fn guards_are_counted_and_default_gain_avoids_them() {
        let rate = DsdRate::Dsd64.hz(44_100);
        let mut m = SigmaDelta::new(DsdModulator::Asdm7, rate);
        let mut bits = Vec::new();
        m.process(&tone(0.708, 1000.0, rate, rate as usize / 10), &mut bits);
        assert_eq!(
            m.guard_engagements(),
            0,
            "0 dBFS − 3 dB must not engage any guard"
        );
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

    #[test]
    fn error_correction_gains_at_least_3_db_over_the_plain_loop_at_dsd64() {
        let asdm5 = snr_at_dsd64(DsdModulator::Asdm5, 0.5);
        let asdm7 = snr_at_dsd64(DsdModulator::Asdm7, 0.5);
        for (ec, plain, name) in [
            (DsdModulator::Asdm5ec, asdm5, "asdm5"),
            (DsdModulator::Asdm7ec, asdm7, "asdm7"),
            (DsdModulator::Asdm7ec8, asdm7, "asdm7"),
        ] {
            let snr = snr_at_dsd64(ec, 0.5);
            println!("{}: {snr:.1} dB vs {name} {plain:.1} dB", ec.id());
            assert!(
                snr >= plain + 3.0,
                "{}: {snr:.1} dB, {name}: {plain:.1} dB",
                ec.id()
            );
        }
    }

    #[test]
    fn error_correction_survives_0_95_fs_and_recovers() {
        let rate = DsdRate::Dsd64.hz(44_100);
        for kind in [
            DsdModulator::Asdm5ec,
            DsdModulator::Asdm7ec,
            DsdModulator::Asdm7ec8,
        ] {
            let mut m = SigmaDelta::new(kind, rate);
            let mut bits = Vec::new();
            // 250 ms of a 0.95 FS tone, then 250 ms of 0.95 FS DC.
            m.process(&tone(0.95, 1000.0, rate, rate as usize / 4), &mut bits);
            m.process(&vec![0.95; rate as usize / 4], &mut bits);
            let window = rate as usize / 8000;
            for chunk in bits[bits.len() / 4..].chunks(window) {
                let ones: u32 = chunk.iter().map(|b| b.count_ones()).sum();
                assert!(
                    ones > 0 && ones < chunk.len() as u32 * 8,
                    "{}: stuck output",
                    kind.id()
                );
            }
            assert!(m.clips() > 0, "{}: the clipper holds 0.95 FS", kind.id());
            let t = m.trellis.as_ref().unwrap();
            assert!(t.cur[..t.count]
                .iter()
                .all(|p| p.z.iter().all(|z| z.is_finite()) && p.cost.is_finite()));
            // Back to −6 dBFS: the loop recovered and keeps the plain loop's SNR floor.
            let pcm_rate = rate / 8;
            let nfft = 1 << 16;
            let hz = coherent_tone_hz(1000.0, pcm_rate, nfft);
            let mut bits = Vec::new();
            m.process(&tone(0.5, hz, rate, (nfft + 4096) * 8), &mut bits);
            let snr = tone_snr_db(&bits, rate, hz, 20_000.0, nfft).0;
            assert!(snr > 110.0, "{} after overload: SNR {snr:.1} dB", kind.id());
        }
    }

    #[test]
    fn error_correction_is_deterministic_and_reset_restarts_it() {
        let rate = DsdRate::Dsd64.hz(48_000);
        let input = tone(0.6, 997.0, rate, 200_000);
        for kind in [DsdModulator::Asdm5ec, DsdModulator::Asdm7ec8] {
            let run = |m: &mut SigmaDelta| {
                let mut out = Vec::new();
                // Odd block sizes: partial bytes carry over between calls.
                for chunk in input.chunks(4093) {
                    m.process(chunk, &mut out);
                }
                m.complete_byte(&mut out);
                out
            };
            let mut a = SigmaDelta::new(kind, rate);
            let mut b = SigmaDelta::new(kind, rate);
            let first = run(&mut a);
            assert_eq!(first, run(&mut b), "{}: two instances agree", kind.id());
            a.reset();
            assert_eq!(first, run(&mut a), "{}: reset restarts", kind.id());
            // One bit per input sample, the K-sample delay notwithstanding.
            assert_eq!(first.len(), input.len().div_ceil(8));
            assert_eq!(a.delay_samples(), kind.lookahead().unwrap().depth);
            assert_eq!(a.guard_engagements(), 0);
        }
    }

    #[test]
    fn lookahead_parameters_are_clamped_and_settable() {
        let l = Lookahead::new(4);
        assert_eq!((l.depth, l.paths, l.weight), (4, 4, DEFAULT_WEIGHT));
        assert_eq!(Lookahead::new(1).depth, 2);
        assert_eq!(Lookahead::new(12).depth, 8);
        let wild = Lookahead {
            depth: 0,
            paths: 1000,
            weight: f64::NAN,
        }
        .clamped();
        assert_eq!(
            (wild.depth, wild.paths, wild.weight),
            (2, 64, DEFAULT_WEIGHT)
        );
        assert_eq!(DsdModulator::Asdm7ec.lookahead().unwrap().depth, 4);
        assert_eq!(DsdModulator::Asdm7ec8.lookahead().unwrap().depth, 8);
        assert!(DsdModulator::Asdm7.lookahead().is_none());
        // Every depth 2…8 runs (here on a plain loop turned into a trellis one).
        let rate = DsdRate::Dsd64.hz(44_100);
        for depth in Lookahead::DEPTH_RANGE {
            let mut m = SigmaDelta::new(DsdModulator::Asdm7, rate);
            assert_eq!(m.delay_samples(), 0);
            m.set_lookahead(Lookahead::new(depth));
            assert_eq!(m.delay_samples(), depth);
            let mut out = Vec::new();
            m.process(&tone(0.5, 1000.0, rate, 80_000), &mut out);
            assert_eq!(out.len(), 10_000);
            let ones: u32 = out.iter().map(|b| b.count_ones()).sum();
            assert!((ones as f64 / 80_000.0 - 0.5).abs() < 0.02, "depth {depth}");
        }
    }

    #[test]
    fn error_correction_at_dsd256_timing() {
        // No hard timing assert (CI machines vary): report the realtime factor of 0.1 s
        // of DSD256 per modulator, one channel.
        let rate = DsdRate::Dsd256.hz(44_100);
        let input = tone(0.5, 1000.0, rate, rate as usize / 10);
        for kind in [
            DsdModulator::Asdm7,
            DsdModulator::Asdm5ec,
            DsdModulator::Asdm7ec,
            DsdModulator::Asdm7ec8,
        ] {
            let mut m = SigmaDelta::new(kind, rate);
            let mut out = Vec::new();
            let t = std::time::Instant::now();
            m.process(&input, &mut out);
            let secs = t.elapsed().as_secs_f64();
            println!(
                "{} at DSD256: 0.1 s of audio in {secs:.3} s ({:.2}× realtime, one channel)",
                kind.id(),
                0.1 / secs
            );
            assert_eq!(out.len(), input.len() / 8);
            assert_eq!(m.guard_engagements(), 0);
        }
    }
    /// FNV-1a over the bit stream of every error-correction configuration on a tone, DC,
    /// a full-scale step and noise: the trellis must produce exactly these bits (the
    /// optimised search is a rewrite of the same algorithm, not a new one).
    fn trellis_digest(kind: DsdModulator, rate: u32) -> u64 {
        let n = 200_000;
        let mut rng = crate::rng::Rng::new(7);
        let mut x: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 997.0 * i as f64 / rate as f64).sin())
            .collect();
        x.extend(std::iter::repeat_n(0.7, n / 4));
        x.extend(std::iter::repeat_n(-0.75, n / 4));
        x.extend((0..n / 2).map(|_| 0.3 * rng.centred()));
        x.extend(std::iter::repeat_n(0.0, n / 4));
        let mut m = SigmaDelta::new(kind, rate);
        let mut bits = Vec::new();
        m.process(&x, &mut bits);
        m.complete_byte(&mut bits);
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in bits.iter().chain(&m.resets().to_le_bytes()) {
            h = (h ^ *b as u64).wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    #[test]
    fn trellis_output_is_bit_identical_to_the_reference_search() {
        // Recorded from the first (straightforward) implementation of the search.
        for (kind, rate, want) in [
            (DsdModulator::Asdm5ec, 2_822_400, 0xce74_2a3a_b4a8_3876u64),
            (DsdModulator::Asdm5ec, 11_289_600, 0x51a4_5e90_ff23_c39c),
            (DsdModulator::Asdm5ec, 24_576_000, 0x08d9_59cf_181c_090d),
            (DsdModulator::Asdm7ec, 2_822_400, 0xde87_ce8a_f90c_f563),
            (DsdModulator::Asdm7ec, 11_289_600, 0x4973_7ec9_f414_f770),
            (DsdModulator::Asdm7ec, 24_576_000, 0x0803_6e2f_2d12_a088),
            (DsdModulator::Asdm7ec8, 2_822_400, 0x2bda_c2a6_20af_3487),
            (DsdModulator::Asdm7ec8, 11_289_600, 0x7c98_e269_e19f_aaa7),
            (DsdModulator::Asdm7ec8, 24_576_000, 0x5589_5761_beee_f990),
        ] {
            assert_eq!(trellis_digest(kind, rate), want, "{kind:?} at {rate} Hz");
        }
    }
}
