//! Sample-rate conversion with very long FIR filters.
//!
//! * Integer upsampling (×2 … ×64 … ×2048, any integer) and small-ratio conversions run the
//!   polyphase decomposition of the long FIR through the partitioned FFT engine of
//!   [`crate::convolution`]: the input block is transformed once and every one of the `L`
//!   sub-filters reads the same frequency-domain delay line. The `L` phases are evaluated
//!   in parallel on the rayon pool, as are the channels.
//! * Rational conversions with a large `L` (44.1 ↔ 48 kHz families: 147/160, 320/147 …)
//!   use a time-domain polyphase bank, because computing 160 phases to keep one in 147
//!   would waste the FFT's advantage.
//! * Cross-family conversions to a high rate (44.1 kHz → 1.536 MHz is ×5120/147) are
//!   cascaded: a time-domain stage does the rational part to the first rate of the target
//!   family (×160/147 → 48 kHz), then the FFT stage does the integer rest (×32). Each stage
//!   runs the chosen filter at its own ratio; the first stage's filter defines the passband
//!   and the second one's transition lies entirely above it.
//!
//! The output is **time aligned** with the input: the linear-phase group delay is absorbed
//! (the first `(N−1)/2` filter outputs are skipped and `flush` pushes the tail out), so a
//! stream of `n` input frames gives exactly `ceil(n·L/M)` output frames. The processing
//! latency a player must account for is reported by [`Resampler::latency_input_frames`].

use crate::cache::CoefficientCache;
use crate::convolution::{choose_block, BlockEngine, OutputScratch, SpectralFilter};
use crate::fir::{FilterKind, FirSpec, Phase};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Longest polyphase sub-filter the time-domain core accepts (per output sample cost).
const MAX_TIME_DOMAIN_SUBFILTER: usize = 8192;

/// Below this many complex MACs per block the `L` phases are evaluated serially.
const PARALLEL_PHASES_THRESHOLD: usize = 1 << 16;

/// Highest PCM rate a device can be driven at: 32 × 48 kHz. Higher "PCM" rates exist only
/// inside the DSD modulator path (and are never written to a device).
pub const MAX_PCM_RATE: u32 = 1_536_000;

/// PCM rates used as the sigma-delta modulator input (64 × 44.1 / 48 kHz): offered only
/// when the output backend reports support for them.
pub const DSD_MODULATOR_PCM_RATES: [u32; 2] = [2_822_400, 3_072_000];

/// The two base-rate families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RateFamily {
    /// 44.1 kHz × 2^k (CD, SACD-derived, DSD64 = 2.8224 MHz).
    F44,
    /// 48 kHz × 2^k (video, pro audio, 3.072 MHz DSD).
    F48,
}

impl RateFamily {
    /// Base rate in Hz.
    pub fn base(self) -> u32 {
        match self {
            RateFamily::F44 => 44_100,
            RateFamily::F48 => 48_000,
        }
    }

    /// `rate = base × 2^k` for this family (`k` is returned), if it is.
    pub fn multiple(self, rate: u32) -> Option<u32> {
        let b = self.base();
        if rate == 0 || rate % b != 0 {
            return None;
        }
        let q = rate / b;
        q.is_power_of_two().then(|| q.trailing_zeros())
    }

    /// The family a rate belongs to (`None` for odd rates like 8 kHz or 22.05 kHz is `F44`
    /// only when it is a power-of-two multiple).
    pub fn of(rate: u32) -> Option<RateFamily> {
        [RateFamily::F44, RateFamily::F48]
            .into_iter()
            .find(|f| f.multiple(rate).is_some())
    }

    /// Every rate of this family up to `max` (inclusive), ascending.
    pub fn rates_up_to(self, max: u32) -> Vec<u32> {
        (0..)
            .map(|k| self.base() << k)
            .take_while(|r| *r <= max)
            .collect()
    }
}

/// Every PCM output rate Cadenza offers: both families from the base rate to
/// [`MAX_PCM_RATE`] (1.4112 and 1.536 MHz), ascending.
pub fn pcm_output_rates() -> Vec<u32> {
    pcm_output_rates_up_to(MAX_PCM_RATE, false)
}

/// PCM output rates up to `max`, optionally including the [`DSD_MODULATOR_PCM_RATES`]
/// (only when the backend reports they can be opened).
pub fn pcm_output_rates_up_to(max: u32, include_modulator_rates: bool) -> Vec<u32> {
    let mut rates: Vec<u32> = RateFamily::F44
        .rates_up_to(max.min(MAX_PCM_RATE))
        .into_iter()
        .chain(RateFamily::F48.rates_up_to(max.min(MAX_PCM_RATE)))
        .collect();
    if include_modulator_rates {
        rates.extend(DSD_MODULATOR_PCM_RATES.iter().copied().filter(|r| *r <= max));
    }
    rates.sort_unstable();
    rates.dedup();
    rates
}

/// The highest rate of `in_rate`'s own family that is `≤ max` (an integer ratio, so the
/// FFT polyphase engine alone handles it); falls back to the highest offered rate `≤ max`
/// for sources outside both families.
pub fn preferred_output_rate(in_rate: u32, max: u32) -> u32 {
    match RateFamily::of(in_rate) {
        Some(f) => f
            .rates_up_to(max)
            .last()
            .copied()
            .unwrap_or(in_rate),
        None => pcm_output_rates_up_to(max, false)
            .last()
            .copied()
            .unwrap_or(in_rate),
    }
}

/// Errors from [`Resampler::new`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResampleError {
    /// Input and output rates are equal; nothing to do.
    #[error("input and output rate are both {0} Hz")]
    SameRate(u32),
    /// A rate or channel count of zero.
    #[error("invalid parameters: {0}")]
    Invalid(&'static str),
}

/// Which engine a resampler stage uses (reported in the signal path).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Engine {
    /// Partitioned FFT polyphase (block size in samples).
    FftPolyphase {
        /// Input block size in frames.
        block: usize,
    },
    /// Direct time-domain polyphase bank.
    TimePolyphase,
}

impl Engine {
    /// Short description for the signal path.
    pub fn label(self) -> String {
        match self {
            Engine::FftPolyphase { block } => format!("FFT polyphase · block {block}"),
            Engine::TimePolyphase => "time-domain polyphase".into(),
        }
    }
}

/// One stage of a (possibly cascaded) resampler, for reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageInfo {
    /// Stage input rate.
    pub in_rate: u32,
    /// Stage output rate.
    pub out_rate: u32,
    /// Upsampling factor.
    pub l: usize,
    /// Decimation factor.
    pub m: usize,
    /// Filter design of the stage.
    pub spec: FirSpec,
    /// Engine of the stage.
    pub engine: Engine,
    /// Stage latency in its own input frames.
    pub latency_input_frames: usize,
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Decompose `h` (designed at `l × input rate`) into `l` sub-filters, each scaled by `l` to
/// make up for the zero stuffing, optionally reversed for straight dot products.
fn polyphase(h: &[f64], l: usize, reversed: bool) -> Vec<Vec<f64>> {
    let sub_len = h.len().div_ceil(l);
    (0..l)
        .into_par_iter()
        .map(|p| {
            let mut sub: Vec<f64> = (0..sub_len)
                .map(|k| h.get(k * l + p).copied().unwrap_or(0.0) * l as f64)
                .collect();
            if reversed {
                sub.reverse();
            }
            sub
        })
        .collect()
}

/// Per-channel state of the FFT engine.
struct FftCore {
    engine: BlockEngine,
    filters: Arc<Vec<SpectralFilter>>,
    /// One inverse-FFT scratch per parallel worker (`phase_out` is split into that many chunks).
    scratch: Vec<OutputScratch>,
    parallel: bool,
    m: usize,
    /// Index at the `l × fs_in` rate of the next upsampled sample the engine will produce.
    u: u64,
    /// The upsampled index of the next sample to keep (decimation by `m`, offset by the delay).
    next_take: u64,
    phase_out: Vec<Vec<f64>>,
}

impl FftCore {
    fn push(&mut self, input: &[f64], out: &mut Vec<f64>) {
        let Self {
            engine,
            filters,
            scratch,
            parallel,
            m,
            u,
            next_take,
            phase_out,
        } = self;
        let m = *m;
        engine.push(input, |e| {
            if *parallel {
                let chunk = phase_out.len().div_ceil(scratch.len());
                phase_out
                    .par_chunks_mut(chunk)
                    .zip(scratch.par_iter_mut())
                    .enumerate()
                    .for_each(|(i, (bufs, scratch))| {
                        for (j, buf) in bufs.iter_mut().enumerate() {
                            buf.clear();
                            e.output_with(&filters[i * chunk + j], scratch, buf);
                        }
                    });
            } else {
                for (p, buf) in phase_out.iter_mut().enumerate() {
                    buf.clear();
                    e.output(&filters[p], buf);
                }
            }
            let block = e.block();
            if m == 1 {
                // Plain upsampling: interleave the phases, skipping the group delay once.
                for j in 0..block {
                    for buf in phase_out.iter() {
                        if *u >= *next_take {
                            out.push(buf[j]);
                        }
                        *u += 1;
                    }
                }
                *next_take = (*next_take).max(*u);
            } else {
                for j in 0..block {
                    for buf in phase_out.iter() {
                        if *u == *next_take {
                            out.push(buf[j]);
                            *next_take += m as u64;
                        }
                        *u += 1;
                    }
                }
            }
        });
    }

    fn reset(&mut self, delay: u64) {
        self.engine.reset();
        self.u = 0;
        self.next_take = delay;
    }
}

/// Per-channel state of the time-domain polyphase engine.
struct TimeCore {
    /// Reversed sub-filters: `sub[p][k'] = h[(S−1−k')·l + p] · l`.
    sub: Arc<Vec<Vec<f64>>>,
    l: u64,
    m: u64,
    sub_len: usize,
    /// Input history; `buf[i]` is input sample `base + i`. Starts with `S−1` zeros.
    buf: Vec<f64>,
    base: i64,
    /// Upsampled-rate index of the next output sample.
    t: u64,
}

impl TimeCore {
    fn new(sub: Arc<Vec<Vec<f64>>>, l: usize, m: usize, delay: u64) -> Self {
        let sub_len = sub[0].len();
        Self {
            sub,
            l: l as u64,
            m: m as u64,
            sub_len,
            buf: vec![0.0; sub_len - 1],
            base: -(sub_len as i64 - 1),
            t: delay,
        }
    }

    fn push(&mut self, input: &[f64], out: &mut Vec<f64>) {
        self.buf.extend_from_slice(input);
        let s = self.sub_len;
        loop {
            let n = (self.t / self.l) as i64;
            let idx = (n - self.base) as usize; // position of x[n] in buf
            if idx >= self.buf.len() {
                break;
            }
            let p = (self.t % self.l) as usize;
            let window = &self.buf[idx + 1 - s..=idx];
            let y: f64 = dot(&self.sub[p], window);
            out.push(y);
            self.t += self.m;
        }
        // Keep only the history the next output can still reach.
        let next_n = (self.t / self.l) as i64;
        let keep_from = (next_n - self.base) as usize;
        let drop = keep_from.saturating_sub(s - 1).min(self.buf.len());
        if drop > 0 {
            self.buf.drain(..drop);
            self.base += drop as i64;
        }
    }

    fn reset(&mut self, delay: u64) {
        self.buf.clear();
        self.buf.resize(self.sub_len - 1, 0.0);
        self.base = -(self.sub_len as i64 - 1);
        self.t = delay;
    }
}

/// Dot product in four independent accumulators (lets the compiler vectorise without
/// reassociating a single serial sum).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f64; 4];
    let chunks = n / 4;
    for i in 0..chunks {
        let k = i * 4;
        acc[0] += a[k] * b[k];
        acc[1] += a[k + 1] * b[k + 1];
        acc[2] += a[k + 2] * b[k + 2];
        acc[3] += a[k + 3] * b[k + 3];
    }
    let mut tail = 0.0;
    for k in chunks * 4..n {
        tail += a[k] * b[k];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

enum Core {
    Fft(Box<FftCore>),
    Time(TimeCore),
}

impl Core {
    fn push(&mut self, input: &[f64], out: &mut Vec<f64>) {
        match self {
            Core::Fft(c) => c.push(input, out),
            Core::Time(c) => c.push(input, out),
        }
    }
    fn reset(&mut self, delay: u64) {
        match self {
            Core::Fft(c) => c.reset(delay),
            Core::Time(c) => c.reset(delay),
        }
    }
}

/// One `L/M` conversion with one filter.
struct Stage {
    in_rate: u32,
    out_rate: u32,
    channels: usize,
    l: usize,
    m: usize,
    spec: FirSpec,
    engine: Engine,
    /// Group delay of the filter in upsampled samples (`(N−1)/2` for linear phase, 0 otherwise).
    delay: u64,
    cores: Vec<Core>,
    input_frames: u64,
    output_frames: u64,
    split: Vec<Vec<f64>>,
    joined: Vec<Vec<f64>>,
}

impl Stage {
    fn new(in_rate: u32, out_rate: u32, channels: usize, spec: FirSpec, h: &[f64]) -> Self {
        let g = gcd(in_rate, out_rate);
        let (l, m) = ((out_rate / g) as usize, (in_rate / g) as usize);
        let delay = match spec.phase {
            Phase::Linear => ((h.len() - 1) / 2) as u64,
            Phase::Minimum => 0,
        };
        let (engine, cores) = if use_fft(l, m) {
            let sub_len = h.len().div_ceil(l);
            let block = choose_block(sub_len);
            let filters: Arc<Vec<SpectralFilter>> = Arc::new(
                polyphase(h, l, false)
                    .par_iter()
                    .map(|s| SpectralFilter::new(s, block))
                    .collect(),
            );
            let depth = filters[0].partitions();
            let work = depth * (block + 1) * l;
            let threads = rayon::current_num_threads();
            let parallel = l > 1 && threads > 1 && work >= PARALLEL_PHASES_THRESHOLD;
            let workers = if parallel { (2 * threads).min(l) } else { 0 };
            let cores = (0..channels)
                .map(|_| {
                    let engine = BlockEngine::new(block, depth);
                    Core::Fft(Box::new(FftCore {
                        scratch: (0..workers).map(|_| engine.scratch()).collect(),
                        engine,
                        filters: Arc::clone(&filters),
                        parallel,
                        m,
                        u: 0,
                        next_take: delay,
                        phase_out: vec![Vec::with_capacity(block); l],
                    }))
                })
                .collect();
            (Engine::FftPolyphase { block }, cores)
        } else {
            let sub = Arc::new(polyphase(h, l, true));
            let cores = (0..channels)
                .map(|_| Core::Time(TimeCore::new(Arc::clone(&sub), l, m, delay)))
                .collect();
            (Engine::TimePolyphase, cores)
        };
        Self {
            in_rate,
            out_rate,
            channels,
            l,
            m,
            spec,
            engine,
            delay,
            cores,
            input_frames: 0,
            output_frames: 0,
            split: vec![Vec::new(); channels],
            joined: vec![Vec::new(); channels],
        }
    }

    fn latency_input_frames(&self) -> usize {
        let look_ahead = (self.delay as usize).div_ceil(self.l);
        let block = match self.engine {
            Engine::FftPolyphase { block } => block,
            Engine::TimePolyphase => 0,
        };
        look_ahead + block
    }

    fn info(&self) -> StageInfo {
        StageInfo {
            in_rate: self.in_rate,
            out_rate: self.out_rate,
            l: self.l,
            m: self.m,
            spec: self.spec.clone(),
            engine: self.engine,
            latency_input_frames: self.latency_input_frames(),
        }
    }

    fn push(&mut self, interleaved: &[f64]) -> Vec<f64> {
        let ch = self.channels;
        let frames = interleaved.len() / ch;
        self.input_frames += frames as u64;
        for (c, buf) in self.split.iter_mut().enumerate() {
            buf.clear();
            buf.extend(interleaved.iter().skip(c).step_by(ch).take(frames));
        }
        if ch > 1 {
            self.cores
                .par_iter_mut()
                .zip(self.split.par_iter().zip(self.joined.par_iter_mut()))
                .for_each(|(core, (inp, out))| {
                    out.clear();
                    core.push(inp, out);
                });
        } else {
            let out = &mut self.joined[0];
            out.clear();
            self.cores[0].push(&self.split[0], out);
        }
        self.interleave()
    }

    fn flush(&mut self) -> Vec<f64> {
        let expected = (self.input_frames * self.l as u64).div_ceil(self.m as u64);
        let step = match self.engine {
            Engine::FftPolyphase { block } => block,
            Engine::TimePolyphase => 256,
        };
        let zeros = vec![0.0; step * self.channels];
        let mut out = Vec::new();
        let mut guard = 0;
        while self.output_frames < expected {
            let before = self.input_frames;
            out.extend(self.push(&zeros));
            self.input_frames = before;
            guard += 1;
            assert!(guard < 1_000_000, "resampler flush did not converge");
        }
        let extra = (self.output_frames - expected) as usize;
        out.truncate(out.len() - extra * self.channels);
        self.reset();
        out
    }

    fn reset(&mut self) {
        for c in &mut self.cores {
            c.reset(self.delay);
        }
        self.input_frames = 0;
        self.output_frames = 0;
    }

    fn interleave(&mut self) -> Vec<f64> {
        let n = self.joined[0].len();
        debug_assert!(self.joined.iter().all(|j| j.len() == n));
        self.output_frames += n as u64;
        let mut out = vec![0.0; n * self.channels];
        for (c, buf) in self.joined.iter().enumerate() {
            for (j, v) in buf.iter().enumerate() {
                out[j * self.channels + c] = *v;
            }
        }
        out
    }
}

fn use_fft(l: usize, m: usize) -> bool {
    m == 1 || l <= 4
}

/// For a rational `l/m` with a large `l`, the smallest divisor `l1 ≥ m` of `l` (so the first
/// stage never decimates) that leaves an integer `l / l1 ≥ 2` for an FFT stage.
fn cascade_split(l: usize, m: usize) -> Option<usize> {
    if use_fft(l, m) {
        return None;
    }
    (m.max(2)..=l / 2)
        .find(|l1| l % l1 == 0)
        .filter(|l1| l / l1 >= 2)
}

/// A streaming, gapless, multi-channel sample-rate converter.
pub struct Resampler {
    in_rate: u32,
    out_rate: u32,
    channels: usize,
    l: usize,
    m: usize,
    stages: Vec<Stage>,
    input_frames: u64,
    output_frames: u64,
}

impl Resampler {
    /// Build a converter from `in_rate` to `out_rate` for `channels` interleaved channels
    /// with the catalogue filter `kind`; coefficients come from (and go to) `cache`.
    pub fn new(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        kind: FilterKind,
        attenuation_db: f64,
        cache: &CoefficientCache,
    ) -> Result<Self, ResampleError> {
        if in_rate == 0 || out_rate == 0 {
            return Err(ResampleError::Invalid("sample rate is zero"));
        }
        if channels == 0 {
            return Err(ResampleError::Invalid("channel count is zero"));
        }
        if in_rate == out_rate {
            return Err(ResampleError::SameRate(in_rate));
        }
        let g = gcd(in_rate, out_rate);
        let (l, m) = ((out_rate / g) as usize, (in_rate / g) as usize);

        let stages = match cascade_split(l, m) {
            Some(l1) => {
                let mid_rate = (in_rate as u64 * l1 as u64 / m as u64) as u32;
                let first = Self::stage(in_rate, mid_rate, channels, kind, attenuation_db, cache);
                // The second stage must pass everything the first one produced, so it is
                // never apodizing; its transition lies above the source band.
                let second_kind = FilterKind {
                    apodizing: false,
                    ..kind
                };
                let second = Self::stage(mid_rate, out_rate, channels, second_kind, attenuation_db, cache);
                vec![first, second]
            }
            None => vec![Self::stage(in_rate, out_rate, channels, kind, attenuation_db, cache)],
        };
        Ok(Self {
            in_rate,
            out_rate,
            channels,
            l,
            m,
            stages,
            input_frames: 0,
            output_frames: 0,
        })
    }

    fn stage(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        kind: FilterKind,
        attenuation_db: f64,
        cache: &CoefficientCache,
    ) -> Stage {
        let g = gcd(in_rate, out_rate);
        let (l, m) = ((out_rate / g) as usize, (in_rate / g) as usize);
        let mut spec = kind.spec_for_ratio(l, m, attenuation_db);
        if !use_fft(l, m) && spec.taps.div_ceil(l) > MAX_TIME_DOMAIN_SUBFILTER {
            // A 4M-tap bank at L=160 would cost 26k multiplies per output sample; cap it.
            let capped = FirSpec::with_window(
                MAX_TIME_DOMAIN_SUBFILTER * l,
                spec.cutoff,
                spec.attenuation_db,
                spec.phase,
                spec.window,
            );
            log::warn!(
                "{} is too long for the {l}/{m} time-domain converter; using {} taps",
                spec.taps,
                capped.taps
            );
            spec = capped;
        }
        let h = cache.get_or_design(&spec);
        Stage::new(in_rate, out_rate, channels, spec, &h)
    }

    /// Build a single-stage converter around an explicit linear- or minimum-phase low-pass
    /// `h` designed for `spec` (the filter must run at `L × in_rate` with its cutoff at the
    /// lower Nyquist).
    pub fn with_coefficients(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        spec: FirSpec,
        h: &[f64],
    ) -> Self {
        let g = gcd(in_rate, out_rate);
        let (l, m) = ((out_rate / g) as usize, (in_rate / g) as usize);
        Self {
            in_rate,
            out_rate,
            channels,
            l,
            m,
            stages: vec![Stage::new(in_rate, out_rate, channels, spec, h)],
            input_frames: 0,
            output_frames: 0,
        }
    }

    /// Input sample rate.
    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }
    /// Output sample rate.
    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }
    /// Channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }
    /// The overall `L/M` ratio.
    pub fn ratio(&self) -> (usize, usize) {
        (self.l, self.m)
    }
    /// The filter that defines the passband (the first stage's).
    pub fn spec(&self) -> &FirSpec {
        &self.stages[0].spec
    }
    /// Which engine runs the (first stage's) filter.
    pub fn engine(&self) -> Engine {
        self.stages[0].engine
    }
    /// Every stage, in order (one for integer and small rational ratios, two for cascades).
    pub fn stages(&self) -> Vec<StageInfo> {
        self.stages.iter().map(Stage::info).collect()
    }
    /// Total taps across the stages.
    pub fn total_taps(&self) -> usize {
        self.stages.iter().map(|s| s.spec.taps).sum()
    }

    /// `×4 · FFT polyphase · block 1024` or `×160/147 time-domain polyphase → ×32 FFT polyphase · block 4096`.
    pub fn describe(&self) -> String {
        self.stages
            .iter()
            .map(|s| {
                let ratio = if s.m == 1 {
                    format!("×{}", s.l)
                } else {
                    format!("×{}/{}", s.l, s.m)
                };
                format!("{ratio} {}", s.engine.label())
            })
            .collect::<Vec<_>>()
            .join(" → ")
    }

    /// Input frames that must arrive before the output for a given instant exists: the
    /// filters' look-ahead plus block buffering. Convert with `in_rate` to seconds.
    pub fn latency_input_frames(&self) -> usize {
        let mut total = 0usize;
        // Later stages' latency is expressed in their own input frames; scale it back.
        let mut scale = 1.0f64;
        for s in &self.stages {
            total += (s.latency_input_frames() as f64 * scale).ceil() as usize;
            scale *= s.m as f64 / s.l as f64;
        }
        total
    }

    /// Latency in seconds (see [`Resampler::latency_input_frames`]).
    pub fn latency_seconds(&self) -> f64 {
        self.latency_input_frames() as f64 / self.in_rate as f64
    }

    /// Output frames produced so far for the input frames pushed so far (gapless: the
    /// count never includes filter tails).
    pub fn frames(&self) -> (u64, u64) {
        (self.input_frames, self.output_frames)
    }

    /// Convert interleaved frames; returns whatever output is ready (possibly nothing, or
    /// several blocks). Call [`Resampler::flush`] at the end of the stream.
    pub fn push(&mut self, interleaved: &[f64]) -> Vec<f64> {
        self.input_frames += (interleaved.len() / self.channels) as u64;
        let mut buf = self.stages[0].push(interleaved);
        for s in &mut self.stages[1..] {
            buf = s.push(&buf);
        }
        self.output_frames += (buf.len() / self.channels) as u64;
        buf
    }

    /// Push silence until every input frame has produced its output, then reset the
    /// filter state. Returns the remaining output.
    pub fn flush(&mut self) -> Vec<f64> {
        let expected = (self.input_frames * self.l as u64).div_ceil(self.m as u64);
        let mut carry = Vec::new();
        for s in &mut self.stages {
            let mut out = if carry.is_empty() { Vec::new() } else { s.push(&carry) };
            out.extend(s.flush());
            carry = out;
        }
        let total = self.output_frames + (carry.len() / self.channels) as u64;
        debug_assert!(total >= expected, "flush produced {total} < {expected} frames");
        let extra = total.saturating_sub(expected) as usize;
        carry.truncate(carry.len().saturating_sub(extra * self.channels));
        self.reset();
        carry
    }

    /// Forget all state (a seek, or a new stream).
    pub fn reset(&mut self) {
        for s in &mut self.stages {
            s.reset();
        }
        self.input_frames = 0;
        self.output_frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::RealFft;
    use std::f64::consts::PI;

    fn sine(freq: f64, rate: u32, seconds: f64, amp: f64) -> Vec<f64> {
        let n = (rate as f64 * seconds) as usize;
        (0..n)
            .map(|i| amp * (2.0 * PI * freq * i as f64 / rate as f64).sin())
            .collect()
    }

    fn kind(taps: usize, phase: Phase) -> FilterKind {
        FilterKind::sinc(phase, false, taps)
    }

    /// Hann-windowed spectrum: returns `(peak_hz, peak_amplitude, max_db_above(limit_hz) relative to peak)`.
    fn analyse(x: &[f64], rate: u32, limit_hz: f64) -> (f64, f64, f64) {
        let n = x.len().next_power_of_two() / 2;
        let x = &x[..n];
        let mut fft = RealFft::new(n);
        let mut win_sum = 0.0;
        let mut buf: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos();
                win_sum += w;
                v * w
            })
            .collect();
        let mut spec = fft.spectrum();
        fft.forward(&mut buf, &mut spec);
        let mags: Vec<f64> = spec.iter().map(|c| c.norm() * 2.0 / win_sum).collect();
        let (peak_bin, _) =
            mags.iter()
                .enumerate()
                .fold((0, 0.0), |a, (i, &m)| if m > a.1 { (i, m) } else { a });
        let hz_per_bin = rate as f64 / n as f64;
        // Parabolic interpolation of the peak position.
        let (a, b, c) = (
            mags[peak_bin - 1].ln(),
            mags[peak_bin].ln(),
            mags[peak_bin + 1].ln(),
        );
        let frac = 0.5 * (a - c) / (a - 2.0 * b + c);
        let peak_hz = (peak_bin as f64 + frac) * hz_per_bin;
        // A pure tone's amplitude is RMS·√2; over a thousand periods this is far more
        // accurate than reading a windowed FFT bin (scalloping loss).
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let peak_amp = rms * 2f64.sqrt();
        let limit_bin = (limit_hz / hz_per_bin).ceil() as usize;
        let worst = mags[limit_bin..].iter().cloned().fold(0.0, f64::max);
        (peak_hz, peak_amp, 20.0 * (worst / peak_amp).log10())
    }

    #[test]
    fn upsampling_x4_keeps_the_tone_and_has_no_images() {
        let cache = CoefficientCache::disabled();
        let mut r = Resampler::new(
            44100,
            176400,
            1,
            kind(1 << 14, Phase::Linear),
            150.0,
            &cache,
        )
        .unwrap();
        assert_eq!(r.ratio(), (4, 1));
        assert_eq!(r.stages().len(), 1);
        let x = sine(1000.0, 44100, 2.0, 0.5);
        let mut y = Vec::new();
        for chunk in x.chunks(1234) {
            y.extend(r.push(chunk));
        }
        y.extend(r.flush());
        assert_eq!(y.len(), x.len() * 4);

        // Time alignment: away from the edges the output *is* the ideal 176.4 kHz sine.
        let ideal = sine(1000.0, 176400, 2.0, 0.5);
        let margin = 176400 / 2;
        let err = y[margin..y.len() - margin]
            .iter()
            .zip(&ideal[margin..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "alignment error {err}");

        let (hz, amp, images_db) = analyse(&y[margin..], 176400, 22050.0);
        assert!((hz - 1000.0).abs() < 1.0, "peak at {hz} Hz");
        let amp_db = 20.0 * (amp / 0.5).log10();
        assert!(amp_db.abs() < 0.1, "amplitude error {amp_db} dB");
        assert!(images_db < -100.0, "images at {images_db} dB");
    }

    #[test]
    fn dc_stays_dc() {
        let cache = CoefficientCache::disabled();
        for phase in [Phase::Linear, Phase::Minimum] {
            let mut r =
                Resampler::new(48000, 384000, 2, kind(1 << 12, phase), 140.0, &cache).unwrap();
            assert_eq!(r.ratio(), (8, 1));
            let x = vec![0.25; 2 * 48000];
            let mut y = r.push(&x);
            y.extend(r.flush());
            assert_eq!(y.len(), x.len() * 8);
            let tail = &y[y.len() / 2..y.len() - 8 * 2048];
            for v in tail {
                assert!((v - 0.25).abs() < 1e-6, "{phase:?}: {v}");
            }
        }
    }

    #[test]
    fn rational_44100_to_48000_keeps_frequency() {
        let cache = CoefficientCache::disabled();
        let mut r =
            Resampler::new(44100, 48000, 1, kind(1 << 16, Phase::Linear), 120.0, &cache).unwrap();
        assert_eq!(r.ratio(), (160, 147));
        assert_eq!(r.engine(), Engine::TimePolyphase);
        assert_eq!(r.stages().len(), 1);
        let x = sine(1000.0, 44100, 1.5, 0.5);
        let mut y = Vec::new();
        for chunk in x.chunks(1000) {
            y.extend(r.push(chunk));
        }
        y.extend(r.flush());
        assert_eq!(y.len() as u64, (x.len() as u64 * 160).div_ceil(147));
        let margin = 48000 / 4;
        let (hz, amp, _) = analyse(&y[margin..], 48000, 23000.0);
        assert!((hz - 1000.0).abs() < 1.0, "peak at {hz} Hz");
        let amp_db = 20.0 * (amp / 0.5).log10();
        assert!(amp_db.abs() < 0.1, "amplitude error {amp_db} dB");
        let ideal = sine(1000.0, 48000, 1.5, 0.5);
        let err = y[margin..y.len() - margin]
            .iter()
            .zip(&ideal[margin..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "alignment error {err}");
    }

    #[test]
    fn cross_family_to_high_rate_is_cascaded_and_exact() {
        // 44.1 kHz → 384 kHz = ×1280/147: 160/147 time domain, then ×8 FFT.
        let cache = CoefficientCache::disabled();
        let mut r =
            Resampler::new(44100, 384000, 2, kind(1 << 14, Phase::Linear), 120.0, &cache).unwrap();
        assert_eq!(r.ratio(), (1280, 147));
        let stages = r.stages();
        assert_eq!(stages.len(), 2, "{}", r.describe());
        assert_eq!((stages[0].l, stages[0].m, stages[0].out_rate), (160, 147, 48000));
        assert_eq!((stages[1].l, stages[1].m), (8, 1));
        assert!(matches!(stages[1].engine, Engine::FftPolyphase { .. }));
        assert!(r.describe().contains("→"));
        let x: Vec<f64> = sine(1000.0, 44100, 1.0, 0.5)
            .into_iter()
            .flat_map(|v| [v, -v])
            .collect();
        let mut y = Vec::new();
        for chunk in x.chunks(2 * 999) {
            y.extend(r.push(chunk));
        }
        y.extend(r.flush());
        assert_eq!(y.len() as u64, 2 * (x.len() as u64 / 2 * 1280).div_ceil(147));
        let left: Vec<f64> = y.iter().step_by(2).copied().collect();
        let ideal = sine(1000.0, 384000, 1.0, 0.5);
        let margin = 384000 / 4;
        let err = left[margin..left.len() - margin]
            .iter()
            .zip(&ideal[margin..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "alignment error {err}");
        let (hz, amp, images_db) = analyse(&left[margin..], 384000, 23000.0);
        assert!((hz - 1000.0).abs() < 1.0, "peak at {hz} Hz");
        assert!((20.0 * (amp / 0.5).log10()).abs() < 0.1);
        assert!(images_db < -100.0, "images at {images_db} dB");
        assert!(r.latency_seconds() > 0.0 && r.latency_seconds() < 2.0);
    }

    #[test]
    fn x32_to_1536k_builds_and_uses_the_cascade() {
        let cache = CoefficientCache::disabled();
        let r = Resampler::new(44100, MAX_PCM_RATE, 2, kind(1 << 12, Phase::Linear), 120.0, &cache)
            .unwrap();
        assert_eq!(r.ratio(), (5120, 147));
        let s = r.stages();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].out_rate, 48000);
        assert_eq!(s[1].l, 32);
        let r = Resampler::new(48000, 1_411_200, 2, kind(1 << 12, Phase::Linear), 120.0, &cache)
            .unwrap();
        let s = r.stages();
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].l, s[0].m, s[0].out_rate), (7, 5, 67200));
        assert_eq!(s[1].l, 21);
    }

    #[test]
    fn decimation_96k_to_48k_keeps_frequency() {
        let cache = CoefficientCache::disabled();
        let mut r =
            Resampler::new(96000, 48000, 1, kind(1 << 12, Phase::Linear), 120.0, &cache).unwrap();
        assert_eq!(r.ratio(), (1, 2));
        let x = sine(5000.0, 96000, 1.0, 0.3);
        let mut y = r.push(&x);
        y.extend(r.flush());
        assert_eq!(y.len(), x.len() / 2);
        let ideal = sine(5000.0, 48000, 1.0, 0.3);
        let margin = 8000;
        let err = y[margin..y.len() - margin]
            .iter()
            .zip(&ideal[margin..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "alignment error {err}");
    }

    #[test]
    fn parallel_phases_match_serial_phases() {
        // ×16 with a long enough filter to cross the parallel threshold, versus ×16 with a
        // short one that stays serial: both must be exact against the ideal sine.
        let cache = CoefficientCache::disabled();
        for taps in [1 << 10, 1 << 16] {
            let mut r =
                Resampler::new(44100, 705600, 2, kind(taps, Phase::Linear), 130.0, &cache).unwrap();
            let x: Vec<f64> = sine(2000.0, 44100, 0.75, 0.4)
                .into_iter()
                .flat_map(|v| [v, 0.5 * v])
                .collect();
            let mut y = r.push(&x);
            y.extend(r.flush());
            assert_eq!(y.len(), x.len() * 16);
            let ideal = sine(2000.0, 705600, 0.75, 0.4);
            let margin = 705600 / 4;
            let n = y.len() / 2;
            let err = (margin..n - margin)
                .map(|i| ((y[2 * i] - ideal[i]).abs()).max((y[2 * i + 1] - 0.5 * ideal[i]).abs()))
                .fold(0.0, f64::max);
            assert!(err < 1e-4, "taps {taps}: alignment error {err}");
        }
    }

    #[test]
    fn same_rate_is_an_error_and_latency_is_reported() {
        let cache = CoefficientCache::disabled();
        assert_eq!(
            Resampler::new(44100, 44100, 2, FilterKind::default(), 150.0, &cache).err(),
            Some(ResampleError::SameRate(44100))
        );
        let r =
            Resampler::new(44100, 88200, 2, kind(1 << 12, Phase::Linear), 150.0, &cache).unwrap();
        assert!(r.latency_input_frames() >= 1024 && r.latency_seconds() < 1.0);
    }

    #[test]
    fn rate_tables() {
        let all = pcm_output_rates();
        assert_eq!(all.first(), Some(&44100));
        assert_eq!(all.last(), Some(&1_536_000));
        assert!(all.contains(&1_411_200) && all.contains(&705_600) && all.contains(&768_000));
        assert_eq!(all.len(), 12);
        assert!(!all.contains(&2_822_400));
        let with = pcm_output_rates_up_to(4_000_000, true);
        assert!(with.contains(&2_822_400) && with.contains(&3_072_000));
        assert_eq!(RateFamily::of(352_800), Some(RateFamily::F44));
        assert_eq!(RateFamily::of(768_000), Some(RateFamily::F48));
        assert_eq!(RateFamily::of(32_000), None);
        assert_eq!(preferred_output_rate(44100, 800_000), 705_600);
        assert_eq!(preferred_output_rate(96000, 1_600_000), 1_536_000);
        assert_eq!(preferred_output_rate(32000, 400_000), 384_000);
    }
}
