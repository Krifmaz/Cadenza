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
//! * On an NVIDIA card (`--features cuda`, `gpu = auto | cuda`) the FFT polyphase stages
//!   run on the device ([`crate::gpu::PolyphaseConvolver`]): one forward transform per
//!   input block, every sub-filter's multiply-accumulate and inverse transform batched,
//!   the phases interleaved on the device. The resampler keeps the delay compensation,
//!   decimation and flush logic, so both cores give the same stream (to f64 rounding). A
//!   CUDA error mid-stream moves the stage to the CPU core with its history replayed.
//!
//! The output is **time aligned** with the input: the linear-phase group delay is absorbed
//! (the first `(N−1)/2` filter outputs are skipped and `flush` pushes the tail out), so a
//! stream of `n` input frames gives exactly `ceil(n·L/M)` output frames. The processing
//! latency a player must account for is reported by [`Resampler::latency_input_frames`].

use crate::cache::CoefficientCache;
use crate::convolution::{
    choose_block, BlockEngine, OutputScratch, PolyphaseBlockConvolver, SpectralFilter,
};
use crate::fir::{FilterKind, FirSpec, Phase};
use crate::gpu::PolyphaseFactory;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
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
        rates.extend(
            DSD_MODULATOR_PCM_RATES
                .iter()
                .copied()
                .filter(|r| *r <= max),
        );
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
        Some(f) => f.rates_up_to(max).last().copied().unwrap_or(in_rate),
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
    /// Where the FFT core runs when not on the CPU (`CUDA (RTX 4060)`); `None` on the CPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
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
    /// A fresh CPU core for one channel.
    fn new(filters: &Arc<Vec<SpectralFilter>>, par: Parallelism, m: usize, delay: u64) -> Self {
        let block = filters[0].block();
        let engine = BlockEngine::new(block, filters[0].partitions());
        Self {
            scratch: (0..par.workers).map(|_| engine.scratch()).collect(),
            engine,
            filters: Arc::clone(filters),
            parallel: par.parallel,
            m,
            u: 0,
            next_take: delay,
            phase_out: vec![Vec::with_capacity(block); filters.len()],
        }
    }

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

/// How the CPU core spreads the `L` phases over the rayon pool.
#[derive(Clone, Copy, Debug)]
struct Parallelism {
    parallel: bool,
    workers: usize,
}

/// Per-channel state of the FFT polyphase core on a GPU (or its CPU model): the engine does
/// one block in, `block × L` interleaved upsampled samples out; this keeps the block
/// gathering, the delay compensation and decimation (exactly as [`FftCore`]) and the last
/// `depth` input blocks, so a failing device can be replaced by a CPU core mid-stream
/// without losing a sample.
struct GpuCore {
    engine: Box<dyn PolyphaseBlockConvolver + Send>,
    /// For the CPU core that takes over after a device error.
    filters: Arc<Vec<SpectralFilter>>,
    par: Parallelism,
    block: usize,
    m: usize,
    /// FDL depth: blocks of history needed to rebuild the CPU core's state.
    depth: usize,
    /// The input block being gathered.
    pending: Vec<f64>,
    /// The last `depth` complete input blocks, oldest first.
    history: VecDeque<Vec<f64>>,
    /// The engine's output for one block.
    out_block: Vec<f64>,
    u: u64,
    next_take: u64,
    /// Shared by the channels of one stage, so a lost device is logged once, not once per
    /// channel.
    reported: Arc<AtomicBool>,
}

impl GpuCore {
    fn new(
        engine: Box<dyn PolyphaseBlockConvolver + Send>,
        filters: &Arc<Vec<SpectralFilter>>,
        par: Parallelism,
        m: usize,
        delay: u64,
        reported: Arc<AtomicBool>,
    ) -> Self {
        let block = engine.block();
        let l = engine.phases();
        let depth = filters[0].partitions().max(1);
        Self {
            engine,
            filters: Arc::clone(filters),
            par,
            block,
            m,
            depth,
            pending: Vec::with_capacity(block),
            history: VecDeque::with_capacity(depth),
            out_block: vec![0.0; block * l],
            u: 0,
            next_take: delay,
            reported,
        }
    }

    /// Push input. On a device error, returns the CPU core that now holds this channel's
    /// complete state (it has already produced the output of the failed block) and how many
    /// samples of `input` were consumed; the caller pushes the rest into it.
    fn push(&mut self, input: &[f64], out: &mut Vec<f64>) -> Option<(Box<FftCore>, usize)> {
        let mut used = 0;
        while used < input.len() {
            let take = (self.block - self.pending.len()).min(input.len() - used);
            self.pending.extend_from_slice(&input[used..used + take]);
            used += take;
            if self.pending.len() < self.block {
                break;
            }
            if let Err(e) = self
                .engine
                .process_block(&self.pending, &mut self.out_block)
            {
                if !self.reported.swap(true, Ordering::Relaxed) {
                    log::error!(
                        "GPU: {e} in the upsampler on {}; this stream continues on the CPU",
                        self.engine.device_label()
                    );
                } else {
                    log::debug!("GPU: {e} in another channel of the upsampler; on the CPU");
                }
                return Some((self.move_to_cpu(out), used));
            }
            self.emit(out);
            let mut done = if self.history.len() == self.depth {
                self.history.pop_front().unwrap_or_default()
            } else {
                Vec::with_capacity(self.block)
            };
            done.clear();
            done.extend_from_slice(&self.pending);
            self.history.push_back(done);
            self.pending.clear();
        }
        None
    }

    /// Delay compensation and decimation of one engine block (the [`FftCore`] logic over
    /// the same sample order).
    fn emit(&mut self, out: &mut Vec<f64>) {
        if self.m == 1 {
            if self.u >= self.next_take {
                out.extend_from_slice(&self.out_block);
                self.u += self.out_block.len() as u64;
            } else {
                for v in &self.out_block {
                    if self.u >= self.next_take {
                        out.push(*v);
                    }
                    self.u += 1;
                }
            }
            self.next_take = self.next_take.max(self.u);
        } else {
            for v in &self.out_block {
                if self.u == self.next_take {
                    out.push(*v);
                    self.next_take += self.m as u64;
                }
                self.u += 1;
            }
        }
    }

    /// Rebuild this channel on the CPU: replay the history into a fresh CPU core (its FDL
    /// then holds exactly what the device held), then run the block that failed through it.
    fn move_to_cpu(&mut self, out: &mut Vec<f64>) -> Box<FftCore> {
        let mut cpu = Box::new(FftCore::new(&self.filters, self.par, self.m, 0));
        for block in &self.history {
            cpu.engine.push(block, |_| {});
        }
        cpu.u = self.u;
        cpu.next_take = self.next_take;
        cpu.push(&self.pending, out);
        self.pending.clear();
        self.history.clear();
        cpu
    }

    fn reset(&mut self, delay: u64) {
        self.engine.reset();
        self.pending.clear();
        self.history.clear();
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
    Gpu(Box<GpuCore>),
    Time(TimeCore),
}

impl Core {
    fn push(&mut self, input: &[f64], out: &mut Vec<f64>) {
        match self {
            Core::Fft(c) => c.push(input, out),
            Core::Time(c) => c.push(input, out),
            Core::Gpu(c) => {
                if let Some((mut cpu, used)) = c.push(input, out) {
                    cpu.push(&input[used..], out);
                    *self = Core::Fft(cpu);
                }
            }
        }
    }
    fn reset(&mut self, delay: u64) {
        match self {
            Core::Fft(c) => c.reset(delay),
            Core::Gpu(c) => c.reset(delay),
            Core::Time(c) => c.reset(delay),
        }
    }
    fn device(&self) -> Option<String> {
        match self {
            Core::Gpu(c) => Some(c.engine.device_label()),
            _ => None,
        }
    }
    fn on_gpu(&self) -> bool {
        matches!(self, Core::Gpu(c) if c.engine.on_gpu())
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
    fn new(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        spec: FirSpec,
        h: &[f64],
        gpu: Option<&dyn PolyphaseFactory>,
    ) -> Self {
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
            let par = Parallelism { parallel, workers };
            let on_device = gpu.and_then(|f| match f.polyphase(&filters, channels) {
                Ok(engines) if engines.len() == channels => Some(engines),
                Ok(_) => {
                    log::warn!("GPU: wrong number of upsampler cores; upsampling on the CPU");
                    None
                }
                Err(e) => {
                    log::warn!("GPU: {e}; the ×{l} upsampler runs on the CPU");
                    None
                }
            });
            let cores = match on_device {
                Some(engines) => {
                    log::info!(
                        "GPU: ×{l}{} upsampler ({} taps) on {}",
                        if m > 1 {
                            format!("/{m}")
                        } else {
                            String::new()
                        },
                        h.len(),
                        engines[0].device_label()
                    );
                    let reported = Arc::new(AtomicBool::new(false));
                    engines
                        .into_iter()
                        .map(|e| {
                            Core::Gpu(Box::new(GpuCore::new(
                                e,
                                &filters,
                                par,
                                m,
                                delay,
                                Arc::clone(&reported),
                            )))
                        })
                        .collect()
                }
                None => (0..channels)
                    .map(|_| Core::Fft(Box::new(FftCore::new(&filters, par, m, delay))))
                    .collect(),
            };
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
            device: self.device(),
        }
    }

    /// The device the FFT core runs on (all channels, else `None`).
    fn device(&self) -> Option<String> {
        let first = self.cores.first()?.device()?;
        self.cores
            .iter()
            .all(|c| c.device().as_deref() == Some(first.as_str()))
            .then_some(first)
    }

    fn on_gpu(&self) -> bool {
        !self.cores.is_empty() && self.cores.iter().all(Core::on_gpu)
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
        Self::new_on(
            in_rate,
            out_rate,
            channels,
            kind,
            attenuation_db,
            cache,
            None,
        )
    }

    /// Like [`Resampler::new`], with the FFT polyphase stages on `gpu` when given (a CUDA
    /// [`crate::gpu::Backend`], or the CPU model of the GPU algorithm in tests). A stage
    /// whose GPU cores cannot be built stays on the CPU (logged).
    pub fn new_on(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        kind: FilterKind,
        attenuation_db: f64,
        cache: &CoefficientCache,
        gpu: Option<&dyn PolyphaseFactory>,
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
                let first = Self::stage(
                    in_rate,
                    mid_rate,
                    channels,
                    kind,
                    attenuation_db,
                    cache,
                    gpu,
                );
                // The second stage must pass everything the first one produced, so it is
                // never apodizing; its transition lies above the source band.
                let second_kind = FilterKind {
                    apodizing: false,
                    ..kind
                };
                let second = Self::stage(
                    mid_rate,
                    out_rate,
                    channels,
                    second_kind,
                    attenuation_db,
                    cache,
                    gpu,
                );
                vec![first, second]
            }
            None => vec![Self::stage(
                in_rate,
                out_rate,
                channels,
                kind,
                attenuation_db,
                cache,
                gpu,
            )],
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

    #[allow(clippy::too_many_arguments)]
    fn stage(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        kind: FilterKind,
        attenuation_db: f64,
        cache: &CoefficientCache,
        gpu: Option<&dyn PolyphaseFactory>,
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
        Stage::new(in_rate, out_rate, channels, spec, &h, gpu)
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
        Self::with_coefficients_on(in_rate, out_rate, channels, spec, h, None)
    }

    /// [`Resampler::with_coefficients`] with the FFT core on `gpu` when given.
    pub fn with_coefficients_on(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        spec: FirSpec,
        h: &[f64],
        gpu: Option<&dyn PolyphaseFactory>,
    ) -> Self {
        let g = gcd(in_rate, out_rate);
        let (l, m) = ((out_rate / g) as usize, (in_rate / g) as usize);
        Self {
            in_rate,
            out_rate,
            channels,
            l,
            m,
            stages: vec![Stage::new(in_rate, out_rate, channels, spec, h, gpu)],
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
    /// `true` when an FFT polyphase stage currently runs on a GPU (a stage that fell back
    /// to the CPU after a device error no longer counts).
    pub fn on_gpu(&self) -> bool {
        self.stages.iter().any(Stage::on_gpu)
    }

    /// The device of the first stage that runs off the CPU (`CUDA (RTX 4060)`).
    pub fn device(&self) -> Option<String> {
        self.stages.iter().find_map(Stage::device)
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
                match s.device() {
                    Some(d) => format!("{ratio} {} · {d}", s.engine.label()),
                    None => format!("{ratio} {}", s.engine.label()),
                }
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
            let mut out = if carry.is_empty() {
                Vec::new()
            } else {
                s.push(&carry)
            };
            out.extend(s.flush());
            carry = out;
        }
        let total = self.output_frames + (carry.len() / self.channels) as u64;
        debug_assert!(
            total >= expected,
            "flush produced {total} < {expected} frames"
        );
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
        let mut r = Resampler::new(
            44100,
            384000,
            2,
            kind(1 << 14, Phase::Linear),
            120.0,
            &cache,
        )
        .unwrap();
        assert_eq!(r.ratio(), (1280, 147));
        let stages = r.stages();
        assert_eq!(stages.len(), 2, "{}", r.describe());
        assert_eq!(
            (stages[0].l, stages[0].m, stages[0].out_rate),
            (160, 147, 48000)
        );
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
        assert_eq!(
            y.len() as u64,
            2 * (x.len() as u64 / 2 * 1280).div_ceil(147)
        );
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
        let r = Resampler::new(
            44100,
            MAX_PCM_RATE,
            2,
            kind(1 << 12, Phase::Linear),
            120.0,
            &cache,
        )
        .unwrap();
        assert_eq!(r.ratio(), (5120, 147));
        let s = r.stages();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].out_rate, 48000);
        assert_eq!(s[1].l, 32);
        let r = Resampler::new(
            48000,
            1_411_200,
            2,
            kind(1 << 12, Phase::Linear),
            120.0,
            &cache,
        )
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
    // ---- The GPU polyphase core, through its CPU model (`gpu::reference`) ----

    use crate::gpu::reference::ReferenceFactory;

    /// A deterministic broadband test signal (tones + LCG noise), interleaved.
    fn test_signal(frames: usize, channels: usize) -> Vec<f64> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        (0..frames * channels)
            .map(|i| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                let t = (i / channels) as f64;
                let c = (i % channels) as f64;
                0.4 * (t * (0.031 + 0.007 * c)).sin() + 0.2 * (t * 0.47).cos() + 0.1 * noise
            })
            .collect()
    }

    /// Push `x` through both converters with the same uneven chunking, flush, and return
    /// both whole streams.
    fn run_pair(
        cpu: &mut Resampler,
        gpu: &mut Resampler,
        x: &[f64],
        chunks: &[usize],
    ) -> (Vec<f64>, Vec<f64>) {
        let ch = cpu.channels();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut at = 0;
        let mut i = 0;
        while at < x.len() {
            let take = (chunks[i % chunks.len()] * ch).min(x.len() - at);
            let ya = cpu.push(&x[at..at + take]);
            let yb = gpu.push(&x[at..at + take]);
            assert_eq!(ya.len(), yb.len(), "same output per push");
            a.extend(ya);
            b.extend(yb);
            at += take;
            i += 1;
        }
        a.extend(cpu.flush());
        b.extend(gpu.flush());
        (a, b)
    }

    /// `max |a − b| / max |a|`.
    fn rel_err(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "stream lengths");
        let peak = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(peak > 0.0);
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
            / peak
    }

    fn gpu_model_matches_cpu(
        id: &str,
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        frames: usize,
    ) {
        let cache = CoefficientCache::disabled();
        let kind = FilterKind::parse(id).unwrap_or_else(|| panic!("{id}"));
        let model = ReferenceFactory::default();
        let mut cpu = Resampler::new(in_rate, out_rate, channels, kind, 150.0, &cache).unwrap();
        let mut gpu = Resampler::new_on(
            in_rate,
            out_rate,
            channels,
            kind,
            150.0,
            &cache,
            Some(&model),
        )
        .unwrap();
        assert_eq!(gpu.device().as_deref(), Some("GPU model (CPU)"), "{id}");
        assert!(
            gpu.describe().ends_with("· GPU model (CPU)"),
            "{}",
            gpu.describe()
        );
        assert!(!gpu.on_gpu(), "the CPU model is not a GPU");
        assert_eq!(gpu.latency_input_frames(), cpu.latency_input_frames());
        let x = test_signal(frames, channels);
        let (a, b) = run_pair(&mut cpu, &mut gpu, &x, &[1, 7, 1000, 333, 4099, 64]);
        let (l, m) = cpu.ratio();
        assert_eq!(
            a.len() as u64,
            channels as u64 * (frames as u64 * l as u64).div_ceil(m as u64)
        );
        let e = rel_err(&a, &b);
        assert!(e < 1e-12, "{id} {in_rate}→{out_rate}: relative error {e:e}");
    }

    #[test]
    fn gpu_core_matches_cpu_core_sinc_l_64k() {
        for l in [2u32, 4, 16, 32, 64] {
            gpu_model_matches_cpu("sinc-l-64k", 44_100, 44_100 * l, 2, 12_000);
        }
    }

    #[test]
    fn gpu_core_matches_cpu_core_poly_sinc_gauss_long() {
        for l in [2u32, 4, 16, 32, 64] {
            gpu_model_matches_cpu("poly-sinc-gauss-l-long", 48_000, 48_000 * l, 1, 20_000);
        }
    }

    #[test]
    fn gpu_core_matches_cpu_core_with_a_1m_tap_filter() {
        gpu_model_matches_cpu("sinc-l-1m", 44_100, 88_200, 1, 40_000);
        gpu_model_matches_cpu("sinc-l-1m", 44_100, 705_600, 2, 9_000);
    }

    #[test]
    fn gpu_core_handles_odd_phase_counts_decimation_and_cascades() {
        // ×3 (odd L: the last sub-filter pair has a zero partner), 3/2 (decimation after
        // the polyphase core), and the ×8 FFT stage after a 160/147 time-domain stage.
        gpu_model_matches_cpu("sinc-l-16k", 16_000, 48_000, 2, 9_000);
        gpu_model_matches_cpu("sinc-m-16k", 32_000, 48_000, 1, 9_000);
        gpu_model_matches_cpu("sinc-l-16k", 44_100, 384_000, 2, 9_000);
    }

    #[test]
    fn gpu_core_reset_and_reuse_match_a_fresh_cpu_converter() {
        let cache = CoefficientCache::disabled();
        let kind = FilterKind::parse("sinc-l-64k").unwrap();
        let model = ReferenceFactory::default();
        let mut gpu =
            Resampler::new_on(44_100, 705_600, 2, kind, 150.0, &cache, Some(&model)).unwrap();
        let x = test_signal(6_000, 2);
        // A seek mid-stream: half a stream, reset, then a whole stream from scratch.
        let _ = gpu.push(&x[..2 * 2_500]);
        gpu.reset();
        for round in 0..2 {
            let mut cpu = Resampler::new(44_100, 705_600, 2, kind, 150.0, &cache).unwrap();
            let (a, b) = run_pair(&mut cpu, &mut gpu, &x, &[513, 2048, 9]);
            let e = rel_err(&a, &b);
            assert!(e < 1e-12, "round {round}: {e:e}");
            // flush() reset both: the next round starts clean.
        }
        // The flush tail is there: the last output frames carry the filter's response to
        // the last input frames, not silence.
        let mut gpu =
            Resampler::new_on(44_100, 176_400, 1, kind, 150.0, &cache, Some(&model)).unwrap();
        let x = vec![0.5; 3_000];
        let mut y = gpu.push(&x);
        assert!(y.len() < 4 * x.len(), "latency holds some output back");
        y.extend(gpu.flush());
        assert_eq!(y.len(), 4 * x.len());
        assert!((y[y.len() - 4 * 200] - 0.5).abs() < 1e-3, "tail flushed");
    }

    #[test]
    fn gpu_failure_mid_stream_continues_on_the_cpu_without_a_glitch() {
        let cache = CoefficientCache::disabled();
        let kind = FilterKind::parse("sinc-l-64k").unwrap();
        // ×16: block 256, 16 partitions → the device "dies" in block 5, inside the FDL depth.
        let model = ReferenceFactory {
            fail_at_block: Some(5),
            ..Default::default()
        };
        let mut cpu = Resampler::new(44_100, 705_600, 2, kind, 150.0, &cache).unwrap();
        let mut gpu =
            Resampler::new_on(44_100, 705_600, 2, kind, 150.0, &cache, Some(&model)).unwrap();
        assert!(gpu.device().is_some());
        let x = test_signal(8_000, 2);
        let (a, b) = run_pair(&mut cpu, &mut gpu, &x, &[100, 3, 777]);
        let e = rel_err(&a, &b);
        assert!(e < 1e-12, "fallback must be seamless: {e:e}");
        assert_eq!(gpu.device(), None, "the stage now runs on the CPU");
        assert!(!gpu.describe().contains("GPU"));

        // And far past the FDL depth (the history replay is what keeps it exact).
        let model = ReferenceFactory {
            fail_at_block: Some(200),
            ..Default::default()
        };
        let mut cpu = Resampler::new(48_000, 192_000, 1, kind, 150.0, &cache).unwrap();
        let mut gpu =
            Resampler::new_on(48_000, 192_000, 1, kind, 150.0, &cache, Some(&model)).unwrap();
        let x = test_signal(400 * 1024, 1);
        let (a, b) = run_pair(&mut cpu, &mut gpu, &x, &[4096, 17]);
        assert!(rel_err(&a, &b) < 1e-12);
        assert_eq!(gpu.device(), None);
    }

    #[test]
    fn gpu_build_failure_keeps_the_stage_on_the_cpu() {
        let cache = CoefficientCache::disabled();
        let kind = FilterKind::parse("sinc-l-16k").unwrap();
        let model = ReferenceFactory {
            fail_build: true,
            ..Default::default()
        };
        let mut cpu = Resampler::new(44_100, 176_400, 2, kind, 150.0, &cache).unwrap();
        let mut gpu =
            Resampler::new_on(44_100, 176_400, 2, kind, 150.0, &cache, Some(&model)).unwrap();
        assert_eq!(gpu.device(), None);
        assert_eq!(gpu.describe(), cpu.describe());
        let x = test_signal(5_000, 2);
        let (a, b) = run_pair(&mut cpu, &mut gpu, &x, &[1000]);
        assert_eq!(a, b, "the same CPU core, bit for bit");
        // A time-domain-only conversion never asks for GPU cores.
        let r = Resampler::new_on(44_100, 48_000, 2, kind, 150.0, &cache, Some(&model)).unwrap();
        assert_eq!(r.device(), None);
        assert_eq!(r.engine(), Engine::TimePolyphase);
    }
}
