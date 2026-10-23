//! Uniformly partitioned FFT convolution.
//!
//! One impulse response of any length is cut into partitions of one block each; every
//! partition's spectrum is precomputed; the input is transformed once per block and kept
//! in a frequency-domain delay line (FDL), so each output block costs one FFT, one
//! multiply-accumulate per partition and one inverse FFT. This is what makes
//! multi-million-tap filters affordable in real time, and the same machinery drives both
//! room-correction convolution and the polyphase upsampler (which shares the FDL between
//! its `L` sub-filters).

use crate::fft::{next_pow2, RealFft};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// Below this many complex multiply-accumulates per block the MAC runs on the calling
/// thread; above it the bins are split across the rayon pool (whole blocks only, so the
/// audio thread never waits on a half-finished partition).
pub const PARALLEL_MAC_THRESHOLD: usize = 1 << 17;

/// The interface every block convolution engine offers (CPU here, CUDA in `gpu::Convolver`):
/// `input` and `out` are exactly one block long and `out` is the convolution output for the
/// block that entered `latency()` samples ago.
pub trait BlockConvolver {
    /// Convolve one block: `input.len() == out.len() == block size`.
    fn process_block(&mut self, input: &[f64], out: &mut [f64]);
    /// Samples between an input block entering and its output leaving (one block for
    /// overlap-save engines).
    fn latency(&self) -> usize;
    /// Forget all history (seek / end of stream); the filter stays loaded.
    fn reset(&mut self);
    /// `true` when this engine runs on a GPU (the signal path marks the `gpu` stage active
    /// only when a chain actually holds one).
    fn on_gpu(&self) -> bool {
        false
    }
}

/// The polyphase upsampler's FFT core on another engine (the CUDA device, or the CPU model
/// of the GPU algorithm in `gpu::reference`): the `L` sub-filters of one long FIR share one
/// forward transform per input block.
///
/// `process_block` takes exactly one input block and writes `block × phases` upsampled
/// samples in output order (`out[j·L + p]` is phase `p` of input sample `j`), with the same
/// one-block overlap-save latency as [`BlockEngine`] driving `L` [`SpectralFilter`]s. The
/// resampler handles delay compensation, decimation and flushing around it, exactly as for
/// its CPU core. An `Err` means the engine is unusable (the resampler then moves the
/// stream to the CPU core and replays the history into it).
pub trait PolyphaseBlockConvolver {
    /// Input block size in frames.
    fn block(&self) -> usize;
    /// `L`, the number of polyphase sub-filters (output samples per input sample).
    fn phases(&self) -> usize;
    /// Upsample one block: `input.len() == block`, `out.len() == block · phases`.
    fn process_block(&mut self, input: &[f64], out: &mut [f64])
        -> Result<(), crate::gpu::GpuError>;
    /// Forget all history; the sub-filter spectra stay loaded.
    fn reset(&mut self);
    /// Where it runs, for the signal path: `CUDA (RTX 4060)`.
    fn device_label(&self) -> String;
    /// `true` when this engine runs on a GPU.
    fn on_gpu(&self) -> bool {
        false
    }
}

/// The time-domain polyphase bank of a rational resampler stage (44.1 ↔ 48 kHz families)
/// on another engine: a GPU, or the CPU model of the GPU algorithm in `gpu::reference`.
///
/// The engine holds the `L` reversed sub-filters of `S` taps each (`sub[p][k]`, as the CPU
/// core holds them). [`RationalDotEngine::run`] computes `count` output samples from a
/// window of input history: output `j` sits at upsampled offset `u_j = phase0 + j·m`, uses
/// sub-filter `p_j = u_j mod L` and the `S` inputs ending at window position
/// `n_j = first + u_j / L`:
///
/// ```text
/// out[j] = Σ_k sub[p_j][k] · window[n_j + 1 − S + k]
/// ```
///
/// The resampler keeps the history, the output clock and the flush logic, so a device
/// error (`Err`) simply hands the same state to the CPU core.
pub trait RationalDotEngine {
    /// `L`, the number of sub-filters.
    fn phases(&self) -> usize;
    /// `S`, the taps per sub-filter.
    fn sub_len(&self) -> usize;
    /// Compute `out.len()` outputs (see the trait docs). `window` must reach every input
    /// the outputs need: `first + (phase0 + (out.len() − 1)·m) / L < window.len()` and
    /// `first + 1 ≥ S`.
    fn run(
        &mut self,
        window: &[f64],
        first: usize,
        phase0: usize,
        m: usize,
        out: &mut [f64],
    ) -> Result<(), crate::gpu::GpuError>;
    /// Where it runs, for the signal path: `CUDA (RTX 4060)`.
    fn device_label(&self) -> String;
    /// `true` when this engine runs on a GPU.
    fn on_gpu(&self) -> bool {
        false
    }
}

/// The spectra of one impulse response cut into partitions of `block` samples.
#[derive(Clone, Debug)]
pub struct SpectralFilter {
    /// Block size in samples each partition covers (the FFT is twice this).
    block: usize,
    /// Length of the original impulse response.
    taps: usize,
    /// Spectrum of each `block`-sample slice of `h`, earliest first.
    partitions: Vec<Vec<Complex<f64>>>,
}

impl SpectralFilter {
    /// Partition `h` for a block size of `block` samples (a power of two).
    pub fn new(h: &[f64], block: usize) -> Self {
        assert!(block.is_power_of_two(), "block size must be a power of two");
        let mut fft = RealFft::new(2 * block);
        let mut time = vec![0.0; 2 * block];
        let partitions = h
            .chunks(block)
            .map(|chunk| {
                time.fill(0.0);
                time[..chunk.len()].copy_from_slice(chunk);
                let mut spec = fft.spectrum();
                fft.forward(&mut time, &mut spec);
                spec
            })
            .collect();
        Self {
            block,
            taps: h.len(),
            partitions,
        }
    }

    /// Block size the filter was partitioned for.
    pub fn block(&self) -> usize {
        self.block
    }

    /// Length of the original impulse response.
    pub fn taps(&self) -> usize {
        self.taps
    }

    /// Number of partitions (the FDL depth this filter needs).
    pub fn partitions(&self) -> usize {
        self.partitions.len()
    }

    /// The partition spectra themselves (`block + 1` bins each, `realfft` half spectra),
    /// for engines that keep their own copy – the CUDA convolver uploads them once.
    pub fn partition_spectra(&self) -> &[Vec<Complex<f64>>] {
        &self.partitions
    }
}

/// Frequency-domain delay line: the spectra of the most recent input blocks.
#[derive(Clone, Debug)]
pub struct Fdl {
    /// Ring of the most recent input spectra.
    spectra: Vec<Vec<Complex<f64>>>,
    /// Slot of the newest spectrum (the ring runs backwards, so `head + p` is `p` blocks ago).
    head: usize,
    /// Spectra pushed so far, capped at the ring size (older slots are still zero).
    count: usize,
}

impl Fdl {
    /// A delay line holding `depth` spectra of `bins` complex values.
    pub fn new(depth: usize, bins: usize) -> Self {
        Self {
            spectra: vec![vec![Complex::new(0.0, 0.0); bins]; depth.max(1)],
            head: 0,
            count: 0,
        }
    }

    /// Insert the newest spectrum (the oldest one is overwritten when full).
    pub fn push(&mut self, spec: &[Complex<f64>]) {
        self.head = (self.head + self.spectra.len() - 1) % self.spectra.len();
        self.spectra[self.head].copy_from_slice(spec);
        self.count = (self.count + 1).min(self.spectra.len());
    }

    /// `out = Σ_p X[t − p] · H_p`: the spectrum of the next output block for `filter`.
    ///
    /// Large products (many partitions × many bins) are split across bins on the rayon
    /// pool; every thread sums all partitions for its range of bins, so there is no
    /// reduction step and the inner loop is a plain fused multiply-add over four `f64`
    /// streams (SIMD friendly).
    pub fn convolve(&self, filter: &SpectralFilter, out: &mut [Complex<f64>]) {
        let depth = self.count.min(filter.partitions.len());
        let work = depth * out.len();
        if work >= PARALLEL_MAC_THRESHOLD && rayon::current_num_threads() > 1 {
            let chunk = (out.len() / (2 * rayon::current_num_threads())).max(1024);
            out.par_chunks_mut(chunk).enumerate().for_each(|(i, o)| {
                let start = i * chunk;
                self.convolve_range(filter, depth, start, o);
            });
        } else {
            self.convolve_range(filter, depth, 0, out);
        }
    }

    /// The MAC for bins `start..start + out.len()`.
    #[inline]
    fn convolve_range(
        &self,
        filter: &SpectralFilter,
        depth: usize,
        start: usize,
        out: &mut [Complex<f64>],
    ) {
        out.fill(Complex::new(0.0, 0.0));
        let n = self.spectra.len();
        let end = start + out.len();
        for (p, h) in filter.partitions.iter().enumerate().take(depth) {
            let x = &self.spectra[(self.head + p) % n][start..end];
            let h = &h[start..end];
            for ((o, a), b) in out.iter_mut().zip(x).zip(h) {
                let re = a.re * b.re - a.im * b.im;
                let im = a.re * b.im + a.im * b.re;
                o.re += re;
                o.im += im;
            }
        }
    }

    /// Forget everything (silence before the next block).
    pub fn clear(&mut self) {
        self.count = 0;
        for s in &mut self.spectra {
            s.fill(Complex::new(0.0, 0.0));
        }
    }
}

/// The inverse-FFT scratch one [`BlockEngine::output_with`] call needs. A caller that
/// evaluates several filters against the same engine in parallel (the polyphase upsampler)
/// owns one per worker.
#[derive(Debug)]
pub struct OutputScratch {
    /// The inverse transform for this block size.
    fft: RealFft,
    /// Output spectrum being accumulated.
    spec: Vec<Complex<f64>>,
    /// Time-domain result of the inverse transform (`2 × block` samples).
    time: Vec<f64>,
}

impl OutputScratch {
    /// Scratch for an engine with the given block size.
    pub fn new(block: usize) -> Self {
        let fft = RealFft::new(2 * block);
        let spec = fft.spectrum();
        Self {
            fft,
            spec,
            time: vec![0.0; 2 * block],
        }
    }
}

/// The overlap-save front end shared by the convolver and the upsampler: gathers input
/// into blocks, transforms each one and keeps the spectra in an [`Fdl`].
#[derive(Debug)]
pub struct BlockEngine {
    /// Input block size in samples.
    block: usize,
    /// `[previous block | current block]`; the current block fills from `block + filled`.
    frame: Vec<f64>,
    /// Samples of the current block received so far.
    filled: usize,
    /// Scratch for single-threaded output.
    scratch: OutputScratch,
    /// Input spectra of the last `depth` blocks.
    fdl: Fdl,
}

impl BlockEngine {
    /// An engine with the given block size whose FDL remembers `depth` blocks.
    pub fn new(block: usize, depth: usize) -> Self {
        let scratch = OutputScratch::new(block);
        Self {
            block,
            frame: vec![0.0; 2 * block],
            filled: 0,
            fdl: Fdl::new(depth, scratch.spec.len()),
            scratch,
        }
    }

    /// Block size in samples.
    pub fn block(&self) -> usize {
        self.block
    }

    /// Samples of the current block gathered so far.
    pub fn pending(&self) -> usize {
        self.filled
    }

    /// A fresh [`OutputScratch`] matching this engine.
    pub fn scratch(&self) -> OutputScratch {
        OutputScratch::new(self.block)
    }

    /// The frequency-domain delay line (the newest block's spectrum first).
    pub fn fdl(&self) -> &Fdl {
        &self.fdl
    }

    /// Feed input; every time a block completes, `on_block` receives the engine so it can
    /// call [`BlockEngine::output`] (or [`BlockEngine::output_with`]) for each filter it is
    /// interested in.
    pub fn push(&mut self, input: &[f64], mut on_block: impl FnMut(&mut BlockEngine)) {
        let mut rest = input;
        while !rest.is_empty() {
            let take = (self.block - self.filled).min(rest.len());
            let start = self.block + self.filled;
            self.frame[start..start + take].copy_from_slice(&rest[..take]);
            self.filled += take;
            rest = &rest[take..];
            if self.filled == self.block {
                self.scratch.time.copy_from_slice(&self.frame);
                self.scratch
                    .fft
                    .forward(&mut self.scratch.time, &mut self.scratch.spec);
                self.fdl.push(&self.scratch.spec);
                on_block(self);
                // The current block becomes the "previous" half of the next frame.
                self.frame.copy_within(self.block.., 0);
                self.filled = 0;
            }
        }
    }

    /// Within `on_block`: the newest output block (`block` samples) for `filter`, appended to `out`.
    pub fn output(&mut self, filter: &SpectralFilter, out: &mut Vec<f64>) {
        let Self { fdl, scratch, .. } = self;
        Self::output_into(fdl, filter, scratch, out);
    }

    /// Like [`BlockEngine::output`] but with caller-owned scratch, so several filters can be
    /// evaluated against the same delay line from several threads at once.
    pub fn output_with(
        &self,
        filter: &SpectralFilter,
        scratch: &mut OutputScratch,
        out: &mut Vec<f64>,
    ) {
        Self::output_into(&self.fdl, filter, scratch, out);
    }

    /// Accumulate `fdl` against `filter`, transform back and append the valid (second) half
    /// of the block to `out` (overlap-save).
    fn output_into(
        fdl: &Fdl,
        filter: &SpectralFilter,
        scratch: &mut OutputScratch,
        out: &mut Vec<f64>,
    ) {
        fdl.convolve(filter, &mut scratch.spec);
        scratch.fft.inverse(&mut scratch.spec, &mut scratch.time);
        // Overlap-save: the first half is circular wrap-around, the second half is valid.
        let block = scratch.time.len() / 2;
        out.extend_from_slice(&scratch.time[block..]);
    }

    /// Forget all history.
    pub fn reset(&mut self) {
        self.frame.fill(0.0);
        self.filled = 0;
        self.fdl.clear();
    }
}

/// Pick a block size for a filter of `taps` coefficients: few enough partitions to be cheap,
/// small enough to keep latency reasonable (≤ 16384 samples ≈ 0.37 s at 44.1 kHz).
pub fn choose_block(taps: usize) -> usize {
    next_pow2(taps / 16).clamp(64, 16384)
}

/// A streaming single-channel convolver (room correction, or any long FIR).
#[derive(Debug)]
pub struct Convolver {
    /// The FDL and block gatherer.
    engine: BlockEngine,
    /// The filter in the frequency domain.
    filter: Arc<SpectralFilter>,
    /// Input samples pushed since the last reset.
    input_samples: u64,
    /// Output samples produced since the last reset (to trim the flush to the exact length).
    output_samples: u64,
}

impl Convolver {
    /// Convolve with `h` using an automatically chosen block size.
    pub fn new(h: &[f64]) -> Self {
        Self::with_block(h, choose_block(h.len()))
    }

    /// Convolve with `h` using a specific block size (power of two).
    pub fn with_block(h: &[f64], block: usize) -> Self {
        Self::from_filter(Arc::new(SpectralFilter::new(h, block)))
    }

    /// Convolve with an already partitioned filter (shared between channels).
    pub fn from_filter(filter: Arc<SpectralFilter>) -> Self {
        let engine = BlockEngine::new(filter.block(), filter.partitions());
        Self {
            engine,
            filter,
            input_samples: 0,
            output_samples: 0,
        }
    }

    /// Block size: output arrives in chunks of this many samples.
    pub fn block(&self) -> usize {
        self.engine.block()
    }

    /// Process `input`, appending whatever complete blocks are ready to `out`.
    pub fn push(&mut self, input: &[f64], out: &mut Vec<f64>) {
        self.input_samples += input.len() as u64;
        let before = out.len();
        let filter = Arc::clone(&self.filter);
        self.engine.push(input, |e| e.output(&filter, out));
        self.output_samples += (out.len() - before) as u64;
    }

    /// Push zeros until every input sample has produced its output (the filter tail is
    /// *not* included: the stream length stays equal to the input length).
    pub fn flush(&mut self, out: &mut Vec<f64>) {
        let pending = self.engine.pending();
        if pending > 0 {
            let zeros = vec![0.0; self.block() - pending];
            self.push(&zeros, out);
            let extra = zeros.len();
            out.truncate(out.len() - extra);
            self.output_samples -= extra as u64;
            self.input_samples -= extra as u64;
        }
        self.engine.reset();
    }

    /// Forget all history.
    pub fn reset(&mut self) {
        self.engine.reset();
        self.input_samples = 0;
        self.output_samples = 0;
    }
}

impl BlockConvolver for Convolver {
    fn process_block(&mut self, input: &[f64], out: &mut [f64]) {
        assert_eq!(
            input.len(),
            self.block(),
            "process_block wants exactly one block"
        );
        assert_eq!(
            out.len(),
            self.block(),
            "process_block wants exactly one block"
        );
        assert_eq!(
            self.engine.pending(),
            0,
            "process_block after a partial push"
        );
        let mut tmp = Vec::with_capacity(self.block());
        self.push(input, &mut tmp);
        out.copy_from_slice(&tmp);
    }

    fn latency(&self) -> usize {
        self.block()
    }

    fn reset(&mut self) {
        Convolver::reset(self);
    }
}

/// Run one convolver per channel over `inputs` (one deinterleaved buffer per channel),
/// appending to the matching `outs`; channels run in parallel on the rayon pool when there
/// is more than one. With `flush`, every convolver is flushed afterwards.
pub fn process_channels(
    convolvers: &mut [Convolver],
    inputs: &[Vec<f64>],
    outs: &mut [Vec<f64>],
    flush: bool,
) {
    assert!(convolvers.len() == inputs.len() && inputs.len() == outs.len());
    let run = |(conv, (inp, out)): (&mut Convolver, (&Vec<f64>, &mut Vec<f64>))| {
        conv.push(inp, out);
        if flush {
            conv.flush(out);
        }
    };
    if convolvers.len() > 1 {
        convolvers
            .par_iter_mut()
            .zip(inputs.par_iter().zip(outs.par_iter_mut()))
            .for_each(run);
    } else {
        convolvers
            .iter_mut()
            .zip(inputs.iter().zip(outs.iter_mut()))
            .for_each(run);
    }
}

/// Load an impulse response from a `.wav` (any channel count; `f64` per channel) or a text
/// file with one coefficient per line (blank lines and `#` comments ignored; one channel).
///
/// Returns `(sample_rate, channels)`; `sample_rate` is `None` for text files.
pub fn load_impulse_response(
    path: &std::path::Path,
) -> Result<(Option<u32>, Vec<Vec<f64>>), ImpulseError> {
    let is_wav = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if is_wav {
        let mut reader =
            hound::WavReader::open(path).map_err(|e| ImpulseError::Wav(e.to_string()))?;
        let spec = reader.spec();
        let ch = spec.channels as usize;
        let samples: Vec<f64> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| s.map(|v| v as f64))
                .collect::<Result<_, _>>(),
            hound::SampleFormat::Int => {
                let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f64;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f64 * scale))
                    .collect::<Result<_, _>>()
            }
        }
        .map_err(|e| ImpulseError::Wav(e.to_string()))?;
        let mut per_channel = vec![Vec::with_capacity(samples.len() / ch.max(1)); ch];
        for frame in samples.chunks(ch) {
            for (c, v) in frame.iter().enumerate() {
                per_channel[c].push(*v);
            }
        }
        Ok((Some(spec.sample_rate), per_channel))
    } else {
        let text = std::fs::read_to_string(path).map_err(ImpulseError::Io)?;
        let mut h = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            // Accept "value" or "index value" columns.
            let v = line.split_whitespace().last().unwrap();
            h.push(
                v.parse::<f64>()
                    .map_err(|_| ImpulseError::Parse(i + 1, line.to_string()))?,
            );
        }
        if h.is_empty() {
            return Err(ImpulseError::Empty);
        }
        Ok((None, vec![h]))
    }
}

/// Errors from [`load_impulse_response`].
#[derive(Debug, thiserror::Error)]
pub enum ImpulseError {
    /// The WAV file could not be read.
    #[error("wav: {0}")]
    Wav(String),
    /// The text file could not be read.
    #[error("io: {0}")]
    Io(std::io::Error),
    /// A line of the text file is not a number.
    #[error("line {0}: not a number: {1:?}")]
    Parse(usize, String),
    /// The file holds no coefficients.
    #[error("impulse response is empty")]
    Empty,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(x: &[f64], h: &[f64]) -> Vec<f64> {
        (0..x.len())
            .map(|n| (0..=n.min(h.len() - 1)).map(|k| h[k] * x[n - k]).sum())
            .collect()
    }

    #[test]
    fn matches_direct_convolution_across_block_boundaries() {
        let h: Vec<f64> = (0..1000)
            .map(|i| ((i as f64) * 0.1).sin() / (1.0 + i as f64))
            .collect();
        let x: Vec<f64> = (0..5000).map(|i| ((i as f64) * 0.7).cos()).collect();
        let want = direct(&x, &h);
        let mut c = Convolver::with_block(&h, 128);
        let mut got = Vec::new();
        // Odd chunk sizes exercise the block assembly.
        for chunk in x.chunks(333) {
            c.push(chunk, &mut got);
        }
        c.flush(&mut got);
        assert_eq!(got.len(), x.len());
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn impulse_through_identity_filter_is_identity() {
        let mut c = Convolver::with_block(&[1.0], 64);
        let x: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let mut out = Vec::new();
        c.push(&x, &mut out);
        c.flush(&mut out);
        assert_eq!(out.len(), 200);
        for (a, b) in x.iter().zip(&out) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn block_convolver_trait_matches_push_and_parallel_mac_matches_serial() {
        // 40 partitions × 2049 bins is above the parallel threshold.
        let h: Vec<f64> = (0..40 * 2048)
            .map(|i| ((i as f64) * 0.013).sin() / (1.0 + i as f64 * 0.01))
            .collect();
        let x: Vec<f64> = (0..8 * 2048).map(|i| ((i as f64) * 0.37).cos()).collect();
        let mut serial = Convolver::with_block(&h, 2048);
        let mut want = Vec::new();
        serial.push(&x, &mut want);

        let mut c = Convolver::with_block(&h, 2048);
        assert_eq!(c.latency(), 2048);
        let mut got = Vec::new();
        let mut blk = vec![0.0; 2048];
        for chunk in x.chunks(2048) {
            c.process_block(chunk, &mut blk);
            got.extend_from_slice(&blk);
        }
        assert_eq!(got.len(), want.len());
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        // And the parallel MAC agrees with a direct evaluation.
        let direct_tail = direct(&x[..3000], &h);
        let mut d = Convolver::with_block(&h, 2048);
        let mut out = Vec::new();
        d.push(&x[..3000], &mut out);
        d.flush(&mut out);
        for (a, b) in direct_tail.iter().zip(&out) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn channels_run_in_parallel_with_identical_results() {
        let h: Vec<f64> = (0..300).map(|i| (-(i as f64) / 40.0).exp()).collect();
        let inputs: Vec<Vec<f64>> = (0..4)
            .map(|c| {
                (0..1000)
                    .map(|i| ((i * (c + 1)) as f64 * 0.05).sin())
                    .collect()
            })
            .collect();
        let mut convs: Vec<Convolver> = (0..4).map(|_| Convolver::with_block(&h, 128)).collect();
        let mut outs = vec![Vec::new(); 4];
        process_channels(&mut convs, &inputs, &mut outs, true);
        for (inp, out) in inputs.iter().zip(&outs) {
            let want = direct(inp, &h);
            assert_eq!(out.len(), want.len());
            for (a, b) in want.iter().zip(out) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn loads_text_and_wav_impulses() {
        let dir = std::env::temp_dir().join(format!("cadenza-ir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let txt = dir.join("ir.txt");
        std::fs::write(&txt, "# comment\n1.0\n\n0.5\n-0.25\n").unwrap();
        let (rate, ch) = load_impulse_response(&txt).unwrap();
        assert_eq!(rate, None);
        assert_eq!(ch, vec![vec![1.0, 0.5, -0.25]]);

        let wav = dir.join("ir.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&wav, spec).unwrap();
        for v in [16384i16, -16384, 8192, 0] {
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();
        let (rate, ch) = load_impulse_response(&wav).unwrap();
        assert_eq!(rate, Some(48000));
        assert_eq!(ch.len(), 2);
        assert!((ch[0][0] - 0.5).abs() < 1e-12 && (ch[1][0] + 0.5).abs() < 1e-12);
        assert!((ch[0][1] - 0.25).abs() < 1e-12 && ch[1][1] == 0.0);
        let _ = std::fs::remove_dir_all(dir);
    }
}
