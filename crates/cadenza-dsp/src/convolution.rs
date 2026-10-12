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
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// The spectra of one impulse response cut into partitions of `block` samples.
#[derive(Clone, Debug)]
pub struct SpectralFilter {
    block: usize,
    taps: usize,
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
}

/// Frequency-domain delay line: the spectra of the most recent input blocks.
#[derive(Clone, Debug)]
pub struct Fdl {
    spectra: Vec<Vec<Complex<f64>>>,
    head: usize,
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
    pub fn convolve(&self, filter: &SpectralFilter, out: &mut [Complex<f64>]) {
        out.fill(Complex::new(0.0, 0.0));
        let n = self.spectra.len();
        for (p, h) in filter.partitions.iter().enumerate().take(self.count) {
            let x = &self.spectra[(self.head + p) % n];
            for ((o, a), b) in out.iter_mut().zip(x).zip(h) {
                *o += a * b;
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

/// The overlap-save front end shared by the convolver and the upsampler: gathers input
/// into blocks, transforms each one and keeps the spectra in an [`Fdl`].
#[derive(Debug)]
pub struct BlockEngine {
    block: usize,
    fft: RealFft,
    /// `[previous block | current block]`; the current block fills from `block + filled`.
    frame: Vec<f64>,
    filled: usize,
    spec: Vec<Complex<f64>>,
    time: Vec<f64>,
    fdl: Fdl,
}

impl BlockEngine {
    /// An engine with the given block size whose FDL remembers `depth` blocks.
    pub fn new(block: usize, depth: usize) -> Self {
        let fft = RealFft::new(2 * block);
        let spec = fft.spectrum();
        Self {
            block,
            frame: vec![0.0; 2 * block],
            filled: 0,
            time: vec![0.0; 2 * block],
            fdl: Fdl::new(depth, spec.len()),
            spec,
            fft,
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

    /// Feed input; every time a block completes, `on_block` receives the engine so it can
    /// call [`BlockEngine::output`] for each filter it is interested in.
    pub fn push(&mut self, input: &[f64], mut on_block: impl FnMut(&mut BlockEngine)) {
        let mut rest = input;
        while !rest.is_empty() {
            let take = (self.block - self.filled).min(rest.len());
            let start = self.block + self.filled;
            self.frame[start..start + take].copy_from_slice(&rest[..take]);
            self.filled += take;
            rest = &rest[take..];
            if self.filled == self.block {
                self.time.copy_from_slice(&self.frame);
                self.fft.forward(&mut self.time, &mut self.spec);
                self.fdl.push(&self.spec);
                on_block(self);
                // The current block becomes the "previous" half of the next frame.
                self.frame.copy_within(self.block.., 0);
                self.filled = 0;
            }
        }
    }

    /// Within `on_block`: the newest output block (`block` samples) for `filter`, appended to `out`.
    pub fn output(&mut self, filter: &SpectralFilter, out: &mut Vec<f64>) {
        self.fdl.convolve(filter, &mut self.spec);
        self.fft.inverse(&mut self.spec, &mut self.time);
        // Overlap-save: the first half is circular wrap-around, the second half is valid.
        out.extend_from_slice(&self.time[self.block..]);
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
    engine: BlockEngine,
    filter: Arc<SpectralFilter>,
    input_samples: u64,
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
