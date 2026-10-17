//! The processing chain a player runs:
//!
//! ```text
//! source PCM (f64) → volume → EQ → convolution → headphones → matrix → resampler → headroom/true-peak/soft clip → quantiser (dither) → integers
//! ```
//!
//! Every stage is optional. With `bit_perfect` requested nothing is built at all and the
//! samples reach the output exactly as decoded; the [`Pipeline::bit_perfect`] report is
//! computed from what the stages *actually do*, so a 0 dB volume, an empty EQ and an
//! unchanged rate also count as untouched.

use crate::cache::CoefficientCache;
use crate::convolution::{BlockConvolver, SpectralFilter};
use crate::dither::{DitherKind, Quantizer};
use crate::dsd::{DsdConfig, DsdStage, OutputMode};
use crate::eq::{Band, Equalizer};
use crate::fir::FilterKind;
use crate::gpu::{Backend as GpuBackend, GpuConfig, GpuUse, PolyphaseFactory};
use crate::headroom::{default_headroom_db, Headroom};
use crate::matrix::{MatrixConfig, MatrixError, MatrixStage};
use crate::resampler::{ResampleError, Resampler};
use crate::signal_path::{rate_label, Endpoint, SignalPath, Stage};
use crate::volume::Volume;
use rayon::prelude::*;
use serde_json::json;
use std::sync::Arc;

/// What the decoder delivers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputFormat {
    /// Sample rate in Hz.
    pub rate: u32,
    /// Source bit depth (a lossy codec reports what its decoder produces, e.g. 16 or 24).
    pub bits: u32,
    /// Channel count.
    pub channels: u32,
    /// Codec name for the signal path (`FLAC`, `MP3`…).
    pub codec: String,
}

/// A loaded impulse response for the convolution stage.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvolutionSpec {
    /// File name or label.
    pub name: String,
    /// Sample rate the impulse was measured at, when known.
    pub rate: Option<u32>,
    /// One impulse (`channels.len() == 1`, applied to every channel) or one per stream channel.
    pub channels: Vec<Vec<f64>>,
}

/// Everything that decides what the chain does.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineConfig {
    /// Bypass everything.
    pub bit_perfect: bool,
    /// Volume (can also be changed live with [`Pipeline::set_volume`]).
    pub volume: Volume,
    /// EQ bands (ignored unless `eq_enabled`).
    pub eq: Vec<Band>,
    /// Master switch for the EQ.
    pub eq_enabled: bool,
    /// Room correction impulse.
    pub convolution: Option<Arc<ConvolutionSpec>>,
    /// Upsampling filter shape.
    pub filter: FilterKind,
    /// Stopband attenuation of the upsampling filter in dB.
    pub attenuation_db: f64,
    /// Output rate; `None` keeps the source rate.
    pub output_rate: Option<u32>,
    /// Output bit depth (16, 24 or 32).
    pub output_bits: u32,
    /// Dither / noise shaping for the quantiser.
    pub dither: DitherKind,
    /// Intersample-peak headroom in dB applied after the resampler; `None` = automatic
    /// (−3 dB when upsampling, 0 dB otherwise, see [`crate::headroom`]).
    pub headroom_db: Option<f64>,
    /// Keep the transparent soft clipper as the last guard before quantisation.
    pub soft_clip: bool,
    /// Output device name, for the signal path only.
    pub output_device: Option<String>,
    /// PCM through the quantiser, or 1-bit DSD through the sigma-delta modulator
    /// (PART dsd; `output_mode` in config.toml).
    pub output_mode: OutputMode,
    /// DSD rate, modulator, packing, gain and oversampling (`dsd_*` keys); used when
    /// `output_mode` is `dsd`.
    pub dsd: DsdConfig,
    /// CUDA offload of the convolution engines (`gpu`, `gpu_device` keys; PART gpu).
    pub gpu: GpuConfig,
    /// Pipeline matrix (`[matrix]` table / `matrix_preset`): routing, per-output chains and
    /// crossovers; may change the channel count (PART matrix, [`crate::matrix`]).
    pub matrix: MatrixConfig,
    /// Crossfeed or HRTF virtual speakers for headphones (stereo sources only,
    /// [`crate::headphone`]).
    pub headphone: crate::headphone::HeadphoneMode,
    /// A filter from the designer; when set it is used instead of `filter`.
    pub filter_design: Option<crate::design::FilterDesign>,
    /// The DAC's own noise floor in dBFS (A-weighted), which `auto` dither chooses from;
    /// `None` assumes [`crate::dither::DEFAULT_DAC_FLOOR_DB`].
    pub dac_noise_floor_db: Option<f64>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            bit_perfect: false,
            volume: Volume::UNITY,
            eq: Vec::new(),
            eq_enabled: true,
            convolution: None,
            filter: FilterKind::default(),
            attenuation_db: crate::fir::DEFAULT_ATTENUATION_DB,
            output_rate: None,
            output_bits: 24,
            dither: DitherKind::Tpdf,
            headroom_db: None,
            soft_clip: true,
            output_device: None,
            output_mode: OutputMode::Pcm,
            dsd: DsdConfig::default(),
            gpu: GpuConfig::default(),
            matrix: MatrixConfig::default(),
            headphone: crate::headphone::HeadphoneMode::Off,
            filter_design: None,
            dac_noise_floor_db: None,
        }
    }
}

/// Errors building a [`Pipeline`].
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// The resampler could not be built.
    #[error("resampler: {0}")]
    Resample(#[from] ResampleError),
    /// The impulse response does not fit the stream.
    #[error("convolution: {0}")]
    Convolution(String),
    /// Unsupported format.
    #[error("{0}")]
    Format(String),
    /// The DSD stage could not be built.
    #[error("dsd: {0}")]
    Dsd(#[from] crate::dsd::DsdError),
    /// The matrix could not be built.
    #[error("matrix: {0}")]
    Matrix(#[from] MatrixError),
}

/// One channel of the room-convolution stage: a [`BlockConvolver`] (CPU engine, or the
/// CUDA engine when the backend has one) behind a small block adapter that gathers whole
/// blocks from the stream and keeps the flush/truncate semantics of
/// [`Convolver::push`] / [`Convolver::flush`].
struct ChannelConvolver {
    engine: Box<dyn BlockConvolver + Send>,
    block: usize,
    /// Input samples not yet forming a whole block.
    pending: Vec<f64>,
    out_block: Vec<f64>,
}

impl ChannelConvolver {
    fn new(engine: Box<dyn BlockConvolver + Send>, block: usize) -> Self {
        Self {
            engine,
            block,
            pending: Vec::with_capacity(block),
            out_block: vec![0.0; block],
        }
    }

    /// Process `input`, appending whatever complete blocks are ready to `out`.
    fn push(&mut self, mut input: &[f64], out: &mut Vec<f64>) {
        while !input.is_empty() {
            let take = (self.block - self.pending.len()).min(input.len());
            self.pending.extend_from_slice(&input[..take]);
            input = &input[take..];
            if self.pending.len() == self.block {
                self.engine
                    .process_block(&self.pending, &mut self.out_block);
                out.extend_from_slice(&self.out_block);
                self.pending.clear();
            }
        }
    }

    /// Push zeros until every input sample has produced its output (the filter tail is
    /// *not* included: the stream length stays equal to the input length), then reset.
    fn flush(&mut self, out: &mut Vec<f64>) {
        let pending = self.pending.len();
        if pending > 0 {
            let zeros = vec![0.0; self.block - pending];
            self.push(&zeros, out);
            out.truncate(out.len() - zeros.len());
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.engine.reset();
    }
}

struct ConvolutionStage {
    convolvers: Vec<ChannelConvolver>,
    split: Vec<Vec<f64>>,
    /// `true` when at least one channel runs on the GPU engine.
    on_gpu: bool,
}

impl ConvolutionStage {
    fn new(
        spec: &ConvolutionSpec,
        channels: usize,
        gpu: &GpuBackend,
    ) -> Result<Self, PipelineError> {
        if spec.channels.is_empty() || spec.channels.iter().any(|c| c.is_empty()) {
            return Err(PipelineError::Convolution(
                "impulse response is empty".into(),
            ));
        }
        let filters: Vec<Arc<SpectralFilter>> = if spec.channels.len() == 1 {
            let filter = Arc::new(SpectralFilter::new(
                &spec.channels[0],
                crate::convolution::choose_block(spec.channels[0].len()),
            ));
            vec![filter; channels]
        } else if spec.channels.len() == channels {
            let block = crate::convolution::choose_block(
                spec.channels.iter().map(|c| c.len()).max().unwrap_or(1),
            );
            spec.channels
                .iter()
                .map(|h| Arc::new(SpectralFilter::new(h, block)))
                .collect()
        } else {
            return Err(PipelineError::Convolution(format!(
                "impulse has {} channels but the stream has {channels}",
                spec.channels.len()
            )));
        };
        let convolvers: Vec<ChannelConvolver> = filters
            .into_iter()
            .map(|f| {
                let block = f.block();
                ChannelConvolver::new(gpu.convolver(f), block)
            })
            .collect();
        let on_gpu = convolvers.iter().any(|c| c.engine.on_gpu());
        Ok(Self {
            convolvers,
            split: vec![Vec::new(); channels],
            on_gpu,
        })
    }

    fn block(&self) -> usize {
        self.convolvers[0].block
    }

    fn run(&mut self, interleaved: &[f64], flush: bool) -> Vec<f64> {
        let ch = self.convolvers.len();
        let frames = interleaved.len() / ch;
        for (c, buf) in self.split.iter_mut().enumerate() {
            buf.clear();
            buf.extend(interleaved.iter().skip(c).step_by(ch).take(frames));
        }
        let mut outs: Vec<Vec<f64>> = vec![Vec::new(); ch];
        let run = |(conv, (inp, out)): (&mut ChannelConvolver, (&Vec<f64>, &mut Vec<f64>))| {
            conv.push(inp, out);
            if flush {
                conv.flush(out);
            }
        };
        // Channels in parallel on the rayon pool for the CPU engine; one device stream at
        // a time for the GPU (whole blocks only, never a half-finished partition).
        if ch > 1 && !self.on_gpu {
            self.convolvers
                .par_iter_mut()
                .zip(self.split.par_iter().zip(outs.par_iter_mut()))
                .for_each(run);
        } else {
            self.convolvers
                .iter_mut()
                .zip(self.split.iter().zip(outs.iter_mut()))
                .for_each(run);
        }
        let n = outs[0].len();
        let mut joined = vec![0.0; n * ch];
        for (c, out) in outs.iter().enumerate() {
            for (j, v) in out.iter().enumerate() {
                joined[j * ch + c] = *v;
            }
        }
        joined
    }

    fn reset(&mut self) {
        self.convolvers.iter_mut().for_each(ChannelConvolver::reset);
    }
}

/// The running chain for one input format.
pub struct Pipeline {
    input: InputFormat,
    config: PipelineConfig,
    volume: Volume,
    eq: Option<Equalizer>,
    convolution: Option<ConvolutionStage>,
    /// Crossfeed / HRTF (stereo only).
    headphone: Option<crate::headphone::HeadphoneStage>,
    /// Routing / crossovers; present only when it changes something. Every stage after it
    /// runs with `out_channels` channels.
    matrix: Option<MatrixStage>,
    /// Channel count after the matrix (= the input's without one).
    out_channels: usize,
    resampler: Option<Resampler>,
    headroom: Option<Headroom>,
    /// Present in DSD output mode: replaces the quantiser.
    dsd: Option<DsdStage>,
    quantizer: Quantizer,
    output_rate: u32,
    scratch: Vec<f64>,
    /// Where the convolution engines run (CUDA or CPU), decided once per configuration.
    gpu: GpuBackend,
}

impl Pipeline {
    /// Build the chain for `input` according to `config`; long filters come from `cache`.
    pub fn new(
        input: InputFormat,
        config: PipelineConfig,
        cache: &CoefficientCache,
    ) -> Result<Self, PipelineError> {
        if input.channels == 0 || input.rate == 0 {
            return Err(PipelineError::Format(
                "input has no channels or no sample rate".into(),
            ));
        }
        if !matches!(config.output_bits, 16 | 24 | 32) {
            return Err(PipelineError::Format(format!(
                "unsupported output bit depth {}",
                config.output_bits
            )));
        }
        let ch = input.channels as usize;
        let bp = config.bit_perfect;
        let volume = if bp { Volume::UNITY } else { config.volume };
        let eq = (!bp && config.eq_enabled && !config.eq.is_empty())
            .then(|| Equalizer::new(input.rate, ch, &config.eq))
            .filter(|e| !e.is_identity());
        // DSD output: the PCM chain runs to the modulator's input rate instead.
        let dsd_mode = !bp && config.output_mode == OutputMode::Dsd;
        let pcm_rate = if bp {
            input.rate
        } else if dsd_mode {
            DsdStage::pcm_input_rate(&config.dsd, crate::dsd::family_base(input.rate))
        } else {
            config.output_rate.unwrap_or(input.rate)
        };
        // Compute backend: probed (and logged) once per `gpu` setting, only when there is
        // a convolution engine to run: the room convolution and the upsampler's FFT
        // polyphase stages run on it (the time-domain rational stage stays on the CPU).
        let has_convolution = !bp && config.convolution.is_some();
        let gpu = if pcm_rate != input.rate || has_convolution {
            GpuBackend::detect(&config.gpu)
        } else {
            GpuBackend::Cpu {
                reason: "nothing to convolve".into(),
            }
        };
        let convolution = match (&config.convolution, bp) {
            (Some(spec), false) => match spec.rate {
                // An impulse made at another rate is resampled to the stream's, so its
                // corrections stay at the same frequencies.
                Some(r) if r != input.rate => {
                    log::info!(
                        "impulse {} resampled from {r} Hz to {} Hz",
                        spec.name,
                        input.rate
                    );
                    let resampled = ConvolutionSpec {
                        name: spec.name.clone(),
                        rate: Some(input.rate),
                        channels: spec
                            .channels
                            .iter()
                            .map(|h| crate::headphone::resample_ir(h, r as f64, input.rate as f64))
                            .collect(),
                    };
                    Some(ConvolutionStage::new(&resampled, ch, &gpu)?)
                }
                _ => Some(ConvolutionStage::new(spec, ch, &gpu)?),
            },
            _ => None,
        };
        let headphone = (!bp && ch == 2)
            .then(|| crate::headphone::HeadphoneStage::new(config.headphone, input.rate))
            .flatten();
        // Matrix: routing, per-output chains, crossovers. Built from the preset; skipped
        // when it is an identity so the chain can stay bit-perfect. From here on the chain
        // has `out_ch` channels.
        let matrix = (!bp && config.matrix.enabled)
            .then(|| MatrixStage::new(&config.matrix, input.rate, ch))
            .transpose()?
            .filter(|m| !m.is_identity());
        let out_ch = matrix
            .as_ref()
            .map(MatrixStage::output_channels)
            .unwrap_or(ch);
        let resampler = if pcm_rate != input.rate {
            let on = gpu.is_gpu().then_some(&gpu as &dyn PolyphaseFactory);
            Some(match &config.filter_design {
                Some(d) => Resampler::with_design(input.rate, pcm_rate, out_ch, d, cache, on)?,
                None => Resampler::new_on(
                    input.rate,
                    pcm_rate,
                    out_ch,
                    config.filter,
                    config.attenuation_db,
                    cache,
                    on,
                )?,
            })
        } else {
            None
        };
        let dsd = dsd_mode
            .then(|| DsdStage::new(pcm_rate, input.rate, out_ch, config.dsd))
            .transpose()?;
        let output_rate = dsd.as_ref().map(DsdStage::dsd_rate).unwrap_or(pcm_rate);
        let dither = if bp { DitherKind::None } else { config.dither };
        let quantizer = Quantizer::for_dac(
            config.output_bits,
            dither,
            out_ch,
            pcm_rate,
            config
                .dac_noise_floor_db
                .unwrap_or(crate::dither::DEFAULT_DAC_FLOOR_DB),
        );
        // Headroom + true-peak meter + soft clipper: the last guard whenever the chain
        // processes anything (never in a bit-perfect chain, so it stays bit-perfect).
        // In DSD mode `dsd_gain_db` is the headroom, so the automatic −3 dB is not stacked.
        let headroom_db = config.headroom_db.unwrap_or_else(|| {
            if dsd_mode {
                0.0
            } else {
                default_headroom_db(resampler.is_some())
            }
        });
        let processing = resampler.is_some()
            || convolution.is_some()
            || eq.is_some()
            || matrix.is_some()
            || headphone.is_some();
        let headroom = (!bp && (headroom_db != 0.0 || processing))
            .then(|| Headroom::new(out_ch, headroom_db, config.soft_clip));
        Ok(Self {
            input,
            config,
            volume,
            eq,
            convolution,
            headphone,
            matrix,
            out_channels: out_ch,
            resampler,
            headroom,
            dsd,
            quantizer,
            output_rate,
            scratch: Vec::new(),
            gpu,
        })
    }

    /// Input format.
    pub fn input(&self) -> &InputFormat {
        &self.input
    }
    /// Configuration the chain was built with.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
    /// Output rate in Hz (the DSD bit rate in DSD mode).
    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }
    /// Output bit depth (1 in DSD mode).
    pub fn output_bits(&self) -> u32 {
        if self.dsd.is_some() {
            1
        } else {
            self.quantizer.bits()
        }
    }
    /// `pcm` or `dsd`: what [`Pipeline::process`] returns.
    pub fn output_mode(&self) -> OutputMode {
        if self.dsd.is_some() {
            OutputMode::Dsd
        } else {
            OutputMode::Pcm
        }
    }
    /// The DSD output stage, in DSD mode.
    pub fn dsd(&self) -> Option<&DsdStage> {
        self.dsd.as_ref()
    }
    /// `(frame rate, valid bits)` of the words [`Pipeline::process`] returns, i.e. what
    /// the device is opened with: PCM `(output_rate, output_bits)`; DoP
    /// `(f_dsd / 16, 24)` – a 24-bit PCM stream to the device; native DSD
    /// `(f_dsd / 32, 32)` words of 32 DSD bits (see `cadenza_dsp::dsd::packing`).
    pub fn device_format(&self) -> (u32, u32) {
        match &self.dsd {
            Some(d) => (d.device_frame_rate(), d.device_word_bits()),
            None => (self.output_rate, self.quantizer.bits()),
        }
    }
    /// Output channel count (after the matrix; equal to the input's without one).
    pub fn channels(&self) -> usize {
        self.out_channels
    }
    /// Input channel count.
    pub fn input_channels(&self) -> usize {
        self.input.channels as usize
    }
    /// The matrix stage, when the chain runs one.
    pub fn matrix(&self) -> Option<&MatrixStage> {
        self.matrix.as_ref()
    }
    /// Current volume.
    pub fn volume(&self) -> Volume {
        self.volume
    }
    /// The headroom / true-peak / soft-clip stage, when the chain has one.
    pub fn headroom(&self) -> Option<&Headroom> {
        self.headroom.as_ref()
    }
    /// The compute backend the convolution engines use (`CUDA (RTX 4060) · 64-bit` or CPU).
    pub fn gpu(&self) -> &GpuBackend {
        &self.gpu
    }
    /// `true` when a stage of this chain actually runs on the GPU (the upsampler's FFT
    /// core and/or the room convolution when the backend is CUDA). A detected card with
    /// nothing to run on it is *not* "in use" and the signal path says so.
    pub fn gpu_in_use(&self) -> bool {
        self.gpu_use().any()
    }

    /// Which engines of this chain run on the GPU right now (an upsampler that fell back
    /// to the CPU after a device error no longer counts).
    pub fn gpu_use(&self) -> GpuUse {
        GpuUse {
            upsampling: self.resampler.as_ref().is_some_and(Resampler::on_gpu),
            convolution: self.convolution.as_ref().is_some_and(|c| c.on_gpu),
        }
    }

    /// Change the volume without rebuilding (ignored in bit-perfect mode).
    pub fn set_volume(&mut self, volume: Volume) {
        if !self.config.bit_perfect {
            self.volume = volume;
            self.config.volume = volume;
        }
    }

    /// Change the EQ without rebuilding. Returns `false` when that is not possible
    /// (the EQ turns on or off, which changes the rest of the chain): rebuild then.
    pub fn set_eq(&mut self, bands: &[Band], enabled: bool) -> bool {
        let wanted = !self.config.bit_perfect
            && enabled
            && bands.iter().take(crate::eq::MAX_BANDS).any(|b| b.enabled);
        match (&mut self.eq, wanted) {
            (Some(eq), true) => {
                eq.retune(bands);
                self.config.eq = bands.to_vec();
                self.config.eq_enabled = enabled;
                true
            }
            (None, false) => {
                self.config.eq = bands.to_vec();
                self.config.eq_enabled = enabled;
                true
            }
            _ => false,
        }
    }

    /// `true` only when nothing in the chain alters any sample.
    pub fn bit_perfect(&self) -> bool {
        self.volume.is_unity()
            && self.eq.is_none()
            && self.convolution.is_none()
            && self.headphone.is_none()
            && self.matrix.is_none()
            && self.resampler.is_none()
            && self.headroom.as_ref().is_none_or(Headroom::is_transparent)
            && self.dsd.is_none()
            && !self.quantizer.kind().adds_noise()
            && self.quantizer.bits() >= self.input.bits
    }

    /// Processing latency in seconds (what the player subtracts from its position).
    pub fn latency_seconds(&self) -> f64 {
        let conv = self
            .convolution
            .as_ref()
            .map(|c| c.block() as f64 / self.input.rate as f64)
            .unwrap_or(0.0);
        let conv = conv
            + self
                .headphone
                .as_ref()
                .map_or(0.0, |h| h.latency_frames() as f64 / self.input.rate as f64);
        conv + self
            .matrix
            .as_ref()
            .map(MatrixStage::latency_seconds)
            .unwrap_or(0.0)
            + self
                .resampler
                .as_ref()
                .map(Resampler::latency_seconds)
                .unwrap_or(0.0)
            + self
                .dsd
                .as_ref()
                .map(DsdStage::latency_seconds)
                .unwrap_or(0.0)
    }

    /// Run interleaved `f64` frames through the chain; returns interleaved integers at the
    /// output rate (possibly empty while the FFT stages gather a block).
    pub fn process(&mut self, pcm: &[f64]) -> Vec<i32> {
        self.run(pcm, false)
    }

    /// Push out everything still inside the chain (end of stream) and reset it.
    pub fn flush(&mut self) -> Vec<i32> {
        let out = self.run(&[], true);
        self.reset();
        out
    }

    /// Drop all state (seek).
    pub fn reset(&mut self) {
        if let Some(eq) = &mut self.eq {
            eq.reset();
        }
        if let Some(c) = &mut self.convolution {
            c.reset();
        }
        if let Some(h) = &mut self.headphone {
            h.reset();
        }
        if let Some(m) = &mut self.matrix {
            m.reset();
        }
        if let Some(r) = &mut self.resampler {
            r.reset();
        }
        if let Some(h) = &mut self.headroom {
            h.reset();
        }
        if let Some(d) = &mut self.dsd {
            d.reset();
        }
        self.quantizer.reset();
    }

    fn run(&mut self, pcm: &[f64], flush: bool) -> Vec<i32> {
        self.scratch.clear();
        self.scratch.extend_from_slice(pcm);
        self.volume.apply(&mut self.scratch);
        if let Some(eq) = &mut self.eq {
            eq.process(&mut self.scratch);
        }
        if let Some(c) = &mut self.convolution {
            self.scratch = c.run(&self.scratch, flush);
        }
        if let Some(h) = &mut self.headphone {
            self.scratch = h.run(&self.scratch, flush);
        }
        if let Some(m) = &mut self.matrix {
            self.scratch = m.process(&self.scratch, flush);
        }
        if let Some(r) = &mut self.resampler {
            let mut out = r.push(&self.scratch);
            if flush {
                out.extend(r.flush());
            }
            self.scratch = out;
        }
        if let Some(h) = &mut self.headroom {
            h.process(&mut self.scratch);
        }
        if let Some(d) = &mut self.dsd {
            let mut words = d.process(&self.scratch);
            if flush {
                words.extend(d.flush());
            }
            return words;
        }
        let mut ints = Vec::with_capacity(self.scratch.len());
        self.quantizer.process(&self.scratch, &mut ints);
        ints
    }

    /// The upsampling filter's id (designed or catalogue).
    fn filter_id(&self) -> String {
        match &self.config.filter_design {
            Some(d) => d.id(),
            None => self.config.filter.id(),
        }
    }

    fn filter_label(&self) -> String {
        match &self.config.filter_design {
            Some(d) => d.label(),
            None => self.config.filter.label(),
        }
    }

    /// Describe the chain.
    pub fn signal_path(&self) -> SignalPath {
        let bp = self.bit_perfect();
        let bypass = |what: &str| {
            if self.config.bit_perfect {
                format!("{what} · bypassed (bit-perfect)")
            } else {
                what.to_string()
            }
        };
        let mut stages = Vec::new();

        let v = self.volume;
        stages.push(Stage {
            id: "volume".into(),
            name: "Volume".into(),
            detail: if v.muted {
                "muted".into()
            } else if v.is_unity() {
                bypass("0 dB")
            } else {
                format!("{:+.1} dB", v.gain_db)
            },
            params: json!({ "gain_db": v.gain_db, "muted": v.muted }),
            active: !v.is_unity(),
        });

        let eq_bands: Vec<&Band> = self.config.eq.iter().filter(|b| b.enabled).collect();
        stages.push(Stage {
            id: "eq".into(),
            name: "Parametric EQ".into(),
            detail: match &self.eq {
                Some(eq) => format!(
                    "{} band{}",
                    eq.active_stages(),
                    if eq.active_stages() == 1 { "" } else { "s" }
                ),
                None if eq_bands.is_empty() => "no bands".into(),
                None => bypass(&format!("{} bands", eq_bands.len())),
            },
            params: json!({ "bands": self.config.eq, "enabled": self.config.eq_enabled }),
            active: self.eq.is_some(),
        });

        if let Some(spec) = &self.config.convolution {
            let taps = spec.channels.iter().map(|c| c.len()).max().unwrap_or(0);
            stages.push(Stage {
                id: "convolution".into(),
                name: "Convolution".into(),
                detail: bypass(&format!(
                    "{} · {taps} taps{}",
                    spec.name,
                    spec.rate.map(|r| format!(" @ {} kHz", rate_label(r))).unwrap_or_default()
                )),
                params: json!({ "name": spec.name, "taps": taps, "rate": spec.rate, "channels": spec.channels.len() }),
                active: self.convolution.is_some(),
            });
        }

        if self.config.headphone != crate::headphone::HeadphoneMode::Off {
            stages.push(Stage {
                id: "headphone".into(),
                name: "Headphones".into(),
                detail: if self.headphone.is_some() {
                    self.config.headphone.label().into()
                } else {
                    bypass(&format!(
                        "{} · stereo sources only",
                        self.config.headphone.label()
                    ))
                },
                params: json!({ "mode": self.config.headphone.id() }),
                active: self.headphone.is_some(),
            });
        }

        match &self.matrix {
            Some(m) => stages.extend(m.stages(&self.config.matrix.preset, true)),
            None => stages.push(Stage {
                id: "matrix".into(),
                name: "Matrix".into(),
                detail: if self.config.matrix.enabled {
                    bypass(&format!(
                        "{} → {} ch · identity",
                        self.input.channels, self.input.channels
                    ))
                } else {
                    "off".into()
                },
                params: json!({
                    "preset": self.config.matrix.preset,
                    "enabled": self.config.matrix.enabled,
                    "input_channels": self.input.channels,
                    "output_channels": self.input.channels,
                    "identity": true,
                    "outputs": [],
                }),
                active: false,
            }),
        }

        match &self.resampler {
            Some(r) => {
                let (l, m) = r.ratio();
                let ratio = if m == 1 {
                    format!("×{l}")
                } else {
                    format!("×{l}/{m}")
                };
                let engine = r.describe();
                stages.push(Stage {
                    id: "resampler".into(),
                    name: "Upsampler".into(),
                    detail: format!(
                        "{} · {ratio} → {} kHz · {engine}",
                        self.filter_label(),
                        rate_label(r.out_rate())
                    ),
                    params: json!({
                        "filter": self.filter_id(),
                        "label": self.filter_label(),
                        "taps": r.spec().taps,
                        "phase": r.spec().phase.name(),
                        "apodizing": match &self.config.filter_design {
                            Some(d) => d.cutoff < 1.0,
                            None => self.config.filter.apodizing,
                        },
                        "attenuation_db": r.spec().attenuation_db,
                        "transition_hz": r.spec().transition_width() * r.in_rate() as f64 * l as f64,
                        "cutoff_hz": r.spec().cutoff * r.in_rate() as f64 * l as f64,
                        "l": l, "m": m,
                        "engine": r.engine(),
                        "device": r.device(),
                        "on_gpu": r.on_gpu(),
                        "stages": r.stages(),
                        "window": r.spec().window.name(),
                        "family": if self.config.filter_design.is_some() {
                            "custom"
                        } else {
                            self.config.filter.family.id()
                        },
                        "stopband_db": r.spec().attenuation_db,
                        "latency_seconds": r.latency_seconds(),
                    }),
                    active: true,
                });
            }
            None => stages.push(Stage {
                id: "resampler".into(),
                name: "Upsampler".into(),
                detail: bypass(&format!("{} kHz unchanged", rate_label(self.input.rate))),
                params: json!({ "filter": self.filter_id(), "rate": self.input.rate }),
                active: false,
            }),
        }

        stages.push(self.gpu.stage_for(&self.config.gpu, self.gpu_use()));

        match &self.headroom {
            Some(h) => stages.push(Stage {
                id: "headroom".into(),
                name: "Headroom".into(),
                detail: h.detail(),
                params: h.report(),
                active: !h.is_transparent(),
            }),
            None => stages.push(Stage {
                id: "headroom".into(),
                name: "Headroom".into(),
                detail: bypass("0 dB"),
                params: json!({ "gain_db": 0.0, "soft_clip": false, "clip_count": 0, "clip_engaged": false }),
                active: false,
            }),
        }

        if let Some(d) = &self.dsd {
            stages.extend(d.stages());
        }

        let q = &self.quantizer;
        let depth_change = q.bits() < self.input.bits;
        if let Some(d) = &self.dsd {
            stages.push(Stage {
                id: "dither".into(),
                name: "Dither".into(),
                detail: format!("not used · 1-bit {}", d.config().rate.label()),
                params: json!({ "kind": "none", "label": "none", "bits": 1 }),
                active: false,
            });
        } else {
            stages.push(Stage {
                id: "dither".into(),
                name: "Dither".into(),
                detail: format!(
                    "{}{} → {}-bit{}",
                    match q.auto_floor() {
                        Some(f) => format!("Auto ({f:.0} dBFS DAC floor): "),
                        None => String::new(),
                    },
                    q.kind().label(),
                    q.bits(),
                    if q.kind().adds_noise() || depth_change { "" } else { " · exact" }
                ),
                params: json!({
                    "kind": q.kind().id(), "label": q.kind().label(), "bits": q.bits(),
                    "shaper": q.coefficients(), "rate": q.rate(), "clipped": q.clipped(),
                    "auto": q.auto_floor().is_some(), "dac_floor_db": q.auto_floor(),
                    "audible_noise_dbfs": crate::dither::audible_noise_dbfs(q.kind(), q.bits(), q.rate()),
                }),
                active: q.kind().adds_noise() || depth_change,
            });
        }

        let input = Endpoint {
            format: self.input.codec.clone(),
            rate: self.input.rate,
            bits: self.input.bits,
            channels: self.input.channels,
            device: None,
        };
        let output = Endpoint {
            format: match &self.dsd {
                Some(d) => d.config().rate.label(),
                None => "PCM".into(),
            },
            rate: self.output_rate,
            bits: self.output_bits(),
            channels: self.out_channels as u32,
            device: self.config.output_device.clone(),
        };
        let label = match &self.dsd {
            // e.g. `FLAC 16/44.1 → 705.6k (sinc-l-1m) → ASDM7 → DSD256 (DoP)`
            Some(d) => format!(
                "{} {} → {}k ({}) → {} → {}",
                input.format,
                input.short(),
                rate_label(d.pcm_rate()),
                self.filter_id(),
                d.modulator().label(),
                d.output_label()
            ),
            None => SignalPath::make_label(&input, &output, bp),
        };
        SignalPath {
            input,
            stages,
            output,
            bit_perfect: bp,
            label,
            latency_seconds: self.latency_seconds(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fir::Phase;
    use std::f64::consts::PI;

    fn input(rate: u32, bits: u32) -> InputFormat {
        InputFormat {
            rate,
            bits,
            channels: 2,
            codec: "FLAC".into(),
        }
    }

    fn grid_samples(bits: u32, n: usize) -> (Vec<i32>, Vec<f64>) {
        let scale = (1u64 << (bits - 1)) as f64;
        let ints: Vec<i32> = (0..n)
            .map(|i| ((i as i64 * 7919) % 65536 - 32768) as i32)
            .collect();
        let f = ints.iter().map(|&i| i as f64 / scale).collect();
        (ints, f)
    }

    #[test]
    fn bit_perfect_request_bypasses_everything() {
        let cfg = PipelineConfig {
            bit_perfect: true,
            volume: Volume {
                gain_db: -10.0,
                muted: false,
            },
            eq: vec![Band::peaking(1000.0, 6.0, 1.0)],
            output_rate: Some(176400),
            output_bits: 16,
            dither: DitherKind::Tpdf,
            ..Default::default()
        };
        let mut p = Pipeline::new(input(44100, 16), cfg, &CoefficientCache::disabled()).unwrap();
        let (ints, f) = grid_samples(16, 2000);
        let mut out = p.process(&f);
        out.extend(p.flush());
        assert_eq!(out, ints);
        assert!(p.bit_perfect());
        let path = p.signal_path();
        assert!(path.bit_perfect);
        assert_eq!(path.label, "FLAC 16/44.1 · bit-perfect");
        assert!(path.stages.iter().all(|s| !s.active));
        assert_eq!(path.output.rate, 44100);
    }

    #[test]
    fn untouched_samples_are_reported_bit_perfect_even_without_the_flag() {
        let cfg = PipelineConfig {
            output_bits: 24,
            dither: DitherKind::None,
            ..Default::default()
        };
        let mut p = Pipeline::new(input(44100, 16), cfg, &CoefficientCache::disabled()).unwrap();
        assert!(p.bit_perfect());
        let (ints, f) = grid_samples(16, 100);
        let out = p.process(&f);
        assert!(out.iter().zip(&ints).all(|(o, i)| *o == i << 8));
        p.set_volume(Volume {
            gain_db: -6.0,
            muted: false,
        });
        assert!(!p.bit_perfect());
        assert!(!p.signal_path().bit_perfect);
        p.set_volume(Volume::UNITY);
        assert!(p.bit_perfect());
    }

    #[test]
    fn dither_or_truncation_is_not_bit_perfect() {
        let cfg = PipelineConfig {
            output_bits: 16,
            dither: DitherKind::None,
            ..Default::default()
        };
        let p = Pipeline::new(input(44100, 24), cfg, &CoefficientCache::disabled()).unwrap();
        assert!(!p.bit_perfect());
        let cfg = PipelineConfig {
            output_bits: 24,
            dither: DitherKind::Tpdf,
            ..Default::default()
        };
        let p = Pipeline::new(input(44100, 16), cfg, &CoefficientCache::disabled()).unwrap();
        assert!(!p.bit_perfect());
    }

    #[test]
    fn eq_retunes_in_place_and_asks_for_a_rebuild_when_it_turns_on_or_off() {
        let cfg = PipelineConfig {
            output_bits: 24,
            eq: vec![Band::peaking(1000.0, -6.0, 1.0)],
            ..Default::default()
        };
        let mut p = Pipeline::new(input(44100, 16), cfg, &CoefficientCache::disabled()).unwrap();
        let eq = |p: &Pipeline| p.eq.as_ref().unwrap().response_db(1000.0);
        assert!((eq(&p) + 6.0).abs() < 1e-6);
        assert!(p.set_eq(&[Band::peaking(1000.0, 3.0, 1.0)], true));
        assert!((eq(&p) - 3.0).abs() < 1e-6);
        // Off, or every band off: the chain changes shape, so the caller rebuilds.
        assert!(!p.set_eq(&[Band::peaking(1000.0, 3.0, 1.0)], false));
        let mut off = Band::peaking(1000.0, 3.0, 1.0);
        off.enabled = false;
        assert!(!p.set_eq(&[off], true));
    }

    #[test]
    fn full_chain_upsamples_with_volume_and_eq() {
        let cfg = PipelineConfig {
            volume: Volume {
                gain_db: -6.0,
                muted: false,
            },
            eq: vec![
                Band::peaking(10000.0, 0.0, 1.0),
                Band::peaking(100.0, 3.0, 2.0),
            ],
            filter: FilterKind::sinc(Phase::Linear, false, 1 << 12),
            output_rate: Some(176400),
            output_bits: 24,
            dither: DitherKind::Lipshitz,
            headroom_db: Some(0.0),
            ..Default::default()
        };
        let mut p = Pipeline::new(input(44100, 16), cfg, &CoefficientCache::disabled()).unwrap();
        assert_eq!(p.output_rate(), 176400);
        let n = 44100;
        let pcm: Vec<f64> = (0..n)
            .flat_map(|i| {
                let v = 0.5 * (2.0 * PI * 1000.0 * i as f64 / 44100.0).sin();
                [v, -v]
            })
            .collect();
        let mut out = Vec::new();
        for chunk in pcm.chunks(2 * 1000) {
            out.extend(p.process(chunk));
        }
        out.extend(p.flush());
        assert_eq!(out.len(), n * 4 * 2);
        let scale = (1u64 << 23) as f64;
        let mid = &out[out.len() / 4..out.len() * 3 / 4];
        let left: Vec<f64> = mid.iter().step_by(2).map(|&v| v as f64 / scale).collect();
        let rms = (left.iter().map(|v| v * v).sum::<f64>() / left.len() as f64).sqrt();
        let db = 20.0 * (rms * 2f64.sqrt() / 0.5).log10();
        assert!((db + 6.0).abs() < 0.05, "level {db} dB");
        let path = p.signal_path();
        assert!(!path.bit_perfect);
        assert_eq!(path.label, "FLAC 16/44.1 → 24/176.4");
        let active: Vec<&str> = path
            .stages
            .iter()
            .filter(|s| s.active && s.id != "gpu")
            .map(|s| s.id.as_str())
            .collect();
        assert_eq!(active, ["volume", "eq", "resampler", "dither"]);
        // The compute stage is active exactly when a GPU runs the upsampler (a machine with
        // one, or CI's software Vulkan with CADENZA_WGPU_SOFTWARE=1).
        let gpu = path.stages.iter().find(|s| s.id == "gpu").unwrap();
        assert_eq!(gpu.active, p.gpu_in_use());
        assert!(path.latency_seconds > 0.0);
        serde_json::to_string(&path).unwrap();
    }

    #[test]
    fn headphone_stage_runs_on_stereo_only_and_shows_in_the_signal_path() {
        use crate::headphone::HeadphoneMode;
        let cache = CoefficientCache::disabled();
        let input = InputFormat {
            rate: 44100,
            bits: 16,
            channels: 2,
            codec: "FLAC".into(),
        };
        let mut p = Pipeline::new(
            input.clone(),
            PipelineConfig {
                headphone: HeadphoneMode::CrossfeedMeier,
                ..Default::default()
            },
            &cache,
        )
        .unwrap();
        assert!(!p.bit_perfect());
        // A hard-left low tone reaches the right channel.
        let pcm: Vec<f64> = (0..8820)
            .flat_map(|i| [0.5 * (i as f64 * 0.01).sin(), 0.0])
            .collect();
        let out = p.process(&pcm);
        assert!(out.iter().skip(2001).step_by(2).any(|w| w.abs() > 1000));
        let sp = p.signal_path();
        let st = sp.stages.iter().find(|s| s.id == "headphone").unwrap();
        assert!(st.active && st.detail.contains("Meier"));
        // HRTF, and its latency counted.
        let p = Pipeline::new(
            input.clone(),
            PipelineConfig {
                headphone: HeadphoneMode::Hrtf30,
                ..Default::default()
            },
            &cache,
        )
        .unwrap();
        assert!(p.latency_seconds() > 0.0);
        // Mono: shown as bypassed.
        let p = Pipeline::new(
            InputFormat {
                channels: 1,
                ..input
            },
            PipelineConfig {
                headphone: HeadphoneMode::Hrtf30,
                ..Default::default()
            },
            &cache,
        )
        .unwrap();
        let sp = p.signal_path();
        let st = sp.stages.iter().find(|s| s.id == "headphone").unwrap();
        assert!(!st.active && st.detail.contains("stereo sources only"));
    }

    #[test]
    fn an_impulse_at_another_rate_is_resampled_to_the_stream() {
        // A 48 kHz correction with a +6 dB shelf below 1 kHz, on a 44.1 kHz stream: the
        // boost must stay at the same frequencies.
        let h48 = crate::room::min_phase(4096, 48000.0, |f| if f < 1000.0 { 6.0 } else { 0.0 });
        let spec = Arc::new(ConvolutionSpec {
            name: "room".into(),
            rate: Some(48000),
            channels: vec![h48],
        });
        let cache = CoefficientCache::disabled();
        let mut p = Pipeline::new(
            InputFormat {
                rate: 44100,
                bits: 24,
                channels: 1,
                codec: "PCM".into(),
            },
            PipelineConfig {
                convolution: Some(spec),
                output_bits: 32,
                dither: DitherKind::None,
                headroom_db: Some(-12.0),
                ..Default::default()
            },
            &cache,
        )
        .unwrap();
        let level = |p: &mut Pipeline, f: f64| {
            p.reset();
            let x: Vec<f64> = (0..44100)
                .map(|n| 0.1 * (2.0 * std::f64::consts::PI * f * n as f64 / 44100.0).sin())
                .collect();
            let mut y = p.process(&x);
            y.extend(p.flush());
            let v: Vec<f64> = y[22050..40000].iter().map(|w| *w as f64).collect();
            (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
        };
        let lo = level(&mut p, 300.0);
        let hi = level(&mut p, 5000.0);
        let db = 20.0 * (lo / hi).log10();
        assert!((db - 6.0).abs() < 0.3, "{db:.2} dB");
        // Just below the 1 kHz edge (at 48 kHz) is still boosted: not shifted by 9 %.
        let edge = 20.0 * (level(&mut p, 900.0) / hi).log10();
        assert!(edge > 3.0, "{edge:.2} dB at 900 Hz");
    }

    #[test]
    fn convolution_stage_applies_the_impulse() {
        let spec = Arc::new(ConvolutionSpec {
            name: "avg".into(),
            rate: Some(48000),
            channels: vec![vec![0.5, 0.5]],
        });
        let cfg = PipelineConfig {
            convolution: Some(spec),
            output_bits: 32,
            dither: DitherKind::None,
            ..Default::default()
        };
        let mut p = Pipeline::new(input(48000, 24), cfg, &CoefficientCache::disabled()).unwrap();
        assert!(!p.bit_perfect());
        let pcm: Vec<f64> = (0..600)
            .map(|i| if i % 2 == 0 { 0.5 } else { 0.0 })
            .collect();
        let mut out = p.process(&pcm);
        out.extend(p.flush());
        assert_eq!(out.len(), 600);
        // Moving average over two frames of the left channel: 0.25 first, then 0.5 forever.
        let scale = (1u64 << 31) as f64;
        assert!((out[0] as f64 / scale - 0.25).abs() < 1e-9);
        for f in 1..300 {
            assert!((out[2 * f] as f64 / scale - 0.5).abs() < 1e-9, "frame {f}");
            assert!((out[2 * f + 1] as f64 / scale).abs() < 1e-9);
        }
        let path = p.signal_path();
        assert!(path
            .stages
            .iter()
            .any(|s| s.id == "convolution" && s.active));
    }

    #[test]
    fn convolution_block_adapter_matches_the_streaming_convolver_and_reports_the_backend() {
        use crate::convolution::Convolver;
        // Two different impulses (one per channel), long enough for several partitions,
        // fed in ragged chunks: the block adapter around the backend's `BlockConvolver`
        // must reproduce the CPU streaming convolver sample for sample, including the
        // flush/truncate semantics (stream length == input length) and a reset.
        let h0: Vec<f64> = (0..700)
            .map(|i| ((i * 7) % 13) as f64 / 13.0 - 0.5)
            .collect();
        let h1: Vec<f64> = (0..650)
            .map(|i| ((i * 5) % 11) as f64 / 11.0 - 0.5)
            .collect();
        let spec = Arc::new(ConvolutionSpec {
            name: "pair".into(),
            rate: None,
            channels: vec![h0.clone(), h1.clone()],
        });
        let gpu = GpuBackend::detect(&GpuConfig::default());
        let mut stage = ConvolutionStage::new(&spec, 2, &gpu).unwrap();
        let block = stage.block();
        let mut reference = [
            Convolver::with_block(&h0, block),
            Convolver::with_block(&h1, block),
        ];
        let n = 5000;
        let pcm: Vec<f64> = (0..2 * n)
            .map(|i| ((i * 37) % 101) as f64 / 101.0 - 0.5)
            .collect();
        for round in 0..2 {
            let mut got = Vec::new();
            let mut pos = 0;
            for (k, len) in [1usize, 333, 64, 1000, 7, 2000, 5000].iter().enumerate() {
                let end = (pos + 2 * len).min(pcm.len());
                let last = k == 6 || end == pcm.len();
                got.extend(stage.run(&pcm[pos..end], last));
                pos = end;
                if last {
                    break;
                }
            }
            assert_eq!(got.len(), 2 * n, "round {round}: stream length");
            let mut want = [Vec::new(), Vec::new()];
            for c in 0..2 {
                let inp: Vec<f64> = pcm.iter().skip(c).step_by(2).copied().collect();
                reference[c].push(&inp, &mut want[c]);
                reference[c].flush(&mut want[c]);
            }
            for f in 0..n {
                for c in 0..2 {
                    assert!(
                        (got[2 * f + c] - want[c][f]).abs() < 1e-12,
                        "round {round} frame {f} ch {c}: {} vs {}",
                        got[2 * f + c],
                        want[c][f]
                    );
                }
            }
            stage.reset();
        }
        // The `gpu` stage is honest about what this chain runs on: on a machine without a
        // CUDA device the backend is the CPU, nothing is "in use" and the stage is inactive;
        // `used` is only ever true when a convolver actually sits on the card.
        assert_eq!(stage.on_gpu, gpu.is_gpu());
        let st = gpu.stage(&GpuConfig::default(), stage.on_gpu);
        assert_eq!(st.active, gpu.is_gpu());
        assert_eq!(st.params["used"], gpu.is_gpu());
        if !gpu.is_gpu() {
            assert!(st.detail.starts_with("CPU · 64-bit"), "{}", st.detail);
        }
        // Without room correction the GPU is claimed only for the upsampler, and only when
        // its FFT core really sits on the card.
        let cfg = PipelineConfig {
            output_rate: Some(96_000),
            filter: FilterKind::sinc(Phase::Linear, false, 1 << 12),
            ..Default::default()
        };
        let p = Pipeline::new(input(48000, 24), cfg, &CoefficientCache::disabled()).unwrap();
        assert_eq!(p.gpu_in_use(), gpu.is_gpu());
        assert!(!p.gpu_use().convolution);
        let path = p.signal_path();
        let g = path.stages.iter().find(|s| s.id == "gpu").unwrap();
        assert_eq!(g.active, gpu.is_gpu());
        assert_eq!(g.params["used"], gpu.is_gpu());
        assert_eq!(g.params["upsampling"], gpu.is_gpu());
        let r = path.stages.iter().find(|s| s.id == "resampler").unwrap();
        assert_eq!(r.params["on_gpu"], gpu.is_gpu());
        if gpu.is_gpu() {
            assert!(g.detail.ends_with("· upsampling"), "{}", g.detail);
        }
    }

    #[test]
    fn dsd_output_mode_runs_the_modulator_and_reports_it() {
        use crate::dsd::packing::unpack_dop;
        use crate::dsd::{DsdModulator, DsdRate, OutputMode};
        let cfg = PipelineConfig {
            filter: FilterKind::sinc(Phase::Linear, false, 1 << 12),
            output_mode: OutputMode::Dsd,
            dsd: crate::dsd::DsdConfig {
                rate: DsdRate::Dsd64,
                modulator: DsdModulator::Asdm5,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut p = Pipeline::new(input(44100, 16), cfg, &CoefficientCache::disabled()).unwrap();
        assert_eq!(p.output_rate(), 2_822_400);
        assert_eq!(p.output_bits(), 1);
        assert_eq!(p.device_format(), (176_400, 24));
        assert_eq!(p.output_mode(), OutputMode::Dsd);
        assert!(!p.bit_perfect());
        // The automatic −3 dB headroom is not stacked on top of dsd_gain_db.
        assert!(p.headroom().is_none_or(|h| h.gain_db() == 0.0));
        let n = 44100;
        let pcm: Vec<f64> = (0..n)
            .flat_map(|i| {
                let v = 0.5 * (2.0 * PI * 1000.0 * i as f64 / 44100.0).sin();
                [v, v]
            })
            .collect();
        let mut words = Vec::new();
        for chunk in pcm.chunks(2 * 1000) {
            words.extend(p.process(chunk));
        }
        words.extend(p.flush());
        // 1 s of DSD64 as DoP: 176 400 frames × 2 channels.
        assert_eq!(words.len(), 176_400 * 2);
        let planar = unpack_dop(&words, 2).expect("valid DoP markers");
        let ones: u32 = planar[0].iter().map(|b| b.count_ones()).sum();
        assert!((ones as f64 / (planar[0].len() * 8) as f64 - 0.5).abs() < 0.01);
        let path = p.signal_path();
        assert_eq!(path.output.format, "DSD64");
        assert_eq!(path.output.rate, 2_822_400);
        assert_eq!(path.output.bits, 1);
        assert_eq!(
            path.label,
            "FLAC 16/44.1 → 352.8k (sinc-l-4k) → ASDM5 → DSD64 (DoP)"
        );
        let ids: Vec<&str> = path.stages.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.contains(&"dsd-oversampler") && ids.contains(&"modulator"));
        let m = path.stages.iter().find(|s| s.id == "modulator").unwrap();
        assert!(m.active && m.detail.contains("ASDM5") && m.detail.contains("DSD64 (DoP)"));
        assert_eq!(m.params["guard_engagements"].as_u64(), Some(0));
        let d = path.stages.iter().find(|s| s.id == "dither").unwrap();
        assert!(!d.active);
        assert!(path.latency_seconds > 0.0);
        serde_json::to_string(&path).unwrap();

        // The summary names the modulator that runs: asdm7ec8 above DSD256 is asdm7ec.
        let cfg = PipelineConfig {
            filter: FilterKind::sinc(Phase::Linear, false, 1 << 12),
            output_mode: OutputMode::Dsd,
            dsd: crate::dsd::DsdConfig {
                rate: DsdRate::Dsd512,
                modulator: DsdModulator::Asdm7ec8,
                ..Default::default()
            },
            ..Default::default()
        };
        let p = Pipeline::new(input(44100, 16), cfg, &CoefficientCache::disabled()).unwrap();
        let label = p.signal_path().label;
        assert!(
            label.contains("→ ASDM7EC →") && !label.contains("ASDM7EC8"),
            "{label}"
        );
    }

    #[test]
    fn headroom_defaults_to_minus_3_db_when_upsampling_and_reports_the_clipper() {
        let cfg = PipelineConfig {
            filter: FilterKind::sinc(Phase::Linear, false, 1 << 12),
            output_rate: Some(176400),
            output_bits: 32,
            dither: DitherKind::None,
            ..Default::default()
        };
        let mut p = Pipeline::new(input(44100, 24), cfg, &CoefficientCache::disabled()).unwrap();
        assert_eq!(p.headroom().unwrap().gain_db(), -3.0);
        let n = 20_000;
        let pcm: Vec<f64> = (0..n)
            .flat_map(|i| {
                let v = 0.9 * (2.0 * PI * 1000.0 * i as f64 / 44100.0).sin();
                [v, v]
            })
            .collect();
        let mut out = p.process(&pcm);
        out.extend(p.flush());
        let scale = (1u64 << 31) as f64;
        let peak = out
            .iter()
            .map(|v| (*v as f64 / scale).abs())
            .fold(0.0, f64::max);
        let peak_db = 20.0 * peak.log10();
        assert!(
            (peak_db - (20.0 * 0.9f64.log10() - 3.0)).abs() < 0.05,
            "peak {peak_db} dB"
        );
        let path = p.signal_path();
        let hr = path.stages.iter().find(|s| s.id == "headroom").unwrap();
        assert!(hr.active);
        assert!(hr.detail.starts_with("-3.0 dB"), "{}", hr.detail);
        assert!(hr.detail.contains("clipper idle"), "{}", hr.detail);
        assert_eq!(hr.params["clip_count"].as_u64(), Some(0));
        // Note: the meters were reset by flush(); a live chain reports the running track.

        // 0 dB headroom on a +1 dBFS-peaking source: the soft clipper engages and says so.
        let cfg = PipelineConfig {
            filter: FilterKind::sinc(Phase::Linear, false, 1 << 12),
            output_rate: Some(176400),
            output_bits: 32,
            dither: DitherKind::None,
            headroom_db: Some(0.0),
            ..Default::default()
        };
        let mut p = Pipeline::new(input(44100, 24), cfg, &CoefficientCache::disabled()).unwrap();
        let hot: Vec<f64> = pcm.iter().map(|v| v * 1.25).collect();
        let out = p.process(&hot);
        assert!(out.iter().all(|v| *v != i32::MAX && *v != i32::MIN));
        let path = p.signal_path();
        let hr = path.stages.iter().find(|s| s.id == "headroom").unwrap();
        assert!(hr.active && hr.detail.contains("engaged"), "{}", hr.detail);
        assert!(hr.params["clip_count"].as_u64().unwrap() > 0);
        assert!(hr.params["true_peak_dbtp"].as_f64().unwrap() > 0.0);
        let d = path.stages.iter().find(|s| s.id == "dither").unwrap();
        assert_eq!(d.params["rate"].as_u64(), Some(176400));

        // No processing at all: no headroom stage, still bit-perfect.
        let cfg = PipelineConfig {
            output_bits: 24,
            dither: DitherKind::None,
            ..Default::default()
        };
        let p = Pipeline::new(input(44100, 16), cfg, &CoefficientCache::disabled()).unwrap();
        assert!(p.headroom().is_none() && p.bit_perfect());
        let hr = p
            .signal_path()
            .stages
            .into_iter()
            .find(|s| s.id == "headroom")
            .unwrap();
        assert!(!hr.active);
    }

    #[test]
    fn matrix_changes_the_channel_count_and_is_reported_per_output() {
        use crate::matrix::MatrixConfig;
        let cfg = PipelineConfig {
            matrix: MatrixConfig::preset("2.1"),
            output_bits: 32,
            dither: DitherKind::None,
            headroom_db: Some(0.0),
            ..Default::default()
        };
        let mut p = Pipeline::new(input(48000, 24), cfg, &CoefficientCache::disabled()).unwrap();
        assert_eq!(p.input_channels(), 2);
        assert_eq!(p.channels(), 3);
        assert!(p.matrix().is_some());
        assert!(!p.bit_perfect());
        let n = 48000;
        let pcm: Vec<f64> = (0..n)
            .flat_map(|i| {
                let v = 0.5 * (2.0 * PI * 20.0 * i as f64 / 48000.0).sin();
                [v, v]
            })
            .collect();
        let mut out = Vec::new();
        for chunk in pcm.chunks(2 * 1000) {
            out.extend(p.process(chunk));
        }
        out.extend(p.flush());
        assert_eq!(out.len(), n * 3);
        let scale = (1u64 << 31) as f64;
        let sub: Vec<f64> = out[3 * n / 2..]
            .iter()
            .skip(2)
            .step_by(3)
            .map(|v| *v as f64 / scale)
            .collect();
        let rms = (sub.iter().map(|v| v * v).sum::<f64>() / sub.len() as f64).sqrt();
        let db = 20.0 * (rms * 2f64.sqrt() / 0.5).log10();
        // Sub = L −6 dB + R −6 dB = the source level; LR24 @ 80 Hz is −0.03 dB at 20 Hz.
        assert!(db.abs() < 0.06, "sub level {db} dB");
        let path = p.signal_path();
        assert_eq!(path.input.channels, 2);
        assert_eq!(path.output.channels, 3);
        let ids: Vec<&str> = path.stages.iter().map(|s| s.id.as_str()).collect();
        assert!(
            ids.contains(&"matrix")
                && ids.contains(&"matrix-out-1")
                && ids.contains(&"matrix-out-3")
        );
        let m = path.stages.iter().find(|s| s.id == "matrix").unwrap();
        assert!(m.active);
        assert_eq!(m.detail, "2 → 3 ch · 2.1");
        assert_eq!(m.params["output_channels"].as_u64(), Some(3));
        let sub = path.stages.iter().find(|s| s.id == "matrix-out-3").unwrap();
        assert_eq!(sub.name, "Out 3 · Sub");
        assert!(sub.detail.contains("LP LR24 @ 80 Hz"), "{}", sub.detail);
        // The matrix comes before the upsampler in the chain.
        let mi = ids.iter().position(|s| *s == "matrix").unwrap();
        let ri = ids.iter().position(|s| *s == "resampler").unwrap();
        assert!(mi < ri);
        serde_json::to_string(&path).unwrap();
    }

    #[test]
    fn identity_matrix_keeps_the_chain_bit_perfect() {
        use crate::matrix::MatrixConfig;
        let cfg = PipelineConfig {
            matrix: MatrixConfig::preset("stereo"),
            output_bits: 24,
            dither: DitherKind::None,
            ..Default::default()
        };
        let mut p = Pipeline::new(input(44100, 16), cfg, &CoefficientCache::disabled()).unwrap();
        assert!(p.matrix().is_none(), "identity matrix must be skipped");
        assert!(p.bit_perfect());
        assert_eq!(p.channels(), 2);
        let (ints, f) = grid_samples(16, 300);
        let out = p.process(&f);
        assert!(out.iter().zip(&ints).all(|(o, i)| *o == i << 8));
        let path = p.signal_path();
        let m = path.stages.iter().find(|s| s.id == "matrix").unwrap();
        assert!(!m.active && m.detail.contains("identity"), "{}", m.detail);
        assert!(path.stages.iter().all(|s| !s.id.starts_with("matrix-out")));
        // Disabled: the stage says "off".
        let p = Pipeline::new(
            input(44100, 16),
            PipelineConfig::default(),
            &CoefficientCache::disabled(),
        )
        .unwrap();
        let m = p
            .signal_path()
            .stages
            .into_iter()
            .find(|s| s.id == "matrix")
            .unwrap();
        assert_eq!(m.detail, "off");
        // A bad preset is a pipeline error.
        let cfg = PipelineConfig {
            matrix: MatrixConfig::preset("2.1"),
            ..Default::default()
        };
        let six = InputFormat {
            rate: 48000,
            bits: 24,
            channels: 6,
            codec: "FLAC".into(),
        };
        let e = Pipeline::new(six, cfg, &CoefficientCache::disabled())
            .err()
            .expect("error");
        assert!(matches!(e, PipelineError::Matrix(_)), "{e}");
    }

    #[test]
    fn matrix_feeds_the_upsampler_with_the_new_channel_count() {
        use crate::matrix::MatrixConfig;
        let cfg = PipelineConfig {
            matrix: MatrixConfig::preset("2-way-active"),
            filter: FilterKind::sinc(Phase::Linear, false, 1 << 12),
            output_rate: Some(176400),
            output_bits: 32,
            dither: DitherKind::None,
            ..Default::default()
        };
        let mut p = Pipeline::new(input(44100, 24), cfg, &CoefficientCache::disabled()).unwrap();
        assert_eq!(p.channels(), 4);
        assert_eq!(p.output_rate(), 176400);
        let n = 4410;
        let pcm: Vec<f64> = (0..n)
            .flat_map(|i| {
                let v = 0.25 * (2.0 * PI * 1000.0 * i as f64 / 44100.0).sin();
                [v, -v]
            })
            .collect();
        let mut out = p.process(&pcm);
        out.extend(p.flush());
        assert_eq!(out.len(), n * 4 * 4);
        assert!(p.latency_seconds() > 0.0);
        assert_eq!(p.signal_path().output.channels, 4);
    }

    #[test]
    fn a_designed_filter_replaces_the_catalogue_one() {
        let d = crate::design::FilterDesign::parse("custom-gaussian-m-c95-t10-s140").unwrap();
        let cache = CoefficientCache::disabled();
        let p = Pipeline::new(
            InputFormat {
                rate: 44100,
                bits: 16,
                channels: 2,
                codec: "FLAC".into(),
            },
            PipelineConfig {
                output_rate: Some(176_400),
                filter_design: Some(d.clone()),
                ..Default::default()
            },
            &cache,
        )
        .unwrap();
        let path = p.signal_path();
        let r = path.stages.iter().find(|s| s.id == "resampler").unwrap();
        assert!(r.active);
        assert_eq!(r.params["filter"], d.id());
        assert_eq!(r.params["family"], "custom");
        assert_eq!(r.params["apodizing"], true);
        assert_eq!(r.params["phase"], "minimum");
        assert_eq!(r.params["window"], "gaussian");
        let cutoff = r.params["cutoff_hz"].as_f64().unwrap();
        assert!((cutoff - 0.95 * 22050.0).abs() < 1.0, "{cutoff}");
    }

    #[test]
    fn auto_dither_follows_the_dac_floor_and_shows_it() {
        let cache = CoefficientCache::disabled();
        let make = |bits: u32, floor: Option<f64>| {
            Pipeline::new(
                InputFormat {
                    rate: 44100,
                    bits: 24,
                    channels: 2,
                    codec: "FLAC".into(),
                },
                PipelineConfig {
                    output_rate: Some(44100),
                    output_bits: bits,
                    dither: DitherKind::Auto,
                    dac_noise_floor_db: floor,
                    ..Default::default()
                },
                &cache,
            )
            .unwrap()
        };
        let stage = |p: &Pipeline| {
            p.signal_path()
                .stages
                .into_iter()
                .find(|s| s.id == "dither")
                .unwrap()
        };
        let d = stage(&make(16, Some(-100.0)));
        assert_eq!(d.params["kind"], "ns9");
        assert_eq!(d.params["auto"], true);
        assert_eq!(d.params["dac_floor_db"], -100.0);
        assert!(
            d.detail.starts_with("Auto (-100 dBFS DAC floor): NS9"),
            "{}",
            d.detail
        );
        assert_eq!(stage(&make(16, None)).params["kind"], "lns15");
        assert_eq!(stage(&make(24, None)).params["kind"], "tpdf");
    }
}
