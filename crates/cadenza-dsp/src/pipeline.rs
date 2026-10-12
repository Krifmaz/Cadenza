//! The processing chain a player runs:
//!
//! ```text
//! source PCM (f64) → volume → EQ → convolution → resampler → quantiser (dither) → integers
//! ```
//!
//! Every stage is optional. With `bit_perfect` requested nothing is built at all and the
//! samples reach the output exactly as decoded; the [`Pipeline::bit_perfect`] report is
//! computed from what the stages *actually do*, so a 0 dB volume, an empty EQ and an
//! unchanged rate also count as untouched.

use crate::cache::CoefficientCache;
use crate::convolution::{Convolver, SpectralFilter};
use crate::dither::{DitherKind, Quantizer};
use crate::eq::{Band, Equalizer};
use crate::fir::FilterKind;
use crate::resampler::{Engine, ResampleError, Resampler};
use crate::signal_path::{rate_label, Endpoint, SignalPath, Stage};
use crate::volume::Volume;
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
    /// Output device name, for the signal path only.
    pub output_device: Option<String>,
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
            output_device: None,
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
}

struct ConvolutionStage {
    convolvers: Vec<Convolver>,
    split: Vec<Vec<f64>>,
}

impl ConvolutionStage {
    fn new(spec: &ConvolutionSpec, channels: usize) -> Result<Self, PipelineError> {
        if spec.channels.is_empty() || spec.channels.iter().any(|c| c.is_empty()) {
            return Err(PipelineError::Convolution(
                "impulse response is empty".into(),
            ));
        }
        let convolvers = if spec.channels.len() == 1 {
            let filter = Arc::new(SpectralFilter::new(
                &spec.channels[0],
                crate::convolution::choose_block(spec.channels[0].len()),
            ));
            (0..channels)
                .map(|_| Convolver::from_filter(Arc::clone(&filter)))
                .collect()
        } else if spec.channels.len() == channels {
            let block = crate::convolution::choose_block(
                spec.channels.iter().map(|c| c.len()).max().unwrap_or(1),
            );
            spec.channels
                .iter()
                .map(|h| Convolver::with_block(h, block))
                .collect()
        } else {
            return Err(PipelineError::Convolution(format!(
                "impulse has {} channels but the stream has {channels}",
                spec.channels.len()
            )));
        };
        Ok(Self {
            convolvers,
            split: vec![Vec::new(); channels],
        })
    }

    fn run(&mut self, interleaved: &[f64], flush: bool) -> Vec<f64> {
        let ch = self.convolvers.len();
        let frames = interleaved.len() / ch;
        for (c, buf) in self.split.iter_mut().enumerate() {
            buf.clear();
            buf.extend(interleaved.iter().skip(c).step_by(ch).take(frames));
        }
        let mut outs: Vec<Vec<f64>> = vec![Vec::new(); ch];
        for ((conv, inp), out) in self
            .convolvers
            .iter_mut()
            .zip(&self.split)
            .zip(outs.iter_mut())
        {
            conv.push(inp, out);
            if flush {
                conv.flush(out);
            }
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
        self.convolvers.iter_mut().for_each(Convolver::reset);
    }
}

/// The running chain for one input format.
pub struct Pipeline {
    input: InputFormat,
    config: PipelineConfig,
    volume: Volume,
    eq: Option<Equalizer>,
    convolution: Option<ConvolutionStage>,
    resampler: Option<Resampler>,
    quantizer: Quantizer,
    output_rate: u32,
    scratch: Vec<f64>,
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
        let convolution = match (&config.convolution, bp) {
            (Some(spec), false) => {
                if let Some(r) = spec.rate {
                    if r != input.rate {
                        log::warn!(
                            "impulse {} was measured at {r} Hz but the stream is {} Hz",
                            spec.name,
                            input.rate
                        );
                    }
                }
                Some(ConvolutionStage::new(spec, ch)?)
            }
            _ => None,
        };
        let output_rate = if bp {
            input.rate
        } else {
            config.output_rate.unwrap_or(input.rate)
        };
        let resampler = if output_rate != input.rate {
            Some(Resampler::new(
                input.rate,
                output_rate,
                ch,
                config.filter,
                config.attenuation_db,
                cache,
            )?)
        } else {
            None
        };
        let dither = if bp { DitherKind::None } else { config.dither };
        let quantizer = Quantizer::new(config.output_bits, dither, ch);
        Ok(Self {
            input,
            config,
            volume,
            eq,
            convolution,
            resampler,
            quantizer,
            output_rate,
            scratch: Vec::new(),
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
    /// Output rate in Hz.
    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }
    /// Output bit depth.
    pub fn output_bits(&self) -> u32 {
        self.quantizer.bits()
    }
    /// Channel count.
    pub fn channels(&self) -> usize {
        self.input.channels as usize
    }
    /// Current volume.
    pub fn volume(&self) -> Volume {
        self.volume
    }

    /// Change the volume without rebuilding (ignored in bit-perfect mode).
    pub fn set_volume(&mut self, volume: Volume) {
        if !self.config.bit_perfect {
            self.volume = volume;
            self.config.volume = volume;
        }
    }

    /// `true` only when nothing in the chain alters any sample.
    pub fn bit_perfect(&self) -> bool {
        self.volume.is_unity()
            && self.eq.is_none()
            && self.convolution.is_none()
            && self.resampler.is_none()
            && !self.quantizer.kind().adds_noise()
            && self.quantizer.bits() >= self.input.bits
    }

    /// Processing latency in seconds (what the player subtracts from its position).
    pub fn latency_seconds(&self) -> f64 {
        let conv = self
            .convolution
            .as_ref()
            .map(|c| c.convolvers[0].block() as f64 / self.input.rate as f64)
            .unwrap_or(0.0);
        conv + self
            .resampler
            .as_ref()
            .map(Resampler::latency_seconds)
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
        if let Some(r) = &mut self.resampler {
            r.reset();
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
        if let Some(r) = &mut self.resampler {
            let mut out = r.push(&self.scratch);
            if flush {
                out.extend(r.flush());
            }
            self.scratch = out;
        }
        let mut ints = Vec::with_capacity(self.scratch.len());
        self.quantizer.process(&self.scratch, &mut ints);
        ints
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

        match &self.resampler {
            Some(r) => {
                let (l, m) = r.ratio();
                let ratio = if m == 1 {
                    format!("×{l}")
                } else {
                    format!("×{l}/{m}")
                };
                let engine = match r.engine() {
                    Engine::FftPolyphase { block } => format!("FFT polyphase · block {block}"),
                    Engine::TimePolyphase => "time-domain polyphase".into(),
                };
                stages.push(Stage {
                    id: "resampler".into(),
                    name: "Upsampler".into(),
                    detail: format!(
                        "{} · {ratio} → {} kHz · {engine}",
                        self.config.filter.label(),
                        rate_label(r.out_rate())
                    ),
                    params: json!({
                        "filter": self.config.filter.id(),
                        "label": self.config.filter.label(),
                        "taps": r.spec().taps,
                        "phase": r.spec().phase.name(),
                        "apodizing": self.config.filter.apodizing,
                        "attenuation_db": r.spec().attenuation_db,
                        "transition_hz": r.spec().transition_width() * r.in_rate() as f64 * l as f64,
                        "cutoff_hz": r.spec().cutoff * r.in_rate() as f64 * l as f64,
                        "l": l, "m": m,
                        "engine": r.engine(),
                        "latency_seconds": r.latency_seconds(),
                    }),
                    active: true,
                });
            }
            None => stages.push(Stage {
                id: "resampler".into(),
                name: "Upsampler".into(),
                detail: bypass(&format!("{} kHz unchanged", rate_label(self.input.rate))),
                params: json!({ "filter": self.config.filter.id(), "rate": self.input.rate }),
                active: false,
            }),
        }

        let q = &self.quantizer;
        let depth_change = q.bits() < self.input.bits;
        stages.push(Stage {
            id: "dither".into(),
            name: "Dither".into(),
            detail: format!(
                "{} → {}-bit{}",
                q.kind().label(),
                q.bits(),
                if q.kind().adds_noise() || depth_change { "" } else { " · exact" }
            ),
            params: json!({ "kind": q.kind().id(), "label": q.kind().label(), "bits": q.bits(), "shaper": q.kind().coefficients() }),
            active: q.kind().adds_noise() || depth_change,
        });

        let input = Endpoint {
            format: self.input.codec.clone(),
            rate: self.input.rate,
            bits: self.input.bits,
            channels: self.input.channels,
            device: None,
        };
        let output = Endpoint {
            format: "PCM".into(),
            rate: self.output_rate,
            bits: q.bits(),
            channels: self.input.channels,
            device: self.config.output_device.clone(),
        };
        let label = SignalPath::make_label(&input, &output, bp);
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
            filter: FilterKind {
                phase: Phase::Linear,
                apodizing: false,
                taps: 1 << 12,
            },
            output_rate: Some(176400),
            output_bits: 24,
            dither: DitherKind::Lipshitz,
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
            .filter(|s| s.active)
            .map(|s| s.id.as_str())
            .collect();
        assert_eq!(active, ["volume", "eq", "resampler", "dither"]);
        assert!(path.latency_seconds > 0.0);
        serde_json::to_string(&path).unwrap();
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
}
