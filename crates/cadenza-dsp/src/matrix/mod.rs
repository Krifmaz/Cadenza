//! Pipeline matrix: N input channels → N output channels (up to 32) through a routing
//! matrix with a gain in dB per cell, then a chain per output channel (crossover
//! low-/high-pass, parametric EQ, delay, polarity, gain), plus stereo utilities (mono sum,
//! balance, L/R swap, mid/side) that fold into the routing matrix at build time.
//!
//! ```text
//!   inputs ─▶ [stereo utility] ─▶ routing (dB cells) ─▶ per output: crossover → EQ → delay → ±gain ─▶ outputs
//! ```
//!
//! It sits before the upsampler in the [`crate::pipeline::Pipeline`] (so IIR crossovers run
//! at the source rate and the resampler sees the final channel count). Everything is `f64`;
//! a configuration that cannot change a sample ([`MatrixStage::is_identity`]) is skipped by
//! the pipeline so the chain stays bit-perfect.
//!
//! Configuration lives in the `[matrix]` table of config.toml (and as the same JSON in the
//! API):
//!
//! ```toml
//! [matrix]
//! enabled = true
//! preset = "custom"           # stereo | mono | swap | mid-side | 2.1 | 2-way-active | 3-way-active | custom
//! [matrix.stereo]             # stereo utilities (2-channel sources)
//! mono = false
//! balance_db = 0.0            # > 0 attenuates the left channel, < 0 the right
//! swap = false
//! mid_side = "off"            # off | encode | decode
//! [[matrix.outputs]]
//! name = "Sub"
//! inputs = [-6.0, -6.0]       # dB from each input channel; ≤ -120 = not connected
//! gain_db = 0.0
//! delay_ms = 0.0
//! delay_samples = 0
//! invert = false
//! lowpass = { kind = "linkwitz-riley", freq = 80.0, slope_db = 24 }
//! eq = [{ kind = "peaking", freq = 40.0, gain_db = 3.0, q = 1.0 }]
//! ```

pub mod crossover;

pub use crossover::{freq_label, CrossoverFilter, CrossoverKind};

use crate::convolution::{choose_block, Convolver, SpectralFilter};
use crate::eq::{Band, Equalizer};
use crate::signal_path::Stage;
use crossover::{fir_band, IirCrossover, Pass};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

/// Largest channel count in or out.
pub const MAX_CHANNELS: usize = 32;
/// A routing cell at or below this gain is "not connected" (JSON has no `-inf`).
pub const OFF_DB: f64 = -120.0;
/// Longest per-channel delay accepted, in seconds.
pub const MAX_DELAY_SECONDS: f64 = 10.0;

/// Errors building a [`MatrixStage`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MatrixError {
    /// Unknown preset name.
    #[error("unknown matrix preset {0:?} (one of stereo, mono, swap, mid-side, 2.1, 2-way-active, 3-way-active, custom)")]
    UnknownPreset(String),
    /// The preset needs a different source channel count.
    #[error("matrix preset {preset:?} needs a {needs}-channel source, the stream has {has}")]
    PresetChannels {
        /// Preset id.
        preset: String,
        /// Channels the preset expects.
        needs: usize,
        /// Channels the stream has.
        has: usize,
    },
    /// 0 or more than [`MAX_CHANNELS`] channels.
    #[error("{0}")]
    Channels(String),
    /// A per-output setting is out of range.
    #[error("matrix output {index} ({name:?}): {reason}")]
    Output {
        /// 1-based output index.
        index: usize,
        /// Output name.
        name: String,
        /// What is wrong.
        reason: String,
    },
}

/// One output channel: which inputs feed it, and its chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputChannel {
    /// Display name (`L`, `Sub`, `L tweeter`…).
    #[serde(default)]
    pub name: String,
    /// Routing row: gain in dB from each input channel, index = input channel. Missing
    /// cells and cells ≤ [`OFF_DB`] are not connected.
    #[serde(default)]
    pub inputs: Vec<f64>,
    /// Output gain in dB (after the crossover and EQ).
    #[serde(default)]
    pub gain_db: f64,
    /// Delay in milliseconds (added to `delay_samples`).
    #[serde(default)]
    pub delay_ms: f64,
    /// Delay in whole samples at the stream rate (added to `delay_ms`).
    #[serde(default)]
    pub delay_samples: u32,
    /// Polarity inversion.
    #[serde(default)]
    pub invert: bool,
    /// Keep what is below this crossover (woofer / sub).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lowpass: Option<CrossoverFilter>,
    /// Keep what is above this crossover (tweeter); with `lowpass` too: a band-pass (mid).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highpass: Option<CrossoverFilter>,
    /// Parametric EQ for this output only (up to [`crate::eq::MAX_BANDS`]).
    #[serde(default)]
    pub eq: Vec<Band>,
}

impl OutputChannel {
    /// An output fed at 0 dB from input `input` alone (of `inputs` channels).
    pub fn from_input(name: &str, input: usize, inputs: usize) -> Self {
        let mut row = vec![OFF_DB; inputs];
        if input < inputs {
            row[input] = 0.0;
        }
        Self {
            name: name.into(),
            inputs: row,
            gain_db: 0.0,
            delay_ms: 0.0,
            delay_samples: 0,
            invert: false,
            lowpass: None,
            highpass: None,
            eq: Vec::new(),
        }
    }

    /// An output summing `cells` (dB per input; `OFF_DB` for none).
    pub fn mixed(name: &str, cells: &[f64]) -> Self {
        let mut o = Self::from_input(name, usize::MAX, cells.len());
        o.inputs = cells.to_vec();
        o
    }

    /// Builder: low-pass.
    pub fn lowpass(mut self, f: CrossoverFilter) -> Self {
        self.lowpass = Some(f);
        self
    }

    /// Builder: high-pass.
    pub fn highpass(mut self, f: CrossoverFilter) -> Self {
        self.highpass = Some(f);
        self
    }

    /// Linear gain from input `i` (0 when not connected).
    pub fn cell_gain(&self, i: usize) -> f64 {
        match self.inputs.get(i) {
            Some(&db) if db > OFF_DB => 10f64.powf(db / 20.0),
            _ => 0.0,
        }
    }

    /// Total delay in samples at `rate`.
    pub fn delay_total_samples(&self, rate: u32) -> usize {
        let ms = (self.delay_ms.max(0.0) * rate as f64 / 1000.0).round() as usize;
        ms + self.delay_samples as usize
    }
}

/// Mid/side utility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum MidSide {
    /// Untouched.
    #[default]
    Off,
    /// `L, R → M = (L+R)/2, S = (L−R)/2`.
    Encode,
    /// `M, S → L = M+S, R = M−S`.
    Decode,
}

/// Stereo utilities, applied to a 2-channel source before the routing matrix (ignored, and
/// reported as such, for other channel counts).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct StereoUtility {
    /// Sum both channels to mono (−6 dB each) on both outputs.
    #[serde(default)]
    pub mono: bool,
    /// Balance in dB: positive attenuates the left channel by that much, negative the right.
    #[serde(default)]
    pub balance_db: f64,
    /// Swap left and right.
    #[serde(default)]
    pub swap: bool,
    /// Mid/side encode or decode.
    #[serde(default)]
    pub mid_side: MidSide,
}

impl StereoUtility {
    /// `true` when nothing is switched on.
    pub fn is_identity(&self) -> bool {
        !self.mono && self.balance_db == 0.0 && !self.swap && self.mid_side == MidSide::Off
    }

    /// The 2×2 linear matrix `out = M · in` (row = output): swap, then balance, then
    /// mono, then mid/side.
    pub fn matrix(&self) -> [[f64; 2]; 2] {
        let mut m = [[1.0, 0.0], [0.0, 1.0]];
        let mul = |a: [[f64; 2]; 2], b: [[f64; 2]; 2]| {
            let mut r = [[0.0; 2]; 2];
            for (i, row) in r.iter_mut().enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = a[i][0] * b[0][j] + a[i][1] * b[1][j];
                }
            }
            r
        };
        if self.swap {
            m = mul([[0.0, 1.0], [1.0, 0.0]], m);
        }
        if self.balance_db != 0.0 {
            let l = 10f64.powf(-self.balance_db.max(0.0) / 20.0);
            let r = 10f64.powf(self.balance_db.min(0.0) / 20.0);
            m = mul([[l, 0.0], [0.0, r]], m);
        }
        if self.mono {
            m = mul([[0.5, 0.5], [0.5, 0.5]], m);
        }
        match self.mid_side {
            MidSide::Off => {}
            MidSide::Encode => m = mul([[0.5, 0.5], [0.5, -0.5]], m),
            MidSide::Decode => m = mul([[1.0, 1.0], [1.0, -1.0]], m),
        }
        m
    }

    /// Short description of what is on (`mono · balance +2.0 dB`), empty when nothing is.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.swap {
            parts.push("L/R swapped".to_string());
        }
        if self.balance_db != 0.0 {
            parts.push(format!("balance {:+.1} dB", self.balance_db));
        }
        if self.mono {
            parts.push("mono".to_string());
        }
        match self.mid_side {
            MidSide::Off => {}
            MidSide::Encode => parts.push("mid/side encode".to_string()),
            MidSide::Decode => parts.push("mid/side decode".to_string()),
        }
        parts.join(" · ")
    }
}

/// The `[matrix]` table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatrixConfig {
    /// Master switch; off = the matrix stage is not built at all.
    #[serde(default)]
    pub enabled: bool,
    /// Preset id (see [`presets`]); `custom` uses `outputs`.
    #[serde(default = "default_preset")]
    pub preset: String,
    /// Output channels for the `custom` preset (empty = identity routing).
    #[serde(default)]
    pub outputs: Vec<OutputChannel>,
    /// Stereo utilities (merged with whatever the preset sets).
    #[serde(default)]
    pub stereo: StereoUtility,
}

/// Serde default for the matrix preset name.
fn default_preset() -> String {
    "stereo".into()
}

impl Default for MatrixConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            preset: default_preset(),
            outputs: Vec::new(),
            stereo: StereoUtility::default(),
        }
    }
}

/// A preset the UI can offer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetInfo {
    /// Id for `matrix_preset` / `[matrix] preset`.
    pub id: &'static str,
    /// Display name.
    pub label: &'static str,
    /// What it does.
    pub description: &'static str,
    /// Source channels it needs (`0` = any).
    pub input_channels: usize,
    /// Output channels it produces (`0` = same as the input).
    pub output_channels: usize,
}

/// Every preset.
pub fn presets() -> &'static [PresetInfo] {
    &[
        PresetInfo {
            id: "stereo",
            label: "Stereo / identity",
            description: "every input goes to the output with the same number, untouched",
            input_channels: 0,
            output_channels: 0,
        },
        PresetInfo {
            id: "mono",
            label: "Mono sum",
            description: "L and R summed at −6 dB each on both outputs",
            input_channels: 2,
            output_channels: 2,
        },
        PresetInfo {
            id: "swap",
            label: "L/R swap",
            description: "left and right exchanged",
            input_channels: 2,
            output_channels: 2,
        },
        PresetInfo {
            id: "mid-side",
            label: "Mid/side",
            description: "outputs M = (L+R)/2 and S = (L−R)/2",
            input_channels: 2,
            output_channels: 2,
        },
        PresetInfo {
            id: "2.1",
            label: "2.1 (stereo + sub)",
            description: "L and R high-passed LR24 at 80 Hz, a third channel with the L+R sum low-passed LR24 at 80 Hz",
            input_channels: 2,
            output_channels: 3,
        },
        PresetInfo {
            id: "2-way-active",
            label: "2-way active",
            description: "L woofer, L tweeter, R woofer, R tweeter: LR24 at 2 kHz",
            input_channels: 2,
            output_channels: 4,
        },
        PresetInfo {
            id: "3-way-active",
            label: "3-way active",
            description: "L woofer/mid/tweeter, R woofer/mid/tweeter: LR24 at 300 Hz and 3 kHz",
            input_channels: 2,
            output_channels: 6,
        },
        PresetInfo {
            id: "custom",
            label: "Custom",
            description: "the [matrix] outputs table as written",
            input_channels: 0,
            output_channels: 0,
        },
    ]
}

/// Preset ids in order.
pub fn preset_ids() -> Vec<&'static str> {
    presets().iter().map(|p| p.id).collect()
}

impl MatrixConfig {
    /// A disabled matrix (identity).
    pub fn off() -> Self {
        Self::default()
    }

    /// An enabled matrix from a preset.
    pub fn preset(id: &str) -> Self {
        Self {
            enabled: true,
            preset: id.into(),
            ..Self::default()
        }
    }

    /// An enabled custom matrix with these outputs.
    pub fn custom(outputs: Vec<OutputChannel>) -> Self {
        Self {
            enabled: true,
            preset: "custom".into(),
            outputs,
            stereo: StereoUtility::default(),
        }
    }

    /// Expand the preset for a source with `input_channels` channels: the returned config
    /// is always `custom` with explicit `outputs`, and `stereo` merges the preset's own
    /// utilities with the user's.
    pub fn resolve(&self, input_channels: usize) -> Result<MatrixConfig, MatrixError> {
        if input_channels == 0 || input_channels > MAX_CHANNELS {
            return Err(MatrixError::Channels(format!(
                "the stream has {input_channels} channels, the matrix handles 1…{MAX_CHANNELS}"
            )));
        }
        let n = input_channels;
        let id = self.preset.trim().to_ascii_lowercase();
        let need_stereo = |needs: usize| {
            if n == needs {
                Ok(())
            } else {
                Err(MatrixError::PresetChannels {
                    preset: id.clone(),
                    needs,
                    has: n,
                })
            }
        };
        let identity = || {
            (0..n)
                .map(|i| OutputChannel::from_input(&channel_name(i, n), i, n))
                .collect()
        };
        let mut stereo = self.stereo.clone();
        let outputs: Vec<OutputChannel> = match id.as_str() {
            "" | "stereo" | "identity" => identity(),
            "custom" => {
                if self.outputs.is_empty() {
                    identity()
                } else {
                    self.outputs.clone()
                }
            }
            "mono" => {
                need_stereo(2)?;
                stereo.mono = true;
                identity()
            }
            "swap" => {
                need_stereo(2)?;
                stereo.swap = true;
                identity()
            }
            "mid-side" | "mid_side" | "ms" => {
                need_stereo(2)?;
                stereo.mid_side = MidSide::Encode;
                vec![
                    OutputChannel::from_input("Mid", 0, 2),
                    OutputChannel::from_input("Side", 1, 2),
                ]
            }
            "2.1" => {
                need_stereo(2)?;
                let x = CrossoverFilter::lr24(80.0);
                vec![
                    OutputChannel::from_input("L", 0, 2).highpass(x),
                    OutputChannel::from_input("R", 1, 2).highpass(x),
                    OutputChannel::mixed("Sub", &[-6.0, -6.0]).lowpass(x),
                ]
            }
            "2-way-active" | "2-way" => {
                need_stereo(2)?;
                let x = CrossoverFilter::lr24(2000.0);
                vec![
                    OutputChannel::from_input("L woofer", 0, 2).lowpass(x),
                    OutputChannel::from_input("L tweeter", 0, 2).highpass(x),
                    OutputChannel::from_input("R woofer", 1, 2).lowpass(x),
                    OutputChannel::from_input("R tweeter", 1, 2).highpass(x),
                ]
            }
            "3-way-active" | "3-way" => {
                need_stereo(2)?;
                let lo = CrossoverFilter::lr24(300.0);
                let hi = CrossoverFilter::lr24(3000.0);
                vec![
                    OutputChannel::from_input("L woofer", 0, 2).lowpass(lo),
                    OutputChannel::from_input("L mid", 0, 2)
                        .highpass(lo)
                        .lowpass(hi),
                    OutputChannel::from_input("L tweeter", 0, 2).highpass(hi),
                    OutputChannel::from_input("R woofer", 1, 2).lowpass(lo),
                    OutputChannel::from_input("R mid", 1, 2)
                        .highpass(lo)
                        .lowpass(hi),
                    OutputChannel::from_input("R tweeter", 1, 2).highpass(hi),
                ]
            }
            other => return Err(MatrixError::UnknownPreset(other.to_string())),
        };
        if outputs.is_empty() || outputs.len() > MAX_CHANNELS {
            return Err(MatrixError::Channels(format!(
                "the matrix has {} outputs, it handles 1…{MAX_CHANNELS}",
                outputs.len()
            )));
        }
        Ok(MatrixConfig {
            enabled: self.enabled,
            preset: "custom".into(),
            outputs,
            stereo,
        })
    }
}

/// Default name of input channel `i` of `n` (`L`/`R` for stereo, `Ch 3` otherwise).
pub fn channel_name(i: usize, n: usize) -> String {
    match (n, i) {
        (1, 0) => "Mono".into(),
        (2, 0) => "L".into(),
        (2, 1) => "R".into(),
        _ => format!("Ch {}", i + 1),
    }
}

/// Fixed-length delay line.
#[derive(Clone, Debug, Default)]
struct DelayLine {
    /// Ring of delayed samples (empty means no delay).
    buf: Vec<f64>,
    /// Next slot to read and overwrite.
    pos: usize,
}

impl DelayLine {
    /// A delay of `samples` samples.
    fn new(samples: usize) -> Self {
        Self {
            buf: vec![0.0; samples],
            pos: 0,
        }
    }

    /// Delay one sample (a pass-through when the delay is zero).
    #[inline]
    fn tick(&mut self, x: f64) -> f64 {
        if self.buf.is_empty() {
            return x;
        }
        let y = self.buf[self.pos];
        self.buf[self.pos] = x;
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
        y
    }

    /// Clear the delayed samples.
    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
    }
}

/// The chain of one output channel.
struct OutputChain {
    /// `(input index, linear gain)` for connected cells, after the stereo utility.
    mix: Vec<(usize, f64)>,
    /// IIR crossover sections, in order (including phase-alignment all-passes).
    iir: Vec<IirCrossover>,
    /// Crossover points of sibling outputs this band is phase-aligned to (`Pass::All` or
    /// `Pass::High` filters appended to `iir`), for the report.
    aligned: Vec<CrossoverFilter>,
    /// Linear-phase FIR crossover of this output, if any.
    fir: Option<Convolver>,
    /// Per-output parametric EQ, if any.
    eq: Option<Equalizer>,
    /// Time alignment of this output.
    delay: DelayLine,
    /// User delay in samples.
    user_delay: usize,
    /// Extra samples added so this channel lines up with the FIR channels.
    align_delay: usize,
    /// Linear gain with the polarity sign folded in.
    gain: f64,
    /// Scratch: this output's mixed block.
    buf: Vec<f64>,
    /// Scratch for the convolver.
    fir_out: Vec<f64>,
}

impl OutputChain {
    /// Mix the connected inputs of `frames` interleaved frames into `buf`, then run the
    /// crossovers, FIR, EQ and delay over it.
    fn run(&mut self, frames: usize, input: &[f64], n_in: usize) {
        self.buf.clear();
        self.buf.resize(frames, 0.0);
        for (f, v) in self.buf.iter_mut().enumerate() {
            let frame = &input[f * n_in..(f + 1) * n_in];
            let mut s = 0.0;
            for &(i, g) in &self.mix {
                s += g * frame[i];
            }
            *v = s;
        }
        if let Some(conv) = &mut self.fir {
            self.fir_out.clear();
            conv.push(&self.buf, &mut self.fir_out);
            debug_assert_eq!(self.fir_out.len(), frames);
            std::mem::swap(&mut self.buf, &mut self.fir_out);
        }
        for x in &mut self.iir {
            x.process(&mut self.buf);
        }
        if let Some(eq) = &mut self.eq {
            eq.process(&mut self.buf);
        }
        let g = self.gain;
        for v in &mut self.buf {
            *v = self.delay.tick(*v) * g;
        }
    }

    /// Clear every filter and the delay of this output.
    fn reset(&mut self) {
        self.iir.iter_mut().for_each(IirCrossover::reset);
        if let Some(c) = &mut self.fir {
            c.reset();
        }
        if let Some(e) = &mut self.eq {
            e.reset();
        }
        self.delay.reset();
    }
}

/// The running matrix for one stream format.
pub struct MatrixStage {
    /// Stream sample rate in Hz.
    rate: u32,
    /// Input channel count.
    n_in: usize,
    /// Output channel count.
    n_out: usize,
    /// The configuration this stage was built from.
    config: MatrixConfig,
    /// One processing chain per output channel.
    chains: Vec<OutputChain>,
    /// Set when any output has a linear-phase FIR: the stage then works in blocks.
    block: Option<usize>,
    /// Longest FIR crossover in taps (0 when there is none).
    fir_taps: usize,
    /// Whether the stereo utility (swap, balance, mono, M/S) is part of the mix.
    stereo_applied: bool,
    /// Input not yet forming a whole block (block mode only).
    pending: Vec<f64>,
    /// Output buffer, kept between calls to save an allocation.
    out: Vec<f64>,
    /// `true` when the configuration cannot change a sample.
    identity: bool,
}

impl MatrixStage {
    /// Build the stage for a stream of `input_channels` channels at `rate` Hz. The preset is
    /// expanded first ([`MatrixConfig::resolve`]).
    pub fn new(
        config: &MatrixConfig,
        rate: u32,
        input_channels: usize,
    ) -> Result<Self, MatrixError> {
        if rate == 0 {
            return Err(MatrixError::Channels("sample rate is zero".into()));
        }
        let resolved = config.resolve(input_channels)?;
        let n_in = input_channels;
        let n_out = resolved.outputs.len();
        let fs = rate as f64;

        // Validate every output first.
        for (idx, o) in resolved.outputs.iter().enumerate() {
            let err = |reason: String| MatrixError::Output {
                index: idx + 1,
                name: o.name.clone(),
                reason,
            };
            if o.inputs.len() > n_in {
                return Err(err(format!(
                    "{} routing cells but the stream has {n_in} channels",
                    o.inputs.len()
                )));
            }
            for x in o.lowpass.iter().chain(o.highpass.iter()) {
                x.validate(fs).map_err(err)?;
            }
            if let (Some(lp), Some(hp)) = (&o.lowpass, &o.highpass) {
                if hp.freq >= lp.freq {
                    return Err(err(format!(
                        "high-pass {} is not below low-pass {}",
                        freq_label(hp.freq),
                        freq_label(lp.freq)
                    )));
                }
            }
            if !(o.delay_ms.is_finite() && o.delay_ms >= 0.0) {
                return Err(err(format!(
                    "delay {} ms is negative or not a number",
                    o.delay_ms
                )));
            }
            if o.delay_total_samples(rate) as f64 > MAX_DELAY_SECONDS * fs {
                return Err(err(format!("delay exceeds {MAX_DELAY_SECONDS} s")));
            }
            if !o.gain_db.is_finite() || o.inputs.iter().any(|v| !v.is_finite()) {
                return Err(err("gain is not a number".into()));
            }
        }

        // Stereo utility folded into the routing matrix.
        let stereo_applied = n_in == 2 && !resolved.stereo.is_identity();
        let s = if stereo_applied {
            resolved.stereo.matrix()
        } else {
            [[1.0, 0.0], [0.0, 1.0]]
        };

        // Linear-phase FIRs share one tap count (so they are complementary) and one block.
        let fir_outputs: Vec<usize> = resolved
            .outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| {
                o.lowpass
                    .is_some_and(|x| x.kind == CrossoverKind::LinearPhase)
                    || o.highpass
                        .is_some_and(|x| x.kind == CrossoverKind::LinearPhase)
            })
            .map(|(i, _)| i)
            .collect();
        let fir_taps = resolved
            .outputs
            .iter()
            .flat_map(|o| o.lowpass.iter().chain(o.highpass.iter()))
            .filter(|x| x.kind == CrossoverKind::LinearPhase)
            .map(|x| x.fir_taps(fs))
            .max()
            .unwrap_or(0);
        let block = (fir_taps > 0).then(|| choose_block(fir_taps));
        let group_delay = if fir_taps > 0 { (fir_taps - 1) / 2 } else { 0 };

        // Outputs fed from a common input form one speaker system: a band that does not take
        // part in a crossover point of that system still runs through that point's all-pass
        // (above its own low-pass) or high-pass (below its own high-pass), so the acoustic
        // sum of an N-way set is `A₁·A₂·…` – flat for Linkwitz–Riley. (FIR bands are
        // complementary by construction and need none of this.)
        let shares_input = |a: &OutputChannel, b: &OutputChannel| {
            stereo_applied || (0..n_in).any(|i| a.cell_gain(i) != 0.0 && b.cell_gain(i) != 0.0)
        };
        let alignment_for = |idx: usize, o: &OutputChannel| -> Vec<(CrossoverFilter, Pass)> {
            let mut out: Vec<(CrossoverFilter, Pass)> = Vec::new();
            if o.lowpass.is_none() && o.highpass.is_none() {
                return out;
            }
            let own = |f: f64| {
                o.lowpass.is_some_and(|x| x.freq == f) || o.highpass.is_some_and(|x| x.freq == f)
            };
            for (j, other) in resolved.outputs.iter().enumerate() {
                if j == idx || !shares_input(o, other) {
                    continue;
                }
                for x in other.lowpass.iter().chain(other.highpass.iter()) {
                    if x.kind == CrossoverKind::LinearPhase
                        || own(x.freq)
                        || out.iter().any(|(y, _)| y.freq == x.freq)
                    {
                        continue;
                    }
                    if o.lowpass.is_some_and(|lp| x.freq > lp.freq) {
                        out.push((*x, Pass::All));
                    } else if o.highpass.is_some_and(|hp| x.freq < hp.freq) {
                        out.push((*x, Pass::High));
                    }
                }
            }
            out.sort_by(|a, b| a.0.freq.total_cmp(&b.0.freq));
            out
        };

        let mut chains = Vec::with_capacity(n_out);
        for (idx, o) in resolved.outputs.iter().enumerate() {
            // Routing row R, combined with the stereo utility: out = Σ_i R[i] · (S · in)[i]
            // = Σ_j (Σ_i R[i] S[i][j]) in[j].
            let mut row: Vec<f64> = (0..n_in).map(|i| o.cell_gain(i)).collect();
            if stereo_applied {
                let r = row.clone();
                for j in 0..2 {
                    row[j] = r[0] * s[0][j] + r[1] * s[1][j];
                }
            }
            let mix: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .filter(|(_, g)| **g != 0.0)
                .map(|(i, g)| (i, *g))
                .collect();

            let is_fir = fir_outputs.contains(&idx);
            let mut iir = Vec::new();
            let mut aligned = Vec::new();
            let mut fir = None;
            if is_fir {
                // Every crossover of an FIR output becomes part of the one FIR band (an IIR
                // side mixed with an FIR side would not be complementary anyway).
                let h = fir_band(
                    fir_taps,
                    o.highpass.map(|x| x.freq),
                    o.lowpass.map(|x| x.freq),
                    fs,
                );
                let filter = Arc::new(SpectralFilter::new(&h, block.unwrap()));
                fir = Some(Convolver::from_filter(filter));
            } else {
                if let Some(x) = &o.lowpass {
                    iir.push(IirCrossover::new(x, Pass::Low, fs));
                }
                if let Some(x) = &o.highpass {
                    iir.push(IirCrossover::new(x, Pass::High, fs));
                }
                for (x, pass) in alignment_for(idx, o) {
                    iir.push(IirCrossover::new(&x, pass, fs));
                    aligned.push(x);
                }
            }
            let eq = (!o.eq.is_empty())
                .then(|| Equalizer::new(rate, 1, &o.eq))
                .filter(|e| !e.is_identity());
            let user_delay = o.delay_total_samples(rate);
            let align_delay = if is_fir { 0 } else { group_delay };
            let gain = 10f64.powf(o.gain_db / 20.0) * if o.invert { -1.0 } else { 1.0 };
            chains.push(OutputChain {
                mix,
                iir,
                aligned,
                fir,
                eq,
                delay: DelayLine::new(user_delay + align_delay),
                user_delay,
                align_delay,
                gain,
                buf: Vec::new(),
                fir_out: Vec::new(),
            });
        }

        let identity = n_in == n_out
            && chains.iter().enumerate().all(|(i, c)| {
                c.mix.len() == 1
                    && c.mix[0] == (i, 1.0)
                    && c.iir.is_empty()
                    && c.fir.is_none()
                    && c.eq.is_none()
                    && c.delay.buf.is_empty()
                    && c.gain == 1.0
            });

        Ok(Self {
            rate,
            n_in,
            n_out,
            config: resolved,
            chains,
            block,
            fir_taps,
            stereo_applied,
            pending: Vec::new(),
            out: Vec::new(),
            identity,
        })
    }

    /// Input channel count.
    pub fn input_channels(&self) -> usize {
        self.n_in
    }

    /// Output channel count.
    pub fn output_channels(&self) -> usize {
        self.n_out
    }

    /// The expanded configuration (`custom` with explicit outputs).
    pub fn config(&self) -> &MatrixConfig {
        &self.config
    }

    /// `true` when the stage cannot change a single sample (identity routing, no chains).
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// Block size the stage buffers in when it has FIR crossovers (`None` = sample by sample).
    pub fn block(&self) -> Option<usize> {
        self.block
    }

    /// Taps of the linear-phase FIR crossovers (0 when none).
    pub fn fir_taps(&self) -> usize {
        self.fir_taps
    }

    /// Latency the player must hide: the block buffering plus the FIR group delay, in
    /// input frames (user delays are intentional and not counted).
    pub fn latency_frames(&self) -> usize {
        self.block.unwrap_or(0)
            + if self.fir_taps > 0 {
                (self.fir_taps - 1) / 2
            } else {
                0
            }
    }

    /// [`MatrixStage::latency_frames`] in seconds.
    pub fn latency_seconds(&self) -> f64 {
        self.latency_frames() as f64 / self.rate as f64
    }

    /// Run interleaved input frames (`n_in` channels) and return interleaved output frames
    /// (`n_out` channels). With FIR crossovers the output arrives in whole blocks; `flush`
    /// pushes out what is pending (the stream length stays equal to the input length).
    pub fn process(&mut self, input: &[f64], flush: bool) -> Vec<f64> {
        assert_eq!(input.len() % self.n_in, 0, "input is not whole frames");
        let mut out = std::mem::take(&mut self.out);
        out.clear();
        match self.block {
            None => {
                let frames = input.len() / self.n_in;
                self.run_frames(frames, input, &mut out);
            }
            Some(block) => {
                self.pending.extend_from_slice(input);
                let mut start = 0;
                while self.pending.len() - start >= block * self.n_in {
                    let chunk = &self.pending[start..start + block * self.n_in];
                    // `chunk` borrows self.pending; copy it out to a frame buffer to run.
                    let frame_buf = chunk.to_vec();
                    self.run_frames(block, &frame_buf, &mut out);
                    start += block * self.n_in;
                }
                self.pending.drain(..start);
                if flush && !self.pending.is_empty() {
                    let frames = self.pending.len() / self.n_in;
                    let mut padded = std::mem::take(&mut self.pending);
                    padded.resize(block * self.n_in, 0.0);
                    self.run_frames(block, &padded, &mut out);
                    out.truncate(out.len() - (block - frames) * self.n_out);
                    self.pending = padded;
                    self.pending.clear();
                    for c in &mut self.chains {
                        if let Some(f) = &mut c.fir {
                            f.reset();
                        }
                    }
                }
            }
        }
        self.out = Vec::with_capacity(out.capacity());
        out
    }

    /// Run `frames` input frames through every output chain (in parallel when they are FIR
    /// blocks) and append the interleaved result to `out`.
    fn run_frames(&mut self, frames: usize, input: &[f64], out: &mut Vec<f64>) {
        if frames == 0 {
            return;
        }
        let n_in = self.n_in;
        if self.block.is_some() && self.chains.len() > 1 {
            self.chains
                .par_iter_mut()
                .for_each(|c| c.run(frames, input, n_in));
        } else {
            for c in &mut self.chains {
                c.run(frames, input, n_in);
            }
        }
        let n_out = self.n_out;
        let base = out.len();
        out.resize(base + frames * n_out, 0.0);
        for (o, c) in self.chains.iter().enumerate() {
            for (f, v) in c.buf.iter().enumerate() {
                out[base + f * n_out + o] = *v;
            }
        }
    }

    /// Forget all state (seek).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.chains.iter_mut().for_each(OutputChain::reset);
    }

    /// `2 → 6 ch · 3-way-active`-style summary.
    pub fn summary(&self, preset: &str) -> String {
        let mut s = format!("{} → {} ch", self.n_in, self.n_out);
        let p = preset.trim();
        if !p.is_empty() && p != "custom" {
            s.push_str(&format!(" · {p}"));
        }
        if self.stereo_applied {
            s.push_str(&format!(" · {}", self.config.stereo.describe()));
        }
        if self.fir_taps > 0 {
            s.push_str(&format!(" · FIR {} taps", self.fir_taps));
        }
        s
    }

    /// Signal-path stages: one `matrix` summary followed by one `matrix-out-<n>` stage per
    /// output channel. `preset` is the id the user chose (for the label); `active` says
    /// whether the pipeline runs the stage.
    pub fn stages(&self, preset: &str, active: bool) -> Vec<Stage> {
        let mut stages = Vec::with_capacity(self.n_out + 1);
        let outputs: Vec<serde_json::Value> = self
            .config
            .outputs
            .iter()
            .enumerate()
            .map(|(i, o)| self.output_params(i, o))
            .collect();
        stages.push(Stage {
            id: "matrix".into(),
            name: "Matrix".into(),
            detail: if active {
                self.summary(preset)
            } else {
                format!("{} · bypassed", self.summary(preset))
            },
            params: json!({
                "preset": preset,
                "input_channels": self.n_in,
                "output_channels": self.n_out,
                "stereo": self.config.stereo,
                "stereo_applied": self.stereo_applied,
                "fir_taps": self.fir_taps,
                "block": self.block,
                "latency_frames": self.latency_frames(),
                "latency_seconds": self.latency_seconds(),
                "identity": self.identity,
                "outputs": outputs,
            }),
            active,
        });
        for (i, o) in self.config.outputs.iter().enumerate() {
            stages.push(Stage {
                id: format!("matrix-out-{}", i + 1),
                name: format!(
                    "Out {} · {}",
                    i + 1,
                    if o.name.is_empty() {
                        channel_name(i, self.n_out)
                    } else {
                        o.name.clone()
                    }
                ),
                detail: self.output_detail(i, o),
                params: self.output_params(i, o),
                active,
            });
        }
        stages
    }

    /// One output's processing summarised for the signal path (sources, filters, EQ, delay).
    fn output_detail(&self, i: usize, o: &OutputChannel) -> String {
        let c = &self.chains[i];
        let mut parts: Vec<String> = Vec::new();
        let sources: Vec<String> = o
            .inputs
            .iter()
            .enumerate()
            .filter(|(_, db)| **db > OFF_DB)
            .map(|(j, db)| {
                let name = channel_name(j, self.n_in);
                if *db == 0.0 {
                    name
                } else {
                    format!("{name} {db:+.1} dB")
                }
            })
            .collect();
        parts.push(if sources.is_empty() {
            "silence".into()
        } else {
            sources.join(" + ")
        });
        if let Some(x) = &o.highpass {
            parts.push(format!("HP {}", x.label()));
        }
        if let Some(x) = &o.lowpass {
            parts.push(format!("LP {}", x.label()));
        }
        if !c.aligned.is_empty() {
            parts.push(format!(
                "aligned to {}",
                c.aligned
                    .iter()
                    .map(|x| freq_label(x.freq))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if let Some(eq) = &c.eq {
            parts.push(format!(
                "EQ {} band{}",
                eq.active_stages(),
                if eq.active_stages() == 1 { "" } else { "s" }
            ));
        }
        if c.user_delay > 0 {
            parts.push(format!(
                "delay {} smp ({:.3} ms)",
                c.user_delay,
                c.user_delay as f64 * 1000.0 / self.rate as f64
            ));
        }
        if o.invert {
            parts.push("inverted".into());
        }
        if o.gain_db != 0.0 {
            parts.push(format!("{:+.1} dB", o.gain_db));
        }
        parts.join(" · ")
    }

    /// One output's processing as JSON for the API.
    fn output_params(&self, i: usize, o: &OutputChannel) -> serde_json::Value {
        let c = &self.chains[i];
        json!({
            "index": i + 1,
            "name": o.name,
            "inputs": o.inputs,
            "sources": o.inputs.iter().enumerate().filter(|(_, db)| **db > OFF_DB)
                .map(|(j, db)| json!({ "input": j + 1, "name": channel_name(j, self.n_in), "gain_db": db })).collect::<Vec<_>>(),
            "gain_db": o.gain_db,
            "invert": o.invert,
            "delay_ms": o.delay_ms,
            "delay_samples": o.delay_samples,
            "delay_total_samples": c.user_delay,
            "align_delay_samples": c.align_delay,
            "lowpass": o.lowpass,
            "highpass": o.highpass,
            "crossover": o.lowpass.or(o.highpass).map(|x| x.kind.id()),
            "phase_aligned_to": c.aligned,
            "iir_sections": c.iir.iter().map(IirCrossover::sections).sum::<usize>(),
            "fir_taps": if c.fir.is_some() { self.fir_taps } else { 0 },
            "eq": o.eq,
            "eq_bands": c.eq.as_ref().map(Equalizer::active_stages).unwrap_or(0),
        })
    }
}

impl std::fmt::Debug for MatrixStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatrixStage")
            .field("rate", &self.rate)
            .field("in", &self.n_in)
            .field("out", &self.n_out)
            .field("block", &self.block)
            .field("fir_taps", &self.fir_taps)
            .field("identity", &self.identity)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn stereo_signal(n: usize, rate: f64) -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                let t = i as f64 / rate;
                [
                    0.5 * (2.0 * PI * 440.0 * t).sin(),
                    0.3 * (2.0 * PI * 1234.5 * t).sin() + 0.1 * (2.0 * PI * 55.0 * t).sin(),
                ]
            })
            .collect()
    }

    fn run_all(stage: &mut MatrixStage, input: &[f64], chunk_frames: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for chunk in input.chunks(chunk_frames * stage.input_channels()) {
            out.extend(stage.process(chunk, false));
        }
        out.extend(stage.process(&[], true));
        out
    }

    fn level_db(x: &[f64]) -> f64 {
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
        20.0 * rms.log10()
    }

    #[test]
    fn identity_matrix_is_bit_identical_and_detected() {
        let cfg = MatrixConfig::preset("stereo");
        let mut stage = MatrixStage::new(&cfg, 44100, 2).unwrap();
        assert!(stage.is_identity());
        assert_eq!(stage.output_channels(), 2);
        let x = stereo_signal(5000, 44100.0);
        let y = run_all(&mut stage, &x, 777);
        assert_eq!(x, y, "identity routing changed a sample");
        // An explicit custom identity table is also detected, and bit-identical.
        let cfg = MatrixConfig::custom(vec![
            OutputChannel::mixed("L", &[0.0, OFF_DB]),
            OutputChannel::mixed("R", &[OFF_DB, 0.0]),
        ]);
        let mut stage = MatrixStage::new(&cfg, 44100, 2).unwrap();
        assert!(stage.is_identity());
        assert_eq!(run_all(&mut stage, &x, 1000), x);
        // Identity for 8 channels too.
        let stage = MatrixStage::new(&MatrixConfig::off(), 48000, 8).unwrap();
        assert!(stage.is_identity() && stage.output_channels() == 8);
        // Anything that touches a sample is not identity.
        let mut o = OutputChannel::from_input("L", 0, 2);
        o.gain_db = -0.1;
        let cfg = MatrixConfig::custom(vec![o, OutputChannel::from_input("R", 1, 2)]);
        assert!(!MatrixStage::new(&cfg, 44100, 2).unwrap().is_identity());
    }

    #[test]
    fn lr24_crossover_sums_flat_within_0_01_db() {
        let rate = 48000u32;
        let x = CrossoverFilter::lr24(1000.0);
        let cfg = MatrixConfig::custom(vec![
            OutputChannel::from_input("low", 0, 1).lowpass(x),
            OutputChannel::from_input("high", 0, 1).highpass(x),
        ]);
        let mut stage = MatrixStage::new(&cfg, rate, 1).unwrap();
        assert_eq!(stage.output_channels(), 2);
        assert!(stage.block().is_none());
        for probe in [50.0, 300.0, 1000.0, 1500.0, 4000.0, 15000.0] {
            stage.reset();
            let n = 48000;
            let input: Vec<f64> = (0..n)
                .map(|i| (2.0 * PI * probe * i as f64 / rate as f64).sin())
                .collect();
            let out = run_all(&mut stage, &input, 4096);
            assert_eq!(out.len(), 2 * n);
            let sum: Vec<f64> = out[n..].chunks(2).map(|f| f[0] + f[1]).collect();
            let db = level_db(&sum) - level_db(&input[n / 2..]);
            assert!(db.abs() < 0.01, "LR24 sum at {probe} Hz: {db:+.4} dB");
        }
        // Three-way too (300 Hz / 3 kHz), via the preset.
        let mut stage = MatrixStage::new(&MatrixConfig::preset("3-way-active"), rate, 2).unwrap();
        assert_eq!(stage.output_channels(), 6);
        for probe in [40.0, 300.0, 1000.0, 3000.0, 12000.0] {
            stage.reset();
            let n = 48000;
            let input: Vec<f64> = (0..n)
                .flat_map(|i| {
                    let v = 0.5 * (2.0 * PI * probe * i as f64 / rate as f64).sin();
                    [v, 0.0]
                })
                .collect();
            let out = run_all(&mut stage, &input, 1000);
            let sum: Vec<f64> = out[6 * n / 2..]
                .chunks(6)
                .map(|f| f[0] + f[1] + f[2])
                .collect();
            let right: Vec<f64> = out.chunks(6).map(|f| f[3] + f[4] + f[5]).collect();
            let db = level_db(&sum)
                - level_db(&input[n..].iter().step_by(2).copied().collect::<Vec<_>>());
            assert!(db.abs() < 0.01, "3-way sum at {probe} Hz: {db:+.4} dB");
            assert!(right.iter().all(|v| *v == 0.0));
        }
        let st = stage.stages("3-way-active", true);
        assert!(
            st[1].detail.contains("aligned to 3 kHz"),
            "{}",
            st[1].detail
        );
        assert!(
            st[3].detail.contains("aligned to 300 Hz"),
            "{}",
            st[3].detail
        );
        assert!(!st[2].detail.contains("aligned"), "{}", st[2].detail);
        assert_eq!(
            st[1].params["phase_aligned_to"][0]["freq"].as_f64(),
            Some(3000.0)
        );
        // A 4-way set sums flat too (woofer gets two all-passes, tweeter two high-passes).
        let f = [150.0, 800.0, 5000.0].map(CrossoverFilter::lr24);
        let cfg = MatrixConfig::custom(vec![
            OutputChannel::from_input("w", 0, 1).lowpass(f[0]),
            OutputChannel::from_input("lm", 0, 1)
                .highpass(f[0])
                .lowpass(f[1]),
            OutputChannel::from_input("um", 0, 1)
                .highpass(f[1])
                .lowpass(f[2]),
            OutputChannel::from_input("t", 0, 1).highpass(f[2]),
        ]);
        let mut stage = MatrixStage::new(&cfg, rate, 1).unwrap();
        for probe in [30.0, 150.0, 400.0, 800.0, 2000.0, 5000.0, 16000.0] {
            stage.reset();
            let n = 48000;
            let input: Vec<f64> = (0..n)
                .map(|i| (2.0 * PI * probe * i as f64 / rate as f64).sin())
                .collect();
            let out = run_all(&mut stage, &input, 4096);
            let sum: Vec<f64> = out[4 * n / 2..].chunks(4).map(|f| f.iter().sum()).collect();
            let db = level_db(&sum) - level_db(&input[n / 2..]);
            assert!(db.abs() < 0.01, "4-way sum at {probe} Hz: {db:+.4} dB");
        }
    }

    #[test]
    fn linear_phase_crossover_sums_to_a_pure_delay_and_aligns_other_channels() {
        let rate = 44100u32;
        let x = CrossoverFilter::linear_phase(2000.0);
        let cfg = MatrixConfig::custom(vec![
            OutputChannel::from_input("low", 0, 2).lowpass(x),
            OutputChannel::from_input("high", 0, 2).highpass(x),
            OutputChannel::from_input("R (no xover)", 1, 2),
        ]);
        let mut stage = MatrixStage::new(&cfg, rate, 2).unwrap();
        let taps = stage.fir_taps();
        assert!(taps > 0 && taps % 2 == 1, "{taps}");
        let gd = (taps - 1) / 2;
        assert!(stage.block().is_some());
        assert_eq!(stage.latency_frames(), stage.block().unwrap() + gd);
        let n = 3 * taps;
        let input = stereo_signal(n, rate as f64);
        let out = run_all(&mut stage, &input, 1234);
        assert_eq!(out.len(), 3 * n);
        for f in gd..n {
            let l_in = input[2 * (f - gd)];
            let r_in = input[2 * (f - gd) + 1];
            let sum = out[3 * f] + out[3 * f + 1];
            assert!((sum - l_in).abs() < 1e-9, "frame {f}: {sum} vs {l_in}");
            assert_eq!(
                out[3 * f + 2],
                r_in,
                "frame {f}: R not aligned to the FIR delay"
            );
        }
    }

    #[test]
    fn delay_is_accurate_to_the_sample() {
        let rate = 48000u32;
        let mut a = OutputChannel::from_input("37 smp", 0, 1);
        a.delay_samples = 37;
        let mut b = OutputChannel::from_input("1 ms", 0, 1);
        b.delay_ms = 1.0; // 48 samples at 48 kHz
        let mut c = OutputChannel::from_input("1 ms + 2", 0, 1);
        c.delay_ms = 1.0;
        c.delay_samples = 2;
        let mut d = OutputChannel::from_input("inverted -6 dB", 0, 1);
        d.invert = true;
        d.gain_db = -6.0;
        let cfg = MatrixConfig::custom(vec![a, b, c, d]);
        let mut stage = MatrixStage::new(&cfg, rate, 1).unwrap();
        let input: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.37).sin()).collect();
        let out = run_all(&mut stage, &input, 300);
        assert_eq!(out.len(), 4 * 2000);
        let g = 10f64.powf(-6.0 / 20.0);
        for f in 0..2000 {
            let fr = &out[4 * f..4 * f + 4];
            let at = |d: usize| if f >= d { input[f - d] } else { 0.0 };
            assert_eq!(fr[0], at(37), "frame {f}");
            assert_eq!(fr[1], at(48), "frame {f}");
            assert_eq!(fr[2], at(50), "frame {f}");
            assert!((fr[3] + g * input[f]).abs() < 1e-15, "frame {f}");
        }
        let st = stage.stages("custom", true);
        assert_eq!(st.len(), 5);
        assert!(st[1].detail.contains("delay 37 smp"), "{}", st[1].detail);
        assert_eq!(st[2].params["delay_total_samples"].as_u64(), Some(48));
        assert!(st[4].detail.contains("inverted") && st[4].detail.contains("-6.0 dB"));
    }

    #[test]
    fn stereo_utilities() {
        let rate = 44100;
        let x = stereo_signal(1000, rate as f64);
        // Mono.
        let mut stage = MatrixStage::new(&MatrixConfig::preset("mono"), rate, 2).unwrap();
        let y = run_all(&mut stage, &x, 100);
        for f in 0..1000 {
            let m = 0.5 * (x[2 * f] + x[2 * f + 1]);
            assert!((y[2 * f] - m).abs() < 1e-15 && (y[2 * f + 1] - m).abs() < 1e-15);
        }
        // Swap.
        let mut stage = MatrixStage::new(&MatrixConfig::preset("swap"), rate, 2).unwrap();
        let y = run_all(&mut stage, &x, 100);
        for f in 0..1000 {
            assert_eq!(y[2 * f], x[2 * f + 1]);
            assert_eq!(y[2 * f + 1], x[2 * f]);
        }
        // Mid/side encode then decode is the identity.
        let mut enc = MatrixStage::new(&MatrixConfig::preset("mid-side"), rate, 2).unwrap();
        let ms = run_all(&mut enc, &x, 100);
        assert!((ms[0] - 0.5 * (x[0] + x[1])).abs() < 1e-15);
        let mut dec_cfg = MatrixConfig::preset("stereo");
        dec_cfg.stereo.mid_side = MidSide::Decode;
        let mut dec = MatrixStage::new(&dec_cfg, rate, 2).unwrap();
        let back = run_all(&mut dec, &ms, 100);
        for (a, b) in back.iter().zip(&x) {
            assert!((a - b).abs() < 1e-12);
        }
        // Balance +3 dB attenuates the left channel only.
        let mut bal = MatrixConfig::preset("stereo");
        bal.stereo.balance_db = 3.0;
        let mut stage = MatrixStage::new(&bal, rate, 2).unwrap();
        assert!(!stage.is_identity());
        let y = run_all(&mut stage, &x, 100);
        let g = 10f64.powf(-3.0 / 20.0);
        for f in 0..1000 {
            assert!((y[2 * f] - g * x[2 * f]).abs() < 1e-15);
            assert_eq!(y[2 * f + 1], x[2 * f + 1]);
        }
        assert_eq!(
            stage.summary("stereo"),
            "2 → 2 ch · stereo · balance +3.0 dB"
        );
        // Utilities are ignored for non-stereo sources.
        let mut six = MatrixConfig::preset("stereo");
        six.stereo.mono = true;
        let stage = MatrixStage::new(&six, rate, 6).unwrap();
        assert!(stage.is_identity());
    }

    #[test]
    fn presets_resolve_and_validate() {
        for p in presets() {
            if p.input_channels == 0 || p.input_channels == 2 {
                let r = MatrixConfig::preset(p.id).resolve(2).unwrap();
                let want = if p.output_channels == 0 {
                    2
                } else {
                    p.output_channels
                };
                assert_eq!(r.outputs.len(), want, "{}", p.id);
                assert_eq!(r.preset, "custom");
            }
        }
        let r = MatrixConfig::preset("2.1").resolve(2).unwrap();
        assert_eq!(r.outputs[2].name, "Sub");
        assert_eq!(r.outputs[2].inputs, vec![-6.0, -6.0]);
        assert_eq!(r.outputs[2].lowpass, Some(CrossoverFilter::lr24(80.0)));
        assert_eq!(r.outputs[0].highpass, Some(CrossoverFilter::lr24(80.0)));
        assert_eq!(
            MatrixConfig::preset("2.1").resolve(6).unwrap_err(),
            MatrixError::PresetChannels {
                preset: "2.1".into(),
                needs: 2,
                has: 6
            }
        );
        assert!(matches!(
            MatrixConfig::preset("7.1-magic").resolve(2),
            Err(MatrixError::UnknownPreset(_))
        ));
        assert!(matches!(
            MatrixConfig::off().resolve(33),
            Err(MatrixError::Channels(_))
        ));
        let many: Vec<OutputChannel> = (0..33)
            .map(|_| OutputChannel::from_input("x", 0, 2))
            .collect();
        assert!(matches!(
            MatrixConfig::custom(many).resolve(2),
            Err(MatrixError::Channels(_))
        ));
        // 32 outputs from 2 inputs is fine.
        let many: Vec<OutputChannel> = (0..32)
            .map(|i| OutputChannel::from_input("x", i % 2, 2))
            .collect();
        let stage = MatrixStage::new(&MatrixConfig::custom(many), 48000, 2).unwrap();
        assert_eq!(stage.output_channels(), 32);
        assert_eq!(stage.stages("custom", true).len(), 33);
        // Per-output validation.
        let mut bad = OutputChannel::from_input("x", 0, 2);
        bad.lowpass = Some(CrossoverFilter::lr24(100.0));
        bad.highpass = Some(CrossoverFilter::lr24(1000.0));
        let e = MatrixStage::new(&MatrixConfig::custom(vec![bad]), 48000, 2).unwrap_err();
        assert!(matches!(e, MatrixError::Output { index: 1, .. }), "{e}");
        let mut bad = OutputChannel::from_input("x", 0, 2);
        bad.inputs = vec![0.0, 0.0, 0.0];
        assert!(MatrixStage::new(&MatrixConfig::custom(vec![bad]), 48000, 2).is_err());
        let mut bad = OutputChannel::from_input("x", 0, 2);
        bad.delay_ms = -1.0;
        assert!(MatrixStage::new(&MatrixConfig::custom(vec![bad]), 48000, 2).is_err());
        let mut bad = OutputChannel::from_input("x", 0, 2);
        bad.lowpass = Some(CrossoverFilter::new(
            CrossoverKind::LinkwitzRiley,
            100.0,
            18,
        ));
        assert!(MatrixStage::new(&MatrixConfig::custom(vec![bad]), 48000, 2).is_err());
    }

    #[test]
    fn per_output_eq_and_signal_path() {
        let rate = 48000u32;
        let mut sub =
            OutputChannel::mixed("Sub", &[-6.0, -6.0]).lowpass(CrossoverFilter::lr24(80.0));
        sub.eq = vec![Band::peaking(40.0, 3.0, 1.0)];
        let cfg = MatrixConfig::custom(vec![
            OutputChannel::from_input("L", 0, 2).highpass(CrossoverFilter::lr24(80.0)),
            OutputChannel::from_input("R", 1, 2).highpass(CrossoverFilter::lr24(80.0)),
            sub,
        ]);
        let mut stage = MatrixStage::new(&cfg, rate, 2).unwrap();
        let n = 48000;
        let input: Vec<f64> = (0..n)
            .flat_map(|i| {
                let v = 0.5 * (2.0 * PI * 40.0 * i as f64 / rate as f64).sin();
                [v, v]
            })
            .collect();
        let out = run_all(&mut stage, &input, 4800);
        let sub_level = level_db(
            &out[3 * n / 2..]
                .iter()
                .skip(2)
                .step_by(3)
                .copied()
                .collect::<Vec<_>>(),
        );
        let in_level = level_db(&input[n..].iter().step_by(2).copied().collect::<Vec<_>>());
        // L+R at −6 dB each = the same level; the +3 dB bell at 40 Hz adds 3 dB, the LR24
        // low-pass an octave above takes its −0.5 dB.
        let lp = IirCrossover::new(&CrossoverFilter::lr24(80.0), Pass::Low, rate as f64);
        let want = 3.0 + lp.response_db(40.0, rate as f64);
        assert!(
            (sub_level - in_level - want).abs() < 0.05,
            "{} vs {want}",
            sub_level - in_level
        );
        let stages = stage.stages("custom", true);
        assert_eq!(stages[0].id, "matrix");
        assert_eq!(stages[0].detail, "2 → 3 ch");
        assert_eq!(stages[0].params["output_channels"].as_u64(), Some(3));
        assert_eq!(stages[3].id, "matrix-out-3");
        assert_eq!(stages[3].name, "Out 3 · Sub");
        assert_eq!(
            stages[3].detail,
            "L -6.0 dB + R -6.0 dB · LP LR24 @ 80 Hz · EQ 1 band"
        );
        assert_eq!(stages[3].params["iir_sections"].as_u64(), Some(2));
        assert_eq!(stages[1].detail, "L · HP LR24 @ 80 Hz");
        assert_eq!(stages[3].params["eq_bands"].as_u64(), Some(1));
        serde_json::to_string(&stages).unwrap();
    }

    #[test]
    fn config_round_trips_through_json_with_defaults() {
        let cfg: MatrixConfig = serde_json::from_str(
            r#"{"enabled": true, "preset": "custom",
                "outputs": [{"name": "Sub", "inputs": [-6, -6], "lowpass": {"freq": 80}},
                            {"inputs": [0], "delay_ms": 2.5, "invert": true}]}"#,
        )
        .unwrap();
        assert_eq!(cfg.outputs[0].lowpass, Some(CrossoverFilter::lr24(80.0)));
        assert_eq!(cfg.outputs[1].delay_ms, 2.5);
        assert!(cfg.outputs[1].invert && cfg.outputs[1].eq.is_empty());
        let s = serde_json::to_string(&cfg).unwrap();
        let back: MatrixConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back, cfg);
        let d: MatrixConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(d, MatrixConfig::default());
        assert_eq!(d.preset, "stereo");
        assert!(!d.enabled);
        assert!(preset_ids().contains(&"3-way-active"));
    }
}
