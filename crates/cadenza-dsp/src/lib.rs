//! Cadenza DSP: pure 64-bit floating point audio processing with no I/O.
//!
//! The crate is organised as independent building blocks plus a [`pipeline::Pipeline`]
//! that chains them in the order a playback engine wants:
//! source PCM → volume → EQ → convolution → matrix → resampler → dither/quantiser.
#![warn(missing_docs)]

pub mod analysis;
pub mod bench;
pub mod cache;
pub mod convolution;
pub mod design;
pub mod dither;
pub mod dsd;
pub mod eq;
pub mod fft;
pub mod fir;
pub mod gpu;
pub mod headphone;
pub mod headroom;
mod kemar;
pub mod matrix;
pub mod pipeline;
pub mod resampler;
pub mod rng;
pub mod room;
pub mod signal_path;
pub mod volume;

pub use cache::CoefficientCache;
pub use dither::DitherKind;
pub use eq::{Band, BandKind};
pub use fir::{Family, FilterInfo, FilterKind, FirSpec, Phase};
pub use gpu::{GpuConfig, GpuMode, GpuPrecision};
pub use headphone::HeadphoneMode;
pub use headroom::Headroom;
pub use matrix::{MatrixConfig, MatrixStage};
pub use pipeline::{ConvolutionSpec, InputFormat, Pipeline, PipelineConfig};
pub use resampler::Resampler;
pub use rustfft::num_complex::Complex;
pub use signal_path::SignalPath;
pub use volume::Volume;
