//! Cadenza DSP: pure 64-bit floating point audio processing with no I/O.
//!
//! The crate is organised as independent building blocks plus a [`pipeline::Pipeline`]
//! that chains them in the order a playback engine wants:
//! source PCM → volume → EQ → convolution → resampler → dither/quantiser.
#![warn(missing_docs)]

pub mod cache;
pub mod convolution;
pub mod dither;
pub mod eq;
pub mod fft;
pub mod fir;
pub mod pipeline;
pub mod resampler;
pub mod rng;
pub mod signal_path;
pub mod volume;

pub use cache::CoefficientCache;
pub use dither::DitherKind;
pub use eq::{Band, BandKind};
pub use fir::{FilterKind, FirSpec, Phase};
pub use pipeline::{ConvolutionSpec, InputFormat, Pipeline, PipelineConfig};
pub use resampler::Resampler;
pub use rustfft::num_complex::Complex;
pub use signal_path::SignalPath;
pub use volume::Volume;
