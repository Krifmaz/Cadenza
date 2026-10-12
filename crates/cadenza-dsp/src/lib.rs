//! Cadenza DSP: pure 64-bit floating point audio processing with no I/O.
//!
//! The crate is organised as independent building blocks plus a [`pipeline::Pipeline`]
//! that chains them in the order a playback engine wants:
//! source PCM → volume → EQ → convolution → resampler → dither/quantiser.
#![warn(missing_docs)]

pub mod cache;
pub mod convolution;
pub mod fft;
pub mod fir;
pub mod resampler;
pub mod rng;

pub use rustfft::num_complex::Complex;
