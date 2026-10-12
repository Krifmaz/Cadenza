//! Cadenza: the playback engine, outputs, control API and Encore client.
//!
//! The binary in `main.rs` is a thin CLI over this library so the integration tests can
//! drive the real engine with the null output.
#![warn(missing_docs)]

pub mod api;
pub mod config;
pub mod decode;
pub mod encore;
pub mod engine;
pub mod output;
