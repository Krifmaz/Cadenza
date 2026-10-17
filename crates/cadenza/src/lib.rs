//! Cadenza: the playback engine, outputs, control API and Encore client.
//!
//! The binary in `main.rs` is a thin CLI over this library so the integration tests can
//! drive the real engine with the null output.
#![warn(missing_docs)]

pub mod airplay;
pub mod api;
pub mod bench;
pub mod config;
pub mod decode;
pub mod embed;
pub mod encore;
pub mod endpoint;
pub mod engine;
pub mod library;
pub mod output;
pub mod render;
pub mod room;
pub mod service;
pub mod system;
pub mod upnp;
