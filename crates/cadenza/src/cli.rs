//! Command line definitions (`clap`).

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Cadenza: bit-perfect / upsampling music playback engine.
#[derive(Parser, Debug)]
#[command(name = "cadenza", version, about, long_about = None)]
pub struct Cli {
    /// Configuration file (default: ~/.config/cadenza/config.toml).
    #[arg(long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Log level (error, warn, info, debug, trace).
    #[arg(long, global = true, default_value = "info")]
    pub log: String,

    #[command(subcommand)]
    pub command: Commands,
}

/// Settings overrides shared by `play` and `serve`.
#[derive(Args, Debug, Clone, Default)]
pub struct Overrides {
    /// Bypass all DSP: source rate and bits straight to the device.
    #[arg(long)]
    pub bitperfect: bool,

    /// Upsampling filter id (see `cadenza filters`), e.g. sinc-l-1m, sinc-ma-256k.
    #[arg(long, value_name = "ID")]
    pub filter: Option<String>,

    /// Dither / noise shaper: none, rectangular, tpdf, ef2, lipshitz, wannamaker9, wannamaker3, gesemann441, gesemann48.
    #[arg(long, value_name = "ID")]
    pub dither: Option<String>,

    /// Output rate: auto, source, or a rate such as 352800 (or 352.8).
    #[arg(long, value_name = "RATE")]
    pub rate: Option<String>,

    /// Output bit depth: 16, 24 or 32.
    #[arg(long, value_name = "BITS")]
    pub bits: Option<u32>,

    /// Output backend: null, cpal, alsa, wasapi.
    #[arg(long, value_name = "BACKEND")]
    pub backend: Option<String>,

    /// Output device id as printed by `cadenza devices`.
    #[arg(long, value_name = "DEVICE")]
    pub device: Option<String>,

    /// Volume in dB (0 = unity).
    #[arg(long, value_name = "DB")]
    pub volume: Option<f64>,

    /// Impulse response file for the convolution stage (WAV or text).
    #[arg(long, value_name = "FILE")]
    pub convolution: Option<String>,
}

/// Subcommands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Play files or URLs (gapless) and exit when done.
    Play {
        /// Files or http(s) URLs.
        #[arg(required = true)]
        items: Vec<String>,

        /// Start at this position in seconds.
        #[arg(long, default_value_t = 0.0)]
        seek: f64,

        #[command(flatten)]
        overrides: Overrides,
    },
    /// Run the control API, the web UI and (if configured) the Encore device.
    Serve {
        /// Address to listen on, e.g. 127.0.0.1:8777 or 0.0.0.0:8777.
        #[arg(long, value_name = "ADDR")]
        listen: Option<String>,

        /// Do not connect to Encore even if configured.
        #[arg(long)]
        no_encore: bool,

        #[command(flatten)]
        overrides: Overrides,
    },
    /// List output devices and what they support.
    Devices,
    /// List the upsampling filters.
    Filters,
    /// List the dither / noise shaping options.
    Dithers,
    /// Print the effective configuration (and where it lives).
    Config,
}
