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

    /// Dither / noise shaper: none, rectangular, tpdf, ef2, lipshitz, wannamaker9, wannamaker3,
    /// gesemann441, gesemann48, ns5, ns9, lns15, ns-adaptive (see `cadenza dithers`).
    #[arg(long, value_name = "ID")]
    pub dither: Option<String>,

    /// Output mode: pcm (quantiser) or dsd (sigma-delta modulator).
    #[arg(long, value_name = "MODE")]
    pub output_mode: Option<String>,

    /// DSD output rate: dsd64, dsd128, dsd256, dsd512, dsd1024, dsd2048 (native only).
    #[arg(long, value_name = "RATE")]
    pub dsd_rate: Option<String>,

    /// DSD modulator: asdm5, asdm7, dsd5v2, dsd7v2.
    #[arg(long, value_name = "ID")]
    pub dsd_modulator: Option<String>,

    /// DSD packing: dop or native.
    #[arg(long, value_name = "PACKING")]
    pub dsd_packing: Option<String>,

    /// Gain before the modulator in dB (headroom, default -3).
    #[arg(long, value_name = "DB")]
    pub dsd_gain: Option<f64>,

    /// DSD oversampling: staged (default) or full (the long FIR runs to the DSD rate).
    #[arg(long, value_name = "MODE")]
    pub dsd_oversampling: Option<String>,

    /// Intersample-peak headroom in dB ("auto" = -3 dB when upsampling).
    #[arg(long, value_name = "DB")]
    pub headroom: Option<String>,

    /// GPU offload: auto, off or cuda.
    #[arg(long, value_name = "MODE")]
    pub gpu: Option<String>,

    /// Pipeline matrix preset: stereo, mono, swap, mid-side, 2.1, 2-way-active, 3-way-active, custom
    /// ("off" disables the matrix).
    #[arg(long, value_name = "PRESET")]
    pub matrix: Option<String>,

    /// Output rate: auto, source, or a rate such as 352800 (or 352.8).
    #[arg(long, value_name = "RATE")]
    pub rate: Option<String>,

    /// Output bit depth: 16, 24 or 32.
    #[arg(long, value_name = "BITS")]
    pub bits: Option<u32>,

    /// Output backend: null, cpal, alsa, wasapi, coreaudio, or endpoint://<name>.
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
    /// List output devices and what they support (rates, bits, DSD, exclusive).
    Devices,
    /// List the upsampling filters with taps, transition band, stopband and latency.
    Filters {
        /// Describe the filters for this source rate (Hz), default 44100.
        #[arg(long, value_name = "RATE")]
        in_rate: Option<u32>,
        /// … and this output rate (Hz), default 705600 (x16).
        #[arg(long, value_name = "RATE")]
        out_rate: Option<u32>,
    },
    /// List the dither / noise shaping options.
    Dithers,
    /// Print the effective configuration (and where it lives).
    Config,
    /// Measure how many seconds of audio per second this machine processes (CPU and, when
    /// present, GPU) for a filter and rate, the convolver, and the DSD modulator at every rate.
    Bench {
        /// Upsampling filter id (default: the configured one).
        #[arg(long, value_name = "ID")]
        filter: Option<String>,
        /// Output rate in Hz (or kHz below 1000); default 705600.
        #[arg(long, value_name = "RATE")]
        rate: Option<String>,
        /// Source rate in Hz; default: the base of the output rate's family.
        #[arg(long, value_name = "RATE")]
        in_rate: Option<u32>,
        /// Seconds of test audio per measurement.
        #[arg(long, default_value_t = 2.0)]
        seconds: f64,
        /// Channels.
        #[arg(long, default_value_t = 2)]
        channels: usize,
        /// Skip the per-rate DSD modulator rows.
        #[arg(long)]
        no_dsd: bool,
        /// Also time the long filter feeding each DSD rate (staged: to f_dsd/8 or /16).
        #[arg(long)]
        dsd_upsampling: bool,
        /// Print JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Run the network endpoint next to the DAC: exposes this machine's output devices to a
    /// Cadenza instance that selects `endpoint://<name>` as its backend.
    Endpoint {
        /// Advertised name (default: "Cadenza endpoint on <hostname>").
        #[arg(long, value_name = "NAME")]
        name: Option<String>,
        /// Address to listen on, e.g. 0.0.0.0:8778.
        #[arg(long, value_name = "ADDR")]
        listen: Option<String>,
        /// Local output backend (alsa, coreaudio, wasapi, cpal, null).
        #[arg(long, value_name = "BACKEND")]
        backend: Option<String>,
        /// Local output device id.
        #[arg(long, value_name = "DEVICE")]
        device: Option<String>,
        /// Jitter buffer in milliseconds.
        #[arg(long, value_name = "MS")]
        buffer_ms: Option<u32>,
        /// Do not advertise over mDNS.
        #[arg(long)]
        no_mdns: bool,
    },
}
