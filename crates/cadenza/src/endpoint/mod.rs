//! The **Cadenza endpoint**: our network audio adapter (what HQPlayer calls an NAA).
//!
//! `cadenza endpoint` runs on a small box next to the DAC and exposes that box's output
//! devices. The main Cadenza instance selects `endpoint://<name>` as its output backend and
//! streams the *final processed* words – PCM or DSD, exactly what it would hand a local
//! device – over TCP. The endpoint keeps a jitter buffer in front of its output and
//! acknowledges every block with its device clock position; the sender paces itself from
//! those acknowledgements, so there is no resampling anywhere and the DAC's clock is the
//! only clock (clock drift between the two machines simply moves the ack timing).
//!
//! * [`protocol`] – the framed TCP protocol (hello/caps, open, frames, acks, control).
//! * [`server`] – the endpoint process: listener, jitter buffer, local output.
//! * [`client`] – [`client::EndpointOutput`], an [`crate::output::Output`] over the wire.
//! * [`discovery`] – mDNS advertisement and browsing (`_cadenza-endpoint._tcp`).

pub mod client;
pub mod discovery;
pub mod protocol;
pub mod server;

use serde::{Deserialize, Serialize};

pub use client::EndpointOutput;
pub use server::{EndpointServer, EndpointStats};

/// Protocol version spoken by this build.
pub const PROTOCOL_VERSION: u32 = 1;
/// mDNS service type.
pub const SERVICE_TYPE: &str = "_cadenza-endpoint._tcp.local.";
/// Default TCP port of `cadenza endpoint`.
pub const DEFAULT_PORT: u16 = 8778;

/// `[endpoint]` in config.toml: how `cadenza endpoint` runs on the DAC box.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EndpointConfig {
    /// Advertised name (default: `Cadenza endpoint on <hostname>`); the main instance
    /// selects it as `endpoint://<name>`.
    pub name: String,
    /// `host:port` to listen on.
    pub listen: String,
    /// Local output backend the endpoint plays through (`alsa`, `coreaudio`, `wasapi`,
    /// `cpal`, `null`; empty = the platform default).
    pub backend: String,
    /// Local output device id (empty = default).
    pub device: String,
    /// Jitter buffer the endpoint fills before starting and refills after an underrun.
    pub buffer_ms: u32,
    /// Advertise over mDNS.
    pub mdns: bool,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            listen: format!("0.0.0.0:{DEFAULT_PORT}"),
            backend: String::new(),
            device: String::new(),
            buffer_ms: 150,
            mdns: true,
        }
    }
}

impl EndpointConfig {
    /// The advertised name, with the hostname filled in when none is configured.
    pub fn effective_name(&self) -> String {
        if self.name.trim().is_empty() {
            format!(
                "Cadenza endpoint on {}",
                gethostname::gethostname().to_string_lossy()
            )
        } else {
            self.name.trim().to_string()
        }
    }
}

/// Run the endpoint until the process ends (`cadenza endpoint`).
pub fn run(config: EndpointConfig) -> anyhow::Result<()> {
    let server = EndpointServer::start(config)?;
    log::info!(
        "endpoint '{}' listening on {} (devices: {})",
        server.name(),
        server.local_addr(),
        crate::output::backends().join(", ")
    );
    server.run_forever();
    Ok(())
}
