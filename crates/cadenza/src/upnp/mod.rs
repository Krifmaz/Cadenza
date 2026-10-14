//! UPnP / DLNA MediaRenderer: control points on the network (BubbleUPnP, mconnect,
//! Audirvana, Kazoo/Linn apps in UPnP mode, foobar2000 …) find Cadenza with SSDP and play
//! into it through AVTransport; RenderingControl drives the engine's volume.
//!
//! The renderer has its own HTTP listener on all interfaces (`[upnp] port`, 49494 by
//! default) – the control API stays on its own address – and is off unless
//! `[upnp] enabled = true` (it opens a port to the network). Only `http(s)` media URIs
//! are played, never local paths.

pub mod description;
pub mod http;
pub mod openhome;
pub mod renderer;
pub mod ssdp;
pub mod xml;

use crate::engine::Engine;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// `[upnp]` in config.toml.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UpnpConfig {
    /// Run the renderer (`cadenza serve`).
    pub enabled: bool,
    /// Name shown in control apps (empty: `Cadenza (hostname)`).
    pub name: String,
    /// HTTP port of the descriptions, control and events (0: any free port).
    pub port: u16,
}

impl Default for UpnpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            name: String::new(),
            port: 49494,
        }
    }
}

impl UpnpConfig {
    /// The friendly name in use.
    pub fn friendly_name(&self) -> String {
        if self.name.trim().is_empty() {
            let host = gethostname::gethostname().to_string_lossy().into_owned();
            format!("Cadenza ({host})")
        } else {
            self.name.trim().to_string()
        }
    }
}

/// FNV-1a over `parts`.
pub fn fnv(parts: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in parts {
        for b in *p {
            h = (h ^ *b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// A version-4-shaped UUID from two 64-bit values.
pub fn uuid_from(a: u64, b: u64) -> String {
    let a = (a & 0xffff_ffff_ffff_0fff) | 0x4000;
    let b = (b & 0x3fff_ffff_ffff_ffff) | 0x8000_0000_0000_0000;
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        a >> 32,
        (a >> 16) & 0xffff,
        a & 0xffff,
        b >> 48,
        b & 0xffff_ffff_ffff
    )
}

/// The device's UDN: stable for a host and name, so control apps remember it.
pub fn udn_for(name: &str) -> String {
    let host = gethostname::gethostname().to_string_lossy().into_owned();
    let a = fnv(&[b"cadenza-renderer", host.as_bytes(), name.as_bytes()]);
    let b = fnv(&[name.as_bytes(), host.as_bytes(), b"cadenza-renderer"]);
    format!("uuid:{}", uuid_from(a, b))
}

/// The running renderer; dropping it stops the announcements (with `ssdp:byebye`).
pub struct UpnpRenderer {
    _ssdp: Option<ssdp::Ssdp>,
    stop: Arc<AtomicBool>,
    /// The HTTP port it listens on.
    pub port: u16,
    /// The UDN it announces.
    pub udn: String,
}

impl Drop for UpnpRenderer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Start the renderer for `engine`: HTTP listener, event notifier and (with `announce`)
/// SSDP. Fails when the port is taken.
pub fn start(engine: Engine, config: &UpnpConfig, announce: bool) -> anyhow::Result<UpnpRenderer> {
    let name = config.friendly_name();
    let udn = udn_for(&name);
    let listener = std::net::TcpListener::bind(("0.0.0.0", config.port))?;
    listener.set_nonblocking(true)?;
    let port = listener.local_addr()?.port();
    let shared = http::Shared::new(
        renderer::Renderer::new(engine, name.clone()),
        name.clone(),
        udn.clone(),
    );
    let stop = Arc::new(AtomicBool::new(false));
    {
        let shared = Arc::clone(&shared);
        let stop = Arc::clone(&stop);
        std::thread::Builder::new()
            .name("cadenza-upnp-http".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        log::error!("UPnP: no runtime: {e}");
                        return;
                    }
                };
                rt.block_on(async move {
                    let Ok(listener) = tokio::net::TcpListener::from_std(listener) else {
                        log::error!("UPnP: listener setup failed");
                        return;
                    };
                    let shutdown = async move {
                        while !stop.load(Ordering::Relaxed) {
                            tokio::time::sleep(Duration::from_millis(250)).await;
                        }
                    };
                    let _ = axum::serve(listener, http::router(shared))
                        .with_graceful_shutdown(shutdown)
                        .await;
                });
            })?;
    }
    {
        let shared = Arc::clone(&shared);
        let stop = Arc::clone(&stop);
        std::thread::Builder::new()
            .name("cadenza-upnp-events".into())
            .spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    http::tick(&shared);
                    std::thread::sleep(Duration::from_millis(500));
                }
            })?;
    }
    let ssdp = if announce {
        match ssdp::Ssdp::start(ssdp::Advert {
            udn: udn.clone(),
            port,
        }) {
            Ok(s) => Some(s),
            Err(e) => {
                log::warn!("UPnP: SSDP on port 1900 failed ({e}); not discoverable");
                None
            }
        }
    } else {
        None
    };
    log::info!("UPnP renderer \"{name}\" on port {port} ({udn})");
    Ok(UpnpRenderer {
        _ssdp: ssdp,
        stop,
        port,
        udn,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn udn_is_stable_and_uuid_shaped() {
        let a = udn_for("Living room");
        assert_eq!(a, udn_for("Living room"));
        assert_ne!(a, udn_for("Kitchen"));
        let u = a.strip_prefix("uuid:").unwrap();
        let parts: Vec<&str> = u.split('-').collect();
        assert_eq!(
            parts.iter().map(|p| p.len()).collect::<Vec<_>>(),
            [8, 4, 4, 4, 12]
        );
        assert!(parts[2].starts_with('4'));
        assert!(matches!(&parts[3][..1], "8" | "9" | "a" | "b"));
        assert!(UpnpConfig::default()
            .friendly_name()
            .starts_with("Cadenza ("));
    }
}
