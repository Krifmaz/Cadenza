//! AirPlay 1 (RAOP) receiver: iPhones, iPads, Macs and iTunes / Music play into Cadenza,
//! and the audio goes through the engine like any other source (upsampling, DSD, EQ…).
//!
//! The receiver advertises `_raop._tcp` over mDNS, speaks RTSP on its own port and
//! receives Apple Lossless over RTP/UDP, decrypting it with the session key the sender
//! wraps for the AirPort Express key (see [`crypto`]). Off unless `[airplay] enabled =
//! true`. AirPlay 2 (multi-room from the Home app, AAC, buffered audio) is a different,
//! closed protocol and is not implemented; senders fall back to AirPlay 1 for receivers
//! that only advertise it.

pub mod audio;
pub mod crypto;
pub mod dacp;
pub mod dmap;
#[cfg(test)]
mod e2e;
pub mod rtp;
pub mod rtsp;

use crate::engine::Engine;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// `[airplay]` in config.toml.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AirplayConfig {
    /// Run the receiver (`cadenza serve`).
    pub enabled: bool,
    /// Name shown on the sender (empty: `Cadenza (hostname)`).
    pub name: String,
    /// RTSP port (0: any free port).
    pub port: u16,
    /// Audio buffered before playing, ms. Senders schedule about 2 s ahead, so 2000
    /// keeps sound in step with video; lower starts faster.
    pub latency_ms: u32,
    /// Follow the sender's volume slider (engine volume); off leaves the volume alone.
    pub volume: bool,
}

impl Default for AirplayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            name: String::new(),
            port: 5000,
            latency_ms: 2000,
            volume: true,
        }
    }
}

impl AirplayConfig {
    /// The advertised name.
    pub fn friendly_name(&self) -> String {
        if self.name.trim().is_empty() {
            let host = gethostname::gethostname().to_string_lossy().into_owned();
            format!("Cadenza ({host})")
        } else {
            self.name.trim().to_string()
        }
    }
}

/// A stable, locally administered "hardware address" for this host and name. Senders
/// expect it in the service name and in the `Apple-Response`.
pub fn hardware_address(name: &str) -> [u8; 6] {
    let host = gethostname::gethostname().to_string_lossy().into_owned();
    let h = crate::upnp::fnv(&[b"cadenza-airplay", host.as_bytes(), name.as_bytes()]);
    let b = h.to_be_bytes();
    [(b[0] & 0xfe) | 0x02, b[1], b[2], b[3], b[4], b[5]]
}

/// The mDNS TXT record of an AirPlay 1 receiver: stereo 44.1/16, PCM / ALAC, plain or
/// RSA-wrapped AES, text / artwork / progress metadata, no password.
pub fn txt_record() -> Vec<(&'static str, &'static str)> {
    vec![
        ("txtvers", "1"),
        ("ch", "2"),
        ("cn", "0,1"),
        ("et", "0,1"),
        ("md", "0,1,2"),
        ("pw", "false"),
        ("sr", "44100"),
        ("ss", "16"),
        ("tp", "UDP"),
        ("vn", "65537"),
        ("vs", "130.14"),
        ("am", "Cadenza"),
        ("sf", "0x4"),
        ("da", "true"),
        ("sv", "false"),
    ]
}

/// The running receiver; dropping it stops it and withdraws the advertisement.
pub struct AirplayReceiver {
    stop: Arc<AtomicBool>,
    mdns: Option<(mdns_sd::ServiceDaemon, String)>,
    /// RTSP port.
    pub port: u16,
    /// The hardware address in use.
    pub hw: [u8; 6],
    /// The advertised service instance (`0211AABBCCDD@Name`).
    pub instance: String,
}

impl Drop for AirplayReceiver {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some((daemon, fullname)) = self.mdns.take() {
            if let Ok(rx) = daemon.unregister(&fullname) {
                let _ = rx.recv_timeout(Duration::from_millis(500));
            }
            let _ = daemon.shutdown();
        }
    }
}

/// Start the receiver for `engine` (with `announce`, on mDNS too).
pub fn start(
    engine: Engine,
    config: &AirplayConfig,
    announce: bool,
) -> anyhow::Result<AirplayReceiver> {
    start_with(engine, config, announce, dacp::mdns_resolver())
}

/// [`start`] with a given way of finding senders' remote controls.
pub fn start_with(
    engine: Engine,
    config: &AirplayConfig,
    announce: bool,
    resolver: dacp::Resolver,
) -> anyhow::Result<AirplayReceiver> {
    let name = config.friendly_name();
    let hw = hardware_address(&name);
    let listener = std::net::TcpListener::bind(("0.0.0.0", config.port))?;
    let port = listener.local_addr()?.port();
    let instance = format!(
        "{}@{name}",
        hw.iter().map(|b| format!("{b:02X}")).collect::<String>()
    );
    let stop = Arc::new(AtomicBool::new(false));
    let shared = rtsp::Shared::new(engine, config.clone(), hw, resolver);
    {
        let stop = Arc::clone(&stop);
        std::thread::Builder::new()
            .name("cadenza-airplay".into())
            .spawn(move || rtsp::serve(listener, shared, stop))?;
    }
    let mdns = if announce {
        match advertise(&instance, port) {
            Ok(m) => Some(m),
            Err(e) => {
                log::warn!("AirPlay: mDNS failed ({e}); not discoverable");
                None
            }
        }
    } else {
        None
    };
    log::info!("AirPlay receiver \"{name}\" on port {port}");
    Ok(AirplayReceiver {
        stop,
        mdns,
        port,
        hw,
        instance,
    })
}

fn advertise(instance: &str, port: u16) -> Result<(mdns_sd::ServiceDaemon, String), String> {
    let daemon = mdns_sd::ServiceDaemon::new().map_err(|e| e.to_string())?;
    let host = gethostname::gethostname().to_string_lossy().into_owned();
    let host = format!(
        "{}.local.",
        host.trim_end_matches(".local").trim_end_matches('.')
    );
    let info = mdns_sd::ServiceInfo::new(
        "_raop._tcp.local.",
        instance,
        &host,
        "",
        port,
        &txt_record()[..],
    )
    .map_err(|e| e.to_string())?
    .enable_addr_auto();
    let fullname = info.get_fullname().to_string();
    daemon.register(info).map_err(|e| e.to_string())?;
    Ok((daemon, fullname))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_address_is_stable_and_local() {
        let a = hardware_address("Living room");
        assert_eq!(a, hardware_address("Living room"));
        assert_ne!(a, hardware_address("Kitchen"));
        assert_eq!(a[0] & 0x03, 0x02); // unicast, locally administered
        let txt = txt_record();
        assert!(txt.contains(&("cn", "0,1")) && txt.contains(&("et", "0,1")));
    }
}
