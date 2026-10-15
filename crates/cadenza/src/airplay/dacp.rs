//! DACP: the sender's remote-control service, so play / pause / next / previous pressed in
//! Cadenza reach the iPhone or Mac that is playing.
//!
//! Every AirPlay request carries `DACP-ID` and `Active-Remote`. The sender advertises
//! `iTunes_Ctrl_<DACP-ID>._dacp._tcp` over mDNS; a command is an HTTP GET of
//! `/ctrl-int/1/<command>` on that port with the `Active-Remote` header.

use crate::decode::live::{ControlFn, LiveControl};
use std::net::{IpAddr, SocketAddr};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Finds a sender's DACP address from its id and IP.
pub type Resolver = Arc<dyn Fn(&str, IpAddr) -> Option<SocketAddr> + Send + Sync>;

/// The DACP path of a command.
pub fn command_path(c: LiveControl) -> &'static str {
    match c {
        LiveControl::Play => "/ctrl-int/1/play",
        LiveControl::Pause => "/ctrl-int/1/pause",
        LiveControl::Next => "/ctrl-int/1/nextitem",
        LiveControl::Prev => "/ctrl-int/1/previtem",
    }
}

/// Look the sender up over mDNS (up to 3 s). The sender's own address is preferred.
pub fn mdns_resolver() -> Resolver {
    Arc::new(|id: &str, peer: IpAddr| {
        let daemon = mdns_sd::ServiceDaemon::new().ok()?;
        let rx = daemon.browse("_dacp._tcp.local.").ok()?;
        let want = format!("itunes_ctrl_{}.", id.to_ascii_lowercase());
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut found = None;
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            match rx.recv_timeout(left) {
                Ok(mdns_sd::ServiceEvent::ServiceResolved(info))
                    if info.get_fullname().to_ascii_lowercase().starts_with(&want) =>
                {
                    let addrs = info.get_addresses();
                    let ip = if addrs.contains(&peer) || addrs.is_empty() {
                        peer
                    } else {
                        *addrs.iter().next()?
                    };
                    found = Some(SocketAddr::new(ip, info.get_port()));
                    break;
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        let _ = daemon.shutdown();
        found
    })
}

/// A control function that sends commands to the sender from a thread of its own (the
/// engine never waits on the network). The address is resolved on first use and kept.
pub fn remote(
    dacp_id: String,
    active_remote: String,
    peer: IpAddr,
    resolve: Resolver,
) -> ControlFn {
    let (tx, rx) = mpsc::channel::<LiveControl>();
    let spawned = std::thread::Builder::new()
        .name("cadenza-airplay-dacp".into())
        .spawn(move || {
            let agent = ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(2))
                .build();
            let mut addr = None;
            while let Ok(c) = rx.recv() {
                if addr.is_none() {
                    addr = resolve(&dacp_id, peer);
                }
                let Some(a) = addr else {
                    log::warn!("AirPlay: the sender's remote control (DACP {dacp_id}) not found");
                    continue;
                };
                let url = format!("http://{a}{}", command_path(c));
                if let Err(e) = agent.get(&url).set("Active-Remote", &active_remote).call() {
                    log::warn!("AirPlay: {url}: {e}");
                    addr = None; // the sender may have moved; look again next time
                }
            }
        });
    if let Err(e) = spawned {
        log::warn!("AirPlay: remote control thread: {e}");
    }
    Arc::new(move |c| {
        let _ = tx.send(c);
    })
}
