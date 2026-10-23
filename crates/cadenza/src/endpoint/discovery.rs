//! mDNS for the endpoint: `cadenza endpoint` advertises `_cadenza-endpoint._tcp` and the
//! main instance browses for it. One daemon per process, started on first use.

use super::SERVICE_TYPE;
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

/// A registered advertisement; unregistered when dropped.
pub struct Advertisement {
    /// The mDNS daemon the service is registered with.
    daemon: ServiceDaemon,
    /// Full service name, to unregister it.
    fullname: String,
}

impl Advertisement {
    /// Advertise `name` on `port` on every interface.
    pub fn start(name: &str, port: u16) -> Result<Self, String> {
        let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
        let host = gethostname::gethostname().to_string_lossy().into_owned();
        let host = format!(
            "{}.local.",
            host.trim_end_matches(".local").trim_end_matches('.')
        );
        let props = [("version", super::PROTOCOL_VERSION.to_string())];
        let info = ServiceInfo::new(SERVICE_TYPE, name, &host, "", port, &props[..])
            .map_err(|e| e.to_string())?
            .enable_addr_auto();
        let fullname = info.get_fullname().to_string();
        daemon.register(info).map_err(|e| e.to_string())?;
        Ok(Self { daemon, fullname })
    }
}

impl Drop for Advertisement {
    fn drop(&mut self) {
        if let Ok(rx) = self.daemon.unregister(&self.fullname) {
            let _ = rx.recv_timeout(Duration::from_millis(500));
        }
        let _ = self.daemon.shutdown();
    }
}

/// The process-wide browser for endpoints.
struct Browser {
    /// Instance name → address.
    found: Mutex<BTreeMap<String, SocketAddr>>,
    /// When browsing started (lookups wait a grace period after it).
    started: Instant,
}

/// The browser, started on first use.
static BROWSER: OnceLock<Arc<Browser>> = OnceLock::new();

/// Instance name from `Name._cadenza-endpoint._tcp.local.`.
fn instance_name(fullname: &str) -> String {
    fullname
        .strip_suffix(SERVICE_TYPE)
        .map(|s| s.trim_end_matches('.'))
        .unwrap_or(fullname)
        .to_string()
}

/// The address to connect to: IPv4 first, then any non-link-local IPv6.
fn pick_addr(info: &ServiceInfo) -> Option<SocketAddr> {
    let mut addrs: Vec<&IpAddr> = info.get_addresses().iter().collect();
    // Prefer IPv4, then anything that is not link-local v6.
    addrs.sort_by_key(|a| match a {
        IpAddr::V4(_) => 0,
        IpAddr::V6(v6) if (v6.segments()[0] & 0xffc0) == 0xfe80 => 2,
        IpAddr::V6(_) => 1,
    });
    addrs.first().map(|a| SocketAddr::new(**a, info.get_port()))
}

/// The browser, starting it (and its listener thread) on first use.
fn browser() -> Arc<Browser> {
    Arc::clone(BROWSER.get_or_init(|| {
        let b = Arc::new(Browser {
            found: Mutex::new(BTreeMap::new()),
            started: Instant::now(),
        });
        let bg = Arc::clone(&b);
        let _ = thread::Builder::new()
            .name("cadenza-mdns".into())
            .spawn(move || {
                let Ok(daemon) = ServiceDaemon::new() else {
                    log::warn!("endpoint discovery: cannot start mDNS");
                    return;
                };
                let Ok(rx) = daemon.browse(SERVICE_TYPE) else {
                    return;
                };
                while let Ok(ev) = rx.recv() {
                    match ev {
                        ServiceEvent::ServiceResolved(info) => {
                            if let Some(addr) = pick_addr(&info) {
                                let name = instance_name(info.get_fullname());
                                log::info!("endpoint discovery: '{name}' at {addr}");
                                bg.found.lock().unwrap().insert(name, addr);
                            }
                        }
                        ServiceEvent::ServiceRemoved(_, fullname) => {
                            bg.found.lock().unwrap().remove(&instance_name(&fullname));
                        }
                        _ => {}
                    }
                }
            });
        b
    }))
}

/// Endpoints known so far; waits up to `grace` after the browser started so a fresh
/// process gets a first answer.
pub fn known(grace: Duration) -> Vec<(String, SocketAddr)> {
    let b = browser();
    let deadline = b.started + grace;
    loop {
        let snapshot: Vec<(String, SocketAddr)> = b
            .found
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        if !snapshot.is_empty() || Instant::now() >= deadline {
            return snapshot;
        }
        thread::sleep(Duration::from_millis(50));
    }
}

/// Find the endpoint called `name` (case-insensitive; empty = any), waiting up to
/// `timeout`.
pub fn find(name: &str, timeout: Duration) -> Option<SocketAddr> {
    let b = browser();
    let deadline = Instant::now() + timeout;
    loop {
        {
            let found = b.found.lock().unwrap();
            if name.is_empty() {
                if let Some((_, a)) = found.iter().next() {
                    return Some(*a);
                }
            } else if let Some((_, a)) = found.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)) {
                return Some(*a);
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_names_are_stripped_of_the_service_type() {
        assert_eq!(
            instance_name("Kitchen DAC._cadenza-endpoint._tcp.local."),
            "Kitchen DAC"
        );
        assert_eq!(instance_name("odd"), "odd");
    }
}
