//! SSDP: answer `M-SEARCH` requests and announce the renderer (`NOTIFY ssdp:alive` at
//! start and every 15 minutes, `ssdp:byebye` at shutdown) on every IPv4 interface.

use super::description::{DEVICE_TYPE, SERVICES};
use socket2::{Domain, Protocol, Socket, Type};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The SSDP multicast group.
pub const GROUP: Ipv4Addr = Ipv4Addr::new(239, 255, 255, 250);
/// The SSDP port.
pub const PORT: u16 = 1900;
/// How long announcements are valid (we re-announce well before).
const MAX_AGE: u32 = 1800;

/// What we advertise.
#[derive(Clone, Debug)]
pub struct Advert {
    /// `uuid:…`.
    pub udn: String,
    /// HTTP port of the description.
    pub port: u16,
}

impl Advert {
    /// Every `(NT/ST, USN)` pair of the device.
    pub fn targets(&self) -> Vec<(String, String)> {
        let mut v = vec![
            (
                "upnp:rootdevice".to_string(),
                format!("{}::upnp:rootdevice", self.udn),
            ),
            (self.udn.clone(), self.udn.clone()),
            (
                DEVICE_TYPE.to_string(),
                format!("{}::{DEVICE_TYPE}", self.udn),
            ),
        ];
        for s in &SERVICES {
            v.push((
                s.service_type.to_string(),
                format!("{}::{}", self.udn, s.service_type),
            ));
        }
        v
    }

    /// The pairs a search target `st` asks for (empty when it is not us).
    pub fn matching(&self, st: &str) -> Vec<(String, String)> {
        let st = st.trim();
        if st == "ssdp:all" {
            return self.targets();
        }
        self.targets()
            .into_iter()
            .filter(|(t, _)| t == st)
            .collect()
    }

    fn location(&self, ip: Ipv4Addr) -> String {
        format!("http://{ip}:{}/upnp/device.xml", self.port)
    }

    /// The unicast answer to a search for `st` with USN `usn`, from interface `ip`.
    pub fn response(&self, st: &str, usn: &str, ip: Ipv4Addr) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nCACHE-CONTROL: max-age={MAX_AGE}\r\nEXT:\r\nLOCATION: {}\r\n\
SERVER: {}\r\nST: {st}\r\nUSN: {usn}\r\nBOOTID.UPNP.ORG: 1\r\nCONFIGID.UPNP.ORG: 1\r\n\r\n",
            self.location(ip),
            server()
        )
    }

    /// A multicast announcement (`ssdp:alive` or `ssdp:byebye`).
    pub fn notify(&self, nt: &str, usn: &str, alive: bool, ip: Ipv4Addr) -> String {
        if alive {
            format!(
                "NOTIFY * HTTP/1.1\r\nHOST: {GROUP}:{PORT}\r\nCACHE-CONTROL: max-age={MAX_AGE}\r\n\
LOCATION: {}\r\nNT: {nt}\r\nNTS: ssdp:alive\r\nSERVER: {}\r\nUSN: {usn}\r\n\
BOOTID.UPNP.ORG: 1\r\nCONFIGID.UPNP.ORG: 1\r\n\r\n",
                self.location(ip),
                server()
            )
        } else {
            format!(
                "NOTIFY * HTTP/1.1\r\nHOST: {GROUP}:{PORT}\r\nNT: {nt}\r\nNTS: ssdp:byebye\r\n\
USN: {usn}\r\nBOOTID.UPNP.ORG: 1\r\nCONFIGID.UPNP.ORG: 1\r\n\r\n"
            )
        }
    }
}

fn server() -> String {
    format!(
        "{}/1.0 UPnP/1.0 Cadenza/{}",
        std::env::consts::OS,
        env!("CARGO_PKG_VERSION")
    )
}

/// The search target of an `M-SEARCH` request, or `None` for anything else.
pub fn parse_search(msg: &str) -> Option<String> {
    let mut lines = msg.lines();
    if !lines
        .next()?
        .trim()
        .eq_ignore_ascii_case("M-SEARCH * HTTP/1.1")
    {
        return None;
    }
    let mut st = None;
    let mut discover = false;
    for line in lines {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        match k.trim().to_ascii_uppercase().as_str() {
            "ST" => st = Some(v.trim().to_string()),
            "MAN" => discover = v.trim().trim_matches('"') == "ssdp:discover",
            _ => {}
        }
    }
    st.filter(|_| discover)
}

/// The non-loopback IPv4 interfaces with their netmasks.
fn interfaces() -> Vec<(Ipv4Addr, Ipv4Addr)> {
    if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|i| match i.addr {
            if_addrs::IfAddr::V4(v4) if !v4.ip.is_loopback() => Some((v4.ip, v4.netmask)),
            _ => None,
        })
        .collect()
}

/// The interface address on `peer`'s subnet (else the first one).
fn local_for(peer: Ipv4Addr, ifaces: &[(Ipv4Addr, Ipv4Addr)]) -> Option<Ipv4Addr> {
    let same = |(ip, mask): &(Ipv4Addr, Ipv4Addr)| {
        u32::from(*ip) & u32::from(*mask) == u32::from(peer) & u32::from(*mask)
    };
    ifaces
        .iter()
        .find(|i| same(i))
        .or_else(|| ifaces.first())
        .map(|i| i.0)
}

/// The running responder; dropping it says goodbye.
pub struct Ssdp {
    stop: Arc<AtomicBool>,
    advert: Advert,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn announce(advert: &Advert, alive: bool) {
    for (ip, _) in interfaces() {
        let Ok(sock) = UdpSocket::bind(SocketAddrV4::new(ip, 0)) else {
            continue;
        };
        let _ = sock.set_multicast_ttl_v4(4);
        for (nt, usn) in advert.targets() {
            let msg = advert.notify(&nt, &usn, alive, ip);
            let _ = sock.send_to(msg.as_bytes(), SocketAddrV4::new(GROUP, PORT));
        }
    }
}

impl Ssdp {
    /// Join the SSDP group on every interface and start answering and announcing.
    pub fn start(advert: Advert) -> std::io::Result<Ssdp> {
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        socket.set_reuse_address(true)?;
        #[cfg(unix)]
        let _ = socket.set_reuse_port(true);
        socket.bind(&SocketAddr::from((Ipv4Addr::UNSPECIFIED, PORT)).into())?;
        let socket: UdpSocket = socket.into();
        let mut joined = 0;
        for (ip, _) in interfaces() {
            if socket.join_multicast_v4(&GROUP, &ip).is_ok() {
                joined += 1;
            }
        }
        if joined == 0 {
            socket.join_multicast_v4(&GROUP, &Ipv4Addr::UNSPECIFIED)?;
        }
        socket.set_read_timeout(Some(Duration::from_millis(500)))?;
        let stop = Arc::new(AtomicBool::new(false));
        let (s, a) = (Arc::clone(&stop), advert.clone());
        let thread = std::thread::Builder::new()
            .name("cadenza-ssdp".into())
            .spawn(move || run(socket, a, s))?;
        Ok(Ssdp {
            stop,
            advert,
            thread: Some(thread),
        })
    }
}

fn run(socket: UdpSocket, advert: Advert, stop: Arc<AtomicBool>) {
    announce(&advert, true);
    let mut last_announce = Instant::now();
    let mut buf = [0u8; 2048];
    while !stop.load(Ordering::Relaxed) {
        if last_announce.elapsed() > Duration::from_secs(MAX_AGE as u64 / 2) {
            announce(&advert, true);
            last_announce = Instant::now();
        }
        let Ok((n, peer)) = socket.recv_from(&mut buf) else {
            continue;
        };
        let Some(st) = parse_search(&String::from_utf8_lossy(&buf[..n])) else {
            continue;
        };
        let IpAddr::V4(peer_ip) = peer.ip() else {
            continue;
        };
        let Some(local) = local_for(peer_ip, &interfaces()) else {
            continue;
        };
        for (st, usn) in advert.matching(&st) {
            let _ = socket.send_to(advert.response(&st, &usn, local).as_bytes(), peer);
        }
    }
}

impl Drop for Ssdp {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        announce(&self.advert, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn advert() -> Advert {
        Advert {
            udn: "uuid:abc".into(),
            port: 49152,
        }
    }

    #[test]
    fn searches_are_parsed_and_answered_for_our_targets_only() {
        let msg = "M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: 2\r\nST: urn:schemas-upnp-org:device:MediaRenderer:1\r\n\r\n";
        assert_eq!(parse_search(msg).as_deref(), Some(DEVICE_TYPE));
        assert_eq!(parse_search("NOTIFY * HTTP/1.1\r\nST: x\r\n"), None);
        assert_eq!(
            parse_search("M-SEARCH * HTTP/1.1\r\nST: x\r\n"),
            None,
            "MAN required"
        );
        let a = advert();
        assert_eq!(a.matching("ssdp:all").len(), 3 + SERVICES.len());
        assert_eq!(
            a.matching("upnp:rootdevice")[0].1,
            "uuid:abc::upnp:rootdevice"
        );
        assert_eq!(a.matching("uuid:abc").len(), 1);
        assert!(a
            .matching("urn:schemas-upnp-org:device:MediaServer:1")
            .is_empty());
        let r = a.response(DEVICE_TYPE, "uuid:abc::x", Ipv4Addr::new(192, 168, 1, 5));
        assert!(r.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(r.contains("LOCATION: http://192.168.1.5:49152/upnp/device.xml\r\n"));
        assert!(r.ends_with("\r\n\r\n"));
        let n = a.notify("upnp:rootdevice", "u", false, Ipv4Addr::LOCALHOST);
        assert!(n.contains("NTS: ssdp:byebye") && !n.contains("LOCATION"));
    }

    #[test]
    fn location_uses_the_interface_on_the_peers_subnet() {
        let ifaces = [
            (Ipv4Addr::new(10, 0, 0, 2), Ipv4Addr::new(255, 0, 0, 0)),
            (
                Ipv4Addr::new(192, 168, 1, 5),
                Ipv4Addr::new(255, 255, 255, 0),
            ),
        ];
        assert_eq!(
            local_for(Ipv4Addr::new(192, 168, 1, 77), &ifaces),
            Some(Ipv4Addr::new(192, 168, 1, 5))
        );
        assert_eq!(
            local_for(Ipv4Addr::new(172, 16, 0, 1), &ifaces),
            Some(Ipv4Addr::new(10, 0, 0, 2))
        );
        assert_eq!(local_for(Ipv4Addr::new(1, 2, 3, 4), &[]), None);
    }
}
