//! The three UDP ports of an AirPlay 1 session.
//!
//! - **audio**: RTP packets (payload type 0x60), one ALAC frame each.
//! - **control**: sync packets from the sender (ignored: we play as soon as the buffer is
//!   full rather than at the sender's clock time), our resend requests for lost packets
//!   and the sender's resent packets.
//! - **timing**: NTP-style time requests and replies.
//!
//! Packets go through a small reorder buffer; a gap is asked for again once and filled
//! with silence if it does not arrive in time.

use super::audio::PacketDecoder;
use crate::decode::live::LiveStream;
use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How long a gap may stay open before it is filled with silence.
const GAP_WAIT: Duration = Duration::from_millis(250);
/// Packets held behind a gap before it is given up anyway.
const GAP_MAX: usize = 128;

enum Msg {
    Packet { seq: u16, ts: u32, payload: Vec<u8> },
    Flush(Option<u32>),
}

/// A header-checked RTP audio packet: `(seq, timestamp, payload)`.
pub fn parse_rtp(pkt: &[u8]) -> Option<(u16, u32, &[u8])> {
    if pkt.len() < 12 || pkt[0] >> 6 != 2 || pkt[1] & 0x7f != 0x60 {
        return None;
    }
    let seq = u16::from_be_bytes([pkt[2], pkt[3]]);
    let ts = u32::from_be_bytes([pkt[4], pkt[5], pkt[6], pkt[7]]);
    Some((seq, ts, &pkt[12..]))
}

/// The running session ports.
pub struct Rtp {
    /// Our audio port.
    pub audio_port: u16,
    /// Our control port.
    pub control_port: u16,
    /// Our timing port.
    pub timing_port: u16,
    tx: Sender<Msg>,
    stop: Arc<AtomicBool>,
}

impl Drop for Rtp {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn bind(family: IpAddr) -> io::Result<UdpSocket> {
    let s = match family {
        IpAddr::V4(_) => UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?,
        IpAddr::V6(_) => UdpSocket::bind((Ipv6Addr::UNSPECIFIED, 0))?,
    };
    s.set_read_timeout(Some(Duration::from_millis(100)))?;
    Ok(s)
}

/// NTP time now (seconds since 1900 in the high word).
fn ntp_now() -> u64 {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs() + 2_208_988_800;
    let frac = ((d.subsec_nanos() as u64) << 32) / 1_000_000_000;
    (secs << 32) | frac
}

impl Rtp {
    /// Bind the ports and start receiving into `stream`. `peer` is the sender's address;
    /// `client_control` / `client_timing` its ports from the SETUP request.
    pub fn start(
        local: IpAddr,
        peer: IpAddr,
        client_control: Option<u16>,
        client_timing: Option<u16>,
        decoder: PacketDecoder,
        stream: Arc<LiveStream>,
    ) -> io::Result<Rtp> {
        let audio = bind(local)?;
        let control = bind(local)?;
        let timing = bind(local)?;
        let ports = (
            audio.local_addr()?.port(),
            control.local_addr()?.port(),
            timing.local_addr()?.port(),
        );
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let resend_to = client_control.map(|p| SocketAddr::new(peer, p));
        let control_out = control.try_clone()?;
        // Audio.
        {
            let (tx, stop) = (tx.clone(), Arc::clone(&stop));
            spawn("cadenza-airplay-audio", move || {
                let mut buf = [0u8; 2048];
                while !stop.load(Ordering::Relaxed) {
                    if let Ok((n, from)) = audio.recv_from(&mut buf) {
                        if from.ip() != peer {
                            continue;
                        }
                        if let Some((seq, ts, payload)) = parse_rtp(&buf[..n]) {
                            let payload = payload.to_vec();
                            if tx.send(Msg::Packet { seq, ts, payload }).is_err() {
                                break;
                            }
                        }
                    }
                }
            })?;
        }
        // Control: resent packets (0x56) carry a whole RTP packet after 4 bytes.
        {
            let (tx, stop) = (tx.clone(), Arc::clone(&stop));
            spawn("cadenza-airplay-control", move || {
                let mut buf = [0u8; 2048];
                while !stop.load(Ordering::Relaxed) {
                    if let Ok((n, from)) = control.recv_from(&mut buf) {
                        if from.ip() != peer || n < 4 || buf[1] & 0x7f != 0x56 {
                            continue;
                        }
                        let mut inner = buf[4..n].to_vec();
                        // The resent packet's own type byte is the audio one.
                        if inner.len() >= 2 {
                            inner[1] = 0x60;
                        }
                        if let Some((seq, ts, payload)) = parse_rtp(&inner) {
                            let payload = payload.to_vec();
                            if tx.send(Msg::Packet { seq, ts, payload }).is_err() {
                                break;
                            }
                        }
                    }
                }
            })?;
        }
        // Timing: answer requests, and ask the sender every 3 s like other receivers do.
        {
            let stop = Arc::clone(&stop);
            let to = client_timing.map(|p| SocketAddr::new(peer, p));
            spawn("cadenza-airplay-timing", move || {
                let mut buf = [0u8; 128];
                let mut last = Instant::now() - Duration::from_secs(10);
                while !stop.load(Ordering::Relaxed) {
                    if let (Some(to), true) = (to, last.elapsed() >= Duration::from_secs(3)) {
                        last = Instant::now();
                        let mut req = [0u8; 32];
                        req[..4].copy_from_slice(&[0x80, 0xd2, 0x00, 0x07]);
                        req[24..].copy_from_slice(&ntp_now().to_be_bytes());
                        let _ = timing.send_to(&req, to);
                    }
                    if let Ok((n, from)) = timing.recv_from(&mut buf) {
                        if n >= 32 && buf[1] & 0x7f == 0x52 && from.ip() == peer {
                            let mut reply = [0u8; 32];
                            reply[..4].copy_from_slice(&[0x80, 0xd3, 0x00, 0x07]);
                            reply[8..16].copy_from_slice(&buf[24..32]);
                            let now = ntp_now().to_be_bytes();
                            reply[16..24].copy_from_slice(&now);
                            reply[24..32].copy_from_slice(&now);
                            let _ = timing.send_to(&reply, from);
                        }
                    }
                }
            })?;
        }
        // The reorder buffer, decoder and the live stream.
        {
            let stop = Arc::clone(&stop);
            spawn("cadenza-airplay-jitter", move || {
                let mut r = Reorder::new(decoder, stream, move |first, count| {
                    if let Some(to) = resend_to {
                        let mut req = [0x80, 0xd5, 0, 1, 0, 0, 0, 0];
                        req[4..6].copy_from_slice(&first.to_be_bytes());
                        req[6..8].copy_from_slice(&count.to_be_bytes());
                        let _ = control_out.send_to(&req, to);
                    }
                });
                r.run(rx, &stop);
            })?;
        }
        Ok(Rtp {
            audio_port: ports.0,
            control_port: ports.1,
            timing_port: ports.2,
            tx,
            stop,
        })
    }

    /// Drop what is buffered (pause / seek at the sender). Packets older than `rtptime`
    /// that are still in flight are ignored.
    pub fn flush(&self, rtptime: Option<u32>) {
        let _ = self.tx.send(Msg::Flush(rtptime));
    }
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) -> io::Result<()> {
    std::thread::Builder::new()
        .name(name.into())
        .spawn(f)
        .map(|_| ())
}

/// Puts packets in sequence order and hands them to the stream.
pub struct Reorder<F: FnMut(u16, u16)> {
    decoder: PacketDecoder,
    stream: Arc<LiveStream>,
    request: F,
    next: Option<u16>,
    pending: HashMap<u16, Vec<u8>>,
    gap_since: Option<Instant>,
    flush_before: Option<u32>,
    /// Packets filled with silence because they never came.
    pub lost: u64,
}

impl<F: FnMut(u16, u16)> Reorder<F> {
    /// A reorder buffer; `request(first, count)` asks the sender to resend.
    pub fn new(decoder: PacketDecoder, stream: Arc<LiveStream>, request: F) -> Self {
        Self {
            decoder,
            stream,
            request,
            next: None,
            pending: HashMap::new(),
            gap_since: None,
            flush_before: None,
            lost: 0,
        }
    }

    fn run(&mut self, rx: Receiver<Msg>, stop: &AtomicBool) {
        while !stop.load(Ordering::Relaxed) && !self.stream.is_closed() {
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(Msg::Packet { seq, ts, payload }) => self.packet(seq, ts, payload),
                Ok(Msg::Flush(t)) => self.flush(t),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            self.check_gap(Instant::now());
        }
    }

    /// Forget everything (sender paused or seeked).
    pub fn flush(&mut self, rtptime: Option<u32>) {
        self.pending.clear();
        self.next = None;
        self.gap_since = None;
        self.flush_before = rtptime;
        self.stream.flush();
    }

    /// One packet in.
    pub fn packet(&mut self, seq: u16, ts: u32, payload: Vec<u8>) {
        if let Some(f) = self.flush_before {
            if (ts.wrapping_sub(f) as i32) < 0 {
                return;
            }
            self.flush_before = None;
        }
        let next = *self.next.get_or_insert(seq);
        let ahead = seq.wrapping_sub(next) as i16;
        if ahead < 0 {
            return; // late or duplicate
        }
        if ahead as usize > 4 * GAP_MAX {
            // A jump (sender restarted its sequence): start over from here.
            self.pending.clear();
            self.next = Some(seq);
            self.gap_since = None;
        }
        self.pending.insert(seq, payload);
        self.deliver();
        if !self.pending.is_empty() && self.gap_since.is_none() {
            self.gap_since = Some(Instant::now());
            let next = self.next.unwrap_or(seq);
            let count = self
                .pending
                .keys()
                .map(|s| s.wrapping_sub(next))
                .min()
                .unwrap_or(1);
            (self.request)(next, count);
        }
    }

    fn deliver(&mut self) {
        while let Some(next) = self.next {
            let Some(p) = self.pending.remove(&next) else {
                break;
            };
            let pcm = self.decoder.decode(&p);
            self.stream.push(pcm);
            self.next = Some(next.wrapping_add(1));
            self.gap_since = None;
        }
    }

    /// Give up on a gap that has been open too long: silence in its place.
    pub fn check_gap(&mut self, now: Instant) {
        while let (Some(since), Some(next)) = (self.gap_since, self.next) {
            if now.duration_since(since) < GAP_WAIT && self.pending.len() < GAP_MAX {
                break;
            }
            self.stream.push(self.decoder.silence());
            self.lost += 1;
            self.next = Some(next.wrapping_add(1));
            self.gap_since = None;
            self.deliver();
            if !self.pending.is_empty() {
                // The next gap starts its clock from the first one's, so a long run of
                // losses is bridged without waiting GAP_WAIT for each packet.
                self.gap_since = Some(since);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::audio::{parse_sdp, tests::SDP_ALAC};
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn l16_packet(v: i16) -> Vec<u8> {
        v.to_be_bytes().repeat(2)
    }

    fn drain(s: &LiveStream) -> Vec<f64> {
        let mut out = Vec::new();
        while s.buffered_frames() > 0 {
            out.extend(s.pull().unwrap());
        }
        out
    }

    #[test]
    fn reorders_requests_and_bridges_gaps() {
        let f = parse_sdp("a=rtpmap:96 L16/44100/2\n").unwrap();
        let stream = Arc::new(LiveStream::new(44100, 2, 16, "PCM", 0));
        let asked = Rc::new(RefCell::new(Vec::new()));
        let a2 = Rc::clone(&asked);
        let mut r = Reorder::new(PacketDecoder::new(&f).unwrap(), Arc::clone(&stream), {
            move |first, n| a2.borrow_mut().push((first, n))
        });
        let v = |x: i16| x as f64 / 32768.0;
        // 65534, 65535, 1 (0 missing, across the wrap), then 0 arrives late-but-in-time.
        r.packet(65534, 0, l16_packet(1));
        r.packet(65535, 352, l16_packet(2));
        r.packet(1, 1056, l16_packet(4));
        assert_eq!(*asked.borrow(), vec![(0, 1)]);
        r.packet(0, 704, l16_packet(3));
        assert_eq!(drain(&stream), [1, 1, 2, 2, 3, 3, 4, 4].map(v));
        // Duplicates and late packets are dropped.
        r.packet(1, 1056, l16_packet(9));
        assert_eq!(stream.buffered_frames(), 0);
        // 3 never comes: after GAP_WAIT it becomes one packet of silence (352 frames).
        r.packet(2, 1408, l16_packet(5));
        r.packet(4, 2112, l16_packet(6));
        r.check_gap(Instant::now());
        assert_eq!(drain(&stream), vec![v(5), v(5)]);
        r.check_gap(Instant::now() + GAP_WAIT);
        let out = drain(&stream);
        assert_eq!(out.len(), 352 * 2 + 2);
        assert_eq!(&out[704..], &[v(6), v(6)]);
        assert_eq!(r.lost, 1);
        // After a flush, packets from before the flush point are ignored.
        r.flush(Some(10_000));
        r.packet(5, 2464, l16_packet(7));
        assert_eq!(stream.buffered_frames(), 0);
        r.packet(40, 10_000, l16_packet(8));
        assert_eq!(drain(&stream), vec![v(8), v(8)]);
    }

    #[test]
    fn rtp_header_checks() {
        let mut p = vec![0x80, 0x60, 0x12, 0x34, 0, 0, 1, 0, 0, 0, 0, 0, 9, 9];
        assert_eq!(parse_rtp(&p), Some((0x1234, 256, &[9u8, 9][..])));
        p[1] = 0xe0; // marker bit set on the first packet
        assert!(parse_rtp(&p).is_some());
        p[1] = 0x54;
        assert!(parse_rtp(&p).is_none());
        assert!(parse_rtp(&p[..8]).is_none());
        let _ = parse_sdp(SDP_ALAC).unwrap();
    }
}
