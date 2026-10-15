//! The RTSP side of an AirPlay 1 session: one thread per sender connection.
//!
//! OPTIONS (with the `Apple-Challenge`), ANNOUNCE (the SDP), SETUP (our UDP ports; the
//! engine starts playing the live stream), RECORD, FLUSH (pause / seek), SET_PARAMETER
//! (volume, track metadata, progress), GET_PARAMETER (volume) and TEARDOWN. A new sender
//! takes over from the one playing, like an AirPort Express does.

use super::audio::{parse_sdp, PacketDecoder, StreamFormat};
use super::rtp::Rtp;
use super::{crypto, dacp, dmap, AirplayConfig};
use crate::decode::live::{self, LiveStream};
use crate::decode::Source;
use crate::engine::{Command, Engine, Track};
use cadenza_dsp::Volume;
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{IpAddr, Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// AirPlay's volume floor: −144 means mute, otherwise −30…0.
const MUTE: f64 = -144.0;
const VOLUME_MIN: f64 = -30.0;

/// State shared by all connections.
pub struct Shared {
    pub(super) engine: Engine,
    pub(super) config: AirplayConfig,
    pub(super) hw: [u8; 6],
    pub(super) resolver: dacp::Resolver,
    /// The connection whose session is playing, and a handle to close it on takeover.
    active: Mutex<Option<(u64, TcpStream)>>,
    next_id: AtomicU64,
}

impl Shared {
    /// Shared state for `engine`.
    pub fn new(
        engine: Engine,
        config: AirplayConfig,
        hw: [u8; 6],
        resolver: dacp::Resolver,
    ) -> Arc<Self> {
        Arc::new(Self {
            engine,
            config,
            hw,
            resolver,
            active: Mutex::new(None),
            next_id: AtomicU64::new(1),
        })
    }
}

/// Accept connections until `stop` is set.
pub fn serve(listener: TcpListener, shared: Arc<Shared>, stop: Arc<AtomicBool>) {
    let _ = listener.set_nonblocking(true);
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                let shared = Arc::clone(&shared);
                let _ = std::thread::Builder::new()
                    .name("cadenza-airplay-rtsp".into())
                    .spawn(move || {
                        if let Err(e) = Conn::new(stream, shared).and_then(|mut c| c.run()) {
                            log::debug!("AirPlay connection: {e}");
                        }
                    });
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                log::warn!("AirPlay accept: {e}");
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
    // Close the playing session too.
    if let Some((_, s)) = shared.active.lock().unwrap().take() {
        let _ = s.shutdown(Shutdown::Both);
    }
}

/// One request.
#[derive(Debug, Default)]
pub struct Request {
    /// `OPTIONS`, `ANNOUNCE` …
    pub method: String,
    /// The request URI.
    pub uri: String,
    /// Headers, names in lower case.
    pub headers: HashMap<String, String>,
    /// The body.
    pub body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(String::as_str)
    }
}

/// Read one request; `None` at the end of the connection.
pub fn read_request(r: &mut impl BufRead) -> io::Result<Option<Request>> {
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        if !line.trim().is_empty() {
            break;
        }
    }
    let mut parts = line.split_whitespace();
    let mut req = Request {
        method: parts.next().unwrap_or("").to_string(),
        uri: parts.next().unwrap_or("").to_string(),
        ..Default::default()
    };
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            req.headers
                .insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let len: usize = req
        .header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if len > 16 << 20 {
        return Err(io::Error::other("request body too large"));
    }
    req.body = vec![0; len];
    r.read_exact(&mut req.body)?;
    Ok(Some(req))
}

/// A reply.
struct Reply {
    status: u16,
    reason: &'static str,
    headers: Vec<(&'static str, String)>,
    body: Vec<u8>,
}

impl Reply {
    fn ok() -> Self {
        Self::status(200, "OK")
    }

    fn status(status: u16, reason: &'static str) -> Self {
        Self {
            status,
            reason,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    fn header(mut self, k: &'static str, v: impl Into<String>) -> Self {
        self.headers.push((k, v.into()));
        self
    }
}

/// A playing session (after SETUP).
struct Session {
    live_id: String,
    stream: Arc<LiveStream>,
    rtp: Rtp,
    track: Track,
    rate: u32,
    /// Covers received (makes each cover URL new, so the UI reloads it).
    covers: u64,
}

struct Conn {
    id: u64,
    shared: Arc<Shared>,
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    local: IpAddr,
    peer: IpAddr,
    format: Option<StreamFormat>,
    session: Option<Session>,
}

impl Conn {
    fn new(stream: TcpStream, shared: Arc<Shared>) -> io::Result<Self> {
        let local = stream.local_addr()?.ip();
        let peer = stream.peer_addr()?.ip();
        let id = shared.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            id,
            shared,
            writer: stream.try_clone()?,
            reader: BufReader::new(stream),
            local,
            peer,
            format: None,
            session: None,
        })
    }

    fn run(&mut self) -> io::Result<()> {
        log::info!("AirPlay: connection from {}", self.peer);
        let result = loop {
            let req = match read_request(&mut self.reader) {
                Ok(Some(r)) => r,
                Ok(None) => break Ok(()),
                Err(e) => break Err(e),
            };
            log::debug!("AirPlay: {} {}", req.method, req.uri);
            let reply = self.handle(&req);
            if let Err(e) = self.send(&req, reply) {
                break Err(e);
            }
        };
        self.teardown();
        result
    }

    fn send(&mut self, req: &Request, reply: Reply) -> io::Result<()> {
        let mut out = format!("RTSP/1.0 {} {}\r\n", reply.status, reply.reason);
        if let Some(cseq) = req.header("cseq") {
            out += &format!("CSeq: {cseq}\r\n");
        }
        out += "Server: AirTunes/105.1\r\n";
        out += "Audio-Jack-Status: connected; type=analog\r\n";
        if let Some(ch) = req.header("apple-challenge") {
            if let Some(resp) = crypto::apple_response(ch, self.local, self.shared.hw) {
                out += &format!("Apple-Response: {resp}\r\n");
            }
        }
        for (k, v) in &reply.headers {
            out += &format!("{k}: {v}\r\n");
        }
        if !reply.body.is_empty() {
            out += &format!("Content-Length: {}\r\n", reply.body.len());
        }
        out += "\r\n";
        let mut bytes = out.into_bytes();
        bytes.extend_from_slice(&reply.body);
        self.writer.write_all(&bytes)
    }

    fn handle(&mut self, req: &Request) -> Reply {
        match req.method.as_str() {
            "OPTIONS" => Reply::ok().header(
                "Public",
                "ANNOUNCE, SETUP, RECORD, PAUSE, FLUSH, TEARDOWN, OPTIONS, GET_PARAMETER, \
                 SET_PARAMETER",
            ),
            "ANNOUNCE" => match parse_sdp(&String::from_utf8_lossy(&req.body)) {
                Ok(f) => {
                    log::info!(
                        "AirPlay: {} {}/{} from {}{}",
                        f.codec_label(),
                        f.bits,
                        f.rate,
                        self.peer,
                        if f.key.is_some() { ", encrypted" } else { "" }
                    );
                    self.format = Some(f);
                    Reply::ok()
                }
                Err(e) => {
                    log::warn!("AirPlay: ANNOUNCE refused: {e}");
                    Reply::status(415, "Unsupported Media Type")
                }
            },
            "SETUP" => self.setup(req),
            "RECORD" => Reply::ok().header("Audio-Latency", "11025"),
            "FLUSH" => {
                if let Some(s) = &self.session {
                    let rtptime = req
                        .header("rtp-info")
                        .and_then(|v| param(v, "rtptime"))
                        .and_then(|v| v.parse().ok());
                    s.rtp.flush(rtptime);
                }
                Reply::ok()
            }
            "SET_PARAMETER" => {
                self.set_parameter(req);
                Reply::ok()
            }
            "GET_PARAMETER" => {
                let body = String::from_utf8_lossy(&req.body);
                if body.lines().any(|l| l.trim() == "volume") {
                    Reply {
                        body: format!("volume: {:.6}\r\n", self.airplay_volume()).into_bytes(),
                        ..Reply::ok()
                    }
                    .header("Content-Type", "text/parameters")
                } else {
                    Reply::ok()
                }
            }
            "TEARDOWN" => {
                self.teardown();
                self.format = None;
                Reply::ok()
            }
            // AirPlay 2 probes (/info, /feedback): harmless to acknowledge.
            "GET" | "POST" => Reply::ok(),
            _ => Reply::status(501, "Not Implemented"),
        }
    }

    fn setup(&mut self, req: &Request) -> Reply {
        let Some(format) = self.format.clone() else {
            return Reply::status(455, "Method Not Valid in This State");
        };
        self.teardown();
        let transport = req.header("transport").unwrap_or("");
        let port = |name| param(transport, name).and_then(|v| v.parse::<u16>().ok());
        let decoder = match PacketDecoder::new(&format) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("AirPlay: {e}");
                return Reply::status(415, "Unsupported Media Type");
            }
        };
        let stream = Arc::new(LiveStream::new(
            format.rate,
            format.channels,
            format.bits,
            format.codec_label(),
            self.shared.config.latency_ms,
        ));
        let rtp = match Rtp::start(
            self.local,
            self.peer,
            port("control_port"),
            port("timing_port"),
            decoder,
            Arc::clone(&stream),
        ) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("AirPlay: UDP ports: {e}");
                return Reply::status(500, "Internal Server Error");
            }
        };
        let transport = format!(
            "RTP/AVP/UDP;unicast;mode=record;server_port={};control_port={};timing_port={}",
            rtp.audio_port, rtp.control_port, rtp.timing_port
        );
        // Take over from whoever was playing.
        {
            let mut active = self.shared.active.lock().unwrap();
            if let Some((_, old)) = active.take() {
                let _ = old.shutdown(Shutdown::Both);
            }
            *active = self.writer.try_clone().ok().map(|s| (self.id, s));
        }
        let live_id = format!("airplay-{}", self.id);
        if let (Some(id), Some(ar)) = (req.header("dacp-id"), req.header("active-remote")) {
            stream.set_control(Some(dacp::remote(
                id.to_string(),
                ar.to_string(),
                self.peer,
                Arc::clone(&self.shared.resolver),
            )));
        }
        live::register(&live_id, Arc::clone(&stream));
        let track = Track {
            id: Some(live_id.clone()),
            source: Source::Live {
                live: live_id.clone(),
            },
            title: "AirPlay".into(),
            artist: req
                .header("x-apple-client-name")
                .map(str::to_string)
                .unwrap_or_else(|| self.peer.to_string()),
            format: Some(format!(
                "AirPlay {} {}/{}",
                format.codec_label(),
                format.bits,
                cadenza_dsp::signal_path::rate_label(format.rate)
            )),
            ..Default::default()
        };
        self.shared.engine.play_queue(vec![track.clone()], 0, 0.0);
        self.session = Some(Session {
            live_id,
            stream,
            rtp,
            track,
            rate: format.rate,
            covers: 0,
        });
        Reply::ok()
            .header("Transport", transport)
            .header("Session", "1")
    }

    fn set_parameter(&mut self, req: &Request) {
        let ctype = req.header("content-type").unwrap_or("");
        if ctype.starts_with("text/parameters") {
            for line in String::from_utf8_lossy(&req.body).lines() {
                let Some((k, v)) = line.split_once(':') else {
                    continue;
                };
                match k.trim() {
                    "volume" => {
                        if let Ok(v) = v.trim().parse::<f64>() {
                            self.set_volume(v);
                        }
                    }
                    "progress" => self.progress(v.trim()),
                    _ => {}
                }
            }
        } else if ctype.starts_with("application/x-dmap-tagged") {
            let tags = dmap::parse(&req.body);
            if let Some(s) = &mut self.session {
                if let Some(t) = tags.title {
                    s.track.title = t;
                }
                if let Some(a) = tags.artist {
                    s.track.artist = a;
                }
                if let Some(a) = tags.album {
                    s.track.album = a;
                }
                self.shared
                    .engine
                    .send(Command::SetTrackInfo(Box::new(s.track.clone())));
            }
        } else if ctype.starts_with("image/") {
            if let Some(s) = &mut self.session {
                let mime = ctype.split(';').next().unwrap_or(ctype).trim();
                if mime == "image/none" || req.body.is_empty() {
                    s.stream.set_cover(None);
                    s.track.cover = None;
                } else {
                    s.stream
                        .set_cover(Some((mime.to_string(), req.body.clone())));
                    s.covers += 1;
                    s.track.cover = Some(format!("/api/live/{}/cover?v={}", s.live_id, s.covers));
                }
                self.shared
                    .engine
                    .send(Command::SetTrackInfo(Box::new(s.track.clone())));
            }
        }
    }

    /// `progress: start/current/end` in RTP time.
    fn progress(&mut self, v: &str) {
        let Some(s) = &mut self.session else { return };
        let p: Vec<u32> = v.split('/').filter_map(|x| x.trim().parse().ok()).collect();
        if p.len() != 3 {
            return;
        }
        let rate = s.rate.max(1) as f64;
        let duration = p[2].wrapping_sub(p[0]) as f64 / rate;
        let position = p[1].wrapping_sub(p[0]) as f64 / rate;
        s.stream.mark_position(position.min(duration));
        s.track.duration = Some(duration);
        self.shared
            .engine
            .send(Command::SetTrackInfo(Box::new(s.track.clone())));
    }

    fn set_volume(&self, v: f64) {
        if !self.shared.config.volume {
            return;
        }
        let vol = if v <= MUTE + 1.0 {
            Volume {
                gain_db: self.shared.engine.status().volume.gain_db,
                muted: true,
            }
        } else {
            Volume {
                gain_db: v.clamp(VOLUME_MIN, 0.0),
                muted: false,
            }
        };
        self.shared.engine.send(Command::SetVolume(vol));
    }

    fn airplay_volume(&self) -> f64 {
        let v = self.shared.engine.status().volume;
        if v.muted {
            MUTE
        } else {
            v.gain_db.clamp(VOLUME_MIN, 0.0)
        }
    }

    /// End the session (if any): the engine plays out what is buffered, then stops.
    fn teardown(&mut self) {
        let Some(s) = self.session.take() else { return };
        log::info!("AirPlay: session from {} ended", self.peer);
        drop(s.rtp);
        live::unregister(&s.live_id);
        let mut active = self.shared.active.lock().unwrap();
        if active.as_ref().is_some_and(|(id, _)| *id == self.id) {
            *active = None;
        }
    }
}

/// `name=value` from a `;`-separated header.
fn param<'a>(header: &'a str, name: &str) -> Option<&'a str> {
    header.split(';').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k.trim() == name).then(|| v.trim())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_parse() {
        let raw = b"ANNOUNCE rtsp://10.0.0.2/123 RTSP/1.0\r\nCSeq: 3\r\nContent-Type: application/sdp\r\nContent-Length: 5\r\n\r\nv=0\r\nOPTIONS * RTSP/1.0\r\nCSeq: 4\r\n\r\n";
        let mut r = BufReader::new(&raw[..]);
        let a = read_request(&mut r).unwrap().unwrap();
        assert_eq!(
            (a.method.as_str(), a.uri.as_str()),
            ("ANNOUNCE", "rtsp://10.0.0.2/123")
        );
        assert_eq!(a.header("cseq"), Some("3"));
        assert_eq!(a.body, b"v=0\r\n");
        let b = read_request(&mut r).unwrap().unwrap();
        assert_eq!(b.method, "OPTIONS");
        assert!(read_request(&mut r).unwrap().is_none());
        assert_eq!(
            param(
                "RTP/AVP/UDP;unicast;control_port=6001;timing_port=6002",
                "timing_port"
            ),
            Some("6002")
        );
        assert_eq!(param("seq=1;rtptime=99", "rtptime"), Some("99"));
    }
}
