//! A scripted AirPlay 1 sender against a real engine (null output): the whole session
//! as an iPhone runs it, encrypted ALAC included.

use super::audio::tests::{alac_frame, SDP_ALAC};
use super::crypto::tests::{encrypt, wrap_key};
use super::dmap::tests::item;
use crate::config::{RateSetting, Settings};
use crate::decode::Source;
use crate::engine::{Engine, EngineOptions, PlayState};
use cadenza_dsp::CoefficientCache;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, UdpSocket};
use std::time::{Duration, Instant};

struct Sender {
    tcp: BufReader<TcpStream>,
    cseq: u32,
}

struct Resp {
    status: u16,
    headers: HashMap<String, String>,
    body: String,
}

impl Sender {
    fn connect(port: u16) -> Self {
        let tcp = TcpStream::connect(("127.0.0.1", port)).unwrap();
        tcp.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        Self {
            tcp: BufReader::new(tcp),
            cseq: 0,
        }
    }

    fn call(&mut self, method: &str, headers: &[(&str, &str)], body: &[u8]) -> Resp {
        self.cseq += 1;
        let mut req = format!(
            "{method} rtsp://127.0.0.1/4242 RTSP/1.0\r\nCSeq: {}\r\n",
            self.cseq
        );
        for (k, v) in headers {
            req += &format!("{k}: {v}\r\n");
        }
        if !body.is_empty() {
            req += &format!("Content-Length: {}\r\n", body.len());
        }
        req += "\r\n";
        let mut bytes = req.into_bytes();
        bytes.extend_from_slice(body);
        self.tcp.get_mut().write_all(&bytes).unwrap();
        let mut line = String::new();
        self.tcp.read_line(&mut line).unwrap();
        let status = line.split_whitespace().nth(1).unwrap().parse().unwrap();
        let mut headers = HashMap::new();
        loop {
            line.clear();
            self.tcp.read_line(&mut line).unwrap();
            if line.trim().is_empty() {
                break;
            }
            let (k, v) = line.split_once(':').unwrap();
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
        let len = headers
            .get("content-length")
            .map(|v| v.parse().unwrap())
            .unwrap_or(0);
        let mut body = vec![0; len];
        self.tcp.read_exact(&mut body).unwrap();
        assert_eq!(headers["cseq"], self.cseq.to_string());
        Resp {
            status,
            headers,
            body: String::from_utf8(body).unwrap(),
        }
    }
}

fn rtp(seq: u16, ts: u32, payload: &[u8], ptype: u8) -> Vec<u8> {
    let mut p = vec![0x80, ptype];
    p.extend_from_slice(&seq.to_be_bytes());
    p.extend_from_slice(&ts.to_be_bytes());
    p.extend_from_slice(&[0, 0, 0, 1]);
    p.extend_from_slice(payload);
    p
}

fn wait_for(what: &str, timeout: Duration, mut f: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < timeout, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn an_airplay_sender_plays_into_the_engine() {
    let mut s = Settings::default();
    s.output.backend = "null".into();
    s.output.rate = RateSetting::Auto;
    s.dsp.bit_perfect = true;
    let engine = Engine::start(
        s,
        EngineOptions {
            cache: CoefficientCache::disabled(),
            realtime: true,
            persist: None,
        },
    );
    let cfg = super::AirplayConfig {
        enabled: true,
        name: "Test room".into(),
        port: 0,
        latency_ms: 200,
        volume: true,
    };
    let rx = super::start(engine.clone(), &cfg, false).unwrap();
    assert!(rx.instance.ends_with("@Test room"));
    let mut snd = Sender::connect(rx.port);

    // OPTIONS with a challenge.
    let r = snd.call(
        "OPTIONS",
        &[("Apple-Challenge", "BwcHBwcHBwcHBwcHBwcHBw")],
        b"",
    );
    assert_eq!(r.status, 200);
    assert!(r.headers["public"].contains("ANNOUNCE"));
    assert!(
        super::crypto::b64_decode(&r.headers["apple-response"])
            .unwrap()
            .len()
            == 256
    );

    // ANNOUNCE an encrypted ALAC stream.
    let key = [0x21u8; 16];
    let iv = [0x42u8; 16];
    let sdp = format!(
        "{SDP_ALAC}a=rsaaeskey:{}\r\na=aesiv:QkJCQkJCQkJCQkJCQkJCQg\r\n",
        wrap_key(key)
    );
    let r = snd.call(
        "ANNOUNCE",
        &[("Content-Type", "application/sdp")],
        sdp.as_bytes(),
    );
    assert_eq!(r.status, 200);

    // SETUP with our control and timing ports.
    let control = UdpSocket::bind("127.0.0.1:0").unwrap();
    let timing = UdpSocket::bind("127.0.0.1:0").unwrap();
    control
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    timing
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let transport = format!(
        "RTP/AVP/UDP;unicast;interleaved=0-1;mode=record;control_port={};timing_port={}",
        control.local_addr().unwrap().port(),
        timing.local_addr().unwrap().port()
    );
    let r = snd.call(
        "SETUP",
        &[
            ("Transport", &transport),
            ("X-Apple-Client-Name", "Test iPhone"),
        ],
        b"",
    );
    assert_eq!(r.status, 200);
    let port = |name: &str| -> u16 {
        r.headers["transport"]
            .split(';')
            .find_map(|kv| kv.strip_prefix(&format!("{name}=")))
            .unwrap()
            .parse()
            .unwrap()
    };
    let (server_audio, server_control, server_timing) = (
        port("server_port"),
        port("control_port"),
        port("timing_port"),
    );
    assert_eq!(
        snd.call("RECORD", &[("RTP-Info", "seq=0;rtptime=0")], b"")
            .status,
        200
    );

    // The receiver asks for the time on our timing port.
    let mut buf = [0u8; 64];
    let (n, _) = timing.recv_from(&mut buf).unwrap();
    assert_eq!((n, buf[1]), (32, 0xd2));
    // And answers our requests.
    let mut req = [0u8; 32];
    req[..2].copy_from_slice(&[0x80, 0xd2]);
    req[24..].copy_from_slice(&[9; 8]);
    timing.send_to(&req, ("127.0.0.1", server_timing)).unwrap();
    let (n, _) = timing.recv_from(&mut buf).unwrap();
    assert_eq!((n, buf[1]), (32, 0xd3));
    assert_eq!(&buf[8..16], &[9; 8]);

    // About 1.2 s of a 1 kHz tone, paced in real time; packet 5 is "lost".
    let audio = UdpSocket::bind("127.0.0.1:0").unwrap();
    let packet = |seq: u16| {
        let samples: Vec<i16> = (0..352 * 2)
            .map(|i| {
                let n = seq as f64 * 352.0 + (i / 2) as f64;
                (8000.0 * (2.0 * std::f64::consts::PI * 1000.0 * n / 44100.0).sin()) as i16
            })
            .collect();
        let mut p = alac_frame(&samples, 352);
        encrypt(key, iv, &mut p);
        p
    };
    let t0 = Instant::now();
    for seq in 0..150u16 {
        if seq != 5 {
            let ptype = if seq == 0 { 0xe0 } else { 0x60 };
            audio
                .send_to(
                    &rtp(seq, seq as u32 * 352, &packet(seq), ptype),
                    ("127.0.0.1", server_audio),
                )
                .unwrap();
        }
        if seq == 7 {
            // The resend request for 5 arrives on our control port; answer it.
            let (n, _) = control.recv_from(&mut buf).unwrap();
            assert_eq!(n, 8);
            assert_eq!(buf[1], 0xd5);
            assert_eq!(u16::from_be_bytes([buf[4], buf[5]]), 5);
            assert_eq!(u16::from_be_bytes([buf[6], buf[7]]), 1);
            let mut resent = vec![0x80, 0xd6, 0, 1];
            resent.extend(rtp(5, 5 * 352, &packet(5), 0x60));
            control
                .send_to(&resent, ("127.0.0.1", server_control))
                .unwrap();
        }
        let due = Duration::from_micros(seq as u64 * 352 * 1_000_000 / 44100);
        if let Some(wait) = due.checked_sub(t0.elapsed()) {
            std::thread::sleep(wait);
        }
    }

    let status = engine.status();
    assert_eq!(status.state, PlayState::Playing);
    let track = status.track.clone().unwrap();
    assert!(matches!(&track.source, Source::Live { live } if live.starts_with("airplay-")));
    assert_eq!(track.artist, "Test iPhone");
    assert_eq!(status.source_format.as_deref(), Some("ALAC 16/44.1"));
    assert!(status.output_frames > 22_050, "{}", status.output_frames);
    let Source::Live { live } = &track.source else {
        unreachable!()
    };
    // Paced in real time behind a 200 ms buffer: nothing ran dry, nothing was dropped.
    assert_eq!(
        crate::decode::live::lookup(live).unwrap().counters(),
        (0, 0)
    );

    // Volume, metadata, progress.
    let r = snd.call(
        "SET_PARAMETER",
        &[("Content-Type", "text/parameters")],
        b"volume: -15.000000\r\n",
    );
    assert_eq!(r.status, 200);
    let mut meta = item(b"minm", b"So What");
    meta.extend(item(b"asar", b"Miles Davis"));
    meta.extend(item(b"asal", b"Kind of Blue"));
    snd.call(
        "SET_PARAMETER",
        &[("Content-Type", "application/x-dmap-tagged")],
        &item(b"mlit", &meta),
    );
    snd.call(
        "SET_PARAMETER",
        &[("Content-Type", "text/parameters")],
        format!(
            "progress: 1000/{}/{}\r\n",
            1000 + 44100 * 30,
            1000 + 44100 * 562
        )
        .as_bytes(),
    );
    wait_for("metadata", Duration::from_secs(3), || {
        let s = engine.status();
        let t = s.track.as_ref().unwrap();
        t.title == "So What"
            && t.album == "Kind of Blue"
            && s.duration == Some(562.0)
            && (s.position - 30.0).abs() < 1.0
            && s.volume.gain_db == -15.0
    });
    let r = snd.call(
        "GET_PARAMETER",
        &[("Content-Type", "text/parameters")],
        b"volume\r\n",
    );
    assert_eq!(r.body, "volume: -15.000000\r\n");
    snd.call(
        "SET_PARAMETER",
        &[("Content-Type", "text/parameters")],
        b"volume: -144.000000\r\n",
    );
    wait_for("mute", Duration::from_secs(3), || {
        engine.status().volume.muted
    });

    // FLUSH (pause on the phone) keeps the session; a second sender then takes over.
    assert_eq!(
        snd.call("FLUSH", &[("RTP-Info", "seq=150;rtptime=52800")], b"")
            .status,
        200
    );
    let mut other = Sender::connect(rx.port);
    other.call(
        "ANNOUNCE",
        &[("Content-Type", "application/sdp")],
        SDP_ALAC.as_bytes(),
    );
    let r = other.call(
        "SETUP",
        &[("Transport", "RTP/AVP/UDP;unicast;mode=record")],
        b"",
    );
    assert_eq!(r.status, 200);
    // The first sender's connection is closed.
    let mut rest = Vec::new();
    let _ = snd.tcp.read_to_end(&mut rest);
    wait_for("takeover", Duration::from_secs(3), || {
        engine
            .status()
            .track
            .as_ref()
            .is_some_and(|t| t.title == "AirPlay" && t.artist == "127.0.0.1")
    });

    // TEARDOWN: the engine plays out and stops.
    let runs = engine.status().runs_finished;
    assert_eq!(other.call("TEARDOWN", &[], b"").status, 200);
    let st = engine.wait_until_finished(runs, Duration::from_secs(10));
    assert!(st.runs_finished > runs);
    assert_eq!(st.state, PlayState::Stopped);
    drop(rx);
    engine.shutdown();
}
