//! The UPnP renderer end to end: a real engine (null output, real time), the renderer's
//! HTTP side on a free port, a little media server for WAV files, and this test acting as
//! the control point – SOAP actions, gapless next URI, volume, and GENA events.

use cadenza::config::{RateSetting, Settings};
use cadenza::engine::{Engine, EngineOptions, PlayState};
use cadenza::upnp::{self, xml, UpnpConfig};
use cadenza_dsp::CoefficientCache;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn wav(seconds: f64, freq: f64) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cur = std::io::Cursor::new(Vec::new());
    {
        let mut w = hound::WavWriter::new(&mut cur, spec).unwrap();
        for i in 0..(44_100.0 * seconds) as usize {
            let v =
                (8000.0 * (2.0 * std::f64::consts::PI * freq * i as f64 / 44_100.0).sin()) as i16;
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();
    }
    cur.into_inner()
}

/// A minimal HTTP/1.1 file server (GET/HEAD, single byte ranges).
fn media_server(files: Vec<(&'static str, Vec<u8>)>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let files = Arc::new(files);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let files = Arc::clone(&files);
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut stream = stream;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    let mut parts = line.split_whitespace();
                    let method = parts.next().unwrap_or("").to_string();
                    let path = parts.next().unwrap_or("").to_string();
                    let mut range = None;
                    loop {
                        let mut h = String::new();
                        if reader.read_line(&mut h).unwrap_or(0) == 0 {
                            return;
                        }
                        let h = h.trim();
                        if h.is_empty() {
                            break;
                        }
                        if let Some((k, v)) = h.split_once(':') {
                            if k.eq_ignore_ascii_case("range") {
                                range = v.trim().strip_prefix("bytes=").map(str::to_string);
                            }
                        }
                    }
                    let Some((_, data)) = files.iter().find(|(p, _)| *p == path) else {
                        let _ = stream
                            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                        continue;
                    };
                    let len = data.len();
                    let (start, end, status) =
                        match range.as_deref().and_then(|r| r.split_once('-')) {
                            Some((a, b)) => {
                                let a: usize = a.parse().unwrap_or(0);
                                let b: usize = b.parse().unwrap_or(len - 1).min(len - 1);
                                (a.min(len), b + 1, "206 Partial Content")
                            }
                            None => (0, len, "200 OK"),
                        };
                    let mut head = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: audio/wav\r\nAccept-Ranges: bytes\r\nContent-Length: {}\r\n",
                        end - start
                    );
                    if status.starts_with("206") {
                        head.push_str(&format!(
                            "Content-Range: bytes {start}-{}/{len}\r\n",
                            end - 1
                        ));
                    }
                    head.push_str("\r\n");
                    if stream.write_all(head.as_bytes()).is_err() {
                        return;
                    }
                    if method == "GET" && stream.write_all(&data[start..end]).is_err() {
                        return;
                    }
                }
            });
        }
    });
    port
}

/// Collects GENA NOTIFY bodies sent to it.
fn event_sink() -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let got = Arc::new(Mutex::new(Vec::new()));
    let g = Arc::clone(&got);
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut len = 0usize;
            let mut sid_seq = String::new();
            loop {
                let mut h = String::new();
                if reader.read_line(&mut h).unwrap_or(0) == 0 {
                    break;
                }
                let t = h.trim().to_string();
                if t.is_empty() {
                    break;
                }
                if let Some((k, v)) = t.split_once(':') {
                    match k.to_ascii_lowercase().as_str() {
                        "content-length" => len = v.trim().parse().unwrap_or(0),
                        "seq" => sid_seq = v.trim().to_string(),
                        _ => {}
                    }
                }
            }
            let mut body = vec![0u8; len];
            let _ = reader.read_exact(&mut body);
            g.lock()
                .unwrap()
                .push(format!("SEQ {sid_seq}\n{}", String::from_utf8_lossy(&body)));
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
        }
    });
    (port, got)
}

struct Cp {
    base: String,
}

impl Cp {
    fn call(&self, service: &str, action: &str, args: &[(&str, &str)]) -> Result<String, String> {
        let body: String = args
            .iter()
            .map(|(k, v)| format!("<{k}>{}</{k}>", xml::escape(v)))
            .collect();
        let st = format!("urn:schemas-upnp-org:service:{service}:1");
        let env = format!(
            r#"<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"><s:Body><u:{action} xmlns:u="{st}">{body}</u:{action}></s:Body></s:Envelope>"#
        );
        match ureq::post(&format!("{}/upnp/control/{service}", self.base))
            .set("Content-Type", "text/xml; charset=\"utf-8\"")
            .set("SOAPACTION", &format!("\"{st}#{action}\""))
            .send_string(&env)
        {
            Ok(r) => Ok(r.into_string().unwrap()),
            Err(ureq::Error::Status(500, r)) => {
                let t = r.into_string().unwrap();
                Err(xml::text(&t, "errorCode").unwrap_or_default())
            }
            Err(e) => panic!("{e}"),
        }
    }

    fn get(&self, service: &str, action: &str, out: &str) -> String {
        let r = self.call(service, action, &[("InstanceID", "0")]).unwrap();
        xml::text(&r, out).unwrap_or_default()
    }

    fn state(&self) -> String {
        self.get("AVTransport", "GetTransportInfo", "CurrentTransportState")
    }

    fn wait_state(&self, want: &str) {
        let t0 = Instant::now();
        while self.state() != want {
            assert!(
                t0.elapsed() < Duration::from_secs(10),
                "never reached {want} (at {})",
                self.state()
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

#[test]
fn a_control_point_drives_the_renderer() {
    let media = media_server(vec![
        ("/a.wav", wav(3.0, 440.0)),
        ("/b.wav", wav(2.0, 880.0)),
    ]);
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
    let cfg = UpnpConfig {
        enabled: true,
        name: "Test renderer".into(),
        port: 0,
    };
    let r = upnp::start(engine.clone(), &cfg, false).unwrap();
    let cp = Cp {
        base: format!("http://127.0.0.1:{}", r.port),
    };

    // Description and service descriptions.
    let d = ureq::get(&format!("{}/upnp/device.xml", cp.base))
        .call()
        .unwrap()
        .into_string()
        .unwrap();
    assert_eq!(
        xml::text(&d, "friendlyName").as_deref(),
        Some("Test renderer")
    );
    assert_eq!(xml::text(&d, "UDN").as_deref(), Some(r.udn.as_str()));
    for svc in ["AVTransport", "RenderingControl", "ConnectionManager"] {
        let scpd = ureq::get(&format!("{}/upnp/{svc}.xml", cp.base))
            .call()
            .unwrap()
            .into_string()
            .unwrap();
        assert!(scpd.contains("<actionList>"), "{svc}");
    }
    assert!(cp
        .get("ConnectionManager", "GetProtocolInfo", "Sink")
        .contains("http-get:*:audio/flac:*"));

    // Events: subscribe to AVTransport; the initial event arrives with SEQ 0.
    let (sink_port, events) = event_sink();
    let sub = ureq::request("SUBSCRIBE", &format!("{}/upnp/event/AVTransport", cp.base))
        .set("CALLBACK", &format!("<http://127.0.0.1:{sink_port}/ev>"))
        .set("NT", "upnp:event")
        .set("TIMEOUT", "Second-300")
        .call()
        .unwrap();
    let sid = sub.header("sid").unwrap().to_string();
    assert!(sid.starts_with("uuid:"));
    assert_eq!(sub.header("timeout"), Some("Second-300"));

    assert_eq!(cp.state(), "NO_MEDIA_PRESENT");
    // Local paths are refused.
    assert_eq!(
        cp.call(
            "AVTransport",
            "SetAVTransportURI",
            &[
                ("InstanceID", "0"),
                ("CurrentURI", "/etc/passwd"),
                ("CurrentURIMetaData", "")
            ]
        ),
        Err("716".into())
    );
    assert_eq!(
        cp.call(
            "AVTransport",
            "Play",
            &[("InstanceID", "0"), ("Speed", "1")]
        ),
        Err("701".into())
    );

    let a = format!("http://127.0.0.1:{media}/a.wav");
    let b = format!("http://127.0.0.1:{media}/b.wav");
    let meta = r#"<DIDL-Lite><item><dc:title>Tone A</dc:title><upnp:artist>Test</upnp:artist><res duration="0:00:03">x</res></item></DIDL-Lite>"#;
    cp.call(
        "AVTransport",
        "SetAVTransportURI",
        &[
            ("InstanceID", "0"),
            ("CurrentURI", &a),
            ("CurrentURIMetaData", meta),
        ],
    )
    .unwrap();
    assert_eq!(cp.state(), "STOPPED");
    cp.call(
        "AVTransport",
        "Play",
        &[("InstanceID", "0"), ("Speed", "1")],
    )
    .unwrap();
    cp.wait_state("PLAYING");
    assert_eq!(engine.status().track.as_ref().unwrap().title, "Tone A");
    let pos = cp
        .call("AVTransport", "GetPositionInfo", &[("InstanceID", "0")])
        .unwrap();
    assert_eq!(xml::text(&pos, "TrackURI").as_deref(), Some(a.as_str()));
    assert!(xml::text(&pos, "TrackMetaData").unwrap().contains("Tone A"));

    // Volume and mute.
    cp.call(
        "RenderingControl",
        "SetVolume",
        &[
            ("InstanceID", "0"),
            ("Channel", "Master"),
            ("DesiredVolume", "40"),
        ],
    )
    .unwrap();
    let t0 = Instant::now();
    while cp.get("RenderingControl", "GetVolume", "CurrentVolume") != "40" {
        assert!(t0.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(20));
    }
    cp.call(
        "RenderingControl",
        "SetMute",
        &[
            ("InstanceID", "0"),
            ("Channel", "Master"),
            ("DesiredMute", "1"),
        ],
    )
    .unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert!(engine.status().volume.muted);
    cp.call(
        "RenderingControl",
        "SetMute",
        &[
            ("InstanceID", "0"),
            ("Channel", "Master"),
            ("DesiredMute", "0"),
        ],
    )
    .unwrap();

    // Pause, resume, seek.
    cp.call("AVTransport", "Pause", &[("InstanceID", "0")])
        .unwrap();
    cp.wait_state("PAUSED_PLAYBACK");
    cp.call(
        "AVTransport",
        "Play",
        &[("InstanceID", "0"), ("Speed", "1")],
    )
    .unwrap();
    cp.wait_state("PLAYING");
    assert_eq!(
        cp.call(
            "AVTransport",
            "Seek",
            &[("InstanceID", "0"), ("Unit", "TRACK_NR"), ("Target", "1")]
        ),
        Err("710".into())
    );
    cp.call(
        "AVTransport",
        "Seek",
        &[
            ("InstanceID", "0"),
            ("Unit", "REL_TIME"),
            ("Target", "0:00:01"),
        ],
    )
    .unwrap();

    // The next URI is queued; the renderer follows the engine onto it without a gap.
    cp.call(
        "AVTransport",
        "SetNextAVTransportURI",
        &[
            ("InstanceID", "0"),
            ("NextURI", &b),
            ("NextURIMetaData", ""),
        ],
    )
    .unwrap();
    let media_info = cp
        .call("AVTransport", "GetMediaInfo", &[("InstanceID", "0")])
        .unwrap();
    assert_eq!(
        xml::text(&media_info, "NextURI").as_deref(),
        Some(b.as_str())
    );
    let t0 = Instant::now();
    loop {
        let pos = cp
            .call("AVTransport", "GetPositionInfo", &[("InstanceID", "0")])
            .unwrap();
        if xml::text(&pos, "TrackURI").as_deref() == Some(b.as_str()) {
            break;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "never moved on to the next URI"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(cp.state(), "PLAYING");
    assert_eq!(
        engine.status().index,
        Some(1),
        "the engine played on from its queue"
    );

    cp.call("AVTransport", "Stop", &[("InstanceID", "0")])
        .unwrap();
    cp.wait_state("STOPPED");
    assert_eq!(engine.status().state, PlayState::Stopped);

    // Events: the initial one, and later changes with increasing SEQ.
    let t0 = Instant::now();
    loop {
        let got = events.lock().unwrap().clone();
        if got.iter().any(|e| e.starts_with("SEQ 0"))
            && got.iter().any(|e| e.contains("PLAYING"))
            && got.len() >= 3
        {
            assert!(got.iter().all(|e| e.contains("LastChange")));
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(5), "events: {got:?}");
        std::thread::sleep(Duration::from_millis(100));
    }
    let unsub = ureq::request(
        "UNSUBSCRIBE",
        &format!("{}/upnp/event/AVTransport", cp.base),
    )
    .set("SID", &sid)
    .call();
    assert!(unsub.is_ok());
    let again = ureq::request(
        "UNSUBSCRIBE",
        &format!("{}/upnp/event/AVTransport", cp.base),
    )
    .set("SID", &sid)
    .call();
    assert!(matches!(again, Err(ureq::Error::Status(412, _))));
    drop(r);
    engine.shutdown();
}
