//! The PipeWire system input (Linux).
//!
//! * `sink`: a node with `media.class = Audio/Sink` named after the config ("Cadenza"):
//!   it shows up as an output device in every app and in the desktop's sound settings.
//!   What apps play to it arrives here in 32-bit float at the graph rate.
//! * `app`: a capture stream linked to one application's playback stream (matched by its
//!   application name, process binary or node name), before the mixer. When the app
//!   closes its stream and opens a new one, the capture follows.
//!
//! The rate is the graph's: for bit-perfect input let the graph switch to the source rate
//! (`default.clock.allowed-rates = [ 44100 48000 88200 96000 176400 192000 ]` in
//! pipewire.conf), and keep the app's own volume at 100 %.

use super::{Feed, SystemConfig, SystemMode};
use pipewire as pw;
use pw::spa;
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::pod::Pod;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

/// A running PipeWire capture; dropping it stops the loop.
pub struct Capture {
    quit: Option<pw::channel::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Capture {
    /// Connect to PipeWire and start capturing into `feed`.
    pub fn start(cfg: &SystemConfig, feed: Arc<Feed>) -> Result<Self, String> {
        let (quit_tx, quit_rx) = pw::channel::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        let cfg = cfg.clone();
        let thread = std::thread::Builder::new()
            .name("cadenza-pipewire".into())
            .spawn(move || {
                if let Err(e) = run(&cfg, feed, quit_rx, &ready_tx) {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| e.to_string())?;
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => Ok(Self {
                quit: Some(quit_tx),
                thread: Some(thread),
            }),
            Ok(Err(e)) => Err(format!("PipeWire: {e}")),
            Err(_) => Err("PipeWire: no answer from the server".into()),
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if let Some(q) = self.quit.take() {
            let _ = q.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The format we take: 32-bit float, any rate and channel count (the graph's).
fn format_param() -> Vec<u8> {
    let mut info = spa::param::audio::AudioInfoRaw::new();
    info.set_format(spa::param::audio::AudioFormat::F32LE);
    let obj = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: info.into(),
    };
    spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .expect("serialising a format pod")
    .0
    .into_inner()
}

/// Does a playback stream's properties name the app we want?
pub fn matches_app(want: &str, props: &[(&str, &str)]) -> bool {
    let want = want.trim().to_ascii_lowercase();
    props.iter().any(|(k, v)| {
        matches!(
            *k,
            "application.name" | "application.process.binary" | "node.name"
        ) && v.to_ascii_lowercase() == want
    })
}

fn run(
    cfg: &SystemConfig,
    feed: Arc<Feed>,
    quit: pw::channel::Receiver<()>,
    ready: &mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    pw::init();
    let e = |e: pw::Error| e.to_string();
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(e)?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(e)?;
    let core = context.connect_rc(None).map_err(e)?;
    let _quit = quit.attach(mainloop.loop_(), {
        let ml = mainloop.clone();
        move |_| ml.quit()
    });

    match cfg.mode {
        SystemMode::Sink => {
            let mut props = pw::properties::PropertiesBox::new();
            props.insert(*pw::keys::MEDIA_TYPE, "Audio");
            props.insert(*pw::keys::MEDIA_CLASS, "Audio/Sink");
            props.insert(*pw::keys::NODE_NAME, node_name(&cfg.name));
            props.insert(*pw::keys::NODE_DESCRIPTION, cfg.name.as_str());
            props.insert("audio.channels", "2");
            props.insert("audio.position", "FL,FR");
            let _sink = Capturing::connect(&core, props, Arc::clone(&feed), None).map_err(e)?;
            let _ = ready.send(Ok(()));
            mainloop.run();
        }
        SystemMode::App => {
            // Watch the registry for playback streams, read each one's properties (the
            // registry's announcement carries only a few) and capture the app's with a
            // stream of its own, dropped when the app's stream ends.
            let current: Rc<RefCell<Option<(u32, Capturing)>>> = Rc::new(RefCell::new(None));
            type Watched = HashMap<u32, (pw::node::Node, pw::node::NodeListener)>;
            let watched: Rc<RefCell<Watched>> = Rc::new(RefCell::new(HashMap::new()));
            let registry = core.get_registry_rc().map_err(e)?;
            let want = cfg.app.clone();
            let reg = registry.clone();
            let (watched_add, watched_remove) = (Rc::clone(&watched), Rc::clone(&watched));
            let (current_add, current_remove) = (Rc::clone(&current), Rc::clone(&current));
            let core_add = core.clone();
            let _registry_listener = registry
                .add_listener_local()
                .global(move |g| {
                    if g.type_ != pw::types::ObjectType::Node
                        || g.props.and_then(|p| p.get("media.class")) != Some("Stream/Output/Audio")
                    {
                        return;
                    }
                    let Ok(node) = reg.bind::<pw::node::Node, _>(g) else {
                        return;
                    };
                    let id = g.id;
                    let want = want.clone();
                    let current = Rc::clone(&current_add);
                    let core = core_add.clone();
                    let feed = Arc::clone(&feed);
                    let listener = node
                        .add_listener_local()
                        .info(move |info| {
                            if current.borrow().is_some() {
                                return;
                            }
                            let Some(p) = info.props() else { return };
                            let kv: Vec<(&str, &str)> = p.iter().collect();
                            if !matches_app(&want, &kv) {
                                return;
                            }
                            log::info!("system input: capturing {want:?} (node {id})");
                            let mut props = pw::properties::PropertiesBox::new();
                            props.insert(*pw::keys::MEDIA_TYPE, "Audio");
                            props.insert(*pw::keys::MEDIA_CATEGORY, "Capture");
                            props.insert(*pw::keys::MEDIA_ROLE, "Music");
                            props.insert(*pw::keys::NODE_NAME, "cadenza-app-capture");
                            // Never fall back to the default source when the app's
                            // stream goes away.
                            props.insert("node.dont-reconnect", "true");
                            match Capturing::connect(&core, props, Arc::clone(&feed), Some(id)) {
                                Ok(c) => *current.borrow_mut() = Some((id, c)),
                                Err(err) => {
                                    log::warn!("system input: capturing {want:?}: {err}")
                                }
                            }
                        })
                        .register();
                    watched_add.borrow_mut().insert(id, (node, listener));
                })
                .global_remove(move |id| {
                    watched_remove.borrow_mut().remove(&id);
                    let mut cur = current_remove.borrow_mut();
                    if cur.as_ref().is_some_and(|(t, _)| *t == id) {
                        log::info!("system input: the app's stream ended");
                        *cur = None;
                    }
                })
                .register();
            let _ = ready.send(Ok(()));
            mainloop.run();
            current.borrow_mut().take();
        }
    }
    Ok(())
}

/// One capture stream and its listener.
struct Capturing {
    stream: pw::stream::StreamRc,
    _listener: pw::stream::StreamListener<u32>,
}

impl Capturing {
    /// A stream with `props` that delivers into `feed`, linked to node `target` (or, with
    /// none, waiting to be linked to: a sink).
    fn connect(
        core: &pw::core::CoreRc,
        props: pw::properties::PropertiesBox,
        feed: Arc<Feed>,
        target: Option<u32>,
    ) -> Result<Self, pw::Error> {
        let stream = pw::stream::StreamRc::new(core.clone(), "cadenza-system", props)?;
        let format_feed = Arc::clone(&feed);
        let listener = stream
            .add_local_listener_with_user_data(0u32)
            .param_changed(move |_, channels, id, param| {
                let Some(param) = param else { return };
                if id != spa::param::ParamType::Format.as_raw() {
                    return;
                }
                let Ok((MediaType::Audio, MediaSubtype::Raw)) = format_utils::parse_format(param)
                else {
                    return;
                };
                let mut info = spa::param::audio::AudioInfoRaw::new();
                if info.parse(param).is_ok() && info.rate() > 0 && info.channels() > 0 {
                    *channels = info.channels();
                    format_feed.set_format(info.rate(), info.channels());
                }
            })
            .process(move |stream, channels| {
                let Some(mut buffer) = stream.dequeue_buffer() else {
                    return;
                };
                let datas = buffer.datas_mut();
                if datas.is_empty() || *channels == 0 {
                    return;
                }
                let d = &mut datas[0];
                let (offset, size) = (d.chunk().offset() as usize, d.chunk().size() as usize);
                if let Some(bytes) = d.data() {
                    let end = (offset + size).min(bytes.len());
                    let samples: Vec<f32> = bytes[offset.min(end)..end]
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    feed.push_f32(&samples);
                }
            })
            .register()?;
        let bytes = format_param();
        let pod = Pod::from_bytes(&bytes).expect("a format pod");
        let mut flags = pw::stream::StreamFlags::MAP_BUFFERS | pw::stream::StreamFlags::RT_PROCESS;
        if target.is_some() {
            flags |= pw::stream::StreamFlags::AUTOCONNECT;
        }
        stream.connect(spa::utils::Direction::Input, target, flags, &mut [pod])?;
        Ok(Self {
            stream,
            _listener: listener,
        })
    }
}

impl Drop for Capturing {
    fn drop(&mut self) {
        let _ = self.stream.disconnect();
    }
}

/// A node name from a display name: `Cadenza` → `cadenza`, spaces to dashes.
fn node_name(name: &str) -> String {
    let n: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    if n.is_empty() {
        "cadenza".into()
    } else {
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_app_matching() {
        assert_eq!(node_name("Cadenza"), "cadenza");
        assert_eq!(node_name("Living Room DAC"), "living-room-dac");
        assert_eq!(node_name("  "), "cadenza");
        let props = [
            ("application.name", "Firefox"),
            ("application.process.binary", "firefox-bin"),
            ("node.name", "Firefox"),
        ];
        assert!(matches_app("firefox", &props));
        assert!(matches_app("FIREFOX-BIN", &props));
        assert!(!matches_app("spotify", &props));
    }

    /// The PipeWire tests share one server; run them one at a time.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Is a PipeWire server reachable (the tests below need one; CI starts it)?
    fn server() -> bool {
        let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_default();
        let ok = !dir.is_empty() && std::path::Path::new(&dir).join("pipewire-0").exists();
        if !ok {
            eprintln!("no PipeWire server: skipped");
        }
        ok
    }

    fn engine() -> crate::engine::Engine {
        let mut s = crate::config::Settings::default();
        s.output.backend = "null".into();
        crate::engine::Engine::start(
            s,
            crate::engine::EngineOptions {
                cache: cadenza_dsp::CoefficientCache::disabled(),
                realtime: true,
                persist: None,
            },
        )
    }

    fn tone_wav(seconds: f64) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "cadenza-pw-{}-{}.wav",
            std::process::id(),
            (seconds * 10.0) as u32
        ));
        let mut w = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 2,
                sample_rate: 48_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for i in 0..(48_000.0 * seconds) as usize {
            let v = 0.5 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48_000.0).sin();
            w.write_sample(v as f32).unwrap();
            w.write_sample(v as f32).unwrap();
        }
        w.finalize().unwrap();
        path
    }

    fn wait(cond: impl Fn() -> bool, secs: u64) -> bool {
        let t = std::time::Instant::now();
        while t.elapsed().as_secs() < secs {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    /// Against a real PipeWire: an app plays to the "Cadenza" sink and the engine gets it.
    #[test]
    fn a_player_reaches_the_engine_through_the_virtual_sink() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        if !server() {
            return;
        }
        let e = engine();
        let cfg = SystemConfig {
            enabled: true,
            name: "Cadenza Test Sink".into(),
            ..Default::default()
        };
        let input = super::super::start(e.clone(), &cfg).unwrap();
        let wav = tone_wav(2.0);
        let status = std::process::Command::new("pw-play")
            .arg("--target")
            .arg("cadenza-test-sink")
            .arg(&wav)
            .status()
            .expect("pw-play");
        assert!(status.success());
        let s = input.stats();
        assert!(s.frames > 48_000, "{s:?}");
        assert_eq!((s.channels, s.backend.as_str()), (2, "pipewire"), "{s:?}");
        assert!(s.rate > 0);
        assert!((s.peak - 0.5).abs() < 0.02, "{s:?}");
        assert!(wait(
            || e.status()
                .track
                .as_ref()
                .is_some_and(|t| t.title == "Cadenza Test Sink (system audio)"),
            5
        ));
        let _ = std::fs::remove_file(wav);
    }

    /// Against a real PipeWire: one app's stream is captured by its name, and a second
    /// stream of the same app is followed after the first ends.
    #[test]
    fn one_app_is_captured_by_name() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        if !server() {
            return;
        }
        // Somewhere for the app to play to (a desktop has its sound card; the test server
        // has none): a second virtual sink, on its own engine.
        let _dump = super::super::start(
            engine(),
            &SystemConfig {
                enabled: true,
                name: "Cadenza Test Dump".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let e = engine();
        let cfg = SystemConfig {
            enabled: true,
            mode: SystemMode::App,
            app: "cadenza-test-app".into(),
            ..Default::default()
        };
        let input = super::super::start(e, &cfg).unwrap();
        let wav = tone_wav(1.5);
        let play = || {
            std::process::Command::new("pw-play")
                .arg("--target")
                .arg("cadenza-test-dump")
                .arg("-P")
                .arg("{ application.name = \"cadenza-test-app\" }")
                .arg(&wav)
                .status()
                .expect("pw-play")
        };
        assert!(play().success());
        let first = input.stats().frames;
        assert!(first > 24_000, "{:?}", input.stats());
        assert!(
            (input.stats().peak - 0.5).abs() < 0.02,
            "{:?}",
            input.stats()
        );
        assert!(play().success());
        assert!(
            input.stats().frames > first + 24_000,
            "the second stream was not followed: {:?}",
            input.stats()
        );
        let _ = std::fs::remove_file(wav);
    }
}
