//! System-wide upsampler: the sound of the whole system, or of one app, played through
//! the engine into the DAC.
//!
//! A capture backend (a PipeWire virtual sink or app capture on Linux, process loopback on
//! Windows, a process tap on macOS) hands its PCM to a [`Feed`]. The feed keeps one
//! [`LiveStream`] per source format and plays it as a live track, so the engine follows
//! the source rate (a new rate starts a new track, as a new file would). The engine keeps
//! the DAC open while the system is quiet: the live stream hands out silence when dry.

use crate::decode::live::{self, LiveStream};
use crate::decode::Source;
use crate::engine::{Engine, PlayState, Track};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(all(windows, feature = "loopback"))]
pub mod loopback;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
pub mod pipewire;
#[cfg(all(target_os = "macos", feature = "tap"))]
pub mod tap;

/// What to capture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SystemMode {
    /// A virtual output device ("Cadenza") apps or the system play to.
    #[default]
    Sink,
    /// One application's audio, before the OS mixer.
    App,
}

/// `[system]` in config.toml.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemConfig {
    /// Run the system input (`cadenza serve`).
    pub enabled: bool,
    /// Virtual device or one app.
    pub mode: SystemMode,
    /// For `app`: the application (its name, process binary or PipeWire node name).
    pub app: String,
    /// Name of the virtual device.
    pub name: String,
    /// Audio buffered before playing, ms: room for the capture and the DAC clocks to
    /// drift and for scheduling jitter.
    pub latency_ms: u32,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: SystemMode::Sink,
            app: String::new(),
            name: "Cadenza".into(),
            latency_ms: 150,
        }
    }
}

impl SystemConfig {
    /// The `--system` flag: `sink` or `app:<name>` (enables the input).
    pub fn parse_flag(flag: &str, base: &SystemConfig) -> Result<SystemConfig, String> {
        let mut c = base.clone();
        c.enabled = true;
        match flag.trim() {
            "sink" => c.mode = SystemMode::Sink,
            f => match f.strip_prefix("app:") {
                Some(app) if !app.trim().is_empty() => {
                    c.mode = SystemMode::App;
                    c.app = app.trim().to_string();
                }
                _ => return Err(format!("--system takes sink or app:<name>, not {flag:?}")),
            },
        }
        Ok(c)
    }
}

/// What a running system input reports.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SystemStats {
    /// The backend (`pipewire`, …).
    pub backend: String,
    /// Source rate in Hz (0 before the first format).
    pub rate: u32,
    /// Source channels.
    pub channels: u32,
    /// Frames received.
    pub frames: u64,
    /// Peak level of the last second, linear.
    pub peak: f64,
    /// Formats seen (each starts a new live track).
    pub formats: u64,
}

/// The live track now fed: rate, channels, stream and its registry id.
type Current = (u32, u32, Arc<LiveStream>, String);

/// Where a capture backend delivers PCM.
pub struct Feed {
    engine: Engine,
    title: String,
    codec: String,
    latency_ms: u32,
    current: Mutex<Option<Current>>,
    frames: AtomicU64,
    formats: AtomicU64,
    peak: Mutex<(f64, u64)>,
    backend: String,
}

impl Feed {
    /// A feed into `engine`; `title` names the live track, `codec` its format label.
    pub fn new(engine: Engine, backend: &str, title: &str, codec: &str, latency_ms: u32) -> Self {
        Self {
            engine,
            title: title.to_string(),
            codec: codec.to_string(),
            latency_ms,
            current: Mutex::new(None),
            frames: AtomicU64::new(0),
            formats: AtomicU64::new(0),
            peak: Mutex::new((0.0, 0)),
            backend: backend.to_string(),
        }
    }

    /// The source format is (now) `rate` Hz, `channels` channels: a new format closes the
    /// previous live track and starts one at the new rate. The engine is only taken over
    /// when it is idle or already playing the system input.
    pub fn set_format(&self, rate: u32, channels: u32) {
        let mut cur = self.current.lock().unwrap();
        if matches!(&*cur, Some((r, c, _, _)) if *r == rate && *c == channels) {
            return;
        }
        let n = self.formats.fetch_add(1, Ordering::Relaxed) + 1;
        let ours = |id: &str| id.starts_with("system-");
        let st = self.engine.status();
        let playing_ours = st
            .track
            .as_ref()
            .and_then(|t| t.id.as_deref())
            .is_some_and(ours);
        let idle = st.track.is_none() || st.state != PlayState::Playing;
        if let Some((_, _, old, id)) = cur.take() {
            old.close();
            live::unregister(&id);
        }
        let stream = Arc::new(LiveStream::new(
            rate,
            channels,
            32,
            &self.codec,
            self.latency_ms,
        ));
        let id = format!("system-{}-{n}", std::process::id());
        live::register(&id, Arc::clone(&stream));
        if idle || playing_ours {
            self.engine.play_queue(
                vec![Track {
                    id: Some(id.clone()),
                    source: Source::Live { live: id.clone() },
                    title: self.title.clone(),
                    format: Some(format!("{} {}", self.codec, rate_label(rate))),
                    ..Default::default()
                }],
                0,
                0.0,
            );
        } else {
            log::info!("system input: {rate} Hz; the engine is busy, not taking it over");
        }
        log::info!("system input: {channels} ch at {rate} Hz");
        *cur = Some((rate, channels, stream, id));
    }

    /// Interleaved `f32` samples in the current format.
    pub fn push_f32(&self, samples: &[f32]) {
        let (stream, ch, rate) = match &*self.current.lock().unwrap() {
            Some((rate, ch, s, _)) => (Arc::clone(s), *ch, *rate),
            None => return,
        };
        let mut peak = 0.0f64;
        let v: Vec<f64> = samples
            .iter()
            .map(|&x| {
                let x = x as f64;
                peak = peak.max(x.abs());
                x
            })
            .collect();
        let frames = (v.len() / ch.max(1) as usize) as u64;
        stream.push(v);
        let total = self.frames.fetch_add(frames, Ordering::Relaxed) + frames;
        let mut p = self.peak.lock().unwrap();
        // A peak meter that holds for about a second.
        if total - p.1 > rate as u64 {
            *p = (peak, total);
        } else if peak > p.0 {
            p.0 = peak;
        }
    }

    /// What has arrived so far.
    pub fn stats(&self) -> SystemStats {
        let cur = self.current.lock().unwrap();
        let (rate, channels) = cur.as_ref().map(|c| (c.0, c.1)).unwrap_or((0, 0));
        SystemStats {
            backend: self.backend.clone(),
            rate,
            channels,
            frames: self.frames.load(Ordering::Relaxed),
            peak: self.peak.lock().unwrap().0,
            formats: self.formats.load(Ordering::Relaxed),
        }
    }

    /// Stop: close the live track.
    pub fn close(&self) {
        if let Some((_, _, s, id)) = self.current.lock().unwrap().take() {
            s.close();
            live::unregister(&id);
        }
    }
}

impl Drop for Feed {
    fn drop(&mut self) {
        self.close();
    }
}

/// Planar channels (one slice per channel, as some capture APIs deliver them) to
/// interleaved frames; a shorter plane ends the output.
pub fn interleave_planar(planes: &[&[f32]]) -> Vec<f32> {
    let frames = planes.iter().map(|p| p.len()).min().unwrap_or(0);
    let mut out = Vec::with_capacity(frames * planes.len());
    for i in 0..frames {
        out.extend(planes.iter().map(|p| p[i]));
    }
    out
}

fn rate_label(rate: u32) -> String {
    if rate % 1000 == 0 {
        format!("{}k", rate / 1000)
    } else {
        format!("{:.1}k", rate as f64 / 1000.0)
    }
}

/// A running system input; dropping it stops the capture.
pub struct SystemInput {
    feed: Arc<Feed>,
    _backend: Box<dyn Send>,
}

impl SystemInput {
    /// What has arrived so far.
    pub fn stats(&self) -> SystemStats {
        self.feed.stats()
    }

    /// The feed (tests).
    pub fn feed(&self) -> &Arc<Feed> {
        &self.feed
    }
}

/// Whether this build has a system input for this platform.
pub const AVAILABLE: bool = cfg!(any(
    all(target_os = "linux", feature = "pipewire"),
    all(windows, feature = "loopback"),
    all(target_os = "macos", feature = "tap")
));

struct Global {
    config: Option<SystemConfig>,
    input: Option<SystemInput>,
    error: Option<String>,
}

static GLOBAL: Mutex<Global> = Mutex::new(Global {
    config: None,
    input: None,
    error: None,
});

/// Start the system input for the life of the process (`cadenza serve`); its state is
/// what [`status`] reports.
pub fn start_global(engine: Engine, cfg: &SystemConfig) {
    let mut g = GLOBAL.lock().unwrap();
    g.input = None;
    g.config = Some(cfg.clone());
    match start(engine, cfg) {
        Ok(i) => {
            log::info!("system input: {:?} running", cfg.mode);
            g.input = Some(i);
            g.error = None;
        }
        Err(e) => {
            log::error!("{e}");
            g.error = Some(e);
        }
    }
}

/// `GET /api/system`: whether this build can, the configuration, and what is arriving.
pub fn status() -> serde_json::Value {
    let g = GLOBAL.lock().unwrap();
    serde_json::json!({
        "available": AVAILABLE,
        "config": g.config,
        "running": g.input.is_some(),
        "stats": g.input.as_ref().map(SystemInput::stats),
        "error": g.error,
    })
}

/// Start the system input this platform and build support.
pub fn start(engine: Engine, cfg: &SystemConfig) -> Result<SystemInput, String> {
    if cfg.mode == SystemMode::App && cfg.app.trim().is_empty() {
        return Err("system input: mode = \"app\" needs app = \"<application>\"".into());
    }
    #[cfg(all(target_os = "linux", feature = "pipewire"))]
    {
        let title = match cfg.mode {
            SystemMode::Sink => format!("{} (system audio)", cfg.name),
            SystemMode::App => cfg.app.clone(),
        };
        let feed = Arc::new(Feed::new(
            engine,
            "pipewire",
            &title,
            "PipeWire",
            cfg.latency_ms,
        ));
        let backend = pipewire::Capture::start(cfg, Arc::clone(&feed))?;
        Ok(SystemInput {
            feed,
            _backend: Box::new(backend),
        })
    }
    #[cfg(all(windows, feature = "loopback"))]
    {
        let title = match cfg.mode {
            SystemMode::Sink => "All apps (system audio)".to_string(),
            SystemMode::App => cfg.app.clone(),
        };
        let feed = Arc::new(Feed::new(
            engine,
            "process-loopback",
            &title,
            "Loopback",
            cfg.latency_ms,
        ));
        let backend = loopback::Capture::start(cfg, Arc::clone(&feed))?;
        Ok(SystemInput {
            feed,
            _backend: Box::new(backend),
        })
    }
    #[cfg(all(target_os = "macos", feature = "tap"))]
    {
        let title = match cfg.mode {
            SystemMode::Sink => format!("{} (system audio)", cfg.name),
            SystemMode::App => cfg.app.clone(),
        };
        let feed = Arc::new(Feed::new(
            engine,
            "process-tap",
            &title,
            "Tap",
            cfg.latency_ms,
        ));
        let backend = tap::Capture::start(cfg, Arc::clone(&feed))?;
        Ok(SystemInput {
            feed,
            _backend: Box::new(backend),
        })
    }
    #[cfg(not(any(
        all(target_os = "linux", feature = "pipewire"),
        all(windows, feature = "loopback"),
        all(target_os = "macos", feature = "tap")
    )))]
    {
        let _ = engine;
        Err("this build has no system input for this platform (build with the pipewire feature on Linux, loopback on Windows, tap on macOS)".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Settings;
    use crate::engine::EngineOptions;
    use cadenza_dsp::CoefficientCache;

    fn engine() -> Engine {
        let mut s = Settings::default();
        s.output.backend = "null".into();
        Engine::start(
            s,
            EngineOptions {
                cache: CoefficientCache::disabled(),
                realtime: true,
                persist: None,
            },
        )
    }

    fn wait(cond: impl Fn() -> bool) -> bool {
        let t = std::time::Instant::now();
        while t.elapsed().as_secs() < 10 {
            if cond() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn the_feed_follows_the_source_rate_and_plays() {
        let e = engine();
        let feed = Feed::new(e.clone(), "test", "System audio", "PCM", 50);
        feed.set_format(48_000, 2);
        let tone: Vec<f32> = (0..48_000 * 2)
            .map(|i| 0.25 * ((i / 2) as f32 * 0.1).sin())
            .collect();
        feed.push_f32(&tone);
        assert!(wait(|| e.status().state == PlayState::Playing
            && e.status().track.as_ref().map(|t| t.title.as_str())
                == Some("System audio")));
        let path = e.status().signal_path.clone();
        assert_eq!(path.map(|p| p.input.rate), Some(48_000));
        // The same format again changes nothing; a new rate starts a new track.
        feed.set_format(48_000, 2);
        assert_eq!(feed.stats().formats, 1);
        feed.set_format(44_100, 2);
        feed.push_f32(&tone);
        assert!(wait(|| e
            .status()
            .signal_path
            .as_ref()
            .is_some_and(|p| p.input.rate == 44_100)));
        let s = feed.stats();
        assert_eq!((s.rate, s.channels, s.formats), (44_100, 2, 2));
        assert_eq!(s.frames, 96_000);
        assert!((s.peak - 0.25).abs() < 0.01, "{}", s.peak);
    }

    #[test]
    fn a_busy_engine_is_not_taken_over() {
        let e = engine();
        let id = "system-test-busy";
        let other = Arc::new(LiveStream::new(44_100, 2, 16, "PCM", 0));
        live::register(id.trim_start_matches("system-"), Arc::clone(&other));
        e.play_queue(
            vec![Track {
                id: Some("someone-else".into()),
                source: Source::Live {
                    live: id.trim_start_matches("system-").into(),
                },
                title: "A record".into(),
                ..Default::default()
            }],
            0,
            0.0,
        );
        other.push(vec![0.0; 44_100 * 2]);
        assert!(wait(|| e.status().state == PlayState::Playing));
        let feed = Feed::new(e.clone(), "test", "System audio", "PCM", 50);
        feed.set_format(48_000, 2);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(
            e.status().track.as_ref().map(|t| t.title.clone()),
            Some("A record".into())
        );
        live::unregister(id.trim_start_matches("system-"));
    }

    #[test]
    fn the_flag_picks_the_mode() {
        let base = SystemConfig::default();
        let s = SystemConfig::parse_flag("sink", &base).unwrap();
        assert!(s.enabled);
        assert_eq!(s.mode, SystemMode::Sink);
        let a = SystemConfig::parse_flag("app:Firefox", &base).unwrap();
        assert_eq!((a.mode, a.app.as_str()), (SystemMode::App, "Firefox"));
        assert!(SystemConfig::parse_flag("app:", &base).is_err());
        assert!(SystemConfig::parse_flag("everything", &base).is_err());
        let st = status();
        assert_eq!(st["available"], AVAILABLE);
    }

    #[test]
    fn planar_to_interleaved() {
        let l = [1.0f32, 2.0, 3.0];
        let r = [-1.0f32, -2.0, -3.0, -4.0];
        assert_eq!(
            interleave_planar(&[&l, &r]),
            vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
        );
        assert!(interleave_planar(&[]).is_empty());
    }
}
