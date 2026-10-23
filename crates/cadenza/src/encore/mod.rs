//! Cadenza as an Encore playback device.
//!
//! * log in (or use a given `encore_session` cookie, or an app token as a bearer),
//! * every 2 s `POST /api/remote/heartbeat` with the player state and run the commands
//!   that come back,
//! * keep `GET /api/remote/events?device=…` open as an SSE stream so commands arrive the
//!   moment they are sent (the heartbeat stays as the fallback; a command's `seq` makes
//!   sure it runs once even when it arrives both ways),
//! * play Encore tracks from `/api/library/tracks/{id}/stream` with the same credentials,
//! * report plays like Encore's own player (`now-playing` when a track starts, `plays`
//!   after half the song or four minutes).
//!
//! [`start`] hands back an [`EncoreLink`]; `cadenza serve` keeps it for the life of the
//! process, the embedded player stops it when the user signs out or changes server.

mod client;
pub mod commands;
pub mod state;

pub use client::Client;

use crate::config::EncoreConfig;
use crate::engine::{Engine, Event};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// A running connection to one Encore server: the heartbeat, event stream and play
/// reporting threads. Dropping the handle does not stop them; [`EncoreLink::stop`] does.
pub struct EncoreLink {
    /// The HTTP client of this connection.
    client: Arc<Client>,
    /// Set to stop the heartbeat and event threads.
    stop: Arc<AtomicBool>,
}

impl EncoreLink {
    /// Ask every thread to end at its next iteration. The heartbeat stops within 10 s, the
    /// play reporter at once, the event reader when its socket times out (60 s at most).
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Whether `stop` was called.
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// Device id this link registers as.
    pub fn device_id(&self) -> &str {
        self.client.device_id()
    }

    /// Device name shown in Encore.
    pub fn device_name(&self) -> &str {
        self.client.device_name()
    }

    /// Encore base URL.
    pub fn server(&self) -> &str {
        self.client.base()
    }

    /// The bearer token this link uses (empty with cookie or password login).
    pub fn token(&self) -> &str {
        self.client.token()
    }

    /// The last failure talking to Encore, `None` after the last success.
    pub fn last_error(&self) -> Option<String> {
        self.client.last_error()
    }
}

/// The one live Encore link and the configuration behind it, so the web UI can log in and
/// out while the process runs. The link's threads are not stopped by dropping it, so a
/// replacement always calls [`EncoreLink::stop`] on the old one first.
static LINK: Mutex<Option<EncoreLink>> = Mutex::new(None);
/// The Encore configuration the current link was started with.
static CONFIG: Mutex<Option<EncoreConfig>> = Mutex::new(None);

/// A callback that writes the Encore configuration back to `config.toml`, registered by
/// `cadenza::service::start`.
type Persist = Box<dyn Fn(&EncoreConfig) + Send>;
/// The registered persistence function, if any.
static PERSIST: Mutex<Option<Persist>> = Mutex::new(None);

/// Register the function that persists the Encore configuration (called once at startup).
pub fn set_persist(f: impl Fn(&EncoreConfig) + Send + 'static) {
    *PERSIST.lock().unwrap() = Some(Box::new(f));
}

/// Persist `cfg` through the registered function, if there is one.
fn persist(cfg: &EncoreConfig) {
    if let Some(f) = PERSIST.lock().unwrap().as_ref() {
        f(cfg);
    }
}

/// Start (or restart) the one process-wide Encore link, replacing any current one. With no
/// server URL it only stops the old link (signed out).
pub fn start_global(engine: Engine, cfg: &EncoreConfig) {
    let mut g = LINK.lock().unwrap();
    if let Some(old) = g.take() {
        old.stop();
    }
    *g = cfg.enabled().then(|| start(Arc::new(cfg.clone()), engine));
    *CONFIG.lock().unwrap() = Some(cfg.clone());
}

/// Apply a configuration from the web UI (log in, out, or switch servers) and save it. A
/// blank password keeps the saved one for the same server and user, so the UI never has to
/// echo the password back.
pub fn set_config(engine: Engine, cfg: &EncoreConfig) {
    let mut cfg = cfg.clone();
    if cfg.password.is_empty() {
        if let Some(old) = CONFIG.lock().unwrap().as_ref() {
            if cfg.url == old.url && cfg.username == old.username {
                cfg.password = old.password.clone();
                cfg.cookie = old.cookie.clone();
                cfg.token = old.token.clone();
            }
        }
    }
    start_global(engine, &cfg);
    persist(&cfg);
}

/// The signed-in server's client, for the in-app library (`None` when signed out).
pub fn client() -> Option<Arc<Client>> {
    LINK.lock().unwrap().as_ref().map(|l| Arc::clone(&l.client))
}

/// `GET /api/encore`: the server and sign-in state, for the web UI's login form. The
/// password is never returned.
pub fn status() -> serde_json::Value {
    let link = LINK.lock().unwrap();
    let cfg = CONFIG.lock().unwrap();
    let c = cfg.as_ref();
    serde_json::json!({
        "configured": link.is_some(),
        "server": c.map(|c| c.url.as_str()).unwrap_or_default(),
        "username": c.map(|c| c.username.as_str()).unwrap_or_default(),
        "device_name": link.as_ref().map(|l| l.device_name()).unwrap_or_default(),
        "error": link.as_ref().and_then(|l| l.last_error()),
        "connected": link.as_ref().is_some_and(|l| l.last_error().is_none()),
    })
}

/// Start the three Encore threads (heartbeat, event stream, play reporting).
pub fn start(config: Arc<EncoreConfig>, engine: Engine) -> EncoreLink {
    let client = Arc::new(Client::new(config));
    let stop = Arc::new(AtomicBool::new(false));
    log::info!(
        "Encore device {} ({}) → {}",
        client.device_name(),
        client.device_id(),
        client.base()
    );
    {
        let (c, e, s) = (Arc::clone(&client), engine.clone(), Arc::clone(&stop));
        thread::Builder::new()
            .name("encore-heartbeat".into())
            .spawn(move || heartbeat_loop(c, e, s))
            .expect("spawn");
    }
    {
        let (c, e, s) = (Arc::clone(&client), engine.clone(), Arc::clone(&stop));
        thread::Builder::new()
            .name("encore-events".into())
            .spawn(move || events_loop(c, e, s))
            .expect("spawn");
    }
    {
        let (c, s) = (Arc::clone(&client), Arc::clone(&stop));
        thread::Builder::new()
            .name("encore-plays".into())
            .spawn(move || plays_loop(c, engine, s))
            .expect("spawn");
    }
    EncoreLink { client, stop }
}

/// Sleep `total` in short steps, returning early (true) when `stop` is raised.
fn wait(stop: &AtomicBool, total: Duration) -> bool {
    let step = Duration::from_millis(100);
    let mut left = total;
    while !left.is_zero() {
        if stop.load(Ordering::SeqCst) {
            return true;
        }
        let d = left.min(step);
        thread::sleep(d);
        left -= d;
    }
    stop.load(Ordering::SeqCst)
}

/// Every 2 s: report the player state with a heartbeat and run the commands it returns;
/// backs off while the server cannot be reached, until `stop` is set.
fn heartbeat_loop(client: Arc<Client>, engine: Engine, stop: Arc<AtomicBool>) {
    let mut failures = 0u32;
    while !stop.load(Ordering::SeqCst) {
        let status = engine.status();
        let body = serde_json::json!({
            "id": client.device_id(),
            "name": client.device_name(),
            "state": state::heartbeat_state_as(&client, &status, client.embedded()),
        });
        let pause = match client.post_json("/api/remote/heartbeat", &body) {
            Ok(resp) => {
                if failures > 0 {
                    log::info!("Encore heartbeat back");
                }
                failures = 0;
                if let Some(cmds) = resp.get("commands").and_then(|c| c.as_array()) {
                    for cmd in cmds {
                        commands::run(&client, &engine, cmd);
                    }
                }
                Duration::from_secs(2)
            }
            Err(e) => {
                failures += 1;
                if failures == 1 || failures % 30 == 0 {
                    log::warn!("Encore heartbeat failed: {e}");
                }
                // A token that is not accepted cannot be refreshed here: keep trying,
                // calmly, until the app hands over a new one or the server is back.
                if client.has_token() {
                    Duration::from_secs(10)
                } else {
                    Duration::from_secs(2)
                }
            }
        };
        if wait(&stop, pause) {
            break;
        }
    }
    log::info!("Encore heartbeat stopped");
}

/// Keep the command stream open; reconnect after a pause whenever it drops.
fn events_loop(client: Arc<Client>, engine: Engine, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::SeqCst) {
        match client.open_stream(&format!("/api/remote/events?device={}", client.device_id())) {
            Ok(reader) => {
                log::info!("Encore event stream open");
                read_sse(BufReader::new(reader), |event, data| {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    if event == "command" {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            if let Some(cmd) = v.get("command") {
                                commands::run(&client, &engine, cmd);
                            }
                        }
                    }
                });
                if !stop.load(Ordering::SeqCst) {
                    log::warn!("Encore event stream closed; reconnecting");
                }
            }
            Err(e) => log::warn!("Encore event stream: {e}"),
        }
        if wait(&stop, Duration::from_secs(2)) {
            break;
        }
    }
    log::info!("Encore event stream stopped");
}

/// Parse an SSE stream, calling `on_event(event_name, data)` for each message.
pub fn read_sse<R: BufRead>(reader: R, mut on_event: impl FnMut(&str, &str)) {
    let mut event = String::from("message");
    let mut data = String::new();
    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.is_empty() {
            if !data.is_empty() {
                on_event(&event, &data);
            }
            event = "message".into();
            data.clear();
        } else if let Some(v) = line.strip_prefix("event:") {
            event = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(v.trim_start());
        }
        // comments (": ping") and ids are ignored
    }
}

/// Report starts and plays of Encore tracks.
fn plays_loop(client: Arc<Client>, engine: Engine, stop: Arc<AtomicBool>) {
    use tokio::sync::broadcast::error::TryRecvError;
    let mut rx = engine.subscribe();
    while !stop.load(Ordering::SeqCst) {
        match rx.try_recv() {
            Ok(Event::TrackStarted(t)) => {
                if let Some(id) = encore_track_id(&t) {
                    if let Err(e) = client.post_json(
                        "/api/library/now-playing",
                        &serde_json::json!({ "track_id": id }),
                    ) {
                        log::debug!("now-playing: {e}");
                    }
                }
            }
            Ok(Event::TrackPlayed { track, seconds }) => {
                if let Some(id) = encore_track_id(&track) {
                    if let Err(e) = client.post_json(
                        "/api/library/plays",
                        &serde_json::json!({ "track_id": id, "seconds": seconds }),
                    ) {
                        log::debug!("plays: {e}");
                    }
                }
            }
            Ok(Event::Status(_)) => {}
            Err(TryRecvError::Lagged(_)) => continue,
            Err(TryRecvError::Empty) => thread::sleep(Duration::from_millis(200)),
            Err(TryRecvError::Closed) => break,
        }
    }
}

/// Only tracks that came from Encore (numeric id and an Encore stream URL) are reported.
fn encore_track_id(t: &crate::engine::Track) -> Option<i64> {
    let id = t.id.as_deref()?.parse::<i64>().ok()?;
    match &t.source {
        crate::decode::Source::Url { url, .. } if url.contains("/api/library/tracks/") => Some(id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Settings;
    use crate::engine::EngineOptions;
    use cadenza_dsp::CoefficientCache;

    #[test]
    fn sse_parser_handles_multi_line_data_and_comments() {
        let text = ": ping\n\nevent: command\ndata: {\"a\":\ndata: 1}\n\ndata: plain\n\n";
        let mut got = Vec::new();
        read_sse(std::io::Cursor::new(text), |e, d| {
            got.push((e.to_string(), d.to_string()))
        });
        assert_eq!(
            got,
            vec![
                ("command".to_string(), "{\"a\":\n1}".to_string()),
                ("message".to_string(), "plain".to_string())
            ]
        );
    }

    #[test]
    fn stop_ends_the_loops_against_a_dead_server() {
        // A port nobody listens on: every request fails fast, the loops keep retrying
        // until `stop`, then end on their own.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let mut s = Settings::default();
        s.output.backend = "null".into();
        let engine = Engine::start(
            s,
            EngineOptions {
                cache: CoefficientCache::disabled(),
                realtime: false,
                persist: None,
            },
        );
        let link = start(
            Arc::new(EncoreConfig {
                url,
                token: "t".into(),
                device_id: "cadenza-desktop-test".into(),
                device_name: "Cadenza on test".into(),
                ..Default::default()
            }),
            engine.clone(),
        );
        assert_eq!(link.device_id(), "cadenza-desktop-test");
        assert_eq!(link.device_name(), "Cadenza on test");
        // The first heartbeat fails and leaves its error behind.
        let t0 = std::time::Instant::now();
        while link.last_error().is_none() && t0.elapsed() < Duration::from_secs(10) {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(link.last_error().is_some());
        assert!(!link.stopped());
        link.stop();
        assert!(link.stopped());
        // The heartbeat thread leaves its 10 s pause early and is gone within a moment.
        let t0 = std::time::Instant::now();
        let mut alive = true;
        while alive && t0.elapsed() < Duration::from_secs(5) {
            thread::sleep(Duration::from_millis(50));
            alive = false;
            // There is no join handle by design; look the threads up by name instead.
            for entry in std::fs::read_dir("/proc/self/task")
                .into_iter()
                .flatten()
                .flatten()
            {
                if let Ok(name) = std::fs::read_to_string(entry.path().join("comm")) {
                    if name.trim() == "encore-heartbea" || name.trim() == "encore-plays" {
                        alive = true;
                    }
                }
            }
        }
        if cfg!(target_os = "linux") {
            assert!(!alive, "heartbeat/plays threads still running after stop");
        }
        engine.shutdown();
    }

    #[test]
    fn blank_password_is_kept_and_sign_out_clears() {
        let saved = Arc::new(Mutex::new(None));
        let s = saved.clone();
        set_persist(move |c: &EncoreConfig| *s.lock().unwrap() = Some(c.clone()));
        let mut st = Settings::default();
        st.output.backend = "null".into();
        let engine = Engine::start(
            st,
            EngineOptions {
                cache: CoefficientCache::disabled(),
                realtime: false,
                persist: None,
            },
        );
        // Pretend we are already signed in (no live link needed to test the merge).
        *CONFIG.lock().unwrap() = Some(EncoreConfig {
            username: "me".into(),
            password: "secret".into(),
            ..Default::default()
        });
        // Re-applying with the same user and a blank password keeps the saved password.
        set_config(
            engine.clone(),
            &EncoreConfig {
                username: "me".into(),
                ..Default::default()
            },
        );
        assert_eq!(saved.lock().unwrap().as_ref().unwrap().password, "secret");
        // Signing out (empty account) clears the password and the link.
        set_config(engine.clone(), &EncoreConfig::default());
        assert_eq!(saved.lock().unwrap().as_ref().unwrap().password, "");
        assert_eq!(status()["configured"], false);
        engine.shutdown();
    }
}
