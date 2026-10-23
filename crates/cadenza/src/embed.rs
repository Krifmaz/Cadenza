//! Cadenza inside another program (the Encore desktop app): the engine with its own
//! config file and coefficient cache under one directory, and an Encore link that can be
//! connected, re-pointed and disconnected at run time. No HTTP server, no CLI.

use crate::config::{Config, Settings};
use crate::encore::EncoreLink;
use crate::engine::{Engine, EngineOptions};
use cadenza_dsp::CoefficientCache;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A running engine plus, when connected, its Encore link.
pub struct EmbeddedPlayer {
    /// Directory holding `config.toml` and the caches.
    dir: PathBuf,
    /// Name the player registers with in Encore.
    device_name: String,
    /// The playback engine.
    engine: Engine,
    /// The Encore connection, when configured.
    link: Mutex<Option<Arc<EncoreLink>>>,
}

impl EmbeddedPlayer {
    /// Load (or create) `<dir>/config.toml`, start the engine with a persist callback that
    /// writes it back and the coefficient cache under `<dir>/cache`. `device_name` is what
    /// Encore shows for this player (`Cadenza on <hostname>`).
    pub fn start(dir: PathBuf, device_name: String) -> anyhow::Result<EmbeddedPlayer> {
        std::fs::create_dir_all(&dir)?;
        let path = Self::config_path_in(&dir);
        let config = Config::load(&path)?;
        if !path.exists() {
            config.save(&path)?;
        }
        let persist_cfg = Mutex::new(config.clone());
        let persist_path = path.clone();
        let persist = Box::new(move |s: &Settings| {
            let mut c = persist_cfg.lock().unwrap_or_else(|e| e.into_inner());
            c.apply(s);
            if let Err(e) = c.save(&persist_path) {
                log::warn!("could not save {}: {e}", persist_path.display());
            }
        });
        let engine = Engine::start(
            config.settings(),
            EngineOptions {
                cache: CoefficientCache::in_dir(dir.join("cache")),
                realtime: true,
                persist: Some(persist),
            },
        );
        log::info!("embedded Cadenza engine started; config {}", path.display());
        Ok(EmbeddedPlayer {
            dir,
            device_name,
            engine,
            link: Mutex::new(None),
        })
    }

    /// `config.toml` inside `dir`.
    fn config_path_in(dir: &Path) -> PathBuf {
        dir.join("config.toml")
    }

    /// `<dir>/config.toml`.
    pub fn config_path(&self) -> PathBuf {
        Self::config_path_in(&self.dir)
    }

    /// The directory everything lives in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The name Encore shows for this player.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// The engine handle (settings, queue, status).
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Connect to `server` with an app `token` as device `device_id`. A previous link is
    /// stopped first; connecting again with the same server, token and id is a no-op.
    pub fn connect(&self, server: &str, token: &str, device_id: &str) -> Arc<EncoreLink> {
        let config = crate::config::EncoreConfig {
            url: server.trim().to_string(),
            token: token.trim().to_string(),
            device_id: device_id.trim().to_string(),
            device_name: self.device_name.clone(),
            embedded: true,
            ..Default::default()
        };
        let mut slot = self.link.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(l) = slot.as_ref() {
            let same = !l.stopped()
                && l.server() == config.url.trim_end_matches('/')
                && l.device_id() == config.device_id
                && l.token() == config.token;
            if same {
                return Arc::clone(l);
            }
            l.stop();
        }
        let link = Arc::new(crate::encore::start(Arc::new(config), self.engine.clone()));
        *slot = Some(Arc::clone(&link));
        link
    }

    /// Stop the Encore link, if any. The engine keeps its settings and stops playing.
    pub fn disconnect(&self) {
        let mut slot = self.link.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(l) = slot.take() {
            l.stop();
            self.engine.send(crate::engine::Command::Stop);
        }
    }

    /// The current link, if connected.
    pub fn link(&self) -> Option<Arc<EncoreLink>> {
        self.link.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// `{running, device_id, name, error}` for the app's `engineStatus()`. Without a link,
    /// `device_id` is `null`, `name` the device name, `error` the engine's last error.
    pub fn status_json(&self) -> serde_json::Value {
        let link = self.link();
        let engine_error = self.engine.status().error.clone();
        serde_json::json!({
            "running": link.is_some(),
            "device_id": link.as_ref().map(|l| l.device_id().to_string()),
            "name": self.device_name,
            "error": link.as_ref().and_then(|l| l.last_error()).or(engine_error),
        })
    }

    /// Stop the link and the engine (the output device is released).
    pub fn shutdown(self) {
        self.disconnect();
        self.engine.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cadenza-embed-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn starts_with_the_null_output_and_persists_a_setting() {
        let dir = temp_dir("persist");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.toml"), "[output]\nbackend = \"null\"\n").unwrap();
        let player = EmbeddedPlayer::start(dir.clone(), "Cadenza on test".into()).unwrap();
        assert_eq!(player.config_path(), dir.join("config.toml"));
        assert_eq!(player.engine().settings().output.backend, "null");
        let st = player.status_json();
        assert_eq!(st["running"], false);
        assert_eq!(st["device_id"], serde_json::Value::Null);
        assert_eq!(st["name"], "Cadenza on test");

        let mut s = player.engine().settings();
        s.dsp.filter = "sinc-l-4k".into();
        s.output.bits = 16;
        player.engine().update_settings(s);
        let t0 = Instant::now();
        let mut saved = Config::default();
        while t0.elapsed() < Duration::from_secs(5) {
            saved = Config::load(&player.config_path()).unwrap();
            if saved.dsp.filter == "sinc-l-4k" {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(saved.dsp.filter, "sinc-l-4k");
        assert_eq!(saved.output.bits, 16);
        assert_eq!(saved.output.backend, "null", "the rest of the file is kept");
        // The coefficient cache lives next to the config.
        let mut s = player.engine().settings();
        s.output.rate = crate::config::RateSetting::Fixed(88_200);
        player.engine().update_settings(s);
        player.shutdown();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn connect_is_idempotent_and_disconnect_stops_the_link() {
        let dir = temp_dir("link");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.toml"), "[output]\nbackend = \"null\"\n").unwrap();
        let player = EmbeddedPlayer::start(dir.clone(), "Cadenza on test".into()).unwrap();
        // Nobody listens here: the link keeps retrying, which is all this test needs.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        drop(listener);
        let a = player.connect(&url, "tok", "cadenza-desktop-1");
        let b = player.connect(&url, "tok", "cadenza-desktop-1");
        assert!(Arc::ptr_eq(&a, &b), "same server, token and id: same link");
        let st = player.status_json();
        assert_eq!(st["running"], true);
        assert_eq!(st["device_id"], "cadenza-desktop-1");
        assert_eq!(st["name"], "Cadenza on test");
        let c = player.connect(&url, "tok2", "cadenza-desktop-1");
        assert!(!Arc::ptr_eq(&a, &c), "a new token re-points the link");
        assert!(a.stopped());
        assert!(!c.stopped());
        player.disconnect();
        assert!(c.stopped());
        assert_eq!(player.status_json()["running"], false);
        player.shutdown();
        let _ = std::fs::remove_dir_all(dir);
    }
}
