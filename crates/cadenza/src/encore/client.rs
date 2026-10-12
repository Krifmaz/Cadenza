//! Blocking HTTP client for one Encore server: session cookie, CSRF header, re-login.

use crate::config::EncoreConfig;
use serde_json::Value;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// See the module documentation.
pub struct Client {
    config: Arc<EncoreConfig>,
    base: String,
    agent: ureq::Agent,
    cookie: Mutex<Option<String>>,
    device_id: String,
    device_name: String,
}

/// FNV-1a, enough to turn a hostname into a stable device id.
fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl Client {
    /// Prepare a client for `config` (nothing is sent yet).
    pub fn new(config: Arc<EncoreConfig>) -> Self {
        let base = config.url.trim().trim_end_matches('/').to_string();
        let host = gethostname::gethostname().to_string_lossy().to_string();
        let device_name = if config.device_name.trim().is_empty() {
            format!("Cadenza on {host}")
        } else {
            config.device_name.clone()
        };
        let device_id = format!("cadenza-{:016x}", fnv1a(&format!("{host}|{base}")));
        let cookie =
            (!config.cookie.trim().is_empty()).then(|| normalise_cookie(config.cookie.trim()));
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(20))
            .timeout_write(Duration::from_secs(20))
            .build();
        Self {
            config,
            base,
            agent,
            cookie: Mutex::new(cookie),
            device_id,
            device_name,
        }
    }

    /// Encore base URL without a trailing slash.
    pub fn base(&self) -> &str {
        &self.base
    }
    /// Stable id sent with every heartbeat.
    pub fn device_id(&self) -> &str {
        &self.device_id
    }
    /// Name shown in Encore's device list.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// The `Cookie` header value, logging in first when needed.
    pub fn cookie(&self) -> Result<String, String> {
        if let Some(c) = self.cookie.lock().unwrap().clone() {
            return Ok(c);
        }
        self.login()
    }

    /// Request headers a track decoder must send to stream from Encore.
    pub fn stream_headers(&self) -> Vec<(String, String)> {
        match self.cookie() {
            Ok(c) => vec![("Cookie".into(), c)],
            Err(_) => Vec::new(),
        }
    }

    /// `POST /api/auth/login` and remember the `encore_session` cookie.
    pub fn login(&self) -> Result<String, String> {
        if self.config.username.is_empty() {
            return Err("no Encore username or cookie configured".into());
        }
        let resp = self
            .agent
            .post(&format!("{}/api/auth/login", self.base))
            .set("X-Requested-With", "encore")
            .set("Accept", "application/json")
            .send_json(serde_json::json!({ "username": self.config.username, "password": self.config.password }))
            .map_err(|e| match e {
                ureq::Error::Status(code, r) => {
                    let body = r.into_string().unwrap_or_default();
                    format!("login failed: HTTP {code} {}", body.chars().take(200).collect::<String>())
                }
                other => format!("login failed: {other}"),
            })?;
        let cookie = resp
            .all("set-cookie")
            .iter()
            .filter_map(|c| c.split(';').next())
            .find(|c| c.trim_start().starts_with("encore_session="))
            .map(|c| c.trim().to_string())
            .ok_or_else(|| "login succeeded but no encore_session cookie came back".to_string())?;
        log::info!("logged in to Encore as {}", self.config.username);
        *self.cookie.lock().unwrap() = Some(cookie.clone());
        Ok(cookie)
    }

    fn forget_cookie(&self) {
        // Only a login-derived cookie can be refreshed; a configured one is kept.
        if self.config.cookie.trim().is_empty() {
            *self.cookie.lock().unwrap() = None;
        }
    }

    /// `POST` JSON, parse the JSON reply; a 401 triggers one re-login and retry.
    pub fn post_json(&self, path: &str, body: &Value) -> Result<Value, String> {
        for attempt in 0..2 {
            let cookie = self.cookie()?;
            let resp = self
                .agent
                .post(&format!("{}{}", self.base, path))
                .set("X-Requested-With", "encore")
                .set("Accept", "application/json")
                .set("Cookie", &cookie)
                .send_json(body.clone());
            match resp {
                Ok(r) => {
                    return r
                        .into_json::<Value>()
                        .map_err(|e| format!("bad JSON from {path}: {e}"))
                }
                Err(ureq::Error::Status(401, _)) if attempt == 0 => {
                    log::info!("Encore session expired; logging in again");
                    self.forget_cookie();
                    continue;
                }
                Err(ureq::Error::Status(code, r)) => {
                    let text = r.into_string().unwrap_or_default();
                    return Err(format!(
                        "{path}: HTTP {code} {}",
                        text.chars().take(200).collect::<String>()
                    ));
                }
                Err(e) => return Err(format!("{path}: {e}")),
            }
        }
        Err(format!("{path}: unauthorized"))
    }

    /// Open a long-lived `GET` (the SSE stream) and hand back its body reader.
    pub fn open_stream(&self, path: &str) -> Result<Box<dyn Read + Send + Sync + 'static>, String> {
        let cookie = self.cookie()?;
        // No read timeout: the server sends a ping every 15 s but a stream is meant to idle.
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .build();
        let resp = agent
            .get(&format!("{}{}", self.base, path))
            .set("Accept", "text/event-stream")
            .set("Cookie", &cookie)
            .call()
            .map_err(|e| match e {
                ureq::Error::Status(401, _) => {
                    self.forget_cookie();
                    "unauthorized".to_string()
                }
                other => other.to_string(),
            })?;
        Ok(resp.into_reader())
    }

    /// Absolute stream URL of an Encore track.
    pub fn track_url(&self, id: i64) -> String {
        format!("{}/api/library/tracks/{id}/stream", self.base)
    }

    /// Absolute cover URL of an Encore album.
    pub fn cover_url(&self, album_id: &str) -> String {
        format!("{}/api/library/albums/{album_id}/cover", self.base)
    }
}

/// `abc` → `encore_session=abc`; a full `name=value` is kept.
fn normalise_cookie(c: &str) -> String {
    if c.contains('=') {
        c.to_string()
    } else {
        format!("encore_session={c}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_id_is_stable_and_short() {
        let cfg = Arc::new(EncoreConfig {
            url: "http://x:8080/".into(),
            cookie: "tok".into(),
            ..Default::default()
        });
        let a = Client::new(Arc::clone(&cfg));
        let b = Client::new(cfg);
        assert_eq!(a.device_id(), b.device_id());
        assert!(a.device_id().len() <= 64 && a.device_id().starts_with("cadenza-"));
        assert_eq!(a.base(), "http://x:8080");
        assert_eq!(a.cookie().unwrap(), "encore_session=tok");
        assert_eq!(a.track_url(5), "http://x:8080/api/library/tracks/5/stream");
    }
}
