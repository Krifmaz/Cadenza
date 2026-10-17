//! Blocking HTTP client for one Encore server: bearer token or session cookie, CSRF
//! header, re-login, last error.

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
    token: Option<String>,
    device_id: String,
    device_name: String,
    last_error: Mutex<Option<String>>,
}

/// FNV-1a, enough to turn a hostname into a stable device id.
pub fn fnv1a(s: &str) -> u64 {
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
        let device_id = if config.device_id.trim().is_empty() {
            format!("cadenza-{:016x}", fnv1a(&format!("{host}|{base}")))
        } else {
            config.device_id.trim().chars().take(64).collect()
        };
        let cookie =
            (!config.cookie.trim().is_empty()).then(|| normalise_cookie(config.cookie.trim()));
        let token = (!config.token.trim().is_empty()).then(|| config.token.trim().to_string());
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
            token,
            device_id,
            device_name,
            last_error: Mutex::new(None),
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
    /// Whether requests carry a bearer token (no cookie, no login).
    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }
    /// The bearer token in use (empty without one).
    pub fn token(&self) -> &str {
        self.token.as_deref().unwrap_or("")
    }
    /// Whether this player runs inside the Encore desktop app.
    pub fn embedded(&self) -> bool {
        self.config.embedded
    }
    /// The last request failure in plain words, `None` after the last success.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().unwrap().clone()
    }

    fn set_error(&self, e: Option<String>) {
        *self.last_error.lock().unwrap() = e;
    }

    /// The `Cookie` header value, logging in first when needed.
    pub fn cookie(&self) -> Result<String, String> {
        if let Some(c) = self.cookie.lock().unwrap().clone() {
            return Ok(c);
        }
        self.login()
    }

    /// The header that authenticates a request: `Authorization: Bearer …` with a token,
    /// otherwise `Cookie: encore_session=…` (logging in first when needed).
    pub fn auth_header(&self) -> Result<(&'static str, String), String> {
        match &self.token {
            Some(t) => Ok(("Authorization", format!("Bearer {t}"))),
            None => Ok(("Cookie", self.cookie()?)),
        }
    }

    /// Request headers a track decoder must send to stream from Encore.
    pub fn stream_headers(&self) -> Vec<(String, String)> {
        match self.auth_header() {
            Ok((k, v)) => vec![(k.to_string(), v)],
            Err(_) => Vec::new(),
        }
    }

    /// `POST /api/auth/login` and remember the `encore_session` cookie.
    pub fn login(&self) -> Result<String, String> {
        if self.token.is_some() {
            return Err("the Encore token was not accepted".into());
        }
        if self.config.username.is_empty() {
            return Err("no Encore username, cookie or token configured".into());
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

    /// `POST` JSON, parse the JSON reply; a 401 triggers one re-login and retry (with a
    /// cookie; a rejected token is just reported). Failures are kept in `last_error`.
    pub fn post_json(&self, path: &str, body: &Value) -> Result<Value, String> {
        let r = self.post_json_inner(path, body);
        match &r {
            Ok(_) => self.set_error(None),
            Err(e) => self.set_error(Some(e.clone())),
        }
        r
    }

    fn post_json_inner(&self, path: &str, body: &Value) -> Result<Value, String> {
        for attempt in 0..2 {
            let (header, value) = self.auth_header()?;
            let resp = self
                .agent
                .post(&format!("{}{}", self.base, path))
                .set("X-Requested-With", "encore")
                .set("Accept", "application/json")
                .set(header, &value)
                .send_json(body.clone());
            match resp {
                Ok(r) => {
                    return r
                        .into_json::<Value>()
                        .map_err(|e| format!("bad JSON from {path}: {e}"))
                }
                Err(ureq::Error::Status(401, _)) if self.token.is_some() => {
                    return Err("Encore did not accept the token (signed out?)".into());
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

    /// `GET` with query parameters and parse the JSON reply (the library browser); a 401
    /// triggers one re-login and retry, as for [`Client::post_json`].
    pub fn get_json(&self, path: &str, query: &[(&str, &str)]) -> Result<Value, String> {
        let r = self.get(path, query).and_then(|r| {
            r.into_json::<Value>()
                .map_err(|e| format!("bad JSON from {path}: {e}"))
        });
        match &r {
            Ok(_) => self.set_error(None),
            Err(e) => self.set_error(Some(e.clone())),
        }
        r
    }

    /// `GET` a binary resource (an album cover): its content type and bytes, at most 16 MB.
    pub fn get_bytes(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<(String, Vec<u8>), String> {
        let r = self.get(path, query)?;
        let kind = r.content_type().to_string();
        let mut body = Vec::new();
        r.into_reader()
            .take(16 << 20)
            .read_to_end(&mut body)
            .map_err(|e| format!("{path}: {e}"))?;
        Ok((kind, body))
    }

    fn get(&self, path: &str, query: &[(&str, &str)]) -> Result<ureq::Response, String> {
        for attempt in 0..2 {
            let (header, value) = self.auth_header()?;
            let mut req = self
                .agent
                .get(&format!("{}{}", self.base, path))
                .set("X-Requested-With", "encore")
                .set(header, &value);
            for (k, v) in query {
                req = req.query(k, v);
            }
            match req.call() {
                Ok(r) => return Ok(r),
                Err(ureq::Error::Status(401, _)) if self.token.is_some() => {
                    return Err("Encore did not accept the token (signed out?)".into());
                }
                Err(ureq::Error::Status(401, _)) if attempt == 0 => {
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

    /// Open a long-lived `GET` (the SSE stream) and hand back its body reader. The read
    /// timeout is 60 s: the server pings every 15 s, so an idle stream stays open while a
    /// stopped link lets go of the socket within a minute.
    pub fn open_stream(&self, path: &str) -> Result<Box<dyn Read + Send + Sync + 'static>, String> {
        let (header, value) = self.auth_header()?;
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(60))
            .build();
        let resp = agent
            .get(&format!("{}{}", self.base, path))
            .set("Accept", "text/event-stream")
            .set(header, &value)
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
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

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
        assert!(!a.has_token());
    }

    #[test]
    fn device_id_override_and_token_headers() {
        let c = Client::new(Arc::new(EncoreConfig {
            url: "http://x".into(),
            token: " t0k3n ".into(),
            device_id: "cadenza-desktop-abc".into(),
            device_name: "Cadenza on box".into(),
            ..Default::default()
        }));
        assert_eq!(c.device_id(), "cadenza-desktop-abc");
        assert_eq!(c.device_name(), "Cadenza on box");
        assert!(c.has_token());
        assert_eq!(
            c.auth_header().unwrap(),
            ("Authorization", "Bearer t0k3n".to_string())
        );
        assert_eq!(
            c.stream_headers(),
            vec![("Authorization".to_string(), "Bearer t0k3n".to_string())]
        );
        // A token never logs in.
        assert!(c.login().is_err());
        let long = "x".repeat(100);
        let c = Client::new(Arc::new(EncoreConfig {
            url: "http://x".into(),
            token: "t".into(),
            device_id: long,
            ..Default::default()
        }));
        assert_eq!(c.device_id().len(), 64);
    }

    /// A one-request HTTP server on localhost; returns the request head it saw.
    fn serve_once(
        status: &'static str,
        body: &'static str,
    ) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let h = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let mut head = String::new();
            let mut len = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
                head.push_str(&line);
                if line == "\r\n" {
                    break;
                }
            }
            let mut body_bytes = vec![0u8; len];
            std::io::Read::read_exact(&mut reader, &mut body_bytes).unwrap();
            let mut stream = reader.into_inner();
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            stream.flush().unwrap();
            head
        });
        (format!("http://{addr}"), h)
    }

    #[test]
    fn post_sends_bearer_and_no_cookie() {
        let (url, server) = serve_once("200 OK", "{\"ok\":true}");
        let c = Client::new(Arc::new(EncoreConfig {
            url,
            token: "secret".into(),
            cookie: "ignored".into(),
            ..Default::default()
        }));
        let v = c
            .post_json("/api/remote/heartbeat", &serde_json::json!({}))
            .unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(c.last_error(), None);
        let head = server.join().unwrap();
        assert!(head.contains("Authorization: Bearer secret\r\n"), "{head}");
        assert!(!head.to_ascii_lowercase().contains("cookie:"), "{head}");
    }

    #[test]
    fn stream_sends_bearer() {
        let (url, server) = serve_once("200 OK", "data: x\n\n");
        let c = Client::new(Arc::new(EncoreConfig {
            url,
            token: "secret".into(),
            ..Default::default()
        }));
        let mut r = c.open_stream("/api/remote/events?device=d").unwrap();
        let mut text = String::new();
        r.read_to_string(&mut text).unwrap();
        assert_eq!(text, "data: x\n\n");
        let head = server.join().unwrap();
        assert!(head.contains("Authorization: Bearer secret\r\n"), "{head}");
        assert!(!head.to_ascii_lowercase().contains("cookie:"), "{head}");
    }

    #[test]
    fn a_rejected_token_is_an_error_not_a_login() {
        let (url, server) = serve_once("401 Unauthorized", "{\"error\":\"no\"}");
        let c = Client::new(Arc::new(EncoreConfig {
            url,
            token: "old".into(),
            username: "u".into(),
            password: "p".into(),
            ..Default::default()
        }));
        let e = c
            .post_json("/api/remote/heartbeat", &serde_json::json!({}))
            .unwrap_err();
        assert!(e.contains("token"), "{e}");
        assert_eq!(c.last_error().as_deref(), Some(e.as_str()));
        // Exactly one request reached the server (no login attempt follows).
        let head = server.join().unwrap();
        assert!(head.starts_with("POST /api/remote/heartbeat"), "{head}");
    }
}
