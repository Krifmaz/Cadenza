//! Where audio comes from: a local file or an HTTP(S) URL.
//!
//! The HTTP source implements Symphonia's `MediaSource` on top of blocking `ureq`
//! requests. Seeking is done lazily with `Range: bytes=<pos>-`: a seek only records the new
//! position, and the next read opens a fresh request there. That is exactly what a
//! container parser needs (a handful of seeks while probing, then long sequential reads).

use serde::{Deserialize, Serialize};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Duration;
use symphonia::core::io::MediaSource;

/// A track's location.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Source {
    /// A local file.
    Path {
        /// Path on disk.
        path: PathBuf,
    },
    /// Something to fetch over HTTP(S).
    Url {
        /// The URL.
        url: String,
        /// Extra request headers (`Cookie`, `Authorization`…).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        headers: Vec<(String, String)>,
    },
    /// PCM pushed in by a receiver in this process (AirPlay); see [`super::live`].
    Live {
        /// The registered stream's id.
        live: String,
    },
}

impl Source {
    /// Interpret a CLI argument: URLs start with `http://` or `https://`, everything else is a path.
    pub fn parse(s: &str) -> Source {
        if s.starts_with("http://") || s.starts_with("https://") {
            Source::Url {
                url: s.to_string(),
                headers: Vec::new(),
            }
        } else {
            Source::Path {
                path: PathBuf::from(s),
            }
        }
    }

    /// A display string (the path or the URL without its query string).
    pub fn display(&self) -> String {
        match self {
            Source::Path { path } => path.display().to_string(),
            Source::Url { url, .. } => url.split('?').next().unwrap_or(url).to_string(),
            Source::Live { live } => format!("live:{live}"),
        }
    }

    /// File extension (from the path or the URL's last segment) to hint the container format.
    pub fn extension(&self) -> Option<String> {
        let name = match self {
            Source::Path { path } => path.file_name()?.to_string_lossy().to_string(),
            Source::Url { url, .. } => url
                .split(['?', '#'])
                .next()?
                .rsplit('/')
                .next()?
                .to_string(),
            Source::Live { .. } => return None,
        };
        let (_, ext) = name.rsplit_once('.')?;
        (!ext.is_empty() && ext.len() <= 5).then(|| ext.to_ascii_lowercase())
    }

    /// Open as a Symphonia media source.
    pub fn open(&self) -> io::Result<(Box<dyn MediaSource>, Option<String>)> {
        match self {
            Source::Path { path } => Ok((Box::new(std::fs::File::open(path)?), None)),
            Source::Url { url, headers } => {
                let src = HttpSource::open(url, headers)?;
                let mime = src.content_type.clone();
                Ok((Box::new(src), mime))
            }
            Source::Live { live } => Err(io::Error::other(format!(
                "live stream {live} is not a byte source"
            ))),
        }
    }
}

/// A seekable, blocking HTTP reader.
pub struct HttpSource {
    url: String,
    headers: Vec<(String, String)>,
    agent: ureq::Agent,
    len: Option<u64>,
    ranges: bool,
    pos: u64,
    body: Option<Box<dyn Read + Send + Sync>>,
    /// The `Content-Type` of the first response, to hint the container.
    pub content_type: Option<String>,
}

impl HttpSource {
    /// Issue the first request (from byte 0) and learn the length and range support.
    pub fn open(url: &str, headers: &[(String, String)]) -> io::Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(15))
            .timeout_read(Duration::from_secs(30))
            .build();
        let mut src = Self {
            url: url.to_string(),
            headers: headers.to_vec(),
            agent,
            len: None,
            ranges: false,
            pos: 0,
            body: None,
            content_type: None,
        };
        src.connect()?;
        Ok(src)
    }

    fn connect(&mut self) -> io::Result<()> {
        let mut req = self
            .agent
            .get(&self.url)
            .set("Range", &format!("bytes={}-", self.pos));
        for (k, v) in &self.headers {
            req = req.set(k, v);
        }
        let resp = match req.call() {
            Ok(r) => r,
            Err(ureq::Error::Status(416, _)) if self.pos > 0 => {
                // Past the end: a read here is simply EOF.
                self.body = Some(Box::new(io::empty()));
                return Ok(());
            }
            Err(ureq::Error::Status(code, r)) => {
                return Err(io::Error::other(format!(
                    "HTTP {code} {} for {}",
                    r.status_text(),
                    self.url
                )));
            }
            Err(e) => return Err(io::Error::other(e.to_string())),
        };
        if self.content_type.is_none() {
            self.content_type = resp
                .header("content-type")
                .map(|s| s.split(';').next().unwrap_or(s).trim().to_string());
        }
        match resp.status() {
            206 => {
                self.ranges = true;
                // Content-Range: bytes <from>-<to>/<total>
                if let Some(total) = resp
                    .header("content-range")
                    .and_then(|cr| cr.rsplit('/').next())
                    .and_then(|t| t.parse().ok())
                {
                    self.len = Some(total);
                }
            }
            _ => {
                // The server ignored the range: we got the whole thing from byte 0.
                if self.pos != 0 {
                    // Skip ahead by reading; slow, but correct.
                    let mut r = resp.into_reader();
                    io::copy(&mut (&mut r).take(self.pos), &mut io::sink())?;
                    self.body = Some(r);
                    return Ok(());
                }
                self.ranges = resp
                    .header("accept-ranges")
                    .is_some_and(|v| v.contains("bytes"));
                if let Some(l) = resp.header("content-length").and_then(|v| v.parse().ok()) {
                    self.len = Some(l);
                }
            }
        }
        self.body = Some(resp.into_reader());
        Ok(())
    }
}

impl Read for HttpSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.body.is_none() {
            if let Some(len) = self.len {
                if self.pos >= len {
                    return Ok(0);
                }
            }
            self.connect()?;
        }
        let n = match self.body.as_mut().unwrap().read(buf) {
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => return self.read(buf),
            Err(e) => {
                // A dropped connection: reopen at the current position once.
                log::warn!("http read failed ({e}); reconnecting at byte {}", self.pos);
                self.body = None;
                self.connect()?;
                self.body.as_mut().unwrap().read(buf)?
            }
        };
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for HttpSource {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let new = match to {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::Current(d) => self.pos as i64 + d,
            SeekFrom::End(d) => match self.len {
                Some(l) => l as i64 + d,
                None => return Err(io::Error::new(io::ErrorKind::Unsupported, "length unknown")),
            },
        };
        if new < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before start",
            ));
        }
        if new as u64 != self.pos {
            self.pos = new as u64;
            self.body = None;
        }
        Ok(self.pos)
    }
}

impl MediaSource for HttpSource {
    fn is_seekable(&self) -> bool {
        self.ranges && self.len.is_some()
    }

    fn byte_len(&self) -> Option<u64> {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sources_and_extensions() {
        let p = Source::parse("/music/a.flac");
        assert_eq!(p.extension().as_deref(), Some("flac"));
        let u = Source::parse("https://x.org/api/library/tracks/12/stream?x=1");
        assert!(matches!(u, Source::Url { .. }));
        assert_eq!(u.extension(), None);
        assert_eq!(
            Source::parse("http://h/a/b.MP3").extension().as_deref(),
            Some("mp3")
        );
        assert_eq!(u.display(), "https://x.org/api/library/tracks/12/stream");
    }
}
