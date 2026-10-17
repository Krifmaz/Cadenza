//! Cover art for a track's `cover` field, straight from where the engine keeps it: an
//! Encore album (`/api/encore/cover/{id}`), a live stream's announced art
//! (`/api/live/{id}/cover`), a web address, or a file.

pub fn fetch(url: &str) -> Option<Vec<u8>> {
    let path = url.split('?').next().unwrap_or(url);
    if let Some(id) = path.strip_prefix("/api/encore/cover/") {
        let c = cadenza::encore::client()?;
        return cadenza::encore::library::cover(&c, id, 600).ok().map(|(_, b)| b);
    }
    if let Some(rest) = path.strip_prefix("/api/live/") {
        let id = rest.strip_suffix("/cover")?;
        let s = cadenza::decode::live::lookup(id)?;
        return s.cover().map(|(_, b)| b.to_vec());
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let r = ureq::get(url).timeout(std::time::Duration::from_secs(10)).call().ok()?;
        let mut v = Vec::new();
        std::io::Read::read_to_end(&mut r.into_reader().take(20 << 20), &mut v).ok()?;
        return Some(v);
    }
    let p = url.strip_prefix("file://").unwrap_or(url);
    std::fs::read(p).ok()
}

use std::io::Read as _;
