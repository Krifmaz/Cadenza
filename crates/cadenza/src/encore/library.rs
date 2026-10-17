//! The Encore library inside Cadenza: browse and search the signed-in server and play its
//! tracks through this player. Requests go through the server-side [`Client`], so the
//! session cookie or token never reaches the web page; covers come through
//! `/api/encore/cover/{album}` for the same reason.

use super::{commands::track_from_encore, Client};
use crate::engine::{Command, Engine, Track};
use serde_json::Value;
use std::sync::Arc;

/// Albums, newest first unless a search is given (`GET /api/library/albums`).
pub fn albums(c: &Client, q: &str, offset: u32, limit: u32) -> Result<Value, String> {
    let (offset, limit) = (offset.to_string(), limit.clamp(1, 500).to_string());
    let sort = if q.is_empty() { "added" } else { "name" };
    let dir = if q.is_empty() { "desc" } else { "asc" };
    c.get_json(
        "/api/library/albums",
        &[
            ("q", q),
            ("offset", &offset),
            ("limit", &limit),
            ("sort", sort),
            ("dir", dir),
        ],
    )
}

/// One album with its tracks (`GET /api/library/albums/{id}`).
pub fn album(c: &Client, id: &str) -> Result<Value, String> {
    if id.is_empty()
        || !id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err("bad album id".into());
    }
    c.get_json(&format!("/api/library/albums/{id}"), &[])
}

/// Tracks matching a search (`GET /api/library/tracks?q=`).
pub fn tracks(c: &Client, q: &str, limit: u32) -> Result<Value, String> {
    let limit = limit.clamp(1, 1000).to_string();
    c.get_json("/api/library/tracks", &[("q", q), ("limit", &limit)])
}

/// An album cover, scaled by Encore to about `size` pixels (0 = original).
pub fn cover(c: &Client, id: &str, size: u32) -> Result<(String, Vec<u8>), String> {
    if id.is_empty()
        || !id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err("bad album id".into());
    }
    let s = size.to_string();
    c.get_bytes(&format!("/api/library/albums/{id}/cover"), &[("s", &s)])
}

/// Encore track objects as queue entries that stream from the server. The cover points at
/// this player's cover proxy, which the page can load without Encore's credentials.
pub fn queue_entries(c: &Client, items: &[Value]) -> Vec<Track> {
    items
        .iter()
        .filter_map(|v| track_from_encore(c, v))
        .map(|mut t| {
            if let Some(a) = &t.album_id {
                t.cover = Some(format!("/api/encore/cover/{a}"));
            }
            t
        })
        .collect()
}

/// Play Encore tracks here: `replace` (play from `index`), `append` or `next`.
pub fn play(
    c: &Arc<Client>,
    engine: &Engine,
    items: &[Value],
    index: usize,
    mode: &str,
) -> Result<usize, String> {
    let tracks = queue_entries(c, items);
    if tracks.is_empty() {
        return Err("no playable Encore tracks".into());
    }
    let n = tracks.len();
    match mode {
        "append" | "enqueue" => engine.send(Command::Enqueue(tracks)),
        "next" | "play_next" => engine.send(Command::PlayNext(tracks)),
        _ => engine.play_queue(tracks, index.min(n - 1), 0.0),
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EncoreConfig;

    fn client() -> Client {
        Client::new(Arc::new(EncoreConfig {
            url: "http://e:8080".into(),
            token: "t".into(),
            ..Default::default()
        }))
    }

    #[test]
    fn library_tracks_stream_from_encore_with_a_local_cover() {
        let c = client();
        let items = vec![
            serde_json::json!({ "id": 5, "title": "A", "artist": "B", "album": "C", "album_id": "ab12", "duration": 10.0 }),
            serde_json::json!({ "title": "no id" }),
        ];
        let t = queue_entries(&c, &items);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].cover.as_deref(), Some("/api/encore/cover/ab12"));
        match &t[0].source {
            crate::decode::Source::Url { url, headers } => {
                assert_eq!(url, "http://e:8080/api/library/tracks/5/stream");
                assert!(headers
                    .iter()
                    .any(|(k, v)| k == "Authorization" && v == "Bearer t"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn album_ids_are_checked_before_they_reach_a_url() {
        let c = client();
        assert!(album(&c, "../x").is_err());
        assert!(cover(&c, "a/b", 0).is_err());
        assert!(album(&c, "").is_err());
    }
}
