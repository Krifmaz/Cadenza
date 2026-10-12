//! Commands Encore sends to a device, and how they map onto the engine.

use super::Client;
use crate::api::control_command;
use crate::config::RateSetting;
use crate::engine::{Command, Engine, Track};
use cadenza_dsp::{DitherKind, FilterKind};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Sequence numbers already executed (a command can arrive pushed *and* with a heartbeat).
static SEEN: Mutex<VecDeque<u64>> = Mutex::new(VecDeque::new());

fn seen_before(seq: u64) -> bool {
    let mut seen = SEEN.lock().unwrap();
    if seen.contains(&seq) {
        return true;
    }
    seen.push_back(seq);
    if seen.len() > 200 {
        seen.pop_front();
    }
    false
}

/// An Encore library track (`{id, title, artist, album, album_id, format, duration}`) as a
/// queue entry streaming from the server with the session cookie.
pub fn track_from_encore(client: &Client, v: &Value) -> Option<Track> {
    let id = v.get("id").and_then(|i| {
        i.as_i64()
            .or_else(|| i.as_str().and_then(|s| s.parse().ok()))
    })?;
    // Encore album ids are strings (hex); accept numbers too
    let album_id = v
        .get("album_id")
        .and_then(|a| {
            a.as_str()
                .map(str::to_string)
                .or_else(|| a.as_i64().map(|n| n.to_string()))
        })
        .filter(|a| !a.is_empty());
    let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    Some(Track {
        id: Some(id.to_string()),
        source: crate::decode::Source::Url {
            url: client.track_url(id),
            headers: client.stream_headers(),
        },
        title: s("title"),
        artist: s("artist"),
        album: s("album"),
        album_id: album_id.clone(),
        duration: v.get("duration").and_then(Value::as_f64),
        format: v.get("format").and_then(Value::as_str).map(str::to_string),
        cover: album_id.as_deref().map(|a| client.cover_url(a)),
    })
}

fn tracks(client: &Client, v: &Value) -> Vec<Track> {
    v.get("tracks")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|t| track_from_encore(client, t))
                .collect()
        })
        .unwrap_or_default()
}

/// Run one command object (`{action, seq?, …}`).
pub fn run(client: &Client, engine: &Engine, cmd: &Value) {
    if let Some(seq) = cmd.get("seq").and_then(Value::as_u64) {
        if seen_before(seq) {
            return;
        }
    }
    let action = cmd.get("action").and_then(Value::as_str).unwrap_or("");
    log::info!("Encore command: {action}");
    let value = cmd
        .get("value")
        .or_else(|| cmd.get("seconds"))
        .or_else(|| cmd.get("level"));
    match action {
        "play_queue" => {
            let list = tracks(client, cmd);
            if list.is_empty() {
                return;
            }
            let index = cmd.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            let position = cmd.get("position").and_then(Value::as_f64).unwrap_or(0.0);
            engine.play_queue(list, index, position);
        }
        "enqueue" => engine.send(Command::Enqueue(tracks(client, cmd))),
        "play_next" => engine.send(Command::PlayNext(tracks(client, cmd))),
        "engine" => {
            if let Some(set) = cmd.get("set").and_then(Value::as_object) {
                apply_engine_settings(engine, set);
            }
        }
        other => match control_command(other, value) {
            Ok(c) => engine.send(c),
            Err(e) => log::warn!("Encore: {e}"),
        },
    }
}

/// `engine {set: {bitperfect|filter|dither|rate|bits: value}}`.
pub fn apply_engine_settings(engine: &Engine, set: &serde_json::Map<String, Value>) {
    let mut s = engine.settings();
    for (k, v) in set {
        match k.as_str() {
            "bitperfect" | "bit_perfect" => {
                s.dsp.bit_perfect = v
                    .as_bool()
                    .or_else(|| v.as_str().map(|t| t == "true" || t == "1"))
                    .unwrap_or(false)
            }
            "filter" => {
                if let Some(id) = v.as_str().filter(|id| FilterKind::parse(id).is_some()) {
                    s.dsp.filter = id.to_string();
                } else {
                    log::warn!("Encore: unknown filter {v}");
                }
            }
            "dither" => {
                if let Some(id) = v.as_str().filter(|id| DitherKind::parse(id).is_some()) {
                    s.dsp.dither = id.to_string();
                } else {
                    log::warn!("Encore: unknown dither {v}");
                }
            }
            "rate" => {
                let parsed = match v {
                    Value::Number(n) => n.as_u64().map(|r| RateSetting::Fixed(r as u32)),
                    Value::String(t) => RateSetting::parse(t),
                    _ => None,
                };
                match parsed {
                    Some(r) => s.output.rate = r,
                    None => log::warn!("Encore: bad rate {v}"),
                }
            }
            "bits" => {
                if let Some(b) = v.as_u64().filter(|b| matches!(b, 16 | 24 | 32)) {
                    s.output.bits = b as u32;
                }
            }
            other => log::debug!("Encore: ignoring engine setting {other}"),
        }
    }
    engine.update_settings(s);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EncoreConfig;
    use std::sync::Arc;

    #[test]
    fn encore_tracks_become_streaming_queue_entries() {
        let client = Client::new(Arc::new(EncoreConfig {
            url: "http://e:8080".into(),
            cookie: "abc".into(),
            ..Default::default()
        }));
        let t = track_from_encore(
            &client,
            &serde_json::json!({ "id": 12, "title": "Song", "artist": "Band", "album": "LP", "album_id": "cc5d7f35", "format": "FLAC", "duration": 200.5 }),
        )
        .unwrap();
        assert_eq!(t.id.as_deref(), Some("12"));
        assert_eq!(
            t.cover.as_deref(),
            Some("http://e:8080/api/library/albums/cc5d7f35/cover")
        );
        match &t.source {
            crate::decode::Source::Url { url, headers } => {
                assert_eq!(url, "http://e:8080/api/library/tracks/12/stream");
                assert_eq!(
                    headers,
                    &vec![("Cookie".to_string(), "encore_session=abc".to_string())]
                );
            }
            _ => panic!("expected a URL"),
        }
        assert!(track_from_encore(&client, &serde_json::json!({ "title": "no id" })).is_none());
    }

    #[test]
    fn duplicate_sequence_numbers_run_once() {
        assert!(!seen_before(900_001));
        assert!(seen_before(900_001));
        assert!(!seen_before(900_002));
    }
}
