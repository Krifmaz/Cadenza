//! The `state` object sent with every heartbeat: what Encore's device list, remote and
//! signal-path dialogs display for this device.

use super::Client;
use crate::config::RateSetting;
use crate::engine::{PlayState, Status, Track};
use cadenza_dsp::signal_path::rate_label;
use cadenza_dsp::{DitherKind, FilterKind};
use serde_json::{json, Value};

/// A queue entry as Encore shows it.
pub fn slim_track(t: &Track) -> Value {
    json!({
        "id": t.id.as_deref().and_then(|s| s.parse::<i64>().ok()).map(Value::from).unwrap_or(Value::Null),
        "title": t.title,
        "artist": t.artist,
        "album": t.album,
        "album_id": t.album_id,
        "format": t.format,
        "duration": t.duration,
    })
}

/// Build the heartbeat state.
pub fn heartbeat_state(client: &Client, s: &Status) -> Value {
    let track = s.track.as_ref();
    let album_id = track.and_then(|t| t.album_id.clone());
    let cover = track.and_then(|t| t.cover.clone()).or_else(|| {
        album_id
            .as_ref()
            .map(|a| format!("/api/library/albums/{a}/cover"))
    });
    let format = track
        .and_then(|t| t.format.clone())
        .or_else(|| s.source_format.clone());
    let signal = s.signal_path.as_ref();
    let rates: Vec<Value> = {
        let mut v = vec![
            json!(["auto", "Auto (highest)"]),
            json!(["source", "Source rate"]),
        ];
        for r in &s.output_rates {
            v.push(json!([r, format!("{} kHz", rate_label(*r))]));
        }
        v
    };
    let rate: Value = match s.settings.output.rate {
        RateSetting::Auto => json!("auto"),
        RateSetting::Source => json!("source"),
        RateSetting::Fixed(r) => json!(r),
    };
    let output = match s.output_format {
        Some(f) => format!("{} · {}", s.output, f),
        None => s.output.clone(),
    };
    let _ = client;
    json!({
        "id": track.and_then(|t| t.id.as_deref()).and_then(|i| i.parse::<i64>().ok()),
        "title": track.map(|t| t.title.clone()).unwrap_or_default(),
        "artist": track.map(|t| t.artist.clone()).unwrap_or_default(),
        "album": track.map(|t| t.album.clone()).unwrap_or_default(),
        "album_id": album_id,
        "format": format,
        "cover": cover,
        "playing": s.state == PlayState::Playing,
        "paused": s.state == PlayState::Paused,
        "position": s.position,
        "duration": s.duration,
        "volume": s.volume_linear,
        "index": s.index,
        "queue": s.queue.iter().map(slim_track).collect::<Vec<_>>(),
        "signal": {
            "label": signal.map(|p| p.label.clone()).unwrap_or_default(),
            "transport": "original",
            "output_rate": s.output_format.map(|f| f.rate),
            "bitperfect": signal.map(|p| p.bit_perfect).unwrap_or(false),
            "stages": signal.map(|p| p.stages.iter().filter(|st| st.active).map(|st| json!({ "name": st.name, "detail": st.detail })).collect::<Vec<_>>()).unwrap_or_default(),
        },
        "engine": {
            "name": "Cadenza",
            "version": env!("CARGO_PKG_VERSION"),
            "bitperfect": s.settings.dsp.bit_perfect,
            "filter": s.settings.dsp.filter,
            "filters": FilterKind::catalog().iter().map(|k| json!([k.id(), k.label()])).collect::<Vec<_>>(),
            "dither": s.settings.dsp.dither,
            "dithers": DitherKind::catalog().iter().map(|d| json!([d.id(), d.label()])).collect::<Vec<_>>(),
            "rate": rate,
            "rates": rates,
            "bits": s.settings.output.bits,
            "output": output,
            "error": s.error,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EncoreConfig, Settings};
    use std::sync::Arc;

    #[test]
    fn state_has_the_fields_encore_reads() {
        let client = Client::new(Arc::new(EncoreConfig {
            url: "http://e".into(),
            cookie: "c".into(),
            ..Default::default()
        }));
        let status = Status {
            state: PlayState::Playing,
            track: Some(Track {
                id: Some("42".into()),
                album_id: Some("7".into()),
                title: "T".into(),
                ..Default::default()
            }),
            index: Some(0),
            position: 12.0,
            settings: Settings::default(),
            output_rates: vec![44100, 88200],
            ..Default::default()
        };
        let v = heartbeat_state(&client, &status);
        assert_eq!(v["id"], 42);
        assert_eq!(v["cover"], "/api/library/albums/7/cover");
        assert_eq!(v["playing"], true);
        assert_eq!(v["engine"]["name"], "Cadenza");
        assert_eq!(v["engine"]["rate"], "auto");
        assert_eq!(v["engine"]["rates"].as_array().unwrap().len(), 4);
        assert!(v["engine"]["filters"].as_array().unwrap().len() > 10);
        assert_eq!(v["signal"]["transport"], "original");
    }
}
