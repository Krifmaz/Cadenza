//! Cadenza as an Encore playback device.
//!
//! * log in (or use a given `encore_session` cookie),
//! * every 2 s `POST /api/remote/heartbeat` with the player state and run the commands
//!   that come back,
//! * keep `GET /api/remote/events?device=…` open as an SSE stream so commands arrive the
//!   moment they are sent (the heartbeat stays as the fallback; a command's `seq` makes
//!   sure it runs once even when it arrives both ways),
//! * play Encore tracks from `/api/library/tracks/{id}/stream` with the session cookie,
//! * report plays like Encore's own player (`now-playing` when a track starts, `plays`
//!   after half the song or four minutes).

mod client;
mod commands;
mod state;

pub use client::Client;

use crate::config::EncoreConfig;
use crate::engine::{Engine, Event};
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Start the three Encore threads (heartbeat, event stream, play reporting).
pub fn start(config: Arc<EncoreConfig>, engine: Engine) {
    let client = Arc::new(Client::new(config));
    log::info!(
        "Encore device {} ({}) → {}",
        client.device_name(),
        client.device_id(),
        client.base()
    );
    {
        let (c, e) = (Arc::clone(&client), engine.clone());
        thread::Builder::new()
            .name("encore-heartbeat".into())
            .spawn(move || heartbeat_loop(c, e))
            .expect("spawn");
    }
    {
        let (c, e) = (Arc::clone(&client), engine.clone());
        thread::Builder::new()
            .name("encore-events".into())
            .spawn(move || events_loop(c, e))
            .expect("spawn");
    }
    thread::Builder::new()
        .name("encore-plays".into())
        .spawn(move || plays_loop(client, engine))
        .expect("spawn");
}

fn heartbeat_loop(client: Arc<Client>, engine: Engine) {
    let mut failures = 0u32;
    loop {
        let status = engine.status();
        let body = serde_json::json!({
            "id": client.device_id(),
            "name": client.device_name(),
            "state": state::heartbeat_state(&client, &status),
        });
        match client.post_json("/api/remote/heartbeat", &body) {
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
            }
            Err(e) => {
                failures += 1;
                if failures == 1 || failures % 30 == 0 {
                    log::warn!("Encore heartbeat failed: {e}");
                }
            }
        }
        thread::sleep(Duration::from_secs(2));
    }
}

/// Keep the command stream open; reconnect after a pause whenever it drops.
fn events_loop(client: Arc<Client>, engine: Engine) {
    loop {
        match client.open_stream(&format!("/api/remote/events?device={}", client.device_id())) {
            Ok(reader) => {
                log::info!("Encore event stream open");
                read_sse(BufReader::new(reader), |event, data| {
                    if event == "command" {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            if let Some(cmd) = v.get("command") {
                                commands::run(&client, &engine, cmd);
                            }
                        }
                    }
                });
                log::warn!("Encore event stream closed; reconnecting");
            }
            Err(e) => log::warn!("Encore event stream: {e}"),
        }
        thread::sleep(Duration::from_secs(2));
    }
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
fn plays_loop(client: Arc<Client>, engine: Engine) {
    let mut rx = engine.subscribe();
    loop {
        match rx.blocking_recv() {
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
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
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
}
