//! The HTTP control API (JSON + Server-Sent Events) and the single-file web UI.
//!
//! Everything is a thin layer over [`Engine`]: reads come from the latest status snapshot,
//! writes are commands. The UI lives in `static/index.html` and is embedded in the binary.

use crate::engine::{Command, Engine, Event, Track};
use crate::output::list_devices;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cadenza_dsp::{DitherKind, FilterKind, Volume};
use futures_util::stream::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;

/// The embedded web UI.
pub const INDEX_HTML: &str = include_str!("../../static/index.html");

#[derive(Clone)]
struct AppState {
    engine: Engine,
}

/// A JSON error with a status code.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

/// Build the router.
pub fn router(engine: Engine) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/status", get(status))
        .route(
            "/api/settings",
            get(get_settings).put(put_settings).post(put_settings),
        )
        .route("/api/devices", get(devices))
        .route("/api/filters", get(filters))
        .route("/api/dithers", get(dithers))
        .route("/api/queue", post(queue))
        .route("/api/control", post(control))
        .route("/api/events", get(events))
        .with_state(AppState { engine })
}

/// Run the server on `listen` until the process ends.
pub async fn serve(listen: &str, engine: Engine) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(listen).await?;
    log::info!(
        "control API and web UI on http://{}/",
        listener.local_addr()?
    );
    axum::serve(listener, router(engine)).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn status(State(st): State<AppState>) -> Json<Value> {
    Json(serde_json::to_value(&*st.engine.status()).unwrap_or(Value::Null))
}

async fn get_settings(State(st): State<AppState>) -> Json<Value> {
    Json(serde_json::to_value(st.engine.settings()).unwrap_or(Value::Null))
}

/// Accepts a full `Settings` object or any subset of its fields (deep-merged).
async fn put_settings(
    State(st): State<AppState>,
    Json(patch): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let mut current = serde_json::to_value(st.engine.settings()).map_err(|e| bad(e.to_string()))?;
    merge(&mut current, &patch);
    let settings: crate::config::Settings =
        serde_json::from_value(current.clone()).map_err(|e| bad(format!("bad settings: {e}")))?;
    if !matches!(settings.output.bits, 16 | 24 | 32) {
        return Err(bad("bits must be 16, 24 or 32"));
    }
    if FilterKind::parse(&settings.dsp.filter).is_none() {
        return Err(bad(format!("unknown filter {:?}", settings.dsp.filter)));
    }
    if DitherKind::parse(&settings.dsp.dither).is_none() {
        return Err(bad(format!("unknown dither {:?}", settings.dsp.dither)));
    }
    st.engine.update_settings(settings);
    Ok(Json(current))
}

/// Recursive JSON merge: objects merge key by key, everything else is replaced.
pub fn merge(into: &mut Value, patch: &Value) {
    match (into, patch) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                match a.get_mut(k) {
                    Some(slot) if slot.is_object() && v.is_object() => merge(slot, v),
                    _ => {
                        a.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (a, b) => *a = b.clone(),
    }
}

async fn devices() -> Json<Value> {
    // Device probing opens devices; keep it off the async threads.
    let list = tokio::task::spawn_blocking(list_devices)
        .await
        .unwrap_or_default();
    Json(json!(list))
}

async fn filters() -> Json<Value> {
    Json(json!(FilterKind::catalog()
        .iter()
        .map(|k| json!({ "id": k.id(), "label": k.label(), "taps": k.taps, "phase": k.phase.name(), "apodizing": k.apodizing }))
        .collect::<Vec<_>>()))
}

async fn dithers() -> Json<Value> {
    Json(json!(DitherKind::catalog()
        .iter()
        .map(|d| json!({ "id": d.id(), "label": d.label(), "design_rate": d.design_rate(), "shaper": d.coefficients() }))
        .collect::<Vec<_>>()))
}

/// One item of a `POST /api/queue` body.
#[derive(Deserialize)]
struct QueueItem {
    url: Option<String>,
    path: Option<String>,
    #[serde(default)]
    headers: Vec<(String, String)>,
    id: Option<Value>,
    #[serde(default)]
    title: String,
    #[serde(default)]
    artist: String,
    #[serde(default)]
    album: String,
    album_id: Option<Value>,
    duration: Option<f64>,
    format: Option<String>,
    cover: Option<String>,
}

#[derive(Deserialize)]
struct QueueBody {
    #[serde(default)]
    items: Vec<QueueItem>,
    #[serde(default)]
    index: usize,
    #[serde(default)]
    position: f64,
    /// `replace` (default, and play), `append`, `next`.
    #[serde(default)]
    mode: Option<String>,
}

impl QueueItem {
    fn into_track(self) -> Result<Track, ApiError> {
        let source = match (self.url, self.path) {
            (Some(u), _) => crate::decode::Source::Url {
                url: u,
                headers: self.headers,
            },
            (None, Some(p)) => crate::decode::Source::Path { path: p.into() },
            (None, None) => return Err(bad("each item needs a url or a path")),
        };
        let title = if self.title.is_empty() {
            source
                .display()
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or("")
                .to_string()
        } else {
            self.title
        };
        Ok(Track {
            id: self.id.map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| v.to_string())
            }),
            source,
            title,
            artist: self.artist,
            album: self.album,
            album_id: self.album_id.map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| v.to_string())
            }),
            duration: self.duration,
            format: self.format,
            cover: self.cover,
        })
    }
}

async fn queue(
    State(st): State<AppState>,
    Json(body): Json<QueueBody>,
) -> Result<Json<Value>, ApiError> {
    let tracks: Vec<Track> = body
        .items
        .into_iter()
        .map(QueueItem::into_track)
        .collect::<Result<_, _>>()?;
    match body.mode.as_deref().unwrap_or("replace") {
        "append" | "enqueue" => st.engine.send(Command::Enqueue(tracks)),
        "next" | "play_next" => st.engine.send(Command::PlayNext(tracks)),
        _ => st.engine.play_queue(tracks, body.index, body.position),
    }
    Ok(Json(json!({ "ok": true })))
}

#[derive(Deserialize)]
struct ControlBody {
    action: String,
    #[serde(default)]
    value: Option<Value>,
}

/// Translate a control action (shared with the Encore command handler).
pub fn control_command(action: &str, value: Option<&Value>) -> Result<Command, String> {
    let num = || {
        value
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("{action} needs a numeric value"))
    };
    Ok(match action {
        "play" => Command::Play,
        "pause" => Command::Pause,
        "toggle" => Command::Toggle,
        "stop" => Command::Stop,
        "next" => Command::Next,
        "prev" | "previous" => Command::Prev,
        "seek" => Command::Seek(num()?),
        "volume" => Command::SetVolume(Volume::from_linear(num()?)),
        "volume_db" => Command::SetVolume(Volume {
            gain_db: num()?,
            muted: false,
        }),
        "mute" => Command::SetVolume(Volume {
            gain_db: 0.0,
            muted: true,
        }),
        "play_index" => Command::PlayIndex(num()? as usize),
        "remove" => Command::Remove(num()? as usize),
        other => return Err(format!("unknown action {other:?}")),
    })
}

async fn control(
    State(st): State<AppState>,
    Json(body): Json<ControlBody>,
) -> Result<Json<Value>, ApiError> {
    let cmd = control_command(&body.action, body.value.as_ref()).map_err(bad)?;
    st.engine.send(cmd);
    // Give the worker a moment so the reply reflects the command.
    tokio::time::sleep(Duration::from_millis(30)).await;
    Ok(Json(
        serde_json::to_value(&*st.engine.status()).unwrap_or(Value::Null),
    ))
}

async fn events(
    State(st): State<AppState>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let first = SseEvent::default()
        .event("status")
        .json_data(&*st.engine.status())
        .unwrap_or_default();
    let live = BroadcastStream::new(st.engine.subscribe()).filter_map(|ev| async move {
        match ev {
            Ok(Event::Status(s)) => Some(Ok(SseEvent::default()
                .event("status")
                .json_data(&*s)
                .unwrap_or_default())),
            Ok(Event::TrackStarted(t)) => Some(Ok(SseEvent::default()
                .event("track")
                .json_data(&t)
                .unwrap_or_default())),
            _ => None,
        }
    });
    let stream = futures_util::stream::once(async move { Ok(first) }).chain(live);
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_is_deep() {
        let mut a = json!({ "output": { "bits": 24, "rate": "auto" }, "dsp": { "filter": "x" } });
        merge(
            &mut a,
            &json!({ "output": { "bits": 16 }, "dsp": { "eq": [1] } }),
        );
        assert_eq!(
            a,
            json!({ "output": { "bits": 16, "rate": "auto" }, "dsp": { "filter": "x", "eq": [1] } })
        );
    }

    #[test]
    fn control_actions_parse() {
        assert!(
            matches!(control_command("seek", Some(&json!(12.5))), Ok(Command::Seek(t)) if t == 12.5)
        );
        assert!(
            matches!(control_command("volume", Some(&json!(1.0))), Ok(Command::SetVolume(v)) if v.gain_db == 0.0)
        );
        assert!(control_command("seek", None).is_err());
        assert!(control_command("dance", None).is_err());
    }
}
