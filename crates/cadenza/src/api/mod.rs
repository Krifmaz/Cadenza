//! The HTTP control API (JSON + Server-Sent Events) and the single-file web UI.
//!
//! Everything is a thin layer over [`Engine`]: reads come from the latest status snapshot,
//! writes are commands. The UI lives in `static/index.html` and is embedded in the binary.

use crate::engine::{Command, Engine, Event, Track};
use crate::output::list_devices;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cadenza_dsp::dsd::{DsdModulator, DsdOversampling, DsdPacking, DsdRate, OutputMode};
use cadenza_dsp::gpu::{Backend as GpuBackend, GpuMode};
use cadenza_dsp::matrix::CrossoverKind;
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
        .route("/api/options", get(options_route))
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
    validate(&settings).map_err(bad)?;
    st.engine.update_settings(settings);
    Ok(Json(current))
}

/// Checks beyond what deserialising enforces (shared with the Encore command handler).
pub fn validate(settings: &crate::config::Settings) -> Result<(), String> {
    let d = &settings.dsp;
    if !matches!(settings.output.bits, 16 | 24 | 32) {
        return Err("bits must be 16, 24 or 32".into());
    }
    if FilterKind::parse(&d.filter).is_none() {
        return Err(format!("unknown filter {:?}", d.filter));
    }
    if DitherKind::parse(&d.dither).is_none() {
        return Err(format!("unknown dither {:?}", d.dither));
    }
    if !(-30.0..=6.0).contains(&d.dsd_gain_db) {
        return Err("dsd_gain_db must be between -30 and +6 dB".into());
    }
    if d.dsd_packing == DsdPacking::Dop && !d.dsd_rate.dop_possible() {
        return Err(format!(
            "{} cannot be carried over DoP; choose native packing",
            d.dsd_rate.label()
        ));
    }
    if let Some(h) = d.headroom_db {
        if !(-30.0..=0.0).contains(&h) {
            return Err("headroom_db must be between -30 and 0 dB (or null for automatic)".into());
        }
    }
    if !(-60.0..=12.0).contains(&d.volume_db) {
        return Err("volume_db must be between -60 and +12 dB".into());
    }
    if settings.matrix.enabled {
        // Presets are validated against a stereo source here; the worker re-resolves them
        // for the actual channel count when a track starts.
        let preset = settings.matrix.preset.trim().to_ascii_lowercase();
        if preset != "custom" && !cadenza_dsp::matrix::preset_ids().contains(&preset.as_str()) {
            return Err(format!(
                "unknown matrix preset {:?}",
                settings.matrix.preset
            ));
        }
        if preset == "custom" {
            settings
                .matrix
                .resolve(2)
                .map_err(|e| format!("matrix: {e}"))?;
            for o in &settings.matrix.outputs {
                for f in o.lowpass.iter().chain(o.highpass.iter()) {
                    f.validate(44_100.0)
                        .map_err(|e| format!("matrix output {:?}: {e}", o.name))?;
                }
            }
        }
    }
    Ok(())
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

/// `?in_rate=44100&out_rate=705600&attenuation_db=150` describe the catalogue for a
/// specific conversion (default: the ×16 reference 44.1 → 705.6 kHz).
#[derive(Deserialize, Default)]
pub struct FilterQuery {
    /// Source rate in Hz.
    pub in_rate: Option<u32>,
    /// Output rate in Hz.
    pub out_rate: Option<u32>,
    /// Stopband attenuation in dB.
    pub attenuation_db: Option<f64>,
}

/// Every filter with taps, transition band, stopband and latency (see `FilterInfo`).
pub fn filter_list(q: &FilterQuery) -> Vec<Value> {
    let (in_rate, out_rate) = match (q.in_rate, q.out_rate) {
        (Some(i), Some(o)) if i > 0 && o > i => (i, o),
        _ => (44_100, 705_600),
    };
    let att = q
        .attenuation_db
        .unwrap_or(cadenza_dsp::fir::DEFAULT_ATTENUATION_DB);
    FilterKind::catalog()
        .iter()
        .map(|k| {
            let info = k.describe(in_rate, out_rate, att);
            let mut v = serde_json::to_value(&info).unwrap_or(Value::Null);
            // Kept for older clients.
            v["taps"] = json!(k.taps);
            v["apodizing"] = json!(k.apodizing);
            v
        })
        .collect()
}

async fn filters(Query(q): Query<FilterQuery>) -> Json<Value> {
    Json(json!(filter_list(&q)))
}

async fn dithers() -> Json<Value> {
    Json(json!(DitherKind::catalog()
        .iter()
        .map(|d| json!({
            "id": d.id(),
            "label": d.label(),
            "design_rate": d.design_rate(),
            "designed_per_rate": d.is_designed(),
            "order": d.order(),
            "shaper": d.coefficients(),
        }))
        .collect::<Vec<_>>()))
}

/// Everything a settings UI needs to offer choices: DSD rates / modulators / packings /
/// oversampling, GPU modes and the detected backend, matrix presets and crossover kinds,
/// output backends and the standard PCM rates.
pub fn options(settings: &crate::config::Settings) -> Value {
    let gpu = GpuBackend::detect(&settings.dsp.gpu);
    let output_modes: Vec<Value> = [OutputMode::Pcm, OutputMode::Dsd]
        .iter()
        .map(|m| json!({ "id": m.id(), "label": m.id().to_ascii_uppercase() }))
        .collect();
    let dsd_rates: Vec<Value> = DsdRate::catalog()
        .iter()
        .map(|r| {
            json!({
                "id": r.id(), "label": r.label(), "multiple": r.multiple(),
                "hz_44": r.hz(44_100), "hz_48": r.hz(48_000), "dop_possible": r.dop_possible(),
            })
        })
        .collect();
    let dsd_modulators: Vec<Value> = DsdModulator::catalog()
        .iter()
        .map(|m| json!({ "id": m.id(), "label": m.label(), "order": m.order(), "dithered": m.dithered() }))
        .collect();
    let dsd_packings: Vec<Value> = [DsdPacking::Dop, DsdPacking::Native]
        .iter()
        .map(|p| json!({ "id": p.id(), "label": p.label() }))
        .collect();
    let dsd_oversampling: Vec<Value> = [DsdOversampling::Staged, DsdOversampling::Full]
        .iter()
        .map(|o| json!({ "id": o.id() }))
        .collect();
    let gpu_modes: Vec<Value> = GpuMode::ALL
        .iter()
        .map(|m| json!({ "id": m.id(), "label": m.label() }))
        .collect();
    let matrix_presets: Vec<Value> = cadenza_dsp::matrix::presets()
        .iter()
        .map(|p| {
            json!({
                "id": p.id, "label": p.label, "description": p.description,
                "input_channels": p.input_channels, "output_channels": p.output_channels,
            })
        })
        .collect();
    let crossover_kinds: Vec<Value> = CrossoverKind::all()
        .iter()
        .map(|k| json!({ "id": k.id(), "label": k.short(), "slopes": k.slopes() }))
        .collect();
    json!({
        "output_modes": output_modes,
        "dsd_rates": dsd_rates,
        "dsd_modulators": dsd_modulators,
        "dsd_packings": dsd_packings,
        "dsd_oversampling": dsd_oversampling,
        "gpu_modes": gpu_modes,
        "gpu": gpu.report(&settings.dsp.gpu),
        "gpu_compiled_in": cadenza_dsp::gpu::COMPILED_IN,
        "matrix_presets": matrix_presets,
        "crossover_kinds": crossover_kinds,
        "backends": crate::output::backends(),
        "pcm_rates": cadenza_dsp::resampler::pcm_output_rates(),
        "max_pcm_rate": crate::output::MAX_PCM_RATE,
        "endpoint_port": crate::endpoint::DEFAULT_PORT,
    })
}

async fn options_route(State(st): State<AppState>) -> Json<Value> {
    Json(options(&st.engine.settings()))
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
    fn filters_options_and_validation() {
        let list = filter_list(&FilterQuery::default());
        assert!(list.len() >= 60);
        let first = &list[0];
        for key in [
            "id",
            "label",
            "family",
            "taps",
            "taps_at_rate",
            "transition_hz",
            "transition_percent",
            "stopband_db",
            "latency_seconds",
            "window",
        ] {
            assert!(first.get(key).is_some(), "missing {key}");
        }
        let x2 = filter_list(&FilterQuery {
            in_rate: Some(44_100),
            out_rate: Some(88_200),
            attenuation_db: None,
        });
        assert_eq!(x2[0]["out_rate"], 88_200);

        let s = crate::config::Settings::default();
        let o = options(&s);
        assert_eq!(o["dsd_rates"].as_array().unwrap().len(), 6);
        assert_eq!(o["dsd_rates"][5]["id"], "dsd2048");
        assert_eq!(o["dsd_rates"][5]["dop_possible"], false);
        assert_eq!(o["dsd_modulators"].as_array().unwrap().len(), 4);
        assert_eq!(o["gpu"]["precision"], "f64");
        assert!(o["matrix_presets"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["id"] == "2.1"));
        assert_eq!(o["max_pcm_rate"], 1_536_000);

        assert!(validate(&s).is_ok());
        let mut bad = s.clone();
        bad.dsp.dsd_rate = DsdRate::Dsd2048;
        bad.dsp.dsd_packing = DsdPacking::Dop;
        assert!(validate(&bad).unwrap_err().contains("DoP"));
        bad.dsp.dsd_packing = DsdPacking::Native;
        assert!(validate(&bad).is_ok());
        let mut bad = s.clone();
        bad.dsp.headroom_db = Some(3.0);
        assert!(validate(&bad).is_err());
        let mut bad = s.clone();
        bad.matrix.enabled = true;
        bad.matrix.preset = "7.1-nonsense".into();
        assert!(validate(&bad).unwrap_err().contains("preset"));
        bad.matrix.preset = "3-way-active".into();
        assert!(validate(&bad).is_ok());
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
