//! The HTTP control API (JSON + Server-Sent Events) and the single-file web UI.
//!
//! Everything is a thin layer over [`Engine`]: reads come from the latest status snapshot,
//! writes are commands. The UI lives in `static/index.html` and is embedded in the binary.

use crate::engine::{Command, Engine, Event, Track};
use crate::output::list_devices;
use axum::extract::{Path, Query, State};
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
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;

/// The embedded web UI.
pub const INDEX_HTML: &str = include_str!("../../static/index.html");

/// State shared by every handler.
#[derive(Clone)]
struct AppState {
    /// Handle to the playback engine.
    engine: Engine,
    /// Room correction state (measurements, the applied filter).
    room: Arc<Mutex<crate::room::Room>>,
    /// The limit test, one at a time.
    limit: Arc<LimitTest>,
}

/// The limit test: one at a time, run on its own thread.
#[derive(Default)]
struct LimitTest {
    /// Where the running (or last) test is.
    progress: Mutex<crate::limit::Progress>,
    /// Set to ask the running test to stop.
    cancel: std::sync::atomic::AtomicBool,
}

/// A JSON error with a status code.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

/// A `400 Bad Request` error with `msg`.
fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

/// Build the router.
pub fn router(engine: Engine) -> Router {
    let room = crate::room::Room::new(
        crate::room::Room::default_dir(),
        crate::room::capture::system(),
    );
    router_with_room(engine, room)
}

/// [`router`] with a given room-correction state (tests use a simulated room).
pub fn router_with_room(engine: Engine, room: crate::room::Room) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/status", get(status))
        .route(
            "/api/settings",
            get(get_settings).put(put_settings).post(put_settings),
        )
        .route("/api/devices", get(devices))
        .route("/api/filters", get(filters))
        .route("/api/filters/response", get(filter_response))
        .route("/api/dithers", get(dithers))
        .route("/api/options", get(options_route))
        .route("/api/queue", post(queue))
        .route("/api/control", post(control))
        .route("/api/events", get(events))
        .route("/api/live/{id}/cover", get(live_cover))
        .route("/api/room", get(room_report))
        .route("/api/room/options", post(room_options))
        .route("/api/room/measure", post(room_measure))
        .route("/api/room/apply", post(room_apply))
        .route("/api/room/remove", post(room_remove))
        .route("/api/room/forget", post(room_forget))
        .route("/api/room/noise", post(room_noise))
        .route("/api/room/verify", post(room_verify))
        .route(
            "/api/system",
            get(|| async { Json(crate::system::status()) }).post(system_set),
        )
        .route(
            "/api/system/apps",
            get(|| async { Json(json!({ "apps": crate::system::apps() })) }),
        )
        .route(
            "/api/encore",
            get(|| async { Json(crate::encore::status()) }).post(encore_set),
        )
        .route("/api/limit", get(limit_status))
        .route("/api/limit/start", post(limit_start))
        .route("/api/limit/cancel", post(limit_cancel))
        .with_state(AppState {
            engine,
            room: Arc::new(Mutex::new(room)),
            limit: Arc::default(),
        })
}

/// Run the server on `listen` until the process ends.
pub async fn serve(listen: &str, engine: Engine) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(listen).await?;
    serve_on(listener, engine).await
}

/// [`serve`] on a socket already bound (the desktop app picks a free port first).
pub async fn serve_on(listener: tokio::net::TcpListener, engine: Engine) -> anyhow::Result<()> {
    log::info!(
        "control API and web UI on http://{}/",
        listener.local_addr()?
    );
    axum::serve(listener, router(engine)).await?;
    Ok(())
}

/// `GET /`: the built-in web UI.
async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// Run `f` on the room state off the async runtime (measurements take seconds).
async fn with_room<T: Send + 'static>(
    st: &AppState,
    f: impl FnOnce(&mut crate::room::Room, &Engine) -> Result<T, String> + Send + 'static,
) -> Result<T, ApiError> {
    let (room, engine) = (Arc::clone(&st.room), st.engine.clone());
    tokio::task::spawn_blocking(move || {
        let mut r = room.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut r, &engine)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(bad)
}

/// `GET /api/room`: the room correction report.
async fn room_report(State(st): State<AppState>) -> Result<Json<Value>, ApiError> {
    with_room(&st, |r, e| Ok(r.report(e))).await.map(Json)
}

/// Change some of the design options (the others stay).
async fn room_options(
    State(st): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    with_room(&st, move |r, e| {
        let mut cur = serde_json::to_value(&r.options).map_err(|e| e.to_string())?;
        if let (Some(c), Some(b)) = (cur.as_object_mut(), body.as_object()) {
            for (k, v) in b {
                c.insert(k.clone(), v.clone());
            }
        }
        let o: cadenza_dsp::room::DesignOptions =
            serde_json::from_value(cur).map_err(|e| e.to_string())?;
        if !(o.low_hz > 0.0 && o.high_hz > o.low_hz && o.taps >= 1024 && o.taps <= 1 << 18) {
            return Err("options out of range".into());
        }
        r.options = o;
        Ok(r.report(e))
    })
    .await
    .map(Json)
}

/// `POST /api/room/measure`: run a sweep measurement on the device and return the report.
async fn room_measure(
    State(st): State<AppState>,
    Json(req): Json<crate::room::MeasureRequest>,
) -> Result<Json<Value>, ApiError> {
    with_room(&st, move |r, e| {
        r.measure(e, &req)?;
        Ok(r.report(e))
    })
    .await
    .map(Json)
}

/// `POST /api/room/apply`: build and apply the correction filter from the measurements.
async fn room_apply(State(st): State<AppState>) -> Result<Json<Value>, ApiError> {
    with_room(&st, |r, e| {
        r.apply(e)?;
        Ok(r.report(e))
    })
    .await
    .map(Json)
}

/// `POST /api/room/remove`: stop using the correction filter.
async fn room_remove(State(st): State<AppState>) -> Result<Json<Value>, ApiError> {
    with_room(&st, |r, e| {
        r.remove(e);
        Ok(r.report(e))
    })
    .await
    .map(Json)
}

/// Body naming one measured channel.
#[derive(Deserialize)]
struct ChannelBody {
    /// Channel name, e.g. `left`.
    channel: String,
}

/// `POST /api/room/forget`: drop one channel's measurement.
async fn room_forget(
    State(st): State<AppState>,
    Json(b): Json<ChannelBody>,
) -> Result<Json<Value>, ApiError> {
    with_room(&st, move |r, e| {
        r.forget(&b.channel)?;
        Ok(r.report(e))
    })
    .await
    .map(Json)
}

/// Body naming an input (capture) device; empty means the default one.
#[derive(Deserialize, Default)]
#[serde(default)]
struct InputBody {
    /// Input device name.
    input: String,
}

/// Measure the DAC's noise floor with an input device and store it for `auto` dither.
async fn room_noise(
    State(st): State<AppState>,
    Json(b): Json<InputBody>,
) -> Result<Json<Value>, ApiError> {
    with_room(&st, move |r, e| {
        let m = r.measure_noise(e, &b.input)?;
        let s = e.settings();
        let rate = e.status().output_format.map(|f| f.rate).unwrap_or(44_100);
        Ok(json!({
            "measurement": m,
            "dac_noise_floor_db": s.dsp.dac_noise_floor_db,
            "auto_choice": cadenza_dsp::dither::auto_choice(s.output.bits, rate, m.floor_db).id(),
        }))
    })
    .await
    .map(Json)
}

/// Loopback verification: bit-perfect check and THD+N through an input.
async fn room_verify(
    State(st): State<AppState>,
    Json(req): Json<crate::room::verify::VerifyRequest>,
) -> Result<Json<Value>, ApiError> {
    with_room(&st, move |r, e| {
        r.verify(e, &req)
            .and_then(|rep| serde_json::to_value(rep).map_err(|e| e.to_string()))
    })
    .await
    .map(Json)
}

/// Cover art a live stream (AirPlay) received from its sender.
async fn live_cover(Path(id): Path<String>) -> Response {
    match crate::decode::live::lookup(&id).and_then(|s| s.cover()) {
        Some((mime, bytes)) => (
            [
                (axum::http::header::CONTENT_TYPE, mime),
                (axum::http::header::CACHE_CONTROL, "no-cache".to_string()),
            ],
            bytes.to_vec(),
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// `GET /api/status`: the engine's current status.
async fn status(State(st): State<AppState>) -> Json<Value> {
    Json(serde_json::to_value(&*st.engine.status()).unwrap_or(Value::Null))
}

/// `GET /api/settings`: the current settings.
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

/// `POST /api/system`: turn the system input on or off and pick the source (whole system,
/// or one application), from the web UI. The body is a `SystemConfig`; the reply is the new
/// status.
async fn system_set(
    State(st): State<AppState>,
    Json(cfg): Json<crate::system::SystemConfig>,
) -> Result<Json<Value>, ApiError> {
    if !crate::system::AVAILABLE {
        return Err(bad("the system input is not available in this build"));
    }
    if cfg.enabled && cfg.mode == crate::system::SystemMode::App && cfg.app.trim().is_empty() {
        return Err(bad("pick an application to capture"));
    }
    let engine = st.engine.clone();
    crate::system::set_config(engine, &cfg);
    Ok(Json(crate::system::status()))
}

/// `POST /api/encore`: sign in to an Encore server (so Cadenza appears as a playback
/// device), switch servers, or sign out with an empty URL.
async fn encore_set(
    State(st): State<AppState>,
    Json(cfg): Json<crate::config::EncoreConfig>,
) -> Result<Json<Value>, ApiError> {
    crate::encore::set_config(st.engine.clone(), &cfg);
    Ok(Json(crate::encore::status()))
}

/// Checks beyond what deserialising enforces (shared with the Encore command handler).
pub fn validate(settings: &crate::config::Settings) -> Result<(), String> {
    let d = &settings.dsp;
    if !matches!(settings.output.bits, 16 | 24 | 32) {
        return Err("bits must be 16, 24 or 32".into());
    }
    if !cadenza_dsp::design::is_filter_id(&d.filter) {
        return Err(format!("unknown filter {:?}", d.filter));
    }
    if DitherKind::parse(&d.dither).is_none() {
        return Err(format!("unknown dither {:?}", d.dither));
    }
    if let Some(f) = d.dac_noise_floor_db {
        let (lo, hi) = crate::render::DAC_FLOOR_RANGE;
        if !(lo..=hi).contains(&f) {
            return Err(format!("dac_noise_floor_db must be {lo} … {hi} dBFS"));
        }
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

/// `GET /api/limit`: progress of the limit test (`idle` when none ran).
async fn limit_status(State(st): State<AppState>) -> Json<Value> {
    let mut p = st.limit.progress.lock().unwrap().clone();
    if p.state.is_empty() {
        p.state = "idle".into();
    }
    Json(json!(p))
}

/// Start the limit test for the current settings and device. Refused while music plays:
/// the test needs the whole machine, and playback would stutter during it.
async fn limit_start(State(st): State<AppState>) -> Result<Json<Value>, ApiError> {
    {
        let mut p = st.limit.progress.lock().unwrap();
        if p.state == "running" {
            return Err(bad("The test is already running."));
        }
        *p = crate::limit::Progress {
            state: "running".into(),
            current: "looking at the device".into(),
            ..Default::default()
        };
    }
    st.limit
        .cancel
        .store(false, std::sync::atomic::Ordering::Relaxed);
    let (engine, limit) = (st.engine.clone(), Arc::clone(&st.limit));
    // The test needs the whole machine, and the audio device must not sit half-open while
    // it runs: pause and let go of the device. Play opens it again afterwards.
    st.engine.send(crate::engine::Command::Release);
    std::thread::Builder::new()
        .name("cadenza-limit".into())
        .spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            let settings = engine.settings();
            let device = crate::limit::find_device(&settings, &list_devices());
            let r = crate::limit::run(&settings, device, &engine.cache(), &limit.cancel, |p| {
                *limit.progress.lock().unwrap() = p.clone();
            });
            let mut p = limit.progress.lock().unwrap();
            match r {
                Ok(report) => {
                    p.done = p.total;
                    p.current.clear();
                    p.report = report;
                    p.state = "done".into();
                }
                Err(e) if e == "cancelled" => p.state = "cancelled".into(),
                Err(e) => {
                    p.state = "error".into();
                    p.error = Some(e);
                }
            }
        })
        .map_err(|e| bad(e.to_string()))?;
    Ok(limit_status(State(st)).await)
}

/// `POST /api/limit/cancel`: ask the running limit test to stop.
async fn limit_cancel(State(st): State<AppState>) -> Json<Value> {
    st.limit
        .cancel
        .store(true, std::sync::atomic::Ordering::Relaxed);
    limit_status(State(st)).await
}

/// `GET /api/devices`: output and input devices on every backend (probed off the async
/// threads).
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

/// `?id=…&in_rate=…&out_rate=…` for `/api/filters/response`.
#[derive(Deserialize)]
struct ResponseQuery {
    /// Filter id (catalogue name or designer string).
    id: String,
    /// Conversion source rate, if any.
    in_rate: Option<u32>,
    /// Conversion target rate, if any.
    out_rate: Option<u32>,
}

/// The response plots of any filter (catalogue or designed) for a conversion, with its
/// description; designs it first if needed (into the engine's filter cache).
async fn filter_response(
    State(st): State<AppState>,
    Query(q): Query<ResponseQuery>,
) -> Result<Json<Value>, ApiError> {
    use cadenza_dsp::design::{response, Filter, MAX_PLOT_TAPS};
    let f = Filter::parse(&q.id).ok_or_else(|| bad(format!("unknown filter {:?}", q.id)))?;
    let (in_rate, out_rate) = match (q.in_rate, q.out_rate) {
        (Some(i), Some(o)) if i >= 8000 && o > i && o <= 6_144_000 => (i, o),
        (None, None) => (44_100, 705_600),
        _ => {
            return Err(bad(
                "in_rate and out_rate must be given, out_rate above in_rate",
            ))
        }
    };
    let att = st.engine.settings().dsp.attenuation_db;
    let cache = st.engine.cache();
    tokio::task::spawn_blocking(move || {
        let (spec, rate) = f.primary(in_rate, out_rate, att);
        let info = f.describe(in_rate, out_rate, att);
        if spec.taps > MAX_PLOT_TAPS {
            return Err(bad(format!(
                "{} taps at this rate: too long to plot (up to {} here)",
                spec.taps,
                MAX_PLOT_TAPS - 1
            )));
        }
        let h = cache.get_or_design(&spec);
        let (p, s) = f.edges(in_rate, out_rate, att);
        let r = response(&h, rate, in_rate as f64 / 2.0, p, s);
        Ok(Json(json!({ "info": info, "response": r })))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
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

/// `GET /api/filters`: the filter catalogue, with the designs that fit a conversion.
async fn filters(Query(q): Query<FilterQuery>) -> Json<Value> {
    Json(json!(filter_list(&q)))
}

/// `GET /api/dithers`: every dither kind with its shaper coefficients.
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
    let gpu_modes: Vec<Value> = GpuMode::available()
        .iter()
        .map(|m| json!({ "id": m.id(), "label": m.label() }))
        .collect();
    let gpu_precisions: Vec<Value> = cadenza_dsp::gpu::GpuPrecision::ALL
        .iter()
        .map(|p| json!({ "id": p.id(), "label": p.label() }))
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
        "gpu_precisions": gpu_precisions,
        "gpu_compiled_in": cadenza_dsp::gpu::COMPILED_IN,
        "gpu_cuda_compiled_in": cadenza_dsp::gpu::CUDA_COMPILED_IN,
        "gpu_wgpu_compiled_in": cadenza_dsp::gpu::WGPU_COMPILED_IN,
        "matrix_presets": matrix_presets,
        "headphone_modes": cadenza_dsp::HeadphoneMode::ALL
            .iter()
            .map(|m| json!({ "id": m.1, "label": m.2 }))
            .collect::<Vec<_>>(),
        "crossover_kinds": crossover_kinds,
        "backends": crate::output::backends(),
        "pcm_rates": cadenza_dsp::resampler::pcm_output_rates(),
        "max_pcm_rate": crate::output::MAX_PCM_RATE,
        "endpoint_port": crate::endpoint::DEFAULT_PORT,
    })
}

/// `GET /api/options`: everything a settings UI offers to choose from.
async fn options_route(State(st): State<AppState>) -> Json<Value> {
    Json(options(&st.engine.settings()))
}

/// One item of a `POST /api/queue` body.
#[derive(Deserialize)]
struct QueueItem {
    /// Stream URL to play (wins over `path`).
    url: Option<String>,
    /// Local file to play.
    path: Option<String>,
    /// HTTP headers to send with `url` (e.g. authentication).
    #[serde(default)]
    headers: Vec<(String, String)>,
    /// Caller's track id (string or number), echoed in the status.
    id: Option<Value>,
    /// Display title (defaults to the file name).
    #[serde(default)]
    title: String,
    /// Display artist.
    #[serde(default)]
    artist: String,
    /// Display album.
    #[serde(default)]
    album: String,
    /// Caller's album id (string or number).
    album_id: Option<Value>,
    /// Duration in seconds, if known ahead of decoding.
    duration: Option<f64>,
    /// Source format hint, e.g. `flac`.
    format: Option<String>,
    /// Cover art URL.
    cover: Option<String>,
}

/// Body of `POST /api/queue`.
#[derive(Deserialize)]
struct QueueBody {
    /// Tracks to queue.
    #[serde(default)]
    items: Vec<QueueItem>,
    /// Queue index to start at (`replace` mode).
    #[serde(default)]
    index: usize,
    /// Start position in seconds (`replace` mode).
    #[serde(default)]
    position: f64,
    /// `replace` (default, and play), `append`, `next`.
    #[serde(default)]
    mode: Option<String>,
}

impl QueueItem {
    /// Turn the item into an engine track; ids become strings and a missing title falls back
    /// to the file name.
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

/// `POST /api/queue`: replace (and play), append to or insert after the current track;
/// disc images and CUE sheets expand into their tracks.
async fn queue(
    State(st): State<AppState>,
    Json(body): Json<QueueBody>,
) -> Result<Json<Value>, ApiError> {
    // Disc images and CUE sheets become their tracks; `index` follows.
    let mut tracks: Vec<Track> = Vec::new();
    let mut index = body.index;
    for (i, item) in body.items.into_iter().enumerate() {
        let expanded = item.into_track()?.expand().map_err(bad)?;
        if i < body.index {
            index += expanded.len().saturating_sub(1);
        }
        tracks.extend(expanded);
    }
    match body.mode.as_deref().unwrap_or("replace") {
        "append" | "enqueue" => st.engine.send(Command::Enqueue(tracks)),
        "next" | "play_next" => st.engine.send(Command::PlayNext(tracks)),
        _ => st.engine.play_queue(tracks, index, body.position),
    }
    Ok(Json(json!({ "ok": true })))
}

/// Body of `POST /api/control`.
#[derive(Deserialize)]
struct ControlBody {
    /// `play`, `pause`, `next`, `seek`, `volume`, …
    action: String,
    /// The action's argument (seconds, dB, …), if it takes one.
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
        "reconnect" => Command::Reconnect,
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

/// `POST /api/control`: run one transport or volume action.
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

/// `GET /api/events`: server-sent events, the current status first, then every status and
/// track change, with a keep-alive ping every 15 s.
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
        assert_eq!(o["dsd_modulators"].as_array().unwrap().len(), 7);
        assert_eq!(o["dsd_modulators"][6]["id"], "asdm7ec8");
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

    #[test]
    fn designed_filters_plot_and_can_be_selected() {
        use crate::engine::{Engine, EngineOptions};
        let mut s = crate::config::Settings::default();
        s.output.backend = "null".into();
        let engine = Engine::start(
            s,
            EngineOptions {
                cache: cadenza_dsp::CoefficientCache::disabled(),
                realtime: false,
                persist: None,
            },
        );
        let api = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = api.local_addr().unwrap();
        api.set_nonblocking(true).unwrap();
        let app = router(engine.clone());
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let l = tokio::net::TcpListener::from_std(api).unwrap();
                axum::serve(l, app).await.unwrap();
            });
        });
        let id = "custom-kaiser-l-c100-t10-s120";
        let j: Value = ureq::get(&format!(
            "http://{addr}/api/filters/response?id={id}&in_rate=48000&out_rate=192000"
        ))
        .call()
        .unwrap()
        .into_json()
        .unwrap();
        assert_eq!(j["info"]["id"], id);
        assert_eq!(j["info"]["family"], "custom");
        assert_eq!(j["response"]["filter_rate"], 192000.0);
        assert!(j["response"]["stopband_db"].as_f64().unwrap() < -119.0);
        assert_eq!(j["response"]["transition"].as_array().unwrap().len(), 800);
        // Catalogue filters plot too; unknown ids and silly rates are refused.
        assert!(
            ureq::get(&format!("http://{addr}/api/filters/response?id=sinc-l-16k"))
                .call()
                .is_ok()
        );
        for q in ["id=nope", "id=sinc-l-16k&in_rate=48000&out_rate=44100"] {
            assert!(matches!(
                ureq::get(&format!("http://{addr}/api/filters/response?{q}")).call(),
                Err(ureq::Error::Status(400, _))
            ));
        }
        // The design is accepted as the filter setting.
        ureq::put(&format!("http://{addr}/api/settings"))
            .send_json(json!({ "dsp": { "filter": id } }))
            .unwrap();
        // The engine applies settings on its own thread.
        let t = std::time::Instant::now();
        while engine.settings().dsp.filter != id && t.elapsed().as_secs() < 10 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(engine.settings().dsp.filter, id);
        assert!(ureq::put(&format!("http://{addr}/api/settings"))
            .send_json(json!({ "dsp": { "filter": "custom-kaiser-l-c100-t10-s999" } }))
            .is_err());
    }
}
