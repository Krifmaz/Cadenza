//! The `state` object sent with every heartbeat: what Encore's device list, remote,
//! signal-path dialog and the Settings → Playback → "Cadenza engine" section display.

use super::Client;
use crate::config::{RateSetting, Settings};
use crate::engine::{PlayState, Status, Track};
use crate::output::{split_backend, DeviceInfo, DsdCaps};
use cadenza_dsp::dsd::{DsdModulator, DsdOversampling, DsdPacking, DsdRate, OutputMode};
use cadenza_dsp::gpu::{Backend as GpuBackend, GpuConfig, GpuMode};
use cadenza_dsp::signal_path::rate_label;
use cadenza_dsp::{DitherKind, FilterKind};
use serde_json::{json, Value};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long a device listing or a GPU probe is reused before it is refreshed.
const REFRESH: Duration = Duration::from_secs(30);

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

/// `"<backend>:<device>"` – the id the engine section uses for an output. `endpoint://x`
/// style backends become `endpoint:x`.
pub fn output_id(backend: &str, device: &str) -> String {
    let (b, d) = split_backend(backend, device);
    format!("{b}:{d}")
}

/// `[id, label]` for one device: the name plus a short note on what it can do.
fn output_entry(d: &DeviceInfo) -> Value {
    let mut notes = Vec::new();
    if let Some(r) = d.rates.iter().max() {
        notes.push(format!("up to {} kHz", rate_label(*r)));
    }
    if d.dsd.any() {
        match d.dsd.max_rate_label() {
            Some(l) => notes.push(l),
            None => notes.push("DSD".into()),
        }
    }
    notes.push(if d.exclusive { "exclusive" } else { "shared" }.into());
    let label = format!("{} · {}", d.name, notes.join(" · "));
    json!([output_id(&d.backend, &d.id), label])
}

/// Every output device of every compiled backend, as `[id, label]` pairs. Probing opens
/// devices, so the listing is refreshed at most every 30 s.
pub fn outputs() -> Vec<Value> {
    type Cached = Option<(Instant, Vec<Value>)>;
    static CACHE: OnceLock<Mutex<Cached>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, list)) = guard.as_ref() {
        if at.elapsed() < REFRESH {
            return list.clone();
        }
    }
    let list: Vec<Value> = crate::output::list_devices()
        .iter()
        .map(output_entry)
        .collect();
    *guard = Some((Instant::now(), list.clone()));
    list
}

/// Whether an NVIDIA card is there (whatever the `gpu` setting says) and its short name.
/// Probing goes through the driver, so the answer is kept for 30 s.
pub fn gpu_available(device: usize) -> (bool, Option<String>) {
    type Cached = Option<(Instant, usize, bool, Option<String>)>;
    static CACHE: OnceLock<Mutex<Cached>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, dev, ok, name)) = guard.as_ref() {
        if *dev == device && at.elapsed() < REFRESH {
            return (*ok, name.clone());
        }
    }
    let probe = GpuBackend::probe(&GpuConfig {
        mode: GpuMode::Auto,
        device,
        ..GpuConfig::default()
    });
    let (ok, name) = match probe {
        Ok(b) => (b.is_gpu(), b.device_short_name()),
        Err(_) => (false, None),
    };
    *guard = Some((Instant::now(), device, ok, name.clone()));
    (ok, name)
}

/// The DSD rates the output device takes: nothing above its ceiling, and DSD2048 only
/// with native DSD (DoP cannot carry it). Unknown capabilities (no device opened yet)
/// list everything DoP can carry.
pub fn dsd_rates(caps: &DsdCaps) -> Vec<Value> {
    let unknown = !caps.any();
    DsdRate::catalog()
        .iter()
        .filter(|r| caps.max_rate.is_none_or(|m| r.hz(44_100) <= m))
        .filter(|r| {
            if r.dop_possible() {
                unknown || caps.dop || !caps.native.is_empty()
            } else {
                !caps.native.is_empty()
            }
        })
        .map(|r| json!([r.id(), r.label()]))
        .collect()
}

/// DoP and native packings the output device takes (both when nothing is known yet).
pub fn dsd_packings(caps: &DsdCaps) -> Vec<Value> {
    let unknown = !caps.any();
    let mut v = Vec::new();
    if unknown || caps.dop {
        v.push(json!([DsdPacking::Dop.id(), DsdPacking::Dop.label()]));
    }
    if unknown || !caps.native.is_empty() {
        v.push(json!([DsdPacking::Native.id(), "Native"]));
    }
    v
}

/// `auto` / `on` / `off` – what the UI shows for the `gpu` setting (`cuda` and `wgpu`
/// are `on`).
pub fn gpu_mode_id(mode: GpuMode) -> &'static str {
    match mode {
        GpuMode::Auto => "auto",
        GpuMode::Off => "off",
        GpuMode::Cuda | GpuMode::Wgpu | GpuMode::On => "on",
    }
}

/// The catalogue as `[id, label]` pairs, plus the current filter when it is a design (so
/// Encore's picker can show it).
fn filter_options(current: &str) -> Vec<Value> {
    let mut v: Vec<Value> = FilterKind::catalog()
        .iter()
        .map(|k| json!([k.id(), k.label()]))
        .collect();
    if let Some(d) = cadenza_dsp::design::FilterDesign::parse(current) {
        v.push(json!([d.id(), d.label()]));
    }
    v
}

/// The `engine` object: the v2 engine description the Encore UIs render.
pub fn engine_description(client: &Client, s: &Status, embedded: bool) -> Value {
    let settings: &Settings = &s.settings;
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
    let rate: Value = match settings.output.rate {
        RateSetting::Auto => json!("auto"),
        RateSetting::Source => json!("source"),
        RateSetting::Fixed(r) => json!(r),
    };
    let output_name = if s.output.is_empty() {
        let (b, d) = split_backend(&settings.output.backend, &settings.output.device);
        if d.is_empty() {
            b.to_string()
        } else {
            format!("{b} · {d}")
        }
    } else {
        s.output.clone()
    };
    let output = match s.output_format {
        Some(f) => format!("{output_name} · {f}"),
        None => output_name,
    };
    let signal = s.signal_path.as_ref();
    let gpu_in_use = signal.is_some_and(|p| p.stages.iter().any(|st| st.id == "gpu" && st.active));
    let (gpu_available, gpu_name) = gpu_available(settings.dsp.gpu.device);
    let host = client
        .device_name()
        .strip_prefix("Cadenza on ")
        .unwrap_or("")
        .to_string();
    let host = if host.is_empty() {
        gethostname::gethostname().to_string_lossy().to_string()
    } else {
        host
    };
    let mut matrix_presets: Vec<Value> = vec![json!(["off", "Off"])];
    matrix_presets.extend(
        cadenza_dsp::matrix::presets()
            .iter()
            .map(|p| json!([p.id, p.label])),
    );
    json!({
        "name": "Cadenza",
        "version": env!("CARGO_PKG_VERSION"),
        "embedded": embedded,
        "host": host,
        "error": s.error,
        "output": output,
        "exclusive": s.output_exclusive,
        "exclusive_wanted": settings.output.backend != "cpal" && settings.output.backend != "null",
        "outputs": outputs(),
        "output_id": output_id(&settings.output.backend, &settings.output.device),
        "bitperfect": settings.dsp.bit_perfect,
        "filter": settings.dsp.filter,
        "filters": filter_options(&settings.dsp.filter),
        "dither": settings.dsp.dither,
        "dithers": DitherKind::catalog().iter().map(|d| json!([d.id(), d.label()])).collect::<Vec<_>>(),
        "rate": rate,
        "rates": rates,
        "bits": settings.output.bits,
        "bits_list": [16, 24, 32],
        "output_mode": settings.dsp.output_mode.id(),
        "output_modes": [[OutputMode::Pcm.id(), "PCM"], [OutputMode::Dsd.id(), "DSD"]],
        "dsd": {
            "rate": settings.dsp.dsd_rate.id(),
            "rates": dsd_rates(&s.output_dsd),
            "modulator": settings.dsp.dsd_modulator.id(),
            "modulators": DsdModulator::catalog().iter().map(|m| json!([m.id(), m.label()])).collect::<Vec<_>>(),
            "packing": settings.dsp.dsd_packing.id(),
            "packings": dsd_packings(&s.output_dsd),
            "gain_db": settings.dsp.dsd_gain_db,
            "oversampling": settings.dsp.dsd_oversampling.id(),
            "oversamplings": [[DsdOversampling::Staged.id(), "Staged"], [DsdOversampling::Full.id(), "Full"]],
            "device": s.output_dsd,
        },
        "headroom_db": settings.dsp.headroom_db,
        "soft_clip": settings.dsp.soft_clip,
        "headphones": settings.dsp.headphones.id(),
        "dac_noise_floor_db": settings.dsp.dac_noise_floor_db,
        "gpu": gpu_mode_id(settings.dsp.gpu.mode),
        "gpu_available": gpu_available,
        "gpu_in_use": gpu_in_use,
        "gpu_name": gpu_name,
        "matrix": if settings.matrix.enabled { json!(settings.matrix.preset) } else { Value::Null },
        "matrix_presets": matrix_presets,
    })
}

/// Build the heartbeat state.
pub fn heartbeat_state(client: &Client, s: &Status) -> Value {
    heartbeat_state_as(client, s, false)
}

/// Build the heartbeat state, saying whether this player lives inside the desktop app.
pub fn heartbeat_state_as(client: &Client, s: &Status, embedded: bool) -> Value {
    let track = s.track.as_ref();
    let album_id = track.and_then(|t| t.album_id.clone());
    // A live stream's cover, and an Encore cover fetched through this player (the in-app
    // library), are served by this player's own API; Encore cannot reach them.
    let cover = track
        .and_then(|t| t.cover.clone())
        .filter(|c| !c.starts_with("/api/live/") && !c.starts_with("/api/encore/"))
        .or_else(|| {
            album_id
                .as_ref()
                .map(|a| format!("/api/library/albums/{a}/cover"))
        });
    let format = track
        .and_then(|t| t.format.clone())
        .or_else(|| s.source_format.clone());
    let signal = s.signal_path.as_ref();
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
        // the computer's own output volume (Encore's "Device volume"); null when its mixer
        // cannot be reached, and Encore then leaves that slider out
        "system_volume": crate::os_volume::get(),
        "index": s.index,
        "queue": s.queue.iter().map(slim_track).collect::<Vec<_>>(),
        "signal": {
            "label": signal.map(|p| p.label.clone()).unwrap_or_default(),
            "transport": "original",
            "output_rate": s.output_format.map(|f| f.rate),
            "bitperfect": signal.map(|p| p.bit_perfect).unwrap_or(false),
            "stages": signal.map(|p| p.stages.iter().filter(|st| st.active).map(|st| json!({ "name": st.name, "detail": st.detail })).collect::<Vec<_>>()).unwrap_or_default(),
        },
        "engine": engine_description(client, s, embedded),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EncoreConfig, Settings};
    use cadenza_dsp::dsd::NativeFormat;
    use std::sync::Arc;

    #[test]
    fn state_has_the_fields_encore_reads() {
        let client = Client::new(Arc::new(EncoreConfig {
            url: "http://e".into(),
            cookie: "c".into(),
            device_name: "Cadenza on box".into(),
            ..Default::default()
        }));
        let mut settings = Settings::default();
        settings.output.backend = "null".into();
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
            settings,
            output_rates: vec![44100, 88200],
            output_dsd: crate::output::null::dsd_caps(),
            output: "null".into(),
            ..Default::default()
        };
        let v = heartbeat_state(&client, &status);
        assert_eq!(v["id"], 42);
        assert_eq!(v["cover"], "/api/library/albums/7/cover");
        assert_eq!(v["playing"], true);
        assert_eq!(v["signal"]["transport"], "original");
        let e = &v["engine"];
        assert_eq!(e["name"], "Cadenza");
        assert_eq!(e["rate"], "auto");
        assert_eq!(e["rates"].as_array().unwrap().len(), 4);
        assert_eq!(e["output_mode"], "pcm");
        assert_eq!(e["dsd"]["rate"], "dsd256");
        assert_eq!(e["dsd"]["modulator"], "asdm7");
        assert_eq!(e["dsd"]["device"]["dop"], true);
        assert_eq!(e["headroom_db"], Value::Null);
        assert_eq!(e["gpu"], "auto");
        assert_eq!(e["matrix"], Value::Null);
        assert_eq!(e["exclusive"], false);
        assert!(e["filters"].as_array().unwrap().len() > 10);
        // The seam the Encore UIs read: every key, in the shape the contract names.
        assert_eq!(e["embedded"], false);
        assert_eq!(e["host"], "box");
        assert_eq!(e["error"], Value::Null);
        assert_eq!(e["output"], "null");
        assert_eq!(e["output_id"], "null:");
        assert_eq!(e["exclusive_wanted"], false);
        let outputs = e["outputs"].as_array().unwrap();
        assert!(outputs.iter().any(|o| o[0] == "null:null"), "{outputs:?}");
        assert!(outputs[0][1].as_str().unwrap().contains("Null output"));
        assert_eq!(e["bitperfect"], false);
        assert_eq!(e["bits"], 24);
        assert_eq!(e["bits_list"], json!([16, 24, 32]));
        assert_eq!(e["output_modes"], json!([["pcm", "PCM"], ["dsd", "DSD"]]));
        assert!(e["dithers"].as_array().unwrap().len() > 3);
        // The null output does native DSD up to DSD1024: DSD2048 is above its ceiling.
        let rates: Vec<&str> = e["dsd"]["rates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r[0].as_str().unwrap())
            .collect();
        assert_eq!(rates, ["dsd64", "dsd128", "dsd256", "dsd512", "dsd1024"]);
        assert_eq!(e["dsd"]["modulators"].as_array().unwrap().len(), 7);
        assert_eq!(e["dsd"]["modulators"][6], json!(["asdm7ec8", "ASDM7EC8"]));
        assert_eq!(e["dsd"]["packing"], "dop");
        assert_eq!(
            e["dsd"]["packings"],
            json!([["dop", "DoP"], ["native", "Native"]])
        );
        assert_eq!(e["dsd"]["gain_db"], -3.0);
        assert_eq!(e["dsd"]["oversampling"], "staged");
        assert_eq!(e["soft_clip"], true);
        assert!(e["gpu_available"].is_boolean());
        if !cadenza_dsp::gpu::COMPILED_IN {
            assert_eq!(e["gpu_available"], false);
            assert_eq!(e["gpu_name"], Value::Null);
        }
        assert_eq!(e["gpu_in_use"], false);
        assert!(e["gpu_name"].is_null() || e["gpu_name"].is_string());
        let presets = e["matrix_presets"].as_array().unwrap();
        assert_eq!(presets[0], json!(["off", "Off"]));
        assert!(presets.iter().any(|p| p[0] == "2.1"));
        // Embedded players say so.
        let v = heartbeat_state_as(&client, &status, true);
        assert_eq!(v["engine"]["embedded"], true);
    }

    #[test]
    fn dsd_lists_follow_the_device() {
        let none = DsdCaps::NONE;
        let ids = |v: Vec<Value>| -> Vec<String> {
            v.iter()
                .map(|r| r[0].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(
            ids(dsd_rates(&none)),
            ["dsd64", "dsd128", "dsd256", "dsd512", "dsd1024"]
        );
        assert_eq!(ids(dsd_packings(&none)), ["dop", "native"]);
        let dop = DsdCaps::dop_only(Some(5_644_800));
        assert_eq!(ids(dsd_rates(&dop)), ["dsd64", "dsd128"]);
        assert_eq!(ids(dsd_packings(&dop)), ["dop"]);
        let native = DsdCaps {
            dop: false,
            native: vec![NativeFormat::U32Be],
            max_rate: Some(48_000 * 2048),
        };
        assert_eq!(ids(dsd_rates(&native)).last().unwrap(), "dsd2048");
        assert_eq!(ids(dsd_packings(&native)), ["native"]);
        assert_eq!(
            output_id("endpoint://Living room", ""),
            "endpoint:Living room"
        );
        assert_eq!(output_id("alsa", "hw:1,0"), "alsa:hw:1,0");
        assert_eq!(gpu_mode_id(GpuMode::Cuda), "on");
    }
}
