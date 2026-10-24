//! Commands Encore sends to a device, and how they map onto the engine.

use super::Client;
use crate::api::control_command;
use crate::config::{RateSetting, Settings};
use crate::engine::{Command, Engine, Track};
use crate::output::split_backend;
use cadenza_dsp::dsd::{DsdModulator, DsdOversampling, DsdPacking, DsdRate, OutputMode};
use cadenza_dsp::gpu::GpuMode;
use cadenza_dsp::DitherKind;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Sequence numbers already executed (a command can arrive pushed *and* with a heartbeat).
static SEEN: Mutex<VecDeque<u64>> = Mutex::new(VecDeque::new());

/// `true` when command `seq` has already run; otherwise remember it (the last 200).
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

/// The tracks in a command's `tracks` array, as engine tracks.
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
        // the computer's own output volume, 0..1 (os_volume); the engine's stays as it is
        "system_volume" => match value.and_then(as_f64) {
            Some(v) => {
                if let Err(e) = crate::os_volume::set(v) {
                    log::warn!("Encore: {e}");
                }
            }
            None => log::warn!("Encore: system_volume needs a numeric value"),
        },
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

/// The exclusive backend of this build, if any (`wasapi`, `coreaudio`, `alsa`), and the
/// shared one (`cpal`).
fn exclusive_backend() -> Option<&'static str> {
    crate::output::backends()
        .into_iter()
        .find(|b| matches!(*b, "wasapi" | "coreaudio" | "alsa"))
}

/// A boolean from JSON: `true`/`false`, or a string like `on`, `1`, `yes`.
fn as_bool(v: &Value) -> Option<bool> {
    v.as_bool()
        .or_else(|| match v.as_str()?.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "on" | "yes" => Some(true),
            "false" | "0" | "off" | "no" => Some(false),
            _ => None,
        })
}

/// A number from JSON, or a string holding one.
fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|t| t.trim().parse().ok()))
}

/// Apply an `engine {set: {…}}` patch to `s`, key by key. Every key of the engine
/// description that is a setting is accepted: `bitperfect`, `filter`, `dither`, `rate`,
/// `bits`, `output` (`<backend>:<device>`), `output_mode`, `dsd_rate`, `dsd_modulator`,
/// `dsd_packing`, `dsd_gain_db`, `dsd_oversampling`, `headroom_db` (number, `null` or
/// `"auto"`), `soft_clip`, `gpu` (`auto` / `on` / `off`), `matrix` (`off` or a preset id)
/// and `exclusive`. Keys with values that make no sense are skipped; the returned list
/// says which, in plain words.
pub fn patch_settings(s: &mut Settings, set: &serde_json::Map<String, Value>) -> Vec<String> {
    let mut ignored = Vec::new();
    for (k, v) in set {
        let mut bad = |what: &str| ignored.push(format!("{k}: {what} {v}"));
        match k.as_str() {
            "bitperfect" | "bit_perfect" => match as_bool(v) {
                Some(b) => s.dsp.bit_perfect = b,
                None => bad("not a boolean"),
            },
            "filter" => match v
                .as_str()
                .filter(|id| cadenza_dsp::design::is_filter_id(id))
            {
                Some(id) => s.dsp.filter = id.to_string(),
                None => bad("unknown filter"),
            },
            "dither" => match v.as_str().filter(|id| DitherKind::parse(id).is_some()) {
                Some(id) => s.dsp.dither = id.to_string(),
                None => bad("unknown dither"),
            },
            "rate" => {
                let parsed = match v {
                    Value::Number(n) => n.as_u64().map(|r| RateSetting::Fixed(r as u32)),
                    Value::String(t) => RateSetting::parse(t),
                    _ => None,
                };
                match parsed {
                    Some(r) => s.output.rate = r,
                    None => bad("bad rate"),
                }
            }
            "bits" => {
                let n = v
                    .as_u64()
                    .or_else(|| v.as_str().and_then(|t| t.trim().parse().ok()));
                match n.filter(|b| matches!(b, 16 | 24 | 32)) {
                    Some(b) => s.output.bits = b as u32,
                    None => bad("bits must be 16, 24 or 32:"),
                }
            }
            "output" => {
                let parsed = v.as_str().and_then(|id| {
                    let (b, d) = id.split_once(':').unwrap_or((id, ""));
                    let (b, d) = split_backend(b, d);
                    crate::output::backends()
                        .contains(&b)
                        .then(|| (b.to_string(), d.to_string()))
                });
                match parsed {
                    Some((b, d)) => {
                        s.output.backend = b;
                        s.output.device = d;
                    }
                    None => bad("unknown output"),
                }
            }
            "output_mode" => match v.as_str().and_then(OutputMode::parse) {
                Some(m) => s.dsp.output_mode = m,
                None => bad("output mode must be pcm or dsd:"),
            },
            "dsd_rate" => match v.as_str().and_then(DsdRate::parse) {
                Some(r) => s.dsp.dsd_rate = r,
                None => bad("unknown DSD rate"),
            },
            "dsd_modulator" => match v.as_str().and_then(DsdModulator::parse) {
                Some(m) => s.dsp.dsd_modulator = m,
                None => bad("unknown DSD modulator"),
            },
            "dsd_packing" => match v.as_str().and_then(DsdPacking::parse) {
                Some(p) => s.dsp.dsd_packing = p,
                None => bad("DSD packing must be dop or native:"),
            },
            "dsd_gain_db" => match as_f64(v).filter(|g| g.is_finite()) {
                Some(g) => s.dsp.dsd_gain_db = g,
                None => bad("not a number"),
            },
            "dsd_oversampling" => match v.as_str().and_then(DsdOversampling::parse) {
                Some(o) => s.dsp.dsd_oversampling = o,
                None => bad("DSD oversampling must be staged or full:"),
            },
            "headroom_db" => {
                let auto = v.is_null()
                    || v.as_str()
                        .is_some_and(|t| t.trim().is_empty() || t.eq_ignore_ascii_case("auto"));
                if auto {
                    s.dsp.headroom_db = None;
                } else {
                    match as_f64(v).filter(|h| h.is_finite()) {
                        Some(h) => s.dsp.headroom_db = Some(h),
                        None => bad("headroom must be dB or auto:"),
                    }
                }
            }
            "soft_clip" => match as_bool(v) {
                Some(b) => s.dsp.soft_clip = b,
                None => bad("not a boolean"),
            },
            "gpu" => {
                let parsed = match v {
                    Value::Bool(true) => "on".parse::<GpuMode>().ok(),
                    Value::Bool(false) => Some(GpuMode::Off),
                    Value::String(t) => t.parse::<GpuMode>().ok(),
                    _ => None,
                };
                match parsed {
                    Some(m) => s.dsp.gpu.mode = m,
                    None => bad("gpu must be auto, on or off:"),
                }
            }
            "dac_noise_floor_db" => match v {
                Value::Null => s.dsp.dac_noise_floor_db = None,
                _ => match v.as_f64().filter(|f| {
                    (crate::render::DAC_FLOOR_RANGE.0..=crate::render::DAC_FLOOR_RANGE.1)
                        .contains(f)
                }) {
                    Some(f) => s.dsp.dac_noise_floor_db = Some(f),
                    None => bad("dac_noise_floor_db must be −160 … −60 dBFS"),
                },
            },
            "headphones" => match v.as_str().and_then(cadenza_dsp::HeadphoneMode::from_id) {
                Some(m) => s.dsp.headphones = m,
                None => bad("unknown headphones mode"),
            },
            "matrix" => {
                let id = match v {
                    Value::Null | Value::Bool(false) => Some("off".to_string()),
                    Value::String(t) => Some(t.trim().to_ascii_lowercase()),
                    _ => None,
                };
                match id.as_deref() {
                    Some("off") | Some("none") | Some("") => s.matrix.enabled = false,
                    Some(p) if cadenza_dsp::matrix::preset_ids().contains(&p) => {
                        s.matrix.enabled = true;
                        s.matrix.preset = p.to_string();
                    }
                    _ => bad("unknown matrix preset"),
                }
            }
            "exclusive" => match as_bool(v) {
                Some(want) => {
                    let (current, _) = split_backend(&s.output.backend, &s.output.device);
                    let is_exclusive = !matches!(current, "cpal" | "null");
                    if want != is_exclusive {
                        let target = if want {
                            exclusive_backend()
                        } else {
                            crate::output::backends()
                                .contains(&"cpal")
                                .then_some("cpal")
                        };
                        match target {
                            Some(b) => {
                                s.output.backend = b.to_string();
                                s.output.device.clear();
                            }
                            None => bad("this build has no output backend for exclusive ="),
                        }
                    }
                }
                None => bad("not a boolean"),
            },
            _ => bad("unknown engine setting"),
        }
    }
    ignored
}

/// `engine {set: {…}}`: patch the settings (see [`patch_settings`]), check the result with
/// the API's validation and hand it to the engine. Bad values are logged and skipped; a
/// patch whose result does not validate is dropped as a whole.
pub fn apply_engine_settings(engine: &Engine, set: &serde_json::Map<String, Value>) {
    let mut s = engine.settings();
    for msg in patch_settings(&mut s, set) {
        log::warn!("Encore: ignoring engine setting {msg}");
    }
    if let Err(e) = crate::api::validate(&s) {
        log::warn!("Encore: engine settings not applied: {e}");
        return;
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
    fn engine_patch_covers_every_setting_and_skips_bad_values() {
        let mut s = Settings::default();
        s.output.backend = "null".into();
        let set = serde_json::json!({
            "bitperfect": true,
            "filter": "sinc-l-1m",
            "dither": "tpdf",
            "rate": 352800,
            "bits": 32,
            "output": "null:null",
            "output_mode": "dsd",
            "dsd_rate": "dsd512",
            "dsd_modulator": "dsd5v2",
            "dsd_packing": "native",
            "dsd_gain_db": -4.5,
            "dsd_oversampling": "full",
            "headroom_db": -2.0,
            "soft_clip": false,
            "gpu": "on",
            "matrix": "2.1",
        });
        let ignored = patch_settings(&mut s, set.as_object().unwrap());
        assert!(ignored.is_empty(), "{ignored:?}");
        assert!(s.dsp.bit_perfect);
        assert_eq!(s.dsp.filter, "sinc-l-1m");
        assert_eq!(s.dsp.dither, "tpdf");
        assert_eq!(s.output.rate, RateSetting::Fixed(352_800));
        assert_eq!(s.output.bits, 32);
        assert_eq!(
            (s.output.backend.as_str(), s.output.device.as_str()),
            ("null", "null")
        );
        assert_eq!(s.dsp.output_mode, OutputMode::Dsd);
        assert_eq!(s.dsp.dsd_rate, DsdRate::Dsd512);
        assert_eq!(s.dsp.dsd_modulator, DsdModulator::Dsd5v2);
        assert_eq!(s.dsp.dsd_packing, DsdPacking::Native);
        assert_eq!(s.dsp.dsd_gain_db, -4.5);
        assert_eq!(s.dsp.dsd_oversampling, DsdOversampling::Full);
        assert_eq!(s.dsp.headroom_db, Some(-2.0));
        assert!(!s.dsp.soft_clip);
        assert_eq!(s.dsp.gpu.mode, "on".parse::<GpuMode>().unwrap());
        assert!(s.matrix.enabled);
        assert_eq!(s.matrix.preset, "2.1");
        assert!(crate::api::validate(&s).is_ok());

        // Other spellings: strings for numbers and booleans, auto headroom, matrix off.
        let set = serde_json::json!({
            "bitperfect": "false",
            "rate": "auto",
            "bits": "24",
            "headroom_db": "auto",
            "gpu": "off",
            "matrix": "off",
            "dsd_gain_db": "-3",
            "exclusive": true,
        });
        let ignored = patch_settings(&mut s, set.as_object().unwrap());
        assert!(!s.dsp.bit_perfect);
        assert_eq!(s.output.rate, RateSetting::Auto);
        assert_eq!(s.output.bits, 24);
        assert_eq!(s.dsp.headroom_db, None);
        assert_eq!(s.dsp.gpu.mode, GpuMode::Off);
        assert!(!s.matrix.enabled);
        assert_eq!(s.dsp.dsd_gain_db, -3.0);
        // `exclusive` moves between the shared and the direct backend of this build; on a
        // build with neither it is the one key that gets skipped.
        if exclusive_backend().is_some() {
            assert!(ignored.is_empty(), "{ignored:?}");
            assert_eq!(s.output.backend, exclusive_backend().unwrap());
            assert_eq!(s.output.device, "");
            let set = serde_json::json!({ "exclusive": false });
            patch_settings(&mut s, set.as_object().unwrap());
            assert_eq!(s.output.backend, "cpal");
        } else {
            assert_eq!(ignored.len(), 1, "{ignored:?}");
        }
        // headroom_db null is automatic too.
        s.dsp.headroom_db = Some(-1.0);
        patch_settings(
            &mut s,
            serde_json::json!({ "headroom_db": null })
                .as_object()
                .unwrap(),
        );
        assert_eq!(s.dsp.headroom_db, None);

        // Every key with a value that makes no sense is skipped and named; nothing changes.
        let before = s.clone();
        let set = serde_json::json!({
            "bitperfect": "maybe",
            "filter": "no-such-filter",
            "dither": 7,
            "rate": "fast",
            "bits": 20,
            "output": "tape:deck",
            "output_mode": "analogue",
            "dsd_rate": "dsd9000",
            "dsd_modulator": "x",
            "dsd_packing": "sideways",
            "dsd_gain_db": "loud",
            "dsd_oversampling": "lots",
            "headroom_db": "much",
            "soft_clip": 3,
            "gpu": "maybe",
            "matrix": "7.1-nonsense",
            "exclusive": "sometimes",
            "colour": "blue",
        });
        let ignored = patch_settings(&mut s, set.as_object().unwrap());
        assert_eq!(ignored.len(), 18, "{ignored:?}");
        assert_eq!(s, before);
        assert!(ignored
            .iter()
            .any(|m| m.starts_with("colour: unknown engine setting")));
    }

    #[test]
    fn invalid_patches_are_not_handed_to_the_engine() {
        // A value every key accepts on its own but the whole does not: DSD2048 over DoP.
        let mut s = Settings::default();
        s.output.backend = "null".into();
        let engine = Engine::start(
            s.clone(),
            crate::engine::EngineOptions {
                cache: cadenza_dsp::CoefficientCache::disabled(),
                realtime: false,
                persist: None,
            },
        );
        let set = serde_json::json!({ "dsd_rate": "dsd2048", "dsd_packing": "dop" });
        apply_engine_settings(&engine, set.as_object().unwrap());
        let set = serde_json::json!({ "bits": 16 });
        apply_engine_settings(&engine, set.as_object().unwrap());
        let t0 = std::time::Instant::now();
        while engine.settings().output.bits != 16
            && t0.elapsed() < std::time::Duration::from_secs(5)
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let got = engine.settings();
        assert_eq!(got.output.bits, 16);
        assert_eq!(
            got.dsp.dsd_rate,
            DsdRate::Dsd256,
            "the invalid patch was dropped whole"
        );
        engine.shutdown();
    }

    #[test]
    fn duplicate_sequence_numbers_run_once() {
        assert!(!seen_before(900_001));
        assert!(seen_before(900_001));
        assert!(!seen_before(900_002));
    }
}
