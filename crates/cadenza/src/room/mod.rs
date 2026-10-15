//! Room correction from the web UI: play a sweep, record it with a measurement
//! microphone, design a correction and load it into the convolution stage.
//!
//! A measurement plays through the engine itself – the same output, rate and volume as
//! music, with the EQ, any correction and the headphone stage switched off for its
//! duration – while the microphone records at its own rate. The sweep is generated at
//! that rate, so the recording deconvolves straight into the impulse response
//! ([`cadenza_dsp::room`]). Each speaker is measured on its own; the correction (one
//! filter per channel, or one for both) is written as a WAV next to the measurements and
//! becomes the `convolution` setting. Impulses are resampled to the stream's rate by the
//! pipeline, so one measurement serves every source rate.

pub mod capture;
pub mod noise;

use crate::decode::live::{self, LiveStream};
use crate::decode::Source;
use crate::engine::{Engine, Track};
use cadenza_dsp::room::{self, DesignOptions, Sweep};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

pub use capture::{Capture, CaptureFactory};

/// What to measure.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct MeasureRequest {
    /// `left` or `right`.
    pub channel: String,
    /// Microphone (empty: the system's default input).
    pub input: String,
    /// Sweep level, dBFS (before the engine's volume).
    pub level_db: f64,
    /// Sweep length, seconds.
    pub seconds: f64,
}

impl Default for MeasureRequest {
    fn default() -> Self {
        Self {
            channel: "left".into(),
            input: String::new(),
            level_db: -12.0,
            seconds: 6.0,
        }
    }
}

/// One speaker's measurement.
#[derive(Clone, Debug)]
pub struct Measurement {
    /// Impulse response.
    pub ir: Vec<f64>,
    /// Its rate.
    pub rate: u32,
    /// Loudest sample the microphone recorded, dBFS.
    pub peak_dbfs: f64,
}

/// The room-correction state behind the API.
pub struct Room {
    dir: PathBuf,
    capture: CaptureFactory,
    /// Left and right.
    measurements: [Option<Measurement>; 2],
    /// Design settings.
    pub options: DesignOptions,
}

const CHANNELS: [&str; 2] = ["left", "right"];

fn channel_index(name: &str) -> Result<usize, String> {
    CHANNELS
        .iter()
        .position(|c| c.eq_ignore_ascii_case(name.trim()))
        .ok_or_else(|| format!("channel {name:?}: left or right"))
}

/// Change the engine's settings and wait (≤ 2 s) until its status shows them.
pub(crate) fn set_and_wait(engine: &Engine, s: crate::config::Settings) {
    let want = s.clone();
    engine.update_settings(s);
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while engine.settings() != want && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn write_wav(path: &Path, rate: u32, channels: &[Vec<f64>]) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: channels.len() as u16,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
    let n = channels.iter().map(Vec::len).max().unwrap_or(0);
    for i in 0..n {
        for c in channels {
            w.write_sample(c.get(i).copied().unwrap_or(0.0) as f32)
                .map_err(|e| e.to_string())?;
        }
    }
    w.finalize().map_err(|e| e.to_string())
}

impl Room {
    /// The default place for measurements and corrections: `<config dir>/cadenza/room`.
    pub fn default_dir() -> PathBuf {
        crate::config::Config::default_path()
            .parent()
            .map(|p| p.join("room"))
            .unwrap_or_else(|| PathBuf::from("room"))
    }

    /// Room correction kept in `dir`, recording through `capture`. Earlier measurements
    /// found there are loaded.
    pub fn new(dir: PathBuf, capture: CaptureFactory) -> Self {
        let mut measurements: [Option<Measurement>; 2] = [None, None];
        for (i, c) in CHANNELS.iter().enumerate() {
            if let Ok((Some(rate), mut ch)) =
                cadenza_dsp::convolution::load_impulse_response(&dir.join(format!("{c}.wav")))
            {
                if let Some(ir) = ch.drain(..).next() {
                    measurements[i] = Some(Measurement {
                        ir,
                        rate,
                        peak_dbfs: f64::NAN,
                    });
                }
            }
        }
        Self {
            dir,
            capture,
            measurements,
            options: DesignOptions::default(),
        }
    }

    /// Where measurements and corrections are kept.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Microphones the capture backend sees.
    pub fn inputs(&self) -> Vec<String> {
        capture::input_devices()
    }

    /// Play a sweep through `engine` on one channel and record it. Blocks for the
    /// sweep's length plus a few seconds. The engine's queue is replaced; its settings
    /// are restored afterwards.
    pub fn measure(&mut self, engine: &Engine, req: &MeasureRequest) -> Result<(), String> {
        let ch = channel_index(&req.channel)?;
        if !(1.0..=30.0).contains(&req.seconds) || !(-60.0..=0.0).contains(&req.level_db) {
            return Err("seconds 1–30 and level_db −60…0, please".into());
        }
        let mut mic = (self.capture)(&req.input)?;
        let saved = engine.settings();
        // The room as it is: no EQ, correction or headphone processing while measuring.
        let mut plain = saved.clone();
        plain.dsp.convolution.clear();
        plain.dsp.eq_enabled = false;
        plain.dsp.headphones = cadenza_dsp::HeadphoneMode::Off;
        plain.dsp.bit_perfect = false;
        set_and_wait(engine, plain);
        let result = self.run_measurement(engine, &mut *mic, ch, req);
        set_and_wait(engine, saved);
        let m = result?;
        let _ = std::fs::create_dir_all(&self.dir);
        write_wav(
            &self.dir.join(format!("{}.wav", CHANNELS[ch])),
            m.rate,
            std::slice::from_ref(&m.ir),
        )?;
        self.measurements[ch] = Some(m);
        Ok(())
    }

    fn run_measurement(
        &self,
        engine: &Engine,
        mic: &mut dyn Capture,
        ch: usize,
        req: &MeasureRequest,
    ) -> Result<Measurement, String> {
        let rate = mic.start(48000)?;
        let sweep = Sweep::new(rate, 10.0, (rate as f64 * 0.45).min(22000.0), req.seconds);
        let gain = 10f64.powf(req.level_db / 20.0);
        // Half a second of quiet, the sweep, a second and a half for the room to decay.
        let pre = rate as usize / 2;
        let post = rate as usize * 3 / 2;
        let frames = pre + sweep.signal.len() + post;
        let mut stereo = vec![0.0; frames * 2];
        for (i, v) in sweep.signal.iter().enumerate() {
            stereo[(pre + i) * 2 + ch] = v * gain;
        }
        let rec = play_and_record(
            engine,
            mic,
            stereo,
            rate,
            &format!("room-sweep-{}", CHANNELS[ch]),
            &format!("Measurement sweep ({})", CHANNELS[ch]),
        )?;
        let peak = rec.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let peak_dbfs = 20.0 * peak.max(1e-12).log10();
        if peak_dbfs < -60.0 {
            return Err(format!(
                "the microphone heard nothing (peak {peak_dbfs:.0} dBFS): check the input and the volume"
            ));
        }
        if peak_dbfs > -0.5 {
            log::warn!("room: the microphone clipped ({peak_dbfs:.1} dBFS); measure lower");
        }
        Ok(Measurement {
            ir: sweep.impulse(&rec, 5.0, 500.0),
            rate,
            peak_dbfs,
        })
    }

    /// Measure the DAC's noise floor with `input` (see [`noise`]) and make it the
    /// `dac_noise_floor_db` setting, which the `auto` dither chooses from.
    pub fn measure_noise(
        &mut self,
        engine: &Engine,
        input: &str,
    ) -> Result<noise::NoiseFloor, String> {
        let mut mic = (self.capture)(input)?;
        let m = noise::measure(engine, &mut *mic)?;
        let mut s = engine.settings();
        s.dsp.dac_noise_floor_db = Some((m.floor_db * 10.0).round() / 10.0);
        set_and_wait(engine, s);
        Ok(m)
    }

    /// Forget a channel's measurement.
    pub fn forget(&mut self, channel: &str) -> Result<(), String> {
        let ch = channel_index(channel)?;
        self.measurements[ch] = None;
        let _ = std::fs::remove_file(self.dir.join(format!("{}.wav", CHANNELS[ch])));
        Ok(())
    }

    /// The corrections for the measured channels (left, right).
    fn designs(&self) -> [Option<room::Correction>; 2] {
        let d = |m: &Option<Measurement>| {
            m.as_ref()
                .map(|m| room::design(&m.ir, m.rate, &self.options))
        };
        [d(&self.measurements[0]), d(&self.measurements[1])]
    }

    /// Everything the UI shows: curves at 200 frequencies (relative to each speaker's
    /// 200 Hz – 2 kHz level), the target, the options.
    pub fn report(&self, engine: &Engine) -> Value {
        let freqs = room::log_grid(20.0, 20000.0, 200);
        let designs = self.designs();
        let channels: Vec<Value> = self
            .measurements
            .iter()
            .zip(&designs)
            .enumerate()
            .filter_map(|(i, (m, d))| {
                let (m, d) = (m.as_ref()?, d.as_ref()?);
                let rel = |v: Vec<f64>| -> Vec<f64> {
                    v.into_iter()
                        .map(|x| ((x - d.reference_db) * 10.0).round() / 10.0)
                        .collect()
                };
                let measured = rel(room::response(&m.ir, m.rate, &freqs, 6.0));
                let corrected = rel(room::response(
                    &room::convolve(&m.ir, &d.fir),
                    m.rate,
                    &freqs,
                    6.0,
                ));
                let filter: Vec<f64> = room::response(&d.fir, m.rate, &freqs, 6.0)
                    .into_iter()
                    .map(|x| (x * 10.0).round() / 10.0)
                    .collect();
                Some(json!({
                    "channel": CHANNELS[i],
                    "rate": m.rate,
                    "peak_dbfs": m.peak_dbfs.is_finite().then_some((m.peak_dbfs * 10.0).round() / 10.0),
                    "measured": measured,
                    "corrected": corrected,
                    "filter": filter,
                }))
            })
            .collect();
        let target: Vec<f64> = freqs
            .iter()
            .map(|f| room::target_db(*f, &self.options))
            .collect();
        let conv = engine.settings().dsp.convolution;
        json!({
            "inputs": self.inputs(),
            "options": self.options,
            "freqs": freqs.iter().map(|f| f.round()).collect::<Vec<_>>(),
            "target": target,
            "channels": channels,
            "applied": (!conv.is_empty() && Path::new(&conv).starts_with(&self.dir)).then_some(conv),
            "dir": self.dir,
        })
    }

    /// Write the correction and make it the engine's convolution. With both speakers
    /// measured each gets its own filter; with one, both channels get its filter.
    pub fn apply(&mut self, engine: &Engine) -> Result<PathBuf, String> {
        let designs = self.designs();
        let (rate, filters): (u32, Vec<Vec<f64>>) = match (&designs, &self.measurements) {
            ([Some(l), Some(r)], [Some(ml), Some(mr)]) => {
                if ml.rate != mr.rate {
                    return Err("left and right were measured at different rates".into());
                }
                (ml.rate, vec![l.fir.clone(), r.fir.clone()])
            }
            ([Some(d), None], [Some(m), _]) | ([None, Some(d)], [_, Some(m)]) => {
                (m.rate, vec![d.fir.clone()])
            }
            _ => return Err("measure a speaker first".into()),
        };
        std::fs::create_dir_all(&self.dir).map_err(|e| e.to_string())?;
        // A new name each time: the engine keeps the filter it loaded per file name.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let path = self.dir.join(format!("correction-{stamp}.wav"));
        write_wav(&path, rate, &filters)?;
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let p = e.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with("correction-") && p != path {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
        let mut s = engine.settings();
        s.dsp.convolution = path.to_string_lossy().into_owned();
        set_and_wait(engine, s);
        Ok(path)
    }

    /// Stop using the correction (if the convolution setting is ours).
    pub fn remove(&mut self, engine: &Engine) {
        let mut s = engine.settings();
        if Path::new(&s.dsp.convolution).starts_with(&self.dir) {
            s.dsp.convolution.clear();
            set_and_wait(engine, s);
        }
    }
}

/// Play interleaved stereo `stereo` at `rate` through `engine` as a live track while `mic`
/// (already started) records; returns the recording. The engine's queue is replaced.
pub(crate) fn play_and_record(
    engine: &Engine,
    mic: &mut dyn Capture,
    stereo: Vec<f64>,
    rate: u32,
    id: &str,
    title: &str,
) -> Result<Vec<f64>, String> {
    let frames = stereo.len() / 2;
    mic.note_played(&stereo, rate);
    let stream = Arc::new(LiveStream::new(rate, 2, 24, title, 0).unbounded());
    live::register(id, Arc::clone(&stream));
    stream.push(stereo);
    stream.close();
    let runs = engine.status().runs_finished;
    engine.play_queue(
        vec![Track {
            id: Some(id.to_string()),
            source: Source::Live {
                live: id.to_string(),
            },
            title: title.to_string(),
            ..Default::default()
        }],
        0,
        0.0,
    );
    let limit = Duration::from_secs_f64(frames as f64 / rate as f64 + 15.0);
    let t0 = std::time::Instant::now();
    let st = engine.wait_until_finished(runs, limit);
    let played_for = t0.elapsed().as_secs_f64();
    // The output queue and the room's decay are still sounding.
    std::thread::sleep(Duration::from_millis(700));
    live::unregister(id);
    let rec = mic.stop()?;
    if st.runs_finished <= runs {
        return Err("the measurement signal did not finish playing".into());
    }
    let duration = frames as f64 / rate as f64;
    if played_for < duration * 0.5 {
        return Err(format!(
            "the measurement signal stopped after {played_for:.1} s of {duration:.1} s{}",
            st.error
                .as_ref()
                .map(|e| format!(": {e}"))
                .unwrap_or_default()
        ));
    }
    Ok(rec)
}

#[cfg(test)]
mod tests {
    use super::capture::SimulatedRoom;
    use super::*;
    use crate::config::{RateSetting, Settings};
    use crate::engine::EngineOptions;
    use cadenza_dsp::eq::{Band, BandKind, Equalizer};
    use cadenza_dsp::CoefficientCache;

    fn speaker(rate: u32, freq: f64, gain_db: f64, delay: usize) -> Vec<f64> {
        let mut x = vec![0.0; rate as usize / 2];
        x[delay] = 0.5;
        Equalizer::new(
            rate,
            1,
            &[Band {
                kind: BandKind::Peaking,
                freq,
                gain_db,
                q: 5.0,
                enabled: true,
            }],
        )
        .process(&mut x);
        x
    }

    #[test]
    fn measure_design_and_apply_through_the_engine() {
        let dir = std::env::temp_dir().join(format!("cadenza-room-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = Settings::default();
        s.output.backend = "null".into();
        s.output.rate = RateSetting::Auto;
        s.dsp.eq_enabled = true;
        s.dsp.headphones = cadenza_dsp::HeadphoneMode::CrossfeedMeier;
        let engine = Engine::start(
            s,
            EngineOptions {
                cache: CoefficientCache::disabled(),
                realtime: true,
                persist: None,
            },
        );
        // The simulated microphone hears the left speaker with a 50 Hz mode, the right
        // one with a 70 Hz mode.
        let rate = 48000;
        let room = SimulatedRoom::new(
            rate,
            speaker(rate, 50.0, 9.0, 300),
            speaker(rate, 70.0, 7.0, 280),
        );
        let mut r = Room::new(dir.clone(), room.factory());
        for ch in ["left", "right"] {
            r.measure(
                &engine,
                &MeasureRequest {
                    channel: ch.into(),
                    seconds: 1.5,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        // The engine's own settings are back.
        let back = engine.settings();
        assert!(back.dsp.eq_enabled);
        assert_eq!(
            back.dsp.headphones,
            cadenza_dsp::HeadphoneMode::CrossfeedMeier
        );
        // The report: both channels, their modes visible, corrected flat in the band.
        let rep = r.report(&engine);
        let freqs: Vec<f64> = rep["freqs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let chans = rep["channels"].as_array().unwrap();
        assert_eq!(chans.len(), 2);
        for (c, mode) in chans.iter().zip([50.0, 70.0]) {
            let curve = |k: &str| -> Vec<f64> {
                c[k].as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            };
            let (m, k) = (curve("measured"), curve("corrected"));
            let at = |v: &[f64], f: f64| {
                let i = freqs.iter().position(|x| *x >= f).unwrap();
                v[i]
            };
            assert!(at(&m, mode) > 5.0, "{}: {}", c["channel"], at(&m, mode));
            let worst = freqs
                .iter()
                .zip(&k)
                .filter(|(f, _)| **f >= 30.0 && **f <= 400.0)
                .map(|(_, v)| v.abs())
                .fold(0.0, f64::max);
            assert!(worst < 2.0, "{}: {worst}", c["channel"]);
        }
        // Apply: a stereo WAV becomes the convolution.
        let path = r.apply(&engine).unwrap();
        assert!(path.starts_with(&dir));
        assert_eq!(engine.settings().dsp.convolution, path.to_string_lossy());
        let (wrate, wch) = cadenza_dsp::convolution::load_impulse_response(&path).unwrap();
        assert_eq!((wrate, wch.len()), (Some(rate), 2));
        assert_eq!(r.report(&engine)["applied"], json!(path.to_string_lossy()));
        // Measurements survive a restart.
        let r2 = Room::new(
            dir.clone(),
            SimulatedRoom::new(rate, vec![1.0], vec![1.0]).factory(),
        );
        assert_eq!(r2.report(&engine)["channels"].as_array().unwrap().len(), 2);
        r.remove(&engine);
        assert!(engine.settings().dsp.convolution.is_empty());
        // Errors: a deaf microphone, a bad channel.
        let mut deaf = Room::new(
            dir.join("x"),
            SimulatedRoom::new(rate, vec![0.0], vec![0.0]).factory(),
        );
        let e = deaf
            .measure(
                &engine,
                &MeasureRequest {
                    seconds: 1.0,
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(e.contains("heard nothing"), "{e}");
        assert!(deaf
            .measure(
                &engine,
                &MeasureRequest {
                    channel: "centre".into(),
                    ..Default::default()
                }
            )
            .is_err());
        engine.shutdown();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn noise_floor_through_the_engine_sets_the_auto_dither() {
        let mut s = Settings::default();
        s.output.backend = "null".into();
        s.output.rate = RateSetting::Auto;
        s.output.bits = 16;
        s.dsp.dither = "auto".into();
        let engine = Engine::start(
            s,
            EngineOptions {
                cache: CoefficientCache::disabled(),
                realtime: true,
                persist: None,
            },
        );
        // Both speakers at half level: the tone reaches the microphone at −20 dBFS, with
        // the simulated input's white noise (±1e-5) on top.
        let rate = 48000;
        let room = SimulatedRoom::new(rate, vec![0.5], vec![0.5]);
        let mut r = Room::new(std::env::temp_dir().join("cadenza-noise-x"), room.factory());
        let m = r.measure_noise(&engine, "").unwrap();
        let rms = 2e-5 / 12f64.sqrt();
        let steps = 20_000;
        let df = (20_000.0 - 20.0) / steps as f64;
        let aw: f64 = (0..steps)
            .map(|i| cadenza_dsp::dither::a_weighting(20.0 + (i as f64 + 0.5) * df).powi(2) * df)
            .sum();
        let want = 10.0 * (rms * rms * aw / (rate as f64 / 2.0) / 0.005).log10() - 20.0;
        assert!(
            (m.floor_db - want).abs() < 1.0,
            "{:.1} vs {want:.1}",
            m.floor_db
        );
        assert!(
            (m.tone_at_input_db + 20.0).abs() < 0.2,
            "{}",
            m.tone_at_input_db
        );
        // The setting is stored and the other settings are back.
        let st = engine.settings();
        assert_eq!(st.dsp.dither, "auto");
        let floor = st.dsp.dac_noise_floor_db.unwrap();
        assert!((floor - m.floor_db).abs() < 0.06);
    }
}
