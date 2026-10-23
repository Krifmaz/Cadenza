//! Loopback verification: play test signals through the engine, record what comes back on
//! an input, and show what the chain actually did instead of trusting it.
//!
//! * **Bit-perfect**: with the engine in bit-perfect mode a pseudo-random 24-bit pattern is
//!   played; the recording is searched for it and compared sample for sample. Only a
//!   digital loopback can pass (S/PDIF or AES out → in, a DAC's digital monitor, a USB
//!   interface's loopback channel); an analog path never matches, which is reported, not
//!   hidden.
//! * **THD+N**: a 997 Hz tone through the chain as configured (upsampling, dither, room
//!   correction…), measured from the recording: THD+N, THD and the harmonics, noise. Through
//!   an analog loopback this is the DAC + ADC; through a digital one, Cadenza's own chain.

use super::capture::Capture;
use super::{play_and_record, set_and_wait};
use crate::engine::Engine;
use cadenza_dsp::analysis::{self, BitCompare, ThdN};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// What to run.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct VerifyRequest {
    /// The input device (empty: the default input).
    pub input: String,
    /// Run the bit-perfect check.
    pub bit_perfect: bool,
    /// Run the THD+N measurement.
    pub thd_n: bool,
    /// Test tone level (dBFS).
    pub level_db: f64,
    /// Test tone length (s).
    pub seconds: f64,
}

impl Default for VerifyRequest {
    fn default() -> Self {
        Self {
            input: String::new(),
            bit_perfect: true,
            thd_n: true,
            level_db: -3.0,
            seconds: 3.0,
        }
    }
}

/// The bit-perfect result.
#[derive(Clone, Debug, Serialize)]
pub struct BitResult {
    /// The comparison.
    #[serde(flatten)]
    pub compare: BitCompare,
    /// Whether the engine reported its chain bit-perfect while playing.
    pub engine_bit_perfect: Option<bool>,
    /// The engine's signal path summary while playing.
    pub signal_path: Option<String>,
    /// Plain-language verdict.
    pub verdict: String,
}

/// The THD+N result.
#[derive(Clone, Debug, Serialize)]
pub struct ThdResult {
    /// The measurement.
    #[serde(flatten)]
    pub measurement: ThdN,
    /// The engine's signal path summary while playing.
    pub signal_path: Option<String>,
}

/// Everything measured.
#[derive(Clone, Debug, Serialize)]
pub struct VerifyReport {
    /// The input's rate.
    pub rate: u32,
    /// The bit-perfect check, if run.
    pub bit_perfect: Option<BitResult>,
    /// The THD+N measurement, if run.
    pub thd_n: Option<ThdResult>,
}

/// Bits of the bit-perfect pattern.
pub const PATTERN_BITS: u32 = 24;

/// Watch the engine while a measurement plays: the signal path of the track `id`.
fn watch(
    engine: &Engine,
    id: &str,
) -> (
    Arc<AtomicBool>,
    std::thread::JoinHandle<Option<(bool, String)>>,
) {
    let stop = Arc::new(AtomicBool::new(false));
    let (e, id, s) = (engine.clone(), id.to_string(), Arc::clone(&stop));
    let seen = Mutex::new(None);
    let h = std::thread::spawn(move || {
        while !s.load(Ordering::Relaxed) {
            let st = e.status();
            if st.track.as_ref().and_then(|t| t.id.as_deref()) == Some(id.as_str()) {
                if let Some(p) = &st.signal_path {
                    *seen.lock().unwrap() = Some((p.bit_perfect, p.label.clone()));
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        seen.into_inner().unwrap()
    });
    (stop, h)
}

/// The bit-perfect pass: play the pattern and compare the recording sample for sample;
/// returns the capture rate and the result.
fn bit_pass(engine: &Engine, mic: &mut dyn Capture) -> Result<(u32, BitResult), String> {
    let rate = mic.start(48_000)?;
    let r = rate as usize;
    let sent = analysis::pattern(2 * r, PATTERN_BITS, 0x00CA_DE2A);
    let scale = (1u64 << (PATTERN_BITS - 1)) as f64;
    // Silence, the pattern on the left (the input records its first channel) and its
    // negation on the right, silence.
    let (pre, post) = (r * 3 / 10, r / 2);
    let mut stereo = vec![0.0; (pre + sent.len() + post) * 2];
    for (i, &v) in sent.iter().enumerate() {
        stereo[(pre + i) * 2] = v as f64 / scale;
        stereo[(pre + i) * 2 + 1] = -(v as f64) / scale;
    }
    let id = "verify-bits";
    let (stop, seen) = watch(engine, id);
    let rec = play_and_record(engine, mic, stereo, rate, id, "Bit-perfect check");
    stop.store(true, Ordering::Relaxed);
    let seen = seen.join().ok().flatten();
    let rec = rec?;
    let got: Vec<i32> = rec
        .iter()
        .map(|&x| (x * scale).round().clamp(-scale, scale - 1.0) as i32)
        .collect();
    let compare = analysis::bit_compare(&sent, &got, 32);
    let verdict = match (&compare, seen.as_ref().map(|s| s.0)) {
        (c, _) if c.bit_perfect => format!(
            "bit-perfect: all {} samples came back unchanged",
            c.compared
        ),
        (c, _) if c.offset.is_some() => format!(
            "not bit-perfect: {} of {} samples differ (first at {})",
            c.mismatches,
            c.compared,
            c.first_mismatch.unwrap_or(0)
        ),
        (_, Some(false)) => {
            "not bit-perfect: the engine's chain alters the samples (see the signal path)".into()
        }
        _ => "the pattern was not found in the recording: an analog path, a level or rate \
              change after the engine, or the wrong input"
            .into(),
    };
    Ok((
        rate,
        BitResult {
            compare,
            engine_bit_perfect: seen.as_ref().map(|s| s.0),
            signal_path: seen.map(|s| s.1),
            verdict,
        },
    ))
}

/// The THD+N pass: play a tone through the chain and analyse the recording; returns the
/// capture rate and the result.
fn thd_pass(
    engine: &Engine,
    mic: &mut dyn Capture,
    req: &VerifyRequest,
) -> Result<(u32, ThdResult), String> {
    let rate = mic.start(48_000)?;
    let r = rate as f64;
    let amp = 10f64.powf(req.level_db / 20.0);
    let n = (req.seconds * r) as usize;
    let (pre, post) = ((0.3 * r) as usize, (0.5 * r) as usize);
    let fade = (0.01 * r) as usize;
    let mut stereo = vec![0.0; (pre + n + post) * 2];
    for i in 0..n {
        let env = (i.min(n - 1 - i) as f64 / fade as f64).min(1.0);
        let v = amp * env * (2.0 * std::f64::consts::PI * 997.0 * i as f64 / r).sin();
        stereo[(pre + i) * 2] = v;
        stereo[(pre + i) * 2 + 1] = v;
    }
    let id = "verify-tone";
    let (stop, seen) = watch(engine, id);
    let rec = play_and_record(engine, mic, stereo, rate, id, "THD+N measurement");
    stop.store(true, Ordering::Relaxed);
    let seen = seen.join().ok().flatten();
    let rec = rec?;
    // The tone in the recording: where the level is within 6 dB of its peak; measure the
    // middle of it.
    let block = (0.01 * r) as usize;
    let env: Vec<f64> = rec
        .chunks(block)
        .map(|c| c.iter().fold(0.0f64, |m, v| m.max(v.abs())))
        .collect();
    let peak = env.iter().cloned().fold(0.0, f64::max);
    if peak < 1e-4 {
        return Err("the input heard nothing: check the input and the volume".into());
    }
    let loud: Vec<usize> = (0..env.len()).filter(|&i| env[i] > peak / 2.0).collect();
    let (a, b) = (loud[0] * block, (loud[loud.len() - 1] + 1) * block);
    let margin = (b - a) / 8;
    let m = analysis::thd_n(&rec[a + margin..b - margin], r, 997.0)
        .ok_or("the recording is too short to measure")?;
    Ok((
        rate,
        ThdResult {
            measurement: m,
            signal_path: seen.map(|s| s.1),
        },
    ))
}

/// Run the requested checks through `engine` with `mic` (a fresh capture per pass). The
/// bit-perfect pass switches the engine to bit-perfect mode for its duration; the THD+N
/// pass uses the settings as they are. The settings are restored afterwards.
pub fn verify(
    engine: &Engine,
    open: &dyn Fn() -> Result<Box<dyn Capture>, String>,
    req: &VerifyRequest,
) -> Result<VerifyReport, String> {
    if !(-60.0..=0.0).contains(&req.level_db) || !(1.0..=20.0).contains(&req.seconds) {
        return Err("level_db −60…0 and seconds 1–20, please".into());
    }
    let mut report = VerifyReport {
        rate: 0,
        bit_perfect: None,
        thd_n: None,
    };
    if req.bit_perfect {
        let saved = engine.settings();
        let mut bp = saved.clone();
        bp.dsp.bit_perfect = true;
        set_and_wait(engine, bp);
        let r = open().and_then(|mut mic| bit_pass(engine, &mut *mic));
        set_and_wait(engine, saved);
        let (rate, b) = r?;
        report.rate = rate;
        report.bit_perfect = Some(b);
    }
    if req.thd_n {
        let mut mic = open()?;
        let (rate, t) = thd_pass(engine, &mut *mic, req)?;
        report.rate = rate;
        report.thd_n = Some(t);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RateSetting, Settings};
    use crate::engine::EngineOptions;
    use cadenza_dsp::CoefficientCache;

    /// A perfect digital loopback: reads back what the null output wrote (its `capture:`
    /// file), optionally through a cubic nonlinearity (an imperfect analog path).
    struct FileLoopback {
        engine: Engine,
        path: std::path::PathBuf,
        start: u64,
        cubic: f64,
    }

    impl Capture for FileLoopback {
        fn start(&mut self, rate: u32) -> Result<u32, String> {
            self.start = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
            Ok(rate)
        }
        fn stop(&mut self) -> Result<Vec<f64>, String> {
            let fmt = self.engine.status().output_format.ok_or("no output")?;
            let bytes = std::fs::read(&self.path).map_err(|e| e.to_string())?;
            let scale = (1u64 << (fmt.bits - 1)) as f64;
            let ch = fmt.channels.max(1) as usize;
            Ok(bytes[self.start as usize..]
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64 / scale)
                .step_by(ch)
                .map(|x| x + self.cubic * x * x * x)
                .collect())
        }
    }

    fn engine(path: &std::path::Path) -> Engine {
        let mut s = Settings::default();
        s.output.backend = "null".into();
        s.output.device = format!("capture:{}", path.display());
        s.output.rate = RateSetting::Fixed(48_000);
        s.output.bits = 24;
        s.dsp.eq_enabled = true;
        s.dsp.eq = vec![cadenza_dsp::Band {
            kind: cadenza_dsp::BandKind::Peaking,
            freq: 3000.0,
            gain_db: -2.0,
            q: 1.0,
            enabled: true,
            slope_db: 12,
        }];
        Engine::start(
            s,
            EngineOptions {
                cache: CoefficientCache::disabled(),
                realtime: true,
                persist: None,
            },
        )
    }

    #[test]
    fn a_digital_loopback_verifies_bit_perfect_and_measures_the_chain() {
        let path = std::env::temp_dir().join(format!("cadenza-verify-{}.raw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let e = engine(&path);
        let open = || -> Result<Box<dyn Capture>, String> {
            Ok(Box::new(FileLoopback {
                engine: e.clone(),
                path: path.clone(),
                start: 0,
                cubic: 0.0,
            }))
        };
        let r = verify(&e, &open, &VerifyRequest::default()).unwrap();
        let b = r.bit_perfect.unwrap();
        assert!(b.compare.bit_perfect, "{b:?}");
        assert_eq!(b.engine_bit_perfect, Some(true));
        assert!(b.verdict.starts_with("bit-perfect"), "{}", b.verdict);
        // The chain as configured: EQ (−2 dB at 3 kHz barely touches 997 Hz), 24-bit TPDF.
        let t = r.thd_n.unwrap().measurement;
        assert!((t.frequency - 997.0).abs() < 0.5);
        assert!((t.level_db + 3.0).abs() < 0.5, "{}", t.level_db);
        assert!(t.thd_n_db < -125.0, "THD+N {}", t.thd_n_db);
        // The settings came back (EQ on, not bit-perfect).
        let s = e.settings();
        assert!(!s.dsp.bit_perfect && s.dsp.eq_enabled);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_changed_path_is_caught() {
        let path =
            std::env::temp_dir().join(format!("cadenza-verify-x-{}.raw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let e = engine(&path);
        // A distorting path: the pattern is never found, and the THD shows the cubic term.
        let k = 0.01;
        let open = || -> Result<Box<dyn Capture>, String> {
            Ok(Box::new(FileLoopback {
                engine: e.clone(),
                path: path.clone(),
                start: 0,
                cubic: k,
            }))
        };
        let r = verify(&e, &open, &VerifyRequest::default()).unwrap();
        let b = r.bit_perfect.unwrap();
        assert!(!b.compare.bit_perfect);
        assert!(b.verdict.contains("not found"), "{}", b.verdict);
        let t = r.thd_n.unwrap().measurement;
        // y = x + k x³ puts k a³ / 4 in the 3rd harmonic: THD = k a² / 4 for the tone's
        // amplitude a where it meets the nonlinearity (the EQ takes a little off 997 Hz).
        let a = 10f64.powf(t.level_db / 20.0);
        let want = 20.0 * (k * a * a / 4.0).log10();
        assert!((t.thd_db - want).abs() < 0.5, "THD {} vs {want}", t.thd_db);
        let _ = std::fs::remove_file(&path);
    }
}
