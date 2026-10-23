//! Measuring the DAC's noise floor with an input device, for the `auto` dither.
//!
//! The engine plays a 997 Hz tone at −20 dBFS, then digital silence (dither off, so the
//! DAC receives exact zeros). The input records both; the noise in the silence, A-weighted
//! over 20 Hz – 20 kHz, relative to the tone gives the floor in dBFS of the DAC without
//! knowing the input's own gain. What is measured includes the input's noise too, so it is
//! an upper bound: a DAC quieter than the ADC measures as the ADC.

use super::capture::Capture;
use super::{play_and_record, set_and_wait};
use crate::engine::Engine;
use cadenza_dsp::dither::a_weighting;
use cadenza_dsp::fft::RealFft;
use serde::Serialize;

/// Test tone frequency (Hz): not a divisor of common rates, so it never sits on a bin pattern.
pub const TONE_HZ: f64 = 997.0;
/// Test tone level (dBFS, sine-referenced).
pub const TONE_DBFS: f64 = -20.0;

/// Silence before the tone, in seconds (lets the input settle).
const PRE_S: f64 = 0.5;
/// Length of the tone, in seconds.
const TONE_S: f64 = 2.0;
/// Gap between the tone and the measured silence, in seconds.
const GAP_S: f64 = 1.0;
/// Length of the measured silence, in seconds.
const QUIET_S: f64 = 3.0;
/// Silence after the measurement, in seconds.
const POST_S: f64 = 0.5;

/// A noise-floor measurement.
#[derive(Clone, Debug, Serialize)]
pub struct NoiseFloor {
    /// The floor in dBFS, A-weighted, 20 Hz – 20 kHz (what `auto` dither uses).
    pub floor_db: f64,
    /// The tone's level at the input (dBFS of the input).
    pub tone_at_input_db: f64,
    /// The noise at the input (dBFS of the input, A-weighted).
    pub noise_at_input_db: f64,
    /// The input's sample rate.
    pub rate: u32,
}

/// The signal played: interleaved stereo at `rate`, the tone on both channels.
pub fn test_signal(rate: u32) -> Vec<f64> {
    let r = rate as f64;
    let (pre, tone) = ((PRE_S * r) as usize, (TONE_S * r) as usize);
    let total = ((PRE_S + TONE_S + GAP_S + QUIET_S + POST_S) * r) as usize;
    let amp = 10f64.powf(TONE_DBFS / 20.0);
    let fade = (0.01 * r) as usize;
    let mut out = vec![0.0; total * 2];
    for i in 0..tone {
        let env = (i.min(tone - 1 - i) as f64 / fade as f64).min(1.0);
        let v = amp * env * (2.0 * std::f64::consts::PI * TONE_HZ * i as f64 / r).sin();
        out[(pre + i) * 2] = v;
        out[(pre + i) * 2 + 1] = v;
    }
    out
}

/// Find the tone and the silence after it in a recording and measure both. `tone_dbfs` is
/// the tone's level at the DAC.
pub fn analyse(rec: &[f64], rate: u32, tone_dbfs: f64) -> Result<NoiseFloor, String> {
    let r = rate as f64;
    let block = (0.02 * r) as usize;
    let env: Vec<f64> = rec
        .chunks(block)
        .map(|c| (c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    let peak = env.iter().cloned().fold(0.0, f64::max);
    if peak < 1e-4 {
        return Err(format!(
            "the input heard nothing (tone at {:.0} dBFS): check the input and the volume",
            20.0 * (peak * 2f64.sqrt()).max(1e-12).log10()
        ));
    }
    let loud: Vec<usize> = (0..env.len()).filter(|&i| env[i] > peak / 4.0).collect();
    let (first, last) = (loud[0], *loud.last().unwrap());
    let tone_blocks = last + 1 - first;
    if (tone_blocks as f64 * 0.02 - TONE_S).abs() > 0.5 {
        return Err(format!(
            "could not find the test tone (heard {:.1} s of sound, expected {TONE_S} s)",
            tone_blocks as f64 * 0.02
        ));
    }
    let tone_start = (first * block) as f64 / r;
    let tone_end = ((last + 1) * block) as f64 / r;
    // The middle of the tone, and the silence from a second after it.
    let seg = |from: f64, len: f64| -> Option<&[f64]> {
        let (a, b) = ((from * r) as usize, ((from + len) * r) as usize);
        (b <= rec.len()).then(|| &rec[a..b])
    };
    let tone = seg(tone_start + 0.25, TONE_S - 0.5).ok_or("the recording is too short")?;
    let quiet = seg(tone_end + GAP_S, QUIET_S - 1.0)
        .ok_or("the recording stopped before the silence was over")?;
    let n = 1usize << (tone.len().min(quiet.len()).ilog2());
    let spectrum = |x: &[f64]| -> Vec<f64> {
        let mut fft = RealFft::new(n);
        let mut buf: Vec<f64> = x[..n]
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
                v * w
            })
            .collect();
        let mut out = fft.spectrum();
        fft.forward(&mut buf, &mut out);
        out.iter().map(|c| c.norm_sqr()).collect()
    };
    let hz = r / n as f64;
    let ts = spectrum(tone);
    let p_tone: f64 = ts
        .iter()
        .enumerate()
        .filter(|(i, _)| (*i as f64 * hz - TONE_HZ).abs() < 30.0)
        .map(|(_, p)| p)
        .sum();
    let qs = spectrum(quiet);
    let top = 20_000f64.min(r / 2.0);
    let p_noise: f64 = qs
        .iter()
        .enumerate()
        .map(|(i, p)| (i as f64 * hz, p))
        .filter(|(f, _)| (20.0..=top).contains(f))
        .map(|(f, p)| p * a_weighting(f).powi(2))
        .sum();
    let rel = 10.0 * (p_noise / p_tone).max(1e-30).log10();
    // The tone's level at the input, from its RMS (sine-referenced dBFS).
    let rms = (tone.iter().map(|v| v * v).sum::<f64>() / tone.len() as f64).sqrt();
    let tone_at_input_db = 20.0 * (rms * 2f64.sqrt()).log10();
    Ok(NoiseFloor {
        floor_db: rel + tone_dbfs,
        tone_at_input_db,
        noise_at_input_db: rel + tone_at_input_db,
        rate,
    })
}

/// Measure through `engine` with `mic`. The engine plays without EQ, convolution,
/// headphone processing or dither; its settings are restored afterwards.
pub fn measure(engine: &Engine, mic: &mut dyn Capture) -> Result<NoiseFloor, String> {
    let vol = engine.status().volume_linear;
    if vol <= 1e-6 {
        return Err("the volume is at zero".into());
    }
    let saved = engine.settings();
    let mut plain = saved.clone();
    plain.dsp.convolution.clear();
    plain.dsp.eq_enabled = false;
    plain.dsp.headphones = cadenza_dsp::HeadphoneMode::Off;
    plain.dsp.bit_perfect = false;
    plain.dsp.dither = "none".into();
    set_and_wait(engine, plain);
    let result = (|| {
        let rate = mic.start(48000)?;
        let rec = play_and_record(
            engine,
            mic,
            test_signal(rate),
            rate,
            "noise-floor",
            "Noise floor measurement",
        )?;
        analyse(&rec, rate, TONE_DBFS + 20.0 * vol.log10())
    })();
    set_and_wait(engine, saved);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White noise of `rms` relative to a −20 dBFS tone: the floor is what the A-weighted
    /// 20 Hz – 20 kHz share of that noise works out to.
    fn expected_floor(rms: f64, rate: u32) -> f64 {
        let steps = 20_000;
        let df = (20_000.0 - 20.0) / steps as f64;
        let aw: f64 = (0..steps)
            .map(|i| a_weighting(20.0 + (i as f64 + 0.5) * df).powi(2) * df)
            .sum();
        let p = rms * rms * aw / (rate as f64 / 2.0);
        let tone = 0.1f64.powi(2) / 2.0;
        10.0 * (p / tone).log10() + TONE_DBFS
    }

    #[test]
    fn analysis_finds_the_tone_and_the_floor() {
        let rate = 48_000;
        let sig = test_signal(rate);
        let mut seed = 5u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * 3f64.sqrt()
        };
        for (gain, rms) in [(1.0, 1e-5), (0.3, 1e-6), (2.0, 3e-5)] {
            // An input with its own gain, 120 ms of latency and white noise.
            let mut rec = vec![0.0; rate as usize * 12 / 100];
            rec.extend(sig.chunks(2).map(|f| f[0] * gain));
            rec.extend(std::iter::repeat_n(0.0, rate as usize));
            for v in &mut rec {
                *v += noise() * rms;
            }
            let m = analyse(&rec, rate, TONE_DBFS).unwrap();
            let want = expected_floor(rms / gain, rate);
            assert!(
                (m.floor_db - want).abs() < 0.5,
                "gain {gain}: {:.2} vs {want:.2}",
                m.floor_db
            );
            assert!((m.tone_at_input_db - (TONE_DBFS + 20.0 * gain.log10())).abs() < 0.1);
        }
        assert!(analyse(&vec![0.0; rate as usize * 8], rate, TONE_DBFS).is_err());
    }
}
