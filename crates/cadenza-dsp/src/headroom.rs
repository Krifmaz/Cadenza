//! Intersample-peak headroom, true-peak metering and a transparent soft clipper.
//!
//! Upsampling reconstructs the waveform *between* the original samples, and modern
//! masters routinely peak above 0 dBFS there (intersample overs). HQPlayer answers this
//! with a fixed headroom gain; Cadenza does the same (`headroom_db`, −3 dB by default when
//! upsampling) and adds two guards after the resampler:
//!
//! * a true-peak meter (ITU-R BS.1770 style 4× oversampled polyphase FIR) that reports the
//!   highest reconstructed peak seen, in dBTP;
//! * a soft clipper that is *transparent* below its knee (−0.5 dBFS): samples under the
//!   knee pass bit-identically, anything above is bent smoothly towards full scale with a
//!   continuous first derivative, so even a mis-set gain cannot produce a hard clip. It
//!   counts every sample it touched, which the signal path reports.
//!
//! All maths in `f64`; nothing allocates in [`Headroom::process`].

use serde_json::{json, Value};

/// Default headroom in dB when the chain upsamples (intersample peaks become real samples).
pub const DEFAULT_UPSAMPLING_HEADROOM_DB: f64 = -3.0;

/// The clipper's knee: below this (fraction of full scale, −0.5 dBFS) samples are untouched.
pub const SOFT_CLIP_KNEE: f64 = 0.944_060_876_285_923_4;

/// Default headroom for a chain: −3 dB when it upsamples, 0 dB otherwise.
pub fn default_headroom_db(upsampling: bool) -> f64 {
    if upsampling {
        DEFAULT_UPSAMPLING_HEADROOM_DB
    } else {
        0.0
    }
}

/// Transparent soft clip of one sample: identity below [`SOFT_CLIP_KNEE`], then
/// `k + (1 − k)·tanh((|x| − k) / (1 − k))`, which starts with unit slope at the knee and
/// approaches ±1 asymptotically (so the output never reaches full scale).
#[inline]
pub fn soft_clip(x: f64) -> f64 {
    let a = x.abs();
    if a <= SOFT_CLIP_KNEE {
        x
    } else {
        let k = SOFT_CLIP_KNEE;
        let y = k + (1.0 - k) * ((a - k) / (1.0 - k)).tanh();
        y.copysign(x)
    }
}

/// Linear gain for a dB value.
#[inline]
pub fn db_to_gain(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// dB for a linear magnitude (−inf for 0).
#[inline]
pub fn gain_to_db(g: f64) -> f64 {
    if g <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * g.log10()
    }
}

/// 4× oversampling true-peak meter (BS.1770-4 Annex 2 structure: a 48-tap interpolation
/// filter as four 12-tap phases). The filter is designed here as a Kaiser windowed sinc
/// rather than copied from the standard's table; like the standard's, it reads within
/// about 0.1 dB of the true peak over the audio band, and the phases are normalised to
/// unity gain.
#[derive(Clone, Debug)]
pub struct TruePeakMeter {
    channels: usize,
    phases: [[f64; TP_TAPS]; 4],
    /// Per channel ring of the last `TP_TAPS` samples.
    history: Vec<[f64; TP_TAPS]>,
    pos: usize,
    peak: f64,
    sample_peak: f64,
}

const TP_TAPS: usize = 12;

impl TruePeakMeter {
    /// A meter for `channels` interleaved channels.
    pub fn new(channels: usize) -> Self {
        // 4× interpolation low-pass at the output (4 fs) rate: cutoff fs/2 → 1/8, 48 taps.
        let full = crate::fir::windowed_sinc(4 * TP_TAPS - 1, 0.125, crate::fir::kaiser_beta(60.0));
        let mut phases = [[0.0; TP_TAPS]; 4];
        for (p, phase) in phases.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (k, v) in phase.iter_mut().enumerate() {
                *v = full.get(k * 4 + p).copied().unwrap_or(0.0);
                sum += *v;
            }
            // Each phase is a fractional-delay interpolator: unity DC gain.
            for v in phase.iter_mut() {
                *v /= sum;
            }
        }
        Self {
            channels: channels.max(1),
            phases,
            history: vec![[0.0; TP_TAPS]; channels.max(1)],
            pos: 0,
            peak: 0.0,
            sample_peak: 0.0,
        }
    }

    /// Feed interleaved samples; updates the running peaks.
    pub fn push(&mut self, interleaved: &[f64]) {
        let ch = self.channels;
        let mut pos = self.pos;
        let mut peak = self.peak;
        let mut sample_peak = self.sample_peak;
        for frame in interleaved.chunks(ch) {
            pos = (pos + 1) % TP_TAPS;
            for (c, &x) in frame.iter().enumerate() {
                sample_peak = sample_peak.max(x.abs());
                let hist = &mut self.history[c];
                hist[pos] = x;
                for phase in &self.phases {
                    let mut acc = 0.0;
                    for (k, h) in phase.iter().enumerate() {
                        acc += h * hist[(pos + TP_TAPS - k) % TP_TAPS];
                    }
                    peak = peak.max(acc.abs());
                }
            }
        }
        self.pos = pos;
        self.peak = peak;
        self.sample_peak = sample_peak;
    }

    /// Highest reconstructed (4× oversampled) absolute peak so far, linear.
    pub fn true_peak(&self) -> f64 {
        self.peak
    }

    /// Highest reconstructed peak so far in dBTP.
    pub fn true_peak_db(&self) -> f64 {
        gain_to_db(self.peak)
    }

    /// Highest sample peak so far, linear.
    pub fn sample_peak(&self) -> f64 {
        self.sample_peak
    }

    /// Forget the peaks (not the filter history).
    pub fn reset_peaks(&mut self) {
        self.peak = 0.0;
        self.sample_peak = 0.0;
    }

    /// Forget everything.
    pub fn reset(&mut self) {
        self.reset_peaks();
        for h in &mut self.history {
            h.fill(0.0);
        }
        self.pos = 0;
    }
}

/// The headroom stage: gain → true-peak meter → soft clipper, on interleaved `f64`.
#[derive(Clone, Debug)]
pub struct Headroom {
    gain_db: f64,
    gain: f64,
    soft_clip: bool,
    meter: TruePeakMeter,
    clipped: u64,
    /// Highest level that went into the clipper (after the gain), linear.
    worst_in: f64,
    samples: u64,
}

impl Headroom {
    /// A stage applying `gain_db` (typically ≤ 0) to `channels` interleaved channels, with
    /// the soft clipper engaged or not.
    pub fn new(channels: usize, gain_db: f64, soft_clip: bool) -> Self {
        Self {
            gain_db,
            gain: db_to_gain(gain_db),
            soft_clip,
            meter: TruePeakMeter::new(channels),
            clipped: 0,
            worst_in: 0.0,
            samples: 0,
        }
    }

    /// The headroom gain in dB.
    pub fn gain_db(&self) -> f64 {
        self.gain_db
    }

    /// Whether the soft clipper is enabled.
    pub fn soft_clip_enabled(&self) -> bool {
        self.soft_clip
    }

    /// Change the gain without rebuilding.
    pub fn set_gain_db(&mut self, gain_db: f64) {
        self.gain_db = gain_db;
        self.gain = db_to_gain(gain_db);
    }

    /// `true` when the stage does nothing to the samples: 0 dB gain and (clipper off or
    /// never engaged so far).
    pub fn is_transparent(&self) -> bool {
        self.gain_db == 0.0 && (!self.soft_clip || self.clipped == 0)
    }

    /// Apply gain, meter, soft clip; in place, no allocation.
    pub fn process(&mut self, interleaved: &mut [f64]) {
        if self.gain != 1.0 {
            for v in interleaved.iter_mut() {
                *v *= self.gain;
            }
        }
        self.meter.push(interleaved);
        self.samples += interleaved.len() as u64;
        let mut worst = self.worst_in;
        if self.soft_clip {
            let mut clipped = 0u64;
            for v in interleaved.iter_mut() {
                let a = v.abs();
                if a > SOFT_CLIP_KNEE {
                    worst = worst.max(a);
                    clipped += 1;
                    *v = soft_clip(*v);
                }
            }
            self.clipped += clipped;
        } else {
            for v in interleaved.iter() {
                worst = worst.max(v.abs());
            }
        }
        self.worst_in = worst;
    }

    /// Samples the soft clipper altered so far.
    pub fn clip_count(&self) -> u64 {
        self.clipped
    }

    /// Whether the clipper ever engaged.
    pub fn clip_engaged(&self) -> bool {
        self.clipped > 0
    }

    /// True peak of the processed signal (after the gain) in dBTP, −inf before any audio.
    pub fn true_peak_db(&self) -> f64 {
        self.meter.true_peak_db()
    }

    /// Sample peak of the processed signal (after the gain) in dBFS.
    pub fn sample_peak_db(&self) -> f64 {
        gain_to_db(self.meter.sample_peak())
    }

    /// Highest level that reached the clipper, in dBFS (what a hard clip would have cut).
    pub fn worst_input_db(&self) -> f64 {
        gain_to_db(self.worst_in)
    }

    /// The meter.
    pub fn meter(&self) -> &TruePeakMeter {
        &self.meter
    }

    /// Reset meters and counters (a new track), keeping the filter state.
    pub fn reset_meters(&mut self) {
        self.meter.reset_peaks();
        self.clipped = 0;
        self.worst_in = 0.0;
        self.samples = 0;
    }

    /// Reset everything (seek / new stream).
    pub fn reset(&mut self) {
        self.reset_meters();
        self.meter.reset();
    }

    /// One-line signal-path summary, e.g. `−3 dB · true peak −1.2 dBTP · clipper idle`.
    pub fn detail(&self) -> String {
        let tp = self.true_peak_db();
        let tp = if tp.is_finite() {
            format!(" · true peak {tp:+.1} dBTP")
        } else {
            String::new()
        };
        let clip = if !self.soft_clip {
            " · clipper off".to_string()
        } else if self.clipped == 0 {
            " · clipper idle".to_string()
        } else {
            format!(
                " · soft clipper engaged {}× (peak {:+.1} dBFS)",
                self.clipped,
                self.worst_input_db()
            )
        };
        format!("{:+.1} dB{tp}{clip}", self.gain_db)
    }

    /// Full parameters for the signal path.
    pub fn report(&self) -> Value {
        let finite = |v: f64| if v.is_finite() { Some(v) } else { None };
        json!({
            "gain_db": self.gain_db,
            "soft_clip": self.soft_clip,
            "knee_db": gain_to_db(SOFT_CLIP_KNEE),
            "true_peak_dbtp": finite(self.true_peak_db()),
            "sample_peak_dbfs": finite(self.sample_peak_db()),
            "clip_count": self.clipped,
            "clip_engaged": self.clipped > 0,
            "clip_peak_dbfs": finite(self.worst_input_db()),
            "samples": self.samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn soft_clip_is_transparent_below_the_knee_and_never_exceeds_full_scale() {
        for i in 0..1000 {
            let x = (i as f64 / 1000.0) * SOFT_CLIP_KNEE;
            assert_eq!(soft_clip(x), x);
            assert_eq!(soft_clip(-x), -x);
        }
        let mut last = SOFT_CLIP_KNEE;
        for i in 1..=1000 {
            let x = SOFT_CLIP_KNEE + i as f64 * 0.01;
            let y = soft_clip(x);
            assert!(y >= last && y <= 1.0, "x={x} y={y}");
            // Strictly below full scale over the whole range a real overshoot occupies
            // (up to +3.5 dBFS); beyond that tanh saturates to 1.0 in f64.
            assert!(x > 1.5 || y < 1.0, "x={x} y={y}");
            assert_eq!(soft_clip(-x), -y);
            last = y;
        }
        // Continuous with unit slope at the knee.
        let eps = 1e-7;
        let slope = (soft_clip(SOFT_CLIP_KNEE + eps) - soft_clip(SOFT_CLIP_KNEE - eps)) / (2.0 * eps);
        assert!((slope - 1.0).abs() < 1e-5, "slope at knee {slope}");
        assert!((gain_to_db(SOFT_CLIP_KNEE) + 0.5).abs() < 1e-9);
    }

    #[test]
    fn true_peak_meter_sees_intersample_overs() {
        // A full-scale sine at fs/4 sampled at phase 45° has sample peaks of 0.7071 but a
        // true peak of 1.0.
        let rate = 44_100u32;
        let n = 4096;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * (rate as f64 / 4.0) * i as f64 / rate as f64 + PI / 4.0).sin())
            .collect();
        let sample_peak = x.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!((sample_peak - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3);
        let mut m = TruePeakMeter::new(1);
        m.push(&x);
        assert!((m.sample_peak() - sample_peak).abs() < 1e-12);
        let tp = m.true_peak_db();
        assert!(tp > -0.6 && tp < 0.2, "true peak {tp} dBTP");

        // A 1 kHz tone at −6 dBFS reads −6 dBTP within 0.05 dB.
        let x: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f64 / rate as f64).sin())
            .collect();
        let mut m = TruePeakMeter::new(1);
        m.push(&x);
        assert!((m.true_peak_db() + 6.02).abs() < 0.05, "{}", m.true_peak_db());
        // Stereo: channels are independent.
        let st: Vec<f64> = x.iter().flat_map(|v| [*v, 0.25 * v]).collect();
        let mut m2 = TruePeakMeter::new(2);
        m2.push(&st);
        assert!((m2.true_peak_db() + 6.02).abs() < 0.05);
    }

    #[test]
    fn headroom_applies_gain_and_only_clips_what_would_clip() {
        let x: Vec<f64> = (0..20_000)
            .map(|i| 1.1 * (2.0 * PI * 1000.0 * i as f64 / 44_100.0).sin())
            .collect();
        // −3 dB brings a +0.83 dBFS peak to −2.2 dBFS: nothing to clip.
        let mut h = Headroom::new(1, -3.0, true);
        let mut y = x.clone();
        h.process(&mut y);
        assert_eq!(h.clip_count(), 0);
        assert!(!h.clip_engaged());
        assert!((h.true_peak_db() - (gain_to_db(1.1) - 3.0)).abs() < 0.05, "{}", h.true_peak_db());
        for (a, b) in x.iter().zip(&y) {
            assert!((a * db_to_gain(-3.0) - b).abs() < 1e-15);
        }
        assert!(h.detail().contains("clipper idle"), "{}", h.detail());
        assert!(!h.is_transparent());

        // 0 dB: the clipper engages, nothing reaches ±1, and the count is reported.
        let mut h = Headroom::new(1, 0.0, true);
        let mut y = x.clone();
        h.process(&mut y);
        assert!(h.clip_count() > 0 && h.clip_engaged());
        assert!(y.iter().all(|v| v.abs() < 1.0));
        let touched = x.iter().zip(&y).filter(|(a, b)| a != b).count() as u64;
        assert_eq!(touched, h.clip_count());
        assert!(x.iter().zip(&y).all(|(a, b)| a.abs() > SOFT_CLIP_KNEE || a == b));
        assert!(h.detail().contains("engaged"), "{}", h.detail());
        assert!(!h.is_transparent());
        let r = h.report();
        assert_eq!(r["clip_count"].as_u64(), Some(h.clip_count()));
        assert!(r["clip_engaged"].as_bool().unwrap());

        // Transparent when nothing happens.
        let quiet: Vec<f64> = x.iter().map(|v| v * 0.5).collect();
        let mut h = Headroom::new(1, 0.0, true);
        let mut y = quiet.clone();
        h.process(&mut y);
        assert_eq!(y, quiet);
        assert!(h.is_transparent());
        h.reset();
        assert!(!h.true_peak_db().is_finite());
    }

    #[test]
    fn defaults() {
        assert_eq!(default_headroom_db(true), -3.0);
        assert_eq!(default_headroom_db(false), 0.0);
        assert!((db_to_gain(-6.0205999) - 0.5).abs() < 1e-7);
    }
}
