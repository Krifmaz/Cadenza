//! Headphone listening: crossfeed and HRTF virtual speakers for stereo.
//!
//! On headphones each ear hears only its own channel, which puts hard-panned sounds
//! inside the head. Two remedies:
//!
//! - **Crossfeed** (Bauer's method, as bs2b does it): each ear also gets the other channel,
//!   low-passed and a little delayed by the filter, and its own channel slightly
//!   high-boosted so the overall balance stays flat. Three classic settings.
//! - **HRTF**: the stereo pair played through measured head-related impulse responses of
//!   two loudspeakers at ±30°, ±45° or ±60° (the MIT KEMAR dummy-head set,
//!   diffuse-field equalised), so the sound comes from in front instead of inside.
//!
//! Both work on stereo only and run at the source rate before the upsampler.

use crate::convolution::Convolver;
use crate::kemar;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// What the headphone stage does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HeadphoneMode {
    /// Nothing.
    #[default]
    Off,
    /// bs2b's default: 700 Hz, 4.5 dB of feed.
    CrossfeedBs2b,
    /// Chu Moy's: 700 Hz, 6 dB.
    CrossfeedChuMoy,
    /// Jan Meier's: 650 Hz, 9.5 dB (the strongest).
    CrossfeedMeier,
    /// Virtual speakers at ±30° (a standard stereo triangle).
    #[serde(rename = "hrtf-30")]
    Hrtf30,
    /// Virtual speakers at ±45°.
    #[serde(rename = "hrtf-45")]
    Hrtf45,
    /// Virtual speakers at ±60° (wide).
    #[serde(rename = "hrtf-60")]
    Hrtf60,
}

impl HeadphoneMode {
    /// Every mode with its id and label (settings menus).
    pub const ALL: [(HeadphoneMode, &'static str, &'static str); 7] = [
        (HeadphoneMode::Off, "off", "Off"),
        (
            HeadphoneMode::CrossfeedBs2b,
            "crossfeed-bs2b",
            "Crossfeed · bs2b (700 Hz, 4.5 dB)",
        ),
        (
            HeadphoneMode::CrossfeedChuMoy,
            "crossfeed-chu-moy",
            "Crossfeed · Chu Moy (700 Hz, 6 dB)",
        ),
        (
            HeadphoneMode::CrossfeedMeier,
            "crossfeed-meier",
            "Crossfeed · Jan Meier (650 Hz, 9.5 dB)",
        ),
        (HeadphoneMode::Hrtf30, "hrtf-30", "HRTF · speakers at ±30°"),
        (HeadphoneMode::Hrtf45, "hrtf-45", "HRTF · speakers at ±45°"),
        (HeadphoneMode::Hrtf60, "hrtf-60", "HRTF · speakers at ±60°"),
    ];

    /// The id (`crossfeed-bs2b` …).
    pub fn id(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|m| m.0 == self)
            .map_or("off", |m| m.1)
    }

    /// A readable name.
    pub fn label(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|m| m.0 == self)
            .map_or("Off", |m| m.2)
    }

    /// Parse an id.
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .find(|m| m.1.eq_ignore_ascii_case(id.trim()))
            .map(|m| m.0)
    }
}

/// One crossfeed channel pair's filters (bs2b's first-order sections).
#[derive(Clone, Debug)]
struct Crossfeed {
    // Low-pass of the opposite channel: y = a0·x + b1·y₋₁
    lo_a0: f64,
    lo_b1: f64,
    // High boost of the own channel: y = a0·x + a1·x₋₁ + b1·y₋₁
    hi_a0: f64,
    hi_a1: f64,
    hi_b1: f64,
    /// Per channel: low-pass state, high-boost input and output state.
    lo: [f64; 2],
    hi_x: [f64; 2],
    hi_y: [f64; 2],
}

impl Crossfeed {
    /// Cut-off `fc` Hz and `feed` dB at sample rate `fs`.
    fn new(fc: f64, feed_db: f64, fs: f64) -> Self {
        let gb_lo = feed_db * -5.0 / 6.0 - 3.0;
        let gb_hi = feed_db / 6.0 - 3.0;
        let g_lo = 10f64.powf(gb_lo / 20.0);
        let g_hi = 1.0 - 10f64.powf(gb_hi / 20.0);
        let fc_hi = fc * 2f64.powf((gb_lo - 20.0 * g_hi.log10()) / 12.0);
        // A centred low tone comes out at unity.
        let gain = 1.0 / (1.0 - g_hi + g_lo);
        let x = (-2.0 * PI * fc / fs).exp();
        let (lo_a0, lo_b1) = (g_lo * (1.0 - x) * gain, x);
        let x = (-2.0 * PI * fc_hi / fs).exp();
        Self {
            lo_a0,
            lo_b1,
            hi_a0: (1.0 - g_hi * (1.0 - x)) * gain,
            hi_a1: -x * gain,
            hi_b1: x,
            lo: [0.0; 2],
            hi_x: [0.0; 2],
            hi_y: [0.0; 2],
        }
    }

    fn process(&mut self, buf: &mut [f64]) {
        for f in buf.chunks_exact_mut(2) {
            let input = [f[0], f[1]];
            for (c, out) in f.iter_mut().enumerate() {
                let own = input[c];
                let other = input[1 - c];
                self.lo[c] = self.lo_a0 * other + self.lo_b1 * self.lo[c];
                let hi = self.hi_a0 * own + self.hi_a1 * self.hi_x[c] + self.hi_b1 * self.hi_y[c];
                self.hi_x[c] = own;
                self.hi_y[c] = hi;
                *out = hi + self.lo[c];
            }
        }
    }

    fn reset(&mut self) {
        self.lo = [0.0; 2];
        self.hi_x = [0.0; 2];
        self.hi_y = [0.0; 2];
    }
}

/// The KEMAR responses for a speaker at `azimuth` to the right: (ipsilateral ear,
/// contralateral ear), at 44.1 kHz.
fn kemar_pair(azimuth: u32) -> (&'static [i16], &'static [i16]) {
    match azimuth {
        45 => (&kemar::KEMAR_45_RIGHT, &kemar::KEMAR_45_LEFT),
        60 => (&kemar::KEMAR_60_RIGHT, &kemar::KEMAR_60_LEFT),
        _ => (&kemar::KEMAR_30_RIGHT, &kemar::KEMAR_30_LEFT),
    }
}

/// Band-limited resampling of a short impulse response from `from` to `to` Hz, keeping
/// its frequency response (at a higher rate the same response takes proportionally more,
/// smaller taps).
pub fn resample_ir(h: &[f64], from: f64, to: f64) -> Vec<f64> {
    if (from - to).abs() < 1e-9 {
        return h.to_vec();
    }
    let r = to / from;
    // Low-pass at the lower Nyquist, a little inside it.
    let c = 0.95 * r.min(1.0);
    // Kernel half-width in input samples.
    let half = 16.0 / c;
    // The output is delayed by `half` so the kernel's leading half is not cut off (a
    // common delay: magnitudes and the ears' relative timing are unchanged).
    let n_out = ((h.len() as f64 + 2.0 * half) * r).ceil() as usize;
    let mut out: Vec<f64> = (0..n_out)
        .map(|n| {
            let t = n as f64 / r - half;
            let lo = (t - half).floor().max(0.0) as usize;
            let hi = ((t + half).ceil() as usize).min(h.len().saturating_sub(1));
            let mut s = 0.0;
            for (k, hk) in h.iter().enumerate().take(hi + 1).skip(lo) {
                let x = t - k as f64;
                let a = c * x;
                let sinc = if a.abs() < 1e-12 {
                    1.0
                } else {
                    (PI * a).sin() / (PI * a)
                };
                let w = x / half;
                let win = if w.abs() >= 1.0 {
                    0.0
                } else {
                    0.5 + 0.5 * (PI * w).cos()
                };
                s += hk * c * sinc * win;
            }
            s / r
        })
        .collect();
    while out.last().is_some_and(|v| v.abs() < 1e-12) {
        out.pop();
    }
    out
}

/// The largest magnitude of `a + b`'s frequency response (sampled finely).
fn peak_gain(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(b.len());
    let points = 512;
    (0..points)
        .map(|i| {
            let w = PI * i as f64 / points as f64;
            let (mut re, mut im) = (0.0, 0.0);
            for k in 0..n {
                let v = a.get(k).unwrap_or(&0.0) + b.get(k).unwrap_or(&0.0);
                re += v * (w * k as f64).cos();
                im -= v * (w * k as f64).sin();
            }
            (re * re + im * im).sqrt()
        })
        .fold(0.0, f64::max)
}

/// Stereo through two virtual speakers: four convolutions.
#[derive(Debug)]
struct Hrtf {
    /// ipsi(L→L), contra(L→R), contra(R→L), ipsi(R→R)
    conv: [Convolver; 4],
}

impl Hrtf {
    fn new(azimuth: u32, fs: f64) -> Self {
        let (ipsi, contra) = kemar_pair(azimuth);
        let to = |v: &[i16]| -> Vec<f64> {
            let h: Vec<f64> = v.iter().map(|x| *x as f64 / 32768.0).collect();
            resample_ir(&h, 44100.0, fs)
        };
        let (mut i, mut c) = (to(ipsi), to(contra));
        // No louder than the original anywhere: a centred sound reaches each ear through
        // both paths, so their sum's peak is brought to 0 dB.
        let g = 1.0 / peak_gain(&i, &c).max(1e-9);
        for v in i.iter_mut().chain(c.iter_mut()) {
            *v *= g;
        }
        let len = i.len().max(c.len());
        i.resize(len, 0.0);
        c.resize(len, 0.0);
        let block = crate::convolution::choose_block(len);
        Self {
            conv: [
                Convolver::with_block(&i, block),
                Convolver::with_block(&c, block),
                Convolver::with_block(&c, block),
                Convolver::with_block(&i, block),
            ],
        }
    }

    fn run(&mut self, buf: &[f64], flush: bool) -> Vec<f64> {
        let frames = buf.len() / 2;
        let l: Vec<f64> = buf.iter().step_by(2).copied().collect();
        let r: Vec<f64> = buf.iter().skip(1).step_by(2).copied().collect();
        let mut o: [Vec<f64>; 4] = Default::default();
        let ins = [&l, &l, &r, &r];
        for (k, conv) in self.conv.iter_mut().enumerate() {
            o[k].reserve(frames + 64);
            conv.push(ins[k], &mut o[k]);
            if flush {
                conv.flush(&mut o[k]);
            }
        }
        let n = o.iter().map(Vec::len).min().unwrap_or(0);
        let mut out = Vec::with_capacity(n * 2);
        for (((ll, lr), rl), rr) in o[0].iter().zip(&o[1]).zip(&o[2]).zip(&o[3]).take(n) {
            out.push(ll + rl); // left ear: L ipsi + R contra
            out.push(lr + rr); // right ear: L contra + R ipsi
        }
        out
    }

    fn reset(&mut self) {
        for c in &mut self.conv {
            c.reset();
        }
    }
}

/// The running stage.
#[derive(Debug)]
pub struct HeadphoneStage {
    mode: HeadphoneMode,
    crossfeed: Option<Box<CrossfeedState>>,
    hrtf: Option<Hrtf>,
}

/// Boxed so the stage stays small.
#[derive(Debug)]
struct CrossfeedState(Crossfeed);

impl HeadphoneStage {
    /// A stage for `mode` at sample rate `fs`; `None` when the mode is off.
    pub fn new(mode: HeadphoneMode, fs: u32) -> Option<Self> {
        let fs = fs as f64;
        let (crossfeed, hrtf) = match mode {
            HeadphoneMode::Off => return None,
            HeadphoneMode::CrossfeedBs2b => (Some(Crossfeed::new(700.0, 4.5, fs)), None),
            HeadphoneMode::CrossfeedChuMoy => (Some(Crossfeed::new(700.0, 6.0, fs)), None),
            HeadphoneMode::CrossfeedMeier => (Some(Crossfeed::new(650.0, 9.5, fs)), None),
            HeadphoneMode::Hrtf30 => (None, Some(Hrtf::new(30, fs))),
            HeadphoneMode::Hrtf45 => (None, Some(Hrtf::new(45, fs))),
            HeadphoneMode::Hrtf60 => (None, Some(Hrtf::new(60, fs))),
        };
        Some(Self {
            mode,
            crossfeed: crossfeed.map(|c| Box::new(CrossfeedState(c))),
            hrtf,
        })
    }

    /// The mode.
    pub fn mode(&self) -> HeadphoneMode {
        self.mode
    }

    /// Frames between input and output (the HRTF convolution's block; crossfeed has none).
    pub fn latency_frames(&self) -> usize {
        self.hrtf.as_ref().map_or(0, |h| h.conv[0].block())
    }

    /// Process interleaved stereo; with `flush`, also what the filters still hold.
    pub fn run(&mut self, buf: &[f64], flush: bool) -> Vec<f64> {
        if let Some(c) = &mut self.crossfeed {
            let mut v = buf.to_vec();
            c.0.process(&mut v);
            return v;
        }
        match &mut self.hrtf {
            Some(h) => h.run(buf, flush),
            None => buf.to_vec(),
        }
    }

    /// Forget all state (seek).
    pub fn reset(&mut self) {
        if let Some(c) = &mut self.crossfeed {
            c.0.reset();
        }
        if let Some(h) = &mut self.hrtf {
            h.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gain of a stereo-in / stereo-out stage at `f` Hz: (left-in → left-out,
    /// left-in → right-out), measured with a sine in the left channel only.
    fn gains(mode: HeadphoneMode, fs: u32, f: f64) -> (f64, f64) {
        let mut st = HeadphoneStage::new(mode, fs).unwrap();
        let n = fs as usize; // one second
        let mut input = vec![0.0; n * 2];
        for i in 0..n {
            input[2 * i] = (2.0 * PI * f * i as f64 / fs as f64).sin();
        }
        let out = st.run(&input, true);
        assert_eq!(out.len(), input.len(), "stream length kept");
        let rms = |c: usize| {
            let v: Vec<f64> = out.iter().skip(c).step_by(2).skip(n / 2).copied().collect();
            (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt() * 2f64.sqrt()
        };
        (rms(0), rms(1))
    }

    /// Output level of the same sine in both channels.
    fn mono_gain(mode: HeadphoneMode, f: f64) -> f64 {
        let mut st = HeadphoneStage::new(mode, 44100).unwrap();
        let n = 44100usize;
        let input: Vec<f64> = (0..2 * n)
            .map(|i| (2.0 * PI * f * (i / 2) as f64 / 44100.0).sin())
            .collect();
        let out = st.run(&input, true);
        let v: Vec<f64> = out.iter().step_by(2).skip(n / 2).copied().collect();
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt() * 2f64.sqrt()
    }

    fn db(x: f64) -> f64 {
        20.0 * x.log10()
    }

    #[test]
    fn crossfeed_feeds_lows_across_and_keeps_the_balance() {
        for (mode, feed) in [
            (HeadphoneMode::CrossfeedBs2b, 4.5),
            (HeadphoneMode::CrossfeedChuMoy, 6.0),
            (HeadphoneMode::CrossfeedMeier, 9.5),
        ] {
            let (own_lo, other_lo) = gains(mode, 44100, 100.0);
            let (own_hi, other_hi) = gains(mode, 44100, 8000.0);
            // At low frequencies the other ear gets the channel `feed` dB down…
            let sep = db(own_lo) - db(other_lo);
            assert!((sep - feed).abs() < 1.0, "{mode:?}: {sep:.2} dB");
            // …and much less of it in the treble.
            assert!(db(own_hi) - db(other_hi) > feed + 8.0, "{mode:?}");
            // A centred low tone at unity; centred treble a little lower (as with bs2b
            // itself: up to about 2.5 dB with the strongest setting).
            let (m_lo, m_hi) = (mono_gain(mode, 100.0), mono_gain(mode, 8000.0));
            assert!(db(m_lo).abs() < 0.1, "{mode:?}: {m_lo}");
            assert!((-3.0..0.0).contains(&db(m_hi)), "{mode:?}: {m_hi}");
        }
        assert!(HeadphoneStage::new(HeadphoneMode::Off, 44100).is_none());
    }

    #[test]
    fn hrtf_puts_the_left_speaker_left_at_every_rate() {
        for fs in [44100u32, 48000, 96000, 352800] {
            let (l, r) = gains(HeadphoneMode::Hrtf30, fs, 3000.0);
            // The near ear is clearly louder at 3 kHz (head shadow).
            assert!(db(l) - db(r) > 5.0, "{fs}: {:.1} dB", db(l) - db(r));
            // Nothing louder than full scale.
            assert!(l < 1.05 && r < 1.05, "{fs}: {l} {r}");
        }
        // Wider speakers: more shadow at 3 kHz.
        let d = |m| {
            let (l, r) = gains(m, 48000, 3000.0);
            db(l) - db(r)
        };
        assert!(d(HeadphoneMode::Hrtf60) > d(HeadphoneMode::Hrtf30));
    }

    #[test]
    fn ids_round_trip() {
        for (m, id, _) in HeadphoneMode::ALL {
            assert_eq!(HeadphoneMode::from_id(id), Some(m));
            assert_eq!(m.id(), id);
            // The serialised name is the id.
            assert_eq!(serde_json::to_value(m).unwrap(), serde_json::json!(id));
        }
        assert_eq!(HeadphoneMode::from_id("nope"), None);
    }

    #[test]
    fn ir_resampling_keeps_the_frequency_response() {
        // A 5-tap smoothing filter: its gain at 1 kHz must not change with the rate.
        let h = [0.1, 0.2, 0.4, 0.2, 0.1];
        let g = |h: &[f64], fs: f64| {
            let w = 2.0 * PI * 1000.0 / fs;
            let (re, im) = h.iter().enumerate().fold((0.0, 0.0), |(re, im), (k, v)| {
                (re + v * (w * k as f64).cos(), im - v * (w * k as f64).sin())
            });
            (re * re + im * im).sqrt()
        };
        let g0 = g(&h, 44100.0);
        for to in [48000.0, 96000.0, 352800.0] {
            let r = resample_ir(&h, 44100.0, to);
            assert!(
                (g(&r, to) / g0 - 1.0).abs() < 0.02,
                "{to}: {} vs {g0}",
                g(&r, to)
            );
        }
    }
}
