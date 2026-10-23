//! Multi-room synchronisation: the pure parts (no sockets, no threads), so they can be
//! tested exactly.
//!
//! - [`ClockSync`]: the offset between the sender's clock and an endpoint's, from ping
//!   round trips (the fastest recent round trip is the most trustworthy).
//! - [`Timeline`]: which stream position the *leader's* DAC presents when, fitted by least
//!   squares over its recent reports. Its slope is the leader's true sample rate.
//! - [`Follower`]: on every other endpoint, the error between when its DAC will present a
//!   frame and when the leader presents it, turned into a playback ratio (a PI loop) or,
//!   for large errors, a jump.
//! - [`Resampler`]: plays PCM at that ratio (a few hundred ppm at most) with a windowed
//!   sinc, so a follower tracks the leader's clock without clicks. DSD cannot be
//!   resampled; a DSD follower slips whole words instead.

use std::collections::VecDeque;

/// Offset of an endpoint's clock from ours: `remote = local + offset`.
#[derive(Clone, Debug, Default)]
pub struct ClockSync {
    /// `(round trip, offset)` of recent pings, nanoseconds.
    samples: VecDeque<(i64, i64)>,
}

impl ClockSync {
    /// Most recent pings kept; the offset comes from the one with the shortest round trip.
    const KEEP: usize = 16;

    /// A ping sent at local `t0`, answered at remote `server`, received at local `t3`.
    pub fn add(&mut self, t0: i64, server: i64, t3: i64) {
        if t3 < t0 {
            return;
        }
        let rtt = t3 - t0;
        let offset = server - (t0 + rtt / 2);
        if self.samples.len() == Self::KEEP {
            self.samples.pop_front();
        }
        self.samples.push_back((rtt, offset));
    }

    /// The offset from the fastest recent round trip (`None` before the first).
    pub fn offset(&self) -> Option<i64> {
        self.samples.iter().min_by_key(|s| s.0).map(|s| s.1)
    }

    /// Its round trip (the uncertainty is half of it).
    pub fn best_rtt(&self) -> Option<i64> {
        self.samples.iter().map(|s| s.0).min()
    }
}

/// A straight line `time = at + (position − origin) × frame_ns` through recent
/// `(position, time)` reports.
#[derive(Clone, Debug, Default)]
pub struct Timeline {
    /// `(stream position, clock ns)` reports, oldest first.
    points: VecDeque<(u64, i64)>,
}

/// Where a stream position plays: `at` for `position`, `frame_ns` per frame after it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    /// Stream position.
    pub position: u64,
    /// When it plays, ns, in whichever clock the anchor is expressed.
    pub at: i64,
    /// Nanoseconds per frame (the leader's real rate).
    pub frame_ns: f64,
}

impl Anchor {
    /// When `position` (fractional) plays.
    pub fn time_of(&self, position: f64) -> f64 {
        self.at as f64 + (position - self.position as f64) * self.frame_ns
    }
}

impl Timeline {
    /// Most reports kept for the fit.
    const KEEP: usize = 64;
    /// One report per 100 ms is kept, so the fit spans about six seconds.
    const MIN_GAP_NS: i64 = 100_000_000;

    /// The leader presented `position` at `time`.
    pub fn add(&mut self, position: u64, time: i64) {
        if self
            .points
            .back()
            .is_some_and(|p| p.0 >= position || time - p.1 < Self::MIN_GAP_NS)
        {
            // Only forward progress makes a point (paused / repeated reports are skipped).
            return;
        }
        if self.points.len() == Self::KEEP {
            self.points.pop_front();
        }
        self.points.push_back((position, time));
    }

    /// Reports kept.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// No reports yet.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Forget (after a seek or pause).
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// The fitted line through the reports, anchored at the newest; `nominal_ns` (one
    /// frame at the nominal rate) is used until two reports span enough time.
    pub fn anchor(&self, nominal_ns: f64) -> Option<Anchor> {
        let last = *self.points.back()?;
        let first = *self.points.front()?;
        let span = (last.0 - first.0) as f64;
        let frame_ns = if self.points.len() >= 4 && span * nominal_ns > 2e8 {
            let n = self.points.len() as f64;
            let (mx, my) = self.points.iter().fold((0.0, 0.0), |(a, b), p| {
                (a + (p.0 - first.0) as f64, b + (p.1 - first.1) as f64)
            });
            let (mx, my) = (mx / n, my / n);
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for p in &self.points {
                let x = (p.0 - first.0) as f64 - mx;
                let y = (p.1 - first.1) as f64 - my;
                sxy += x * y;
                sxx += x * x;
            }
            let slope = sxy / sxx;
            // A wild fit (pause, clock step) falls back to nominal.
            if (slope / nominal_ns - 1.0).abs() < 1e-3 {
                slope
            } else {
                nominal_ns
            }
        } else {
            nominal_ns
        };
        // Anchor at the line's value at the newest position (less noisy than the point).
        let n = self.points.len() as f64;
        let mean_err = self
            .points
            .iter()
            .map(|p| p.1 as f64 - (last.1 as f64 + (p.0 as f64 - last.0 as f64) * frame_ns))
            .sum::<f64>()
            / n;
        Some(Anchor {
            position: last.0,
            at: (last.1 as f64 + mean_err).round() as i64,
            frame_ns,
        })
    }
}

/// What a follower does with its error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Correction {
    /// Play at this ratio (input frames per output frame).
    Ratio(f64),
    /// Drop this many input frames (we are late).
    Drop(usize),
    /// Play this many frames of silence first (we are early).
    Insert(usize),
}

/// The follower's control loop.
#[derive(Clone, Debug)]
pub struct Follower {
    /// Stream rate in frames per second (to turn a jump into frames).
    rate: f64,
    /// Integral term of the controller, clamped to ±20 ms·s.
    integral: f64,
    /// Smoothed error in seconds, once there is one.
    filtered: Option<f64>,
    /// Beyond this many seconds of error, jump instead of steering.
    pub jump_s: f64,
}

impl Follower {
    /// Steering gain (per second): an error of 1 ms asks for 1 ms/s × `KP` of catch-up.
    const KP: f64 = 1.0;
    /// Critically damped with `KP` (KP² / 4): settles in a few seconds, no overshoot.
    const KI: f64 = 0.25;
    /// Largest ratio deviation (1000 ppm), well below anything audible.
    const MAX_DEV: f64 = 1e-3;

    /// A follower for a stream at `rate` frames per second.
    pub fn new(rate: u32) -> Self {
        Self {
            rate: rate as f64,
            integral: 0.0,
            filtered: None,
            jump_s: 0.010,
        }
    }

    /// Start over (after a jump, a seek or a pause).
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.filtered = None;
    }

    /// The error `late_s` (positive: our DAC will present the frame after the leader's),
    /// measured every `dt_s`.
    pub fn update(&mut self, late_s: f64, dt_s: f64) -> Correction {
        if late_s.abs() > self.jump_s {
            self.reset();
            let frames = (late_s.abs() * self.rate).round() as usize;
            return if late_s > 0.0 {
                Correction::Drop(frames)
            } else {
                Correction::Insert(frames)
            };
        }
        // Light smoothing: device queue readings jitter by a few frames.
        let e = match self.filtered {
            Some(f) => f + 0.2 * (late_s - f),
            None => late_s,
        };
        self.filtered = Some(e);
        self.integral = (self.integral + e * dt_s).clamp(-0.02, 0.02);
        let dev = (Self::KP * e + Self::KI * self.integral).clamp(-Self::MAX_DEV, Self::MAX_DEV);
        Correction::Ratio(1.0 + dev)
    }
}

/// Variable-ratio resampler for interleaved PCM at ratios near 1.
pub struct Resampler {
    /// Interleaved channel count.
    channels: usize,
    /// Kernel, `PHASES + 1` rows of `TAPS`.
    kernel: Vec<f64>,
    /// Input history, interleaved.
    buf: VecDeque<f64>,
    /// Position of the next output in input frames, relative to `buf`'s first frame.
    t: f64,
    /// Input frames consumed in total (dropped from `buf` or skipped).
    consumed: f64,
}

/// Taps of the windowed-sinc interpolation kernel.
const TAPS: usize = 48;
/// Taps on either side of the interpolation point.
const HALF: usize = TAPS / 2;
/// Fractional positions the kernel is tabulated at (linearly interpolated in between).
const PHASES: usize = 512;

/// Modified Bessel function of the first kind, order 0 (for the Kaiser window).
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..50 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-17 * sum {
            break;
        }
    }
    sum
}

impl Resampler {
    /// A resampler for `channels` channels.
    pub fn new(channels: usize) -> Self {
        let beta = 9.0;
        let cutoff = 0.95;
        let mut kernel = vec![0.0; (PHASES + 1) * TAPS];
        for p in 0..=PHASES {
            let frac = p as f64 / PHASES as f64;
            for k in 0..TAPS {
                // Tap k sits at input offset (k − HALF + 1) from floor(t); distance to t:
                let x = k as f64 - (HALF as f64 - 1.0) - frac;
                let s = if x == 0.0 {
                    cutoff
                } else {
                    (std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
                };
                let w = x / HALF as f64;
                let win = if w.abs() >= 1.0 {
                    0.0
                } else {
                    bessel_i0(beta * (1.0 - w * w).sqrt()) / bessel_i0(beta)
                };
                kernel[p * TAPS + k] = s * win;
            }
            // Unit gain at DC for every phase: no low-frequency ripple as the phase moves.
            let row = &mut kernel[p * TAPS..(p + 1) * TAPS];
            let sum: f64 = row.iter().sum();
            for v in row {
                *v /= sum;
            }
        }
        let ch = channels.max(1);
        Self {
            channels: ch,
            kernel,
            // Start with half a kernel of silence so the first output is the first input.
            buf: std::iter::repeat_n(0.0, (HALF - 1) * ch).collect(),
            t: (HALF - 1) as f64,
            consumed: -((HALF - 1) as f64),
        }
    }

    /// Input frames consumed so far, as of the next output frame (fractional).
    pub fn input_position(&self) -> f64 {
        self.consumed + self.t
    }

    /// Input frames waiting (not yet fully used).
    pub fn pending_frames(&self) -> usize {
        self.buf.len() / self.channels
    }

    /// Add input.
    pub fn push(&mut self, input: &[f64]) {
        self.buf.extend(input.iter().copied());
    }

    /// Skip `frames` input frames (a jump).
    pub fn skip(&mut self, frames: usize) {
        self.t += frames as f64;
        self.trim();
    }

    /// Drop input history the next output's kernel no longer reaches.
    fn trim(&mut self) {
        let base = self.t.floor() as usize;
        let drop = base
            .saturating_sub(HALF - 1)
            .min(self.buf.len() / self.channels);
        if drop > 0 {
            self.buf.drain(..drop * self.channels);
            self.t -= drop as f64;
            self.consumed += drop as f64;
        }
    }

    /// Produce up to `max_frames` output frames at `ratio`, as far as the input reaches.
    pub fn produce(&mut self, ratio: f64, max_frames: usize, out: &mut Vec<f64>) -> usize {
        let ch = self.channels;
        let avail = self.buf.len() / ch;
        let mut n = 0;
        while n < max_frames {
            let base = self.t.floor() as usize;
            if base + HALF >= avail {
                break;
            }
            let pos = (self.t - base as f64) * PHASES as f64;
            let p = pos.floor() as usize;
            let a = pos - p as f64;
            let k0 = &self.kernel[p * TAPS..(p + 1) * TAPS];
            let k1 = &self.kernel[(p + 1).min(PHASES) * TAPS..((p + 1).min(PHASES) + 1) * TAPS];
            let start = base + 1 - HALF;
            for c in 0..ch {
                let mut s = 0.0;
                for k in 0..TAPS {
                    let w = k0[k] + a * (k1[k] - k0[k]);
                    s += w * self.buf[(start + k) * ch + c];
                }
                out.push(s);
            }
            self.t += ratio;
            n += 1;
        }
        self.trim();
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_offset_uses_the_fastest_round_trip() {
        let mut c = ClockSync::default();
        assert_eq!(c.offset(), None);
        // True offset +5000 ns; one slow, asymmetric round trip.
        c.add(1000, 1000 + 5000 + 300, 1600); // rtt 600 → offset 5000
        c.add(2000, 2000 + 5000 + 4000, 2000 + 4100); // slow return: rtt 4100
        c.add(3000, 3000 + 5000 + 100, 3200); // rtt 200 → offset 5000
        assert_eq!(c.offset(), Some(5000));
        assert_eq!(c.best_rtt(), Some(200));
    }

    #[test]
    fn timeline_fits_the_leaders_true_rate() {
        // A DAC 80 ppm fast at 44.1 kHz, reported every 10 ms with ±20 µs jitter (one in
        // ten reports is kept).
        let frame_ns = 1e9 / (44100.0 * (1.0 + 80e-6));
        let mut t = Timeline::default();
        let mut seed = 1u64;
        for i in 0..600u64 {
            let pos = 1000 + i * 441;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let jitter = ((seed >> 33) % 40_000) as i64 - 20_000;
            t.add(pos, 7_000_000 + (pos as f64 * frame_ns) as i64 + jitter);
        }
        t.add(500, 0); // backwards: ignored
        let a = t.anchor(1e9 / 44100.0).unwrap();
        assert!((a.frame_ns / frame_ns - 1.0).abs() < 5e-6, "{}", a.frame_ns);
        let truth = 7_000_000.0 + a.position as f64 * frame_ns;
        assert!(
            (a.at as f64 - truth).abs() < 10_000.0,
            "{}",
            a.at as f64 - truth
        );
        assert!((a.time_of(a.position as f64 + 441.0) - (truth + 441.0 * frame_ns)).abs() < 1e4);
    }

    #[test]
    fn follower_steers_small_errors_and_jumps_big_ones() {
        let mut f = Follower::new(48000);
        assert_eq!(f.update(0.05, 0.01), Correction::Drop(2400));
        assert_eq!(f.update(-0.02, 0.01), Correction::Insert(960));
        match f.update(0.001, 0.01) {
            Correction::Ratio(r) => assert!(r > 1.0 && r <= 1.001, "{r}"),
            c => panic!("{c:?}"),
        }
        // Simulate a follower whose DAC runs 200 ppm slow: the loop must hold it.
        let mut f = Follower::new(48000);
        let mut late = 0.0f64;
        let mut worst_after = 0.0f64;
        for step in 0..4000 {
            let dt = 0.01;
            let r = match f.update(late, dt) {
                Correction::Ratio(r) => r,
                c => panic!("{c:?}"),
            };
            // Playing input at ratio r advances input r× faster; a slow DAC loses 200 ppm.
            late += dt * (1.0 - r * (1.0 - 200e-6));
            if step > 2000 {
                worst_after = worst_after.max(late.abs());
            }
        }
        // Within a fifth of a sample at 44.1 kHz once settled.
        assert!(worst_after < 5e-6, "{worst_after}");
    }

    #[test]
    fn resampler_passes_audio_and_tracks_position() {
        // A 1 kHz sine at 48 kHz through a 1.0005 ratio: output is the same sine,
        // read 0.05 % faster, with the input position advancing by the ratio.
        let rate = 48000.0;
        let mut r = Resampler::new(1);
        let input: Vec<f64> = (0..48000)
            .map(|n| (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / rate).sin() * 0.5)
            .collect();
        r.push(&input);
        let ratio = 1.0005;
        let mut out = Vec::new();
        let n = r.produce(ratio, 40000, &mut out);
        assert_eq!(n, 40000);
        assert!((r.input_position() - 40000.0 * ratio).abs() < 1e-6);
        let mut worst = 0.0f64;
        for (i, y) in out.iter().enumerate().skip(64) {
            let x = i as f64 * ratio;
            let want = (2.0 * std::f64::consts::PI * 1000.0 * x / rate).sin() * 0.5;
            worst = worst.max((y - want).abs());
        }
        // Better than −90 dB of the signal.
        assert!(worst < 0.5 * 3e-5, "{worst}");
        // At ratio 1 it is a delay-free copy of a band-limited signal (to the filter's
        // passband ripple), two channels kept apart.
        let mut r = Resampler::new(2);
        let x: Vec<f64> = (0..4000)
            .map(|i| {
                let n = (i / 2) as f64;
                let f = if i % 2 == 0 { 3000.0 } else { 15000.0 };
                (2.0 * std::f64::consts::PI * f * n / rate).sin() * 0.5
            })
            .collect();
        r.push(&x);
        let mut out = Vec::new();
        r.produce(1.0, 1900, &mut out);
        let worst = out
            .iter()
            .zip(&x)
            .skip(128)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-4, "{worst}");
        // Skips move the position.
        let before = r.input_position();
        r.skip(10);
        assert!((r.input_position() - before - 10.0).abs() < 1e-9);
    }
}
