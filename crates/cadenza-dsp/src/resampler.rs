//! Sample-rate conversion with very long FIR filters.
//!
//! * Integer upsampling (×2 … ×32, any integer) and small-ratio conversions run the
//!   polyphase decomposition of the long FIR through the partitioned FFT engine of
//!   [`crate::convolution`]: the input block is transformed once and every one of the `L`
//!   sub-filters reads the same frequency-domain delay line.
//! * Rational conversions with a large `L` (44.1 ↔ 48 kHz families: 147/160, 320/147 …)
//!   use a time-domain polyphase bank, because computing 160 phases to keep one in 147
//!   would waste the FFT's advantage.
//!
//! The output is **time aligned** with the input: the linear-phase group delay is absorbed
//! (the first `(N−1)/2` filter outputs are skipped and `flush` pushes the tail out), so a
//! stream of `n` input frames gives exactly `ceil(n·L/M)` output frames. The processing
//! latency a player must account for is reported by [`Resampler::latency_input_frames`].

use crate::cache::CoefficientCache;
use crate::convolution::{choose_block, BlockEngine, SpectralFilter};
use crate::fir::{FilterKind, FirSpec, Phase};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Longest polyphase sub-filter the time-domain core accepts (per output sample cost).
const MAX_TIME_DOMAIN_SUBFILTER: usize = 8192;

/// Errors from [`Resampler::new`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResampleError {
    /// Input and output rates are equal; nothing to do.
    #[error("input and output rate are both {0} Hz")]
    SameRate(u32),
    /// A rate or channel count of zero.
    #[error("invalid parameters: {0}")]
    Invalid(&'static str),
}

/// Which engine a resampler uses (reported in the signal path).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Engine {
    /// Partitioned FFT polyphase (block size in samples).
    FftPolyphase {
        /// Input block size in frames.
        block: usize,
    },
    /// Direct time-domain polyphase bank.
    TimePolyphase,
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Decompose `h` (designed at `l × input rate`) into `l` sub-filters, each scaled by `l` to
/// make up for the zero stuffing, optionally reversed for straight dot products.
fn polyphase(h: &[f64], l: usize, reversed: bool) -> Vec<Vec<f64>> {
    let sub_len = h.len().div_ceil(l);
    (0..l)
        .map(|p| {
            let mut sub: Vec<f64> = (0..sub_len)
                .map(|k| h.get(k * l + p).copied().unwrap_or(0.0) * l as f64)
                .collect();
            if reversed {
                sub.reverse();
            }
            sub
        })
        .collect()
}

/// Per-channel state of the FFT engine.
struct FftCore {
    engine: BlockEngine,
    filters: Arc<Vec<SpectralFilter>>,
    l: usize,
    m: usize,
    /// Index at the `l × fs_in` rate of the next upsampled sample the engine will produce.
    u: u64,
    /// The upsampled index of the next sample to keep (decimation by `m`, offset by the delay).
    next_take: u64,
    phase_out: Vec<Vec<f64>>,
}

impl FftCore {
    fn push(&mut self, input: &[f64], out: &mut Vec<f64>) {
        let Self {
            engine,
            filters,
            l,
            m,
            u,
            next_take,
            phase_out,
        } = self;
        let (l, m) = (*l, *m);
        engine.push(input, |e| {
            for (p, buf) in phase_out.iter_mut().enumerate() {
                buf.clear();
                e.output(&filters[p], buf);
            }
            let block = e.block();
            if m == 1 {
                // Plain upsampling: interleave the phases, skipping the group delay once.
                for j in 0..block {
                    for buf in phase_out.iter() {
                        if *u >= *next_take {
                            out.push(buf[j]);
                        }
                        *u += 1;
                    }
                }
                *next_take = (*next_take).max(*u);
            } else {
                for j in 0..block {
                    for buf in phase_out.iter() {
                        if *u == *next_take {
                            out.push(buf[j]);
                            *next_take += m as u64;
                        }
                        *u += 1;
                    }
                }
            }
        });
        let _ = l;
    }

    fn reset(&mut self, delay: u64) {
        self.engine.reset();
        self.u = 0;
        self.next_take = delay;
    }
}

/// Per-channel state of the time-domain polyphase engine.
struct TimeCore {
    /// Reversed sub-filters: `sub[p][k'] = h[(S−1−k')·l + p] · l`.
    sub: Arc<Vec<Vec<f64>>>,
    l: u64,
    m: u64,
    sub_len: usize,
    /// Input history; `buf[i]` is input sample `base + i`. Starts with `S−1` zeros.
    buf: Vec<f64>,
    base: i64,
    /// Upsampled-rate index of the next output sample.
    t: u64,
}

impl TimeCore {
    fn new(sub: Arc<Vec<Vec<f64>>>, l: usize, m: usize, delay: u64) -> Self {
        let sub_len = sub[0].len();
        Self {
            sub,
            l: l as u64,
            m: m as u64,
            sub_len,
            buf: vec![0.0; sub_len - 1],
            base: -(sub_len as i64 - 1),
            t: delay,
        }
    }

    fn push(&mut self, input: &[f64], out: &mut Vec<f64>) {
        self.buf.extend_from_slice(input);
        let s = self.sub_len;
        loop {
            let n = (self.t / self.l) as i64;
            let idx = (n - self.base) as usize; // position of x[n] in buf
            if idx >= self.buf.len() {
                break;
            }
            let p = (self.t % self.l) as usize;
            let window = &self.buf[idx + 1 - s..=idx];
            let y: f64 = self.sub[p].iter().zip(window).map(|(a, b)| a * b).sum();
            out.push(y);
            self.t += self.m;
        }
        // Keep only the history the next output can still reach.
        let next_n = (self.t / self.l) as i64;
        let keep_from = (next_n - self.base) as usize;
        let drop = keep_from.saturating_sub(s - 1).min(self.buf.len());
        if drop > 0 {
            self.buf.drain(..drop);
            self.base += drop as i64;
        }
    }

    fn reset(&mut self, delay: u64) {
        self.buf.clear();
        self.buf.resize(self.sub_len - 1, 0.0);
        self.base = -(self.sub_len as i64 - 1);
        self.t = delay;
    }
}

enum Core {
    Fft(FftCore),
    Time(TimeCore),
}

impl Core {
    fn push(&mut self, input: &[f64], out: &mut Vec<f64>) {
        match self {
            Core::Fft(c) => c.push(input, out),
            Core::Time(c) => c.push(input, out),
        }
    }
    fn reset(&mut self, delay: u64) {
        match self {
            Core::Fft(c) => c.reset(delay),
            Core::Time(c) => c.reset(delay),
        }
    }
}

/// A streaming, gapless, multi-channel sample-rate converter.
pub struct Resampler {
    in_rate: u32,
    out_rate: u32,
    channels: usize,
    l: usize,
    m: usize,
    spec: FirSpec,
    engine: Engine,
    /// Group delay of the filter in upsampled samples (`(N−1)/2` for linear phase, 0 otherwise).
    delay: u64,
    cores: Vec<Core>,
    input_frames: u64,
    output_frames: u64,
    split: Vec<Vec<f64>>,
    joined: Vec<Vec<f64>>,
}

impl Resampler {
    /// Build a converter from `in_rate` to `out_rate` for `channels` interleaved channels
    /// with the catalogue filter `kind`; coefficients come from (and go to) `cache`.
    pub fn new(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        kind: FilterKind,
        attenuation_db: f64,
        cache: &CoefficientCache,
    ) -> Result<Self, ResampleError> {
        if in_rate == 0 || out_rate == 0 {
            return Err(ResampleError::Invalid("sample rate is zero"));
        }
        if channels == 0 {
            return Err(ResampleError::Invalid("channel count is zero"));
        }
        if in_rate == out_rate {
            return Err(ResampleError::SameRate(in_rate));
        }
        let g = gcd(in_rate, out_rate);
        let (l, m) = ((out_rate / g) as usize, (in_rate / g) as usize);
        let use_fft = m == 1 || l <= 4;
        let mut spec = kind.spec_for_ratio(l, m, attenuation_db);
        if !use_fft && spec.taps.div_ceil(l) > MAX_TIME_DOMAIN_SUBFILTER {
            // A 4M-tap bank at L=160 would cost 26k multiplies per output sample; cap it.
            let capped = FirSpec::from_taps(
                MAX_TIME_DOMAIN_SUBFILTER * l,
                spec.cutoff,
                spec.attenuation_db,
                spec.phase,
            );
            log::warn!(
                "{} is too long for the {l}/{m} time-domain converter; using {} taps",
                spec.taps,
                capped.taps
            );
            spec = capped;
        }
        let h = cache.get_or_design(&spec);
        Ok(Self::with_coefficients(
            in_rate, out_rate, channels, spec, &h,
        ))
    }

    /// Build a converter around an explicit linear- or minimum-phase low-pass `h` designed
    /// for `spec` (the filter must run at `L × in_rate` with its cutoff at the lower Nyquist).
    pub fn with_coefficients(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        spec: FirSpec,
        h: &[f64],
    ) -> Self {
        let g = gcd(in_rate, out_rate);
        let (l, m) = ((out_rate / g) as usize, (in_rate / g) as usize);
        let delay = match spec.phase {
            Phase::Linear => ((h.len() - 1) / 2) as u64,
            Phase::Minimum => 0,
        };
        let use_fft = m == 1 || l <= 4;
        let (engine, cores) = if use_fft {
            let sub_len = h.len().div_ceil(l);
            let block = choose_block(sub_len);
            let filters: Arc<Vec<SpectralFilter>> = Arc::new(
                polyphase(h, l, false)
                    .iter()
                    .map(|s| SpectralFilter::new(s, block))
                    .collect(),
            );
            let depth = filters[0].partitions();
            let cores = (0..channels)
                .map(|_| {
                    Core::Fft(FftCore {
                        engine: BlockEngine::new(block, depth),
                        filters: Arc::clone(&filters),
                        l,
                        m,
                        u: 0,
                        next_take: delay,
                        phase_out: vec![Vec::with_capacity(block); l],
                    })
                })
                .collect();
            (Engine::FftPolyphase { block }, cores)
        } else {
            let sub = Arc::new(polyphase(h, l, true));
            let cores = (0..channels)
                .map(|_| Core::Time(TimeCore::new(Arc::clone(&sub), l, m, delay)))
                .collect();
            (Engine::TimePolyphase, cores)
        };
        Self {
            in_rate,
            out_rate,
            channels,
            l,
            m,
            spec,
            engine,
            delay,
            cores,
            input_frames: 0,
            output_frames: 0,
            split: vec![Vec::new(); channels],
            joined: vec![Vec::new(); channels],
        }
    }

    /// Input sample rate.
    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }
    /// Output sample rate.
    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }
    /// Channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }
    /// The `L/M` ratio.
    pub fn ratio(&self) -> (usize, usize) {
        (self.l, self.m)
    }
    /// The filter actually in use.
    pub fn spec(&self) -> &FirSpec {
        &self.spec
    }
    /// Which engine runs the filter.
    pub fn engine(&self) -> Engine {
        self.engine
    }

    /// Input frames that must arrive before the output for a given instant exists: the
    /// filter's look-ahead plus block buffering. Convert with `in_rate` to seconds.
    pub fn latency_input_frames(&self) -> usize {
        let look_ahead = (self.delay as usize).div_ceil(self.l);
        let block = match self.engine {
            Engine::FftPolyphase { block } => block,
            Engine::TimePolyphase => 0,
        };
        look_ahead + block
    }

    /// Latency in seconds (see [`Resampler::latency_input_frames`]).
    pub fn latency_seconds(&self) -> f64 {
        self.latency_input_frames() as f64 / self.in_rate as f64
    }

    /// Output frames produced so far for the input frames pushed so far (gapless: the
    /// count never includes filter tails).
    pub fn frames(&self) -> (u64, u64) {
        (self.input_frames, self.output_frames)
    }

    /// Convert interleaved frames; returns whatever output is ready (possibly nothing, or
    /// several blocks). Call [`Resampler::flush`] at the end of the stream.
    pub fn push(&mut self, interleaved: &[f64]) -> Vec<f64> {
        let ch = self.channels;
        let frames = interleaved.len() / ch;
        self.input_frames += frames as u64;
        for (c, buf) in self.split.iter_mut().enumerate() {
            buf.clear();
            buf.extend(interleaved.iter().skip(c).step_by(ch).take(frames));
        }
        for (core, (inp, out)) in self
            .cores
            .iter_mut()
            .zip(self.split.iter().zip(self.joined.iter_mut()))
        {
            out.clear();
            core.push(inp, out);
        }
        self.interleave()
    }

    /// Push silence until every input frame has produced its output, then reset the
    /// filter state. Returns the remaining output.
    pub fn flush(&mut self) -> Vec<f64> {
        let expected = (self.input_frames * self.l as u64).div_ceil(self.m as u64);
        let step = match self.engine {
            Engine::FftPolyphase { block } => block,
            Engine::TimePolyphase => 256,
        };
        let zeros = vec![0.0; step * self.channels];
        let mut out = Vec::new();
        let mut guard = 0;
        while self.output_frames < expected {
            let before = self.input_frames;
            out.extend(self.push(&zeros));
            self.input_frames = before;
            guard += 1;
            assert!(guard < 1_000_000, "resampler flush did not converge");
        }
        let extra = (self.output_frames - expected) as usize;
        out.truncate(out.len() - extra * self.channels);
        self.reset();
        out
    }

    /// Forget all state (a seek, or a new stream).
    pub fn reset(&mut self) {
        for c in &mut self.cores {
            c.reset(self.delay);
        }
        self.input_frames = 0;
        self.output_frames = 0;
    }

    fn interleave(&mut self) -> Vec<f64> {
        let n = self.joined[0].len();
        debug_assert!(self.joined.iter().all(|j| j.len() == n));
        self.output_frames += n as u64;
        let mut out = vec![0.0; n * self.channels];
        for (c, buf) in self.joined.iter().enumerate() {
            for (j, v) in buf.iter().enumerate() {
                out[j * self.channels + c] = *v;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::RealFft;
    use std::f64::consts::PI;

    fn sine(freq: f64, rate: u32, seconds: f64, amp: f64) -> Vec<f64> {
        let n = (rate as f64 * seconds) as usize;
        (0..n)
            .map(|i| amp * (2.0 * PI * freq * i as f64 / rate as f64).sin())
            .collect()
    }

    fn kind(taps: usize, phase: Phase) -> FilterKind {
        FilterKind {
            phase,
            apodizing: false,
            taps,
        }
    }

    /// Hann-windowed spectrum: returns `(peak_hz, peak_amplitude, max_db_above(limit_hz) relative to peak)`.
    fn analyse(x: &[f64], rate: u32, limit_hz: f64) -> (f64, f64, f64) {
        let n = x.len().next_power_of_two() / 2;
        let x = &x[..n];
        let mut fft = RealFft::new(n);
        let mut win_sum = 0.0;
        let mut buf: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos();
                win_sum += w;
                v * w
            })
            .collect();
        let mut spec = fft.spectrum();
        fft.forward(&mut buf, &mut spec);
        let mags: Vec<f64> = spec.iter().map(|c| c.norm() * 2.0 / win_sum).collect();
        let (peak_bin, peak) =
            mags.iter()
                .enumerate()
                .fold((0, 0.0), |a, (i, &m)| if m > a.1 { (i, m) } else { a });
        let hz_per_bin = rate as f64 / n as f64;
        // Parabolic interpolation of the peak position.
        let (a, b, c) = (
            mags[peak_bin - 1].ln(),
            mags[peak_bin].ln(),
            mags[peak_bin + 1].ln(),
        );
        let frac = 0.5 * (a - c) / (a - 2.0 * b + c);
        let peak_hz = (peak_bin as f64 + frac) * hz_per_bin;
        // A pure tone's amplitude is RMS·√2; over a thousand periods this is far more
        // accurate than reading a windowed FFT bin (scalloping loss).
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let peak = rms * 2f64.sqrt();
        let limit_bin = (limit_hz / hz_per_bin).ceil() as usize;
        let worst = mags[limit_bin..].iter().cloned().fold(0.0, f64::max);
        (peak_hz, peak, 20.0 * (worst / peak).log10())
    }

    #[test]
    fn upsampling_x4_keeps_the_tone_and_has_no_images() {
        let cache = CoefficientCache::disabled();
        let mut r = Resampler::new(
            44100,
            176400,
            1,
            kind(1 << 14, Phase::Linear),
            150.0,
            &cache,
        )
        .unwrap();
        assert_eq!(r.ratio(), (4, 1));
        let x = sine(1000.0, 44100, 2.0, 0.5);
        let mut y = Vec::new();
        for chunk in x.chunks(1234) {
            y.extend(r.push(chunk));
        }
        y.extend(r.flush());
        assert_eq!(y.len(), x.len() * 4);

        // Time alignment: away from the edges the output *is* the ideal 176.4 kHz sine.
        let ideal = sine(1000.0, 176400, 2.0, 0.5);
        let margin = 176400 / 2;
        let err = y[margin..y.len() - margin]
            .iter()
            .zip(&ideal[margin..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "alignment error {err}");

        let (hz, amp, images_db) = analyse(&y[margin..], 176400, 22050.0);
        assert!((hz - 1000.0).abs() < 1.0, "peak at {hz} Hz");
        let amp_db = 20.0 * (amp / 0.5).log10();
        assert!(amp_db.abs() < 0.1, "amplitude error {amp_db} dB");
        assert!(images_db < -100.0, "images at {images_db} dB");
    }

    #[test]
    fn dc_stays_dc() {
        let cache = CoefficientCache::disabled();
        for phase in [Phase::Linear, Phase::Minimum] {
            let mut r =
                Resampler::new(48000, 384000, 2, kind(1 << 12, phase), 140.0, &cache).unwrap();
            assert_eq!(r.ratio(), (8, 1));
            let x = vec![0.25; 2 * 48000];
            let mut y = r.push(&x);
            y.extend(r.flush());
            assert_eq!(y.len(), x.len() * 8);
            let tail = &y[y.len() / 2..y.len() - 8 * 2048];
            for v in tail {
                assert!((v - 0.25).abs() < 1e-6, "{phase:?}: {v}");
            }
        }
    }

    #[test]
    fn rational_44100_to_48000_keeps_frequency() {
        let cache = CoefficientCache::disabled();
        let mut r =
            Resampler::new(44100, 48000, 1, kind(1 << 16, Phase::Linear), 120.0, &cache).unwrap();
        assert_eq!(r.ratio(), (160, 147));
        assert_eq!(r.engine(), Engine::TimePolyphase);
        let x = sine(1000.0, 44100, 1.5, 0.5);
        let mut y = Vec::new();
        for chunk in x.chunks(1000) {
            y.extend(r.push(chunk));
        }
        y.extend(r.flush());
        assert_eq!(y.len() as u64, (x.len() as u64 * 160).div_ceil(147));
        let margin = 48000 / 4;
        let (hz, amp, _) = analyse(&y[margin..], 48000, 23000.0);
        assert!((hz - 1000.0).abs() < 1.0, "peak at {hz} Hz");
        let amp_db = 20.0 * (amp / 0.5).log10();
        assert!(amp_db.abs() < 0.1, "amplitude error {amp_db} dB");
        let ideal = sine(1000.0, 48000, 1.5, 0.5);
        let err = y[margin..y.len() - margin]
            .iter()
            .zip(&ideal[margin..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "alignment error {err}");
    }

    #[test]
    fn decimation_96k_to_48k_keeps_frequency() {
        let cache = CoefficientCache::disabled();
        let mut r =
            Resampler::new(96000, 48000, 1, kind(1 << 12, Phase::Linear), 120.0, &cache).unwrap();
        assert_eq!(r.ratio(), (1, 2));
        let x = sine(5000.0, 96000, 1.0, 0.3);
        let mut y = r.push(&x);
        y.extend(r.flush());
        assert_eq!(y.len(), x.len() / 2);
        let ideal = sine(5000.0, 48000, 1.0, 0.3);
        let margin = 8000;
        let err = y[margin..y.len() - margin]
            .iter()
            .zip(&ideal[margin..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "alignment error {err}");
    }

    #[test]
    fn same_rate_is_an_error_and_latency_is_reported() {
        let cache = CoefficientCache::disabled();
        assert_eq!(
            Resampler::new(44100, 44100, 2, FilterKind::default(), 150.0, &cache).err(),
            Some(ResampleError::SameRate(44100))
        );
        let r =
            Resampler::new(44100, 88200, 2, kind(1 << 12, Phase::Linear), 150.0, &cache).unwrap();
        assert!(r.latency_input_frames() >= 1024 && r.latency_seconds() < 1.0);
    }
}
