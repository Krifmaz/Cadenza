//! Throughput benchmarks: how many seconds of audio a filter processes per wall-clock
//! second on this machine (the "realtime factor"; below 1 the machine cannot keep up).
//!
//! `cadenza bench` (in the player crate) prints these as a table; the DSD modulator path
//! uses [`run_with`] at its PCM-equivalent rates so users see where their CPU tops out.

use crate::cache::CoefficientCache;
use crate::convolution::Convolver;
use crate::fir::FilterKind;
use crate::resampler::{RateFamily, ResampleError, Resampler};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Result of one benchmark run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchResult {
    /// What ran (filter id, or `convolution`).
    pub filter: String,
    /// Input rate in Hz.
    pub in_rate: u32,
    /// Output rate in Hz.
    pub out_rate: u32,
    /// Channels processed.
    pub channels: usize,
    /// Total taps actually designed (all stages).
    pub taps: usize,
    /// Engine description (`×16 FFT polyphase · block 4096`, …).
    pub engine: String,
    /// Threads in the rayon pool.
    pub threads: usize,
    /// Seconds of audio pushed through.
    pub audio_seconds: f64,
    /// Wall-clock seconds the processing took (excluding design / cache load).
    pub wall_seconds: f64,
    /// `audio_seconds / wall_seconds`: seconds of audio per second.
    pub realtime_factor: f64,
    /// Seconds spent designing or loading the coefficients.
    pub design_seconds: f64,
}

impl BenchResult {
    /// `true` when the machine keeps up (with a 20 % margin).
    pub fn realtime(&self) -> bool {
        self.realtime_factor >= 1.2
    }
}

/// Benchmark `filter` upsampling the base rate of `out_rate`'s family (44.1 or 48 kHz;
/// 44.1 kHz for rates outside both) to `out_rate`, stereo, two seconds of audio, with the
/// default on-disk coefficient cache. Returns seconds-of-audio-per-second in
/// [`BenchResult::realtime_factor`].
pub fn run(filter: FilterKind, out_rate: u32) -> Result<BenchResult, ResampleError> {
    let in_rate = RateFamily::of(out_rate).map(RateFamily::base).unwrap_or(44_100);
    run_with(
        filter,
        in_rate,
        out_rate,
        2,
        2.0,
        crate::fir::DEFAULT_ATTENUATION_DB,
        &CoefficientCache::default_location(),
    )
}

/// Benchmark a specific conversion: `in_rate → out_rate`, `channels`, `audio_seconds` of a
/// test tone, with the filter's `attenuation_db` and coefficients from `cache`.
pub fn run_with(
    filter: FilterKind,
    in_rate: u32,
    out_rate: u32,
    channels: usize,
    audio_seconds: f64,
    attenuation_db: f64,
    cache: &CoefficientCache,
) -> Result<BenchResult, ResampleError> {
    let t0 = Instant::now();
    let mut r = Resampler::new(in_rate, out_rate, channels, filter, attenuation_db, cache)?;
    let design_seconds = t0.elapsed().as_secs_f64();

    let frames = (in_rate as f64 * audio_seconds.max(0.1)) as usize;
    let block = 4096usize;
    let tone: Vec<f64> = (0..block * channels)
        .map(|i| {
            let n = (i / channels) as f64;
            0.5 * (2.0 * std::f64::consts::PI * 997.0 * n / in_rate as f64).sin()
        })
        .collect();
    // Warm up (first block allocates the output buffers).
    let _ = r.push(&tone);
    r.reset();

    let t1 = Instant::now();
    let mut done = 0usize;
    let mut sink = 0.0f64;
    while done < frames {
        let out = r.push(&tone);
        sink += out.first().copied().unwrap_or(0.0);
        done += block;
    }
    let out = r.flush();
    sink += out.first().copied().unwrap_or(0.0);
    let wall_seconds = t1.elapsed().as_secs_f64().max(1e-9);
    std::hint::black_box(sink);

    let audio_seconds = done as f64 / in_rate as f64;
    Ok(BenchResult {
        filter: filter.id(),
        in_rate,
        out_rate,
        channels,
        taps: r.total_taps(),
        engine: r.describe(),
        threads: rayon::current_num_threads(),
        audio_seconds,
        wall_seconds,
        realtime_factor: audio_seconds / wall_seconds,
        design_seconds,
    })
}

/// Benchmark the partitioned convolver with a `taps`-long impulse at `rate`, `channels`
/// channels, `audio_seconds` of audio (channels run in parallel).
pub fn convolution(taps: usize, rate: u32, channels: usize, audio_seconds: f64) -> BenchResult {
    let t0 = Instant::now();
    let h: Vec<f64> = (0..taps.max(1))
        .map(|i| (-(i as f64) / taps.max(1) as f64 * 8.0).exp() * ((i as f64) * 0.01).cos())
        .collect();
    let mut convs: Vec<Convolver> = (0..channels.max(1)).map(|_| Convolver::new(&h)).collect();
    let design_seconds = t0.elapsed().as_secs_f64();
    let block = convs[0].block();
    let frames = (rate as f64 * audio_seconds.max(0.1)) as usize;
    let inputs: Vec<Vec<f64>> = (0..channels.max(1))
        .map(|c| {
            (0..block)
                .map(|i| 0.5 * ((i * (c + 1)) as f64 * 0.013).sin())
                .collect()
        })
        .collect();
    let mut outs: Vec<Vec<f64>> = vec![Vec::with_capacity(block); channels.max(1)];
    let t1 = Instant::now();
    let mut done = 0usize;
    while done < frames {
        for o in &mut outs {
            o.clear();
        }
        crate::convolution::process_channels(&mut convs, &inputs, &mut outs, false);
        done += block;
    }
    let wall_seconds = t1.elapsed().as_secs_f64().max(1e-9);
    std::hint::black_box(&outs);
    let audio_seconds = done as f64 / rate as f64;
    BenchResult {
        filter: "convolution".into(),
        in_rate: rate,
        out_rate: rate,
        channels: channels.max(1),
        taps: taps.max(1),
        engine: format!("partitioned FFT · block {block}"),
        threads: rayon::current_num_threads(),
        audio_seconds,
        wall_seconds,
        realtime_factor: audio_seconds / wall_seconds,
        design_seconds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fir::Phase;

    #[test]
    fn upsampling_bench_reports_a_positive_factor() {
        let r = run_with(
            FilterKind::sinc(Phase::Linear, false, 1 << 12),
            44_100,
            176_400,
            2,
            0.25,
            120.0,
            &CoefficientCache::disabled(),
        )
        .unwrap();
        assert_eq!((r.in_rate, r.out_rate, r.channels), (44_100, 176_400, 2));
        assert!(r.realtime_factor > 0.0 && r.realtime_factor.is_finite());
        assert!(r.audio_seconds >= 0.25);
        assert!(r.engine.contains("FFT polyphase"));
        assert_eq!(r.taps, (1 << 12) + 1);
        assert!(r.threads >= 1);
        serde_json::to_string(&r).unwrap();
        assert!(run_with(
            FilterKind::default(),
            48_000,
            48_000,
            2,
            0.1,
            120.0,
            &CoefficientCache::disabled()
        )
        .is_err());
    }

    #[test]
    fn convolution_bench_runs() {
        let r = convolution(1 << 14, 48_000, 2, 0.2);
        assert!(r.realtime_factor > 0.0);
        assert_eq!(r.taps, 1 << 14);
    }
}
