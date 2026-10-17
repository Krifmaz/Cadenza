//! The limit test: how heavy a filter this machine can run, for the device and settings
//! in use, before playback would stutter.
//!
//! It builds the same pipeline playback would (same device format, DSD rate, EQ, room
//! correction, GPU choice) with longer and longer filters, pushes test audio through each
//! and times it. "Speed" is seconds of audio per second of work: below 1 the machine falls
//! behind and the sound breaks up. A filter counts as safe from 1.5× up, so other programs,
//! a busy disk or a hot laptop do not tip it over; 1.1–1.5× is tight. The ladder stops at
//! the first filter that is too slow, since every longer one is slower still.

use crate::config::Settings;
use crate::output::{self, DeviceInfo, Output, OutputFormat, Probe};
use cadenza_dsp::dsd::DsdConfig;
use cadenza_dsp::fir::Family;
use cadenza_dsp::pipeline::InputFormat;
use cadenza_dsp::{CoefficientCache, FilterKind, Pipeline, Volume};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Safe from here up (seconds of audio per second).
pub const SAFE: f64 = 1.5;
/// Below this the machine cannot keep up.
pub const TOO_SLOW: f64 = 1.1;

/// Source rate the test plays (CD; most music).
const SOURCE_RATE: u32 = 44_100;
/// Frames per chunk, about what a decoder hands over.
const CHUNK: usize = 4096;
/// Audio pushed through before timing starts (fills the FFT stages).
const WARMUP_SECONDS: f64 = 1.0;
/// A single chunk taking longer than this could empty the device's buffer even when the
/// average keeps up (long FFT filters work in bursts).
const WORST_CALL: f64 = 0.25;
/// Each filter is timed for this much audio, or this much wall time, whichever ends first.
const MEASURE_SECONDS: f64 = 10.0;
const MEASURE_WALL: Duration = Duration::from_secs(6);

/// One filter tried.
#[derive(Clone, Debug, Serialize)]
pub struct Row {
    /// Filter id (what goes in the settings).
    pub filter: String,
    /// Filter length in taps.
    pub taps: usize,
    /// Seconds of audio per second of work (0 when it did not run).
    pub speed: f64,
    /// The longest a single chunk took, in seconds (long filters work in bursts).
    pub worst_seconds: f64,
    /// Seconds spent designing the filter (once; then it is cached).
    pub design_seconds: f64,
    /// `safe`, `tight`, `too_slow`, `error`.
    pub verdict: String,
    /// What happened, in words.
    pub note: String,
    /// The filter in the settings right now.
    pub current: bool,
}

/// What the test found.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Report {
    /// The device and the format it would be opened in.
    pub output: String,
    /// What the conversion is (`44.1 kHz → 705.6 kHz 24-bit`, …).
    pub conversion: String,
    /// Where the heavy work runs (CPU / GPU).
    pub compute: String,
    /// Filters tried, shortest first.
    pub rows: Vec<Row>,
    /// The longest safe filter, if any.
    pub recommended: Option<String>,
    /// A sentence for the user.
    pub summary: String,
}

/// Where a running test is.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Progress {
    /// `idle`, `running`, `done`, `cancelled`, `error`.
    pub state: String,
    /// Filters done / to do.
    pub done: usize,
    /// Filters on the ladder.
    pub total: usize,
    /// What is being timed now.
    pub current: String,
    /// The result so far (complete when `state` is `done`).
    pub report: Report,
    /// Why it failed (`state` = `error`).
    pub error: Option<String>,
}

/// The filters to try, shortest first: the family, phase and window of `current` at every
/// length the catalogue has (the long-sinc family above 4M taps), so the answer is "your
/// kind of filter, as long as this".
pub fn ladder(current: &str) -> Vec<FilterKind> {
    let cur = FilterKind::parse(current).unwrap_or_default();
    let mut lengths: Vec<usize> = Family::ALL
        .iter()
        .flat_map(|f| f.catalog_taps().iter().copied())
        .filter(|t| *t >= 1 << 12)
        .collect();
    lengths.sort_unstable();
    lengths.dedup();
    // Every catalogue length that the family does not have becomes the closest family
    // that does: sinc up to 4M, sinc-xtr beyond.
    let mut out: Vec<FilterKind> = Vec::new();
    for taps in lengths {
        let family = if cur.family.catalog_taps().contains(&taps) {
            cur.family
        } else if Family::Sinc.catalog_taps().contains(&taps) {
            Family::Sinc
        } else if Family::SincXtr.catalog_taps().contains(&taps) {
            Family::SincXtr
        } else {
            continue;
        };
        let k = FilterKind {
            family,
            taps,
            ..cur
        };
        if !out.iter().any(|o| o.taps == k.taps) {
            out.push(k);
        }
    }
    out
}

/// The device the settings name, as listed (or `None` when it is not connected).
pub fn find_device(settings: &Settings, devices: &[DeviceInfo]) -> Option<DeviceInfo> {
    let (backend, device) =
        output::split_backend(&settings.output.backend, &settings.output.device);
    devices
        .iter()
        .find(|d| d.backend == backend && (d.id == device || (device.is_empty() && d.default)))
        .or_else(|| {
            devices
                .iter()
                .find(|d| d.backend == backend && device.is_empty())
        })
        .cloned()
}

/// The format playback would open `device` in for a CD source, and the DSD choice.
fn target(settings: &Settings, device: &DeviceInfo) -> (OutputFormat, Option<DsdConfig>) {
    let s = settings;
    let probe = Probe(device.clone());
    let channels = if s.matrix.enabled {
        s.matrix
            .resolve(2)
            .map(|m| m.outputs.len() as u32)
            .unwrap_or(2)
    } else {
        2
    };
    if s.dsp.dsd_output() {
        let family = cadenza_dsp::dsd::family_base(SOURCE_RATE);
        if let Some((rate, packing)) =
            output::choose_dsd(&probe, family, s.dsp.dsd_rate, s.dsp.dsd_packing, channels)
        {
            let d = DsdConfig {
                rate,
                packing,
                ..s.dsp.dsd()
            };
            return (
                OutputFormat::dsd(rate.hz(family), packing, channels),
                Some(d),
            );
        }
    }
    let bits = [s.output.bits, 32, 24, 16]
        .into_iter()
        .find(|b| probe.supports(SOURCE_RATE, *b))
        .unwrap_or(s.output.bits);
    let rate = match s.output.rate {
        crate::config::RateSetting::Source => SOURCE_RATE,
        crate::config::RateSetting::Auto => {
            output::choose_rate(&probe, SOURCE_RATE, None, s.output.max_rate, bits)
        }
        crate::config::RateSetting::Fixed(r) => {
            output::choose_rate(&probe, SOURCE_RATE, Some(r), s.output.max_rate, bits)
        }
    };
    (OutputFormat::pcm(rate, bits, channels), None)
}

fn khz(rate: u32) -> String {
    let k = rate as f64 / 1000.0;
    if k.fract() == 0.0 {
        format!("{k:.0} kHz")
    } else {
        format!("{k} kHz")
    }
}

/// Time `filter` in the playback chain. Returns (speed, design seconds).
fn time_filter(
    settings: &Settings,
    filter: &FilterKind,
    got: OutputFormat,
    dsd: Option<DsdConfig>,
    device: &str,
    cache: &CoefficientCache,
    cancel: &AtomicBool,
) -> Result<(f64, f64, f64), String> {
    let mut s = settings.clone();
    s.dsp.filter = filter.id();
    let convolution = match s.dsp.convolution.trim() {
        "" => None,
        path => cadenza_dsp::convolution::load_impulse_response(std::path::Path::new(path))
            .ok()
            .map(|(rate, channels)| {
                std::sync::Arc::new(cadenza_dsp::ConvolutionSpec {
                    name: path.to_string(),
                    rate,
                    channels,
                })
            }),
    };
    let volume = Volume {
        gain_db: s.dsp.volume_db,
        muted: false,
    };
    let config = crate::engine::pipeline_config(
        &s,
        volume,
        convolution,
        got,
        SOURCE_RATE,
        dsd,
        device.to_string(),
    );
    let input = InputFormat {
        rate: SOURCE_RATE,
        bits: 16,
        channels: 2,
        codec: "test".into(),
    };
    let t0 = Instant::now();
    let mut p = Pipeline::new(input, config, cache).map_err(|e| e.to_string())?;
    let design = t0.elapsed().as_secs_f64();

    // Noise at -12 dBFS: the work does not depend on what the music is.
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut chunk = vec![0.0f64; CHUNK * 2];
    let mut fill = |c: &mut [f64]| {
        for v in c.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *v = ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.5;
        }
    };
    let chunk_seconds = CHUNK as f64 / SOURCE_RATE as f64;
    let mut warm = 0.0;
    let warm_start = Instant::now();
    while warm < WARMUP_SECONDS {
        fill(&mut chunk);
        std::hint::black_box(p.process(&chunk));
        warm += chunk_seconds;
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        // Hopeless already (a second of audio took more than a few seconds): stop.
        if warm_start.elapsed() > MEASURE_WALL {
            let wall = warm_start.elapsed().as_secs_f64();
            return Ok((warm / wall, wall, design));
        }
    }
    let start = Instant::now();
    let mut audio = 0.0;
    let mut worst = 0.0f64;
    while audio < MEASURE_SECONDS && (audio < 2.0 * chunk_seconds || start.elapsed() < MEASURE_WALL)
    {
        fill(&mut chunk);
        let t = Instant::now();
        std::hint::black_box(p.process(&chunk));
        worst = worst.max(t.elapsed().as_secs_f64());
        audio += chunk_seconds;
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
    }
    Ok((audio / start.elapsed().as_secs_f64(), worst, design))
}

/// Run the test for `settings` on `device` (`None`: not connected, so an ideal device is
/// assumed). `progress` sees every step; `cancel` stops it between chunks.
pub fn run(
    settings: &Settings,
    device: Option<DeviceInfo>,
    cache: &CoefficientCache,
    cancel: &AtomicBool,
    mut progress: impl FnMut(&Progress),
) -> Result<Report, String> {
    let mut report = Report::default();
    if settings.dsp.bit_perfect {
        report.summary = "Bit-perfect is on: the music goes to the device untouched, so no \
            filter runs and there is nothing to test. Turn bit-perfect off to upsample."
            .into();
        return Ok(report);
    }
    let connected = device.is_some();
    let device = device.unwrap_or_else(output::null::device_info);
    let (got, dsd) = target(settings, &device);
    let device_label = format!("{} · {}", device.backend, device.name);
    report.output = format!(
        "{device_label}{}",
        if connected {
            ""
        } else {
            " (not connected: assuming it takes every rate)"
        }
    );
    report.conversion = match &dsd {
        Some(d) => format!(
            "{} → {} ({})",
            khz(SOURCE_RATE),
            d.rate.label(),
            d.modulator.label()
        ),
        None => format!("{} → {} {}-bit", khz(SOURCE_RATE), khz(got.rate), got.bits),
    };
    report.compute = cadenza_dsp::gpu::Backend::detect(&settings.dsp.gpu).label();

    if dsd.is_none() && got.rate == SOURCE_RATE {
        report.summary = format!(
            "Your output runs at the music's own rate ({}), so no upsampling filter runs and \
             the filter choice makes no difference. Your PC has plenty of room. To use the \
             filters, pick a higher PCM rate (if your device takes one).",
            khz(got.rate)
        );
        return Ok(report);
    }

    let ladder = ladder(&settings.dsp.filter);
    let current = FilterKind::parse(&settings.dsp.filter).map(|k| k.id());
    let mut p = Progress {
        state: "running".into(),
        total: ladder.len(),
        ..Default::default()
    };
    let mut stopped = false;
    for (i, k) in ladder.iter().enumerate() {
        let id = k.id();
        let mut row = Row {
            filter: id.clone(),
            taps: k.taps,
            speed: 0.0,
            worst_seconds: 0.0,
            design_seconds: 0.0,
            verdict: String::new(),
            note: String::new(),
            current: current.as_deref() == Some(id.as_str()),
        };
        if stopped {
            row.verdict = "too_slow".into();
            row.note = "not tried: a shorter filter was already too slow".into();
            report.rows.push(row);
            continue;
        }
        p.done = i;
        p.current = format!("{} ({} taps)", id, k.taps);
        p.report = report.clone();
        progress(&p);
        match time_filter(settings, k, got, dsd, &device_label, cache, cancel) {
            Ok((speed, worst, design)) => {
                row.speed = speed;
                row.worst_seconds = worst;
                row.design_seconds = design;
                row.verdict = if speed >= SAFE && worst <= WORST_CALL {
                    "safe"
                } else if speed >= TOO_SLOW {
                    "tight"
                } else {
                    "too_slow"
                }
                .into();
                row.note = match row.verdict.as_str() {
                    "safe" => format!("runs {speed:.1}× faster than playback"),
                    "tight" if speed >= SAFE => format!(
                        "fast enough on average ({speed:.1}×), but works in bursts of up to {:.0} ms: \
                         may click on a device with a small buffer",
                        worst * 1000.0
                    ),
                    "tight" => format!(
                        "only {speed:.2}× faster than playback: may stutter when the PC is busy"
                    ),
                    _ => format!("{speed:.2}× playback speed: would stutter"),
                };
                if design > 2.0 {
                    row.note += &format!(
                        "; first play waits {design:.0} s while it is designed (then cached)"
                    );
                }
                stopped = row.verdict == "too_slow";
            }
            Err(e) if e == "cancelled" => return Err(e),
            Err(e) => {
                row.verdict = "error".into();
                row.note = e;
            }
        }
        report.rows.push(row);
    }
    report.recommended = report
        .rows
        .iter()
        .rev()
        .find(|r| r.verdict == "safe")
        .map(|r| r.filter.clone());
    let cur_row = report.rows.iter().find(|r| r.current);
    report.summary = match (&report.recommended, cur_row) {
        (None, _) => "Even the shortest filter is not safe at this rate. Lower the PCM rate \
            or DSD rate, turn on the GPU if you have one, or close other programs."
            .into(),
        (Some(best), Some(c)) if c.verdict == "too_slow" || c.verdict == "tight" => format!(
            "Your current filter ({}) is {} here. The longest safe one is {best}.",
            c.filter,
            if c.verdict == "tight" {
                "cutting it close"
            } else {
                "too heavy"
            }
        ),
        (Some(best), _) => format!(
            "The longest filter that runs safely here is {best}. Longer filters cut more \
             sharply and ring less before the beat, but most people hear little past 1M taps."
        ),
    };
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ladder_keeps_the_kind_of_filter_and_grows() {
        let l = ladder("sinc-ma-4k");
        assert!(l.windows(2).all(|w| w[0].taps < w[1].taps));
        assert!(l
            .iter()
            .all(|k| k.phase == FilterKind::parse("sinc-ma-4k").unwrap().phase));
        assert_eq!(l.last().unwrap().family, Family::SincXtr);
        let cf = ladder(
            &FilterKind {
                family: Family::ClosedForm,
                ..FilterKind::parse("sinc-l-1m").unwrap()
            }
            .id(),
        );
        assert!(cf.iter().any(|k| k.family == Family::ClosedForm));
    }

    #[test]
    fn same_rate_output_needs_no_test() {
        let mut s = Settings::default();
        s.output.rate = crate::config::RateSetting::Source;
        let mut d = output::null::device_info();
        d.rates = vec![44_100];
        let r = run(
            &s,
            Some(d),
            &CoefficientCache::disabled(),
            &AtomicBool::new(false),
            |_| {},
        )
        .unwrap();
        assert!(r.rows.is_empty());
        assert!(r.summary.contains("no upsampling"), "{}", r.summary);
    }

    #[test]
    fn short_filters_are_safe_and_the_ladder_stops_at_the_first_slow_one() {
        let mut s = Settings::default();
        s.dsp.filter = "sinc-l-4k".into();
        s.output.rate = crate::config::RateSetting::Fixed(176_400);
        let mut seen = 0;
        let cancel = AtomicBool::new(false);
        // Only the first two lengths, to keep the test short.
        let r = run(&s, None, &CoefficientCache::disabled(), &cancel, |p| {
            seen += 1;
            if p.done >= 2 {
                cancel.store(true, Ordering::Relaxed);
            }
        });
        assert!(seen >= 2);
        assert_eq!(r.unwrap_err(), "cancelled");
        let short = &ladder("sinc-l-4k")[0];
        let (speed, _, _) = time_filter(
            &s,
            short,
            OutputFormat::pcm(176_400, 24, 2),
            None,
            "null",
            &CoefficientCache::disabled(),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(speed > SAFE, "4k taps to 176.4 kHz at {speed}×");
    }
}
