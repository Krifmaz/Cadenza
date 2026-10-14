//! `cadenza bench`: realtime factors (seconds of audio per wall-clock second) for the
//! machine this runs on – the long-FIR upsampler on the CPU and, when a card is present,
//! its convolution engine on the GPU; room convolution; and the DSD modulator at every
//! DSD rate (optionally with the long filter that feeds it), so users see where their
//! CPU/GPU tops out before they pick DSD1024 or `dsd_oversampling = full`.
//!
//! The DSP crate does the measuring (`cadenza_dsp::bench`, `cadenza_dsp::dsd::stage::
//! bench_modulator`, `cadenza_dsp::gpu::Backend::convolver`); this module decides what to
//! run and formats the table.

use cadenza_dsp::bench::{self as dsp_bench, BenchResult};
use cadenza_dsp::convolution::{choose_block, BlockConvolver, Convolver, SpectralFilter};
use cadenza_dsp::dsd::stage::{bench_modulator, DsdStage};
use cadenza_dsp::dsd::{DsdConfig, DsdModulator, DsdOversampling, DsdRate};
use cadenza_dsp::gpu::{Backend, GpuConfig};
use cadenza_dsp::resampler::RateFamily;
use cadenza_dsp::signal_path::rate_label;
use cadenza_dsp::{CoefficientCache, FilterKind};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// What to measure.
#[derive(Clone, Debug)]
pub struct BenchOptions {
    /// The upsampling filter.
    pub filter: FilterKind,
    /// Source rate in Hz.
    pub in_rate: u32,
    /// PCM output rate in Hz.
    pub out_rate: u32,
    /// Channels.
    pub channels: usize,
    /// Seconds of test audio per PCM measurement.
    pub seconds: f64,
    /// Stopband attenuation handed to the filter design.
    pub attenuation_db: f64,
    /// GPU selection (`gpu`, `gpu_device`).
    pub gpu: GpuConfig,
    /// Also time the modulator at every DSD rate.
    pub dsd: bool,
    /// Which modulator.
    pub modulator: DsdModulator,
    /// Also time the long filter feeding each DSD rate.
    pub dsd_upsampling: bool,
    /// With `dsd_upsampling`: the filter runs all the way to the DSD rate (`full`) instead
    /// of to f_dsd/8 (or /16).
    pub dsd_full: bool,
    /// Coefficient cache.
    pub cache: CoefficientCache,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            filter: FilterKind::default(),
            in_rate: 44_100,
            out_rate: 705_600,
            channels: 2,
            seconds: 2.0,
            attenuation_db: cadenza_dsp::fir::DEFAULT_ATTENUATION_DB,
            gpu: GpuConfig::default(),
            dsd: true,
            modulator: DsdModulator::Asdm7,
            dsd_upsampling: false,
            dsd_full: false,
            cache: CoefficientCache::default_location(),
        }
    }
}

/// One line of the table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchRow {
    /// What was measured (`upsampling`, `convolver`, `room-convolution`, `modulator`,
    /// `dsd-upsampling`).
    pub kind: String,
    /// Where it ran: `CPU (4 threads)` or `CUDA (RTX 4060)`.
    pub compute: String,
    /// Human description (filter id, DSD rate, taps…).
    pub detail: String,
    /// Input rate in Hz.
    pub in_rate: u32,
    /// Output rate in Hz.
    pub out_rate: u32,
    /// Channels measured (0 = per channel).
    pub channels: usize,
    /// Seconds of audio processed per wall-clock second; `None` when the measurement could
    /// not run (see `note`).
    pub realtime_factor: Option<f64>,
    /// Seconds spent designing or loading coefficients (0 when nothing was designed).
    pub design_seconds: f64,
    /// Why there is no number, or a remark.
    pub note: String,
}

impl BenchRow {
    /// `true` when the machine keeps up (20 % margin), `false` otherwise or when unknown.
    pub fn realtime(&self) -> bool {
        self.realtime_factor.is_some_and(|f| f >= 1.2)
    }
}

fn threads_label() -> String {
    format!(
        "CPU ({} threads)",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    )
}

fn from_result(kind: &str, compute: String, detail: String, r: &BenchResult) -> BenchRow {
    BenchRow {
        kind: kind.into(),
        compute,
        detail,
        in_rate: r.in_rate,
        out_rate: r.out_rate,
        channels: r.channels,
        realtime_factor: Some(r.realtime_factor),
        design_seconds: r.design_seconds,
        note: String::new(),
    }
}

/// Run a block convolver over `seconds` of audio per channel (channels one after the
/// other, as a worst case) and return the realtime factor.
fn time_convolver(
    make: impl Fn() -> Box<dyn BlockConvolver + Send>,
    block: usize,
    rate: u32,
    channels: usize,
    seconds: f64,
) -> f64 {
    let mut convs: Vec<Box<dyn BlockConvolver + Send>> =
        (0..channels.max(1)).map(|_| make()).collect();
    let input: Vec<f64> = (0..block)
        .map(|i| 0.5 * (i as f64 * 0.0137).sin())
        .collect();
    let mut out = vec![0.0f64; block];
    // Warm up: first call allocates / uploads.
    for c in &mut convs {
        c.process_block(&input, &mut out);
    }
    let frames = (rate as f64 * seconds.max(0.05)) as usize;
    let t = Instant::now();
    let mut done = 0usize;
    while done < frames {
        for c in &mut convs {
            c.process_block(&input, &mut out);
        }
        done += block;
    }
    let wall = t.elapsed().as_secs_f64().max(1e-9);
    std::hint::black_box(&out);
    done as f64 / rate as f64 / wall
}

/// The conversion ratio `l/m` in lowest terms.
fn ratio(in_rate: u32, out_rate: u32) -> (usize, usize) {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let g = gcd(in_rate, out_rate).max(1);
    ((out_rate / g) as usize, (in_rate / g) as usize)
}

/// Run every measurement in `o`.
pub fn run(o: &BenchOptions) -> Vec<BenchRow> {
    let mut rows = Vec::new();
    let cpu = threads_label();

    // 1. The upsampler as the pipeline runs it (CPU).
    match dsp_bench::run_with(
        o.filter,
        o.in_rate,
        o.out_rate,
        o.channels,
        o.seconds,
        o.attenuation_db,
        &o.cache,
    ) {
        Ok(r) => rows.push(from_result(
            "upsampling",
            cpu.clone(),
            format!("{} · {} taps · {}", r.filter, r.taps, r.engine),
            &r,
        )),
        Err(e) => rows.push(BenchRow {
            kind: "upsampling".into(),
            compute: cpu.clone(),
            detail: o.filter.id(),
            in_rate: o.in_rate,
            out_rate: o.out_rate,
            channels: o.channels,
            realtime_factor: None,
            design_seconds: 0.0,
            note: e.to_string(),
        }),
    }

    // 2. The filter's convolution engine alone, CPU and (when present) GPU, so the two
    //    numbers are directly comparable.
    let (l, m) = ratio(o.in_rate, o.out_rate);
    let spec = o.filter.spec_for_ratio(l, m, o.attenuation_db);
    let t0 = Instant::now();
    let h = o.cache.get_or_design(&spec);
    let design_seconds = t0.elapsed().as_secs_f64();
    let block = choose_block(h.len());
    let filter = Arc::new(SpectralFilter::new(&h, block));
    let detail = format!("{} · {} taps · block {block}", o.filter.id(), h.len());
    let f = Arc::clone(&filter);
    let factor = time_convolver(
        move || Box::new(Convolver::from_filter(Arc::clone(&f))) as Box<dyn BlockConvolver + Send>,
        block,
        o.out_rate,
        o.channels,
        o.seconds,
    );
    rows.push(BenchRow {
        kind: "convolver".into(),
        compute: cpu.clone(),
        detail: detail.clone(),
        in_rate: o.out_rate,
        out_rate: o.out_rate,
        channels: o.channels,
        realtime_factor: Some(factor),
        design_seconds,
        note: "the FFT engine behind the upsampler, at the output rate".into(),
    });
    let backend = Backend::detect(&o.gpu);
    if backend.is_gpu() {
        let b = backend.clone();
        let f = Arc::clone(&filter);
        let factor = time_convolver(
            move || b.convolver(Arc::clone(&f)),
            block,
            o.out_rate,
            o.channels,
            o.seconds,
        );
        rows.push(BenchRow {
            kind: "convolver".into(),
            compute: backend.label(),
            detail,
            in_rate: o.out_rate,
            out_rate: o.out_rate,
            channels: o.channels,
            realtime_factor: Some(factor),
            design_seconds: 0.0,
            note: "same engine on the GPU (one block per host round trip)".into(),
        });
    } else {
        rows.push(BenchRow {
            kind: "convolver".into(),
            compute: "GPU".into(),
            detail,
            in_rate: o.out_rate,
            out_rate: o.out_rate,
            channels: o.channels,
            realtime_factor: None,
            design_seconds: 0.0,
            note: match &backend {
                Backend::Cpu { reason } => format!("not run: {reason}"),
                Backend::Cuda { .. } => String::new(),
            },
        });
    }

    // 3. Room correction with a typical 65536-tap impulse at the output rate.
    let r = dsp_bench::convolution(65_536, o.out_rate, o.channels, o.seconds);
    rows.push(from_result(
        "room-convolution",
        cpu.clone(),
        format!("{} taps · {}", r.taps, r.engine),
        &r,
    ));

    // 4. The modulator at every DSD rate (one channel; the stage runs channels in parallel).
    if o.dsd {
        let family = RateFamily::of(o.in_rate)
            .map(RateFamily::base)
            .unwrap_or(44_100);
        let dsd_seconds = (o.seconds / 8.0).clamp(0.05, 0.5);
        for rate in DsdRate::catalog() {
            let hz = rate.hz(family);
            let samples = (hz as f64 * dsd_seconds) as usize;
            let factor = bench_modulator(o.modulator, hz, samples);
            rows.push(BenchRow {
                kind: "modulator".into(),
                compute: "CPU (1 thread per channel)".into(),
                detail: format!(
                    "{} · {} · {} MHz",
                    rate.label(),
                    o.modulator.label(),
                    rate_label(hz)
                ),
                in_rate: hz,
                out_rate: hz,
                channels: 1,
                realtime_factor: Some(factor),
                design_seconds: 0.0,
                note: "per channel; channels run on separate cores".into(),
            });
            if o.dsd_upsampling {
                let cfg = DsdConfig {
                    rate: *rate,
                    oversampling: if o.dsd_full {
                        DsdOversampling::Full
                    } else {
                        DsdOversampling::Staged
                    },
                    ..DsdConfig::default()
                };
                let pcm_rate = DsdStage::pcm_input_rate(&cfg, family);
                let mode = if o.dsd_full {
                    format!("{}× (full)", rate.multiple())
                } else {
                    format!(
                        "staged, ×{} short stage after",
                        cadenza_dsp::dsd::stage::short_stage_factor(*rate)
                    )
                };
                match dsp_bench::run_with(
                    o.filter,
                    o.in_rate,
                    pcm_rate,
                    o.channels,
                    (o.seconds / 4.0).max(0.1),
                    o.attenuation_db,
                    &o.cache,
                ) {
                    Ok(r) => rows.push(from_result(
                        "dsd-upsampling",
                        cpu.clone(),
                        format!("{} · {} · {} taps · {mode}", rate.label(), r.filter, r.taps),
                        &r,
                    )),
                    Err(e) => rows.push(BenchRow {
                        kind: "dsd-upsampling".into(),
                        compute: cpu.clone(),
                        detail: format!("{} · {} · {mode}", rate.label(), o.filter.id()),
                        in_rate: o.in_rate,
                        out_rate: pcm_rate,
                        channels: o.channels,
                        realtime_factor: None,
                        design_seconds: 0.0,
                        note: e.to_string(),
                    }),
                }
            }
        }
    }
    rows
}

/// `44.1k`, `705.6k`, `2.8224M`.
fn hz(rate: u32) -> String {
    if rate >= 1_000_000 {
        let m = format!("{:.4}", rate as f64 / 1e6);
        format!("{}M", m.trim_end_matches('0').trim_end_matches('.'))
    } else {
        format!("{}k", rate_label(rate))
    }
}

/// Format the rows as a text table.
pub fn table(rows: &[BenchRow]) -> String {
    let mut s = format!(
        "{:<17} {:<26} {:>9} {:>9} {:>3} {:>9}  {}\n",
        "WHAT", "COMPUTE", "IN", "OUT", "CH", "REALTIME", "DETAIL"
    );
    for r in rows {
        let factor = match r.realtime_factor {
            Some(f) if f >= 100.0 => format!("{f:.0}×"),
            Some(f) => format!("{f:.1}×"),
            None => "–".into(),
        };
        let mark = match r.realtime_factor {
            Some(f) if f >= 1.2 => "",
            Some(_) => " (too slow)",
            None => "",
        };
        let note = if r.note.is_empty() {
            String::new()
        } else {
            format!(" — {}", r.note)
        };
        s.push_str(&format!(
            "{:<17} {:<26} {:>9} {:>9} {:>3} {:>9}  {}{}{}\n",
            r.kind,
            r.compute.chars().take(26).collect::<String>(),
            hz(r.in_rate),
            hz(r.out_rate),
            r.channels,
            factor,
            r.detail,
            mark,
            note
        ));
    }
    s.push_str("\nREALTIME = seconds of audio processed per second; below 1.2× the machine will not keep up.\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_runs_every_row_kind_quickly() {
        let o = BenchOptions {
            filter: FilterKind::parse("sinc-l-4k").unwrap(),
            in_rate: 44_100,
            out_rate: 176_400,
            channels: 2,
            seconds: 0.1,
            cache: CoefficientCache::disabled(),
            gpu: GpuConfig::default(),
            dsd: true,
            dsd_upsampling: false,
            ..Default::default()
        };
        let rows = run(&o);
        let kinds: Vec<&str> = rows.iter().map(|r| r.kind.as_str()).collect();
        assert!(kinds.contains(&"upsampling"));
        assert!(kinds.contains(&"convolver"));
        assert!(kinds.contains(&"room-convolution"));
        assert_eq!(
            rows.iter().filter(|r| r.kind == "modulator").count(),
            DsdRate::catalog().len()
        );
        for r in &rows {
            if r.compute == "GPU" {
                // No card here: the row says why instead of a number.
                assert!(
                    r.realtime_factor.is_none() && r.note.starts_with("not run"),
                    "{r:?}"
                );
            } else {
                assert!(r.realtime_factor.unwrap() > 0.0, "{r:?}");
            }
        }
        let dsd2048 = rows
            .iter()
            .find(|r| r.detail.starts_with("DSD2048"))
            .unwrap();
        assert_eq!(dsd2048.in_rate, 90_316_800);
        let text = table(&rows);
        assert!(
            text.contains("REALTIME") && text.contains("DSD64"),
            "{text}"
        );
        let json = serde_json::to_string(&rows).unwrap();
        assert!(json.contains("\"kind\":\"modulator\""));
    }

    #[test]
    fn dsd_upsampling_rows_use_the_staged_pcm_rate() {
        let o = BenchOptions {
            filter: FilterKind::parse("sinc-l-4k").unwrap(),
            in_rate: 48_000,
            out_rate: 192_000,
            seconds: 0.1,
            cache: CoefficientCache::disabled(),
            dsd: true,
            dsd_upsampling: true,
            ..Default::default()
        };
        let rows = run(&o);
        let up: Vec<&BenchRow> = rows.iter().filter(|r| r.kind == "dsd-upsampling").collect();
        assert_eq!(up.len(), DsdRate::catalog().len());
        // DSD64 staged: 3.072 MHz / 8 = 384 kHz; DSD512: 24.576 MHz / 16 = 1.536 MHz.
        assert_eq!(up[0].out_rate, 384_000);
        assert_eq!(up[3].out_rate, 1_536_000);
        assert!(ratio(44_100, 48_000) == (160, 147));
        assert_eq!(hz(2_822_400), "2.8224M");
        assert_eq!(hz(90_316_800), "90.3168M");
        assert_eq!(hz(705_600), "705.6k");
    }
}
