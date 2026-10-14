//! `cadenza render`: the binary as Encore drives it (PCM on stdin → s32le on stdout).

use cadenza_dsp::{
    CoefficientCache, DitherKind, FilterKind, InputFormat, Pipeline, PipelineConfig,
};
use std::f64::consts::PI;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_cadenza")
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("cadenza-render-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run `cadenza render <args>` with `input` on stdin.
fn render(args: &[&str], input: Vec<u8>) -> Output {
    let mut child = Command::new(bin())
        .arg("render")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
    });
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    out
}

/// Interleaved stereo 1 kHz sine at `amp`.
fn tone(rate: u32, seconds: f64, amp: f64) -> Vec<f64> {
    let frames = (rate as f64 * seconds) as usize;
    let mut out = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let v = amp * (2.0 * PI * 1000.0 * i as f64 / rate as f64).sin();
        out.push(v);
        out.push(v);
    }
    out
}

fn f64le(samples: &[f64]) -> Vec<u8> {
    samples.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn s32le(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn describe_prints_one_json_line() {
    let out = Command::new(bin())
        .args(["render", "--describe"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(
        out.stderr.is_empty(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    assert_eq!(text.trim_end().lines().count(), 1);
    let d: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
    assert_eq!(d["name"], "cadenza");
    assert_eq!(d["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(d["protocol"], 1);
    assert_eq!(d["in_formats"], serde_json::json!(["f32le", "f64le"]));
    assert_eq!(d["out_formats"], serde_json::json!(["s32le"]));
    assert_eq!(d["gpu"], cadenza_dsp::gpu::COMPILED_IN);
    let filters: Vec<&str> = d["filters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f.as_str().unwrap())
        .collect();
    assert!(filters.contains(&"sinc-l-64k"));
    assert_eq!(filters.len(), FilterKind::catalog().len());
    let dithers = d["dithers"].as_array().unwrap();
    assert!(dithers.iter().any(|x| x == "tpdf"));
    let keys: Vec<&str> = d["settings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["key"].as_str().unwrap())
        .collect();
    for k in ["filter", "dither", "headroom_db", "soft_clip"] {
        assert!(keys.contains(&k), "{k} missing from {keys:?}");
    }
    assert_eq!(keys.contains(&"gpu"), cadenza_dsp::gpu::COMPILED_IN);
}

#[test]
fn tone_matches_the_in_process_pipeline_sample_for_sample() {
    let cache_dir = temp_dir("tone");
    let (in_rate, out_rate) = (44_100u32, 96_000u32);
    let seconds = 1.5;
    let amp = 0.5;
    let input = tone(in_rate, seconds, amp);
    let out = render(
        &[
            "--in-rate",
            "44100",
            "--channels",
            "2",
            "--in-format",
            "f64le",
            "--rate",
            "96000",
            "--filter",
            "sinc-l-4k",
            "--dither",
            "tpdf",
            "--bits",
            "24",
            "--out-format",
            "s32le",
            "--headroom-db",
            "0",
            "--gpu",
            "off",
            "--cache-dir",
            cache_dir.to_str().unwrap(),
        ],
        f64le(&input),
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stderr.is_empty(),
        "nothing on stderr without --progress"
    );
    assert_eq!(out.stdout.len() % 8, 0, "whole stereo s32 frames");
    let got = s32le(&out.stdout);

    // length: the whole input at the new rate (the resampler tail is flushed)
    let frames = got.len() / 2;
    let expected = (input.len() / 2) as f64 * out_rate as f64 / in_rate as f64;
    assert!(
        (frames as f64 - expected).abs() <= 2.0,
        "{frames} frames, expected ≈{expected}"
    );
    // peak: −6 dBFS of full-scale s32 (24-bit values left-justified)
    let peak = got.iter().map(|v| (*v as f64).abs()).fold(0.0, f64::max) / 2f64.powi(31);
    assert!((peak - amp).abs() < 0.01, "peak {peak}");
    assert!(got.iter().all(|v| v & 0xff == 0), "24-bit values << 8");

    // exactly what Encore's in-process path produces for the same settings
    let cfg = PipelineConfig {
        filter: FilterKind::parse("sinc-l-4k").unwrap(),
        output_rate: Some(out_rate),
        output_bits: 24,
        dither: DitherKind::Tpdf,
        headroom_db: Some(0.0),
        gpu: cadenza_dsp::GpuConfig {
            mode: cadenza_dsp::GpuMode::Off,
            device: 0,
        },
        ..PipelineConfig::default()
    };
    let fmt = InputFormat {
        rate: in_rate,
        bits: 16,
        channels: 2,
        codec: "FLAC".into(),
    };
    let mut pipe = Pipeline::new(fmt, cfg, &CoefficientCache::in_dir(&cache_dir)).unwrap();
    // fed in Encore's 64 KiB blocks
    let mut reference: Vec<i32> = Vec::new();
    for block in input.chunks(64 * 1024 / 8) {
        reference.extend(pipe.process(block).into_iter().map(|v| v << 8));
    }
    reference.extend(pipe.flush().into_iter().map(|v| v << 8));
    assert_eq!(got.len(), reference.len());
    assert!(
        got == reference,
        "render differs from the in-process pipeline"
    );

    // f32le input of the same tone also renders at the right length
    let f32_bytes: Vec<u8> = input
        .iter()
        .flat_map(|v| (*v as f32).to_le_bytes())
        .collect();
    let out = render(
        &[
            "--in-rate",
            "44100",
            "--in-format",
            "f32le",
            "--rate",
            "96000",
            "--set",
            "filter=sinc-l-4k",
            "--set",
            "gpu=off",
            "--cache-dir",
            cache_dir.to_str().unwrap(),
        ],
        f32_bytes,
    );
    assert!(out.status.success());
    assert_eq!(out.stdout.len(), reference.len() * 4);
    let _ = std::fs::remove_dir_all(&cache_dir);
}

#[test]
fn bad_values_exit_2_and_unknown_keys_only_warn() {
    let cache_dir = temp_dir("errors");
    let cache = cache_dir.to_str().unwrap();
    let silence = vec![0u8; 44_100 * 8 * 2 / 10];
    for args in [
        vec!["--in-rate", "44100", "--set", "dither=loud"],
        vec!["--in-rate", "44100", "--filter", "nope"],
        vec!["--in-rate", "44100", "--set", "headroom_db=abc"],
        vec!["--in-rate", "44100", "--in-format", "s16le"],
        vec!["--in-rate", "44100", "--out-format", "f32le"],
        vec!["--rate", "96000"],
    ] {
        let mut a = args.clone();
        a.extend(["--cache-dir", cache]);
        let out = render(&a, silence.clone());
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(!out.stderr.is_empty(), "{args:?}: a message on stderr");
        assert!(out.stdout.is_empty());
    }
    let out = render(
        &[
            "--in-rate",
            "44100",
            "--rate",
            "88200",
            "--set",
            "filter=sinc-l-256",
            "--set",
            "colour=blue",
            "--set",
            "soft_clip=false",
            "--progress",
            "--cache-dir",
            cache,
        ],
        silence.clone(),
    );
    assert!(out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("colour"), "warned: {err}");
    assert!(err.contains("{\"in_seconds\":"), "progress line: {err}");
    assert_eq!(
        out.stdout.len(),
        silence.len() * 2 / 2,
        "×2 rate, 8 → 4 bytes"
    );
    let _ = std::fs::remove_dir_all(&cache_dir);
}
