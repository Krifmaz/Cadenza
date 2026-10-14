//! `cadenza render`: a pure PCM filter for Encore's server-side processing.
//!
//! Interleaved PCM comes in on stdin (`f32le` or `f64le` at `--in-rate`), goes through the
//! same [`Pipeline`] Encore runs in-process (filter, output rate, quantiser with dither,
//! headroom, soft clipper) and leaves on stdout as `s32le`: the quantised value shifted up
//! to full scale (24-bit output = the value << 8), byte for byte what Encore pipes to its
//! FLAC encoder. No decoding, no encoding, no device.
//!
//! The engine describes itself (`--describe`): protocol version, formats and a schema of
//! every setting it takes, so Encore can draw the settings without knowing them in advance.
//! Settings are passed as `--set key=value`; unknown keys are ignored with a warning (an
//! older engine with a newer Encore), bad values are an error.

use cadenza_dsp::gpu::{Backend as GpuBackend, GpuConfig, GpuMode, GpuPrecision};
use cadenza_dsp::{
    CoefficientCache, DitherKind, FilterKind, InputFormat, Pipeline, PipelineConfig,
};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

/// Version of the stdin/stdout contract with Encore. Bumped only when an older Encore
/// could no longer drive this engine.
pub const PROTOCOL: u32 = 1;

/// Buffer size for stdin and stdout (and the read block).
pub const IO_BUFFER: usize = 256 * 1024;

/// Input sample format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InFormat {
    /// 32-bit float, little endian.
    F32le,
    /// 64-bit float, little endian.
    F64le,
}

impl InFormat {
    /// Every accepted id.
    pub const IDS: [&'static str; 2] = ["f32le", "f64le"];

    /// Parse `f32le` / `f64le`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32le" => Some(InFormat::F32le),
            "f64le" => Some(InFormat::F64le),
            _ => None,
        }
    }

    /// Bytes per sample.
    pub fn bytes(self) -> usize {
        match self {
            InFormat::F32le => 4,
            InFormat::F64le => 8,
        }
    }
}

/// Output sample formats.
pub const OUT_FORMATS: [&str; 1] = ["s32le"];

/// The engine settings render takes (everything besides the PCM format flags); each one is
/// a key of the `--describe` schema and of `--set`.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineSettings {
    /// Upsampling filter.
    pub filter: FilterKind,
    /// Dither / noise shaper.
    pub dither: DitherKind,
    /// Level after the resampler in dB (0 = the original level, as Encore renders).
    pub headroom_db: f64,
    /// Transparent soft clipper before the quantiser.
    pub soft_clip: bool,
    /// GPU mode and device.
    pub gpu: GpuConfig,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            filter: FilterKind::default(),
            dither: DitherKind::Tpdf,
            headroom_db: 0.0,
            soft_clip: true,
            gpu: GpuConfig::default(),
        }
    }
}

/// Lowest and highest `headroom_db`.
pub const HEADROOM_RANGE: (f64, f64) = (-12.0, 0.0);

/// The schema keys, in display order.
pub const KEYS: [&str; 7] = [
    "filter",
    "dither",
    "headroom_db",
    "soft_clip",
    "gpu",
    "gpu_device",
    "gpu_precision",
];

fn parse_bool(v: &str) -> Option<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// `auto` / `on` / `off` (the schema's ids) plus the config spellings (`cuda`, `wgpu`,
/// `cpu`…).
pub fn parse_gpu(v: &str) -> Result<GpuMode, String> {
    v.parse::<GpuMode>()
        .map_err(|_| format!("bad value {v:?} for gpu (auto, on or off)"))
}

impl EngineSettings {
    /// Set `key` from its text value. `Ok(false)`: the key is unknown (the caller warns and
    /// goes on); `Err`: the value is bad.
    pub fn set(&mut self, key: &str, value: &str) -> Result<bool, String> {
        let v = value.trim();
        match key.trim() {
            "filter" => {
                self.filter = FilterKind::parse(v).ok_or_else(|| {
                    format!("unknown filter {v:?} (see `cadenza filters` or --describe)")
                })?
            }
            "dither" => {
                self.dither = DitherKind::parse(v).ok_or_else(|| {
                    format!("unknown dither {v:?} (see `cadenza dithers` or --describe)")
                })?
            }
            "headroom_db" => {
                let db: f64 = v
                    .parse()
                    .ok()
                    .filter(|d: &f64| d.is_finite())
                    .ok_or_else(|| format!("bad value {v:?} for headroom_db (a number in dB)"))?;
                if db < HEADROOM_RANGE.0 || db > HEADROOM_RANGE.1 {
                    return Err(format!(
                        "headroom_db {db} is out of range ({} to {} dB)",
                        HEADROOM_RANGE.0, HEADROOM_RANGE.1
                    ));
                }
                self.headroom_db = db;
            }
            "soft_clip" => {
                self.soft_clip = parse_bool(v)
                    .ok_or_else(|| format!("bad value {v:?} for soft_clip (true or false)"))?
            }
            "gpu" => self.gpu.mode = parse_gpu(v)?,
            "gpu_device" => {
                self.gpu.device = v
                    .parse()
                    .map_err(|_| format!("bad value {v:?} for gpu_device (a GPU index)"))?
            }
            "gpu_precision" => {
                self.gpu.precision = v.parse::<GpuPrecision>().map_err(|_| {
                    format!("bad value {v:?} for gpu_precision (auto, f64 or double-f32)")
                })?
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Apply `key=value` pairs (`--set`); unknown keys are reported through `warn`.
    pub fn apply_sets(
        &mut self,
        sets: &[String],
        mut warn: impl FnMut(&str),
    ) -> Result<(), String> {
        for s in sets {
            let (k, v) = s
                .split_once('=')
                .ok_or_else(|| format!("--set {s:?}: expected key=value"))?;
            if !self.set(k, v)? {
                warn(&format!(
                    "warning: ignoring unknown setting {:?} (this engine knows: {})",
                    k.trim(),
                    KEYS.join(", ")
                ));
            }
        }
        Ok(())
    }

    /// The pipeline configuration for an output at `rate` Hz and `bits`.
    pub fn pipeline_config(&self, rate: u32, bits: u32) -> PipelineConfig {
        PipelineConfig {
            filter: self.filter,
            output_rate: Some(rate),
            output_bits: bits,
            dither: self.dither,
            headroom_db: Some(self.headroom_db),
            soft_clip: self.soft_clip,
            gpu: self.gpu,
            ..PipelineConfig::default()
        }
    }
}

/// CUDA devices on this machine, `(index, name)` (empty without the `cuda` feature, a
/// driver or a card).
pub fn cuda_devices() -> Vec<(usize, String)> {
    let mut out = Vec::new();
    if !cadenza_dsp::gpu::CUDA_COMPILED_IN {
        return out;
    }
    for device in 0..16 {
        match GpuBackend::probe(&GpuConfig {
            mode: GpuMode::Cuda,
            device,
            ..GpuConfig::default()
        }) {
            Ok(GpuBackend::Cuda { name, .. }) => out.push((device, name)),
            _ => break,
        }
    }
    out
}

/// The GPUs `gpu_device` indexes, `(index, name)`: the CUDA devices when there are any,
/// otherwise the Vulkan / Metal / DX12 adapters (`AMD Radeon RX 7800 XT (Vulkan)`).
pub fn gpu_devices() -> Vec<(usize, String)> {
    let cuda = cuda_devices();
    if !cuda.is_empty() {
        return cuda;
    }
    #[cfg(feature = "wgpu")]
    {
        cadenza_dsp::gpu::wgpu_backend::adapters()
            .into_iter()
            .map(|a| (a.index, format!("{} ({})", a.name, a.api)))
            .collect()
    }
    #[cfg(not(feature = "wgpu"))]
    {
        Vec::new()
    }
}

/// The GPU APIs compiled into this build: `["cuda", "wgpu"]`.
fn gpu_apis() -> Vec<&'static str> {
    let mut v = Vec::new();
    if cadenza_dsp::gpu::CUDA_COMPILED_IN {
        v.push("cuda");
    }
    if cadenza_dsp::gpu::WGPU_COMPILED_IN {
        v.push("wgpu");
    }
    v
}

/// Help text of the `gpu` setting, naming the GPU APIs this build has.
fn gpu_help() -> String {
    let apis = match (
        cadenza_dsp::gpu::CUDA_COMPILED_IN,
        cadenza_dsp::gpu::WGPU_COMPILED_IN,
    ) {
        (true, true) => "an NVIDIA GPU (CUDA) or any Vulkan / Metal / DX12 GPU (AMD, Intel, Apple)",
        (true, false) => "an NVIDIA GPU (CUDA)",
        _ => "a Vulkan / Metal / DX12 GPU (AMD, Intel, Apple, NVIDIA)",
    };
    format!(
        "Run the long filters on {apis}. Auto uses one when present; the CPU is used otherwise. CUDA and 64-bit-capable cards compute the same 64-bit math as the CPU."
    )
}

/// The settings schema for `--describe`: every key `--set` takes, with its type, label,
/// options, default and help. `gpu` is listed only in GPU builds (`cuda` / `wgpu`),
/// `gpu_device` only when a GPU was found, `gpu_precision` only in `wgpu` builds.
pub fn settings_schema(devices: &[(usize, String)]) -> Vec<Value> {
    let d = EngineSettings::default();
    let filters: Vec<Value> = FilterKind::catalog()
        .into_iter()
        .map(|f| json!([f.id(), f.label()]))
        .collect();
    let dithers: Vec<Value> = DitherKind::catalog()
        .iter()
        .map(|k| json!([k.id(), k.label()]))
        .collect();
    let mut out = vec![
        json!({
            "key": "filter", "label": "Upsampling filter", "type": "select",
            "options": filters, "default": d.filter.id(),
            "help": "The interpolation filter. Longer filters are sharper and use more CPU (or GPU).",
        }),
        json!({
            "key": "dither", "label": "Dither / noise shaping", "type": "select",
            "options": dithers, "default": d.dither.id(),
            "help": "How the 24-bit output is quantised. TPDF is plain dither; the noise shapers move the noise above the audible band.",
        }),
        json!({
            "key": "headroom_db", "label": "Headroom (dB)", "type": "number",
            "min": HEADROOM_RANGE.0, "max": HEADROOM_RANGE.1, "step": 0.5, "default": d.headroom_db,
            "help": "Level change after upsampling. 0 keeps the original level; the soft clipper catches intersample overs.",
        }),
        json!({
            "key": "soft_clip", "label": "Soft clipper", "type": "bool", "default": d.soft_clip,
            "help": "Transparent soft clipper before quantisation, so intersample peaks never wrap or hard-clip.",
        }),
    ];
    if cadenza_dsp::gpu::COMPILED_IN {
        out.push(json!({
            "key": "gpu", "label": "GPU processing", "type": "select",
            "options": [["auto", "Auto"], ["on", "On"], ["off", "Off"]], "default": "auto",
            "help": gpu_help(),
        }));
        if !devices.is_empty() {
            let options: Vec<Value> = devices
                .iter()
                .map(|(i, n)| json!([i.to_string(), n]))
                .collect();
            out.push(json!({
                "key": "gpu_device", "label": "GPU", "type": "select",
                "options": options, "default": "0",
                "help": "Which GPU processes the audio.",
            }));
        }
        if cadenza_dsp::gpu::WGPU_COMPILED_IN {
            let options: Vec<Value> = GpuPrecision::ALL
                .iter()
                .map(|p| json!([p.id(), p.label()]))
                .collect();
            out.push(json!({
                "key": "gpu_precision", "label": "GPU precision", "type": "select",
                "options": options, "default": GpuPrecision::default().id(),
                "help": "AMD, Intel and Apple GPUs: Auto computes in 64-bit where the card has it and in double-f32 (about 48 bits, rounding noise near -280 dB) where it does not (Apple, Intel Arc). Double-f32 is much faster on most consumer cards. CUDA always uses 64-bit.",
            }));
        }
    }
    out
}

/// The one-line `--describe` answer.
pub fn describe() -> Value {
    let devices = gpu_devices();
    json!({
        "name": "cadenza",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": PROTOCOL,
        "in_formats": InFormat::IDS,
        "out_formats": OUT_FORMATS,
        "gpu": cadenza_dsp::gpu::COMPILED_IN,
        "gpu_apis": gpu_apis(),
        "filters": FilterKind::catalog().into_iter().map(|f| f.id()).collect::<Vec<_>>(),
        "dithers": DitherKind::catalog().iter().map(|k| k.id()).collect::<Vec<_>>(),
        "settings": settings_schema(&devices),
    })
}

/// One render.
#[derive(Clone, Debug)]
pub struct RenderOptions {
    /// Source rate (Hz).
    pub in_rate: u32,
    /// Interleaved channels.
    pub channels: u32,
    /// Input sample format.
    pub in_format: InFormat,
    /// Output rate (Hz).
    pub rate: u32,
    /// Quantiser bit depth (16, 24 or 32), left-justified in each s32 sample.
    pub bits: u32,
    /// Engine settings.
    pub settings: EngineSettings,
    /// Coefficient cache for the long filters.
    pub cache: CoefficientCache,
}

/// Interleaved quantised integers → s32le bytes, left-justified to full scale.
pub fn encode_s32le(ints: &[i32], bits: u32, out: &mut Vec<u8>) {
    let shift = 32 - bits.clamp(1, 32);
    out.clear();
    out.reserve(ints.len() * 4);
    for &v in ints {
        out.extend_from_slice(&(v << shift).to_le_bytes());
    }
}

/// Run `input` through the pipeline into `output` until EOF, then flush the resampler tail.
/// `progress` gets `{"in_seconds":x}` lines, at most one per second. Returns the output
/// frames written.
pub fn run<R: Read, W: Write>(
    opts: &RenderOptions,
    mut input: R,
    mut output: W,
    mut progress: Option<&mut dyn Write>,
) -> Result<u64, String> {
    if opts.channels == 0 || opts.channels > 64 {
        return Err(format!("bad channel count {}", opts.channels));
    }
    if opts.in_rate == 0 || opts.rate == 0 {
        return Err("rates must be above 0".into());
    }
    if !matches!(opts.bits, 16 | 24 | 32) {
        return Err(format!(
            "unsupported bit depth {} (16, 24 or 32)",
            opts.bits
        ));
    }
    let ch = opts.channels as usize;
    let input_format = InputFormat {
        rate: opts.in_rate,
        bits: 32,
        channels: opts.channels,
        codec: "PCM".into(),
    };
    let mut pipe = Pipeline::new(
        input_format,
        opts.settings.pipeline_config(opts.rate, opts.bits),
        &opts.cache,
    )
    .map_err(|e| e.to_string())?;
    let sample = opts.in_format.bytes();
    let frame = sample * ch;
    let mut buf = vec![0u8; IO_BUFFER];
    let mut pending: Vec<u8> = Vec::with_capacity(IO_BUFFER + frame);
    let mut samples: Vec<f64> = Vec::with_capacity(IO_BUFFER / sample);
    let mut bytes: Vec<u8> = Vec::with_capacity(IO_BUFFER * 2);
    let mut frames_in: u64 = 0;
    let mut frames_out: u64 = 0;
    let mut last = Instant::now();
    let write_err = |e: std::io::Error| format!("writing the output: {e}");
    loop {
        let n = match input.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("reading the input: {e}")),
        };
        pending.extend_from_slice(&buf[..n]);
        let whole = pending.len() / frame * frame;
        if whole == 0 {
            continue;
        }
        samples.clear();
        match opts.in_format {
            InFormat::F64le => samples.extend(
                pending[..whole]
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap())),
            ),
            InFormat::F32le => samples.extend(
                pending[..whole]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64),
            ),
        }
        pending.drain(..whole);
        frames_in += (whole / frame) as u64;
        let ints = pipe.process(&samples);
        frames_out += (ints.len() / ch) as u64;
        encode_s32le(&ints, opts.bits, &mut bytes);
        output.write_all(&bytes).map_err(write_err)?;
        if let Some(p) = progress.as_mut() {
            if last.elapsed() >= Duration::from_secs(1) {
                last = Instant::now();
                let line = json!({ "in_seconds": frames_in as f64 / opts.in_rate as f64 });
                let _ = writeln!(p, "{line}");
                let _ = p.flush();
            }
        }
    }
    let ints = pipe.flush();
    frames_out += (ints.len() / ch) as u64;
    encode_s32le(&ints, opts.bits, &mut bytes);
    output.write_all(&bytes).map_err(write_err)?;
    output.flush().map_err(write_err)?;
    if let Some(p) = progress.as_mut() {
        let line = json!({ "in_seconds": frames_in as f64 / opts.in_rate as f64 });
        let _ = writeln!(p, "{line}");
        let _ = p.flush();
    }
    Ok(frames_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_parses_every_key_and_rejects_bad_values() {
        let mut s = EngineSettings::default();
        assert_eq!(s.set("filter", "sinc-m-256k"), Ok(true));
        assert_eq!(s.filter.id(), "sinc-m-256k");
        assert_eq!(s.set("dither", "ns9"), Ok(true));
        assert_eq!(s.dither, DitherKind::Ns9);
        assert_eq!(s.set("headroom_db", "-3"), Ok(true));
        assert_eq!(s.headroom_db, -3.0);
        assert_eq!(s.set("soft_clip", "false"), Ok(true));
        assert!(!s.soft_clip);
        assert_eq!(s.set("gpu", "on"), Ok(true));
        assert_eq!(s.gpu.mode, GpuMode::On);
        assert_eq!(s.set("gpu", "wgpu"), Ok(true));
        assert_eq!(s.gpu.mode, GpuMode::Wgpu);
        assert_eq!(s.set("gpu_precision", "double-f32"), Ok(true));
        assert_eq!(s.gpu.precision, GpuPrecision::DoubleF32);
        assert!(s.set("gpu_precision", "f16").is_err());
        assert_eq!(s.set("gpu", "off"), Ok(true));
        assert_eq!(s.gpu.mode, GpuMode::Off);
        assert_eq!(s.set("gpu_device", "1"), Ok(true));
        assert_eq!(s.gpu.device, 1);
        assert_eq!(s.set("from_the_future", "x"), Ok(false));
        assert!(s.set("filter", "nope").is_err());
        assert!(s.set("dither", "nope").is_err());
        assert!(s.set("headroom_db", "loud").is_err());
        assert!(s.set("headroom_db", "6").is_err());
        assert!(s.set("soft_clip", "maybe").is_err());
        assert!(s.set("gpu", "sometimes").is_err());
        assert!(s.set("gpu_device", "-1").is_err());
        let mut warned = Vec::new();
        let mut s = EngineSettings::default();
        s.apply_sets(&["dither=lipshitz".into(), "colour=blue".into()], |w| {
            warned.push(w.to_string())
        })
        .unwrap();
        assert_eq!(s.dither, DitherKind::Lipshitz);
        assert_eq!(warned.len(), 1);
        assert!(warned[0].contains("colour"));
        assert!(s.apply_sets(&["no-equals".into()], |_| {}).is_err());
    }

    #[test]
    fn schema_covers_every_key_with_a_valid_default() {
        let devices = vec![(0, "NVIDIA Test".to_string())];
        let schema = settings_schema(&devices);
        let keys: Vec<&str> = schema.iter().map(|s| s["key"].as_str().unwrap()).collect();
        if cadenza_dsp::gpu::WGPU_COMPILED_IN {
            assert_eq!(keys, KEYS);
        } else if cadenza_dsp::gpu::COMPILED_IN {
            assert_eq!(keys, &KEYS[..6]);
        } else {
            assert_eq!(keys, &KEYS[..4]);
        }
        for s in &schema {
            assert!(s["label"].as_str().is_some_and(|l| !l.is_empty()));
            assert!(s["help"].as_str().is_some_and(|l| !l.is_empty()));
            let ty = s["type"].as_str().unwrap();
            assert!(matches!(ty, "select" | "bool" | "number"));
            let default = match &s["default"] {
                Value::String(v) => v.clone(),
                other => other.to_string(),
            };
            let mut e = EngineSettings::default();
            assert_eq!(e.set(s["key"].as_str().unwrap(), &default), Ok(true));
            if ty == "select" {
                let opts = s["options"].as_array().unwrap();
                assert!(opts.iter().any(|o| o[0] == s["default"]));
                for o in opts {
                    assert_eq!(
                        e.set(s["key"].as_str().unwrap(), o[0].as_str().unwrap()),
                        Ok(true)
                    );
                }
            }
        }
        // no GPU found: no gpu_device entry
        assert!(settings_schema(&[])
            .iter()
            .all(|s| s["key"] != "gpu_device"));
    }

    #[test]
    fn left_justifies_like_encore() {
        let mut b = Vec::new();
        encode_s32le(&[1, -1, (1 << 23) - 1], 24, &mut b);
        assert_eq!(&b[0..4], &(256i32).to_le_bytes());
        assert_eq!(&b[4..8], &(-256i32).to_le_bytes());
        assert_eq!(&b[8..12], &(((1i32 << 23) - 1) << 8).to_le_bytes());
        encode_s32le(&[5], 32, &mut b);
        assert_eq!(b, 5i32.to_le_bytes());
    }
}
