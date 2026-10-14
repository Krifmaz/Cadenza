//! CUDA offload of the partitioned convolution, in 64-bit, for NVIDIA cards.
//!
//! The GPU runs the engine both the long-FIR resampler and room convolution are built on:
//! the filter's partition spectra are uploaded once (`f64` complex), and per block the
//! input goes up, is transformed by our own Stockham radix-4/2 FFT kernels (CUDA C in
//! `kernels.cu`, compiled on the user's machine by the driver's NVRTC), multiplied and
//! accumulated against every partition from a frequency-delay line that stays on the
//! device, inverse transformed and downloaded. The sigma-delta modulator is sequential
//! and stays on the CPU.
//!
//! Building needs no CUDA toolkit: `cudarc` is used with its `dynamic-loading` feature,
//! so the binary links nothing and loads `libcuda` / `nvcuda.dll` at run time. Without the
//! `cuda` cargo feature, without an NVIDIA driver (≥ 525) or without a card, [`Backend::detect`]
//! picks the CPU, says so once in the log, and everything else behaves exactly the same.
//!
//! Settings: `gpu = auto | off | cuda` ([`GpuMode`]) and `gpu_device = 0` ([`GpuConfig`]).
//! The signal path shows `CUDA (RTX 4060) · 64-bit` ([`Backend::label`]).

pub mod kernels;
pub mod reference;

#[cfg(feature = "cuda")]
mod cuda;

use crate::convolution::{BlockConvolver, Convolver as CpuConvolver, SpectralFilter};
use crate::signal_path::Stage;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
pub use cuda::Convolver;

/// `true` when this binary was built with `--features cuda` (the only way CUDA can ever be used).
pub const COMPILED_IN: bool = cfg!(feature = "cuda");

/// Minimum NVIDIA driver version (the one that ships CUDA 12 and its NVRTC).
pub const MIN_DRIVER_VERSION: u32 = 525;

/// Whether to use the GPU (`gpu` in config.toml and the settings API).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuMode {
    /// Use CUDA when a device is present, the CPU otherwise.
    #[default]
    Auto,
    /// Never touch the GPU.
    Off,
    /// Require CUDA; fall back to the CPU with a warning when it is unavailable.
    Cuda,
}

impl GpuMode {
    /// Every mode, for UIs.
    pub const ALL: [GpuMode; 3] = [GpuMode::Auto, GpuMode::Off, GpuMode::Cuda];

    /// Config id: `auto`, `off`, `cuda`.
    pub fn id(self) -> &'static str {
        match self {
            GpuMode::Auto => "auto",
            GpuMode::Off => "off",
            GpuMode::Cuda => "cuda",
        }
    }

    /// Display name.
    pub fn label(self) -> &'static str {
        match self {
            GpuMode::Auto => "Automatic",
            GpuMode::Off => "Off (CPU)",
            GpuMode::Cuda => "CUDA",
        }
    }
}

impl std::str::FromStr for GpuMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "automatic" | "" => Ok(GpuMode::Auto),
            "off" | "cpu" | "none" | "false" => Ok(GpuMode::Off),
            "cuda" | "gpu" | "on" | "true" => Ok(GpuMode::Cuda),
            other => Err(format!("unknown gpu mode {other:?} (auto, off or cuda)")),
        }
    }
}

impl fmt::Display for GpuMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// The GPU settings (`gpu` and `gpu_device` keys; `#[serde(flatten)]` them into a settings
/// struct, or nest them as `{"mode": …, "device": …}` – both spellings deserialise).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    /// `auto`, `off` or `cuda`.
    #[serde(rename = "gpu", alias = "mode")]
    pub mode: GpuMode,
    /// CUDA device ordinal (0 = the first card).
    #[serde(rename = "gpu_device", alias = "device")]
    pub device: usize,
}

/// Why the GPU could not be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GpuError {
    /// `gpu = off`.
    #[error("gpu = off")]
    Disabled,
    /// The binary was built without `--features cuda`.
    #[error("built without the `cuda` feature")]
    NotCompiled,
    /// `libcuda` / `nvcuda.dll` could not be loaded (no NVIDIA driver on this machine).
    #[error("no NVIDIA driver found ({0}); CUDA needs driver {MIN_DRIVER_VERSION} or newer")]
    NoDriver(String),
    /// The driver loaded but reports no usable device (or not the requested ordinal).
    #[error("{0}")]
    NoDevice(String),
    /// A CUDA driver call failed.
    #[error("CUDA driver: {0}")]
    Driver(String),
    /// NVRTC rejected the kernel source.
    #[error("NVRTC: {0}")]
    Compile(String),
}

/// What the convolution engines will run on, decided once per configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Plain CPU engine (rayon-parallel), with the reason the GPU was not chosen.
    Cpu {
        /// Why (`gpu = off`, `no NVIDIA driver`…), for the log line and the signal path.
        reason: String,
    },
    /// A CUDA device, ready to build [`Convolver`]s.
    Cuda {
        /// Device ordinal.
        device: usize,
        /// Device name as the driver reports it (`NVIDIA GeForce RTX 4060`).
        name: String,
        /// Compute capability `(major, minor)`.
        compute: (i32, i32),
    },
}

impl Backend {
    /// Decide for `config`: probe the driver and the device (and remember the answer, so a
    /// pipeline rebuilt per track does not re-probe or re-log). The choice – and the reason
    /// for a CPU fallback – is logged once.
    pub fn detect(config: &GpuConfig) -> Backend {
        static LAST: Mutex<Option<(GpuConfig, Backend)>> = Mutex::new(None);
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((c, b)) = last.as_ref() {
            if c == config {
                return b.clone();
            }
        }
        let backend = match Self::probe(config) {
            Ok(b) => {
                log::info!("GPU: {} available", b.label());
                b
            }
            Err(e) => {
                let reason = e.to_string();
                match (config.mode, &e) {
                    (GpuMode::Off, _) => log::info!("GPU: off (CPU · 64-bit)"),
                    (GpuMode::Cuda, _) => {
                        log::warn!("GPU: CUDA requested but {reason}; using the CPU (64-bit)")
                    }
                    (GpuMode::Auto, GpuError::NotCompiled) => {
                        log::debug!("GPU: {reason}; using the CPU (64-bit)")
                    }
                    (GpuMode::Auto, _) => {
                        log::info!("GPU: CUDA unavailable ({reason}); using the CPU (64-bit)")
                    }
                }
                Backend::Cpu { reason }
            }
        };
        *last = Some((*config, backend.clone()));
        backend
    }

    /// Probe without caching or logging: `Ok` with a CUDA backend, or why not.
    pub fn probe(config: &GpuConfig) -> Result<Backend, GpuError> {
        if config.mode == GpuMode::Off {
            return Err(GpuError::Disabled);
        }
        #[cfg(feature = "cuda")]
        {
            cuda::probe(config.device)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::NotCompiled)
        }
    }

    /// `true` for a CUDA backend.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Backend::Cuda { .. })
    }

    /// Short device name: `RTX 4060` for `NVIDIA GeForce RTX 4060`.
    pub fn device_short_name(&self) -> Option<String> {
        match self {
            Backend::Cuda { name, .. } => Some(short_device_name(name)),
            Backend::Cpu { .. } => None,
        }
    }

    /// `CUDA (RTX 4060) · 64-bit` or `CPU · 64-bit`, for the signal path.
    pub fn label(&self) -> String {
        match self {
            Backend::Cuda { name, .. } => format!("CUDA ({}) · 64-bit", short_device_name(name)),
            Backend::Cpu { .. } => "CPU · 64-bit".to_string(),
        }
    }

    /// The signal path stage for the compute backend (id `gpu`). `used` says whether a
    /// convolution engine of this chain actually runs on the device: the stage is active
    /// (and the UI badge green) only then – a detected card that no stage of the current
    /// chain uses is reported as `available, not used by this chain yet`.
    pub fn stage(&self, config: &GpuConfig, used: bool) -> Stage {
        let used = used && self.is_gpu();
        let detail = match self {
            Backend::Cuda { .. } if used => format!("{} · room convolution", self.label()),
            Backend::Cuda { .. } => {
                format!("{} · available, not used by this chain yet", self.label())
            }
            Backend::Cpu { reason } if config.mode == GpuMode::Off => self.label(),
            Backend::Cpu { reason } => format!("{} · {reason}", self.label()),
        };
        let mut params = self.report(config);
        params["used"] = json!(used);
        Stage {
            id: "gpu".into(),
            name: "Compute".into(),
            detail,
            params,
            active: used,
        }
    }

    /// Full parameters for the signal path / API.
    pub fn report(&self, config: &GpuConfig) -> serde_json::Value {
        match self {
            Backend::Cuda {
                device,
                name,
                compute,
            } => json!({
                "mode": config.mode.id(),
                "backend": "cuda",
                "label": self.label(),
                "device": device,
                "device_name": name,
                "compute_capability": format!("{}.{}", compute.0, compute.1),
                "precision": "f64",
                "compiled_in": COMPILED_IN,
            }),
            Backend::Cpu { reason } => json!({
                "mode": config.mode.id(),
                "backend": "cpu",
                "label": self.label(),
                "device": config.device,
                "reason": reason,
                "precision": "f64",
                "compiled_in": COMPILED_IN,
            }),
        }
    }

    /// A block convolver for `filter` on this backend: a [`Convolver`] on CUDA, or the CPU
    /// [`crate::convolution::Convolver`]. A CUDA failure while building (out of device
    /// memory, kernel compile error…) is logged and falls back to the CPU.
    pub fn convolver(&self, filter: Arc<SpectralFilter>) -> Box<dyn BlockConvolver + Send> {
        #[cfg(feature = "cuda")]
        if let Backend::Cuda { device, .. } = self {
            match Convolver::from_filter(Arc::clone(&filter), *device) {
                Ok(c) => return Box::new(c),
                Err(e) => log::warn!("GPU: {e}; this convolver runs on the CPU"),
            }
        }
        Box::new(CpuConvolver::from_filter(filter))
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

/// `NVIDIA GeForce RTX 4060 Laptop GPU` → `RTX 4060 Laptop GPU`.
pub fn short_device_name(name: &str) -> String {
    let mut s = name.trim();
    for prefix in ["NVIDIA ", "GeForce ", "Tesla ", "Quadro "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.trim_start();
        }
    }
    if s.is_empty() {
        name.trim().to_string()
    } else {
        s.to_string()
    }
}

/// The GPU convolver type when the crate is built without the `cuda` feature: it has no
/// values, so the API (and code written against it) compiles either way, and every
/// constructor returns [`GpuError::NotCompiled`].
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub enum Convolver {}

#[cfg(not(feature = "cuda"))]
impl Convolver {
    /// Always `Err(GpuError::NotCompiled)` in this build.
    pub fn from_filter(_filter: Arc<SpectralFilter>, _device: usize) -> Result<Self, GpuError> {
        Err(GpuError::NotCompiled)
    }

    /// Always `Err(GpuError::NotCompiled)` in this build.
    pub fn new(_h: &[f64], _block: usize, _device: usize) -> Result<Self, GpuError> {
        Err(GpuError::NotCompiled)
    }
}

#[cfg(not(feature = "cuda"))]
impl BlockConvolver for Convolver {
    fn process_block(&mut self, _input: &[f64], _out: &mut [f64]) {
        match *self {}
    }
    fn latency(&self) -> usize {
        match *self {}
    }
    fn reset(&mut self) {
        match *self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A tiny `log` sink that counts records mentioning the CPU fallback.
    struct CountingLogger;
    static FALLBACK_LINES: AtomicUsize = AtomicUsize::new(0);
    impl log::Log for CountingLogger {
        fn enabled(&self, _: &log::Metadata) -> bool {
            true
        }
        fn log(&self, record: &log::Record) {
            let msg = record.args().to_string();
            if msg.starts_with("GPU:") && msg.contains("CPU") {
                FALLBACK_LINES.fetch_add(1, Ordering::SeqCst);
            }
        }
        fn flush(&self) {}
    }

    #[test]
    fn cpu_fallback_is_chosen_and_logged_without_a_device() {
        // Another test (or the player) may already own the global logger; then we only
        // check the choice, not the log line.
        let logger_installed = log::set_logger(&CountingLogger).is_ok();
        log::set_max_level(log::LevelFilter::Trace);

        // `cuda` requested: there is no device on this machine (and in the default build
        // no CUDA support at all), so the CPU is chosen and a warning says why.
        let config = GpuConfig {
            mode: GpuMode::Cuda,
            device: 0,
        };
        let before = FALLBACK_LINES.load(Ordering::SeqCst);
        let backend = Backend::detect(&config);
        assert!(!backend.is_gpu(), "no GPU here, got {backend:?}");
        assert_eq!(backend.label(), "CPU · 64-bit");
        let Backend::Cpu { reason } = &backend else {
            unreachable!()
        };
        if COMPILED_IN {
            assert!(
                reason.contains("driver") || reason.contains("device") || reason.contains("CUDA"),
                "{reason}"
            );
        } else {
            assert_eq!(reason, &GpuError::NotCompiled.to_string());
        }
        if logger_installed {
            assert_eq!(
                FALLBACK_LINES.load(Ordering::SeqCst),
                before + 1,
                "one log line"
            );
            // Cached: asking again for the same config neither re-probes nor re-logs.
            let again = Backend::detect(&config);
            assert_eq!(again, backend);
            assert_eq!(FALLBACK_LINES.load(Ordering::SeqCst), before + 1);
        }

        // The stage and report say CPU and carry the reason.
        let stage = backend.stage(&config, true);
        assert_eq!(stage.id, "gpu");
        assert!(!stage.active);
        assert!(
            stage.detail.starts_with("CPU · 64-bit · "),
            "{}",
            stage.detail
        );
        assert_eq!(stage.params["backend"], "cpu");
        assert_eq!(stage.params["precision"], "f64");
        assert_eq!(stage.params["mode"], "cuda");

        // `off` never probes and reports plainly.
        let off = Backend::detect(&GpuConfig {
            mode: GpuMode::Off,
            device: 0,
        });
        assert_eq!(
            off,
            Backend::Cpu {
                reason: "gpu = off".into()
            }
        );
        assert_eq!(
            off.stage(
                &GpuConfig {
                    mode: GpuMode::Off,
                    device: 0
                },
                true
            )
            .detail,
            "CPU · 64-bit"
        );

        // The convolver factory hands out a working CPU engine.
        let filter = Arc::new(SpectralFilter::new(&[0.5, 0.25], 64));
        let mut conv = backend.convolver(filter);
        assert_eq!(conv.latency(), 64);
        let input: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let mut out = vec![0.0; 64];
        conv.process_block(&input, &mut out);
        assert!((out[1] - 0.5).abs() < 1e-12 && (out[2] - 1.25).abs() < 1e-12);
    }

    #[test]
    fn gpu_convolver_constructor_reports_why_without_a_device() {
        let err = Convolver::new(&[1.0], 64, 0).expect_err("no GPU on this machine");
        if COMPILED_IN {
            assert!(matches!(
                err,
                GpuError::NoDriver(_) | GpuError::NoDevice(_) | GpuError::Driver(_)
            ));
        } else {
            assert_eq!(err, GpuError::NotCompiled);
        }
    }

    #[test]
    fn config_keys_and_labels() {
        let c: GpuConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c, GpuConfig::default());
        assert_eq!(c.mode, GpuMode::Auto);
        let c: GpuConfig = serde_json::from_str(r#"{"gpu":"cuda","gpu_device":1}"#).unwrap();
        assert_eq!(c.mode, GpuMode::Cuda);
        assert_eq!(c.device, 1);
        let nested: GpuConfig = serde_json::from_str(r#"{"mode":"off","device":2}"#).unwrap();
        assert_eq!(nested.mode, GpuMode::Off);
        assert_eq!(nested.device, 2);
        let s = serde_json::to_value(c).unwrap();
        assert_eq!(s, json!({"gpu": "cuda", "gpu_device": 1}));
        // config.toml spelling through the flattened keys.
        #[derive(Deserialize)]
        struct Dsp {
            #[serde(flatten)]
            gpu: GpuConfig,
        }
        let d: Dsp = serde_json::from_str(r#"{"gpu":"off"}"#).unwrap();
        assert_eq!(d.gpu.mode, GpuMode::Off);

        assert_eq!("CUDA".parse::<GpuMode>().unwrap(), GpuMode::Cuda);
        assert_eq!("cpu".parse::<GpuMode>().unwrap(), GpuMode::Off);
        assert!("metal".parse::<GpuMode>().is_err());
        for m in GpuMode::ALL {
            assert_eq!(m.id().parse::<GpuMode>().unwrap(), m);
        }

        let cuda = Backend::Cuda {
            device: 0,
            name: "NVIDIA GeForce RTX 4060".into(),
            compute: (8, 9),
        };
        assert_eq!(cuda.label(), "CUDA (RTX 4060) · 64-bit");
        assert_eq!(cuda.device_short_name().unwrap(), "RTX 4060");
        assert_eq!(short_device_name("Tesla T4"), "T4");
        assert_eq!(short_device_name("NVIDIA"), "NVIDIA");
        let stage = cuda.stage(&GpuConfig::default(), true);
        assert!(stage.active);
        assert_eq!(stage.detail, "CUDA (RTX 4060) · 64-bit · room convolution");
        assert_eq!(stage.params["compute_capability"], "8.9");
        assert_eq!(stage.params["precision"], "f64");
        assert_eq!(stage.params["used"], true);
        // Detected but not driving any engine of this chain: honest, and not active.
        let idle = cuda.stage(&GpuConfig::default(), false);
        assert!(!idle.active);
        assert_eq!(
            idle.detail,
            "CUDA (RTX 4060) · 64-bit · available, not used by this chain yet"
        );
        assert_eq!(idle.params["used"], false);
    }
}
