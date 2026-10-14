//! GPU offload of the partitioned convolution: CUDA (64-bit) for NVIDIA cards, and `wgpu`
//! (Vulkan, Metal, DX12) for AMD, Intel, Apple Silicon and NVIDIA cards.
//!
//! The GPU runs the engine both the long-FIR resampler and room convolution are built on:
//! the filter's partition spectra are uploaded once (`f64` complex), and per block the
//! input goes up, is transformed by our own Stockham radix-4/2 FFT kernels (CUDA C in
//! `kernels.cu`, compiled on the user's machine by the driver's NVRTC), multiplied and
//! accumulated against every partition from a frequency-delay line that stays on the
//! device, inverse transformed and downloaded. The sigma-delta modulator is sequential
//! and stays on the CPU.
//!
//! The upsampler's FFT polyphase core runs there too ([`PolyphaseConvolver`]): the `L`
//! sub-filter spectra are uploaded once per stage (shared by the channels), and per input
//! block the device does one forward transform into the delay line, one batched
//! multiply-accumulate over every sub-filter (packed two per complex spectrum), one
//! batched inverse transform and the interleaving into output order; only the finished,
//! interleaved block comes back. Delay compensation, decimation and flushing stay in the
//! resampler, identical for both cores. A CUDA error while building leaves that stage on
//! the CPU; one mid-stream moves it to the CPU core with the history replayed (no gap, no
//! glitch), logged once.
//!
//! Building needs no CUDA toolkit: `cudarc` is used with its `dynamic-loading` feature,
//! so the binary links nothing and loads `libcuda` / `nvcuda.dll` at run time. Without the
//! `cuda` cargo feature, without an NVIDIA driver (≥ 525) or without a card, [`Backend::detect`]
//! picks the CPU, says so once in the log, and everything else behaves exactly the same.
//!
//! **wgpu** (`--features wgpu`): the same polyphase core written in WGSL ([`wgsl`]), run
//! through `wgpu` on Vulkan (Linux, Windows, Android), Metal (macOS) or DX12 (Windows).
//! Cards with 64-bit shader support compute in native `f64`; the others (Apple Silicon,
//! Intel Arc, most mobile GPUs) in double-f32 – two `f32` per value, about 48 bits of
//! mantissa – and the signal path says which. A device is opened once, its kernels
//! compiled once, and it must pass a self-test against the CPU before it is used. Room
//! convolution runs there too (the polyphase core with one phase), with the same
//! seamless move to the CPU if the device fails mid-stream.
//!
//! Settings: `gpu = auto | on | off | cuda | wgpu` ([`GpuMode`]), `gpu_device = 0` and
//! `gpu_precision = auto | f64 | double-f32` ([`GpuConfig`]). `auto` tries CUDA first, then
//! wgpu (`on` too, with a warning when there is no GPU). The signal path shows `CUDA (RTX 4060) · 64-bit` or
//! `Metal (Apple M2) · double-f32 (≈48-bit)` ([`Backend::label`]).

pub mod kernels;
pub mod reference;
pub mod wgsl;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

use crate::convolution::{
    BlockConvolver, Convolver as CpuConvolver, PolyphaseBlockConvolver, RationalDotEngine,
    SpectralFilter,
};
use crate::signal_path::Stage;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;
use std::sync::{Arc, Mutex};
pub use wgsl::Arith;

#[cfg(feature = "cuda")]
pub use cuda::{Convolver, PolyphaseConvolver};

/// Builds the polyphase upsampler cores of one resampler stage on some engine: the CUDA
/// device ([`Backend`]) or the CPU model of the GPU algorithm
/// ([`reference::ReferenceFactory`]). The resampler calls it once per FFT stage.
pub trait PolyphaseFactory: Sync {
    /// One core per channel for the `L = filters.len()` sub-filters (all partitioned for
    /// the same block). The filter spectra should be uploaded once and shared.
    fn polyphase(
        &self,
        filters: &[SpectralFilter],
        channels: usize,
    ) -> Result<Vec<Box<dyn PolyphaseBlockConvolver + Send>>, GpuError>;

    /// One time-domain polyphase engine per channel for a rational stage (44.1 ↔ 48 kHz
    /// families): the `L` reversed sub-filters `sub` of `S` taps each, uploaded once and
    /// shared. The default keeps the stage on the CPU.
    fn rational(
        &self,
        sub: &[Vec<f64>],
        channels: usize,
    ) -> Result<Vec<Box<dyn RationalDotEngine + Send>>, GpuError> {
        let _ = (sub, channels);
        Err(GpuError::Disabled)
    }
}

impl PolyphaseFactory for Backend {
    /// CUDA or wgpu cores on a GPU backend; `Err` on the CPU backend (the caller then keeps
    /// its own CPU core).
    fn polyphase(
        &self,
        filters: &[SpectralFilter],
        channels: usize,
    ) -> Result<Vec<Box<dyn PolyphaseBlockConvolver + Send>>, GpuError> {
        match self {
            Backend::Cuda { device, .. } => {
                PolyphaseConvolver::for_channels(filters, channels, *device)
            }
            Backend::Wgpu { device, arith, .. } => {
                wgpu_polyphase(filters, channels, *device, *arith)
            }
            Backend::Cpu { reason } => Err(cpu_reason(reason)),
        }
    }

    /// CUDA or wgpu rational engines on a GPU backend; `Err` on the CPU backend.
    #[allow(unused_variables)]
    fn rational(
        &self,
        sub: &[Vec<f64>],
        channels: usize,
    ) -> Result<Vec<Box<dyn RationalDotEngine + Send>>, GpuError> {
        match self {
            Backend::Cuda { device, .. } => {
                #[cfg(feature = "cuda")]
                {
                    cuda::RationalEngine::for_channels(sub, channels, *device)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(GpuError::NotCompiled)
                }
            }
            Backend::Wgpu { device, arith, .. } => {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_backend::RationalEngine::for_channels(
                        sub,
                        channels,
                        *device,
                        (*arith).into(),
                    )
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    Err(GpuError::NotCompiled)
                }
            }
            Backend::Cpu { reason } => Err(cpu_reason(reason)),
        }
    }
}

/// The error a CPU backend answers with: `NotCompiled` or `Disabled`.
fn cpu_reason(reason: &str) -> GpuError {
    if reason == GpuError::NotCompiled.to_string() {
        GpuError::NotCompiled
    } else {
        GpuError::Disabled
    }
}

/// Which engines of a chain run on the GPU (for the `gpu` signal path stage).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuUse {
    /// The upsampler's FFT polyphase core.
    pub upsampling: bool,
    /// The room convolution.
    pub convolution: bool,
}

impl GpuUse {
    /// Anything at all.
    pub fn any(self) -> bool {
        self.upsampling || self.convolution
    }

    /// `upsampling + room convolution`, `upsampling`, `room convolution`.
    pub fn label(self) -> &'static str {
        match (self.upsampling, self.convolution) {
            (true, true) => "upsampling + room convolution",
            (true, false) => "upsampling",
            (false, true) => "room convolution",
            (false, false) => "",
        }
    }
}

/// `true` when this binary has any GPU support (`--features cuda` and/or `--features wgpu`).
pub const COMPILED_IN: bool = cfg!(any(feature = "cuda", feature = "wgpu"));

/// `true` when this binary was built with `--features cuda` (the only way CUDA can ever be used).
pub const CUDA_COMPILED_IN: bool = cfg!(feature = "cuda");

/// `true` when this binary was built with `--features wgpu` (Vulkan, Metal, DX12).
pub const WGPU_COMPILED_IN: bool = cfg!(feature = "wgpu");

/// wgpu polyphase cores on adapter `device` in `arith`.
#[allow(unused_variables)]
fn wgpu_polyphase(
    filters: &[SpectralFilter],
    channels: usize,
    device: usize,
    arith: Arith,
) -> Result<Vec<Box<dyn PolyphaseBlockConvolver + Send>>, GpuError> {
    #[cfg(feature = "wgpu")]
    {
        wgpu_backend::PolyphaseCore::for_channels(filters, channels, device, arith.into())
    }
    #[cfg(not(feature = "wgpu"))]
    {
        Err(GpuError::NotCompiled)
    }
}

/// Minimum NVIDIA driver version (the one that ships CUDA 12 and its NVRTC).
pub const MIN_DRIVER_VERSION: u32 = 525;

/// Whether to use the GPU (`gpu` in config.toml and the settings API).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuMode {
    /// Use a GPU when one is present (CUDA first, then wgpu), the CPU otherwise.
    #[default]
    Auto,
    /// Never touch the GPU.
    Off,
    /// Require CUDA; fall back to the CPU with a warning when it is unavailable.
    Cuda,
    /// Require wgpu (Vulkan, Metal, DX12); fall back to the CPU with a warning when it is
    /// unavailable.
    Wgpu,
    /// Use any GPU (CUDA first, then wgpu); fall back to the CPU with a warning when there
    /// is none.
    On,
}

impl GpuMode {
    /// Every mode, for UIs.
    pub const ALL: [GpuMode; 5] = [
        GpuMode::Auto,
        GpuMode::On,
        GpuMode::Off,
        GpuMode::Cuda,
        GpuMode::Wgpu,
    ];

    /// The modes this build can honour (`auto` and `off` always; `cuda` and `wgpu` when
    /// compiled in), for UIs.
    pub fn available() -> Vec<GpuMode> {
        Self::ALL
            .into_iter()
            .filter(|m| match m {
                GpuMode::Cuda => CUDA_COMPILED_IN,
                GpuMode::Wgpu => WGPU_COMPILED_IN,
                _ => true,
            })
            .collect()
    }

    /// Config id: `auto`, `off`, `cuda`.
    pub fn id(self) -> &'static str {
        match self {
            GpuMode::Auto => "auto",
            GpuMode::Off => "off",
            GpuMode::Cuda => "cuda",
            GpuMode::Wgpu => "wgpu",
            GpuMode::On => "on",
        }
    }

    /// Display name.
    pub fn label(self) -> &'static str {
        match self {
            GpuMode::Auto => "Automatic",
            GpuMode::Off => "Off (CPU)",
            GpuMode::Cuda => "CUDA",
            GpuMode::Wgpu => "Vulkan / Metal / DX12",
            GpuMode::On => "On (any GPU)",
        }
    }
}

impl std::str::FromStr for GpuMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "automatic" | "" => Ok(GpuMode::Auto),
            "off" | "cpu" | "none" | "false" => Ok(GpuMode::Off),
            "cuda" => Ok(GpuMode::Cuda),
            "wgpu" | "vulkan" | "metal" | "dx12" => Ok(GpuMode::Wgpu),
            "gpu" | "on" | "true" => Ok(GpuMode::On),
            other => Err(format!(
                "unknown gpu mode {other:?} (auto, on, off, cuda or wgpu)"
            )),
        }
    }
}

impl fmt::Display for GpuMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// The arithmetic of the wgpu kernels (`gpu_precision`). CUDA always computes in `f64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuPrecision {
    /// Native `f64` where the card has it, double-f32 elsewhere.
    #[default]
    Auto,
    /// Native `f64` only: a card without it is not used (the CPU runs instead).
    F64,
    /// Double-f32 (about 48 bits) even where `f64` exists – much faster on consumer
    /// cards, whose `f64` units run at 1/16 … 1/64 of their `f32` rate.
    DoubleF32,
}

impl GpuPrecision {
    /// Every choice, for UIs.
    pub const ALL: [GpuPrecision; 3] = [
        GpuPrecision::Auto,
        GpuPrecision::F64,
        GpuPrecision::DoubleF32,
    ];

    /// Config id: `auto`, `f64`, `double-f32`.
    pub fn id(self) -> &'static str {
        match self {
            GpuPrecision::Auto => "auto",
            GpuPrecision::F64 => "f64",
            GpuPrecision::DoubleF32 => "double-f32",
        }
    }

    /// Display name.
    pub fn label(self) -> &'static str {
        match self {
            GpuPrecision::Auto => "Automatic (64-bit where the GPU has it)",
            GpuPrecision::F64 => "64-bit only",
            GpuPrecision::DoubleF32 => "Double-f32 (≈48-bit, fastest)",
        }
    }
}

impl From<Arith> for GpuPrecision {
    fn from(a: Arith) -> Self {
        match a {
            Arith::F64 => GpuPrecision::F64,
            Arith::DoubleF32 => GpuPrecision::DoubleF32,
        }
    }
}

impl std::str::FromStr for GpuPrecision {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "automatic" | "" => Ok(GpuPrecision::Auto),
            "f64" | "64" | "64-bit" | "double" => Ok(GpuPrecision::F64),
            "double-f32" | "df64" | "48-bit" | "fast" => Ok(GpuPrecision::DoubleF32),
            other => Err(format!(
                "unknown gpu precision {other:?} (auto, f64 or double-f32)"
            )),
        }
    }
}

impl fmt::Display for GpuPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// The GPU settings (`gpu`, `gpu_device` and `gpu_precision` keys; `#[serde(flatten)]` them
/// into a settings struct, or nest them as `{"mode": …, "device": …, "precision": …}` –
/// both spellings deserialise).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    /// `auto`, `on`, `off`, `cuda` or `wgpu`.
    #[serde(rename = "gpu", alias = "mode")]
    pub mode: GpuMode,
    /// Device index (0 = the first card) of the API in use.
    #[serde(rename = "gpu_device", alias = "device")]
    pub device: usize,
    /// Arithmetic of the wgpu kernels.
    #[serde(rename = "gpu_precision", alias = "precision")]
    pub precision: GpuPrecision,
}

/// Why the GPU could not be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GpuError {
    /// `gpu = off`.
    #[error("gpu = off")]
    Disabled,
    /// The binary was built without the GPU feature asked for (`cuda` / `wgpu`).
    #[error("built without GPU support (the `cuda` / `wgpu` features)")]
    NotCompiled,
    /// The GPU driver could not be loaded (no NVIDIA driver for CUDA; no Vulkan, Metal or
    /// DX12 device for wgpu).
    #[error("no GPU driver found ({0})")]
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
    /// A wgpu device (opened, kernels compiled, self-test passed).
    Wgpu {
        /// Adapter index.
        device: usize,
        /// Adapter name (`AMD Radeon RX 7800 XT`, `Apple M2`).
        name: String,
        /// `Vulkan`, `Metal` or `DX12`.
        api: String,
        /// The arithmetic the kernels run in.
        arith: Arith,
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
                    (GpuMode::Cuda | GpuMode::Wgpu | GpuMode::On, _) => log::warn!(
                        "GPU: {} requested but {reason}; using the CPU (64-bit)",
                        config.mode.label()
                    ),
                    (GpuMode::Auto, GpuError::NotCompiled) => {
                        log::debug!("GPU: {reason}; using the CPU (64-bit)")
                    }
                    (GpuMode::Auto, _) => {
                        log::info!("GPU: unavailable ({reason}); using the CPU (64-bit)")
                    }
                }
                Backend::Cpu { reason }
            }
        };
        *last = Some((*config, backend.clone()));
        backend
    }

    /// Probe without caching or logging: `Ok` with a GPU backend, or why not. `auto` tries
    /// CUDA, then wgpu. (A wgpu device is opened, compiled and self-tested on its first
    /// probe; later probes reuse it.)
    pub fn probe(config: &GpuConfig) -> Result<Backend, GpuError> {
        match config.mode {
            GpuMode::Off => Err(GpuError::Disabled),
            GpuMode::Cuda => probe_cuda(config.device),
            GpuMode::Wgpu => probe_wgpu(config.device, config.precision),
            GpuMode::Auto | GpuMode::On => match probe_cuda(config.device) {
                Ok(b) => Ok(b),
                Err(ce) => match probe_wgpu(config.device, config.precision) {
                    Ok(b) => Ok(b),
                    Err(we) => Err(match (ce, we) {
                        (GpuError::NotCompiled, other) | (other, GpuError::NotCompiled) => other,
                        (ce, we) => GpuError::NoDevice(format!("CUDA: {ce}; wgpu: {we}")),
                    }),
                },
            },
        }
    }

    /// `true` for a GPU backend.
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Backend::Cpu { .. })
    }

    /// Short device name: `RTX 4060` for `NVIDIA GeForce RTX 4060`.
    pub fn device_short_name(&self) -> Option<String> {
        match self {
            Backend::Cuda { name, .. } | Backend::Wgpu { name, .. } => {
                Some(short_device_name(name))
            }
            Backend::Cpu { .. } => None,
        }
    }

    /// `CUDA (RTX 4060) · 64-bit`, `Metal (Apple M2) · double-f32 (≈48-bit)` or
    /// `CPU · 64-bit`, for the signal path.
    pub fn label(&self) -> String {
        match self {
            Backend::Cuda { name, .. } => format!("CUDA ({}) · 64-bit", short_device_name(name)),
            Backend::Wgpu {
                name, api, arith, ..
            } => format!("{api} ({}) · {}", short_device_name(name), arith.label()),
            Backend::Cpu { .. } => "CPU · 64-bit".to_string(),
        }
    }

    /// The signal path stage for the compute backend (id `gpu`). `used` says whether a
    /// convolution engine of this chain actually runs on the device: the stage is active
    /// (and the UI badge green) only then – a detected card that no stage of the current
    /// chain uses is reported as `available, not used by this chain yet`.
    pub fn stage(&self, config: &GpuConfig, used: bool) -> Stage {
        self.stage_for(
            config,
            GpuUse {
                upsampling: false,
                convolution: used,
            },
        )
    }

    /// The signal path stage for the compute backend, naming what runs on the device:
    /// `CUDA (RTX 4060) · 64-bit · upsampling + room convolution`.
    pub fn stage_for(&self, config: &GpuConfig, uses: GpuUse) -> Stage {
        let used = uses.any() && self.is_gpu();
        let detail = match self {
            Backend::Cuda { .. } | Backend::Wgpu { .. } if used => {
                format!("{} · {}", self.label(), uses.label())
            }
            Backend::Cuda { .. } | Backend::Wgpu { .. } => {
                format!("{} · available, not used by this chain yet", self.label())
            }
            Backend::Cpu { reason } if config.mode == GpuMode::Off => self.label(),
            Backend::Cpu { reason } => format!("{} · {reason}", self.label()),
        };
        let mut params = self.report(config);
        params["used"] = json!(used);
        params["upsampling"] = json!(used && uses.upsampling);
        params["convolution"] = json!(used && uses.convolution);
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
            Backend::Wgpu {
                device,
                name,
                api,
                arith,
            } => json!({
                "mode": config.mode.id(),
                "backend": "wgpu",
                "api": api,
                "label": self.label(),
                "device": device,
                "device_name": name,
                "precision": arith.id(),
                "precision_setting": config.precision.id(),
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

    /// A block convolver for `filter` on this backend: a [`Convolver`] on CUDA, the wgpu
    /// polyphase core with one phase on wgpu, or the CPU
    /// [`crate::convolution::Convolver`]. A GPU failure while building (out of device
    /// memory, kernel compile error…) is logged and falls back to the CPU.
    pub fn convolver(&self, filter: Arc<SpectralFilter>) -> Box<dyn BlockConvolver + Send> {
        #[cfg(feature = "cuda")]
        if let Backend::Cuda { device, .. } = self {
            match Convolver::from_filter(Arc::clone(&filter), *device) {
                Ok(c) => return Box::new(c),
                Err(e) => log::warn!("GPU: {e}; this convolver runs on the CPU"),
            }
        }
        if let Backend::Wgpu { .. } = self {
            match self.polyphase(std::slice::from_ref(&*filter), 1) {
                Ok(mut cores) => {
                    return Box::new(PhaseConvolver::new(cores.pop().expect("one core"), filter))
                }
                Err(e) => log::warn!("GPU: {e}; this convolver runs on the CPU"),
            }
        }
        Box::new(CpuConvolver::from_filter(filter))
    }
}

/// CUDA probe (or `NotCompiled`).
#[allow(unused_variables)]
fn probe_cuda(device: usize) -> Result<Backend, GpuError> {
    #[cfg(feature = "cuda")]
    {
        cuda::probe(device)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(GpuError::NotCompiled)
    }
}

/// wgpu probe (or `NotCompiled`).
#[allow(unused_variables)]
fn probe_wgpu(device: usize, precision: GpuPrecision) -> Result<Backend, GpuError> {
    #[cfg(feature = "wgpu")]
    {
        wgpu_backend::probe(device, precision)
    }
    #[cfg(not(feature = "wgpu"))]
    {
        Err(GpuError::NotCompiled)
    }
}

/// A one-phase polyphase core used as a plain block convolver (room convolution on wgpu).
/// It keeps the last `partitions` input blocks, so a device failure mid-stream moves the
/// stream to a CPU convolver with its history replayed – no gap, no glitch, logged once.
pub struct PhaseConvolver {
    gpu: Option<Box<dyn PolyphaseBlockConvolver + Send>>,
    cpu: Option<CpuConvolver>,
    filter: Arc<SpectralFilter>,
    history: std::collections::VecDeque<Vec<f64>>,
    label: String,
}

impl PhaseConvolver {
    /// Wrap `core` (one phase, built for `filter`).
    pub fn new(core: Box<dyn PolyphaseBlockConvolver + Send>, filter: Arc<SpectralFilter>) -> Self {
        assert_eq!(core.phases(), 1, "a convolver has one phase");
        assert_eq!(
            core.block(),
            filter.block(),
            "core and filter disagree on the block"
        );
        Self {
            label: core.device_label(),
            gpu: Some(core),
            cpu: None,
            filter,
            history: std::collections::VecDeque::new(),
        }
    }

    /// Where it runs now: `Vulkan (RX 7800 XT)` or `CPU` after a device failure.
    pub fn device_label(&self) -> String {
        if self.gpu.is_some() {
            self.label.clone()
        } else {
            "CPU".into()
        }
    }
}

impl fmt::Debug for PhaseConvolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("gpu::PhaseConvolver")
            .field("device", &self.device_label())
            .field("block", &self.filter.block())
            .finish()
    }
}

impl BlockConvolver for PhaseConvolver {
    fn process_block(&mut self, input: &[f64], out: &mut [f64]) {
        if let Some(core) = &mut self.gpu {
            match core.process_block(input, out) {
                Ok(()) => {
                    self.history.push_back(input.to_vec());
                    if self.history.len() > self.filter.partitions().max(1) {
                        self.history.pop_front();
                    }
                    return;
                }
                Err(e) => {
                    log::error!(
                        "GPU: {e} during convolution; moving it to the CPU (history replayed)"
                    );
                    self.gpu = None;
                    let mut cpu = CpuConvolver::from_filter(Arc::clone(&self.filter));
                    let mut scratch = vec![0.0; input.len()];
                    for block in self.history.drain(..) {
                        cpu.process_block(&block, &mut scratch);
                    }
                    self.cpu = Some(cpu);
                }
            }
        }
        if let Some(cpu) = &mut self.cpu {
            cpu.process_block(input, out);
        }
    }

    fn latency(&self) -> usize {
        self.filter.block()
    }

    fn reset(&mut self) {
        self.history.clear();
        if let Some(core) = &mut self.gpu {
            core.reset();
        }
        if let Some(cpu) = &mut self.cpu {
            cpu.reset();
        }
    }

    fn on_gpu(&self) -> bool {
        self.gpu.is_some()
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
    for prefix in [
        "NVIDIA ", "GeForce ", "Tesla ", "Quadro ", "AMD ", "Radeon ",
    ] {
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

/// The GPU polyphase upsampler core when the crate is built without the `cuda` feature:
/// no values; [`PolyphaseConvolver::for_channels`] returns [`GpuError::NotCompiled`].
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub enum PolyphaseConvolver {}

#[cfg(not(feature = "cuda"))]
impl PolyphaseConvolver {
    /// Always `Err(GpuError::NotCompiled)` in this build.
    pub fn for_channels(
        _filters: &[SpectralFilter],
        _channels: usize,
        _device: usize,
    ) -> Result<Vec<Box<dyn PolyphaseBlockConvolver + Send>>, GpuError> {
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
            ..GpuConfig::default()
        };
        let before = FALLBACK_LINES.load(Ordering::SeqCst);
        let backend = Backend::detect(&config);
        assert!(!backend.is_gpu(), "no GPU here, got {backend:?}");
        assert_eq!(backend.label(), "CPU · 64-bit");
        let Backend::Cpu { reason } = &backend else {
            unreachable!()
        };
        if CUDA_COMPILED_IN {
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
            ..GpuConfig::default()
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
                    ..GpuConfig::default()
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
        if CUDA_COMPILED_IN {
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
        assert_eq!(
            s,
            json!({"gpu": "cuda", "gpu_device": 1, "gpu_precision": "auto"})
        );
        let p: GpuConfig =
            serde_json::from_str(r#"{"gpu":"wgpu","gpu_precision":"double-f32"}"#).unwrap();
        assert_eq!(
            (p.mode, p.precision),
            (GpuMode::Wgpu, GpuPrecision::DoubleF32)
        );
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
        assert_eq!("metal".parse::<GpuMode>().unwrap(), GpuMode::Wgpu);
        assert_eq!("Vulkan".parse::<GpuMode>().unwrap(), GpuMode::Wgpu);
        assert!("opencl".parse::<GpuMode>().is_err());
        assert_eq!("on".parse::<GpuMode>().unwrap(), GpuMode::On);
        for p in GpuPrecision::ALL {
            assert_eq!(p.id().parse::<GpuPrecision>().unwrap(), p);
        }
        assert!("f16".parse::<GpuPrecision>().is_err());
        assert!(GpuMode::available().contains(&GpuMode::Auto));
        assert_eq!(
            GpuMode::available().contains(&GpuMode::Wgpu),
            WGPU_COMPILED_IN
        );
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
        // What runs there, by engine.
        let both = cuda.stage_for(
            &GpuConfig::default(),
            GpuUse {
                upsampling: true,
                convolution: true,
            },
        );
        assert!(both.active);
        assert_eq!(
            both.detail,
            "CUDA (RTX 4060) · 64-bit · upsampling + room convolution"
        );
        assert_eq!(both.params["upsampling"], true);
        assert_eq!(both.params["convolution"], true);
        let up = cuda.stage_for(
            &GpuConfig::default(),
            GpuUse {
                upsampling: true,
                convolution: false,
            },
        );
        assert_eq!(up.detail, "CUDA (RTX 4060) · 64-bit · upsampling");
        assert_eq!(up.params["convolution"], false);
        // The CPU backend never claims either, whatever the chain asks for.
        let cpu = Backend::Cpu { reason: "x".into() };
        let st = cpu.stage_for(
            &GpuConfig::default(),
            GpuUse {
                upsampling: true,
                convolution: true,
            },
        );
        assert!(!st.active);
        assert_eq!(st.params["upsampling"], false);
    }

    #[test]
    fn cpu_backend_builds_no_polyphase_cores() {
        let filters = vec![SpectralFilter::new(&[1.0, 0.5], 64); 2];
        let off = Backend::Cpu {
            reason: GpuError::Disabled.to_string(),
        };
        assert_eq!(off.polyphase(&filters, 2).err(), Some(GpuError::Disabled));
        let none = Backend::Cpu {
            reason: GpuError::NotCompiled.to_string(),
        };
        assert_eq!(
            none.polyphase(&filters, 2).err(),
            Some(GpuError::NotCompiled)
        );
        let err = PolyphaseConvolver::for_channels(&filters, 1, 0)
            .err()
            .unwrap();
        if CUDA_COMPILED_IN {
            assert!(matches!(
                err,
                GpuError::NoDriver(_) | GpuError::NoDevice(_) | GpuError::Driver(_)
            ));
        } else {
            assert_eq!(err, GpuError::NotCompiled);
        }
    }

    #[test]
    fn wgpu_backend_labels_and_report() {
        let metal = Backend::Wgpu {
            device: 0,
            name: "Apple M2 Pro".into(),
            api: "Metal".into(),
            arith: Arith::DoubleF32,
        };
        assert!(metal.is_gpu());
        assert_eq!(metal.label(), "Metal (Apple M2 Pro) · double-f32 (≈48-bit)");
        let amd = Backend::Wgpu {
            device: 1,
            name: "AMD Radeon RX 7800 XT".into(),
            api: "Vulkan".into(),
            arith: Arith::F64,
        };
        assert_eq!(amd.label(), "Vulkan (RX 7800 XT) · 64-bit");
        let st = amd.stage_for(
            &GpuConfig::default(),
            GpuUse {
                upsampling: true,
                convolution: false,
            },
        );
        assert!(st.active);
        assert_eq!(st.detail, "Vulkan (RX 7800 XT) · 64-bit · upsampling");
        assert_eq!(st.params["backend"], "wgpu");
        assert_eq!(st.params["api"], "Vulkan");
        assert_eq!(st.params["precision"], "f64");
        assert_eq!(
            metal.report(&GpuConfig::default())["precision"],
            "double-f32"
        );
    }

    #[test]
    fn phase_convolver_moves_to_the_cpu_without_a_glitch() {
        use super::reference::ReferenceFactory;
        let h: Vec<f64> = (0..5 * 128 + 9)
            .map(|i| ((i as f64) * 0.05).sin() / (1.0 + i as f64 * 0.01))
            .collect();
        let filter = Arc::new(SpectralFilter::new(&h, 128));
        let x: Vec<f64> = (0..20 * 128).map(|i| ((i as f64) * 0.37).cos()).collect();
        let mut cpu = CpuConvolver::from_filter(Arc::clone(&filter));
        let factory = ReferenceFactory {
            fail_at_block: Some(9),
            fail_build: false,
        };
        let core = factory
            .polyphase(std::slice::from_ref(&*filter), 1)
            .unwrap()
            .pop()
            .unwrap();
        let mut gpu = PhaseConvolver::new(core, Arc::clone(&filter));
        assert_eq!(gpu.latency(), cpu.latency());
        let mut want = vec![0.0; 128];
        let mut got = vec![0.0; 128];
        for (b, chunk) in x.chunks(128).enumerate() {
            cpu.process_block(chunk, &mut want);
            gpu.process_block(chunk, &mut got);
            assert_eq!(gpu.on_gpu(), b < 9, "block {b}");
            for (p, q) in want.iter().zip(&got) {
                assert!((p - q).abs() < 1e-10, "block {b}: {p} vs {q}");
            }
        }
        assert_eq!(gpu.device_label(), "CPU");
    }
}
