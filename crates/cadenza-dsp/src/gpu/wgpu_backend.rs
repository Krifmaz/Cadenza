//! The `wgpu` implementation (compiled only with `--features wgpu`): the polyphase
//! upsampler core of [`super::wgsl`] on Vulkan, Metal or DX12, so AMD, Intel, Apple
//! Silicon and NVIDIA cards all work.
//!
//! One logical device per `(adapter, arithmetic)` is opened on first use and kept for the
//! life of the process (pipelines compiled once). Opening it runs a self-test: a small
//! polyphase core on the device against the CPU model of the CUDA algorithm in `f64`; a
//! device whose result is off by more than 1e-10 (a compiler that broke the double-f32
//! error terms, a driver bug) is refused and the stage stays on the CPU.
//!
//! Software adapters (lavapipe, SwiftShader, WARP) are skipped unless
//! `CADENZA_WGPU_SOFTWARE=1` is set: they are slower than the CPU engine and exist here
//! only so CI can run the kernels.

use super::reference::{check_polyphase, pair_spectra, ReferencePolyphase};
use super::wgsl::{
    block_program, dispatch_size, from_device_bytes, rational_source, source, to_device_bytes,
    twiddles, Arith, Params, RationalModel, RationalParams, ENTRY_POINTS, MAX_GROUPS_X,
};
use super::{Backend, GpuError, GpuPrecision};
use crate::convolution::{PolyphaseBlockConvolver, RationalDotEngine, SpectralFilter};
use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex, OnceLock};

/// Accuracy the self-test demands, relative to the peak (double-f32 reaches about 1e-13).
const SELF_TEST_TOLERANCE: f64 = 1e-10;

/// One usable adapter.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    /// Index for `gpu_device`.
    pub index: usize,
    /// Name the driver reports (`AMD Radeon RX 7800 XT`, `Apple M2`).
    pub name: String,
    /// `Vulkan`, `Metal` or `DX12`.
    pub api: &'static str,
    /// Native `f64` in shaders (`SHADER_F64`).
    pub f64: bool,
    /// A software rasteriser.
    pub software: bool,
}

fn instance() -> Option<&'static wgpu::Instance> {
    static INSTANCE: OnceLock<Option<wgpu::Instance>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| {
            if wgpu::Instance::enabled_backend_features().is_empty() {
                return None;
            }
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
            desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
            Some(wgpu::Instance::new(desc))
        })
        .as_ref()
}

fn api_name(b: wgpu::Backend) -> &'static str {
    match b {
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "DX12",
        _ => "GPU",
    }
}

fn allow_software() -> bool {
    std::env::var("CADENZA_WGPU_SOFTWARE").is_ok_and(|v| v == "1" || v == "true")
}

/// The usable adapters, in `gpu_device` order: one entry per physical GPU (Vulkan
/// preferred over DX12 when a card shows up on both), discrete cards first.
fn enumerate() -> Vec<(wgpu::Adapter, AdapterInfo)> {
    let Some(instance) = instance() else {
        return Vec::new();
    };
    let mut all: Vec<wgpu::Adapter> =
        pollster::block_on(instance.enumerate_adapters(
            wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        ));
    let rank_api = |b: wgpu::Backend| match b {
        wgpu::Backend::Vulkan | wgpu::Backend::Metal => 0,
        _ => 1,
    };
    let rank_type = |t: wgpu::DeviceType| match t {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::Cpu => 3,
    };
    all.sort_by_key(|a| {
        let i = a.get_info();
        (
            rank_type(i.device_type),
            rank_api(i.backend),
            i.name.clone(),
        )
    });
    let software = allow_software();
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for adapter in all {
        let info = adapter.get_info();
        let is_soft = info.device_type == wgpu::DeviceType::Cpu;
        if is_soft && !software {
            continue;
        }
        // Compute shaders with storage buffers are needed (not a downlevel GL device).
        if !adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            continue;
        }
        let key = (info.vendor, info.device, info.name.clone());
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let desc = AdapterInfo {
            index: out.len(),
            name: info.name.clone(),
            api: api_name(info.backend),
            f64: adapter.features().contains(wgpu::Features::SHADER_F64),
            software: is_soft,
        };
        out.push((adapter, desc));
    }
    out
}

/// The usable adapters on this machine (empty without any).
pub fn adapters() -> Vec<AdapterInfo> {
    enumerate().into_iter().map(|(_, d)| d).collect()
}

/// Resolve `precision` for an adapter: `Err` when `f64` is required but missing.
fn arith_for(info: &AdapterInfo, precision: GpuPrecision) -> Result<Arith, GpuError> {
    match precision {
        GpuPrecision::Auto => Ok(if info.f64 {
            Arith::F64
        } else {
            Arith::DoubleF32
        }),
        GpuPrecision::DoubleF32 => Ok(Arith::DoubleF32),
        GpuPrecision::F64 if info.f64 => Ok(Arith::F64),
        GpuPrecision::F64 => Err(GpuError::NoDevice(format!(
            "{} has no 64-bit shader support (gpu_precision = f64)",
            info.name
        ))),
    }
}

/// Probe adapter `device` and describe it (opening it, compiling the kernels and running
/// the self-test the first time).
pub fn probe(device: usize, precision: GpuPrecision) -> Result<Backend, GpuError> {
    let shared = open(device, precision)?;
    Ok(Backend::Wgpu {
        device,
        name: shared.name.clone(),
        api: shared.api.to_string(),
        arith: shared.arith,
    })
}

/// What every core on one device shares.
pub(crate) struct Shared {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    rational: wgpu::ComputePipeline,
    rational_layout: wgpu::BindGroupLayout,
    name: String,
    api: &'static str,
    arith: Arith,
    /// Stride of the per-dispatch parameters in the uniform buffer.
    param_stride: u64,
    max_binding: u64,
    max_buffer: u64,
    /// The first device error reported asynchronously (lost device, validation).
    error: Arc<Mutex<Option<String>>>,
}

impl Shared {
    fn take_error(&self) -> Option<String> {
        self.error.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Open (or reuse) the device for `(index, precision)`.
fn open(index: usize, precision: GpuPrecision) -> Result<Arc<Shared>, GpuError> {
    type Opened = HashMap<(usize, GpuPrecision), Result<Arc<Shared>, GpuError>>;
    static OPEN: Mutex<Option<Opened>> = Mutex::new(None);
    let mut map = OPEN.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(r) = map.get(&(index, precision)) {
        if let Ok(s) = r {
            if s.take_error().is_none() {
                return Ok(Arc::clone(s));
            }
        } else {
            return r.clone();
        }
    }
    let r = open_uncached(index, precision);
    map.insert((index, precision), r.clone());
    r
}

fn open_uncached(index: usize, precision: GpuPrecision) -> Result<Arc<Shared>, GpuError> {
    if instance().is_none() {
        return Err(GpuError::NoDriver(
            "no wgpu backend for this platform".into(),
        ));
    }
    let mut list = enumerate();
    if list.is_empty() {
        return Err(GpuError::NoDriver(
            "no Vulkan, Metal or DX12 GPU found".into(),
        ));
    }
    if index >= list.len() {
        return Err(GpuError::NoDevice(format!(
            "gpu_device = {index} but only {} GPU{} found",
            list.len(),
            if list.len() == 1 { " was" } else { "s were" }
        )));
    }
    let (adapter, info) = list.swap_remove(index);
    let arith = arith_for(&info, precision)?;
    let limits = adapter.limits();
    let features = if arith == Arith::F64 {
        wgpu::Features::SHADER_F64
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cadenza"),
        required_features: features,
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .map_err(|e| GpuError::Driver(format!("{}: {e}", info.name)))?;
    let error = Arc::new(Mutex::new(None::<String>));
    {
        let error = Arc::clone(&error);
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let mut slot = error.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(e.to_string());
            }
        }));
    }
    {
        let error = Arc::clone(&error);
        device.set_device_lost_callback(move |reason, msg| {
            let mut slot = error.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(format!("device lost ({reason:?}): {msg}"));
            }
        });
    }

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cadenza kernels"),
        source: wgpu::ShaderSource::Wgsl(source(arith).into()),
    });
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cadenza"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(Params::BYTES as u64),
                },
                count: None,
            },
            storage(1, true),
            storage(2, false),
            storage(3, false),
            storage(4, true),
            storage(5, true),
            storage(6, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cadenza"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let pipelines = ENTRY_POINTS
        .iter()
        .map(|e| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect();
    let rational_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cadenza rational"),
        source: wgpu::ShaderSource::Wgsl(rational_source(arith).into()),
    });
    let rational_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cadenza rational"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(RationalParams::BYTES as u64),
                },
                count: None,
            },
            storage(1, true),
            storage(2, true),
            storage(3, false),
        ],
    });
    let rational_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cadenza rational"),
        bind_group_layouts: &[Some(&rational_layout)],
        ..Default::default()
    });
    let rational = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rational_dot"),
        layout: Some(&rational_pipeline_layout),
        module: &rational_module,
        entry_point: Some("rational_dot"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(GpuError::Compile(format!("{}: {e}", info.name)));
    }
    let align = limits.min_uniform_buffer_offset_alignment.max(1) as u64;
    let shared = Arc::new(Shared {
        device,
        queue,
        pipelines,
        layout,
        rational,
        rational_layout,
        name: info.name.clone(),
        api: info.api,
        arith,
        param_stride: (Params::BYTES as u64).div_ceil(align) * align,
        max_binding: limits.max_storage_buffer_binding_size,
        max_buffer: limits.max_buffer_size,
        error,
    });
    self_test(&shared)?;
    rational_self_test(&shared)?;
    log::info!(
        "GPU: {} ({}) opened, {}, self-test passed",
        info.name,
        info.api,
        arith.label()
    );
    Ok(shared)
}

/// A small polyphase core on the device against the CPU model of the CUDA algorithm.
fn self_test(shared: &Arc<Shared>) -> Result<(), GpuError> {
    let (l, block) = (5, 256);
    let h: Vec<f64> = (0..6 * block * l + 7)
        .map(|i| ((i as f64) * 0.011).sin() / (1.0 + i as f64 * 0.003))
        .collect();
    let subs: Vec<SpectralFilter> = (0..l)
        .map(|p| {
            let s: Vec<f64> = h.iter().skip(p).step_by(l).copied().collect();
            SpectralFilter::new(&s, block)
        })
        .collect();
    let mut dev = PolyphaseCore::for_channels_on(Arc::clone(shared), &subs, 1)?;
    let mut model = ReferencePolyphase::new(&subs)?;
    let mut want = vec![0.0; block * l];
    let mut got = vec![0.0; block * l];
    let mut worst: f64 = 0.0;
    for b in 0..10 {
        let x: Vec<f64> = (0..block)
            .map(|i| ((b * block + i) as f64 * 0.37).cos() * 0.9)
            .collect();
        model.process_block(&x, &mut want)?;
        dev[0].process_block(&x, &mut got)?;
        let peak = want.iter().fold(1e-30f64, |m, v| m.max(v.abs()));
        for (p, q) in want.iter().zip(&got) {
            worst = worst.max((p - q).abs() / peak);
        }
    }
    if !worst.is_finite() || worst > SELF_TEST_TOLERANCE {
        return Err(GpuError::Driver(format!(
            "{} ({}) failed the {} self-test: error {worst:.1e} relative (limit {SELF_TEST_TOLERANCE:.0e})",
            shared.name,
            shared.api,
            shared.arith.id()
        )));
    }
    log::debug!(
        "GPU: {} self-test error {worst:.1e} ({})",
        shared.name,
        shared.arith.id()
    );
    Ok(())
}

/// The rational kernel on the device against its CPU model in `f64`.
fn rational_self_test(shared: &Arc<Shared>) -> Result<(), GpuError> {
    let (l, s) = (3, 600);
    let sub: Vec<Vec<f64>> = (0..l)
        .map(|p| {
            (0..s)
                .map(|k| (((k * l + p) as f64) * 0.013).sin() / (1.0 + k as f64 * 0.01))
                .collect()
        })
        .collect();
    let window: Vec<f64> = (0..3000).map(|i| (i as f64 * 0.37).cos()).collect();
    let mut dev = RationalEngine::for_channels_on(Arc::clone(shared), &sub, 1)?;
    let mut model = RationalModel::<f64>::new(&sub);
    let (mut want, mut got) = (vec![0.0; 900], vec![0.0; 900]);
    model.run(&window, s - 1, 1, 2, &mut want)?;
    dev[0].run(&window, s - 1, 1, 2, &mut got)?;
    let peak = want.iter().fold(1e-30f64, |m, v| m.max(v.abs()));
    let worst = want
        .iter()
        .zip(&got)
        .map(|(p, q)| (p - q).abs() / peak)
        .fold(0.0, f64::max);
    if !worst.is_finite() || worst > SELF_TEST_TOLERANCE {
        return Err(GpuError::Driver(format!(
            "{} ({}) failed the rational {} self-test: error {worst:.1e} relative",
            shared.name,
            shared.api,
            shared.arith.id()
        )));
    }
    Ok(())
}

/// Submit `enc` (which ends by copying the results into `staging`), wait, and read
/// `out.len()` values back.
fn submit_and_read(
    sh: &Shared,
    enc: wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    out: &mut [f64],
) -> Result<(), GpuError> {
    let bytes = (out.len() * 8) as u64;
    let index = sh.queue.submit([enc.finish()]);
    let (tx, rx) = mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, 0..bytes, move |r| {
        let _ = tx.send(r);
    });
    sh.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .map_err(|e| GpuError::Driver(format!("{}: {e}", sh.name)))?;
    rx.recv()
        .map_err(|_| GpuError::Driver("readback never completed".into()))?
        .map_err(|e| GpuError::Driver(format!("{}: {e}", sh.name)))?;
    {
        let view = staging
            .get_mapped_range(0..bytes)
            .map_err(|e| GpuError::Driver(format!("{}: {e}", sh.name)))?;
        from_device_bytes(&view, sh.arith, out);
    }
    staging.unmap();
    if let Some(e) = sh.take_error() {
        return Err(GpuError::Driver(e));
    }
    Ok(())
}

/// The rational resampler's time-domain polyphase bank on a wgpu device (one per channel;
/// the sub-filters are shared): per call, upload the window and the parameters, one
/// workgroup per output sample, read the outputs back.
pub struct RationalEngine {
    shared: Arc<Shared>,
    sub: Arc<wgpu::Buffer>,
    l: usize,
    s: usize,
    params: wgpu::Buffer,
    win: wgpu::Buffer,
    out: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl std::fmt::Debug for RationalEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("gpu::wgpu::RationalEngine")
            .field("device", &self.shared.name)
            .field("arith", &self.shared.arith)
            .field("phases", &self.l)
            .field("taps", &self.s)
            .finish()
    }
}

impl RationalEngine {
    /// Build `channels` engines for the reversed sub-filters `sub` on adapter `device`.
    pub fn for_channels(
        sub: &[Vec<f64>],
        channels: usize,
        device: usize,
        precision: GpuPrecision,
    ) -> Result<Vec<Box<dyn RationalDotEngine + Send>>, GpuError> {
        Self::for_channels_on(open(device, precision)?, sub, channels)
    }

    fn for_channels_on(
        shared: Arc<Shared>,
        sub: &[Vec<f64>],
        channels: usize,
    ) -> Result<Vec<Box<dyn RationalDotEngine + Send>>, GpuError> {
        let (l, s) = (sub.len(), sub.first().map_or(0, Vec::len));
        if l == 0 || s == 0 {
            return Err(GpuError::Driver("empty rational filter".into()));
        }
        let bytes = (l * s * 8) as u64;
        if bytes > shared.max_binding.min(shared.max_buffer) {
            return Err(GpuError::Driver(format!(
                "the rational filter needs a {} MB buffer; {} allows {} MB",
                bytes >> 20,
                shared.name,
                shared.max_binding.min(shared.max_buffer) >> 20
            )));
        }
        let dev = &shared.device;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let flat: Vec<f64> = sub.iter().flatten().copied().collect();
        let table = Arc::new(buffer(
            dev,
            "rational sub-filters",
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ));
        shared
            .queue
            .write_buffer(&table, 0, &to_device_bytes(&flat, shared.arith));
        let mut out: Vec<Box<dyn RationalDotEngine + Send>> = Vec::new();
        for _ in 0..channels.max(1) {
            out.push(Box::new(Self::with_capacity(
                Arc::clone(&shared),
                Arc::clone(&table),
                l,
                s,
                4 * s,
                1024,
            )));
        }
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(GpuError::Driver(format!("{}: {e}", shared.name)));
        }
        Ok(out)
    }

    fn with_capacity(
        shared: Arc<Shared>,
        sub: Arc<wgpu::Buffer>,
        l: usize,
        s: usize,
        win_cap: usize,
        out_cap: usize,
    ) -> Self {
        let dev = &shared.device;
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let params = buffer(
            dev,
            "rational params",
            RationalParams::BYTES as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let win = buffer(dev, "rational window", (win_cap * 8) as u64, rw);
        let out = buffer(
            dev,
            "rational out",
            (out_cap * 8) as u64,
            rw | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = buffer(
            dev,
            "rational readback",
            (out_cap * 8) as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cadenza rational"),
            layout: &shared.rational_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sub.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: win.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        Self {
            shared,
            sub,
            l,
            s,
            params,
            win,
            out,
            staging,
            bind,
        }
    }
}

impl RationalDotEngine for RationalEngine {
    fn phases(&self) -> usize {
        self.l
    }

    fn sub_len(&self) -> usize {
        self.s
    }

    fn run(
        &mut self,
        window: &[f64],
        first: usize,
        phase0: usize,
        m: usize,
        out: &mut [f64],
    ) -> Result<(), GpuError> {
        if out.is_empty() {
            return Ok(());
        }
        if let Some(e) = self.shared.take_error() {
            return Err(GpuError::Driver(e));
        }
        let win_cap = (self.win.size() / 8) as usize;
        let out_cap = (self.out.size() / 8) as usize;
        if window.len() > win_cap || out.len() > out_cap {
            *self = Self::with_capacity(
                Arc::clone(&self.shared),
                Arc::clone(&self.sub),
                self.l,
                self.s,
                window.len().next_power_of_two().max(win_cap),
                out.len().next_power_of_two().max(out_cap),
            );
        }
        let sh = Arc::clone(&self.shared);
        let p = RationalParams {
            s: self.s as u32,
            l: self.l as u32,
            m: m as u32,
            phase0: phase0 as u32,
            first: first as u32,
            count: out.len() as u32,
            one: 1.0,
        };
        sh.queue.write_buffer(&self.params, 0, &p.to_bytes());
        sh.queue
            .write_buffer(&self.win, 0, &to_device_bytes(window, sh.arith));
        let mut enc = sh
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&sh.rational);
            pass.set_bind_group(0, &self.bind, &[]);
            let groups = out.len() as u32;
            let x = groups.min(MAX_GROUPS_X);
            pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
        }
        enc.copy_buffer_to_buffer(&self.out, 0, &self.staging, 0, (out.len() * 8) as u64);
        submit_and_read(&sh, enc, &self.staging, out)
    }

    fn device_label(&self) -> String {
        let sh = &self.shared;
        format!("{} ({})", sh.api, super::short_device_name(&sh.name))
    }

    fn on_gpu(&self) -> bool {
        true
    }
}

/// What the channels of one stage share: the packed sub-filter spectra and the twiddles.
struct Stage {
    shared: Arc<Shared>,
    g: wgpu::Buffer,
    tw: wgpu::Buffer,
    block: usize,
    l: usize,
    parts: usize,
}

/// The upsampler's FFT polyphase core on a wgpu device (one per channel; the spectra are
/// shared). Per input block: upload the frame and the dispatch parameters, record the
/// program of [`block_program`] in one compute pass, copy the interleaved output to a
/// mapped buffer and read it back.
pub struct PolyphaseCore {
    stage: Arc<Stage>,
    head: usize,
    count: usize,
    frame: Vec<f64>,
    io: wgpu::Buffer,
    params: wgpu::Buffer,
    staging: wgpu::Buffer,
    bg_ab: wgpu::BindGroup,
    bg_ba: wgpu::BindGroup,
    // Held so the buffers the bind groups reference stay alive (and for clarity).
    _a: wgpu::Buffer,
    _b: wgpu::Buffer,
    _fdl: wgpu::Buffer,
}

impl std::fmt::Debug for PolyphaseCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("gpu::wgpu::PolyphaseCore")
            .field("device", &self.stage.shared.name)
            .field("arith", &self.stage.shared.arith)
            .field("block", &self.stage.block)
            .field("phases", &self.stage.l)
            .finish()
    }
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT),
        usage,
        mapped_at_creation: false,
    })
}

impl PolyphaseCore {
    /// Build `channels` cores for the `L = filters.len()` sub-filters on adapter `device`.
    pub fn for_channels(
        filters: &[SpectralFilter],
        channels: usize,
        device: usize,
        precision: GpuPrecision,
    ) -> Result<Vec<Box<dyn PolyphaseBlockConvolver + Send>>, GpuError> {
        Self::for_channels_on(open(device, precision)?, filters, channels)
    }

    fn for_channels_on(
        shared: Arc<Shared>,
        filters: &[SpectralFilter],
        channels: usize,
    ) -> Result<Vec<Box<dyn PolyphaseBlockConvolver + Send>>, GpuError> {
        let (block, parts) = check_polyphase(filters)?;
        let l = filters.len();
        let n = 2 * block;
        let pairs = l.div_ceil(2);
        let g_bytes = (pairs * parts * 2 * n * 8) as u64;
        let work_bytes = (pairs * 2 * n * 8) as u64;
        let biggest = g_bytes.max(work_bytes).max((parts * 2 * n * 8) as u64);
        if biggest > shared.max_binding.min(shared.max_buffer) {
            return Err(GpuError::Driver(format!(
                "this filter needs a {} MB buffer; {} allows {} MB",
                biggest >> 20,
                shared.name,
                shared.max_binding.min(shared.max_buffer) >> 20
            )));
        }
        let dev = &shared.device;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let s = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let g = buffer(dev, "spectra", g_bytes, s);
        shared.queue.write_buffer(
            &g,
            0,
            &to_device_bytes(&pair_spectra(filters), shared.arith),
        );
        let tw = buffer(dev, "twiddles", (2 * n * 8) as u64, s);
        shared
            .queue
            .write_buffer(&tw, 0, &to_device_bytes(&twiddles(n), shared.arith));
        let stage = Arc::new(Stage {
            shared: Arc::clone(&shared),
            g,
            tw,
            block,
            l,
            parts,
        });
        let prog_len = block_program(block, l, parts, 0, 1).len() as u64;
        let mut cores: Vec<Box<dyn PolyphaseBlockConvolver + Send>> = Vec::new();
        for _ in 0..channels.max(1) {
            let rw = s | wgpu::BufferUsages::COPY_SRC;
            let a = buffer(dev, "a", work_bytes, rw);
            let b = buffer(dev, "b", work_bytes, rw);
            let fdl = buffer(dev, "fdl", (parts * 2 * n * 8) as u64, rw);
            let io = buffer(dev, "io", (n.max(block * l) * 8) as u64, rw);
            let params = buffer(
                dev,
                "params",
                prog_len * shared.param_stride,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let staging = buffer(
                dev,
                "readback",
                (block * l * 8) as u64,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            let bind = |x: &wgpu::Buffer, y: &wgpu::Buffer| {
                dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cadenza"),
                    layout: &shared.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &params,
                                offset: 0,
                                size: wgpu::BufferSize::new(Params::BYTES as u64),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: x.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: y.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: fdl.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: stage.g.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: stage.tw.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: io.as_entire_binding(),
                        },
                    ],
                })
            };
            let bg_ab = bind(&a, &b);
            let bg_ba = bind(&b, &a);
            cores.push(Box::new(PolyphaseCore {
                stage: Arc::clone(&stage),
                head: 0,
                count: 0,
                frame: vec![0.0; n],
                io,
                params,
                staging,
                bg_ab,
                bg_ba,
                _a: a,
                _b: b,
                _fdl: fdl,
            }));
        }
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(GpuError::Driver(format!("{}: {e}", shared.name)));
        }
        if let Some(e) = shared.take_error() {
            return Err(GpuError::Driver(e));
        }
        Ok(cores)
    }

    fn run_block(&mut self, input: &[f64], out: &mut [f64]) -> Result<(), GpuError> {
        let st = Arc::clone(&self.stage);
        let sh = &st.shared;
        if let Some(e) = sh.take_error() {
            return Err(GpuError::Driver(e));
        }
        let (block, l) = (st.block, st.l);
        self.frame[block..].copy_from_slice(input);
        let head = (self.head + st.parts - 1) % st.parts;
        let count = (self.count + 1).min(st.parts);
        let prog = block_program(block, l, st.parts, head, count);
        let stride = sh.param_stride as usize;
        let mut pbytes = vec![0u8; prog.len() * stride];
        for (i, d) in prog.iter().enumerate() {
            pbytes[i * stride..i * stride + Params::BYTES].copy_from_slice(&d.params.to_bytes());
        }
        sh.queue.write_buffer(&self.params, 0, &pbytes);
        sh.queue
            .write_buffer(&self.io, 0, &to_device_bytes(&self.frame, sh.arith));
        let mut enc = sh
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            for (i, d) in prog.iter().enumerate() {
                pass.set_pipeline(&sh.pipelines[d.entry]);
                let bg = if d.x_is_a { &self.bg_ab } else { &self.bg_ba };
                pass.set_bind_group(0, bg, &[(i * stride) as u32]);
                let (gx, gy) = dispatch_size(d.params.items as usize);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        let out_bytes = (block * l * 8) as u64;
        enc.copy_buffer_to_buffer(&self.io, 0, &self.staging, 0, out_bytes);
        submit_and_read(sh, enc, &self.staging, out)?;
        self.head = head;
        self.count = count;
        self.frame.copy_within(block.., 0);
        Ok(())
    }
}

impl PolyphaseBlockConvolver for PolyphaseCore {
    fn block(&self) -> usize {
        self.stage.block
    }

    fn phases(&self) -> usize {
        self.stage.l
    }

    fn process_block(&mut self, input: &[f64], out: &mut [f64]) -> Result<(), GpuError> {
        assert_eq!(input.len(), self.stage.block, "one block in");
        assert_eq!(out.len(), self.stage.block * self.stage.l, "block × L out");
        self.run_block(input, out)
    }

    fn reset(&mut self) {
        // The delay line is cleared lazily through `count`, as on CUDA.
        self.frame.fill(0.0);
        self.head = 0;
        self.count = 0;
    }

    fn device_label(&self) -> String {
        let sh = &self.stage.shared;
        format!("{} ({})", sh.api, super::short_device_name(&sh.name))
    }

    fn on_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse, validate and translate both sources the way the backends will: SPIR-V
    /// (Vulkan), MSL (Metal, double-f32 only: Metal has no f64) and HLSL (DX12).
    #[test]
    fn wgsl_validates_and_translates_for_every_backend() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        for (arith, src) in [
            (Arith::F64, source(Arith::F64)),
            (Arith::DoubleF32, source(Arith::DoubleF32)),
            (Arith::F64, rational_source(Arith::F64)),
            (Arith::DoubleF32, rational_source(Arith::DoubleF32)),
        ] {
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{arith:?}: {}", e.emit_to_string(&src)));
            let caps = match arith {
                Arith::F64 => Capabilities::FLOAT64,
                Arith::DoubleF32 => Capabilities::empty(),
            };
            let info = Validator::new(ValidationFlags::all(), caps)
                .validate(&module)
                .unwrap_or_else(|e| panic!("{arith:?}: {}", e.emit_to_string(&src)));
            let entries: &[&str] = if src.contains("fn rational_dot(") {
                &["rational_dot"]
            } else {
                &ENTRY_POINTS
            };
            for e in entries {
                assert!(module.entry_points.iter().any(|ep| ep.name == *e), "{e}");
            }
            let spv = naga::back::spv::write_vec(
                &module,
                &info,
                &naga::back::spv::Options::default(),
                None,
            )
            .unwrap_or_else(|e| panic!("{arith:?} SPIR-V: {e}"));
            assert!(spv.len() > 100);
            let mut hlsl = String::new();
            let opts = naga::back::hlsl::Options::default();
            let pipeline = naga::back::hlsl::PipelineOptions::default();
            naga::back::hlsl::Writer::new(&mut hlsl, &opts, &pipeline)
                .write(&module, &info, None)
                .unwrap_or_else(|e| panic!("{arith:?} HLSL: {e}"));
            assert!(hlsl.contains(entries[entries.len() - 1]));
            if arith == Arith::DoubleF32 {
                let (msl, _) = naga::back::msl::write_string(
                    &module,
                    &info,
                    &naga::back::msl::Options::default(),
                    &naga::back::msl::PipelineOptions::default(),
                )
                .unwrap_or_else(|e| panic!("MSL: {e}"));
                assert!(msl.contains("4097.0"), "Dekker split present");
            }
        }
    }

    #[test]
    fn probe_without_a_gpu_is_an_error_not_a_panic() {
        match probe(0, GpuPrecision::Auto) {
            Ok(Backend::Wgpu { name, .. }) => assert!(!name.is_empty()),
            Ok(other) => panic!("probe returned {other:?}"),
            Err(GpuError::NoDriver(_) | GpuError::NoDevice(_) | GpuError::Driver(_)) => {}
            Err(e) => panic!("unexpected error {e}"),
        }
        assert!(probe(usize::MAX, GpuPrecision::Auto).is_err());
    }

    /// Runs on any machine with a GPU (or with `CADENZA_WGPU_SOFTWARE=1` and lavapipe):
    /// the device core against its CPU model and the resampler against the CPU engine.
    #[test]
    #[ignore = "needs a Vulkan / Metal / DX12 device"]
    fn device_polyphase_matches_the_model_and_the_cpu_resampler() {
        use crate::cache::CoefficientCache;
        use crate::fir::FilterKind;
        use crate::resampler::Resampler;
        let list = adapters();
        assert!(!list.is_empty(), "no adapter");
        for info in &list {
            for precision in [GpuPrecision::DoubleF32, GpuPrecision::Auto] {
                let backend = probe(info.index, precision).expect("device opens");
                let cache = CoefficientCache::disabled();
                let kind = FilterKind::parse("sinc-l-64k").unwrap();
                for (inr, outr) in [(44_100, 88_200), (44_100, 705_600), (16_000, 48_000)] {
                    let mut cpu = Resampler::new(inr, outr, 2, kind, 150.0, &cache).unwrap();
                    let mut gpu =
                        Resampler::new_on(inr, outr, 2, kind, 150.0, &cache, Some(&backend))
                            .unwrap();
                    assert!(gpu.on_gpu(), "{}", gpu.describe());
                    let x: Vec<f64> = (0..2 * 20_000)
                        .map(|i| ((i as f64) * 0.0137).sin() * 0.5)
                        .collect();
                    let mut a = cpu.push(&x);
                    a.extend(cpu.flush());
                    let mut b = gpu.push(&x);
                    b.extend(gpu.flush());
                    assert_eq!(a.len(), b.len());
                    let peak = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    let err = a
                        .iter()
                        .zip(&b)
                        .map(|(p, q)| (p - q).abs())
                        .fold(0.0, f64::max);
                    let tol = if backend.label().contains("64-bit") {
                        1e-12
                    } else {
                        1e-10
                    };
                    assert!(
                        err / peak < tol,
                        "{} {inr}→{outr}: {err:e}",
                        backend.label()
                    );
                    eprintln!(
                        "{} {inr}→{outr}: {:.1e} relative",
                        backend.label(),
                        err / peak
                    );
                }
            }
        }
    }

    /// Room convolution on the device (the one-phase core behind [`super::PhaseConvolver`])
    /// against the CPU convolver.
    #[test]
    #[ignore = "needs a Vulkan / Metal / DX12 device"]
    fn device_convolver_matches_the_cpu_convolver() {
        use crate::convolution::{BlockConvolver, Convolver as CpuConvolver};
        let backend = probe(0, GpuPrecision::Auto).expect("device opens");
        let h: Vec<f64> = (0..9 * 512 + 5)
            .map(|i| ((i as f64) * 0.013).sin() / (1.0 + i as f64 * 0.01))
            .collect();
        let filter = Arc::new(SpectralFilter::new(&h, 512));
        let mut cpu = CpuConvolver::from_filter(Arc::clone(&filter));
        let mut gpu = backend.convolver(Arc::clone(&filter));
        assert!(gpu.on_gpu());
        assert_eq!(gpu.latency(), cpu.latency());
        let mut want = vec![0.0; 512];
        let mut got = vec![0.0; 512];
        for b in 0..16 {
            let x: Vec<f64> = (0..512)
                .map(|i| ((b * 512 + i) as f64 * 0.37).cos())
                .collect();
            cpu.process_block(&x, &mut want);
            gpu.process_block(&x, &mut got);
            for (p, q) in want.iter().zip(&got) {
                assert!((p - q).abs() < 1e-10, "{p} vs {q}");
            }
        }
    }

    /// The rational (44.1 ↔ 48 kHz) stage on the device against the CPU resampler, in both
    /// arithmetics, alone and in the 44.1 kHz → 1.536 MHz cascade.
    #[test]
    #[ignore = "needs a Vulkan / Metal / DX12 device"]
    fn device_rational_stage_matches_the_cpu_resampler() {
        use crate::cache::CoefficientCache;
        use crate::fir::FilterKind;
        use crate::resampler::Resampler;
        let cache = CoefficientCache::disabled();
        let kind = FilterKind::parse("sinc-l-64k").unwrap();
        for precision in [GpuPrecision::DoubleF32, GpuPrecision::Auto] {
            let backend = probe(0, precision).expect("device opens");
            for (inr, outr) in [(44_100, 48_000), (48_000, 44_100), (44_100, 1_536_000)] {
                let mut cpu = Resampler::new(inr, outr, 2, kind, 150.0, &cache).unwrap();
                let mut gpu =
                    Resampler::new_on(inr, outr, 2, kind, 150.0, &cache, Some(&backend)).unwrap();
                assert!(
                    gpu.stages().iter().all(|s| s.device.is_some()),
                    "{}",
                    gpu.describe()
                );
                let x: Vec<f64> = (0..2 * 12_000)
                    .map(|i| ((i as f64) * 0.0137).sin() * 0.5)
                    .collect();
                let (mut a, mut b) = (Vec::new(), Vec::new());
                for chunk in x.chunks(2 * 1_000) {
                    a.extend(cpu.push(chunk));
                    b.extend(gpu.push(chunk));
                }
                a.extend(cpu.flush());
                b.extend(gpu.flush());
                assert_eq!(a.len(), b.len());
                let peak = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let err = a
                    .iter()
                    .zip(&b)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0.0, f64::max);
                let tol = if backend.label().contains("64-bit") {
                    1e-12
                } else {
                    1e-10
                };
                eprintln!(
                    "{} {inr}→{outr}: {:.1e} relative",
                    backend.label(),
                    err / peak
                );
                assert!(
                    err / peak < tol,
                    "{} {inr}→{outr}: {err:e}",
                    backend.label()
                );
            }
        }
    }
}
