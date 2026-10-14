//! The CUDA implementation (compiled only with `--features cuda`).
//!
//! `cudarc` is used with `dynamic-loading`: nothing links against CUDA at build time, and
//! the driver library is looked up when first needed. `cudarc` *panics* when it cannot find
//! the library, so [`probe`] checks for `libcuda` / `nvcuda.dll` itself (with `libloading`)
//! before any `cudarc` call, and turns the absence into [`GpuError::NoDriver`].

use super::kernels::{grid_for, pass_schedule, KERNEL_SOURCE, PROGRAM_NAME, THREADS_PER_BLOCK};
use super::reference::{check_polyphase, filter_spectra, pair_spectra};
use super::{Backend, GpuError, MIN_DRIVER_VERSION};
use crate::convolution::{
    BlockConvolver, PolyphaseBlockConvolver, RationalDotEngine, SpectralFilter,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions};
use std::sync::Arc;

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError::Driver(e.to_string())
    }
}

impl From<CompileError> for GpuError {
    fn from(e: CompileError) -> Self {
        GpuError::Compile(match e {
            CompileError::CompileError { log, .. } => log.to_string_lossy().into_owned(),
            other => other.to_string(),
        })
    }
}

/// Library names the NVIDIA driver installs on each OS.
fn driver_library_names() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["nvcuda.dll"]
    } else if cfg!(target_os = "macos") {
        &[]
    } else {
        &["libcuda.so.1", "libcuda.so"]
    }
}

/// Load (and immediately drop) the driver library to find out whether it is installed.
fn driver_present() -> Result<(), GpuError> {
    let names = driver_library_names();
    if names.is_empty() {
        return Err(GpuError::NoDriver(
            "CUDA is not available on this operating system".into(),
        ));
    }
    let mut last = String::new();
    for name in names {
        // SAFETY: loading the NVIDIA driver library runs its initialisers, which is exactly
        // what `cudarc` does next; we hold no symbols from it.
        match unsafe { libloading::Library::new(*name) } {
            Ok(_lib) => return Ok(()),
            Err(e) => last = format!("{name}: {e}"),
        }
    }
    Err(GpuError::NoDriver(format!(
        "{last}; CUDA needs NVIDIA driver {MIN_DRIVER_VERSION} or newer"
    )))
}

/// Probe device `device` and describe it, without compiling anything.
pub fn probe(device: usize) -> Result<Backend, GpuError> {
    driver_present()?;
    let count = CudaContext::device_count()?;
    if count <= 0 {
        return Err(GpuError::NoDevice(
            "the NVIDIA driver reports no CUDA device".into(),
        ));
    }
    if device >= count as usize {
        return Err(GpuError::NoDevice(format!(
            "gpu_device = {device} but only {count} CUDA device{} present",
            if count == 1 { " is" } else { "s are" }
        )));
    }
    let ctx = CudaContext::new(device)?;
    let name = ctx.name()?;
    let compute = ctx.compute_capability()?;
    Ok(Backend::Cuda {
        device,
        name,
        compute,
    })
}

/// Compile the kernels with NVRTC and load them into `ctx`.
fn compile(ctx: &Arc<CudaContext>) -> Result<Arc<CudaModule>, GpuError> {
    let opts = CompileOptions {
        name: Some(PROGRAM_NAME.to_string()),
        // Keep every operation IEEE double: no fast-math, exact division and sqrt.
        use_fast_math: Some(false),
        prec_div: Some(true),
        prec_sqrt: Some(true),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts)?;
    Ok(ctx.load_module(ptx)?)
}

/// One-dimensional launch over `items` work items.
fn cfg(items: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid_for(items), 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// A partitioned convolver whose FFTs, multiply-accumulate and frequency-delay line live
/// on a CUDA device, in `f64`. Implements [`BlockConvolver`] like the CPU engine.
pub struct Convolver {
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    k_pack: CudaFunction,
    k_radix2: CudaFunction,
    k_radix4: CudaFunction,
    k_copy: CudaFunction,
    k_mac: CudaFunction,
    k_unpack: CudaFunction,
    block: usize,
    /// Transform size (`2 · block`).
    n: usize,
    /// Partitions = FDL slots.
    depth: usize,
    head: usize,
    count: usize,
    /// `[previous block | current block]` on the host.
    frame: Vec<f64>,
    d_in: CudaSlice<f64>,
    d_a: CudaSlice<f64>,
    d_b: CudaSlice<f64>,
    d_fdl: CudaSlice<f64>,
    d_h: CudaSlice<f64>,
    d_out: CudaSlice<f64>,
    device_name: String,
}

impl std::fmt::Debug for Convolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("gpu::Convolver")
            .field("device", &self.device_name)
            .field("block", &self.block)
            .field("partitions", &self.depth)
            .finish()
    }
}

impl Convolver {
    /// Build for an already partitioned filter on CUDA device `device`: compiles the
    /// kernels with NVRTC, uploads the partition spectra (`partitions × 2·block` complex
    /// `f64`, i.e. `partitions × block × 32` bytes) and allocates an FDL of the same size.
    pub fn from_filter(filter: Arc<SpectralFilter>, device: usize) -> Result<Self, GpuError> {
        let Backend::Cuda { name, .. } = probe(device)? else {
            unreachable!("probe returns a CUDA backend or an error")
        };
        let ctx = CudaContext::new(device)?;
        let stream = ctx.default_stream();
        let module = compile(&ctx)?;
        let load = |n: &str| module.load_function(n);

        let block = filter.block();
        let n = 2 * block;
        let depth = filter.partitions().max(1);
        let spectra = filter_spectra(&filter);
        let d_h = stream.clone_htod(&spectra)?;
        let d_fdl = stream.alloc_zeros::<f64>(depth * 2 * n)?;
        Ok(Self {
            k_pack: load("pack_real")?,
            k_radix2: load("stockham_radix2")?,
            k_radix4: load("stockham_radix4")?,
            k_copy: load("copy_complex")?,
            k_mac: load("fdl_mac")?,
            k_unpack: load("unpack_real_scaled")?,
            d_in: stream.alloc_zeros::<f64>(n)?,
            d_a: stream.alloc_zeros::<f64>(2 * n)?,
            d_b: stream.alloc_zeros::<f64>(2 * n)?,
            d_out: stream.alloc_zeros::<f64>(block)?,
            d_fdl,
            d_h,
            stream,
            _module: module,
            block,
            n,
            depth,
            head: 0,
            count: 0,
            frame: vec![0.0; n],
            device_name: name,
        })
    }

    /// Partition `h` for `block` and build on device `device`.
    pub fn new(h: &[f64], block: usize, device: usize) -> Result<Self, GpuError> {
        Self::from_filter(Arc::new(SpectralFilter::new(h, block)), device)
    }

    /// Block size (also the latency).
    pub fn block(&self) -> usize {
        self.block
    }

    /// Partitions held on the device.
    pub fn partitions(&self) -> usize {
        self.depth
    }

    /// The device name as the driver reports it.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Bytes of device memory this convolver holds.
    pub fn device_bytes(&self) -> usize {
        8 * (self.n + 2 * 2 * self.n + 2 * self.depth * 2 * self.n + self.block)
    }

    fn cfg(items: usize) -> LaunchConfig {
        cfg(items)
    }

    /// In-place FFT of `d_a` (ping-ponging through `d_b`); the result ends in `d_a`.
    fn fft(&mut self, dir: i32) -> Result<(), DriverError> {
        let n = self.n as i32;
        let mut in_a = true;
        for (radix, ns) in pass_schedule(self.n) {
            let ns_i = ns as i32;
            let (src, dst) = if in_a {
                (&self.d_a, &mut self.d_b)
            } else {
                (&self.d_b, &mut self.d_a)
            };
            let (func, items) = if radix == 4 {
                (&self.k_radix4, self.n / 4)
            } else {
                (&self.k_radix2, self.n / 2)
            };
            // SAFETY: argument types and order match the kernel signatures in kernels.cu;
            // every index the kernels compute is bounded by `n`, the buffers hold `2n`.
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(src)
                    .arg(dst)
                    .arg(&n)
                    .arg(&ns_i)
                    .arg(&dir)
                    .launch(Self::cfg(items))?;
            }
            in_a = !in_a;
        }
        if !in_a {
            std::mem::swap(&mut self.d_a, &mut self.d_b);
        }
        Ok(())
    }

    fn run_block(&mut self, input: &[f64], out: &mut [f64]) -> Result<(), DriverError> {
        let (block, n) = (self.block, self.n);
        self.frame[block..].copy_from_slice(input);
        self.stream.memcpy_htod(&self.frame, &mut self.d_in)?;
        let n_i = n as i32;
        // SAFETY: see `fft`.
        unsafe {
            self.stream
                .launch_builder(&self.k_pack)
                .arg(&self.d_in)
                .arg(&mut self.d_a)
                .arg(&n_i)
                .launch(Self::cfg(n))?;
        }
        self.fft(-1)?;
        self.head = (self.head + self.depth - 1) % self.depth;
        self.count = (self.count + 1).min(self.depth);
        let slot = self.head as i32;
        let depth = self.count as i32;
        let slots = self.depth as i32;
        let scale = 1.0 / n as f64;
        unsafe {
            self.stream
                .launch_builder(&self.k_copy)
                .arg(&self.d_a)
                .arg(&mut self.d_fdl)
                .arg(&n_i)
                .arg(&slot)
                .launch(Self::cfg(n))?;
            self.stream
                .launch_builder(&self.k_mac)
                .arg(&self.d_fdl)
                .arg(&self.d_h)
                .arg(&mut self.d_a)
                .arg(&n_i)
                .arg(&depth)
                .arg(&slot)
                .arg(&slots)
                .launch(Self::cfg(n))?;
        }
        self.fft(1)?;
        let block_i = block as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k_unpack)
                .arg(&self.d_a)
                .arg(&mut self.d_out)
                .arg(&block_i)
                .arg(&scale)
                .launch(Self::cfg(block))?;
        }
        self.stream.memcpy_dtoh(&self.d_out, out)?;
        self.stream.synchronize()?;
        self.frame.copy_within(block.., 0);
        Ok(())
    }

    /// Forget all history (the FDL is cleared lazily through `count`).
    pub fn reset(&mut self) {
        self.frame.fill(0.0);
        self.count = 0;
    }
}

impl BlockConvolver for Convolver {
    fn process_block(&mut self, input: &[f64], out: &mut [f64]) {
        assert_eq!(
            input.len(),
            self.block,
            "process_block wants exactly one block"
        );
        assert_eq!(
            out.len(),
            self.block,
            "process_block wants exactly one block"
        );
        if let Err(e) = self.run_block(input, out) {
            // A device error mid-stream cannot be recovered here; the player sees silence
            // for this block and the log says why.
            log::error!("GPU: CUDA error during convolution: {e}");
            out.fill(0.0);
        }
    }

    fn latency(&self) -> usize {
        self.block
    }

    fn reset(&mut self) {
        Convolver::reset(self);
    }

    fn on_gpu(&self) -> bool {
        true
    }
}

/// What the channels of one polyphase stage share on the device: the compiled kernels and
/// the packed sub-filter spectra (uploaded once).
struct PolyShared {
    _module: Arc<CudaModule>,
    k_pack: CudaFunction,
    k_radix2: CudaFunction,
    k_radix4: CudaFunction,
    k_radix2_batch: CudaFunction,
    k_radix4_batch: CudaFunction,
    k_copy: CudaFunction,
    k_mac: CudaFunction,
    k_interleave: CudaFunction,
    /// `pairs × parts × 2n` doubles: see [`pair_spectra`].
    d_g: CudaSlice<f64>,
    block: usize,
    n: usize,
    l: usize,
    pairs: usize,
    parts: usize,
    device_name: String,
}

/// The upsampler's FFT polyphase core on a CUDA device, in `f64` (one per channel, each on
/// its own stream; the sub-filter spectra are shared). Per input block: upload the block,
/// one forward transform into the device-resident delay line, one `poly_fdl_mac` launch
/// over all sub-filter pairs, one batched inverse transform, `poly_interleave`, and the
/// download of the `block × L` interleaved output. Mirrors
/// [`super::reference::ReferencePolyphase`] kernel for kernel.
pub struct PolyphaseConvolver {
    shared: Arc<PolyShared>,
    stream: Arc<CudaStream>,
    head: usize,
    count: usize,
    frame: Vec<f64>,
    d_in: CudaSlice<f64>,
    d_a: CudaSlice<f64>,
    d_b: CudaSlice<f64>,
    d_fdl: CudaSlice<f64>,
    d_acc: CudaSlice<f64>,
    d_acc2: CudaSlice<f64>,
    d_out: CudaSlice<f64>,
}

impl std::fmt::Debug for PolyphaseConvolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("gpu::PolyphaseConvolver")
            .field("device", &self.shared.device_name)
            .field("block", &self.shared.block)
            .field("phases", &self.shared.l)
            .field("partitions", &self.shared.parts)
            .finish()
    }
}

impl PolyphaseConvolver {
    /// Build `channels` cores for the `L = filters.len()` sub-filters on CUDA device
    /// `device`: compile the kernels, upload the packed spectra once
    /// (`ceil(L/2) × partitions × 2·block` complex `f64`), and give each channel its own
    /// stream, delay line and work buffers.
    pub fn for_channels(
        filters: &[SpectralFilter],
        channels: usize,
        device: usize,
    ) -> Result<Vec<Box<dyn PolyphaseBlockConvolver + Send>>, GpuError> {
        let (block, parts) = check_polyphase(filters)?;
        let Backend::Cuda { name, .. } = probe(device)? else {
            unreachable!("probe returns a CUDA backend or an error")
        };
        let ctx = CudaContext::new(device)?;
        // Streams first, so every buffer below is created with cross-stream event tracking.
        let streams = (0..channels.max(1))
            .map(|_| ctx.new_stream())
            .collect::<Result<Vec<_>, _>>()?;
        let module = compile(&ctx)?;
        let load = |n: &str| module.load_function(n);
        let l = filters.len();
        let n = 2 * block;
        let pairs = l.div_ceil(2);
        let g = pair_spectra(filters);
        let d_g = streams[0].clone_htod(&g)?;
        streams[0].synchronize()?;
        let shared = Arc::new(PolyShared {
            k_pack: load("pack_real")?,
            k_radix2: load("stockham_radix2")?,
            k_radix4: load("stockham_radix4")?,
            k_radix2_batch: load("stockham_radix2_batch")?,
            k_radix4_batch: load("stockham_radix4_batch")?,
            k_copy: load("copy_complex")?,
            k_mac: load("poly_fdl_mac")?,
            k_interleave: load("poly_interleave")?,
            _module: module,
            d_g,
            block,
            n,
            l,
            pairs,
            parts,
            device_name: name,
        });
        streams
            .into_iter()
            .map(|stream| {
                let core = PolyphaseConvolver {
                    d_in: stream.alloc_zeros::<f64>(n)?,
                    d_a: stream.alloc_zeros::<f64>(2 * n)?,
                    d_b: stream.alloc_zeros::<f64>(2 * n)?,
                    d_fdl: stream.alloc_zeros::<f64>(parts * 2 * n)?,
                    d_acc: stream.alloc_zeros::<f64>(pairs * 2 * n)?,
                    d_acc2: stream.alloc_zeros::<f64>(pairs * 2 * n)?,
                    d_out: stream.alloc_zeros::<f64>(block * l)?,
                    shared: Arc::clone(&shared),
                    stream,
                    head: 0,
                    count: 0,
                    frame: vec![0.0; n],
                };
                Ok(Box::new(core) as Box<dyn PolyphaseBlockConvolver + Send>)
            })
            .collect()
    }

    /// Bytes of device memory one channel holds on its own (the shared spectra not counted).
    pub fn device_bytes_per_channel(&self) -> usize {
        let (n, s) = (self.shared.n, &self.shared);
        8 * (n + 2 * 2 * n + s.parts * 2 * n + 2 * s.pairs * 2 * n + s.block * s.l)
    }

    fn run_block(&mut self, input: &[f64], out: &mut [f64]) -> Result<(), DriverError> {
        let s = Arc::clone(&self.shared);
        let (block, n) = (s.block, s.n);
        self.frame[block..].copy_from_slice(input);
        self.stream.memcpy_htod(&self.frame, &mut self.d_in)?;
        let n_i = n as i32;
        // SAFETY (all launches): argument types and order match the kernel signatures in
        // kernels.cu; every index the kernels compute is bounded by the sizes passed, and
        // the buffers were allocated with exactly those sizes.
        unsafe {
            self.stream
                .launch_builder(&s.k_pack)
                .arg(&self.d_in)
                .arg(&mut self.d_a)
                .arg(&n_i)
                .launch(cfg(n))?;
        }
        // Forward transform of the block, once for all phases (result in d_a).
        let mut in_a = true;
        for (radix, ns) in pass_schedule(n) {
            let ns_i = ns as i32;
            let dir = -1i32;
            let (src, dst) = if in_a {
                (&self.d_a, &mut self.d_b)
            } else {
                (&self.d_b, &mut self.d_a)
            };
            let (func, items) = if radix == 4 {
                (&s.k_radix4, n / 4)
            } else {
                (&s.k_radix2, n / 2)
            };
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(src)
                    .arg(dst)
                    .arg(&n_i)
                    .arg(&ns_i)
                    .arg(&dir)
                    .launch(cfg(items))?;
            }
            in_a = !in_a;
        }
        if !in_a {
            std::mem::swap(&mut self.d_a, &mut self.d_b);
        }
        self.head = (self.head + s.parts - 1) % s.parts;
        self.count = (self.count + 1).min(s.parts);
        let slot = self.head as i32;
        let depth = self.count as i32;
        let slots = s.parts as i32;
        let parts = s.parts as i32;
        let pairs = s.pairs as i32;
        unsafe {
            self.stream
                .launch_builder(&s.k_copy)
                .arg(&self.d_a)
                .arg(&mut self.d_fdl)
                .arg(&n_i)
                .arg(&slot)
                .launch(cfg(n))?;
            self.stream
                .launch_builder(&s.k_mac)
                .arg(&self.d_fdl)
                .arg(&s.d_g)
                .arg(&mut self.d_acc)
                .arg(&n_i)
                .arg(&depth)
                .arg(&slot)
                .arg(&slots)
                .arg(&parts)
                .arg(&pairs)
                .launch(cfg(s.pairs * n))?;
        }
        // Every pair's inverse transform in one launch per pass (result in d_acc).
        let mut in_acc = true;
        for (radix, ns) in pass_schedule(n) {
            let ns_i = ns as i32;
            let dir = 1i32;
            let (src, dst) = if in_acc {
                (&self.d_acc, &mut self.d_acc2)
            } else {
                (&self.d_acc2, &mut self.d_acc)
            };
            let (func, items) = if radix == 4 {
                (&s.k_radix4_batch, s.pairs * n / 4)
            } else {
                (&s.k_radix2_batch, s.pairs * n / 2)
            };
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(src)
                    .arg(dst)
                    .arg(&n_i)
                    .arg(&ns_i)
                    .arg(&dir)
                    .arg(&pairs)
                    .launch(cfg(items))?;
            }
            in_acc = !in_acc;
        }
        if !in_acc {
            std::mem::swap(&mut self.d_acc, &mut self.d_acc2);
        }
        let block_i = block as i32;
        let l_i = s.l as i32;
        let scale = 1.0 / n as f64;
        unsafe {
            self.stream
                .launch_builder(&s.k_interleave)
                .arg(&self.d_acc)
                .arg(&mut self.d_out)
                .arg(&block_i)
                .arg(&l_i)
                .arg(&scale)
                .launch(cfg(block * s.l))?;
        }
        self.stream.memcpy_dtoh(&self.d_out, out)?;
        self.stream.synchronize()?;
        self.frame.copy_within(block.., 0);
        Ok(())
    }
}

impl PolyphaseBlockConvolver for PolyphaseConvolver {
    fn block(&self) -> usize {
        self.shared.block
    }

    fn phases(&self) -> usize {
        self.shared.l
    }

    fn process_block(&mut self, input: &[f64], out: &mut [f64]) -> Result<(), GpuError> {
        assert_eq!(input.len(), self.shared.block, "one block in");
        assert_eq!(
            out.len(),
            self.shared.block * self.shared.l,
            "block × L out"
        );
        self.run_block(input, out).map_err(GpuError::from)
    }

    fn reset(&mut self) {
        // The delay line is cleared lazily through `count`; `head` restarts like the model.
        self.frame.fill(0.0);
        self.head = 0;
        self.count = 0;
    }

    fn device_label(&self) -> String {
        format!(
            "CUDA ({})",
            super::short_device_name(&self.shared.device_name)
        )
    }

    fn on_gpu(&self) -> bool {
        true
    }
}

/// What the channels of one rational stage share: the kernel and the sub-filters.
struct RationalShared {
    _module: Arc<CudaModule>,
    kernel: CudaFunction,
    /// `l × s` doubles, the reversed sub-filters one after another.
    d_sub: CudaSlice<f64>,
    l: usize,
    s: usize,
    device_name: String,
}

/// The rational resampler's time-domain polyphase bank on a CUDA device (one per channel,
/// each on its own stream; the sub-filters are shared): per call, upload the input window,
/// one block of 256 threads per output sample, download the outputs. Mirrors
/// [`super::wgsl::rational_dot`].
pub struct RationalEngine {
    shared: Arc<RationalShared>,
    stream: Arc<CudaStream>,
    d_win: CudaSlice<f64>,
    d_out: CudaSlice<f64>,
}

impl RationalEngine {
    /// Build `channels` engines for the reversed sub-filters `sub` on CUDA device `device`.
    pub fn for_channels(
        sub: &[Vec<f64>],
        channels: usize,
        device: usize,
    ) -> Result<Vec<Box<dyn RationalDotEngine + Send>>, GpuError> {
        let Backend::Cuda { name, .. } = probe(device)? else {
            unreachable!("probe returns a CUDA backend or an error")
        };
        let (l, s) = (sub.len(), sub.first().map_or(0, Vec::len));
        if l == 0 || s == 0 {
            return Err(GpuError::Driver("empty rational filter".into()));
        }
        let ctx = CudaContext::new(device)?;
        let streams = (0..channels.max(1))
            .map(|_| ctx.new_stream())
            .collect::<Result<Vec<_>, _>>()?;
        let module = compile(&ctx)?;
        let flat: Vec<f64> = sub.iter().flatten().copied().collect();
        let d_sub = streams[0].clone_htod(&flat)?;
        streams[0].synchronize()?;
        let shared = Arc::new(RationalShared {
            kernel: module.load_function("rational_dot")?,
            _module: module,
            d_sub,
            l,
            s,
            device_name: name,
        });
        streams
            .into_iter()
            .map(|stream| {
                Ok(Box::new(RationalEngine {
                    d_win: stream.alloc_zeros::<f64>(4 * s)?,
                    d_out: stream.alloc_zeros::<f64>(1024)?,
                    shared: Arc::clone(&shared),
                    stream,
                }) as Box<dyn RationalDotEngine + Send>)
            })
            .collect()
    }

    fn run_on_device(
        &mut self,
        window: &[f64],
        first: usize,
        phase0: usize,
        m: usize,
        out: &mut [f64],
    ) -> Result<(), DriverError> {
        let sh = Arc::clone(&self.shared);
        if self.d_win.len() < window.len() {
            self.d_win = self
                .stream
                .alloc_zeros::<f64>(window.len().next_power_of_two())?;
        }
        if self.d_out.len() < out.len() {
            self.d_out = self
                .stream
                .alloc_zeros::<f64>(out.len().next_power_of_two())?;
        }
        self.stream
            .memcpy_htod(window, &mut self.d_win.slice_mut(..window.len()))?;
        let (s, l, m) = (sh.s as i32, sh.l as i32, m as i32);
        let (phase0, first, count) = (phase0 as i32, first as i32, out.len() as i32);
        let cfg = LaunchConfig {
            grid_dim: (out.len() as u32, 1, 1),
            block_dim: (super::wgsl::RATIONAL_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument types and order match `rational_dot` in kernels.cu; the caller
        // guarantees every window index the outputs need is inside `window` (the trait
        // contract), the device buffers are at least that long, and `d_out` holds `count`.
        unsafe {
            self.stream
                .launch_builder(&sh.kernel)
                .arg(&sh.d_sub)
                .arg(&self.d_win)
                .arg(&mut self.d_out)
                .arg(&s)
                .arg(&l)
                .arg(&m)
                .arg(&phase0)
                .arg(&first)
                .arg(&count)
                .launch(cfg)?;
        }
        self.stream
            .memcpy_dtoh(&self.d_out.slice(..out.len()), out)?;
        self.stream.synchronize()
    }
}

impl RationalDotEngine for RationalEngine {
    fn phases(&self) -> usize {
        self.shared.l
    }

    fn sub_len(&self) -> usize {
        self.shared.s
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
        self.run_on_device(window, first, phase0, m, out)
            .map_err(GpuError::from)
    }

    fn device_label(&self) -> String {
        format!(
            "CUDA ({})",
            super::short_device_name(&self.shared.device_name)
        )
    }

    fn on_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolution::Convolver as CpuConvolver;

    /// Runs only on a machine with an NVIDIA card: `cargo test --features cuda -- --ignored`.
    #[test]
    #[ignore = "needs a CUDA device"]
    fn gpu_matches_cpu_convolver_bit_for_bit_within_f64_rounding() {
        let h: Vec<f64> = (0..37 * 1024 + 5)
            .map(|i| ((i as f64) * 0.013).sin() / (1.0 + i as f64 * 0.01))
            .collect();
        let x: Vec<f64> = (0..16 * 1024).map(|i| ((i as f64) * 0.37).cos()).collect();
        let filter = Arc::new(SpectralFilter::new(&h, 1024));
        let mut cpu = CpuConvolver::from_filter(Arc::clone(&filter));
        let mut gpu = Convolver::from_filter(filter, 0).expect("CUDA device");
        assert_eq!(gpu.latency(), cpu.latency());
        let mut want = vec![0.0; 1024];
        let mut got = vec![0.0; 1024];
        for chunk in x.chunks(1024) {
            cpu.process_block(chunk, &mut want);
            gpu.process_block(chunk, &mut got);
            for (a, b) in want.iter().zip(&got) {
                assert!((a - b).abs() < 1e-10, "{a} vs {b}");
            }
        }
    }

    /// Runs only on a machine with an NVIDIA card: the device polyphase core against its
    /// CPU model and the CPU engine, ×2 … ×64 and an odd L.
    #[test]
    #[ignore = "needs a CUDA device"]
    fn gpu_polyphase_matches_the_reference_and_the_cpu_resampler() {
        use crate::cache::CoefficientCache;
        use crate::fir::FilterKind;
        use crate::resampler::Resampler;
        let backend = probe(0).expect("CUDA device");
        let cache = CoefficientCache::disabled();
        let kind = FilterKind::parse("sinc-l-64k").unwrap();
        for (inr, outr) in [
            (44_100, 88_200),
            (44_100, 176_400),
            (44_100, 705_600),
            (44_100, 1_411_200),
            (44_100, 2_822_400),
            (16_000, 48_000),
        ] {
            let mut cpu = Resampler::new(inr, outr, 2, kind, 150.0, &cache).unwrap();
            let mut gpu =
                Resampler::new_on(inr, outr, 2, kind, 150.0, &cache, Some(&backend)).unwrap();
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
            assert!(err / peak < 1e-12, "{inr}→{outr}: {err:e}");
        }
        // Device core vs its model, block by block.
        let h: Vec<f64> = (0..8 * 4096 + 3)
            .map(|i| ((i as f64) * 0.011).sin() / (1.0 + i as f64 * 0.003))
            .collect();
        let subs: Vec<SpectralFilter> = (0..5)
            .map(|p| {
                let s: Vec<f64> = h.iter().skip(p).step_by(5).copied().collect();
                SpectralFilter::new(&s, 512)
            })
            .collect();
        let mut dev = PolyphaseConvolver::for_channels(&subs, 1, 0).expect("CUDA device");
        let mut model = super::super::reference::ReferencePolyphase::new(&subs).unwrap();
        let mut want = vec![0.0; 512 * 5];
        let mut got = vec![0.0; 512 * 5];
        for b in 0..12 {
            let x: Vec<f64> = (0..512)
                .map(|i| ((b * 512 + i) as f64 * 0.37).cos())
                .collect();
            model.process_block(&x, &mut want).unwrap();
            dev[0].process_block(&x, &mut got).unwrap();
            for (p, q) in want.iter().zip(&got) {
                assert!((p - q).abs() < 1e-11, "{p} vs {q}");
            }
        }
    }

    /// Runs only on a machine with an NVIDIA card: the rational stage on the device against
    /// the CPU resampler (44.1 → 48 kHz and back, and the 44.1 kHz → 1.536 MHz cascade).
    #[test]
    #[ignore = "needs a CUDA device"]
    fn gpu_rational_stage_matches_the_cpu_resampler() {
        use crate::cache::CoefficientCache;
        use crate::fir::FilterKind;
        use crate::resampler::Resampler;
        let backend = probe(0).expect("CUDA device");
        let cache = CoefficientCache::disabled();
        let kind = FilterKind::parse("sinc-l-64k").unwrap();
        for (inr, outr) in [(44_100, 48_000), (48_000, 44_100), (44_100, 1_536_000)] {
            let mut cpu = Resampler::new(inr, outr, 2, kind, 150.0, &cache).unwrap();
            let mut gpu =
                Resampler::new_on(inr, outr, 2, kind, 150.0, &cache, Some(&backend)).unwrap();
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
            assert!(err / peak < 1e-12, "{inr}→{outr}: {err:e}");
        }
    }

    #[test]
    fn probe_on_a_machine_without_a_card_is_an_error_not_a_panic() {
        // On the development machine (no GPU) this must come back as a clean error; on a
        // machine with a card it must describe it.
        match probe(0) {
            Ok(Backend::Cuda { name, .. }) => assert!(!name.is_empty()),
            Ok(other) => panic!("probe returned {other:?}"),
            Err(GpuError::NoDriver(_) | GpuError::NoDevice(_) | GpuError::Driver(_)) => {}
            Err(e) => panic!("unexpected error {e}"),
        }
        assert!(matches!(
            probe(usize::MAX),
            Err(GpuError::NoDriver(_) | GpuError::NoDevice(_) | GpuError::Driver(_))
        ));
    }
}
