//! The CUDA implementation (compiled only with `--features cuda`).
//!
//! `cudarc` is used with `dynamic-loading`: nothing links against CUDA at build time, and
//! the driver library is looked up when first needed. `cudarc` *panics* when it cannot find
//! the library, so [`probe`] checks for `libcuda` / `nvcuda.dll` itself (with `libloading`)
//! before any `cudarc` call, and turns the absence into [`GpuError::NoDriver`].

use super::kernels::{grid_for, pass_schedule, KERNEL_SOURCE, PROGRAM_NAME, THREADS_PER_BLOCK};
use super::reference::filter_spectra;
use super::{Backend, GpuError};
use crate::convolution::{BlockConvolver, SpectralFilter};
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
    Err(GpuError::NoDriver(last))
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
        let opts = CompileOptions {
            name: Some(PROGRAM_NAME.to_string()),
            // Keep every operation IEEE double: no fast-math, exact division and sqrt.
            use_fast_math: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts)?;
        let module = ctx.load_module(ptx)?;
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
        LaunchConfig {
            grid_dim: (grid_for(items), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
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
