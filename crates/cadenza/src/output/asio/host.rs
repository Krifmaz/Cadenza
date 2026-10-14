//! The ASIO host: a safe wrapper around a driver object, the sample-type conversions, and
//! the double-buffer machinery that feeds the driver from its callback thread.
//!
//! Platform independent on purpose: the driver is only ever reached through its method
//! table, so the tests drive this code with a fake driver written in Rust, and only the
//! loader (registry + COM) is Windows-specific.
//!
//! **Stream model.** `write` converts the engine's words into per-channel units (PCM
//! words, or for native DSD one byte of 8 one-bit samples per channel) and appends them
//! to a ring. The driver calls `bufferSwitch(index)` once per half buffer from its own
//! thread; the callback takes exactly one half's worth of frames from the ring,
//! converts them to the channels' sample type in place, and calls `outputReady` when the
//! driver supports it. A short ring is filled with silence (counted as an underrun);
//! silence is zero for PCM and DoP and the idle pattern `0x69` for native DSD.
//!
//! ASIO allows one loaded driver per process, so the callbacks (plain C functions with no
//! user pointer) reach the active stream through one global.

use super::sys::{self, BufferInfo, Callbacks, ChannelInfo, IAsio, IoFormat};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::Thread;

/// A failed driver call.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("ASIO {call} failed: {reason}")]
pub struct AsioError {
    /// The method that failed.
    pub call: &'static str,
    /// The driver's error code (or 0 for a host-side check).
    pub code: i32,
    /// What it means (the driver's message when it gives one).
    pub reason: String,
}

impl AsioError {
    fn code(call: &'static str, code: i32) -> Self {
        Self {
            call,
            code,
            reason: sys::error_name(code).to_string(),
        }
    }

    /// A host-side failure (no driver error code).
    pub fn host(call: &'static str, reason: impl Into<String>) -> Self {
        Self {
            call,
            code: 0,
            reason: reason.into(),
        }
    }
}

fn check(call: &'static str, code: i32) -> Result<(), AsioError> {
    if code == sys::ASE_OK || code == sys::ASE_SUCCESS {
        Ok(())
    } else {
        Err(AsioError::code(call, code))
    }
}

/// The driver's buffer size limits, in samples (one-bit samples in DSD mode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferSizes {
    /// Smallest.
    pub min: i32,
    /// Largest.
    pub max: i32,
    /// What the driver (and its control panel) prefers.
    pub preferred: i32,
    /// Step between sizes (−1: powers of two).
    pub granularity: i32,
}

/// One reference to a driver object (released on drop).
#[derive(Debug)]
pub struct Driver {
    ptr: NonNull<IAsio>,
    name: String,
}

// The driver is used from one thread at a time (the engine's); only `outputReady` is
// called from the driver's own callback thread, which ASIO permits.
unsafe impl Send for Driver {}

fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

impl Driver {
    /// Take over one reference to `ptr`.
    ///
    /// # Safety
    /// `ptr` must point to a live ASIO driver object whose method table follows
    /// [`sys::IAsioVtbl`], and the caller must own the reference it passes.
    pub unsafe fn from_raw(ptr: *mut IAsio, name: impl Into<String>) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self {
            ptr,
            name: name.into(),
        })
    }

    fn vt(&self) -> &sys::IAsioVtbl {
        // SAFETY: the object is live while we hold a reference (`from_raw` contract).
        unsafe { &*(*self.ptr.as_ptr()).vtbl }
    }

    fn this(&self) -> *mut IAsio {
        self.ptr.as_ptr()
    }

    /// The object pointer, for `outputReady` from the callback. It stays valid while
    /// this `Driver` lives; streams holding a copy are closed before it is dropped.
    pub fn as_raw(&self) -> *mut IAsio {
        self.this()
    }

    /// The registry name it was loaded under.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `init(sysHandle)` (a window handle, passed through as an integer); on failure the driver's own message explains why (no device
    /// connected, in use by another program…).
    pub fn init(&self, sys_handle: isize) -> Result<(), AsioError> {
        // SAFETY (all calls below): `this` is the live driver object and every pointer
        // argument points to a local of the type the method expects.
        let ok = unsafe { (self.vt().init)(self.this(), sys_handle as *mut c_void) };
        if ok != 0 {
            return Ok(());
        }
        let mut msg = [0u8; 128];
        unsafe { (self.vt().get_error_message)(self.this(), msg.as_mut_ptr()) };
        let reason = c_string(&msg);
        Err(AsioError::host(
            "init",
            if reason.is_empty() {
                "the driver refused to start".to_string()
            } else {
                reason
            },
        ))
    }

    /// The driver's own name.
    pub fn driver_name(&self) -> String {
        let mut name = [0u8; 64];
        unsafe { (self.vt().get_driver_name)(self.this(), name.as_mut_ptr()) };
        c_string(&name)
    }

    /// `(inputs, outputs)`.
    pub fn channels(&self) -> Result<(i32, i32), AsioError> {
        let (mut i, mut o) = (0, 0);
        check("getChannels", unsafe {
            (self.vt().get_channels)(self.this(), &mut i, &mut o)
        })?;
        Ok((i, o))
    }

    /// Buffer size limits.
    pub fn buffer_sizes(&self) -> Result<BufferSizes, AsioError> {
        let (mut min, mut max, mut preferred, mut granularity) = (0, 0, 0, 0);
        check("getBufferSize", unsafe {
            (self.vt().get_buffer_size)(
                self.this(),
                &mut min,
                &mut max,
                &mut preferred,
                &mut granularity,
            )
        })?;
        Ok(BufferSizes {
            min,
            max,
            preferred,
            granularity,
        })
    }

    /// `(input, output)` latency in samples.
    pub fn latencies(&self) -> Result<(i32, i32), AsioError> {
        let (mut i, mut o) = (0, 0);
        check("getLatencies", unsafe {
            (self.vt().get_latencies)(self.this(), &mut i, &mut o)
        })?;
        Ok((i, o))
    }

    /// Whether the driver can run at `rate` (the DSD bit rate in DSD mode).
    pub fn can_sample_rate(&self, rate: f64) -> bool {
        unsafe { (self.vt().can_sample_rate)(self.this(), rate) == sys::ASE_OK }
    }

    /// The current sample rate.
    pub fn sample_rate(&self) -> Result<f64, AsioError> {
        let mut r = 0.0;
        check("getSampleRate", unsafe {
            (self.vt().get_sample_rate)(self.this(), &mut r)
        })?;
        Ok(r)
    }

    /// Set the sample rate.
    pub fn set_sample_rate(&self, rate: f64) -> Result<(), AsioError> {
        check("setSampleRate", unsafe {
            (self.vt().set_sample_rate)(self.this(), rate)
        })
    }

    /// One output (or input) channel's description.
    pub fn channel_info(&self, channel: i32, input: bool) -> Result<ChannelInfo, AsioError> {
        let mut info = ChannelInfo {
            channel,
            is_input: input as i32,
            is_active: 0,
            channel_group: 0,
            sample_type: -1,
            name: [0; 32],
        };
        check("getChannelInfo", unsafe {
            (self.vt().get_channel_info)(self.this(), &mut info)
        })?;
        Ok(info)
    }

    fn io_format(&self, selector: i32, format_type: i32) -> bool {
        let mut f = IoFormat::new(format_type);
        let r = unsafe {
            (self.vt().future)(
                self.this(),
                selector,
                &mut f as *mut IoFormat as *mut c_void,
            )
        };
        r == sys::ASE_SUCCESS
    }

    /// Whether the driver can switch to DSD (`kAsioCanDoIoFormat`).
    pub fn can_do_dsd(&self) -> bool {
        self.io_format(sys::FUTURE_CAN_DO_IO_FORMAT, sys::IO_FORMAT_DSD)
    }

    /// Switch between PCM and DSD (before the buffers are created). Drivers without DSD
    /// support refuse both requests, which is fine for PCM.
    pub fn set_dsd(&self, dsd: bool) -> bool {
        let t = if dsd {
            sys::IO_FORMAT_DSD
        } else {
            sys::IO_FORMAT_PCM
        };
        self.io_format(sys::FUTURE_SET_IO_FORMAT, t)
    }

    /// Create the double buffers for `infos` (`buffer_size` samples each).
    ///
    /// # Safety
    /// `callbacks` must stay valid until [`Driver::dispose_buffers`].
    pub unsafe fn create_buffers(
        &self,
        infos: &mut [BufferInfo],
        buffer_size: i32,
        callbacks: *mut Callbacks,
    ) -> Result<(), AsioError> {
        check("createBuffers", unsafe {
            (self.vt().create_buffers)(
                self.this(),
                infos.as_mut_ptr(),
                infos.len() as i32,
                buffer_size,
                callbacks,
            )
        })
    }

    /// Release the buffers.
    pub fn dispose_buffers(&self) -> Result<(), AsioError> {
        check("disposeBuffers", unsafe {
            (self.vt().dispose_buffers)(self.this())
        })
    }

    /// Start the clock (the callbacks begin).
    pub fn start(&self) -> Result<(), AsioError> {
        check("start", unsafe { (self.vt().start)(self.this()) })
    }

    /// Stop the clock (no callback after this returns).
    pub fn stop(&self) -> Result<(), AsioError> {
        check("stop", unsafe { (self.vt().stop)(self.this()) })
    }

    /// `outputReady()`: `true` when the driver supports the early-output hint.
    pub fn output_ready(&self) -> bool {
        unsafe { (self.vt().output_ready)(self.this()) == sys::ASE_OK }
    }

    /// Open the driver's control panel.
    pub fn control_panel(&self) -> Result<(), AsioError> {
        check("controlPanel", unsafe {
            (self.vt().control_panel)(self.this())
        })
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        // SAFETY: we own one reference (`from_raw`).
        unsafe { (self.vt().release)(self.this()) };
    }
}

/// How one channel's samples are laid out in the driver's buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    /// 16-bit integer.
    Int16 {
        /// Big-endian bytes.
        big_endian: bool,
    },
    /// 24-bit packed integer (3 bytes).
    Int24 {
        /// Big-endian bytes.
        big_endian: bool,
    },
    /// 32-bit integer, `bits` valid bits right-aligned (`bits` = 32: full scale).
    Int32 {
        /// Valid bits (right-aligned).
        bits: u32,
        /// Big-endian bytes.
        big_endian: bool,
    },
    /// 32-bit float, full scale ±1.
    Float32 {
        /// Big-endian bytes.
        big_endian: bool,
    },
    /// 64-bit float, full scale ±1.
    Float64 {
        /// Big-endian bytes.
        big_endian: bool,
    },
    /// DSD, 8 one-bit samples per byte, the first in the most significant bit.
    DsdMsb1,
    /// DSD, 8 one-bit samples per byte, the first in the least significant bit.
    DsdLsb1,
}

impl SampleFormat {
    /// From an `ASIOSampleType` (`None` for the types this host does not write).
    pub fn from_asio(t: i32) -> Option<Self> {
        Some(match t {
            sys::ST_INT16_LSB => Self::Int16 { big_endian: false },
            sys::ST_INT16_MSB => Self::Int16 { big_endian: true },
            sys::ST_INT24_LSB => Self::Int24 { big_endian: false },
            sys::ST_INT24_MSB => Self::Int24 { big_endian: true },
            sys::ST_INT32_LSB => Self::Int32 {
                bits: 32,
                big_endian: false,
            },
            sys::ST_INT32_MSB => Self::Int32 {
                bits: 32,
                big_endian: true,
            },
            sys::ST_INT32_LSB16 => Self::Int32 {
                bits: 16,
                big_endian: false,
            },
            sys::ST_INT32_LSB18 => Self::Int32 {
                bits: 18,
                big_endian: false,
            },
            sys::ST_INT32_LSB20 => Self::Int32 {
                bits: 20,
                big_endian: false,
            },
            sys::ST_INT32_LSB24 => Self::Int32 {
                bits: 24,
                big_endian: false,
            },
            sys::ST_INT32_MSB16 => Self::Int32 {
                bits: 16,
                big_endian: true,
            },
            sys::ST_INT32_MSB24 => Self::Int32 {
                bits: 24,
                big_endian: true,
            },
            sys::ST_FLOAT32_LSB => Self::Float32 { big_endian: false },
            sys::ST_FLOAT32_MSB => Self::Float32 { big_endian: true },
            sys::ST_FLOAT64_LSB => Self::Float64 { big_endian: false },
            sys::ST_FLOAT64_MSB => Self::Float64 { big_endian: true },
            sys::ST_DSD_INT8_MSB1 => Self::DsdMsb1,
            sys::ST_DSD_INT8_LSB1 => Self::DsdLsb1,
            _ => return None,
        })
    }

    /// Bytes per sample in the driver's buffer (per byte of 8 samples for DSD).
    pub fn bytes(self) -> usize {
        match self {
            Self::Int16 { .. } => 2,
            Self::Int24 { .. } => 3,
            Self::Int32 { .. } | Self::Float32 { .. } => 4,
            Self::Float64 { .. } => 8,
            Self::DsdMsb1 | Self::DsdLsb1 => 1,
        }
    }

    /// A DSD layout.
    pub fn is_dsd(self) -> bool {
        matches!(self, Self::DsdMsb1 | Self::DsdLsb1)
    }

    /// The word size to open PCM with: 16 for 16-bit channels, 32 for full 32-bit
    /// integer channels when 32 bits are asked for, 24 otherwise (float carries 24 bits
    /// exactly).
    pub fn pcm_bits(self, requested: u32) -> u32 {
        match self {
            Self::Int16 { .. } | Self::Int32 { bits: 16, .. } => 16,
            Self::Int32 { bits: 32, .. } if requested >= 32 => 32,
            Self::Float64 { .. } if requested >= 32 => 32,
            _ => 24,
        }
    }

    /// Integer samples of at least 24 bits: DoP markers survive untouched.
    pub fn carries_dop(self) -> bool {
        match self {
            Self::Int24 { .. } => true,
            Self::Int32 { bits, .. } => bits >= 24,
            _ => false,
        }
    }

    /// Write one PCM word `v` (right-justified, `bits` valid bits) to `dst`.
    #[inline]
    pub fn put_pcm(self, dst: &mut [u8], v: i32, bits: u32) {
        let to = |target: u32| -> i32 {
            if bits >= target {
                v >> (bits - target)
            } else {
                v << (target - bits)
            }
        };
        let full = (1u64 << (bits - 1)) as f64;
        match self {
            Self::Int16 { big_endian } => {
                let s = to(16) as i16;
                dst[..2].copy_from_slice(&if big_endian {
                    s.to_be_bytes()
                } else {
                    s.to_le_bytes()
                });
            }
            Self::Int24 { big_endian } => {
                let b = to(24).to_le_bytes();
                if big_endian {
                    dst[..3].copy_from_slice(&[b[2], b[1], b[0]]);
                } else {
                    dst[..3].copy_from_slice(&b[..3]);
                }
            }
            Self::Int32 {
                bits: target,
                big_endian,
            } => {
                let s = to(target);
                dst[..4].copy_from_slice(&if big_endian {
                    s.to_be_bytes()
                } else {
                    s.to_le_bytes()
                });
            }
            Self::Float32 { big_endian } => {
                let f = (v as f64 / full) as f32;
                dst[..4].copy_from_slice(&if big_endian {
                    f.to_be_bytes()
                } else {
                    f.to_le_bytes()
                });
            }
            Self::Float64 { big_endian } => {
                let f = v as f64 / full;
                dst[..8].copy_from_slice(&if big_endian {
                    f.to_be_bytes()
                } else {
                    f.to_le_bytes()
                });
            }
            Self::DsdMsb1 | Self::DsdLsb1 => unreachable!("DSD channels take bytes"),
        }
    }

    /// Write one DSD byte (8 samples, the oldest in the MSB) to `dst`.
    #[inline]
    pub fn put_dsd(self, dst: &mut [u8], byte: u8) {
        dst[0] = match self {
            Self::DsdLsb1 => byte.reverse_bits(),
            _ => byte,
        };
    }
}

/// DSD silence: the idle pattern (equal ones and zeros, no DC).
pub const DSD_SILENCE: u8 = 0x69;

/// Split native DSD words (interleaved, 32 one-bit samples per channel each, the oldest
/// in the MSB) into byte frames: four frames of one byte per channel per word frame.
pub fn dsd_words_to_units(words: &[i32], channels: usize, out: &mut Vec<i32>) {
    for frame in words.chunks_exact(channels) {
        for k in 0..4 {
            for w in frame {
                out.push(w.to_be_bytes()[k] as i32);
            }
        }
    }
}

/// Everything the driver's callback thread needs, shared with the writer.
pub struct Shared {
    /// Units (PCM words, or DSD bytes) waiting for the driver, interleaved.
    ring: Mutex<VecDeque<i32>>,
    capacity: usize,
    channels: usize,
    /// Frames per half buffer (PCM samples; DSD bytes).
    frames: usize,
    /// The driver's buffers per channel: `[half 0, half 1]`.
    buffers: Vec<[*mut u8; 2]>,
    format: SampleFormat,
    /// Valid bits of the PCM words.
    bits: u32,
    /// Driver pointer for `outputReady` (when supported).
    output_ready: Option<NonNull<IAsio>>,
    paused: AtomicBool,
    /// The writer is draining: a short last half buffer is the end, not an underrun.
    draining: AtomicBool,
    /// Callbacks that found the ring short.
    pub underruns: AtomicU64,
    /// Half buffers delivered.
    pub switches: AtomicU64,
    /// The driver asked for a reset (or its rate changed under us).
    pub reset_request: AtomicBool,
    waiter: Mutex<Option<Thread>>,
}

// The raw buffer pointers are written only from the callback thread, between the
// driver's `start` and `stop`; everything else is atomics and mutexes.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    /// State for `channels` channels of `format`, `frames` per half buffer.
    pub fn new(
        buffers: Vec<[*mut u8; 2]>,
        format: SampleFormat,
        bits: u32,
        frames: usize,
        capacity_frames: usize,
        output_ready: Option<NonNull<IAsio>>,
    ) -> Self {
        let channels = buffers.len();
        Self {
            ring: Mutex::new(VecDeque::with_capacity(capacity_frames * channels)),
            capacity: capacity_frames.max(2 * frames) * channels,
            channels,
            frames,
            buffers,
            format,
            bits,
            output_ready,
            paused: AtomicBool::new(false),
            draining: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            switches: AtomicU64::new(0),
            reset_request: AtomicBool::new(false),
            waiter: Mutex::new(None),
        }
    }

    /// Frames per half buffer.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Frames the ring can hold.
    pub fn capacity_frames(&self) -> usize {
        self.capacity / self.channels
    }

    /// Frames queued.
    pub fn queued_frames(&self) -> usize {
        self.ring.lock().unwrap_or_else(|e| e.into_inner()).len() / self.channels
    }

    /// Append as many whole frames of `units` as fit; returns the units taken.
    pub fn push(&self, units: &[i32]) -> usize {
        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        let room = (self.capacity - ring.len()) / self.channels * self.channels;
        let n = room.min(units.len());
        ring.extend(&units[..n]);
        n
    }

    /// Drop everything queued.
    pub fn clear(&self) {
        self.ring.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// The writer is draining (`true`) or writing again (`false`).
    pub fn set_draining(&self, draining: bool) {
        self.draining.store(draining, Ordering::Release);
    }

    /// Output silence while `true` (the queue is kept).
    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Release);
    }

    /// Register the current thread to be unparked when room frees up.
    pub fn wait_for_room(&self, timeout: std::time::Duration) {
        *self.waiter.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::thread::current());
        std::thread::park_timeout(timeout);
    }

    /// Silence in both halves (before the first start).
    pub fn prime(&self) {
        let bytes = self.format.bytes();
        for bufs in &self.buffers {
            for half in bufs {
                // SAFETY: as in `fill`.
                let dst = unsafe { std::slice::from_raw_parts_mut(*half, self.frames * bytes) };
                for f in 0..self.frames {
                    let d = &mut dst[f * bytes..(f + 1) * bytes];
                    if self.format.is_dsd() {
                        self.format.put_dsd(d, DSD_SILENCE);
                    } else {
                        self.format.put_pcm(d, 0, self.bits);
                    }
                }
            }
        }
    }

    /// Fill half `index` of every channel's buffer (the driver's `bufferSwitch`).
    pub fn fill(&self, index: usize) {
        let index = index & 1;
        let n = self.frames;
        let bytes = self.format.bytes();
        let silence_unit: i32 = if self.format.is_dsd() {
            DSD_SILENCE as i32
        } else {
            0
        };
        let mut short = false;
        {
            let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
            let paused = self.paused.load(Ordering::Acquire);
            for f in 0..n {
                let take = !paused && ring.len() >= self.channels;
                if !paused && !take {
                    short = true;
                }
                for bufs in &self.buffers {
                    let unit = if take {
                        ring.pop_front().unwrap_or(silence_unit)
                    } else {
                        silence_unit
                    };
                    // SAFETY: the driver gave us `frames` samples of `bytes` each in both
                    // halves of every channel (createBuffers), valid until disposeBuffers.
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(bufs[index].add(f * bytes), bytes)
                    };
                    if self.format.is_dsd() {
                        self.format.put_dsd(dst, unit as u8);
                    } else {
                        self.format.put_pcm(dst, unit, self.bits);
                    }
                }
            }
        }
        if short && !self.draining.load(Ordering::Acquire) {
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
        self.switches.fetch_add(1, Ordering::Release);
        if let Some(d) = self.output_ready {
            // SAFETY: the driver is live while the stream exists; outputReady is the
            // documented call from inside bufferSwitch.
            unsafe { ((*(*d.as_ptr()).vtbl).output_ready)(d.as_ptr()) };
        }
        if let Ok(mut w) = self.waiter.try_lock() {
            if let Some(t) = w.take() {
                t.unpark();
            }
        }
    }
}

/// The stream the callbacks feed (one per process, an ASIO rule).
static ACTIVE: AtomicPtr<Shared> = AtomicPtr::new(std::ptr::null_mut());

/// Make `shared` the target of the callbacks (holds a reference until [`deactivate`]).
pub fn activate(shared: &Arc<Shared>) {
    let old = ACTIVE.swap(
        Arc::into_raw(Arc::clone(shared)) as *mut Shared,
        Ordering::AcqRel,
    );
    if !old.is_null() {
        // SAFETY: every pointer stored in ACTIVE came from `Arc::into_raw`.
        drop(unsafe { Arc::from_raw(old) });
    }
}

/// Detach the callbacks (call after the driver has stopped).
pub fn deactivate() {
    let old = ACTIVE.swap(std::ptr::null_mut(), Ordering::AcqRel);
    if !old.is_null() {
        // SAFETY: as in `activate`.
        drop(unsafe { Arc::from_raw(old) });
    }
}

fn with_active(f: impl FnOnce(&Shared)) {
    let p = ACTIVE.load(Ordering::Acquire);
    if !p.is_null() {
        // SAFETY: ACTIVE holds a strong reference until `deactivate`, which runs only
        // after the driver stopped calling back.
        f(unsafe { &*p });
    }
}

extern "C" fn buffer_switch(index: i32, _direct: i32) {
    with_active(|s| s.fill(index as usize));
}

extern "C" fn sample_rate_did_change(_rate: f64) {
    with_active(|s| s.reset_request.store(true, Ordering::Release));
}

extern "C" fn asio_message(
    selector: i32,
    value: i32,
    _message: *mut c_void,
    _opt: *mut f64,
) -> i32 {
    match selector {
        sys::MSG_SELECTOR_SUPPORTED => matches!(
            value,
            sys::MSG_RESET_REQUEST
                | sys::MSG_ENGINE_VERSION
                | sys::MSG_RESYNC_REQUEST
                | sys::MSG_LATENCIES_CHANGED
                | sys::MSG_SUPPORTS_TIME_INFO
                | sys::MSG_OVERLOAD
        ) as i32,
        sys::MSG_ENGINE_VERSION => 2,
        sys::MSG_RESET_REQUEST => {
            with_active(|s| s.reset_request.store(true, Ordering::Release));
            1
        }
        sys::MSG_RESYNC_REQUEST | sys::MSG_LATENCIES_CHANGED | sys::MSG_OVERLOAD => 1,
        // We use the plain bufferSwitch, not the time-info variant.
        sys::MSG_SUPPORTS_TIME_INFO => 0,
        _ => 0,
    }
}

extern "C" fn buffer_switch_time_info(
    params: *mut sys::AsioTime,
    index: i32,
    direct: i32,
) -> *mut sys::AsioTime {
    buffer_switch(index, direct);
    params
}

/// The callback table handed to `createBuffers` (static: drivers keep the pointer).
pub fn callbacks() -> *mut Callbacks {
    static mut CALLBACKS: Callbacks = Callbacks {
        buffer_switch,
        sample_rate_did_change,
        asio_message,
        buffer_switch_time_info,
    };
    // Never written after initialisation; drivers only read it.
    std::ptr::addr_of_mut!(CALLBACKS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_types_map_and_convert() {
        let f = SampleFormat::from_asio(sys::ST_INT32_LSB).unwrap();
        let mut b = [0u8; 4];
        f.put_pcm(&mut b, 0x12_3456, 24);
        assert_eq!(i32::from_le_bytes(b), 0x1234_5600);
        let f = SampleFormat::from_asio(sys::ST_INT24_LSB).unwrap();
        let mut b = [0u8; 3];
        f.put_pcm(&mut b, -2, 24);
        assert_eq!(b, [0xfe, 0xff, 0xff]);
        let f = SampleFormat::from_asio(sys::ST_INT24_MSB).unwrap();
        f.put_pcm(&mut b, 0x01_0203, 24);
        assert_eq!(b, [0x01, 0x02, 0x03]);
        let f = SampleFormat::from_asio(sys::ST_INT16_LSB).unwrap();
        let mut b = [0u8; 2];
        f.put_pcm(&mut b, 0x12_3456, 24);
        assert_eq!(i16::from_le_bytes(b), 0x1234);
        let f = SampleFormat::from_asio(sys::ST_INT32_LSB24).unwrap();
        let mut b = [0u8; 4];
        f.put_pcm(&mut b, -5, 24);
        assert_eq!(i32::from_le_bytes(b), -5);
        // Float carries 24 bits exactly.
        let f = SampleFormat::from_asio(sys::ST_FLOAT32_LSB).unwrap();
        f.put_pcm(&mut b, -(1 << 23), 24);
        assert_eq!(f32::from_le_bytes(b), -1.0);
        f.put_pcm(&mut b, 1, 24);
        assert_eq!((f32::from_le_bytes(b) as f64 * (1u64 << 23) as f64), 1.0);
        // DSD bit order.
        let mut b = [0u8; 1];
        SampleFormat::DsdLsb1.put_dsd(&mut b, 0b1000_0001 | 0b0100_0000);
        assert_eq!(b[0], 0b1000_0011);
        SampleFormat::DsdMsb1.put_dsd(&mut b, 0x69);
        assert_eq!(b[0], 0x69);
        assert!(SampleFormat::from_asio(sys::ST_DSD_INT8_NER8).is_none());
        // Which channel types carry DoP and what PCM depth they open with.
        assert!(SampleFormat::Int24 { big_endian: false }.carries_dop());
        assert!(!SampleFormat::Float32 { big_endian: false }.carries_dop());
        assert!(!SampleFormat::Int16 { big_endian: false }.carries_dop());
        assert_eq!(SampleFormat::Int16 { big_endian: false }.pcm_bits(24), 16);
        assert_eq!(
            SampleFormat::Int32 {
                bits: 32,
                big_endian: false
            }
            .pcm_bits(32),
            32
        );
    }

    #[test]
    fn dsd_words_split_into_byte_frames_in_time_order() {
        let words = [0x0102_0304, 0x1112_1314u32 as i32];
        let mut out = Vec::new();
        dsd_words_to_units(&words, 2, &mut out);
        assert_eq!(out, [0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 0x04, 0x14]);
    }
}
