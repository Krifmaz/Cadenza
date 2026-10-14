//! CoreAudio exclusive output (macOS): **hog mode** on the device, the device's nominal
//! sample rate switched to the stream's rate, and the stream's *physical* format switched
//! to an integer format of the requested depth where the device offers one ("integer
//! mode"); otherwise the device's 32-bit float physical format is used and we convert.
//!
//! Only compiled on macOS with feature `coreaudio`. It uses the raw `coreaudio-sys`
//! bindings for the few HAL calls needed (property get/set, IOProc create/start/stop) and
//! computes every four-character property selector itself, so it does not depend on which
//! constant names a given SDK's bindings happen to expose.
//!
//! Data flow: `write` pushes interleaved words into a ring buffer and blocks while more
//! than ~0.5 s is queued; the HAL's IOProc (real-time thread, no allocation) pulls from
//! the ring into the device buffer and leaves silence on underrun.
//!
//! **DSD** is DoP only (24-bit integer PCM at `f_dsd / 16`); CoreAudio has no native DSD
//! stream format.

use super::{DeviceInfo, DsdCaps, Encoding, Output, OutputError, OutputFormat, STANDARD_RATES};
use cadenza_dsp::dsd::{DsdPacking, DsdRate};
use coreaudio_sys::{
    AudioBufferList, AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID, AudioDeviceIOProcID,
    AudioDeviceStart, AudioDeviceStop, AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize,
    AudioObjectID, AudioObjectPropertyAddress, AudioObjectSetPropertyData,
    AudioStreamBasicDescription, AudioTimeStamp, OSStatus,
};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Four-character codes as CoreAudio wants them.
const fn fourcc(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}

// AudioObject / AudioHardware / AudioDevice / AudioStream property selectors, scopes and
// elements (AudioHardwareBase.h, AudioHardware.h). Spelled out so no SDK rename bites.
const K_SYSTEM_OBJECT: AudioObjectID = 1;
const K_SCOPE_GLOBAL: u32 = fourcc(b"glob");
const K_SCOPE_OUTPUT: u32 = fourcc(b"outp");
const K_ELEMENT_MAIN: u32 = 0;
const K_HW_DEVICES: u32 = fourcc(b"dev#");
const K_HW_DEFAULT_OUTPUT: u32 = fourcc(b"dOut");
const K_OBJ_NAME: u32 = fourcc(b"lnam");
const K_DEV_UID: u32 = fourcc(b"uid ");
const K_DEV_STREAMS: u32 = fourcc(b"stm#");
const K_DEV_HOG_MODE: u32 = fourcc(b"oink");
const K_DEV_NOMINAL_RATE: u32 = fourcc(b"nsrt");
const K_DEV_AVAILABLE_RATES: u32 = fourcc(b"nsr#");
const K_DEV_BUFFER_FRAME_SIZE: u32 = fourcc(b"fsiz");
const K_DEV_STREAM_CONFIG: u32 = fourcc(b"slay");
const K_STREAM_PHYSICAL_FORMAT: u32 = fourcc(b"pft ");
const K_STREAM_AVAILABLE_PHYSICAL_FORMATS: u32 = fourcc(b"pfta");
const K_FORMAT_LINEAR_PCM: u32 = fourcc(b"lpcm");
const FLAG_IS_FLOAT: u32 = 1 << 0;
const FLAG_IS_SIGNED_INTEGER: u32 = 1 << 2;
const FLAG_IS_PACKED: u32 = 1 << 3;
const FLAG_IS_ALIGNED_HIGH: u32 = 1 << 4;
const FLAG_IS_NON_INTERLEAVED: u32 = 1 << 5;

/// `AudioValueRange` (AudioHardwareBase.h).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ValueRange {
    minimum: f64,
    maximum: f64,
}

/// `AudioStreamRangedDescription`: a format plus the sample-rate range it is valid for.
#[repr(C)]
#[derive(Clone, Copy)]
struct RangedDescription {
    format: AudioStreamBasicDescription,
    rate_range: ValueRange,
}

// CoreFoundation strings for device names and UIDs (CFString.h).
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFStringGetLength(s: *const c_void) -> isize;
    fn CFStringGetCString(s: *const c_void, buf: *mut u8, size: isize, encoding: u32) -> u8;
    fn CFRelease(cf: *const c_void);
}
const CF_UTF8: u32 = 0x0800_0100;

fn addr(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: K_ELEMENT_MAIN,
    }
}

fn check(status: OSStatus, what: &str) -> Result<(), OutputError> {
    if status == 0 {
        Ok(())
    } else {
        Err(OutputError::Backend(format!(
            "CoreAudio {what}: OSStatus {status}"
        )))
    }
}

/// Read one fixed-size property.
fn get<T: Copy>(object: AudioObjectID, selector: u32, scope: u32) -> Result<T, OutputError> {
    let a = addr(selector, scope);
    let mut size = size_of::<T>() as u32;
    let mut value = std::mem::MaybeUninit::<T>::uninit();
    let st = unsafe {
        AudioObjectGetPropertyData(
            object,
            &a,
            0,
            ptr::null(),
            &mut size,
            value.as_mut_ptr() as *mut c_void,
        )
    };
    check(st, "get property")?;
    // SAFETY: the HAL filled `size` bytes of a `T`-sized buffer on success.
    Ok(unsafe { value.assume_init() })
}

/// Read a variable-length array property.
fn get_vec<T: Copy>(
    object: AudioObjectID,
    selector: u32,
    scope: u32,
) -> Result<Vec<T>, OutputError> {
    let a = addr(selector, scope);
    let mut size = 0u32;
    let st = unsafe { AudioObjectGetPropertyDataSize(object, &a, 0, ptr::null(), &mut size) };
    check(st, "get property size")?;
    let n = size as usize / size_of::<T>();
    let mut v: Vec<T> = Vec::with_capacity(n);
    let st = unsafe {
        AudioObjectGetPropertyData(
            object,
            &a,
            0,
            ptr::null(),
            &mut size,
            v.as_mut_ptr() as *mut c_void,
        )
    };
    check(st, "get property")?;
    // SAFETY: the HAL wrote `size` bytes of `T`s into the capacity we reserved.
    unsafe { v.set_len(size as usize / size_of::<T>()) };
    Ok(v)
}

fn set<T: Copy>(
    object: AudioObjectID,
    selector: u32,
    scope: u32,
    value: &T,
) -> Result<(), OutputError> {
    let a = addr(selector, scope);
    let st = unsafe {
        AudioObjectSetPropertyData(
            object,
            &a,
            0,
            ptr::null(),
            size_of::<T>() as u32,
            value as *const T as *const c_void,
        )
    };
    check(st, "set property")
}

/// A CFString property as a Rust string (the CFString is released).
fn get_string(object: AudioObjectID, selector: u32) -> Option<String> {
    let cf: *const c_void = get(object, selector, K_SCOPE_GLOBAL).ok()?;
    if cf.is_null() {
        return None;
    }
    let len = unsafe { CFStringGetLength(cf) };
    let mut buf = vec![0u8; (len as usize) * 4 + 1];
    let ok = unsafe { CFStringGetCString(cf, buf.as_mut_ptr(), buf.len() as isize, CF_UTF8) };
    unsafe { CFRelease(cf) };
    if ok == 0 {
        return None;
    }
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    Some(String::from_utf8_lossy(&buf[..end]).into_owned())
}

fn output_streams(device: AudioObjectID) -> Vec<AudioObjectID> {
    get_vec::<AudioObjectID>(device, K_DEV_STREAMS, K_SCOPE_OUTPUT).unwrap_or_default()
}

fn output_channels(device: AudioObjectID) -> Option<u32> {
    // kAudioDevicePropertyStreamConfiguration is an AudioBufferList; sum the channel counts.
    let a = addr(K_DEV_STREAM_CONFIG, K_SCOPE_OUTPUT);
    let mut size = 0u32;
    if unsafe { AudioObjectGetPropertyDataSize(device, &a, 0, ptr::null(), &mut size) } != 0 {
        return None;
    }
    let mut raw = vec![0u8; size as usize];
    if unsafe {
        AudioObjectGetPropertyData(
            device,
            &a,
            0,
            ptr::null(),
            &mut size,
            raw.as_mut_ptr() as *mut c_void,
        )
    } != 0
    {
        return None;
    }
    // SAFETY: the HAL filled an AudioBufferList with mNumberBuffers trailing AudioBuffers.
    let list = unsafe { &*(raw.as_ptr() as *const AudioBufferList) };
    let n = list.mNumberBuffers as usize;
    let buffers = unsafe { std::slice::from_raw_parts(list.mBuffers.as_ptr(), n) };
    Some(buffers.iter().map(|b| b.mNumberChannels).sum())
}

fn available_rates(device: AudioObjectID) -> Vec<u32> {
    let ranges: Vec<ValueRange> =
        get_vec(device, K_DEV_AVAILABLE_RATES, K_SCOPE_GLOBAL).unwrap_or_default();
    STANDARD_RATES
        .iter()
        .copied()
        .filter(|r| {
            ranges
                .iter()
                .any(|g| g.minimum <= *r as f64 && *r as f64 <= g.maximum)
        })
        .collect()
}

/// Physical formats of the first output stream.
fn physical_formats(stream: AudioObjectID) -> Vec<RangedDescription> {
    get_vec(stream, K_STREAM_AVAILABLE_PHYSICAL_FORMATS, K_SCOPE_GLOBAL).unwrap_or_default()
}

/// Integer bit depths the stream can be switched to at `rate`.
fn integer_depths(formats: &[RangedDescription], rate: u32) -> Vec<u32> {
    let mut v: Vec<u32> = formats
        .iter()
        .filter(|f| {
            f.format.mFormatID == K_FORMAT_LINEAR_PCM
                && f.format.mFormatFlags & FLAG_IS_SIGNED_INTEGER != 0
                && f.format.mFormatFlags & FLAG_IS_NON_INTERLEAVED == 0
                && (f.format.mSampleRate == rate as f64
                    || (f.rate_range.minimum <= rate as f64 && rate as f64 <= f.rate_range.maximum))
        })
        .map(|f| f.format.mBitsPerChannel)
        .filter(|b| matches!(b, 16 | 24 | 32))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Pick the physical format for `rate`/`bits`: an interleaved integer format with
/// exactly `bits` valid bits, else any 32-bit float format at that rate.
fn pick_physical(
    formats: &[RangedDescription],
    rate: u32,
    bits: u32,
) -> Option<AudioStreamBasicDescription> {
    let at_rate = |f: &RangedDescription| {
        f.format.mFormatID == K_FORMAT_LINEAR_PCM
            && f.format.mFormatFlags & FLAG_IS_NON_INTERLEAVED == 0
            && (f.format.mSampleRate == rate as f64
                || (f.rate_range.minimum <= rate as f64 && rate as f64 <= f.rate_range.maximum))
    };
    let int = formats.iter().find(|f| {
        at_rate(f)
            && f.format.mFormatFlags & FLAG_IS_SIGNED_INTEGER != 0
            && f.format.mBitsPerChannel == bits
    });
    let chosen = int.or_else(|| {
        formats.iter().find(|f| {
            at_rate(f)
                && f.format.mFormatFlags & FLAG_IS_FLOAT != 0
                && f.format.mBitsPerChannel == 32
        })
    })?;
    let mut d = chosen.format;
    d.mSampleRate = rate as f64;
    Some(d)
}

fn all_devices() -> Vec<AudioObjectID> {
    get_vec(K_SYSTEM_OBJECT, K_HW_DEVICES, K_SCOPE_GLOBAL).unwrap_or_default()
}

fn default_output() -> Option<AudioObjectID> {
    get::<AudioObjectID>(K_SYSTEM_OBJECT, K_HW_DEFAULT_OUTPUT, K_SCOPE_GLOBAL).ok()
}

/// Find a device by UID or name (empty = default output).
fn find_device(id: &str) -> Result<(AudioObjectID, String), OutputError> {
    if id.is_empty() || id == "default" {
        let d = default_output()
            .ok_or_else(|| OutputError::NoDevice("no default output device".into()))?;
        return Ok((
            d,
            get_string(d, K_OBJ_NAME).unwrap_or_else(|| "CoreAudio".into()),
        ));
    }
    for d in all_devices() {
        let uid = get_string(d, K_DEV_UID).unwrap_or_default();
        let name = get_string(d, K_OBJ_NAME).unwrap_or_default();
        if uid == id || name == id {
            return Ok((d, name));
        }
    }
    Err(OutputError::NoDevice(id.to_string()))
}

/// Every output device with its rates, integer depths and DoP reach.
pub fn list_devices() -> Vec<DeviceInfo> {
    let default = default_output();
    let mut out = Vec::new();
    for d in all_devices() {
        let streams = output_streams(d);
        if streams.is_empty() {
            continue;
        }
        let rates = available_rates(d);
        let formats = physical_formats(streams[0]);
        let mut bits: Vec<u32> = rates
            .iter()
            .flat_map(|r| integer_depths(&formats, *r))
            .collect();
        bits.sort_unstable();
        bits.dedup();
        if bits.is_empty() {
            // Float-only device: we convert, so every depth "works" but is not integer mode.
            bits = vec![16, 24, 32];
        }
        let dop_max = DsdRate::catalog()
            .iter()
            .filter(|r| r.dop_possible())
            .flat_map(|r| [r.hz(44_100), r.hz(48_000)])
            .filter(|hz| rates.contains(&(hz / 16)))
            .max();
        out.push(DeviceInfo {
            backend: "coreaudio".into(),
            id: get_string(d, K_DEV_UID).unwrap_or_else(|| d.to_string()),
            name: get_string(d, K_OBJ_NAME).unwrap_or_else(|| format!("device {d}")),
            rates,
            bits,
            channels: output_channels(d),
            default: default == Some(d),
            exclusive: true,
            dsd: DsdCaps::dop_only(dop_max),
        });
    }
    out
}

/// Shared with the IOProc.
struct Ring {
    queue: VecDeque<i32>,
    paused: bool,
    underruns: u64,
    /// 0 = write integers of `bits` valid bits into 32-bit words (left-aligned when
    /// the physical format says so), 16 = i16 words, 32 with `float` = f32.
    store_bits: u32,
    float: bool,
    aligned_high: bool,
    valid_bits: u32,
}

/// See the module documentation.
pub struct CoreAudioOutput {
    device: AudioObjectID,
    device_name: String,
    format: Option<OutputFormat>,
    ring: Arc<(Mutex<Ring>, Condvar)>,
    proc_id: AudioDeviceIOProcID,
    hogged: bool,
    integer_mode: bool,
    high_water: usize,
    /// Physical format before we touched it, to restore on close.
    saved_format: Option<(AudioObjectID, AudioStreamBasicDescription)>,
    saved_rate: Option<f64>,
}

// The HAL calls the IOProc from its own thread with our ring; the object itself is only
// touched by the engine thread.
unsafe impl Send for CoreAudioOutput {}

/// The IOProc: fill the output buffers from the ring, converting to the physical format.
unsafe extern "C" fn io_proc(
    _device: AudioObjectID,
    _now: *const AudioTimeStamp,
    _input: *const AudioBufferList,
    _input_time: *const AudioTimeStamp,
    output: *mut AudioBufferList,
    _output_time: *const AudioTimeStamp,
    client: *mut c_void,
) -> OSStatus {
    if output.is_null() || client.is_null() {
        return 0;
    }
    let ring = &*(client as *const (Mutex<Ring>, Condvar));
    let list = &mut *output;
    let n = list.mNumberBuffers as usize;
    let buffers = std::slice::from_raw_parts_mut(list.mBuffers.as_mut_ptr(), n);
    // try_lock: never block the real-time thread; a contended lock means silence once.
    let Ok(mut r) = ring.0.try_lock() else {
        for b in buffers.iter_mut() {
            ptr::write_bytes(b.mData as *mut u8, 0, b.mDataByteSize as usize);
        }
        return 0;
    };
    for b in buffers.iter_mut() {
        let bytes = b.mDataByteSize as usize;
        if r.paused {
            ptr::write_bytes(b.mData as *mut u8, 0, bytes);
            continue;
        }
        let shift = 32 - r.valid_bits;
        if r.float {
            let out = std::slice::from_raw_parts_mut(b.mData as *mut f32, bytes / 4);
            let scale = 1.0 / (1u64 << (r.valid_bits - 1)) as f32;
            for o in out.iter_mut() {
                *o = match r.queue.pop_front() {
                    Some(v) => v as f32 * scale,
                    None => {
                        r.underruns += 1;
                        0.0
                    }
                };
            }
        } else if r.store_bits == 16 {
            let out = std::slice::from_raw_parts_mut(b.mData as *mut i16, bytes / 2);
            for o in out.iter_mut() {
                *o = match r.queue.pop_front() {
                    Some(v) => v as i16,
                    None => {
                        r.underruns += 1;
                        0
                    }
                };
            }
        } else {
            let out = std::slice::from_raw_parts_mut(b.mData as *mut i32, bytes / 4);
            for o in out.iter_mut() {
                let v = match r.queue.pop_front() {
                    Some(v) => v,
                    None => {
                        r.underruns += 1;
                        0
                    }
                };
                *o = if r.aligned_high { v << shift } else { v };
            }
        }
    }
    drop(r);
    ring.1.notify_all();
    0
}

impl CoreAudioOutput {
    /// Prepare for `device` (UID or name as printed by `cadenza devices`; empty = default).
    pub fn new(device: &str) -> Result<Self, OutputError> {
        let (id, device_name) = find_device(device)?;
        Ok(Self {
            device: id,
            device_name,
            format: None,
            ring: Arc::new((
                Mutex::new(Ring {
                    queue: VecDeque::new(),
                    paused: false,
                    underruns: 0,
                    store_bits: 32,
                    float: false,
                    aligned_high: false,
                    valid_bits: 24,
                }),
                Condvar::new(),
            )),
            proc_id: None,
            hogged: false,
            integer_mode: false,
            high_water: 0,
            saved_format: None,
            saved_rate: None,
        })
    }

    /// Whether the physical format is an integer one (bit-perfect to the driver).
    pub fn integer_mode(&self) -> bool {
        self.integer_mode
    }

    fn hog(&mut self, on: bool) -> Result<(), OutputError> {
        // kAudioDevicePropertyHogMode is a pid_t: -1 = free, our pid = hogged by us.
        let want: i32 = if on { std::process::id() as i32 } else { -1 };
        set(self.device, K_DEV_HOG_MODE, K_SCOPE_GLOBAL, &want)?;
        self.hogged = on;
        Ok(())
    }

    fn wait_rate(&self, rate: f64) -> bool {
        // Rate switches are asynchronous; poll briefly for the device to settle.
        for _ in 0..50 {
            if get::<f64>(self.device, K_DEV_NOMINAL_RATE, K_SCOPE_GLOBAL).ok() == Some(rate) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }
}

impl Output for CoreAudioOutput {
    fn name(&self) -> String {
        format!(
            "coreaudio hog · {}{}",
            self.device_name,
            if self.integer_mode { " · integer" } else { "" }
        )
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        self.close();
        if matches!(
            format.encoding,
            Encoding::Dsd {
                packing: DsdPacking::Native,
                ..
            }
        ) {
            return Err(OutputError::Unsupported {
                device: self.device_name.clone(),
                format,
            });
        }
        let streams = output_streams(self.device);
        let stream = *streams
            .first()
            .ok_or_else(|| OutputError::NoDevice(self.device_name.clone()))?;
        let formats = physical_formats(stream);
        let physical =
            pick_physical(&formats, format.rate, format.bits).ok_or(OutputError::Unsupported {
                device: self.device_name.clone(),
                format,
            })?;

        // 1. Take the device, 2. switch the rate, 3. switch the physical format.
        if let Err(e) = self.hog(true) {
            log::warn!("coreaudio: hog mode refused ({e}); continuing shared");
        }
        self.saved_rate = get::<f64>(self.device, K_DEV_NOMINAL_RATE, K_SCOPE_GLOBAL).ok();
        if self.saved_rate != Some(format.rate as f64) {
            set(
                self.device,
                K_DEV_NOMINAL_RATE,
                K_SCOPE_GLOBAL,
                &(format.rate as f64),
            )?;
            if !self.wait_rate(format.rate as f64) {
                return Err(OutputError::Unsupported {
                    device: self.device_name.clone(),
                    format,
                });
            }
        }
        let current: AudioStreamBasicDescription =
            get(stream, K_STREAM_PHYSICAL_FORMAT, K_SCOPE_GLOBAL)?;
        self.saved_format = Some((stream, current));
        if let Err(e) = set(stream, K_STREAM_PHYSICAL_FORMAT, K_SCOPE_GLOBAL, &physical) {
            log::warn!("coreaudio: physical format switch refused ({e}); using the current one");
        }
        let actual: AudioStreamBasicDescription =
            get(stream, K_STREAM_PHYSICAL_FORMAT, K_SCOPE_GLOBAL)?;
        self.integer_mode = actual.mFormatFlags & FLAG_IS_SIGNED_INTEGER != 0;
        let store_bits = actual.mBytesPerFrame * 8 / actual.mChannelsPerFrame.max(1);
        {
            let mut r = self.ring.0.lock().unwrap();
            r.queue.clear();
            r.paused = false;
            r.float = actual.mFormatFlags & FLAG_IS_FLOAT != 0;
            r.store_bits = store_bits;
            r.aligned_high = actual.mFormatFlags & FLAG_IS_ALIGNED_HIGH != 0
                || (self.integer_mode && actual.mBitsPerChannel == 32 && format.bits < 32);
            r.valid_bits = format.bits;
        }
        if actual.mChannelsPerFrame != format.channels {
            return Err(OutputError::Unsupported {
                device: self.device_name.clone(),
                format,
            });
        }
        // Keep roughly 20 ms per callback.
        let frames: u32 = (format.rate / 50).clamp(64, 4096);
        let _ = set(
            self.device,
            K_DEV_BUFFER_FRAME_SIZE,
            K_SCOPE_GLOBAL,
            &frames,
        );

        let mut proc_id: AudioDeviceIOProcID = None;
        let client = Arc::as_ptr(&self.ring) as *mut c_void;
        let st =
            unsafe { AudioDeviceCreateIOProcID(self.device, Some(io_proc), client, &mut proc_id) };
        check(st, "create IOProc")?;
        self.proc_id = proc_id;
        check(unsafe { AudioDeviceStart(self.device, proc_id) }, "start")?;
        self.format = Some(format);
        self.high_water = (format.rate as usize / 2) * format.channels as usize;
        let _ = FLAG_IS_PACKED;
        Ok(format)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        if !matches!(bits, 16 | 24 | 32) {
            return false;
        }
        if !available_rates(self.device).contains(&rate) {
            return false;
        }
        let Some(stream) = output_streams(self.device).first().copied() else {
            return false;
        };
        pick_physical(&physical_formats(stream), rate, bits).is_some()
    }

    fn dsd_caps(&self) -> DsdCaps {
        let rates = available_rates(self.device);
        let max = DsdRate::catalog()
            .iter()
            .filter(|r| r.dop_possible())
            .flat_map(|r| [r.hz(44_100), r.hz(48_000)])
            .filter(|hz| rates.contains(&(hz / 16)) && self.supports(hz / 16, 24))
            .max();
        DsdCaps::dop_only(max)
    }

    fn exclusive(&self) -> bool {
        true
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        if self.format.is_none() {
            return Err(OutputError::Backend("coreaudio device not open".into()));
        }
        let (lock, cv) = &*self.ring;
        let mut ring = lock.lock().unwrap();
        while ring.queue.len() > self.high_water {
            ring = cv.wait_timeout(ring, Duration::from_millis(100)).unwrap().0;
        }
        ring.queue.extend(frames.iter().copied());
        Ok(())
    }

    fn pause(&mut self, paused: bool) -> Result<(), OutputError> {
        self.ring.0.lock().unwrap().paused = paused;
        Ok(())
    }

    fn drain(&mut self) -> Result<(), OutputError> {
        let (lock, cv) = &*self.ring;
        let mut ring = lock.lock().unwrap();
        let mut waited = 0;
        while !ring.queue.is_empty() && waited < 50 {
            ring = cv.wait_timeout(ring, Duration::from_millis(100)).unwrap().0;
            waited += 1;
        }
        Ok(())
    }

    fn discard(&mut self) -> Result<(), OutputError> {
        self.ring.0.lock().unwrap().queue.clear();
        Ok(())
    }

    fn close(&mut self) {
        if let Some(id) = self.proc_id.take() {
            unsafe {
                let _ = AudioDeviceStop(self.device, Some(id));
                let _ = AudioDeviceDestroyIOProcID(self.device, Some(id));
            }
        }
        if let Some((stream, desc)) = self.saved_format.take() {
            let _ = set(stream, K_STREAM_PHYSICAL_FORMAT, K_SCOPE_GLOBAL, &desc);
        }
        if let Some(rate) = self.saved_rate.take() {
            let _ = set(self.device, K_DEV_NOMINAL_RATE, K_SCOPE_GLOBAL, &rate);
        }
        if self.hogged {
            let _ = self.hog(false);
        }
        self.format = None;
        self.integer_mode = false;
        let mut ring = self.ring.0.lock().unwrap();
        ring.queue.clear();
        ring.paused = false;
    }

    fn queued_frames(&self) -> usize {
        let ch = self.format.map(|f| f.channels as usize).unwrap_or(2).max(1);
        self.ring.0.lock().unwrap().queue.len() / ch
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }
}

impl Drop for CoreAudioOutput {
    fn drop(&mut self) {
        self.close();
    }
}
