//! The binary interface of an ASIO driver, written from the published interface facts
//! (method order, structure layouts, constants); no Steinberg header is used or shipped.
//!
//! A driver is an in-process COM object whose interface pointer leads to a table of 24
//! methods: the three of `IUnknown`, then the ASIO ones below in this exact order. `long`
//! is 32 bits on Windows, `ASIOBool` is a `long`, sample rates are `double`, and the
//! structures are packed to 4 bytes. On 64-bit Windows the methods use the platform's
//! single calling convention; 32-bit drivers use `thiscall` (the `this` pointer in ECX).

use std::ffi::c_void;

/// `ASIOError` success value.
pub const ASE_OK: i32 = 0;
/// `ASIOFuture` success value (future calls return this, not `ASE_OK`).
pub const ASE_SUCCESS: i32 = 0x3f48_47a0;
/// The requested feature or channel is not present.
pub const ASE_NOT_PRESENT: i32 = -1000;
/// Hardware malfunction.
pub const ASE_HW_MALFUNCTION: i32 = -999;
/// A parameter was invalid.
pub const ASE_INVALID_PARAMETER: i32 = -998;
/// The call is not valid in the driver's current state.
pub const ASE_INVALID_MODE: i32 = -997;
/// The sample position does not advance (hardware stopped).
pub const ASE_SP_NOT_ADVANCING: i32 = -996;
/// No clock.
pub const ASE_NO_CLOCK: i32 = -995;
/// Out of memory.
pub const ASE_NO_MEMORY: i32 = -994;

/// A readable name for an `ASIOError`.
pub fn error_name(e: i32) -> &'static str {
    match e {
        ASE_OK | ASE_SUCCESS => "ok",
        ASE_NOT_PRESENT => "not present",
        ASE_HW_MALFUNCTION => "hardware malfunction",
        ASE_INVALID_PARAMETER => "invalid parameter",
        ASE_INVALID_MODE => "invalid mode",
        ASE_SP_NOT_ADVANCING => "sample position not advancing",
        ASE_NO_CLOCK => "no clock",
        ASE_NO_MEMORY => "out of memory",
        _ => "unknown error",
    }
}

// Sample types (`ASIOSampleType`).
/// 16-bit big endian.
pub const ST_INT16_MSB: i32 = 0;
/// 24-bit packed big endian.
pub const ST_INT24_MSB: i32 = 1;
/// 32-bit big endian.
pub const ST_INT32_MSB: i32 = 2;
/// 32-bit float big endian.
pub const ST_FLOAT32_MSB: i32 = 3;
/// 64-bit float big endian.
pub const ST_FLOAT64_MSB: i32 = 4;
/// 32-bit container, 16 valid bits, right-aligned, big endian.
pub const ST_INT32_MSB16: i32 = 8;
/// 32-bit container, 24 valid bits, right-aligned, big endian.
pub const ST_INT32_MSB24: i32 = 11;
/// 16-bit little endian.
pub const ST_INT16_LSB: i32 = 16;
/// 24-bit packed little endian.
pub const ST_INT24_LSB: i32 = 17;
/// 32-bit little endian.
pub const ST_INT32_LSB: i32 = 18;
/// 32-bit float little endian.
pub const ST_FLOAT32_LSB: i32 = 19;
/// 64-bit float little endian.
pub const ST_FLOAT64_LSB: i32 = 20;
/// 32-bit container, 16 valid bits, right-aligned, little endian.
pub const ST_INT32_LSB16: i32 = 24;
/// 32-bit container, 18 valid bits.
pub const ST_INT32_LSB18: i32 = 25;
/// 32-bit container, 20 valid bits.
pub const ST_INT32_LSB20: i32 = 26;
/// 32-bit container, 24 valid bits, right-aligned, little endian.
pub const ST_INT32_LSB24: i32 = 27;
/// DSD, 8 one-bit samples per byte, the first in the least significant bit.
pub const ST_DSD_INT8_LSB1: i32 = 32;
/// DSD, 8 one-bit samples per byte, the first in the most significant bit.
pub const ST_DSD_INT8_MSB1: i32 = 33;
/// DSD, one sample per byte.
pub const ST_DSD_INT8_NER8: i32 = 40;

// `ASIOFuture` selectors used here.
/// Switch the driver between PCM and DSD (`*mut IoFormat`).
pub const FUTURE_SET_IO_FORMAT: i32 = 0x2311_1961;
/// Read the current I/O format (`*mut IoFormat`).
pub const FUTURE_GET_IO_FORMAT: i32 = 0x2311_1983;
/// Ask whether a format is possible (`*mut IoFormat`).
pub const FUTURE_CAN_DO_IO_FORMAT: i32 = 0x2311_2004;

/// `ASIOIoFormatType`: PCM.
pub const IO_FORMAT_PCM: i32 = 0;
/// `ASIOIoFormatType`: DSD.
pub const IO_FORMAT_DSD: i32 = 1;

/// `ASIOIoFormat`: the format type followed by reserved space, 512 bytes in all.
#[repr(C, packed(4))]
#[derive(Clone, Copy)]
pub struct IoFormat {
    /// `IO_FORMAT_PCM` or `IO_FORMAT_DSD` (`-1` when the driver refused it).
    pub format_type: i32,
    /// Reserved, zero.
    pub future: [u8; 508],
}

impl IoFormat {
    /// A request for `format_type`.
    pub fn new(format_type: i32) -> Self {
        Self {
            format_type,
            future: [0; 508],
        }
    }
}

// Host messages (`asioMessage` selectors).
/// "Do you support selector `value`?"
pub const MSG_SELECTOR_SUPPORTED: i32 = 1;
/// "Which ASIO version is the host?"
pub const MSG_ENGINE_VERSION: i32 = 2;
/// The driver needs to be reset (stop, dispose, recreate the buffers).
pub const MSG_RESET_REQUEST: i32 = 3;
/// The buffer size changed (unused by drivers in practice).
pub const MSG_BUFFER_SIZE_CHANGE: i32 = 4;
/// The driver lost sync; the host should resync.
pub const MSG_RESYNC_REQUEST: i32 = 5;
/// The latencies changed.
pub const MSG_LATENCIES_CHANGED: i32 = 6;
/// "Do you support `bufferSwitchTimeInfo`?"
pub const MSG_SUPPORTS_TIME_INFO: i32 = 7;
/// "Do you support time code?"
pub const MSG_SUPPORTS_TIME_CODE: i32 = 8;
/// The driver detected an overload.
pub const MSG_OVERLOAD: i32 = 15;

/// `ASIOChannelInfo`.
#[repr(C, packed(4))]
#[derive(Clone, Copy)]
pub struct ChannelInfo {
    /// Channel index (in).
    pub channel: i32,
    /// `1` for an input (in).
    pub is_input: i32,
    /// Out: the channel is active.
    pub is_active: i32,
    /// Out: channel group.
    pub channel_group: i32,
    /// Out: sample type.
    pub sample_type: i32,
    /// Out: name, NUL-terminated.
    pub name: [u8; 32],
}

/// `ASIOBufferInfo`.
#[repr(C, packed(4))]
#[derive(Clone, Copy)]
pub struct BufferInfo {
    /// `1` for an input.
    pub is_input: i32,
    /// Channel index.
    pub channel: i32,
    /// Out: the two halves of the double buffer.
    pub buffers: [*mut c_void; 2],
}

/// `ASIOSamples` / `ASIOTimeStamp`: a 64-bit value as two 32-bit halves.
#[repr(C, packed(4))]
#[derive(Clone, Copy, Default)]
pub struct Int64Parts {
    /// High 32 bits.
    pub hi: u32,
    /// Low 32 bits.
    pub lo: u32,
}

/// `ASIOTime` (only passed through: this host does not use time info).
#[repr(C)]
pub struct AsioTime {
    _opaque: [u8; 0],
}

/// The calling convention of the driver's methods: `thiscall` for 32-bit x86, the
/// platform's C convention everywhere else (64-bit Windows has only one).
#[cfg(target_arch = "x86")]
macro_rules! method {
    ($($arg:ty),* => $ret:ty) => { unsafe extern "thiscall" fn(*mut IAsio $(, $arg)*) -> $ret };
    ($($arg:ty),*) => { unsafe extern "thiscall" fn(*mut IAsio $(, $arg)*) };
}
#[cfg(not(target_arch = "x86"))]
macro_rules! method {
    ($($arg:ty),* => $ret:ty) => { unsafe extern "system" fn(*mut IAsio $(, $arg)*) -> $ret };
    ($($arg:ty),*) => { unsafe extern "system" fn(*mut IAsio $(, $arg)*) };
}

/// The driver's method table, in interface order.
#[repr(C)]
pub struct IAsioVtbl {
    /// `IUnknown::QueryInterface`.
    pub query_interface: method!(*const c_void, *mut *mut c_void => i32),
    /// `IUnknown::AddRef`.
    pub add_ref: method!( => u32),
    /// `IUnknown::Release`.
    pub release: method!( => u32),
    /// `init(sysHandle)` → `ASIOBool`.
    pub init: method!(*mut c_void => i32),
    /// `getDriverName(char[32])`.
    pub get_driver_name: method!(*mut u8),
    /// `getDriverVersion()`.
    pub get_driver_version: method!( => i32),
    /// `getErrorMessage(char[124])`.
    pub get_error_message: method!(*mut u8),
    /// `start()`.
    pub start: method!( => i32),
    /// `stop()`.
    pub stop: method!( => i32),
    /// `getChannels(&inputs, &outputs)`.
    pub get_channels: method!(*mut i32, *mut i32 => i32),
    /// `getLatencies(&input, &output)`.
    pub get_latencies: method!(*mut i32, *mut i32 => i32),
    /// `getBufferSize(&min, &max, &preferred, &granularity)`.
    pub get_buffer_size: method!(*mut i32, *mut i32, *mut i32, *mut i32 => i32),
    /// `canSampleRate(rate)`.
    pub can_sample_rate: method!(f64 => i32),
    /// `getSampleRate(&rate)`.
    pub get_sample_rate: method!(*mut f64 => i32),
    /// `setSampleRate(rate)`.
    pub set_sample_rate: method!(f64 => i32),
    /// `getClockSources(clocks, &count)` (not used).
    pub get_clock_sources: method!(*mut c_void, *mut i32 => i32),
    /// `setClockSource(index)` (not used).
    pub set_clock_source: method!(i32 => i32),
    /// `getSamplePosition(&samples, &timestamp)`.
    pub get_sample_position: method!(*mut Int64Parts, *mut Int64Parts => i32),
    /// `getChannelInfo(&info)`.
    pub get_channel_info: method!(*mut ChannelInfo => i32),
    /// `createBuffers(infos, count, bufferSize, &callbacks)`.
    pub create_buffers: method!(*mut BufferInfo, i32, i32, *mut Callbacks => i32),
    /// `disposeBuffers()`.
    pub dispose_buffers: method!( => i32),
    /// `controlPanel()`.
    pub control_panel: method!( => i32),
    /// `future(selector, params)`.
    pub future: method!(i32, *mut c_void => i32),
    /// `outputReady()`.
    pub output_ready: method!( => i32),
}

/// An ASIO driver object: a pointer to its method table.
#[repr(C)]
pub struct IAsio {
    /// The method table.
    pub vtbl: *const IAsioVtbl,
}

/// The host callbacks handed to `createBuffers` (plain C functions on every platform).
#[repr(C)]
pub struct Callbacks {
    /// `bufferSwitch(index, directProcess)`.
    pub buffer_switch: extern "C" fn(i32, i32),
    /// `sampleRateDidChange(rate)`.
    pub sample_rate_did_change: extern "C" fn(f64),
    /// `asioMessage(selector, value, message, opt)`.
    pub asio_message: extern "C" fn(i32, i32, *mut c_void, *mut f64) -> i32,
    /// `bufferSwitchTimeInfo(params, index, directProcess)`.
    pub buffer_switch_time_info: extern "C" fn(*mut AsioTime, i32, i32) -> *mut AsioTime,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn layouts_match_the_driver_abi() {
        assert_eq!(size_of::<IoFormat>(), 512);
        assert_eq!(size_of::<ChannelInfo>(), 52);
        assert_eq!(offset_of!(ChannelInfo, sample_type), 16);
        assert_eq!(offset_of!(ChannelInfo, name), 20);
        assert_eq!(size_of::<Int64Parts>(), 8);
        // Two longs and two pointers.
        assert_eq!(size_of::<BufferInfo>(), 8 + 2 * size_of::<usize>());
        assert_eq!(offset_of!(BufferInfo, buffers), 8);
        // IUnknown's three methods and the 21 of ASIO.
        assert_eq!(size_of::<IAsioVtbl>(), 24 * size_of::<usize>());
        assert_eq!(size_of::<Callbacks>(), 4 * size_of::<usize>());
        assert_eq!(error_name(ASE_NOT_PRESENT), "not present");
        assert_eq!(ASE_NO_MEMORY, -994);
    }
}
