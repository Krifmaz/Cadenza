//! The Windows loader: installed ASIO drivers are listed under `HKLM\SOFTWARE\ASIO`
//! (one subkey per driver, its COM class in the `CLSID` value) and created in-process
//! with `CoCreateInstance`, asking for the class id as the interface id – an ASIO
//! convention, since the interface has no registered IID of its own.
//!
//! Only drivers matching the program's bitness are visible (a 64-bit Cadenza sees the
//! 64-bit drivers; 32-bit ones live under `WOW6432Node` and could not be loaded anyway).
//! COM is initialised single-threaded on the thread that loads the driver, which is what
//! most drivers expect; the output must then be used from that same thread.

use super::host::{AsioError, Driver};
use super::{AsioOutput, DriverSource};
use crate::output::{DeviceInfo, OutputError};
use std::ffi::c_void;
use std::sync::Arc;
use windows_sys::core::GUID;
use windows_sys::Win32::Foundation::{ERROR_SUCCESS, RPC_E_CHANGED_MODE};
use windows_sys::Win32::System::Com::{
    CLSIDFromString, CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER,
    COINIT_APARTMENTTHREADED,
};
use windows_sys::Win32::System::Registry::{
    RegCloseKey, RegEnumKeyExW, RegOpenKeyExW, RegQueryValueExW, HKEY, HKEY_LOCAL_MACHINE,
    KEY_READ, REG_SZ,
};
use windows_sys::Win32::UI::WindowsAndMessaging::GetDesktopWindow;

const ASIO_KEY: &str = "SOFTWARE\\ASIO";

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn from_wide(w: &[u16]) -> String {
    let end = w.iter().position(|&c| c == 0).unwrap_or(w.len());
    String::from_utf16_lossy(&w[..end])
}

/// A registry key, closed on drop.
struct Key(HKEY);

impl Key {
    fn open(parent: HKEY, path: &str) -> Option<Key> {
        let mut key: HKEY = std::ptr::null_mut();
        let p = wide(path);
        // SAFETY: valid NUL-terminated path and out pointer.
        let r = unsafe { RegOpenKeyExW(parent, p.as_ptr(), 0, KEY_READ, &mut key) };
        (r == ERROR_SUCCESS).then_some(Key(key))
    }

    fn subkeys(&self) -> Vec<String> {
        let mut out = Vec::new();
        for i in 0.. {
            let mut name = [0u16; 256];
            let mut len = name.len() as u32;
            // SAFETY: buffer and length describe `name`; the other outputs are optional.
            let r = unsafe {
                RegEnumKeyExW(
                    self.0,
                    i,
                    name.as_mut_ptr(),
                    &mut len,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
            if r != ERROR_SUCCESS {
                break;
            }
            out.push(from_wide(&name[..len as usize]));
        }
        out
    }

    fn string(&self, value: &str) -> Option<String> {
        let v = wide(value);
        let mut data = [0u16; 128];
        let mut bytes = (data.len() * 2) as u32;
        let mut ty = 0;
        // SAFETY: the data buffer is `bytes` long.
        let r = unsafe {
            RegQueryValueExW(
                self.0,
                v.as_ptr(),
                std::ptr::null(),
                &mut ty,
                data.as_mut_ptr() as *mut u8,
                &mut bytes,
            )
        };
        (r == ERROR_SUCCESS && ty == REG_SZ).then(|| from_wide(&data[..bytes as usize / 2]))
    }
}

impl Drop for Key {
    fn drop(&mut self) {
        // SAFETY: an open key.
        unsafe { RegCloseKey(self.0) };
    }
}

fn init_com() {
    // S_FALSE (already initialised) and RPC_E_CHANGED_MODE (the thread is already in
    // the multithreaded apartment, e.g. WASAPI was used here) both leave COM usable.
    // SAFETY: plain call.
    let r = unsafe { CoInitializeEx(std::ptr::null(), COINIT_APARTMENTTHREADED as u32) };
    if r < 0 && r != RPC_E_CHANGED_MODE {
        log::warn!("ASIO: CoInitializeEx failed ({r:#x})");
    }
}

/// The installed drivers, from the registry.
pub struct RegistrySource;

impl DriverSource for RegistrySource {
    fn names(&self) -> Vec<String> {
        Key::open(HKEY_LOCAL_MACHINE, ASIO_KEY)
            .map(|k| k.subkeys())
            .unwrap_or_default()
    }

    fn load(&self, name: &str) -> Result<Driver, AsioError> {
        init_com();
        let key = Key::open(HKEY_LOCAL_MACHINE, &format!("{ASIO_KEY}\\{name}"))
            .ok_or_else(|| AsioError::host("load", format!("{name}: not in the registry")))?;
        let clsid_text = key
            .string("CLSID")
            .ok_or_else(|| AsioError::host("load", format!("{name}: no CLSID")))?;
        let mut clsid: GUID = unsafe { std::mem::zeroed() };
        let w = wide(&clsid_text);
        // SAFETY: valid string and out pointer.
        if unsafe { CLSIDFromString(w.as_ptr(), &mut clsid) } < 0 {
            return Err(AsioError::host(
                "load",
                format!("{name}: bad CLSID {clsid_text}"),
            ));
        }
        let mut obj: *mut c_void = std::ptr::null_mut();
        // SAFETY: the class id doubles as the interface id (ASIO convention).
        let hr = unsafe {
            CoCreateInstance(
                &clsid,
                std::ptr::null_mut(),
                CLSCTX_INPROC_SERVER,
                &clsid,
                &mut obj,
            )
        };
        if hr < 0 || obj.is_null() {
            return Err(AsioError::host(
                "load",
                format!("{name}: CoCreateInstance failed ({hr:#x}); is the driver 64-bit?"),
            ));
        }
        // SAFETY: CoCreateInstance returned an owned reference to the driver object.
        unsafe { Driver::from_raw(obj as *mut super::sys::IAsio, name) }
            .ok_or_else(|| AsioError::host("load", "null driver"))
    }

    fn sys_handle(&self) -> isize {
        // SAFETY: plain call.
        unsafe { GetDesktopWindow() as isize }
    }
}

/// Open driver `name` (`""`: the first installed one).
pub fn open(name: &str) -> Result<AsioOutput, OutputError> {
    Ok(AsioOutput::with_source(Arc::new(RegistrySource), name)?.hold())
}

/// The installed drivers, probed afresh on every listing (so one plugged in later shows
/// up), except the one playback holds: that one is listed from what it reported when it
/// opened, never loaded a second time.
pub fn list_devices() -> Vec<DeviceInfo> {
    let src: Arc<dyn DriverSource> = Arc::new(RegistrySource);
    super::list_devices_from(&src)
}
