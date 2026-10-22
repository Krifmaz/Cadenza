//! The Windows system input: process loopback (Windows 10 2004 / build 19041 and later).
//!
//! * `app`: the audio of one application and the processes it started, captured before the
//!   system mixer combines it with other apps (`PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE`).
//! * `sink`: Windows has no virtual device without a signed driver, so this mode takes
//!   *everything except Cadenza* (`…EXCLUDE_TARGET_PROCESS_TREE` on our own process). Make
//!   another device the Windows default (or mute it) so the sound is not heard twice, and
//!   let Cadenza play to the DAC in exclusive mode.
//!
//! Process loopback delivers 32-bit float at the rate we ask for. We ask for the default
//! output device's shared-mode rate, which is the rate Windows mixes the app at anyway: set
//! that device's format (Sound settings → Advanced) to the source rate for a path without
//! resampling, and keep the app's volume at 100 %.

use super::{Feed, SystemConfig, SystemMode};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
use windows::core::{implement, Interface, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::{
    eConsole, eRender, ActivateAudioInterfaceAsync, IActivateAudioInterfaceAsyncOperation,
    IActivateAudioInterfaceCompletionHandler, IActivateAudioInterfaceCompletionHandler_Impl,
    IAudioCaptureClient, IAudioClient, IMMDeviceEnumerator, MMDeviceEnumerator,
    AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
    AUDCLNT_STREAMFLAGS_EVENTCALLBACK, AUDCLNT_STREAMFLAGS_LOOPBACK, AUDIOCLIENT_ACTIVATION_PARAMS,
    AUDIOCLIENT_ACTIVATION_PARAMS_0, AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
    AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS, PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE,
    PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE, VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
    WAVEFORMATEX,
};
use windows::Win32::System::Com::StructuredStorage::{
    PROPVARIANT, PROPVARIANT_0, PROPVARIANT_0_0, PROPVARIANT_0_0_0,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, IAgileObject, IAgileObject_Impl, BLOB,
    CLSCTX_ALL, COINIT_MULTITHREADED,
};
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::Threading::{CreateEventW, SetEvent, WaitForSingleObject};
use windows::Win32::System::Variant::VT_BLOB;

/// `WAVE_FORMAT_IEEE_FLOAT`.
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
/// Channels we capture.
const CHANNELS: u16 = 2;

/// A running process-loopback capture; dropping it stops the thread.
pub struct Capture {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Capture {
    /// Start capturing into `feed`.
    pub fn start(cfg: &SystemConfig, feed: Arc<Feed>) -> Result<Self, String> {
        let (pid, mode) = match cfg.mode {
            SystemMode::App => {
                let pid = find_process(&cfg.app)
                    .ok_or_else(|| format!("no running process {:?}", cfg.app))?;
                (pid, PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE)
            }
            SystemMode::Sink => (
                std::process::id(),
                PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE,
            ),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        let thread_stop = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("cadenza-loopback".into())
            .spawn(move || {
                // SAFETY: COM is initialised for this thread and torn down at its end.
                unsafe {
                    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                    if let Err(e) = run(pid, mode, &feed, &thread_stop, &ready_tx) {
                        let _ = ready_tx.send(Err(e));
                    }
                    CoUninitialize();
                }
            })
            .map_err(|e| e.to_string())?;
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => Ok(Self {
                stop,
                thread: Some(thread),
            }),
            Ok(Err(e)) => Err(format!("process loopback: {e}")),
            Err(_) => Err("process loopback: activation did not complete".into()),
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The process id of a running program by its executable name (`firefox`, `firefox.exe`,
/// any case); the lowest id wins (usually the parent of a process tree).
/// The executable names (without `.exe`) of every running process, for the web UI's app
/// picker.
pub fn process_names() -> Vec<String> {
    let mut out = Vec::new();
    // SAFETY: a snapshot handle we close; entries are plain data.
    unsafe {
        let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else {
            return out;
        };
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        let mut ok = Process32FirstW(snap, &mut entry).is_ok();
        while ok {
            let len = entry
                .szExeFile
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(entry.szExeFile.len());
            let exe = String::from_utf16_lossy(&entry.szExeFile[..len]);
            let name = exe_name(&exe);
            if !name.is_empty() {
                out.push(name);
            }
            ok = Process32NextW(snap, &mut entry).is_ok();
        }
        let _ = CloseHandle(snap);
    }
    out
}

/// The process id of the first running program called `name` (with or without `.exe`).
pub fn find_process(name: &str) -> Option<u32> {
    let want = exe_name(name);
    // SAFETY: a snapshot handle we close; entries are plain data.
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0).ok()?;
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        let mut best: Option<u32> = None;
        let mut ok = Process32FirstW(snap, &mut entry).is_ok();
        while ok {
            let len = entry
                .szExeFile
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(entry.szExeFile.len());
            let exe = String::from_utf16_lossy(&entry.szExeFile[..len]);
            if exe_name(&exe) == want {
                best = Some(best.map_or(entry.th32ProcessID, |b| b.min(entry.th32ProcessID)));
            }
            ok = Process32NextW(snap, &mut entry).is_ok();
        }
        let _ = CloseHandle(snap);
        best
    }
}

/// `Firefox.EXE` → `firefox`.
pub fn exe_name(name: &str) -> String {
    let n = name.trim().to_ascii_lowercase();
    n.strip_suffix(".exe").unwrap_or(&n).to_string()
}

/// The capture format: 32-bit float stereo at `rate`.
pub fn float_format(rate: u32) -> WAVEFORMATEX {
    let block = CHANNELS * 4;
    WAVEFORMATEX {
        wFormatTag: WAVE_FORMAT_IEEE_FLOAT,
        nChannels: CHANNELS,
        nSamplesPerSec: rate,
        nAvgBytesPerSec: rate * block as u32,
        nBlockAlign: block,
        wBitsPerSample: 32,
        cbSize: 0,
    }
}

/// Completion of `ActivateAudioInterfaceAsync`: signals an event.
#[implement(IActivateAudioInterfaceCompletionHandler, IAgileObject)]
struct Done(HANDLE);

impl IActivateAudioInterfaceCompletionHandler_Impl for Done_Impl {
    fn ActivateCompleted(
        &self,
        _op: windows::core::Ref<'_, IActivateAudioInterfaceAsyncOperation>,
    ) -> windows::core::Result<()> {
        // SAFETY: the event handle lives until the waiter has returned.
        unsafe { SetEvent(self.0) }
    }
}

impl IAgileObject_Impl for Done_Impl {}

/// The default output device's shared-mode rate (what Windows mixes at), or 48 kHz.
unsafe fn mix_rate() -> u32 {
    let rate = || -> windows::core::Result<u32> {
        let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        let dev = en.GetDefaultAudioEndpoint(eRender, eConsole)?;
        let client: IAudioClient = dev.Activate(CLSCTX_ALL, None)?;
        let f = client.GetMixFormat()?;
        let r = (*f).nSamplesPerSec;
        windows::Win32::System::Com::CoTaskMemFree(Some(f as *const _));
        Ok(r)
    };
    rate().unwrap_or(48_000)
}

unsafe fn activate(
    pid: u32,
    mode: windows::Win32::Media::Audio::PROCESS_LOOPBACK_MODE,
) -> Result<IAudioClient, String> {
    let e = |e: windows::core::Error| e.to_string();
    let mut params = AUDIOCLIENT_ACTIVATION_PARAMS {
        ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
            ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                TargetProcessId: pid,
                ProcessLoopbackMode: mode,
            },
        },
    };
    let prop = PROPVARIANT {
        Anonymous: PROPVARIANT_0 {
            Anonymous: std::mem::ManuallyDrop::new(PROPVARIANT_0_0 {
                vt: VT_BLOB,
                wReserved1: 0,
                wReserved2: 0,
                wReserved3: 0,
                Anonymous: PROPVARIANT_0_0_0 {
                    blob: BLOB {
                        cbSize: std::mem::size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>() as u32,
                        pBlobData: &mut params as *mut _ as *mut u8,
                    },
                },
            }),
        },
    };
    let event = CreateEventW(None, false, false, PCWSTR::null()).map_err(e)?;
    let handler: IActivateAudioInterfaceCompletionHandler = Done(event).into();
    let op = ActivateAudioInterfaceAsync(
        VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        &IAudioClient::IID,
        Some(&prop),
        &handler,
    )
    .map_err(e)?;
    let waited = WaitForSingleObject(event, 5000);
    let _ = CloseHandle(event);
    if waited != WAIT_OBJECT_0 {
        return Err("timed out".into());
    }
    let mut hr = windows::core::HRESULT(0);
    let mut iface: Option<windows::core::IUnknown> = None;
    op.GetActivateResult(&mut hr, &mut iface).map_err(e)?;
    hr.ok().map_err(e)?;
    iface
        .ok_or_else(|| "no interface".to_string())?
        .cast::<IAudioClient>()
        .map_err(e)
}

unsafe fn run(
    pid: u32,
    mode: windows::Win32::Media::Audio::PROCESS_LOOPBACK_MODE,
    feed: &Feed,
    stop: &AtomicBool,
    ready: &mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let e = |e: windows::core::Error| e.to_string();
    let rate = mix_rate();
    log::info!("process loopback: activating for pid {pid} at {rate} Hz");
    let client = activate(pid, mode)?;
    log::info!("process loopback: activated");
    let format = float_format(rate);
    client
        .Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK
                | AUDCLNT_STREAMFLAGS_EVENTCALLBACK
                | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
            200_000, // 20 ms in 100 ns units
            0,
            &format,
            None,
        )
        .map_err(e)?;
    let event = CreateEventW(None, false, false, PCWSTR::null()).map_err(e)?;
    client.SetEventHandle(event).map_err(e)?;
    let capture: IAudioCaptureClient = client.GetService().map_err(e)?;
    client.Start().map_err(e)?;
    log::info!("process loopback: capturing");
    feed.set_format(rate, CHANNELS as u32);
    let _ = ready.send(Ok(()));
    let mut buf: Vec<f32> = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        if WaitForSingleObject(event, 100) != WAIT_OBJECT_0 {
            continue;
        }
        while matches!(capture.GetNextPacketSize(), Ok(n) if n > 0) {
            let mut data = std::ptr::null_mut();
            let mut got = 0u32;
            let mut flags = 0u32;
            if capture
                .GetBuffer(&mut data, &mut got, &mut flags, None, None)
                .is_err()
            {
                break;
            }
            let n = got as usize * CHANNELS as usize;
            buf.clear();
            if flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0 || data.is_null() {
                buf.resize(n, 0.0);
            } else {
                buf.extend_from_slice(std::slice::from_raw_parts(data as *const f32, n));
            }
            let _ = capture.ReleaseBuffer(got);
            feed.push_f32(&buf);
        }
    }
    let _ = client.Stop();
    let _ = CloseHandle(event);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_formats() {
        assert_eq!(exe_name("Firefox.EXE"), "firefox");
        assert_eq!(exe_name(" spotify "), "spotify");
        let f = float_format(44_100);
        assert_eq!(
            (f.nChannels, f.nBlockAlign, f.nAvgBytesPerSec),
            (2, 8, 352_800)
        );
        // This very process is found by its own name.
        let me = std::env::current_exe().unwrap();
        let name = me.file_name().unwrap().to_string_lossy().to_string();
        let pid = find_process(&name).unwrap();
        assert!(pid > 0);
        assert!(find_process("no-such-program-cadenza-test").is_none());
    }
}
