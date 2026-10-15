//! The macOS system input: Core Audio process taps (macOS 14.2 and later), no driver.
//!
//! * `app`: a tap on one application's processes (matched by bundle id or process name),
//!   muted, so the app is heard only through Cadenza.
//! * `sink`: a tap on every process except Cadenza, muted: the whole system's sound goes
//!   through Cadenza instead of straight to the default output – a virtual output device
//!   without installing one.
//!
//! A tap is read through a private aggregate device whose clock is the default output
//! device's (with drift compensation); the tap delivers 32-bit float at that device's
//! rate. Set the default device's rate (Audio MIDI Setup) to the source rate for a path
//! without resampling, and keep the app's volume at 100 %. macOS asks once for permission
//! to record system audio (for the terminal or app that runs Cadenza).

use super::{interleave_planar, Feed, SystemConfig, SystemMode};
use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_core_audio::{
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceMainSubDeviceKey,
    kAudioAggregateDeviceNameKey, kAudioAggregateDeviceSubDeviceListKey,
    kAudioAggregateDeviceTapAutoStartKey, kAudioAggregateDeviceTapListKey,
    kAudioAggregateDeviceUIDKey, kAudioDevicePropertyDeviceUID,
    kAudioHardwarePropertyDefaultOutputDevice, kAudioHardwarePropertyProcessObjectList,
    kAudioHardwarePropertyTranslatePIDToProcessObject, kAudioObjectPropertyElementMain,
    kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject, kAudioProcessPropertyBundleID,
    kAudioProcessPropertyPID, kAudioSubDeviceUIDKey, kAudioSubTapDriftCompensationKey,
    kAudioSubTapUIDKey, kAudioTapPropertyFormat, AudioDeviceCreateIOProcID,
    AudioDeviceDestroyIOProcID, AudioDeviceIOProcID, AudioDeviceStart, AudioDeviceStop,
    AudioHardwareCreateAggregateDevice, AudioHardwareCreateProcessTap,
    AudioHardwareDestroyAggregateDevice, AudioHardwareDestroyProcessTap,
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress, AudioObjectPropertySelector, CATapDescription, CATapMuteBehavior,
};
use objc2_core_audio_types::{
    kAudioFormatFlagIsNonInterleaved, AudioBuffer, AudioBufferList, AudioStreamBasicDescription,
    AudioTimeStamp,
};
use objc2_core_foundation::{CFDictionary, CFString};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSUUID};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

/// A running tap; dropping it stops and removes everything it created.
pub struct Capture {
    tap: AudioObjectID,
    device: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    ctx: *mut Ctx,
}

// SAFETY: the ids are plain integers; the context is only touched by the IO thread until
// `drop` has stopped and destroyed the IO proc.
unsafe impl Send for Capture {}

struct Ctx {
    feed: Arc<Feed>,
    channels: usize,
    planar: bool,
    buf: Vec<f32>,
}

fn addr(selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

/// Read a fixed-size property.
unsafe fn get<T: Copy>(
    obj: AudioObjectID,
    selector: AudioObjectPropertySelector,
    qualifier: Option<&[u8]>,
    init: T,
) -> Result<T, String> {
    let mut a = addr(selector);
    let mut v = init;
    let mut size = std::mem::size_of::<T>() as u32;
    let (qs, qp) = qualifier.map_or((0, std::ptr::null()), |q| {
        (q.len() as u32, q.as_ptr() as *const c_void)
    });
    let st = AudioObjectGetPropertyData(
        obj,
        NonNull::from(&mut a),
        qs,
        qp,
        NonNull::from(&mut size),
        NonNull::from(&mut v).cast(),
    );
    if st != 0 {
        return Err(format!("property {selector:#x} of {obj}: OSStatus {st}"));
    }
    Ok(v)
}

/// Read an array-of-ids property.
unsafe fn get_ids(
    obj: AudioObjectID,
    selector: AudioObjectPropertySelector,
) -> Result<Vec<AudioObjectID>, String> {
    let mut a = addr(selector);
    let mut size = 0u32;
    let st = AudioObjectGetPropertyDataSize(
        obj,
        NonNull::from(&mut a),
        0,
        std::ptr::null(),
        NonNull::from(&mut size),
    );
    if st != 0 {
        return Err(format!("property {selector:#x} size: OSStatus {st}"));
    }
    let mut v = vec![0 as AudioObjectID; size as usize / std::mem::size_of::<AudioObjectID>()];
    if v.is_empty() {
        return Ok(v);
    }
    let st = AudioObjectGetPropertyData(
        obj,
        NonNull::from(&mut a),
        0,
        std::ptr::null(),
        NonNull::from(&mut size),
        NonNull::new(v.as_mut_ptr()).unwrap().cast(),
    );
    if st != 0 {
        return Err(format!("property {selector:#x}: OSStatus {st}"));
    }
    v.truncate(size as usize / std::mem::size_of::<AudioObjectID>());
    Ok(v)
}

/// Read a CFString property as a Rust string.
unsafe fn get_string(obj: AudioObjectID, selector: AudioObjectPropertySelector) -> Option<String> {
    let p: *const CFString = get(obj, selector, None, std::ptr::null()).ok()?;
    let s = p.as_ref()?;
    let out = s.to_string();
    objc2_core_foundation::CFRetained::from_raw(NonNull::from(s));
    Some(out)
}

/// The name of process `pid` (`Music`, `firefox`).
fn process_name(pid: i32) -> Option<String> {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer and its length are passed together.
    let n = unsafe { libc::proc_name(pid, buf.as_mut_ptr() as *mut c_void, buf.len() as u32) };
    (n > 0).then(|| String::from_utf8_lossy(&buf[..n as usize]).into_owned())
}

/// Does a process (by bundle id and name) match what the user asked for? Case-insensitive
/// against the whole bundle id, its last part, or the process name.
pub fn matches(want: &str, bundle: Option<&str>, name: Option<&str>) -> bool {
    let want = want.trim().to_ascii_lowercase();
    let eq = |s: &str| s.to_ascii_lowercase() == want;
    bundle.is_some_and(|b| eq(b) || b.rsplit('.').next().is_some_and(eq)) || name.is_some_and(eq)
}

/// Core Audio process objects of the app called `want`.
unsafe fn find_app(want: &str) -> Result<Vec<AudioObjectID>, String> {
    let all = get_ids(
        kAudioObjectSystemObject as AudioObjectID,
        kAudioHardwarePropertyProcessObjectList,
    )?;
    Ok(all
        .into_iter()
        .filter(|&p| {
            let pid: i32 = get(p, kAudioProcessPropertyPID, None, -1).unwrap_or(-1);
            let bundle = get_string(p, kAudioProcessPropertyBundleID);
            matches(want, bundle.as_deref(), process_name(pid).as_deref())
        })
        .collect())
}

/// Our own process object (to leave Cadenza out of a global tap).
unsafe fn own_process() -> Option<AudioObjectID> {
    let pid = std::process::id() as i32;
    let id: AudioObjectID = get(
        kAudioObjectSystemObject as AudioObjectID,
        kAudioHardwarePropertyTranslatePIDToProcessObject,
        Some(&pid.to_ne_bytes()),
        0,
    )
    .ok()?;
    (id != 0).then_some(id)
}

unsafe extern "C-unwind" fn io_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    _output: NonNull<AudioBufferList>,
    _output_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> i32 {
    let ctx = &mut *(client as *mut Ctx);
    let list = input.as_ref();
    let n = list.mNumberBuffers as usize;
    let first = list.mBuffers.as_ptr();
    let buffers: Vec<&AudioBuffer> = (0..n).map(|i| &*first.add(i)).collect();
    let slice = |b: &AudioBuffer| -> &[f32] {
        if b.mData.is_null() {
            &[]
        } else {
            std::slice::from_raw_parts(b.mData as *const f32, b.mDataByteSize as usize / 4)
        }
    };
    if ctx.planar && n > 1 {
        let planes: Vec<&[f32]> = buffers
            .iter()
            .take(ctx.channels)
            .map(|b| slice(b))
            .collect();
        ctx.buf = interleave_planar(&planes);
    } else if let Some(b) = buffers.first() {
        ctx.buf.clear();
        ctx.buf.extend_from_slice(slice(b));
    }
    ctx.feed.push_f32(&ctx.buf);
    0
}

impl Capture {
    /// Create the tap and its aggregate device and start delivering into `feed`.
    pub fn start(cfg: &SystemConfig, feed: Arc<Feed>) -> Result<Self, String> {
        // SAFETY: Core Audio calls with valid arguments; everything created here is
        // released in `drop` (or below on the error paths).
        unsafe {
            let desc: Retained<CATapDescription> = match cfg.mode {
                SystemMode::App => {
                    let procs = find_app(&cfg.app)?;
                    if procs.is_empty() {
                        return Err(format!(
                            "no audio process {:?} (it must be running and have played sound)",
                            cfg.app
                        ));
                    }
                    let nums: Vec<Retained<NSNumber>> =
                        procs.iter().map(|&p| NSNumber::new_u32(p)).collect();
                    CATapDescription::initStereoMixdownOfProcesses(
                        CATapDescription::alloc(),
                        &NSArray::from_retained_slice(&nums),
                    )
                }
                SystemMode::Sink => {
                    let nums: Vec<Retained<NSNumber>> =
                        own_process().into_iter().map(NSNumber::new_u32).collect();
                    CATapDescription::initStereoGlobalTapButExcludeProcesses(
                        CATapDescription::alloc(),
                        &NSArray::from_retained_slice(&nums),
                    )
                }
            };
            desc.setName(&NSString::from_str(&format!("{} tap", cfg.name)));
            desc.setPrivate(true);
            desc.setMuteBehavior(CATapMuteBehavior::Muted);
            let mut tap: AudioObjectID = 0;
            let st = AudioHardwareCreateProcessTap(Some(&desc), &mut tap);
            if st != 0 {
                return Err(format!(
                    "creating the process tap failed (OSStatus {st}; macOS 14.2 or later, and the permission to record system audio, are needed)"
                ));
            }
            let fail = |e: String, device: AudioObjectID| {
                if device != 0 {
                    AudioHardwareDestroyAggregateDevice(device);
                }
                AudioHardwareDestroyProcessTap(tap);
                Err(e)
            };
            let fmt: AudioStreamBasicDescription =
                match get(tap, kAudioTapPropertyFormat, None, std::mem::zeroed()) {
                    Ok(f) => f,
                    Err(e) => return fail(e, 0),
                };
            let tap_uid = desc.UUID().UUIDString();
            let out: AudioObjectID = get(
                kAudioObjectSystemObject as AudioObjectID,
                kAudioHardwarePropertyDefaultOutputDevice,
                None,
                0,
            )
            .unwrap_or(0);
            let out_uid = get_string(out, kAudioDevicePropertyDeviceUID).unwrap_or_default();
            let key = |k: &std::ffi::CStr| NSString::from_str(k.to_str().unwrap_or_default());
            let sub_tap = NSDictionary::from_retained_objects(
                &[
                    &*key(kAudioSubTapUIDKey),
                    &*key(kAudioSubTapDriftCompensationKey),
                ],
                &[
                    Retained::into_super(Retained::into_super(tap_uid.clone())),
                    Retained::into_super(Retained::into_super(NSNumber::new_bool(true))).into(),
                ],
            );
            let sub_dev = NSDictionary::from_retained_objects(
                &[&*key(kAudioSubDeviceUIDKey)],
                &[Retained::into_super(Retained::into_super(
                    NSString::from_str(&out_uid),
                ))],
            );
            let uid = NSUUID::new().UUIDString();
            let agg: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
                NSDictionary::from_retained_objects(
                    &[
                        &*key(kAudioAggregateDeviceNameKey),
                        &*key(kAudioAggregateDeviceUIDKey),
                        &*key(kAudioAggregateDeviceMainSubDeviceKey),
                        &*key(kAudioAggregateDeviceIsPrivateKey),
                        &*key(kAudioAggregateDeviceTapAutoStartKey),
                        &*key(kAudioAggregateDeviceSubDeviceListKey),
                        &*key(kAudioAggregateDeviceTapListKey),
                    ],
                    &[
                        Retained::into_super(Retained::into_super(NSString::from_str(&format!(
                            "{} capture",
                            cfg.name
                        )))),
                        Retained::into_super(Retained::into_super(uid)),
                        Retained::into_super(Retained::into_super(NSString::from_str(&out_uid))),
                        Retained::into_super(Retained::into_super(NSNumber::new_bool(true))).into(),
                        Retained::into_super(Retained::into_super(NSNumber::new_bool(true))).into(),
                        Retained::into_super(Retained::into_super(NSArray::from_retained_slice(
                            &[sub_dev],
                        ))),
                        Retained::into_super(Retained::into_super(NSArray::from_retained_slice(
                            &[sub_tap],
                        ))),
                    ],
                );
            // NSDictionary is toll-free bridged to CFDictionary.
            let cf = &*(Retained::as_ptr(&agg) as *const CFDictionary);
            let mut device: AudioObjectID = 0;
            let st = AudioHardwareCreateAggregateDevice(cf, NonNull::from(&mut device));
            if st != 0 {
                return fail(format!("creating the aggregate device: OSStatus {st}"), 0);
            }
            let channels = fmt.mChannelsPerFrame.max(1) as usize;
            let ctx = Box::into_raw(Box::new(Ctx {
                feed: Arc::clone(&feed),
                channels,
                planar: fmt.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0,
                buf: Vec::new(),
            }));
            let mut proc_id: AudioDeviceIOProcID = None;
            let st = AudioDeviceCreateIOProcID(
                device,
                Some(io_proc),
                ctx as *mut c_void,
                NonNull::from(&mut proc_id),
            );
            if st != 0 {
                drop(Box::from_raw(ctx));
                return fail(format!("creating the IO proc: OSStatus {st}"), device);
            }
            feed.set_format(fmt.mSampleRate.round() as u32, channels as u32);
            let st = AudioDeviceStart(device, proc_id);
            if st != 0 {
                AudioDeviceDestroyIOProcID(device, proc_id);
                drop(Box::from_raw(ctx));
                return fail(format!("starting the capture: OSStatus {st}"), device);
            }
            Ok(Self {
                tap,
                device,
                proc_id,
                ctx,
            })
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        // SAFETY: stop the IO first; after that nothing touches the context.
        unsafe {
            AudioDeviceStop(self.device, self.proc_id);
            AudioDeviceDestroyIOProcID(self.device, self.proc_id);
            AudioHardwareDestroyAggregateDevice(self.device);
            AudioHardwareDestroyProcessTap(self.tap);
            drop(Box::from_raw(self.ctx));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_matching_and_own_process_name() {
        assert!(matches("music", Some("com.apple.Music"), Some("Music")));
        assert!(matches("com.apple.music", Some("com.apple.Music"), None));
        assert!(matches("firefox", None, Some("firefox")));
        assert!(!matches("spotify", Some("com.apple.Music"), Some("Music")));
        // This very process has a name.
        let me = process_name(std::process::id() as i32).unwrap();
        assert!(!me.is_empty());
    }
}
