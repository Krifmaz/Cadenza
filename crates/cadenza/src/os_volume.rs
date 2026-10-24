//! The operating system's own output volume (the speaker icon in the taskbar or menu bar),
//! as opposed to the engine's volume stage.
//!
//! Encore shows it as "Device volume" next to the player volume when this player is
//! controlled from another device: the heartbeat state reports it as `system_volume`
//! (0..1, 0 while muted) and the `system_volume {value}` command sets it. A build or a
//! machine that cannot reach the mixer reports nothing, and Encore leaves the slider out.
//!
//! How it is reached, per platform:
//! - Linux: the first of `wpctl` (PipeWire), `pactl` (PulseAudio) and `amixer` (ALSA)
//!   that answers, on the default output.
//! - macOS: `osascript`, the system's output volume.
//! - Windows (the `os-volume` feature): the default render endpoint's
//!   `IAudioEndpointVolume`.
//!
//! Reads are cached for a moment, so the 2 s heartbeat does not start a process every time,
//! and a mixer that cannot be reached is not asked again for a while.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long a read level is reused before the mixer is asked again.
const FRESH: Duration = Duration::from_millis(1500);

/// How long "no mixer here" is believed before trying again (a desktop session may start
/// after the service, or PipeWire may restart).
const RETRY: Duration = Duration::from_secs(30);

/// The last read: when, and what it gave (`None`: no mixer could be reached).
static CACHE: Mutex<Option<(Instant, Option<f64>)>> = Mutex::new(None);

/// The system output volume, 0..1 (0 while muted), or `None` when this machine's mixer
/// cannot be reached. Cached for [`FRESH`] (a failure for [`RETRY`]).
pub fn get() -> Option<f64> {
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, level)) = *cache {
        let keep = if level.is_some() { FRESH } else { RETRY };
        if at.elapsed() < keep {
            return level;
        }
    }
    let level = backend::read().map(|l| l.clamp(0.0, 1.0));
    *cache = Some((Instant::now(), level));
    level
}

/// Set the system output volume to `level` (0..1, clamped) and unmute it, so the level
/// set is the level heard. The cache takes the new level at once, so the next heartbeat
/// reports it without asking the mixer.
pub fn set(level: f64) -> Result<(), String> {
    if !level.is_finite() {
        return Err("system_volume needs a number between 0 and 1".into());
    }
    let level = level.clamp(0.0, 1.0);
    backend::write(level)?;
    *CACHE.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), Some(level)));
    Ok(())
}

/// A level as a whole percentage for the command-line mixers (0..=100).
#[cfg_attr(windows, allow(dead_code))]
fn percent(level: f64) -> u32 {
    (level.clamp(0.0, 1.0) * 100.0).round() as u32
}

/// `wpctl get-volume` output (`Volume: 0.45` or `Volume: 0.45 [MUTED]`) as a level, 0 while
/// muted. PipeWire allows boosting past 1.0; that reads as 1.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_wpctl(out: &str) -> Option<f64> {
    let rest = out.trim().strip_prefix("Volume:")?.trim();
    let level: f64 = rest.split_whitespace().next()?.parse().ok()?;
    if rest.contains("[MUTED]") {
        return Some(0.0);
    }
    Some(level.clamp(0.0, 1.0))
}

/// The first `NN%` in a mixer's output (`pactl get-sink-volume`, `amixer get Master`) as a
/// level, 0 when the output also says it is muted (`Mute: yes`, `[off]`).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_percent(out: &str, muted: bool) -> Option<f64> {
    if muted {
        return Some(0.0);
    }
    let at = out.find('%')?;
    // walk back over the digits in front of the first '%'
    let digits: String = out[..at]
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let n: f64 = digits.parse().ok()?;
    Some((n / 100.0).clamp(0.0, 1.0))
}

/// Run a mixer command and return what it printed, `None` when it is missing or fails.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn run(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(target_os = "linux")]
mod backend {
    //! Linux: PipeWire's `wpctl`, PulseAudio's `pactl`, then ALSA's `amixer`.
    use super::{parse_percent, parse_wpctl, percent, run};

    /// The default output's level from the first mixer that answers.
    pub fn read() -> Option<f64> {
        if let Some(l) =
            run("wpctl", &["get-volume", "@DEFAULT_AUDIO_SINK@"]).and_then(|o| parse_wpctl(&o))
        {
            return Some(l);
        }
        if let Some(out) = run("pactl", &["get-sink-volume", "@DEFAULT_SINK@"]) {
            let muted = run("pactl", &["get-sink-mute", "@DEFAULT_SINK@"])
                .is_some_and(|m| m.contains("yes"));
            if let Some(l) = parse_percent(&out, muted) {
                return Some(l);
            }
        }
        // amixer prints `[45%] [on]` per channel; `[off]` is muted
        run("amixer", &["get", "Master"]).and_then(|o| parse_percent(&o, o.contains("[off]")))
    }

    /// Set the default output's level (and unmute it) with the first mixer that takes it.
    pub fn write(level: f64) -> Result<(), String> {
        let p = format!("{}%", percent(level));
        // wpctl takes a plain factor; three decimals is finer than any slider
        let factor = format!("{level:.3}");
        if run("wpctl", &["set-volume", "@DEFAULT_AUDIO_SINK@", &factor]).is_some() {
            let _ = run("wpctl", &["set-mute", "@DEFAULT_AUDIO_SINK@", "0"]);
            return Ok(());
        }
        if run("pactl", &["set-sink-volume", "@DEFAULT_SINK@", &p]).is_some() {
            let _ = run("pactl", &["set-sink-mute", "@DEFAULT_SINK@", "0"]);
            return Ok(());
        }
        if run("amixer", &["-q", "set", "Master", &p, "unmute"]).is_some() {
            return Ok(());
        }
        Err("No system mixer answered (wpctl, pactl or amixer)".into())
    }
}

#[cfg(target_os = "macos")]
mod backend {
    //! macOS: AppleScript's volume settings through `osascript`.
    use super::{percent, run};

    /// The output volume (0..100 in AppleScript), 0 while muted.
    pub fn read() -> Option<f64> {
        let out = run("osascript", &["-e", "get volume settings"])?;
        // `output volume:45, input volume:50, alert volume:100, output muted:false`
        let muted = out.contains("output muted:true");
        let vol = out
            .split(',')
            .find(|p| p.trim().starts_with("output volume:"))?;
        let n: f64 = vol
            .trim()
            .trim_start_matches("output volume:")
            .trim()
            .parse()
            .ok()?;
        Some(if muted {
            0.0
        } else {
            (n / 100.0).clamp(0.0, 1.0)
        })
    }

    /// Set the output volume and unmute it.
    pub fn write(level: f64) -> Result<(), String> {
        let script = format!(
            "set volume output volume {} without output muted",
            percent(level)
        );
        run("osascript", &["-e", &script])
            .map(|_| ())
            .ok_or_else(|| "osascript could not set the volume".into())
    }
}

#[cfg(all(windows, feature = "os-volume"))]
mod backend {
    //! Windows: the default render endpoint's `IAudioEndpointVolume` (the taskbar's volume).
    use windows::Win32::Media::Audio::Endpoints::IAudioEndpointVolume;
    use windows::Win32::Media::Audio::{
        eConsole, eRender, IMMDeviceEnumerator, MMDeviceEnumerator,
    };
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED,
    };

    /// The default output's volume interface. COM is joined on this thread first; a thread
    /// already in another apartment keeps it (the call still works there).
    fn endpoint() -> windows::core::Result<IAudioEndpointVolume> {
        // SAFETY: plain COM calls on interfaces this function owns; nothing outlives it but
        // the returned interface, which is reference counted.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let dev = en.GetDefaultAudioEndpoint(eRender, eConsole)?;
            dev.Activate(CLSCTX_ALL, None)
        }
    }

    /// The master volume scalar, 0 while muted.
    pub fn read() -> Option<f64> {
        let ep = endpoint().ok()?;
        // SAFETY: getters on a live interface.
        unsafe {
            if ep.GetMute().ok()?.as_bool() {
                return Some(0.0);
            }
            ep.GetMasterVolumeLevelScalar().ok().map(f64::from)
        }
    }

    /// Set the master volume scalar and unmute.
    pub fn write(level: f64) -> Result<(), String> {
        let ep = endpoint().map_err(|e| format!("No default output: {e}"))?;
        // SAFETY: setters on a live interface; no event context GUID is passed.
        unsafe {
            ep.SetMasterVolumeLevelScalar(level as f32, std::ptr::null())
                .map_err(|e| format!("Could not set the volume: {e}"))?;
            let _ = ep.SetMute(false, std::ptr::null());
        }
        Ok(())
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    all(windows, feature = "os-volume")
)))]
mod backend {
    //! No system mixer support in this build: nothing is reported, nothing is set.

    /// Always `None`: there is no mixer to read.
    pub fn read() -> Option<f64> {
        None
    }

    /// Always an error: there is no mixer to set.
    pub fn write(_level: f64) -> Result<(), String> {
        Err("This build cannot change the system volume".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wpctl_output_reads_as_a_level_and_muted_reads_as_zero() {
        assert_eq!(parse_wpctl("Volume: 0.45\n"), Some(0.45));
        assert_eq!(parse_wpctl("Volume: 0.45 [MUTED]\n"), Some(0.0));
        // PipeWire can boost past 100 %
        assert_eq!(parse_wpctl("Volume: 1.30"), Some(1.0));
        assert_eq!(parse_wpctl("nonsense"), None);
    }

    #[test]
    fn the_first_percentage_is_the_level() {
        let pactl = "Volume: front-left: 29491 /  45% / -20.81 dB,   front-right: 29491 /  45% / -20.81 dB\n";
        assert_eq!(parse_percent(pactl, false), Some(0.45));
        let amixer = "Simple mixer control 'Master',0\n  Mono: Playback 52 [80%] [-15.00dB] [on]\n";
        assert_eq!(parse_percent(amixer, false), Some(0.8));
        assert_eq!(parse_percent(amixer, true), Some(0.0));
        assert_eq!(parse_percent("no level here", false), None);
    }

    #[test]
    fn percentages_round_and_clamp() {
        assert_eq!(percent(0.444), 44);
        assert_eq!(percent(0.445), 45);
        assert_eq!(percent(1.7), 100);
        assert_eq!(percent(-1.0), 0);
    }

    #[test]
    fn a_level_that_is_not_a_number_is_refused() {
        assert!(set(f64::NAN).is_err());
    }
}
