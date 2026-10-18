//! ASIO output (Windows): the DAC driver's own low-latency interface, which also carries
//! native DSD – the way most DACs on Windows take DSD without DoP.
//!
//! No Steinberg SDK is used or shipped (the FlexASIO approach): [`sys`] spells out the
//! driver's binary interface, [`host`] drives a driver through it, and on Windows the
//! loader reads the installed drivers from `HKLM\SOFTWARE\ASIO` and creates the chosen
//! one with `CoCreateInstance`. Everything except the loader is platform independent and
//! tested against a fake driver.
//!
//! * PCM: the sample rate is set on the driver, every output channel's sample type is
//!   honoured (16/24/32-bit integers in either byte order, right-aligned 32-bit
//!   containers, 32/64-bit float – 24-bit words are exact in all of them).
//! * DoP: 24-bit PCM at `f_dsd / 16`, only on integer channels of at least 24 bits (a
//!   float path would not keep the markers bit-exact).
//! * Native DSD: the driver is switched to DSD with `kAsioSetIoFormat`, the "sample rate"
//!   becomes the DSD bit rate, and each buffer holds one byte (8 one-bit samples) per
//!   channel per frame, in the bit order the channel type says.
//! * The driver's reset request (its control panel changed the buffer size, the device
//!   was replugged…) reopens the stream with the same format on the next write.
//!
//! The device id is the driver's registry name (`ASIO4ALL v2`, `RME Fireface USB`…), or
//! empty / `default` for the first installed driver.

pub mod host;
pub mod sys;

#[cfg(test)]
mod fake;
#[cfg(all(windows, feature = "asio"))]
pub mod windows;

use super::{DeviceInfo, DsdCaps, Encoding, Output, OutputError, OutputFormat, STANDARD_RATES};
use cadenza_dsp::dsd::{DsdPacking, DsdRate, NativeFormat};
use host::{AsioError, Driver, SampleFormat, Shared};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Where drivers come from: the Windows registry, or a fake in the tests.
pub trait DriverSource: Send + Sync {
    /// Installed driver names, in registry order.
    fn names(&self) -> Vec<String>;
    /// Create the driver `name` (not yet initialised).
    fn load(&self, name: &str) -> Result<Driver, AsioError>;
    /// The window handle `init` gets (drivers parent their dialogs to it), as an integer.
    fn sys_handle(&self) -> isize {
        0
    }
}

impl From<AsioError> for OutputError {
    fn from(e: AsioError) -> Self {
        OutputError::Backend(e.to_string())
    }
}

/// What a driver can do, read once when it is opened.
#[derive(Clone, Debug, PartialEq)]
pub struct Probe {
    /// Output channels.
    pub outputs: u32,
    /// The first output channel's PCM sample format.
    pub format: Option<SampleFormat>,
    /// Supported PCM rates (of [`STANDARD_RATES`]).
    pub rates: Vec<u32>,
    /// DSD capabilities (DoP on integer channels, native when the driver switches to DSD).
    pub dsd: DsdCaps,
}

impl Probe {
    fn run(driver: &Driver) -> Result<Self, AsioError> {
        let (_, outputs) = driver.channels()?;
        if outputs <= 0 {
            return Err(AsioError::host("getChannels", "the driver has no outputs"));
        }
        driver.set_dsd(false);
        let format = SampleFormat::from_asio(driver.channel_info(0, false)?.sample_type);
        // Above 1536 kHz there is only DSD; some drivers say yes to those rates in PCM mode.
        let rates: Vec<u32> = STANDARD_RATES
            .iter()
            .copied()
            .filter(|&r| r <= super::MAX_PCM_RATE && driver.can_sample_rate(r as f64))
            .collect();
        let dop_max = format
            .filter(|f| f.carries_dop())
            .and_then(|_| rates.iter().max().map(|&r| r * 16));
        let mut native_max = None;
        if driver.can_do_dsd() && driver.set_dsd(true) {
            let dsd_format = SampleFormat::from_asio(driver.channel_info(0, false)?.sample_type);
            if dsd_format.is_some_and(SampleFormat::is_dsd) {
                for r in DsdRate::catalog() {
                    for base in [44_100, 48_000] {
                        let hz = r.hz(base);
                        if driver.can_sample_rate(hz as f64) {
                            native_max = Some(native_max.map_or(hz, |m: u32| m.max(hz)));
                        }
                    }
                }
            }
            driver.set_dsd(false);
        }
        let dsd = DsdCaps {
            dop: dop_max.is_some(),
            native: if native_max.is_some() {
                vec![NativeFormat::U8]
            } else {
                Vec::new()
            },
            max_rate: match (dop_max, native_max) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            },
        };
        log::info!(
            "ASIO {}: {outputs} outputs, {format:?}, PCM rates {rates:?}, DoP up to {dop_max:?}, native DSD up to {native_max:?}",
            driver.name()
        );
        Ok(Self {
            outputs: outputs as u32,
            format,
            rates,
            dsd,
        })
    }

    /// PCM bit depths the channels take.
    pub fn bits(&self) -> Vec<u32> {
        match self.format {
            Some(f) if f.pcm_bits(32) == 16 => vec![16],
            Some(f) if f.pcm_bits(32) == 32 => vec![16, 24, 32],
            Some(_) => vec![16, 24],
            None => Vec::new(),
        }
    }
}

struct Stream {
    shared: Arc<Shared>,
    format: OutputFormat,
    started: bool,
    /// Units per output frame in the ring: 1, or 4 bytes per native DSD word.
    units_per_frame: usize,
}

/// See the module documentation.
pub struct AsioOutput {
    /// Keeps the loader (and on Windows its COM state) alive with the driver.
    _source: Arc<dyn DriverSource>,
    driver: Driver,
    probe: Probe,
    stream: Option<Stream>,
    written: u64,
    units: Vec<i32>,
    /// Registered in [`IN_USE`] (the output playback holds).
    held: bool,
}

/// The drivers playback holds open, as listed: a listing reuses these instead of loading
/// a driver a second time, which some drivers do not survive while they stream.
static IN_USE: std::sync::Mutex<Vec<DeviceInfo>> = std::sync::Mutex::new(Vec::new());

fn device_info(name: &str, p: &Probe) -> DeviceInfo {
    DeviceInfo {
        backend: "asio".into(),
        id: name.to_string(),
        name: name.to_string(),
        rates: p.rates.clone(),
        bits: p.bits(),
        channels: Some(p.outputs),
        default: false,
        exclusive: true,
        dsd: p.dsd.clone(),
    }
}

/// How long `write` / `drain` wait for a driver that stopped calling back.
const STALL: Duration = Duration::from_secs(3);

impl AsioOutput {
    /// Load and initialise driver `name` (`""` / `default`: the first installed one).
    pub fn with_source(source: Arc<dyn DriverSource>, name: &str) -> Result<Self, OutputError> {
        let names = source.names();
        let name = if name.is_empty() || name == "default" {
            names
                .first()
                .cloned()
                .ok_or_else(|| OutputError::NoDevice("no ASIO driver is installed".into()))?
        } else if names.iter().any(|n| n == name) {
            name.to_string()
        } else {
            return Err(OutputError::NoDevice(format!(
                "ASIO driver {name:?} (installed: {})",
                names.join(", ")
            )));
        };
        let driver = source.load(&name)?;
        driver.init(source.sys_handle())?;
        let probe = Probe::run(&driver)?;
        Ok(Self {
            _source: source,
            driver,
            probe,
            stream: None,
            written: 0,
            units: Vec::new(),
            held: false,
        })
    }

    /// Mark this as the output playback holds: listings show it from here on without
    /// loading the driver again.
    pub fn hold(mut self) -> Self {
        let info = device_info(self.driver.name(), &self.probe);
        let mut in_use = IN_USE.lock().unwrap_or_else(|e| e.into_inner());
        in_use.retain(|d| d.id != info.id);
        in_use.push(info);
        self.held = true;
        self
    }

    /// What the driver can do.
    pub fn probe(&self) -> &Probe {
        &self.probe
    }

    /// Open the driver's control panel (buffer size, clock source…).
    pub fn control_panel(&self) -> Result<(), OutputError> {
        Ok(self.driver.control_panel()?)
    }

    fn unsupported(&self, format: OutputFormat) -> OutputError {
        OutputError::Unsupported {
            device: self.name(),
            format,
        }
    }

    fn refused(&self, format: OutputFormat, why: &str) -> OutputError {
        OutputError::Backend(format!("{} refused {format}: {why}", self.name()))
    }

    fn open_stream(&mut self, format: OutputFormat) -> Result<Stream, OutputError> {
        let d = &self.driver;
        let ch = format.channels as usize;
        if ch == 0 || ch as u32 > self.probe.outputs {
            return Err(self.unsupported(format));
        }
        let native = matches!(
            format.encoding,
            Encoding::Dsd {
                packing: DsdPacking::Native,
                ..
            }
        );
        let rate = match format.encoding {
            Encoding::Dsd {
                rate,
                packing: DsdPacking::Native,
            } => rate,
            _ => format.rate,
        };
        if native {
            if !d.set_dsd(true) {
                return Err(self.refused(format, "the driver would not switch to DSD"));
            }
        } else {
            d.set_dsd(false);
        }
        if !d.can_sample_rate(rate as f64) {
            return Err(self.refused(format, &format!("the driver does not take {rate} Hz")));
        }
        d.set_sample_rate(rate as f64)?;
        let mut sample = None;
        for c in 0..ch {
            let t = d.channel_info(c as i32, false)?.sample_type;
            let f = SampleFormat::from_asio(t).ok_or_else(|| {
                OutputError::Backend(format!("ASIO channel {c}: unsupported sample type {t}"))
            })?;
            match sample {
                None => sample = Some(f),
                Some(s) if s != f => {
                    return Err(OutputError::Backend(
                        "ASIO output channels use different sample types".into(),
                    ))
                }
                Some(_) => {}
            }
        }
        let sample = sample.expect("at least one channel");
        if native != sample.is_dsd() {
            return Err(self.refused(
                format,
                if native {
                    "the driver stayed in PCM after switching to DSD"
                } else {
                    "the driver stayed in DSD"
                },
            ));
        }
        let mut opened = format;
        match format.encoding {
            Encoding::Pcm => opened.bits = sample.pcm_bits(format.bits),
            Encoding::Dsd {
                packing: DsdPacking::Dop,
                ..
            } if !sample.carries_dop() => return Err(self.unsupported(format)),
            Encoding::Dsd { .. } => {}
        }
        let sizes = d.buffer_sizes()?;
        let size = sizes.preferred.max(sizes.min).max(1);
        if native && size % 8 != 0 {
            return Err(OutputError::Backend(format!(
                "ASIO DSD buffer of {size} samples is not a whole number of bytes"
            )));
        }
        let frames = if native { size / 8 } else { size } as usize;
        let mut infos: Vec<sys::BufferInfo> = (0..ch)
            .map(|c| sys::BufferInfo {
                is_input: 0,
                channel: c as i32,
                buffers: [std::ptr::null_mut(); 2],
            })
            .collect();
        // SAFETY: the callback table is a static.
        unsafe { d.create_buffers(&mut infos, size, host::callbacks())? };
        let buffers: Vec<[*mut u8; 2]> = infos
            .iter()
            .map(|i| {
                let b = i.buffers;
                [b[0] as *mut u8, b[1] as *mut u8]
            })
            .collect();
        if buffers.iter().any(|b| b[0].is_null() || b[1].is_null()) {
            let _ = d.dispose_buffers();
            return Err(OutputError::Backend(
                "ASIO createBuffers gave no buffers".into(),
            ));
        }
        // `outputReady` before the start only tells whether the driver supports it.
        let output_ready = d
            .output_ready()
            .then(|| std::ptr::NonNull::new(self.driver.as_raw()))
            .flatten();
        // A tenth of a second of queue (at least four half buffers).
        let units_per_sec = if native { rate / 8 } else { rate } as usize;
        let shared = Arc::new(Shared::new(
            buffers,
            sample,
            opened.bits,
            frames,
            (units_per_sec / 10).max(4 * frames),
            output_ready,
        ));
        shared.prime();
        host::activate(&shared);
        log::info!(
            "ASIO: {} open, {} {}ch, buffer {size}{}, {:?}",
            self.driver.name(),
            opened.label(),
            ch,
            if native { " bits" } else { " frames" },
            sample
        );
        Ok(Stream {
            shared,
            format: opened,
            started: false,
            units_per_frame: if native { 4 } else { 1 },
        })
    }

    fn close_stream(&mut self) {
        if let Some(s) = self.stream.take() {
            if s.started {
                let _ = self.driver.stop();
            }
            host::deactivate();
            let _ = self.driver.dispose_buffers();
            if s.format.is_dsd() {
                self.driver.set_dsd(false);
            }
        }
    }

    fn start(&mut self) -> Result<(), OutputError> {
        if let Some(s) = &mut self.stream {
            if !s.started {
                self.driver.start()?;
                s.started = true;
            }
        }
        Ok(())
    }

    /// The driver asked for a reset: reopen with the same format (the queue is lost).
    fn handle_reset(&mut self) -> Result<(), OutputError> {
        let Some(format) = self
            .stream
            .as_ref()
            .filter(|s| s.shared.reset_request.load(Ordering::Acquire))
            .map(|s| s.format)
        else {
            return Ok(());
        };
        log::warn!("ASIO: {} requested a reset; reopening", self.driver.name());
        self.close_stream();
        self.stream = Some(self.open_stream(format)?);
        Ok(())
    }

    /// Wait until the callback consumed something (or the driver stalled).
    fn wait(&mut self, since: &mut (u64, Instant)) -> Result<(), OutputError> {
        let s = self.stream.as_ref().expect("open");
        s.shared.wait_for_room(Duration::from_millis(5));
        let switches = s.shared.switches.load(Ordering::Acquire);
        if switches != since.0 {
            *since = (switches, Instant::now());
        } else if since.1.elapsed() > STALL {
            return Err(OutputError::Backend(format!(
                "ASIO driver {} stopped calling back",
                self.driver.name()
            )));
        }
        Ok(())
    }

    /// Underruns (half buffers that found the queue short) since the stream opened.
    pub fn underruns(&self) -> u64 {
        self.stream
            .as_ref()
            .map_or(0, |s| s.shared.underruns.load(Ordering::Relaxed))
    }
}

impl Drop for AsioOutput {
    fn drop(&mut self) {
        self.close_stream();
        if self.held {
            let name = self.driver.name().to_string();
            IN_USE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|d| d.id != name);
        }
    }
}

impl Output for AsioOutput {
    fn name(&self) -> String {
        format!("asio · {}", self.driver.name())
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        self.close_stream();
        let s = self.open_stream(format)?;
        let opened = s.format;
        self.stream = Some(s);
        Ok(opened)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        self.probe.rates.contains(&rate) && self.probe.bits().contains(&bits)
    }

    fn dsd_caps(&self) -> DsdCaps {
        self.probe.dsd.clone()
    }

    fn exclusive(&self) -> bool {
        true
    }

    fn write(&mut self, words: &[i32]) -> Result<(), OutputError> {
        self.handle_reset()?;
        let s = self
            .stream
            .as_ref()
            .ok_or_else(|| OutputError::Backend("ASIO stream not open".into()))?;
        let ch = s.format.channels as usize;
        self.units.clear();
        if s.units_per_frame == 4 {
            host::dsd_words_to_units(words, ch, &mut self.units);
        } else {
            self.units
                .extend_from_slice(&words[..words.len() / ch * ch]);
        }
        let units = std::mem::take(&mut self.units);
        let mut at = 0;
        let mut since = (0u64, Instant::now());
        while at < units.len() {
            let shared = Arc::clone(&self.stream.as_ref().expect("open").shared);
            at += shared.push(&units[at..]);
            if at < units.len() {
                // Queue full: the clock must run.
                self.start()?;
                self.wait(&mut since)?;
            }
        }
        self.units = units;
        let s = self.stream.as_ref().expect("open");
        if !s.started && s.shared.queued_frames() >= 2 * s.shared.frames() {
            self.start()?;
        }
        self.written += (words.len() / ch) as u64;
        Ok(())
    }

    fn pause(&mut self, paused: bool) -> Result<(), OutputError> {
        if let Some(s) = &self.stream {
            s.shared.set_paused(paused);
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), OutputError> {
        if self
            .stream
            .as_ref()
            .is_none_or(|s| s.shared.queued_frames() == 0)
        {
            return Ok(());
        }
        self.start()?;
        self.stream
            .as_ref()
            .expect("open")
            .shared
            .set_draining(true);
        let r = self.drain_wait();
        if let Some(s) = &self.stream {
            s.shared.set_draining(false);
        }
        r
    }

    fn discard(&mut self) -> Result<(), OutputError> {
        if let Some(s) = &self.stream {
            s.shared.clear();
        }
        Ok(())
    }

    fn close(&mut self) {
        self.close_stream();
    }

    fn queued_frames(&self) -> usize {
        self.stream
            .as_ref()
            .map_or(0, |s| s.shared.queued_frames() / s.units_per_frame)
    }

    fn frames_written(&self) -> u64 {
        self.written
    }

    fn format(&self) -> Option<OutputFormat> {
        self.stream.as_ref().map(|s| s.format)
    }
}

impl AsioOutput {
    fn drain_wait(&mut self) -> Result<(), OutputError> {
        let mut since = (0u64, Instant::now());
        while self.stream.as_ref().expect("open").shared.queued_frames() > 0 {
            self.wait(&mut since)?;
        }
        // Two more half buffers: the last data has left the double buffer.
        let target = self
            .stream
            .as_ref()
            .expect("open")
            .shared
            .switches
            .load(Ordering::Acquire)
            + 2;
        while self
            .stream
            .as_ref()
            .expect("open")
            .shared
            .switches
            .load(Ordering::Acquire)
            < target
        {
            self.wait(&mut since)?;
        }
        Ok(())
    }
}

/// The devices (installed drivers) of `source`, each probed once.
pub fn list_devices_from(source: &Arc<dyn DriverSource>) -> Vec<DeviceInfo> {
    let names = source.names();
    names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            let held = IN_USE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .find(|d| d.id == *name)
                .cloned();
            let mut info = match held {
                Some(d) => d,
                None => match AsioOutput::with_source(Arc::clone(source), name) {
                    Ok(o) => device_info(name, o.probe()),
                    Err(e) => {
                        log::debug!("ASIO: {name}: {e}");
                        return None;
                    }
                },
            };
            info.default = i == 0;
            Some(info)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::fake::{Fake, FakeSource, Spec};
    use super::*;
    use std::sync::Mutex;

    /// ASIO has one active stream per process: the tests take turns.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    fn source(spec: Spec) -> (Arc<Fake>, Arc<dyn DriverSource>) {
        let f = Fake::new(spec);
        let src: Arc<dyn DriverSource> =
            Arc::new(FakeSource(vec![("Fake".into(), Arc::clone(&f))]));
        (f, src)
    }

    fn ramp(frames: usize, ch: usize, bits: u32) -> Vec<i32> {
        let max = (1i64 << (bits - 1)) - 1;
        (0..frames * ch)
            .map(|i| ((i as i64 * 7919 + i as i64 / ch as i64 * 31) % (2 * max) - max) as i32)
            .collect()
    }

    /// The samples of one channel as the fake recorded them (`Int32LSB`), without the
    /// trailing silence after the data.
    fn int32_channel(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn probe_reads_rates_depths_and_dsd() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let (f, src) = source(Spec::default());
        let out = AsioOutput::with_source(src, "").unwrap();
        let p = out.probe();
        assert_eq!(p.outputs, 2);
        assert!(p.rates.contains(&384_000) && !p.rates.contains(&768_000));
        assert_eq!(p.bits(), vec![16, 24, 32]);
        assert!(p.dsd.dop);
        assert_eq!(p.dsd.native, vec![NativeFormat::U8]);
        // DoP at 384 kHz × 16 = 6.144 MHz; native up to DSD256 (11.2896 MHz).
        assert_eq!(p.dsd.max_rate, Some(11_289_600));
        assert!(out.supports(96_000, 24) && !out.supports(96_000, 20));
        assert!(out.exclusive());
        assert_eq!(out.name(), "asio · Fake");
        assert!(
            !f.state.lock().unwrap().dsd,
            "probing leaves the driver in PCM"
        );
        drop(out);
        assert_eq!(
            f.state.lock().unwrap().refs,
            0,
            "the driver reference is released"
        );
    }

    #[test]
    fn unknown_driver_and_refused_init_are_clean_errors() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let (_f, src) = source(Spec::default());
        let e = AsioOutput::with_source(Arc::clone(&src), "Nope")
            .err()
            .unwrap();
        assert!(matches!(e, OutputError::NoDevice(_)), "{e}");
        let (_f, src) = source(Spec {
            init_ok: false,
            ..Spec::default()
        });
        let e = AsioOutput::with_source(src, "Fake").err().unwrap();
        assert!(e.to_string().contains("no hardware"), "{e}");
        let empty: Arc<dyn DriverSource> = Arc::new(FakeSource(Vec::new()));
        assert!(AsioOutput::with_source(empty, "").is_err());
    }

    #[test]
    fn pcm_reaches_the_driver_buffers_bit_exact_and_gapless() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let (f, src) = source(Spec::default());
        let mut out = AsioOutput::with_source(src, "Fake").unwrap();
        let fmt = OutputFormat::pcm(96_000, 24, 2);
        assert_eq!(out.open(fmt).unwrap(), fmt);
        assert_eq!(f.state.lock().unwrap().rate, 96_000.0);
        let clock = f.run(Duration::from_micros(200));
        let x = ramp(30_000, 2, 24);
        for chunk in x.chunks(2 * 777) {
            out.write(chunk).unwrap();
        }
        out.drain().unwrap();
        f.halt();
        clock.join().unwrap();
        assert_eq!(out.frames_written(), 30_000);
        assert_eq!(
            out.underruns(),
            0,
            "the queue never ran dry while data was coming"
        );
        for (c, bytes) in f.played().iter().enumerate() {
            let got = int32_channel(bytes);
            let want: Vec<i32> = x.iter().skip(c).step_by(2).map(|v| v << 8).collect();
            assert_eq!(&got[..want.len()], &want[..], "channel {c}");
            assert!(
                got[want.len()..].iter().all(|&v| v == 0),
                "silence after the data"
            );
        }
        assert!(f.state.lock().unwrap().output_ready_calls > 0);
        out.close();
        assert!(
            f.state.lock().unwrap().callbacks.is_none(),
            "buffers disposed"
        );
    }

    #[test]
    fn native_dsd_switches_the_driver_and_keeps_bit_order() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        for (dsd_type, reverse) in [
            (sys::ST_DSD_INT8_MSB1, false),
            (sys::ST_DSD_INT8_LSB1, true),
        ] {
            let (f, src) = source(Spec {
                dsd_type: Some(dsd_type),
                ..Spec::default()
            });
            let mut out = AsioOutput::with_source(src, "Fake").unwrap();
            let fmt = OutputFormat::dsd(2_822_400, DsdPacking::Native, 2);
            assert_eq!(out.open(fmt).unwrap(), fmt);
            {
                let s = f.state.lock().unwrap();
                assert!(s.dsd, "driver switched to DSD");
                assert_eq!(s.rate, 2_822_400.0, "the DSD bit rate is the sample rate");
            }
            let clock = f.run(Duration::from_micros(200));
            let words: Vec<i32> = (0..2 * 4000u32)
                .map(|i| (i.wrapping_mul(0x9E37_79B9) ^ 0x5A5A_1234) as i32)
                .collect();
            out.write(&words).unwrap();
            assert!(out.queued_frames() <= 4000);
            out.drain().unwrap();
            f.halt();
            clock.join().unwrap();
            for (c, played) in f.played().iter().enumerate() {
                let want: Vec<u8> = words
                    .iter()
                    .skip(c)
                    .step_by(2)
                    .flat_map(|w| w.to_be_bytes())
                    .map(|b| if reverse { b.reverse_bits() } else { b })
                    .collect();
                assert_eq!(&played[..want.len()], &want[..], "{dsd_type} ch {c}");
                let idle = if reverse { 0x69u8.reverse_bits() } else { 0x69 };
                assert!(
                    played[want.len()..].iter().all(|&b| b == idle),
                    "DSD idle after the data"
                );
            }
            out.close();
            assert!(!f.state.lock().unwrap().dsd, "back to PCM on close");
        }
    }

    #[test]
    fn dop_needs_integer_channels_and_native_needs_a_dsd_driver() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let (_f, src) = source(Spec {
            pcm_type: sys::ST_FLOAT32_LSB,
            dsd_type: None,
            ..Spec::default()
        });
        let mut out = AsioOutput::with_source(src, "Fake").unwrap();
        assert!(!out.dsd_caps().any());
        let dop = OutputFormat::dsd(2_822_400, DsdPacking::Dop, 2);
        assert!(matches!(
            out.open(dop),
            Err(OutputError::Unsupported { .. })
        ));
        let native = OutputFormat::dsd(2_822_400, DsdPacking::Native, 2);
        let e = out.open(native).unwrap_err().to_string();
        assert!(e.contains("would not switch to DSD"), "{e}");
        // PCM through float channels is fine (24 bits are exact in f32).
        assert_eq!(out.open(OutputFormat::pcm(48_000, 24, 2)).unwrap().bits, 24);
        // A rate the driver lacks, or more channels than it has.
        assert!(out.open(OutputFormat::pcm(768_000, 24, 2)).is_err());
        assert!(out.open(OutputFormat::pcm(48_000, 24, 6)).is_err());
        // 16-bit channels open 16-bit.
        let (_f, src) = source(Spec {
            pcm_type: sys::ST_INT16_LSB,
            ..Spec::default()
        });
        let mut out = AsioOutput::with_source(src, "Fake").unwrap();
        assert_eq!(out.open(OutputFormat::pcm(44_100, 24, 2)).unwrap().bits, 16);
        assert!(!out.dsd_caps().dop);
    }

    #[test]
    fn pause_plays_silence_and_a_reset_request_reopens_the_stream() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let (f, src) = source(Spec::default());
        let mut out = AsioOutput::with_source(src, "Fake").unwrap();
        let fmt = OutputFormat::pcm(48_000, 24, 2);
        out.open(fmt).unwrap();
        out.write(&ramp(2_000, 2, 24)).unwrap();
        out.pause(true).unwrap();
        // Paused: the clock runs, the queue stays.
        for i in 0..6 {
            f.state.lock().unwrap().started = true;
            f.tick(i & 1);
        }
        assert_eq!(out.queued_frames(), 2_000);
        assert!(f.played()[0].iter().all(|&b| b == 0));
        out.pause(false).unwrap();
        f.tick(0);
        assert_eq!(out.queued_frames(), 2_000 - 256);
        // The driver asks for a reset; the next write reopens it.
        f.request_reset();
        out.write(&ramp(100, 2, 24)).unwrap();
        assert_eq!(out.format(), Some(fmt));
        assert_eq!(
            out.queued_frames(),
            100,
            "the reopened stream starts with the new data"
        );
        out.discard().unwrap();
        assert_eq!(out.queued_frames(), 0);
    }

    #[test]
    fn devices_list_every_installed_driver() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let a = Fake::new(Spec::default());
        let b = Fake::new(Spec {
            init_ok: false,
            ..Spec::default()
        });
        let src: Arc<dyn DriverSource> = Arc::new(FakeSource(vec![
            ("Good".into(), a),
            ("Unplugged".into(), b),
        ]));
        let list = list_devices_from(&src);
        assert_eq!(list.len(), 1, "a driver that will not start is not listed");
        assert_eq!(list[0].backend, "asio");
        assert_eq!(list[0].id, "Good");
        assert!(list[0].default && list[0].exclusive);
        assert!(list[0].dsd.dop && !list[0].dsd.native.is_empty());
    }

    #[test]
    fn a_listing_never_loads_the_driver_playback_holds_again() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let (f, src) = source(Spec::default());
        let out = AsioOutput::with_source(Arc::clone(&src), "Fake")
            .unwrap()
            .hold();
        let loads = f.state.lock().unwrap().refs;
        let list = list_devices_from(&src);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "Fake");
        assert_eq!(
            f.state.lock().unwrap().refs,
            loads,
            "listed from the held entry"
        );
        // Let go: the next listing probes it again.
        drop(out);
        let _ = list_devices_from(&src);
        assert!(IN_USE.lock().unwrap().is_empty());
    }
}
