//! Recording the measurement microphone.
//!
//! [`Capture`] is what a measurement needs from an input: start at about a rate, stop and
//! hand over the samples. The real one uses cpal (any USB measurement microphone the OS
//! sees); [`SimulatedRoom`] is a room in software for the tests.

use std::sync::{Arc, Mutex};

/// A recording in progress.
pub trait Capture: Send {
    /// Start recording at (about) `rate` Hz; returns the rate actually used.
    fn start(&mut self, rate: u32) -> Result<u32, String>;
    /// What is about to be played (interleaved stereo at the capture rate). Real inputs
    /// ignore it; the simulation records it through its room.
    fn note_played(&mut self, _stereo: &[f64], _rate: u32) {}
    /// Stop and return the recording (the first input channel).
    fn stop(&mut self) -> Result<Vec<f64>, String>;
}

/// Opens a capture on an input device by name (empty: the default input).
pub type CaptureFactory = Box<dyn Fn(&str) -> Result<Box<dyn Capture>, String> + Send + Sync>;

/// Input devices the system offers (empty without cpal).
pub fn input_devices() -> Vec<String> {
    #[cfg(feature = "cpal")]
    {
        use cpal::traits::{DeviceTrait, HostTrait};
        cpal::default_host()
            .input_devices()
            .map(|it| it.filter_map(|d| d.name().ok()).collect())
            .unwrap_or_default()
    }
    #[cfg(not(feature = "cpal"))]
    {
        Vec::new()
    }
}

/// The capture factory for real microphones.
pub fn system() -> CaptureFactory {
    Box::new(|name: &str| -> Result<Box<dyn Capture>, String> {
        #[cfg(feature = "cpal")]
        {
            Ok(Box::new(cpal_capture::CpalCapture::new(name)))
        }
        #[cfg(not(feature = "cpal"))]
        {
            let _ = name;
            Err("this build has no audio input (the cpal feature is off)".into())
        }
    })
}

#[cfg(feature = "cpal")]
mod cpal_capture {
    use super::Capture;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};

    /// A cpal input stream, run on a thread of its own (streams are not `Send` on every
    /// platform).
    pub struct CpalCapture {
        name: String,
        stop: Option<mpsc::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
        samples: Arc<Mutex<Vec<f64>>>,
    }

    impl CpalCapture {
        pub fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                stop: None,
                thread: None,
                samples: Arc::default(),
            }
        }
    }

    impl Capture for CpalCapture {
        fn start(&mut self, rate: u32) -> Result<u32, String> {
            let (ready_tx, ready_rx) = mpsc::channel::<Result<u32, String>>();
            let (stop_tx, stop_rx) = mpsc::channel::<()>();
            let name = self.name.clone();
            let samples = Arc::clone(&self.samples);
            let thread = std::thread::Builder::new()
                .name("cadenza-mic".into())
                .spawn(move || {
                    let open = || -> Result<(cpal::Stream, u32), String> {
                        let host = cpal::default_host();
                        let device = if name.is_empty() {
                            host.default_input_device()
                        } else {
                            host.input_devices()
                                .map_err(|e| e.to_string())?
                                .find(|d| d.name().is_ok_and(|n| n == name))
                        }
                        .ok_or_else(|| format!("no input device {name:?}"))?;
                        // The wanted rate if the device offers it, else its default.
                        let config = device
                            .supported_input_configs()
                            .map_err(|e| e.to_string())?
                            .find(|c| {
                                c.min_sample_rate().0 <= rate && c.max_sample_rate().0 >= rate
                            })
                            .map(|c| c.with_sample_rate(cpal::SampleRate(rate)))
                            .map_or_else(
                                || device.default_input_config().map_err(|e| e.to_string()),
                                Ok,
                            )?;
                        let got = config.sample_rate().0;
                        let ch = config.channels() as usize;
                        let fmt = config.sample_format();
                        let cfg: cpal::StreamConfig = config.into();
                        let err = |e| log::warn!("microphone: {e}");
                        macro_rules! stream {
                            ($t:ty, $scale:expr) => {{
                                let s = Arc::clone(&samples);
                                device.build_input_stream(
                                    &cfg,
                                    move |data: &[$t], _| {
                                        let mut v = s.lock().unwrap();
                                        v.extend(data.iter().step_by(ch).map(|x| $scale(*x)));
                                    },
                                    err,
                                    None,
                                )
                            }};
                        }
                        let stream = match fmt {
                            SampleFormat::F32 => stream!(f32, |x: f32| x as f64),
                            SampleFormat::I16 => stream!(i16, |x: i16| x as f64 / 32768.0),
                            SampleFormat::I32 => {
                                stream!(i32, |x: i32| x as f64 / 2_147_483_648.0)
                            }
                            other => return Err(format!("microphone sample format {other:?}")),
                        }
                        .map_err(|e| e.to_string())?;
                        stream.play().map_err(|e| e.to_string())?;
                        Ok((stream, got))
                    };
                    match open() {
                        Ok((stream, got)) => {
                            let _ = ready_tx.send(Ok(got));
                            let _ = stop_rx.recv();
                            drop(stream);
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                        }
                    }
                })
                .map_err(|e| e.to_string())?;
            self.samples.lock().unwrap().clear();
            let got = ready_rx
                .recv()
                .map_err(|_| "microphone thread ended".to_string())??;
            self.stop = Some(stop_tx);
            self.thread = Some(thread);
            Ok(got)
        }

        fn stop(&mut self) -> Result<Vec<f64>, String> {
            if let Some(s) = self.stop.take() {
                let _ = s.send(());
            }
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
            Ok(std::mem::take(&mut *self.samples.lock().unwrap()))
        }
    }
}

/// A room in software: what the microphone "hears" is the played stereo through the left
/// and right speakers' impulse responses, plus a little noise and some latency.
#[derive(Clone)]
pub struct SimulatedRoom {
    rate: u32,
    left: Arc<Vec<f64>>,
    right: Arc<Vec<f64>>,
}

impl SimulatedRoom {
    /// A room whose speakers have these impulse responses at `rate`.
    pub fn new(rate: u32, left: Vec<f64>, right: Vec<f64>) -> Self {
        Self {
            rate,
            left: Arc::new(left),
            right: Arc::new(right),
        }
    }

    /// A capture factory recording in this room.
    pub fn factory(&self) -> CaptureFactory {
        let room = self.clone();
        Box::new(move |_| {
            Ok(Box::new(SimCapture {
                room: room.clone(),
                rec: Arc::default(),
            }) as Box<dyn Capture>)
        })
    }
}

struct SimCapture {
    room: SimulatedRoom,
    rec: Arc<Mutex<Vec<f64>>>,
}

impl Capture for SimCapture {
    fn start(&mut self, _rate: u32) -> Result<u32, String> {
        Ok(self.room.rate)
    }

    fn note_played(&mut self, stereo: &[f64], _rate: u32) {
        let l: Vec<f64> = stereo.iter().step_by(2).copied().collect();
        let r: Vec<f64> = stereo.iter().skip(1).step_by(2).copied().collect();
        let a = cadenza_dsp::room::convolve(&l, &self.room.left);
        let b = cadenza_dsp::room::convolve(&r, &self.room.right);
        // 30 ms of output and input latency in front.
        let mut rec = vec![0.0; self.room.rate as usize * 3 / 100];
        let mut seed = 99u64;
        rec.extend(a.iter().zip(&b).map(|(x, y)| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x + y + ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2e-5
        }));
        *self.rec.lock().unwrap() = rec;
    }

    fn stop(&mut self) -> Result<Vec<f64>, String> {
        Ok(std::mem::take(&mut *self.rec.lock().unwrap()))
    }
}
