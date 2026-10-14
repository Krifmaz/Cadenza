//! [`EndpointOutput`]: an [`Output`] that streams to a Cadenza endpoint over TCP.
//!
//! `write` sends a block and then blocks until the endpoint's acknowledgements say fewer
//! than `high_water` frames are outstanding (sent − played), so the engine is paced by the
//! endpoint's DAC clock exactly as it would be by a local device's. The device id is
//! `endpoint://<name>` (mDNS instance name), `endpoint://<host>:<port>` or either followed
//! by `/<remote device id>` to pick one of the endpoint's devices.

use super::discovery;
use super::protocol::{Ack, Caps, Hello, Message, Open};
use super::PROTOCOL_VERSION;
use crate::output::{DeviceInfo, DsdCaps, Output, OutputError, OutputFormat};
use std::io::{self, BufReader, BufWriter};
use std::net::{Shutdown, SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Outstanding frames the sender keeps at the endpoint (jitter buffer + device queue).
const HIGH_WATER_SECONDS: f64 = 0.5;
/// Connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
/// How long to wait for an ack before declaring the endpoint gone.
const ACK_TIMEOUT: Duration = Duration::from_secs(5);

/// The pacing rule, separated from the socket so it can be simulated: send while fewer
/// than `high_water` frames are outstanding, where outstanding = sent − played at the
/// endpoint (its clock, from the acks).
#[derive(Debug, Clone)]
pub struct Pacer {
    high_water: u64,
    sent: u64,
    played: u64,
    queued: u64,
}

impl Pacer {
    /// A pacer keeping at most `high_water` frames outstanding.
    pub fn new(high_water: u64) -> Self {
        Self {
            high_water,
            sent: 0,
            played: 0,
            queued: 0,
        }
    }
    /// Frames sent but not yet played by the endpoint.
    pub fn outstanding(&self) -> u64 {
        self.sent.saturating_sub(self.played)
    }
    /// Whether another block may go out now.
    pub fn may_send(&self) -> bool {
        self.outstanding() < self.high_water
    }
    /// A block of `frames` was sent.
    pub fn sent(&mut self, frames: u64) {
        self.sent += frames;
    }
    /// An ack arrived with the endpoint's clock and queue.
    pub fn on_ack(&mut self, played: u64, queued: u64) {
        self.played = played.max(self.played);
        self.queued = queued;
    }
    /// Frames the endpoint reports as queued (jitter buffer + device).
    pub fn queued(&self) -> u64 {
        self.queued
    }
    /// Frames sent so far.
    pub fn total_sent(&self) -> u64 {
        self.sent
    }
    /// Forget everything (after discard / reopen).
    pub fn reset(&mut self) {
        self.sent = 0;
        self.played = 0;
        self.queued = 0;
    }
}

/// Parse `name`, `host:port`, `name/device` or `host:port/device`.
fn split_target(spec: &str) -> (&str, &str) {
    let spec = spec.trim().trim_start_matches("endpoint://");
    match spec.split_once('/') {
        Some((t, d)) => (t, d),
        None => (spec, ""),
    }
}

/// Resolve the target to a socket address: literal `host:port`, else mDNS by name.
fn resolve(target: &str) -> Result<SocketAddr, OutputError> {
    if let Ok(a) = target.parse::<SocketAddr>() {
        return Ok(a);
    }
    if target.contains(':') {
        if let Ok(mut it) = target.to_socket_addrs() {
            if let Some(a) = it.next() {
                return Ok(a);
            }
        }
    }
    discovery::find(target, Duration::from_secs(3)).ok_or_else(|| {
        OutputError::NoDevice(format!("endpoint '{target}' not found on the network"))
    })
}

/// What arrives from the endpoint.
enum Incoming {
    Ack(Ack),
    Opened(OutputFormat),
    Drained,
    Error(String),
    Gone(String),
}

/// See the module documentation.
pub struct EndpointOutput {
    label: String,
    remote_device: String,
    caps: Caps,
    device: DeviceInfo,
    writer: Arc<Mutex<BufWriter<TcpStream>>>,
    /// A handle on the socket to shut it down on drop (the reader thread holds a clone).
    socket: TcpStream,
    rx: Receiver<Incoming>,
    _reader: thread::JoinHandle<()>,
    format: Option<OutputFormat>,
    pacer: Pacer,
    seq: u64,
    position: u64,
    frames_written: u64,
    last_ack: Option<Ack>,
}

impl EndpointOutput {
    /// Connect to `spec` (`name`, `host:port`, optionally `/<device>`), say hello and
    /// fetch the endpoint's capabilities. Nothing is opened yet.
    pub fn connect(spec: &str) -> Result<Self, OutputError> {
        let (target, remote_device) = split_target(spec);
        let addr = resolve(target)?;
        let stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
            .map_err(|e| OutputError::NoDevice(format!("endpoint {target} ({addr}): {e}")))?;
        let _ = stream.set_nodelay(true);
        let mut reader = BufReader::new(stream.try_clone().map_err(io_err)?);
        let socket = stream.try_clone().map_err(io_err)?;
        let mut writer = BufWriter::new(stream);
        Message::Hello(Hello {
            version: PROTOCOL_VERSION,
            name: format!(
                "Cadenza on {}",
                gethostname::gethostname().to_string_lossy()
            ),
        })
        .write_to(&mut writer)
        .map_err(io_err)?;
        let caps = match Message::read_from(&mut reader).map_err(io_err)? {
            Message::Caps(c) => c,
            Message::Error(e) => return Err(OutputError::Backend(format!("endpoint: {e}"))),
            other => {
                return Err(OutputError::Backend(format!(
                    "endpoint: expected Caps, got {other:?}"
                )))
            }
        };
        let device = caps
            .devices
            .iter()
            .find(|d| {
                if remote_device.is_empty() {
                    d.default
                } else {
                    d.id == remote_device
                }
            })
            .or_else(|| caps.devices.first())
            .cloned()
            .ok_or_else(|| {
                OutputError::NoDevice(format!("endpoint {target} has no output devices"))
            })?;
        let (tx, rx) = mpsc::channel();
        let reader_thread = thread::Builder::new()
            .name("cadenza-endpoint-rx".into())
            .spawn(move || loop {
                match Message::read_from(&mut reader) {
                    Ok(Message::Ack(a)) => {
                        if tx.send(Incoming::Ack(a)).is_err() {
                            break;
                        }
                    }
                    Ok(Message::Opened(f)) => {
                        let _ = tx.send(Incoming::Opened(f));
                    }
                    Ok(Message::Drained) => {
                        let _ = tx.send(Incoming::Drained);
                    }
                    Ok(Message::Error(e)) => {
                        let _ = tx.send(Incoming::Error(e));
                    }
                    Ok(other) => log::debug!("endpoint: ignoring {other:?}"),
                    Err(e) => {
                        let _ = tx.send(Incoming::Gone(e.to_string()));
                        break;
                    }
                }
            })
            .map_err(io_err)?;
        let label = format!("endpoint · {} · {}", caps.name, device.name);
        Ok(Self {
            label,
            remote_device: device.id.clone(),
            caps,
            device,
            writer: Arc::new(Mutex::new(writer)),
            socket,
            rx,
            _reader: reader_thread,
            format: None,
            pacer: Pacer::new(0),
            seq: 0,
            position: 0,
            frames_written: 0,
            last_ack: None,
        })
    }

    /// The endpoint's capabilities as received.
    pub fn caps(&self) -> &Caps {
        &self.caps
    }
    /// The remote device in use.
    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }
    /// The last acknowledgement (clock, queue, underruns).
    pub fn last_ack(&self) -> Option<Ack> {
        self.last_ack
    }

    fn send(&self, m: &Message) -> Result<(), OutputError> {
        m.write_to(&mut *self.writer.lock().unwrap())
            .map_err(io_err)
    }

    /// Handle one incoming message; returns it for the caller to match on.
    fn apply(&mut self, m: &Incoming) {
        if let Incoming::Ack(a) = m {
            self.pacer.on_ack(a.played, a.queued);
            if let Some(prev) = self.last_ack {
                if a.underruns > prev.underruns {
                    log::warn!(
                        "endpoint: {} underrun(s) at the endpoint (total {})",
                        a.underruns - prev.underruns,
                        a.underruns
                    );
                }
            }
            self.last_ack = Some(*a);
        }
    }

    /// Drain whatever has arrived without blocking.
    fn poll(&mut self) -> Result<(), OutputError> {
        while let Ok(m) = self.rx.try_recv() {
            self.apply(&m);
            if let Incoming::Gone(e) = m {
                return Err(OutputError::Backend(format!(
                    "endpoint connection lost: {e}"
                )));
            }
        }
        Ok(())
    }

    /// Block until `pred` is satisfied by an incoming message or `timeout` passes.
    fn wait_for(
        &mut self,
        timeout: Duration,
        mut pred: impl FnMut(&Incoming) -> bool,
    ) -> Result<(), OutputError> {
        let deadline = Instant::now() + timeout;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(OutputError::Backend(
                    "endpoint: timed out waiting for a reply".into(),
                ));
            }
            match self.rx.recv_timeout(deadline - now) {
                Ok(m) => {
                    self.apply(&m);
                    match &m {
                        Incoming::Gone(e) => {
                            return Err(OutputError::Backend(format!(
                                "endpoint connection lost: {e}"
                            )))
                        }
                        Incoming::Error(e) => {
                            return Err(OutputError::Backend(format!("endpoint: {e}")))
                        }
                        _ => {}
                    }
                    if pred(&m) {
                        return Ok(());
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(OutputError::Backend(
                        "endpoint: timed out waiting for a reply".into(),
                    ))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(OutputError::Backend("endpoint connection lost".into()))
                }
            }
        }
    }
}

impl Drop for EndpointOutput {
    fn drop(&mut self) {
        self.close();
        // Ends the reader thread and lets the endpoint see EOF.
        let _ = self.socket.shutdown(Shutdown::Both);
    }
}

fn io_err(e: io::Error) -> OutputError {
    OutputError::Backend(e.to_string())
}

impl Output for EndpointOutput {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        self.send(&Message::Open(Open {
            device: self.remote_device.clone(),
            format,
        }))?;
        let mut got = None;
        self.wait_for(Duration::from_secs(10), |m| match m {
            Incoming::Opened(f) => {
                got = Some(*f);
                true
            }
            _ => false,
        })?;
        let got = got.unwrap();
        self.format = Some(got);
        self.pacer = Pacer::new((got.rate as f64 * HIGH_WATER_SECONDS) as u64);
        self.seq = 0;
        self.position = 0;
        self.last_ack = None;
        Ok(got)
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        let d = &self.device;
        (d.rates.is_empty() || d.rates.contains(&rate))
            && (d.bits.is_empty() || d.bits.contains(&bits))
    }

    fn dsd_caps(&self) -> DsdCaps {
        self.device.dsd.clone()
    }

    fn exclusive(&self) -> bool {
        self.device.exclusive
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        let fmt = self
            .format
            .ok_or_else(|| OutputError::Backend("endpoint stream not open".into()))?;
        let n = (frames.len() / fmt.channels.max(1) as usize) as u64;
        self.poll()?;
        // Pace: wait for acks until the endpoint has room.
        while !self.pacer.may_send() {
            self.wait_for(ACK_TIMEOUT, |m| matches!(m, Incoming::Ack(_)))?;
        }
        self.send(&Message::Frames {
            seq: self.seq,
            position: self.position,
            words: frames.to_vec(),
        })?;
        self.seq += 1;
        self.position += n;
        self.frames_written += n;
        self.pacer.sent(n);
        Ok(())
    }

    fn pause(&mut self, paused: bool) -> Result<(), OutputError> {
        self.send(&Message::Pause(paused))
    }

    fn drain(&mut self) -> Result<(), OutputError> {
        if self.format.is_none() {
            return Ok(());
        }
        self.send(&Message::Drain)?;
        self.wait_for(Duration::from_secs(30), |m| matches!(m, Incoming::Drained))?;
        self.pacer.reset();
        Ok(())
    }

    fn discard(&mut self) -> Result<(), OutputError> {
        self.send(&Message::Discard)?;
        // Everything outstanding is gone at the endpoint: do not wait for it to "play".
        self.pacer.reset();
        Ok(())
    }

    fn close(&mut self) {
        if self.format.take().is_some() {
            let _ = self.send(&Message::Close);
        }
        self.pacer.reset();
    }

    fn queued_frames(&self) -> usize {
        self.pacer.outstanding() as usize
    }

    fn frames_written(&self) -> u64 {
        self.frames_written
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }
}

/// Endpoints found on the network so far, as devices of the `endpoint` backend:
/// `endpoint://<name>` for each endpoint's default device. Discovery runs in the
/// background from the first call; the first listing waits up to a second.
pub fn list_devices() -> Vec<DeviceInfo> {
    discovery::known(Duration::from_secs(1))
        .into_iter()
        .map(|(name, addr)| DeviceInfo {
            backend: "endpoint".into(),
            id: format!("endpoint://{name}"),
            name: format!("{name} ({addr})"),
            rates: Vec::new(),
            bits: Vec::new(),
            channels: None,
            default: false,
            exclusive: true,
            dsd: DsdCaps::NONE,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_specs_split_into_endpoint_and_device() {
        assert_eq!(split_target("endpoint://kitchen"), ("kitchen", ""));
        assert_eq!(split_target("kitchen/hw:1,0"), ("kitchen", "hw:1,0"));
        assert_eq!(
            split_target("endpoint://10.0.0.5:8778/hw:CARD=DAC,DEV=0"),
            ("10.0.0.5:8778", "hw:CARD=DAC,DEV=0")
        );
        assert_eq!(resolve("127.0.0.1:1").unwrap().port(), 1);
    }

    #[test]
    fn pacer_keeps_outstanding_below_high_water() {
        let mut p = Pacer::new(1000);
        assert!(p.may_send());
        p.sent(600);
        p.sent(600);
        assert!(!p.may_send());
        assert_eq!(p.outstanding(), 1200);
        p.on_ack(300, 900);
        assert_eq!(p.outstanding(), 900);
        assert!(p.may_send());
        // Acks never move the clock backwards.
        p.on_ack(100, 0);
        assert_eq!(p.outstanding(), 900);
        p.reset();
        assert_eq!(p.outstanding(), 0);
    }
}
