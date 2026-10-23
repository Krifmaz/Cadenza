//! The endpoint process: accepts one sender at a time, keeps a jitter buffer and plays
//! through a local [`Output`].
//!
//! Two threads per connection: the *reader* parses messages and pushes words into the
//! [`JitterBuffer`]; the *player* waits until the buffer holds `buffer_ms` of audio, then
//! moves 10 ms chunks into the output (which blocks on the device clock). On underrun it
//! plays silence, counts it and waits for the buffer to refill before resuming, so a
//! network hiccup costs one gap instead of a cascade. Every received block is answered
//! with an [`Ack`] carrying the player's clock position.

use super::protocol::{Ack, Caps, Message, Open, Schedule};
use super::sync::{Correction, Follower, Resampler, Timeline};
use super::{discovery, EndpointConfig, MIN_PROTOCOL_VERSION, PROTOCOL_VERSION};
use crate::output::{self, DeviceInfo, Encoding, Output, OutputError, OutputFormat};
use cadenza_dsp::dsd::DsdPacking;
use std::io::{self, BufReader, BufWriter};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Chunk the player hands to the output at a time.
const CHUNK_MS: u64 = 10;

/// The jitter buffer: words in, words out, with prefill and underrun accounting. Pure
/// data structure (no I/O) so the drift behaviour can be simulated in tests.
#[derive(Debug)]
pub struct JitterBuffer {
    /// Buffered interleaved device words, oldest first.
    words: std::collections::VecDeque<i32>,
    /// Interleaved channel count.
    channels: usize,
    /// Frames to hold before playback starts (and restarts after an underrun).
    prefill_frames: usize,
    /// Playing or (re)buffering.
    running: bool,
    /// Frames consumed by the player (its clock).
    played: u64,
    /// Frames received.
    received: u64,
    /// Times the buffer ran dry while playing.
    underruns: u64,
}

impl JitterBuffer {
    /// A buffer for `channels` channels holding `prefill_frames` before it starts.
    pub fn new(channels: usize, prefill_frames: usize) -> Self {
        Self {
            words: std::collections::VecDeque::new(),
            channels: channels.max(1),
            prefill_frames,
            running: false,
            played: 0,
            received: 0,
            underruns: 0,
        }
    }

    /// Frames waiting.
    pub fn queued_frames(&self) -> usize {
        self.words.len() / self.channels
    }
    /// Frames the player has consumed.
    pub fn played(&self) -> u64 {
        self.played
    }
    /// Frames received.
    pub fn received(&self) -> u64 {
        self.received
    }
    /// Times the buffer ran dry while playing.
    pub fn underruns(&self) -> u64 {
        self.underruns
    }
    /// Playing (as opposed to filling)?
    pub fn running(&self) -> bool {
        self.running
    }

    /// Add received words.
    pub fn push(&mut self, words: &[i32]) {
        self.words.extend(words.iter().copied());
        self.received += (words.len() / self.channels) as u64;
        if !self.running && self.queued_frames() >= self.prefill_frames {
            self.running = true;
        }
    }

    /// Take up to `frames` frames for the output: `Some(words)` while playing, `None`
    /// while (re)filling. A dry buffer while playing counts an underrun and switches to
    /// filling; the caller plays silence for that chunk.
    pub fn pop(&mut self, frames: usize) -> Option<Vec<i32>> {
        if !self.running {
            return None;
        }
        if self.words.is_empty() {
            self.underruns += 1;
            self.running = false;
            return None;
        }
        let n = frames.min(self.queued_frames()) * self.channels;
        let out: Vec<i32> = self.words.drain(..n).collect();
        self.played += (n / self.channels) as u64;
        Some(out)
    }

    /// The next word, if any.
    pub fn peek(&self) -> Option<i32> {
        self.words.front().copied()
    }

    /// Throw away up to `frames` frames (multi-room: skipping what is already late);
    /// returns how many.
    pub fn drop_frames(&mut self, frames: usize) -> usize {
        let n = frames.min(self.queued_frames());
        self.words.drain(..n * self.channels);
        self.played += n as u64;
        n
    }

    /// Take everything (drain): the remaining words, regardless of prefill.
    pub fn pop_all(&mut self) -> Vec<i32> {
        let out: Vec<i32> = self.words.drain(..).collect();
        self.played += (out.len() / self.channels) as u64;
        self.running = false;
        out
    }

    /// Drop everything queued (discard) and go back to filling. The played / received
    /// counts start over too, as the sender's do.
    pub fn clear(&mut self) {
        self.words.clear();
        self.running = false;
        self.played = 0;
        self.received = 0;
    }
}

/// Counters the endpoint exposes (tests, logs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EndpointStats {
    /// Connections accepted.
    pub connections: u64,
    /// Frames written to the local output in total.
    pub frames_played: u64,
    /// Underruns in total.
    pub underruns: u64,
    /// Streams opened.
    pub opens: u64,
}

/// Totals across every connection, for the endpoint's status.
#[derive(Default)]
struct Counters {
    /// Senders accepted.
    connections: AtomicU64,
    /// Frames written to the output.
    frames_played: AtomicU64,
    /// Underruns of the jitter buffer.
    underruns: AtomicU64,
    /// Streams opened by senders.
    opens: AtomicU64,
}

/// A running endpoint.
pub struct EndpointServer {
    /// The configuration the endpoint runs with.
    config: EndpointConfig,
    /// Name advertised to senders.
    name: String,
    /// Address the listener is bound to.
    addr: SocketAddr,
    /// Set to stop the accept thread.
    stop: Arc<AtomicBool>,
    /// Totals shown in the status.
    counters: Arc<Counters>,
    /// The thread accepting connections, joined on shutdown.
    accept_thread: Option<thread::JoinHandle<()>>,
    /// Last format a sender opened, for the status line.
    last_format: Arc<Mutex<Option<OutputFormat>>>,
    /// The mDNS advertisement; dropping it withdraws the endpoint.
    _mdns: Option<discovery::Advertisement>,
}

impl EndpointServer {
    /// Bind, advertise and start accepting in the background.
    pub fn start(config: EndpointConfig) -> io::Result<Self> {
        let listener = TcpListener::bind(&config.listen)?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let name = config.effective_name();
        let stop = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Counters::default());
        let last_format = Arc::new(Mutex::new(None));
        let mdns = if config.mdns {
            match discovery::Advertisement::start(&name, addr.port()) {
                Ok(a) => Some(a),
                Err(e) => {
                    log::warn!("endpoint: mDNS advertisement failed: {e}");
                    None
                }
            }
        } else {
            None
        };
        let accept_thread = {
            let stop = Arc::clone(&stop);
            let counters = Arc::clone(&counters);
            let config = config.clone();
            let name = name.clone();
            let last_format = Arc::clone(&last_format);
            let busy = Arc::new(AtomicBool::new(false));
            thread::Builder::new()
                .name("cadenza-endpoint".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, peer)) => {
                                counters.connections.fetch_add(1, Ordering::Relaxed);
                                let _ = stream.set_nonblocking(false);
                                let _ = stream.set_nodelay(true);
                                // One sender at a time: a second one is told so and dropped.
                                if busy.swap(true, Ordering::AcqRel) {
                                    log::warn!(
                                        "endpoint: refusing {peer}: another sender is connected"
                                    );
                                    let _ = Message::Error(
                                        "endpoint busy: another sender is connected".into(),
                                    )
                                    .write_to(&mut &stream);
                                    continue;
                                }
                                log::info!("endpoint: connection from {peer}");
                                let session = Session {
                                    config: config.clone(),
                                    name: name.clone(),
                                    counters: Arc::clone(&counters),
                                    last_format: Arc::clone(&last_format),
                                };
                                let busy_flag = Arc::clone(&busy);
                                let spawned = thread::Builder::new()
                                    .name("cadenza-endpoint-session".into())
                                    .spawn(move || {
                                        if let Err(e) = session.serve(stream) {
                                            if e.kind() != io::ErrorKind::UnexpectedEof {
                                                log::warn!(
                                                    "endpoint: session with {peer} ended: {e}"
                                                );
                                            }
                                        }
                                        log::info!("endpoint: {peer} disconnected");
                                        busy_flag.store(false, Ordering::Release);
                                    });
                                if spawned.is_err() {
                                    busy.store(false, Ordering::Release);
                                }
                            }
                            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                                thread::sleep(Duration::from_millis(50));
                            }
                            Err(e) => {
                                log::error!("endpoint: accept: {e}");
                                thread::sleep(Duration::from_millis(500));
                            }
                        }
                    }
                })?
        };
        Ok(Self {
            config,
            name,
            addr,
            stop,
            counters,
            accept_thread: Some(accept_thread),
            last_format,
            _mdns: mdns,
        })
    }

    /// Advertised name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Bound address (useful with port 0 in tests).
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }
    /// The configuration.
    pub fn config(&self) -> &EndpointConfig {
        &self.config
    }
    /// Counters so far.
    pub fn stats(&self) -> EndpointStats {
        EndpointStats {
            connections: self.counters.connections.load(Ordering::Relaxed),
            frames_played: self.counters.frames_played.load(Ordering::Relaxed),
            underruns: self.counters.underruns.load(Ordering::Relaxed),
            opens: self.counters.opens.load(Ordering::Relaxed),
        }
    }
    /// The format the current / last sender opened.
    pub fn last_format(&self) -> Option<OutputFormat> {
        *self.last_format.lock().unwrap()
    }

    /// Block until the process is killed.
    pub fn run_forever(mut self) {
        if let Some(t) = self.accept_thread.take() {
            let _ = t.join();
        }
    }

    /// Stop accepting; a session in progress finishes when its sender disconnects.
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.accept_thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for EndpointServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Devices the endpoint offers: its configured default first, then everything local.
pub fn local_devices(config: &EndpointConfig) -> Vec<DeviceInfo> {
    let backend = if config.backend.is_empty() {
        output::default_backend().to_string()
    } else {
        config.backend.clone()
    };
    let mut all: Vec<DeviceInfo> = output::list_devices()
        .into_iter()
        .filter(|d| d.backend != "endpoint")
        .collect();
    let default_pos = all.iter().position(|d| {
        d.backend == backend && (d.id == config.device || (config.device.is_empty() && d.default))
    });
    let mut out = Vec::new();
    if let Some(i) = default_pos {
        let mut d = all.remove(i);
        d.default = true;
        out.push(d);
    } else if let Ok(o) = output::open_backend(&backend, &config.device) {
        // Not in the listing (e.g. `default`): describe it from the opened output.
        out.push(DeviceInfo {
            backend: backend.clone(),
            id: config.device.clone(),
            name: o.name(),
            rates: output::STANDARD_RATES
                .iter()
                .copied()
                .filter(|r| o.supports(*r, 24) || o.supports(*r, 32) || o.supports(*r, 16))
                .collect(),
            bits: [16, 24, 32]
                .into_iter()
                .filter(|b| o.supports(44100, *b) || o.supports(48000, *b))
                .collect(),
            channels: None,
            default: true,
            exclusive: o.exclusive(),
            dsd: o.dsd_caps(),
        });
    }
    for d in &mut all {
        d.default = false;
    }
    out.extend(all);
    out
}

/// An open stream: shared state, the player thread (which returns the output when it
/// stops) and the format in use.
type OpenStream = (
    Arc<Stream>,
    thread::JoinHandle<Box<dyn Output>>,
    OutputFormat,
);

/// One sender's connection: what its reader thread needs.
struct Session {
    /// The endpoint's configuration.
    config: EndpointConfig,
    /// Name announced to the sender.
    name: String,
    /// Totals shared with the server.
    counters: Arc<Counters>,
    /// Last opened format, shared with the server's status.
    last_format: Arc<Mutex<Option<OutputFormat>>>,
}

/// State shared between the reader and the player thread of one open stream.
struct Stream {
    /// Received words waiting to be played.
    buffer: Mutex<JitterBuffer>,
    /// Wakes the player when words arrive or a flag changes.
    cv: Condvar,
    /// Set to end the player thread.
    stop: AtomicBool,
    /// Set by the sender's `Drain`: play out what is buffered, then report.
    drain: AtomicBool,
    /// Set by the player once a drain has played out.
    drained: AtomicBool,
    /// Set while the sender has paused the stream.
    paused: AtomicBool,
    /// Frames the output reports as queued inside the device.
    device_queued: AtomicU64,
    /// Last block sequence number received (echoed in clock acks).
    last_seq: AtomicU64,
    /// Nanoseconds since the session started, for the acks' clock field.
    started: Instant,
    /// The timeline to play on (version 2 sync), if the sender set one.
    schedule: Mutex<Option<Schedule>>,
    /// Stream position of the first frame after `Open` / `Discard`.
    base: Mutex<Option<u64>>,
    /// Set by `Discard`: the player starts its sync state over.
    reset: AtomicBool,
    /// The last presentation measurement: (stream position at the DAC, clock ns).
    presenting: Mutex<(u64, u64)>,
}

impl Session {
    /// Serve one sender until it disconnects: answer the handshake, then open, feed and close
    /// streams, each with its own player thread.
    fn serve(&self, stream: TcpStream) -> io::Result<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let writer = Arc::new(Mutex::new(BufWriter::new(stream)));
        let send = |m: &Message| -> io::Result<()> { m.write_to(&mut *writer.lock().unwrap()) };

        // Handshake.
        match Message::read_from(&mut reader)? {
            Message::Hello(h) => {
                if !(MIN_PROTOCOL_VERSION..=PROTOCOL_VERSION).contains(&h.version) {
                    send(&Message::Error(format!(
                        "protocol version {} not supported (endpoint speaks {PROTOCOL_VERSION})",
                        h.version
                    )))?;
                    return Ok(());
                }
                log::info!("endpoint: hello from '{}'", h.name);
            }
            other => {
                send(&Message::Error("expected Hello".into()))?;
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected Hello, got {other:?}"),
                ));
            }
        }
        send(&Message::Caps(Caps {
            name: self.name.clone(),
            version: PROTOCOL_VERSION,
            devices: local_devices(&self.config),
            buffer_ms: self.config.buffer_ms,
        }))?;

        let mut open: Option<OpenStream> = None;
        let mut output: Option<Box<dyn Output>> = None;
        let mut expected_seq = 0u64;
        let start = Instant::now();
        loop {
            let msg = match Message::read_from(&mut reader) {
                Ok(m) => m,
                Err(e) => {
                    if let Some((st, player, _)) = open.take() {
                        st.stop.store(true, Ordering::Relaxed);
                        st.cv.notify_all();
                        if let Ok(mut o) = player.join() {
                            let _ = o.discard();
                            o.close();
                        }
                    }
                    return Err(e);
                }
            };
            match msg {
                Message::Open(Open { device, format }) => {
                    if let Some((st, player, _)) = open.take() {
                        st.stop.store(true, Ordering::Relaxed);
                        st.cv.notify_all();
                        if let Ok(o) = player.join() {
                            output = Some(o);
                        }
                    }
                    let mut out = match output.take() {
                        Some(o) => o,
                        None => match self.open_output(&device) {
                            Ok(o) => o,
                            Err(e) => {
                                send(&Message::Error(e.to_string()))?;
                                continue;
                            }
                        },
                    };
                    let got = match out.open(format) {
                        Ok(f) => f,
                        Err(e) => {
                            send(&Message::Error(e.to_string()))?;
                            output = Some(out);
                            continue;
                        }
                    };
                    self.counters.opens.fetch_add(1, Ordering::Relaxed);
                    *self.last_format.lock().unwrap() = Some(got);
                    log::info!("endpoint: opened {got} on {}", out.name());
                    let prefill = (got.rate as u64 * self.config.buffer_ms as u64 / 1000) as usize;
                    let st = Arc::new(Stream {
                        buffer: Mutex::new(JitterBuffer::new(got.channels as usize, prefill)),
                        cv: Condvar::new(),
                        stop: AtomicBool::new(false),
                        drain: AtomicBool::new(false),
                        drained: AtomicBool::new(false),
                        paused: AtomicBool::new(false),
                        device_queued: AtomicU64::new(0),
                        last_seq: AtomicU64::new(0),
                        started: start,
                        schedule: Mutex::new(None),
                        base: Mutex::new(None),
                        reset: AtomicBool::new(false),
                        presenting: Mutex::new((Ack::UNKNOWN, 0)),
                    });
                    let player = {
                        let st = Arc::clone(&st);
                        let counters = Arc::clone(&self.counters);
                        let writer = Arc::clone(&writer);
                        thread::Builder::new()
                            .name("cadenza-endpoint-player".into())
                            .spawn(move || play(st, out, got, counters, writer))?
                    };
                    open = Some((st, player, got));
                    expected_seq = 0;
                    send(&Message::Opened(got))?;
                }
                Message::Frames {
                    seq,
                    position,
                    words,
                } => {
                    let Some((st, _, _)) = &open else {
                        send(&Message::Error("no stream open".into()))?;
                        continue;
                    };
                    if seq != expected_seq {
                        log::warn!("endpoint: block {seq} after {expected_seq} (gap)");
                    }
                    expected_seq = seq + 1;
                    st.last_seq.store(seq, Ordering::Relaxed);
                    st.base.lock().unwrap().get_or_insert(position);
                    let ack = {
                        let mut b = st.buffer.lock().unwrap();
                        b.push(&words);
                        Ack {
                            seq,
                            received: b.received(),
                            played: b.played(),
                            queued: b.queued_frames() as u64
                                + st.device_queued.load(Ordering::Relaxed),
                            underruns: b.underruns(),
                            mono_ns: start.elapsed().as_nanos() as u64,
                            presenting: Ack::UNKNOWN,
                        }
                    };
                    st.cv.notify_all();
                    send(&Message::Ack(ack))?;
                }
                Message::Pause(p) => {
                    if let Some((st, _, _)) = &open {
                        st.paused.store(p, Ordering::Relaxed);
                        st.cv.notify_all();
                    }
                }
                Message::Discard => {
                    if let Some((st, _, _)) = &open {
                        st.buffer.lock().unwrap().clear();
                        *st.base.lock().unwrap() = None;
                        *st.presenting.lock().unwrap() = (Ack::UNKNOWN, 0);
                        st.reset.store(true, Ordering::Relaxed);
                        st.cv.notify_all();
                    }
                }
                Message::Sync { id, t0 } => {
                    // Answered at once: the round trip is the uncertainty.
                    send(&Message::SyncReply {
                        id,
                        t0,
                        server: start.elapsed().as_nanos() as u64,
                    })?;
                }
                Message::Schedule(sch) => {
                    if let Some((st, _, _)) = &open {
                        *st.schedule.lock().unwrap() = Some(sch);
                        st.cv.notify_all();
                    }
                }
                Message::Drain => {
                    if let Some((st, _, _)) = &open {
                        st.drained.store(false, Ordering::Relaxed);
                        st.drain.store(true, Ordering::Relaxed);
                        st.cv.notify_all();
                        // Wait for the player to report the drain done (bounded).
                        let deadline = Instant::now() + Duration::from_secs(30);
                        while !st.drained.load(Ordering::Relaxed) && Instant::now() < deadline {
                            thread::sleep(Duration::from_millis(5));
                        }
                    }
                    send(&Message::Drained)?;
                }
                Message::Close => {
                    if let Some((st, player, _)) = open.take() {
                        st.stop.store(true, Ordering::Relaxed);
                        st.cv.notify_all();
                        if let Ok(mut o) = player.join() {
                            o.close();
                            output = Some(o);
                        }
                    }
                }
                Message::Error(e) => log::warn!("endpoint: sender reports: {e}"),
                other => {
                    send(&Message::Error(format!("unexpected {other:?}")))?;
                }
            }
        }
    }

    /// Open the local output for a stream (the configured backend and `device`).
    fn open_output(&self, device: &str) -> Result<Box<dyn Output>, OutputError> {
        let backend = if self.config.backend.is_empty() {
            output::default_backend().to_string()
        } else {
            self.config.backend.clone()
        };
        // A device id from our Caps list may name another local backend.
        let devices = local_devices(&self.config);
        let (backend, device) = match devices
            .iter()
            .find(|d| !device.is_empty() && d.id == device)
        {
            Some(d) => (d.backend.clone(), d.id.clone()),
            None => (
                backend,
                if device.is_empty() {
                    self.config.device.clone()
                } else {
                    device.to_string()
                },
            ),
        };
        if backend == "endpoint" {
            return Err(OutputError::Backend(
                "an endpoint cannot chain to another endpoint".into(),
            ));
        }
        output::open_backend(&backend, &device)
    }
}

/// The player thread: jitter buffer → output. It also sends the *clock acks*: after every
/// chunk the device has taken, an [`Ack`] with the new `played` count goes out, so a sender
/// that has stopped sending (its high-water mark reached) still sees the DAC clock advance
/// and knows when it may continue. Those acks are what pace the sender; the per-block acks
/// from the reader only confirm receipt.
///
/// With a [`Schedule`] (multi-room) the player starts so that the scheduled position
/// reaches the DAC on time, and a follower keeps steering to the timeline: PCM through
/// the variable-ratio [`Resampler`], DSD by slipping whole words, big errors by a jump.
fn play(
    st: Arc<Stream>,
    mut out: Box<dyn Output>,
    format: OutputFormat,
    counters: Arc<Counters>,
    writer: Arc<Mutex<BufWriter<TcpStream>>>,
) -> Box<dyn Output> {
    let clock_ack = |st: &Stream| {
        let b = st.buffer.lock().unwrap();
        let (presenting, at) = *st.presenting.lock().unwrap();
        let ack = Ack {
            seq: st.last_seq.load(Ordering::Relaxed),
            received: b.received(),
            played: b.played(),
            queued: b.queued_frames() as u64 + st.device_queued.load(Ordering::Relaxed),
            underruns: b.underruns(),
            mono_ns: if presenting == Ack::UNKNOWN {
                st.started.elapsed().as_nanos() as u64
            } else {
                at
            },
            presenting,
        };
        drop(b);
        // A failed send means the connection is gone; the reader thread reports that.
        let _ = Message::Ack(ack).write_to(&mut *writer.lock().unwrap());
    };
    let chunk_frames = (format.rate as u64 * CHUNK_MS / 1000).max(1) as usize;
    let silence_word = if format.is_dsd() {
        // DSD silence is 0x69 bytes; DoP words carry it under the marker but a plain
        // 0x69696969 pattern is "silence" for both packings as far as the DAC is concerned.
        0x6969_6969u32 as i32
    } else {
        0
    };
    let silence = vec![silence_word; chunk_frames * format.channels as usize];
    let mut paused_device = false;
    // Underrun *events* already counted into `counters` (the jitter buffer counts events;
    // this loop writes many silence chunks per event and must not count each one).
    let mut underruns_seen = 0u64;
    let mut sync = SyncPlayer::new(format);
    loop {
        if st.stop.load(Ordering::Relaxed) {
            break;
        }
        let paused = st.paused.load(Ordering::Relaxed);
        if paused != paused_device {
            let _ = out.pause(paused);
            paused_device = paused;
            // The timeline does not survive a pause: start over on the next schedule.
            sync.restart();
        }
        if st.reset.swap(false, Ordering::Relaxed) {
            sync = SyncPlayer::new(format);
        }
        if st.drain.load(Ordering::Relaxed) {
            let rest = st.buffer.lock().unwrap().pop_all();
            if !rest.is_empty() && out.write(&rest).is_ok() {
                counters.frames_played.fetch_add(
                    (rest.len() / format.channels as usize) as u64,
                    Ordering::Relaxed,
                );
            }
            let _ = out.drain();
            st.device_queued.store(0, Ordering::Relaxed);
            clock_ack(&st);
            st.drain.store(false, Ordering::Relaxed);
            st.drained.store(true, Ordering::Relaxed);
            continue;
        }
        let schedule = *st.schedule.lock().unwrap();
        if let (Some(sch), false) = (schedule, paused) {
            match sync.step(&st, &mut *out, sch, chunk_frames) {
                Ok(Some(frames)) => {
                    counters
                        .frames_played
                        .fetch_add(frames as u64, Ordering::Relaxed);
                    st.device_queued
                        .store(out.queued_frames() as u64, Ordering::Relaxed);
                    clock_ack(&st);
                    continue;
                }
                Ok(None) => {
                    let b = st.buffer.lock().unwrap();
                    let _ = st.cv.wait_timeout(b, Duration::from_millis(5));
                    continue;
                }
                Err(e) => {
                    log::error!("endpoint: output write: {e}");
                    thread::sleep(Duration::from_millis(100));
                    continue;
                }
            }
        }
        let chunk = {
            let mut b = st.buffer.lock().unwrap();
            match b.pop(chunk_frames) {
                Some(c) => Some(c),
                None => {
                    if b.underruns() > 0 && !b.running() && b.queued_frames() == 0 {
                        // Ran dry: play one chunk of silence, then wait for refill. The
                        // event count follows the jitter buffer's, so the stats and the
                        // sender's acks agree.
                        let u = b.underruns();
                        if u > underruns_seen {
                            counters
                                .underruns
                                .fetch_add(u - underruns_seen, Ordering::Relaxed);
                            underruns_seen = u;
                        }
                        drop(b);
                        let _ = out.write(&silence);
                        clock_ack(&st);
                        // A real device paces the silence write; a non-pacing one would
                        // spin here, so give the network a moment to refill.
                        let b = st.buffer.lock().unwrap();
                        let _ = st.cv.wait_timeout(b, Duration::from_millis(2));
                        continue;
                    }
                    // Filling: wait for data (or a control event).
                    let _ = st.cv.wait_timeout(b, Duration::from_millis(20));
                    None
                }
            }
        };
        if let Some(words) = chunk {
            // While paused the device holds what we hand it (its queue), like a local output.
            match out.write(&words) {
                Ok(()) => {
                    counters.frames_played.fetch_add(
                        (words.len() / format.channels as usize) as u64,
                        Ordering::Relaxed,
                    );
                    st.device_queued
                        .store(out.queued_frames() as u64, Ordering::Relaxed);
                    clock_ack(&st);
                }
                Err(e) => {
                    log::error!("endpoint: output write: {e}");
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }
    let _ = out.discard();
    out
}

/// The player's multi-room state for one stream.
struct SyncPlayer {
    /// Format of the stream as opened on the device.
    format: OutputFormat,
    /// Interleaved channel count.
    channels: usize,
    /// DoP frames carry alternating markers: slips and silence come in pairs.
    dop: bool,
    /// Whether the first frame has been scheduled on the timeline.
    started: bool,
    /// The control loop that tracks the leader's clock.
    follower: Follower,
    /// Fine rate correction while following (created on the first correction).
    resampler: Option<Resampler>,
    /// Input frames consumed outside the resampler (start trims, drops, DSD).
    consumed: u64,
    /// Frames written to the output (silence included).
    out_written: u64,
    /// Since output frame `.0`, output frame `n` is stream position `n + .1` (the data
    /// run in progress; silence, slips and resampling end it).
    run: Option<(u64, i64)>,
    /// Our own DAC's clock: output frames presented over time. Its slope is our real
    /// sample rate, so a follower plays at (leader rate ÷ ours) from the start and the
    /// control loop only trims what is left.
    dac: Timeline,
}

impl SyncPlayer {
    /// Sync state for a stream opened as `format`.
    fn new(format: OutputFormat) -> Self {
        let dop = matches!(
            format.encoding,
            Encoding::Dsd {
                packing: DsdPacking::Dop,
                ..
            }
        );
        Self {
            format,
            channels: format.channels.max(1) as usize,
            dop,
            started: false,
            follower: Follower::new(format.rate),
            resampler: None,
            consumed: 0,
            out_written: 0,
            run: None,
            dac: Timeline::default(),
        }
    }

    /// Write data or silence, keeping the output → stream position map.
    fn write(
        &mut self,
        out: &mut dyn Output,
        words: &[i32],
        data_at: Option<u64>,
    ) -> Result<(), OutputError> {
        let n = (words.len() / self.channels) as u64;
        match (data_at, self.run) {
            (Some(p), Some((from, off)))
                if from as i64 + off <= p as i64 && self.out_written as i64 + off == p as i64 => {}
            (Some(p), _) => self.run = Some((self.out_written, p as i64 - self.out_written as i64)),
            (None, _) => self.run = None,
        }
        out.write(words)?;
        self.out_written += n;
        Ok(())
    }

    /// Wait for the timeline again (after a pause): the next step re-aligns by jumping.
    fn restart(&mut self) {
        self.follower.reset();
        self.dac.clear();
    }

    /// Smallest number of frames that can be dropped or inserted (2 for DoP, so the markers
    /// stay paired; 1 otherwise).
    fn unit(&self) -> usize {
        if self.dop {
            2
        } else {
            1
        }
    }

    /// Silence frames, keeping DoP's marker alternation with `next` (the next word).
    fn silence(&self, frames: usize, next: Option<i32>) -> Vec<i32> {
        let mut v = Vec::with_capacity(frames * self.channels);
        if self.dop {
            let first = next.map_or(0x05, |w| ((w as u32 >> 16) & 0xff) as u8);
            let other = if first == 0x05 { 0xfa } else { 0x05 };
            for i in 0..frames {
                let m = if i % 2 == 0 { first } else { other } as u32;
                v.extend(std::iter::repeat_n(
                    ((m << 16) | 0x6969) as i32,
                    self.channels,
                ));
            }
        } else {
            let w = if self.format.is_dsd() {
                0x6969_6969u32 as i32
            } else {
                0
            };
            v.resize(frames * self.channels, w);
        }
        v
    }

    /// `frames` rounded down to a whole number of units.
    fn round_unit(&self, frames: usize) -> usize {
        frames / self.unit() * self.unit()
    }

    /// Stream position of the next frame to reach the output.
    fn position(&self, base: u64) -> f64 {
        base as f64
            + self.consumed as f64
            + self.resampler.as_ref().map_or(0.0, |r| r.input_position())
    }

    /// One step: write something and return the frames written, or `None` to wait.
    fn step(
        &mut self,
        st: &Stream,
        out: &mut dyn Output,
        sch: Schedule,
        chunk: usize,
    ) -> Result<Option<usize>, OutputError> {
        let Some(base) = *st.base.lock().unwrap() else {
            return Ok(None);
        };
        let rate = self.format.rate as f64;
        let now_ns = st.started.elapsed().as_nanos() as f64;
        let queued = out.queued_frames() as u64;
        if self.started {
            self.dac
                .add(self.out_written.saturating_sub(queued), now_ns as i64);
        }
        // Our DAC's real frame period (the nominal one until measured).
        let own_ns = self
            .dac
            .anchor(1e9 / rate)
            .map_or(1e9 / rate, |a| a.frame_ns);
        let present_ns = now_ns + queued as f64 * own_ns;
        let pos = self.position(base);
        let target = sch.at_ns as f64 + (pos - sch.position as f64) * sch.frame_ns;
        let late_s = (present_ns - target) / 1e9;
        let ch = self.channels;
        let mut b = st.buffer.lock().unwrap();
        let next_word = b.peek();

        if !self.started {
            if !b.running() {
                return Ok(None); // still filling the jitter buffer
            }
            if late_s < 0.0 {
                // Early: silence up to the start (a chunk at a time).
                let n = self.round_unit(((-late_s * rate).round() as usize).min(chunk));
                if n == 0 || -late_s * rate < chunk as f64 {
                    self.started = true;
                }
                drop(b);
                if n > 0 {
                    let w = self.silence(n, next_word);
                    self.write(out, &w, None)?;
                }
                return Ok(Some(n));
            }
            // Late: skip what should already have played.
            let n = self.round_unit((late_s * rate).round() as usize);
            let got = b.drop_frames(n);
            self.consumed += got as u64;
            if got == n {
                self.started = true;
            }
            return Ok(Some(0));
        }

        let mut ratio = 1.0;
        if sch.follow {
            match self.follower.update(late_s, CHUNK_MS as f64 / 1000.0) {
                Correction::Drop(n) => {
                    let n = self.round_unit(n);
                    match &mut self.resampler {
                        // Frames the resampler holds come first: skip inside it.
                        Some(r) => r.skip(n),
                        None => {
                            let got = b.drop_frames(n);
                            self.consumed += got as u64;
                        }
                    }
                    return Ok(Some(0));
                }
                Correction::Insert(n) => {
                    drop(b);
                    let n = self.round_unit(n);
                    let w = self.silence(n, next_word);
                    self.write(out, &w, None)?;
                    return Ok(Some(n));
                }
                // Feed-forward: the leader's rate over ours, trimmed by the loop.
                Correction::Ratio(r) => ratio = r * own_ns / sch.frame_ns,
            }
            if self.format.is_dsd() {
                // DSD cannot be resampled: slip one unit when off by more than one.
                let unit_s = self.unit() as f64 / rate;
                if late_s > unit_s {
                    let got = b.drop_frames(self.unit());
                    self.consumed += got as u64;
                } else if late_s < -unit_s {
                    drop(b);
                    let w = self.silence(self.unit(), next_word);
                    self.write(out, &w, None)?;
                    return Ok(Some(self.unit()));
                }
                ratio = 1.0;
            }
        }

        if ratio == 1.0 && self.resampler.is_none() {
            let Some(words) = b.pop(chunk) else {
                return Ok(None);
            };
            drop(b);
            let n = words.len() / ch;
            let at = self.position(base) as u64;
            self.consumed += n as u64;
            self.write(out, &words, Some(at))?;
            self.measure(st, out);
            return Ok(Some(n));
        }

        // PCM follower: through the variable-ratio resampler.
        let r = self.resampler.get_or_insert_with(|| Resampler::new(ch));
        if r.pending_frames() < chunk + 64 {
            if let Some(words) = b.pop(chunk) {
                let scale = (1u64 << (self.format.bits - 1)) as f64;
                let input: Vec<f64> = words.iter().map(|w| *w as f64 / scale).collect();
                r.push(&input);
            }
        }
        drop(b);
        let mut buf = Vec::with_capacity(chunk * ch);
        let n = r.produce(ratio, chunk, &mut buf);
        if n == 0 {
            return Ok(None);
        }
        let scale = (1u64 << (self.format.bits - 1)) as f64;
        let (lo, hi) = (-scale, scale - 1.0);
        let words: Vec<i32> = buf
            .iter()
            .map(|v| (v * scale).round().clamp(lo, hi) as i32)
            .collect();
        // Resampled output has no exact position map (and only the leader's matters).
        self.write(out, &words, None)?;
        Ok(Some(n))
    }

    /// Record what the DAC presents now, for the acks (the leader's timeline): known only
    /// while the frame at the DAC belongs to the current data run.
    fn measure(&self, st: &Stream, out: &dyn Output) {
        let now = st.started.elapsed().as_nanos() as u64;
        let at_dac = self.out_written.saturating_sub(out.queued_frames() as u64);
        let presenting = match self.run {
            Some((from, off)) if at_dac >= from => (at_dac as i64 + off) as u64,
            _ => Ack::UNKNOWN,
        };
        *st.presenting.lock().unwrap() = (presenting, now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_buffer_prefills_and_counts_underruns() {
        let mut j = JitterBuffer::new(2, 100);
        assert_eq!(j.pop(10), None);
        j.push(&[0; 2 * 99]);
        assert!(!j.running());
        assert_eq!(j.pop(10), None); // still filling, no underrun counted
        assert_eq!(j.underruns(), 0);
        j.push(&[0; 2]);
        assert!(j.running());
        assert_eq!(j.pop(60).unwrap().len(), 120);
        assert_eq!(j.pop(60).unwrap().len(), 80);
        assert_eq!(j.played(), 100);
        assert_eq!(j.pop(10), None);
        assert_eq!(j.underruns(), 1);
        assert!(!j.running());
        j.push(&[1; 2 * 100]);
        assert_eq!(j.pop_all().len(), 200);
        assert_eq!(j.received(), 200);
    }

    /// Drift simulation: the endpoint's DAC clock runs `ppm` faster or slower than the
    /// sender's notion of time. The sender only sends while the endpoint reports less than
    /// its high-water mark outstanding (what `EndpointOutput` does from acks), so the
    /// buffer must neither run dry nor grow without bound over a long run.
    fn simulate(ppm: f64, seconds: f64) -> (u64, usize) {
        let rate = 48_000u64;
        let channels = 2usize;
        let prefill = (rate / 10) as usize; // 100 ms
        let high_water = (rate / 2) as usize; // sender keeps ≤ 0.5 s outstanding
        let block = 480usize; // 10 ms blocks
        let mut j = JitterBuffer::new(channels, prefill);
        let mut pacer = super::super::client::Pacer::new(high_water as u64);
        let dac_rate = rate as f64 * (1.0 + ppm * 1e-6);
        let mut dac_time = 0.0f64; // seconds of DAC consumption
        let mut consumed_frac = 0.0f64;
        let mut max_queued = 0usize;
        let steps = (seconds * 1000.0) as usize; // 1 ms steps of sender wall clock
        for _ in 0..steps {
            // Sender: fill while allowed.
            while pacer.may_send() {
                j.push(&vec![0; block * channels]);
                pacer.sent(block as u64);
                // The ack comes back immediately in this simulation.
                pacer.on_ack(j.played(), j.queued_frames() as u64);
            }
            // DAC: consume 1 ms of its own clock.
            dac_time += 0.001;
            consumed_frac += dac_rate * 0.001;
            let want = consumed_frac.floor() as usize;
            consumed_frac -= want as f64;
            if want > 0 {
                let _ = j.pop(want);
            }
            pacer.on_ack(j.played(), j.queued_frames() as u64);
            max_queued = max_queued.max(j.queued_frames());
        }
        let _ = dac_time;
        (j.underruns(), max_queued)
    }

    /// Real loopback over localhost: an endpoint with the null output, a client that
    /// streams PCM, then DoP, and drains. Every frame must reach the endpoint's output.
    #[test]
    fn loopback_over_localhost_with_the_null_output() {
        use crate::endpoint::EndpointOutput;
        use crate::output::Output;
        use cadenza_dsp::dsd::{DsdPacking, Packer};

        let mut server = EndpointServer::start(EndpointConfig {
            name: "test endpoint".into(),
            listen: "127.0.0.1:0".into(),
            backend: "null".into(),
            device: String::new(),
            buffer_ms: 20,
            mdns: false,
        })
        .unwrap();
        let addr = server.local_addr();
        let mut out = EndpointOutput::connect(&format!("endpoint://{addr}")).unwrap();
        assert_eq!(out.caps().name, "test endpoint");
        assert!(out.name().starts_with("endpoint · test endpoint"));
        assert!(
            out.device().dsd.dop,
            "the null device at the endpoint does DoP"
        );
        assert!(out.supports(705_600, 24));
        assert!(out.exclusive());

        // PCM: 2 s of 48 kHz stereo in 10 ms blocks (the null output is not realtime,
        // so pacing is exercised only by the acks, which is what we want to test).
        let fmt = OutputFormat::pcm(48_000, 24, 2);
        assert_eq!(out.open(fmt).unwrap(), fmt);
        let block = vec![1i32; 480 * 2];
        for _ in 0..200 {
            out.write(&block).unwrap();
        }
        out.drain().unwrap();
        assert_eq!(out.frames_written(), 96_000);
        let ack = out.last_ack().unwrap();
        assert_eq!(ack.received, 96_000);
        assert_eq!(server.stats().frames_played, 96_000);
        assert_eq!(server.last_format(), Some(fmt));

        // DoP DSD64 through the same connection: the endpoint opens its output in DSD and
        // the null output there verifies the marker sequence.
        let dsd = OutputFormat::dsd(2_822_400, DsdPacking::Dop, 2);
        assert_eq!(out.open(dsd).unwrap(), dsd);
        let mut packer = Packer::new(DsdPacking::Dop, 2);
        let bytes = vec![0x69u8; 2 * 1764]; // 10 ms of DSD64 per channel
        let mut words = Vec::new();
        for _ in 0..50 {
            words.clear();
            packer.push(&[&bytes, &bytes], &mut words);
            out.write(&words).unwrap();
        }
        out.drain().unwrap();
        assert_eq!(server.last_format(), Some(dsd));
        assert_eq!(server.stats().frames_played, 96_000 + 50 * 1764);
        // (The non-realtime null output consumes instantly, so the jitter buffer does run
        // dry here; underrun behaviour is covered by the drift simulation.)
        assert_eq!(server.stats().opens, 2);
        out.close();
        drop(out);
        server.shutdown();
        assert_eq!(server.stats().connections, 1);
        // A second sender while one is connected is refused with an error.
        let server2 = EndpointServer::start(EndpointConfig {
            listen: "127.0.0.1:0".into(),
            backend: "null".into(),
            mdns: false,
            ..EndpointConfig::default()
        })
        .unwrap();
        let first = EndpointOutput::connect(&server2.local_addr().to_string()).unwrap();
        let second = EndpointOutput::connect(&server2.local_addr().to_string());
        assert!(second.is_err(), "second sender must be refused");
        drop(first);
    }

    #[test]
    fn unsupported_format_is_reported_not_fatal() {
        use crate::endpoint::EndpointOutput;
        use crate::output::Output;
        use cadenza_dsp::dsd::DsdPacking;

        let server = EndpointServer::start(EndpointConfig {
            listen: "127.0.0.1:0".into(),
            backend: "null".into(),
            mdns: false,
            ..EndpointConfig::default()
        })
        .unwrap();
        let mut out = EndpointOutput::connect(&format!("{}", server.local_addr())).unwrap();
        // DSD2048 is above the null device's ceiling.
        let err = out
            .open(OutputFormat::dsd(90_316_800, DsdPacking::Native, 2))
            .unwrap_err();
        assert!(err.to_string().contains("not supported"), "{err}");
        // The connection survives and a good format opens afterwards.
        assert!(out.open(OutputFormat::pcm(44_100, 16, 2)).is_ok());
        assert!(EndpointOutput::connect("127.0.0.1:1").is_err());
    }

    #[test]
    fn ack_pacing_absorbs_clock_drift_without_resampling() {
        for ppm in [-500.0, -50.0, 0.0, 50.0, 500.0] {
            let (underruns, max_queued) = simulate(ppm, 600.0);
            assert_eq!(underruns, 0, "underruns at {ppm} ppm");
            // Bounded by the sender's high-water mark plus one block.
            assert!(
                max_queued <= 24_000 + 480,
                "queue grew to {max_queued} at {ppm} ppm"
            );
        }
    }
}
