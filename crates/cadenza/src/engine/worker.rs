//! The playback thread.
//!
//! One loop: take a command if there is one, then (while playing) decode a packet, push it
//! through the pipeline and write the result to the output. The output blocks when its
//! buffer is full, which paces everything to the device clock. Gapless playback falls out
//! of the design: when a track ends, the next decoder is opened and the pipeline keeps
//! running, so the filter history carries across the boundary.

use super::decoder::TrackDecoder;
use super::{Command, Event, PlayState, Shared, Status, Track};
use crate::config::{RateSetting, Settings};
use crate::output::{self, DsdCaps, Output, OutputFormat};
use cadenza_dsp::dsd::{DsdConfig, DsdPacking, DsdRate, OutputMode, Packer};
use cadenza_dsp::pipeline::PipelineError;
use cadenza_dsp::signal_path::{Endpoint, SignalPath, Stage};
use cadenza_dsp::{
    CoefficientCache, ConvolutionSpec, DitherKind, FilterKind, Pipeline, PipelineConfig, Volume,
};
use serde_json::json;
use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Bit-perfect DSD pass-through: the source's 1-bit stream packed for the device, nothing
/// else in between.
struct Passthrough {
    /// Packs the planar DSD bytes into device words (DoP or native).
    packer: Packer,
    /// The signal path reported for this chain.
    path: SignalPath,
}

/// `(rate, bits, channels, dsd)` of the current track, its label and its duration.
type TrackInfo = ((u32, u32, u32, bool), String, Option<f64>);

/// What the worker runs the decoded audio through.
#[allow(clippy::large_enum_variant)]
enum Chain {
    /// The DSP pipeline (PCM in, PCM or modulated DSD out).
    Pipeline(Box<Pipeline>),
    /// DSD bits straight to a DoP / native DSD output.
    Passthrough(Passthrough),
}

impl Chain {
    /// New EQ bands in place; `false` when the chain has to be rebuilt instead.
    fn set_eq(&mut self, bands: &[cadenza_dsp::Band], enabled: bool) -> bool {
        match self {
            Chain::Pipeline(p) => p.set_eq(bands, enabled),
            Chain::Passthrough(_) => false,
        }
    }

    /// Run decoded PCM through the pipeline (a pass-through chain takes no PCM).
    fn process_pcm(&mut self, pcm: &[f64]) -> Vec<i32> {
        match self {
            Chain::Pipeline(p) => p.process(pcm),
            Chain::Passthrough(_) => Vec::new(),
        }
    }

    /// Pack DSD bits for a pass-through output (a pipeline chain takes no DSD).
    fn process_dsd(&mut self, planar: &[Vec<u8>]) -> Vec<i32> {
        match self {
            Chain::Passthrough(p) => {
                let refs: Vec<&[u8]> = planar.iter().map(|c| c.as_slice()).collect();
                let mut out = Vec::new();
                p.packer.push(&refs, &mut out);
                out
            }
            Chain::Pipeline(_) => Vec::new(),
        }
    }

    /// Push out what the chain still holds at the end of the stream (filter tails, partial
    /// words).
    fn flush(&mut self) -> Vec<i32> {
        match self {
            Chain::Pipeline(p) => p.flush(),
            Chain::Passthrough(p) => {
                let mut out = Vec::new();
                p.packer.flush(&mut out);
                p.packer.reset();
                out
            }
        }
    }

    /// Clear the chain's history (after a seek or a stop).
    fn reset(&mut self) {
        match self {
            Chain::Pipeline(p) => p.reset(),
            Chain::Passthrough(p) => p.packer.reset(),
        }
    }

    /// Apply a new volume without rebuilding the chain.
    fn set_volume(&mut self, v: Volume) {
        if let Chain::Pipeline(p) = self {
            p.set_volume(v);
        }
    }

    /// Processing latency of the chain in seconds.
    fn latency_seconds(&self) -> f64 {
        match self {
            Chain::Pipeline(p) => p.latency_seconds(),
            Chain::Passthrough(_) => 0.0,
        }
    }

    /// The current signal path for the status.
    fn signal_path(&self) -> SignalPath {
        match self {
            Chain::Pipeline(p) => p.signal_path(),
            Chain::Passthrough(p) => p.path.clone(),
        }
    }

    /// `true` for DSD pass-through.
    fn is_passthrough(&self) -> bool {
        matches!(self, Chain::Passthrough(_))
    }
}

/// Open the next track this long before the current one ends, so HTTP streams are ready.
const PREFETCH_SECONDS: f64 = 10.0;

/// One decoded chunk.
enum Chunk {
    /// Interleaved PCM in `[-1, 1]`.
    Pcm(Vec<f64>),
    /// Planar DSD bytes, one buffer per channel.
    Dsd(Vec<Vec<u8>>),
}

/// State of the playback thread: the queue, the decoder, the chain and the device.
struct Worker {
    /// Commands from the engine handle.
    rx: Receiver<Command>,
    /// Status and event channels shared with the handle.
    shared: Arc<Shared>,
    /// Current settings.
    settings: Settings,
    /// Where designed filter coefficients are cached.
    cache: CoefficientCache,
    /// Pace the null output in real time (otherwise it runs as fast as it can).
    realtime: bool,
    /// Saves changed settings, when the host gave a way to.
    persist: Option<super::PersistFn>,

    /// The play queue.
    queue: Vec<Track>,
    /// Index in `queue` of the current track.
    index: Option<usize>,
    /// Transport state.
    state: PlayState,
    /// Volume as applied to the pipeline.
    volume: Volume,

    /// Decoder of the current track.
    decoder: Option<TrackDecoder>,
    /// `(rate, bits, channels, dsd)` of the current track, and its label.
    info: Option<TrackInfo>,
    /// The next track's decoder, opened ahead for gapless playback: `(queue index, decoder)`.
    prefetched: Option<(usize, TrackDecoder)>,
    /// What the decoded audio runs through.
    chain: Option<Chain>,
    /// The open device, if any.
    output: Option<Box<dyn Output>>,
    /// Name of the open device (for the status).
    output_name: String,
    /// Format the device was opened with.
    output_format: Option<OutputFormat>,
    /// Format that was asked for when the device was last opened. It differs from
    /// `output_format` when the device took another bit depth (or a shared stream);
    /// remembering it stops the next track of the same format reopening the device.
    output_wanted: Option<OutputFormat>,
    /// Rates the device reports it can open.
    output_rates: Vec<u32>,
    /// DSD formats the device reports it can take.
    output_dsd: DsdCaps,
    /// Whether the device is held exclusively.
    output_exclusive: bool,
    /// Where to pick up after the device went away mid-track (play resumes there).
    resume_at: Option<f64>,
    /// `(path, impulse)` so the file is only parsed once.
    convolution: Option<(String, Arc<ConvolutionSpec>)>,

    /// Last playback error, shown in the status.
    error: Option<String>,
    /// Tracks that finished or were stopped (lets clients tell a new run from the old one).
    runs_finished: u64,
    /// Seconds of the current track actually played (for play reporting).
    played: f64,
    /// Decoder position at the previous chunk (to count `played`).
    last_pos: f64,
    /// Whether the current track has been reported as played.
    play_reported: bool,
    /// When the status was last published.
    last_publish: Instant,
    /// The status changed since it was last published.
    dirty: bool,
    /// Settings changed by a volume drag, saved once the dragging settles.
    save_at: Option<Instant>,
    /// A settings change is rebuilding: reopen the device without waiting for it to play
    /// out what it holds.
    no_drain: bool,
}

/// The playback thread's body: handle commands (all that are waiting at once), pump audio
/// while playing, publish the status when it changed or every 500 ms, and close the
/// device on shutdown.
pub(super) fn run(
    rx: Receiver<Command>,
    shared: Arc<Shared>,
    settings: Settings,
    options: super::EngineOptions,
) {
    let volume = Volume {
        gain_db: settings.dsp.volume_db,
        muted: settings.dsp.muted,
    };
    let mut w = Worker {
        rx,
        shared,
        settings,
        cache: options.cache,
        realtime: options.realtime,
        persist: options.persist,
        queue: Vec::new(),
        index: None,
        state: PlayState::Stopped,
        volume,
        decoder: None,
        info: None,
        prefetched: None,
        chain: None,
        output: None,
        output_name: String::new(),
        output_format: None,
        output_wanted: None,
        output_rates: Vec::new(),
        output_dsd: DsdCaps::NONE,
        output_exclusive: false,
        resume_at: None,
        convolution: None,
        error: None,
        runs_finished: 0,
        played: 0.0,
        last_pos: 0.0,
        play_reported: false,
        last_publish: Instant::now(),
        dirty: true,
        save_at: None,
        no_drain: false,
    };
    w.publish();
    'run: loop {
        let cmd = if w.state == PlayState::Playing {
            match w.rx.try_recv() {
                Ok(c) => Some(c),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => break,
            }
        } else {
            match w.rx.recv_timeout(Duration::from_millis(250)) {
                Ok(c) => Some(c),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        };
        if let Some(cmd) = cmd {
            if matches!(cmd, Command::Shutdown) {
                break;
            }
            w.handle(cmd);
            // Everything else waiting goes now too, not one per block: a volume drag
            // sends dozens, which would otherwise trail behind the slider.
            loop {
                match w.rx.try_recv() {
                    Ok(Command::Shutdown) => break 'run,
                    Ok(c) => w.handle(c),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break 'run,
                }
            }
        }
        if w.save_at.is_some_and(|t| Instant::now() >= t) {
            w.save_at = None;
            w.persist_settings();
        }
        if w.state == PlayState::Playing {
            w.pump();
        }
        if w.dirty
            || (w.state == PlayState::Playing
                && w.last_publish.elapsed() > Duration::from_millis(500))
        {
            w.publish();
        }
    }
    if w.save_at.take().is_some() {
        w.persist_settings();
    }
    w.stop_playback(true);
    if let Some(o) = &mut w.output {
        o.close();
    }
}

impl Worker {
    // ----- commands -------------------------------------------------------------------

    /// Apply one command from the handle.
    fn handle(&mut self, cmd: Command) {
        log::debug!("command {cmd:?}");
        self.dirty = true;
        // A live stream (AirPlay): transport commands go to the sender as well. Next and
        // previous are the sender's alone; play / pause / stop also act here.
        if let Some(TrackDecoder::Live(d)) = &self.decoder {
            use crate::decode::live::LiveControl;
            let c = match &cmd {
                Command::Play => Some(LiveControl::Play),
                Command::Pause | Command::Stop => Some(LiveControl::Pause),
                // Toggle comes back here as Play or Pause.
                Command::Next => Some(LiveControl::Next),
                Command::Prev => Some(LiveControl::Prev),
                _ => None,
            };
            if let Some(c) = c {
                let taken = d.stream().control(c);
                if taken && matches!(c, LiveControl::Next | LiveControl::Prev) {
                    return;
                }
            }
        }
        match cmd {
            Command::Play => match self.state {
                PlayState::Playing => {}
                PlayState::Paused if self.output.is_some() => self.resume(),
                PlayState::Paused | PlayState::Stopped => {
                    if self.queue.is_empty() {
                        return;
                    }
                    let at = match self.state {
                        PlayState::Paused => self.position(),
                        _ => self.resume_at.take().unwrap_or(0.0),
                    };
                    let i = self.index.unwrap_or(0).min(self.queue.len() - 1);
                    self.stop_playback(false);
                    self.start_index(i, at);
                }
            },
            Command::Release => {
                if self.state == PlayState::Playing {
                    self.handle(Command::Pause);
                }
                if self.state == PlayState::Paused {
                    // Keep the place: play restarts the track there on a fresh device.
                    let at = self.position();
                    self.stop_playback(false);
                    self.resume_at = Some(at);
                }
                self.drop_output();
            }
            Command::Reconnect => {
                let resume = (self.state != PlayState::Stopped)
                    .then(|| (self.index, self.position(), self.state));
                self.stop_playback(false);
                self.drop_output();
                if let Some((Some(i), at, was)) = resume {
                    self.start_index(i, at);
                    if was == PlayState::Paused {
                        self.handle(Command::Pause);
                    }
                }
            }
            Command::Pause => {
                if self.state == PlayState::Playing {
                    if let Some(o) = &mut self.output {
                        if let Err(e) = o.pause(true) {
                            log::warn!("pause: {e}");
                        }
                    }
                    self.state = PlayState::Paused;
                }
            }
            Command::Toggle => {
                let cmd = if self.state == PlayState::Playing {
                    Command::Pause
                } else {
                    Command::Play
                };
                self.handle(cmd);
            }
            Command::Stop => self.stop_playback(false),
            Command::Next => self.skip(1),
            Command::Prev => {
                if self.position() > 3.0 {
                    self.seek(0.0);
                } else {
                    self.skip(-1);
                }
            }
            Command::Seek(t) => self.seek(t),
            Command::SetVolume(v) => {
                self.volume = v;
                self.settings.dsp.volume_db = v.gain_db;
                self.settings.dsp.muted = v.muted;
                if let Some(c) = &mut self.chain {
                    c.set_volume(v);
                }
                self.save_at
                    .get_or_insert_with(|| Instant::now() + Duration::from_secs(1));
            }
            Command::SetQueue {
                tracks,
                index,
                position,
            } => {
                self.stop_playback(false);
                self.queue = tracks;
                self.index = None;
                if !self.queue.is_empty() {
                    self.start_index(index.min(self.queue.len() - 1), position);
                }
            }
            Command::PlayIndex(i) => {
                if i < self.queue.len() {
                    self.stop_playback(false);
                    self.start_index(i, 0.0);
                }
            }
            Command::Enqueue(tracks) => {
                let was_empty = self.queue.is_empty();
                self.queue.extend(tracks);
                if was_empty && self.state == PlayState::Stopped && !self.queue.is_empty() {
                    self.start_index(0, 0.0);
                }
            }
            Command::PlayNext(tracks) => {
                let at = self.index.map(|i| i + 1).unwrap_or(0).min(self.queue.len());
                let tail = self.queue.split_off(at);
                self.queue.extend(tracks);
                self.queue.extend(tail);
                self.prefetched = None;
                if self.state == PlayState::Stopped && self.index.is_none() {
                    self.start_index(0, 0.0);
                }
            }
            Command::Remove(i) => {
                if i < self.queue.len() {
                    if Some(i) == self.index {
                        // Removing what is playing: play what follows it (or stop).
                        self.stop_playback(false);
                        self.queue.remove(i);
                        if i < self.queue.len() {
                            self.start_index(i, 0.0);
                        } else {
                            self.index = None;
                        }
                    } else {
                        self.queue.remove(i);
                        if let Some(cur) = self.index {
                            if i < cur {
                                self.index = Some(cur - 1);
                            }
                        }
                        self.prefetched = None;
                    }
                }
            }
            Command::UpdateSettings(new) => self.update_settings(*new),
            Command::SetTrackInfo(t) => {
                if let Some(cur) = self.index.and_then(|i| self.queue.get_mut(i)) {
                    if cur.source == t.source {
                        let source = std::mem::take(&mut cur.source);
                        *cur = Track { source, ..*t };
                    }
                }
            }
            Command::Shutdown => {}
        }
    }

    /// Swap in new settings: update in place what can be (volume, EQ), and rebuild the chain
    /// or reopen the device when the change needs it.
    fn update_settings(&mut self, new: Settings) {
        let old = self.settings.clone();
        self.settings = new;
        self.volume = Volume {
            gain_db: self.settings.dsp.volume_db,
            muted: self.settings.dsp.muted,
        };
        let output_changed = old.output != self.settings.output;
        let mut dsp = self.settings.dsp.clone();
        let mut old_dsp = old.dsp.clone();
        dsp.volume_db = 0.0;
        dsp.muted = false;
        old_dsp.volume_db = 0.0;
        old_dsp.muted = false;
        let mut chain_changed = old_dsp != dsp || old.matrix != self.settings.matrix;
        // Only the EQ moved (a band dragged on the page): retune it in place, no rebuild.
        let eq_only = chain_changed && !output_changed && old.matrix == self.settings.matrix && {
            let mut a = old_dsp.clone();
            a.eq = dsp.eq.clone();
            a.eq_enabled = dsp.eq_enabled;
            a == dsp
        };
        if eq_only {
            if let Some(c) = &mut self.chain {
                if c.set_eq(&self.settings.dsp.eq, self.settings.dsp.eq_enabled) {
                    chain_changed = false;
                }
            }
        }
        // A DSD pass-through chain depends on the volume too (any gain means re-modulating).
        let passthrough_affected = self.chain.as_ref().is_some_and(Chain::is_passthrough)
            && old.dsp.volume_db != self.settings.dsp.volume_db;
        if let Some(c) = &mut self.chain {
            c.set_volume(self.volume);
        }
        if (output_changed || chain_changed || passthrough_affected) && self.decoder.is_some() {
            // Switch now: what the old chain still holds is dropped (a moment of music;
            // with a long filter, playing it out took seconds before the change was heard).
            self.chain = None;
            self.no_drain = true;
            if output_changed {
                if let Some(o) = &mut self.output {
                    o.close();
                }
                self.output = None;
                self.output_format = None;
            }
            if self.info.is_some() {
                if let Err(e) = self.build_chain() {
                    self.fail(format!("rebuilding the chain: {e}"));
                }
            }
            self.no_drain = false;
        } else if output_changed {
            self.output = None;
            self.output_format = None;
        }
        self.persist_settings();
    }

    /// Hand the settings to the host's persistence callback.
    fn persist_settings(&self) {
        if let Some(p) = &self.persist {
            p(&self.settings);
        }
    }

    // ----- transport ------------------------------------------------------------------

    /// Start queue entry `i` at `position` seconds: open its decoder (or take the prefetched
    /// one), build the chain and start playing.
    fn start_index(&mut self, i: usize, position: f64) {
        // Starting from silence: open the device afresh, so one that was unplugged and
        // plugged in again (or whose driver changed) is found. Track changes keep it.
        if self.state == PlayState::Stopped && self.settings.output.backend != "null" {
            self.drop_output();
        }
        self.index = Some(i);
        self.error = None;
        self.resume_at = None;
        let track = self.queue[i].clone();
        let mut decoder = match self.take_prefetched(i) {
            Some(d) => d,
            None => match TrackDecoder::open(&track.source) {
                Ok(d) => d,
                Err(e) => {
                    self.fail(format!("{}: {e}", track.source.display()));
                    self.skip_failed(i);
                    return;
                }
            },
        };
        if position > 0.0 {
            if let Err(e) = decoder.seek(position) {
                log::warn!("seek to {position}: {e}");
            }
        }
        let label = decoder.source_label();
        self.info = Some((decoder.format_key(), label.clone(), decoder.duration()));
        self.decoder = Some(decoder);
        self.played = 0.0;
        self.last_pos = position;
        self.play_reported = false;
        if let Err(e) = self.build_chain() {
            self.fail(format!("cannot play {label}: {e}"));
            // Try the device again on the next play rather than keep a broken one.
            self.drop_output();
            self.decoder = None;
            self.state = PlayState::Stopped;
            self.runs_finished += 1;
            return;
        }
        if let Some(o) = &mut self.output {
            let _ = o.pause(false);
        }
        self.state = PlayState::Playing;
        self.emit(Event::TrackStarted(track));
    }

    /// A track that cannot be opened is skipped (unless it is the last one).
    fn skip_failed(&mut self, i: usize) {
        if i + 1 < self.queue.len() {
            self.start_index(i + 1, 0.0);
        } else {
            self.state = PlayState::Stopped;
            self.runs_finished += 1;
        }
    }

    /// Unpause the device and continue playing.
    fn resume(&mut self) {
        if let Some(o) = &mut self.output {
            if let Err(e) = o.pause(false) {
                log::warn!("resume: {e}");
            }
        }
        self.state = PlayState::Playing;
    }

    /// Stop: drop the decoders and the chain; with `drain` the device plays out what it holds.
    fn stop_playback(&mut self, drain: bool) {
        self.decoder = None;
        self.info = None;
        self.prefetched = None;
        if let Some(c) = &mut self.chain {
            if drain {
                let tail = c.flush();
                self.write_out(&tail);
            } else {
                c.reset();
            }
        }
        if let Some(o) = &mut self.output {
            let r = if drain { o.drain() } else { o.discard() };
            if let Err(e) = r {
                log::warn!("output: {e}");
            }
        }
        if self.state != PlayState::Stopped {
            self.runs_finished += 1;
        }
        self.state = PlayState::Stopped;
        self.played = 0.0;
        self.last_pos = 0.0;
        self.dirty = true;
    }

    /// Jump `delta` entries in the queue (stays put past either end).
    fn skip(&mut self, delta: i64) {
        let Some(cur) = self.index else { return };
        let target = cur as i64 + delta;
        if target < 0 || target as usize >= self.queue.len() {
            if delta > 0 {
                self.stop_playback(false);
            }
            return;
        }
        let was = self.state;
        self.stop_playback(false);
        if was == PlayState::Stopped {
            self.index = Some(target as usize);
        } else {
            self.start_index(target as usize, 0.0);
        }
    }

    /// Seek the current track to `t` seconds and clear the chain's history.
    fn seek(&mut self, t: f64) {
        let Some(d) = &mut self.decoder else { return };
        match d.seek(t) {
            Ok(actual) => {
                self.last_pos = actual;
                if let Some(c) = &mut self.chain {
                    c.reset();
                }
                if let Some(o) = &mut self.output {
                    let _ = o.discard();
                }
            }
            Err(e) => log::warn!("seek: {e}"),
        }
    }

    // ----- the chain ------------------------------------------------------------------

    /// Open the configured device if none is open; the error is the reason it failed.
    fn ensure_output(&mut self) -> Result<(), String> {
        if self.output.is_some() {
            return Ok(());
        }
        let (backend, device) = (
            self.settings.output.backend.as_str(),
            self.settings.output.device.as_str(),
        );
        let out: Box<dyn Output> = match backend {
            "null" => Box::new(output::null::NullOutput::for_device(self.realtime, device)),
            _ => output::open_backend(backend, device).map_err(|e| {
                format!("{backend} · {device} isn't available ({e}). Is it plugged in? Press play to try again.")
            })?,
        };
        self.output_name = out.name();
        let bits = self.settings.output.bits;
        self.output_rates = output::STANDARD_RATES
            .iter()
            .copied()
            .filter(|r| out.supports(*r, bits))
            .collect();
        self.output_dsd = out.dsd_caps();
        self.output_exclusive = out.exclusive();
        self.output = Some(out);
        Ok(())
    }

    /// Channel count after the matrix (what the device is opened with).
    fn output_channels(&self, input_channels: u32) -> u32 {
        if !self.settings.matrix.enabled || self.settings.dsp.bit_perfect {
            return input_channels;
        }
        match self.settings.matrix.resolve(input_channels as usize) {
            Ok(m) => m.outputs.len() as u32,
            Err(_) => input_channels,
        }
    }

    /// Whether a DSD source can go to the device untouched: DSD output wanted (or
    /// bit-perfect), pass-through allowed, nothing that would alter the samples, and the
    /// device takes the source rate (in the configured packing or DoP).
    fn passthrough_target(
        &self,
        decoder: &TrackDecoder,
        channels: u32,
    ) -> Option<(DsdRate, DsdPacking)> {
        let s = &self.settings;
        let src = decoder.dsd_rate()?;
        if !s.dsp.dsd_passthrough || !(s.dsp.bit_perfect || s.dsp.output_mode == OutputMode::Dsd) {
            return None;
        }
        let untouched = s.dsp.bit_perfect
            || (self.volume.is_unity()
                && !(s.dsp.eq_enabled && s.dsp.eq.iter().any(|b| b.enabled))
                && s.dsp.convolution.trim().is_empty()
                && !s.matrix.enabled);
        if !untouched {
            return None;
        }
        let (rate, family) = DsdRate::from_hz(src)?;
        let out = self.output.as_ref()?;
        let (got_rate, packing) =
            output::choose_dsd(out.as_ref(), family, rate, s.dsp.dsd_packing, channels)?;
        (got_rate == rate).then_some((rate, packing))
    }

    /// The configured convolution impulse, parsed once per path; `None` when off or unreadable.
    fn load_convolution(&mut self) -> Option<Arc<ConvolutionSpec>> {
        let path = self.settings.dsp.convolution.trim().to_string();
        if path.is_empty() {
            self.convolution = None;
            return None;
        }
        if let Some((p, spec)) = &self.convolution {
            if *p == path {
                return Some(Arc::clone(spec));
            }
        }
        match cadenza_dsp::convolution::load_impulse_response(std::path::Path::new(&path)) {
            Ok((rate, channels)) => {
                let spec = Arc::new(ConvolutionSpec {
                    name: path.clone(),
                    rate,
                    channels,
                });
                self.convolution = Some((path, Arc::clone(&spec)));
                Some(spec)
            }
            Err(e) => {
                self.error = Some(format!("convolution {path}: {e}"));
                log::error!("convolution {path}: {e}");
                None
            }
        }
    }

    /// Build the chain for the current decoder and (re)open the output to match.
    fn build_chain(&mut self) -> Result<(), String> {
        self.ensure_output()?;
        let s = self.settings.clone();
        let decoder = self.decoder.as_ref().ok_or("no track")?;
        let input = decoder.pcm_input();
        let src_info = decoder.info();
        let src_label = decoder.source_label();
        let channels = self.output_channels(input.channels);

        // ---- DSD pass-through: the bits go to the device untouched --------------------
        if let Some((rate, packing)) = self.passthrough_target(decoder, channels) {
            let dsd_hz = src_info.rate;
            let wanted = OutputFormat::dsd(dsd_hz, packing, channels);
            self.open_output(wanted)?;
            let packer = Packer::new(packing, channels as usize);
            let out_label = format!("{} ({})", rate.label(), packing.label());
            let input_ep = Endpoint {
                format: src_info.codec.clone(),
                rate: dsd_hz,
                bits: 1,
                channels,
                device: None,
            };
            let output_ep = Endpoint {
                format: rate.label(),
                rate: dsd_hz,
                bits: 1,
                channels,
                device: Some(self.output_name.clone()),
            };
            let path = SignalPath {
                stages: vec![Stage {
                    id: "dsd-passthrough".into(),
                    name: "DSD pass-through".into(),
                    detail: format!("{} bits untouched → {out_label}", rate.label()),
                    params: json!({
                        "rate": rate.id(),
                        "rate_hz": dsd_hz,
                        "packing": packing.id(),
                        "device_frame_rate": packer.frame_rate(dsd_hz),
                        "device_word_bits": packer.word_bits(),
                    }),
                    active: true,
                }],
                bit_perfect: true,
                label: format!(
                    "{} {} → {out_label} · bit-perfect",
                    src_info.codec,
                    rate.label()
                ),
                latency_seconds: 0.0,
                input: input_ep,
                output: output_ep,
            };
            log::info!("{}", path.label);
            self.chain = Some(Chain::Passthrough(Passthrough { packer, path }));
            return Ok(());
        }

        // ---- DSD output through the modulator ------------------------------------------
        let mut dsd: Option<DsdConfig> = None;
        if s.dsp.dsd_output() {
            let family = cadenza_dsp::dsd::family_base(input.rate);
            let out = self.output.as_ref().unwrap();
            match output::choose_dsd(
                out.as_ref(),
                family,
                s.dsp.dsd_rate,
                s.dsp.dsd_packing,
                channels,
            ) {
                Some((rate, packing)) => {
                    if rate != s.dsp.dsd_rate || packing != s.dsp.dsd_packing {
                        log::warn!(
                            "{} does not take {} {}; using {} {}",
                            self.output_name,
                            s.dsp.dsd_rate.label(),
                            s.dsp.dsd_packing.label(),
                            rate.label(),
                            packing.label()
                        );
                    }
                    dsd = Some(DsdConfig {
                        rate,
                        packing,
                        ..s.dsp.dsd()
                    });
                }
                None => {
                    let msg = format!(
                        "{} does not accept DSD; playing PCM instead",
                        self.output_name
                    );
                    log::warn!("{msg}");
                    self.error = Some(msg);
                }
            }
        }

        // ---- which rate and depth the device gets ----------------------------------------
        let source_bits = input.bits;
        let out = self.output.as_mut().unwrap();
        let wanted = if let Some(d) = &dsd {
            let family = cadenza_dsp::dsd::family_base(input.rate);
            OutputFormat::dsd(d.rate.hz(family), d.packing, channels)
        } else {
            let (rate, bits) = if s.dsp.bit_perfect {
                let bits = [s.output.bits.max(source_bits), 32, 24, 16]
                    .into_iter()
                    .find(|b| *b >= source_bits && out.supports(input.rate, *b))
                    .unwrap_or(s.output.bits);
                (input.rate, bits)
            } else {
                let bits = [s.output.bits, 32, 24, 16]
                    .into_iter()
                    .find(|b| out.supports(input.rate, *b))
                    .unwrap_or(s.output.bits);
                let rate = match s.output.rate {
                    RateSetting::Source => input.rate,
                    RateSetting::Auto => {
                        output::choose_rate(out.as_ref(), input.rate, None, s.output.max_rate, bits)
                    }
                    RateSetting::Fixed(r) => output::choose_rate(
                        out.as_ref(),
                        input.rate,
                        Some(r),
                        s.output.max_rate,
                        bits,
                    ),
                };
                (rate, bits)
            };
            OutputFormat::pcm(rate, bits, channels)
        };
        if let Some(d) = dsd.as_mut() {
            // Drivers say yes to DSD rates they then refuse (or the DAC can't play): step
            // down through the rates, native before DoP, before giving up.
            let family = cadenza_dsp::dsd::family_base(input.rate);
            let mut ladder: Vec<(DsdRate, DsdPacking)> = Vec::new();
            let other = match d.packing {
                DsdPacking::Native => DsdPacking::Dop,
                DsdPacking::Dop => DsdPacking::Native,
            };
            for p in [d.packing, other] {
                for &r in DsdRate::catalog().iter().rev() {
                    // DoP above DSD256 needs a PCM carrier over 768 kHz, which no DAC takes.
                    let dop_ok = r.dop_possible() && r <= DsdRate::Dsd256;
                    if r <= d.rate && (p == DsdPacking::Native || dop_ok) {
                        ladder.push((r, p));
                    }
                }
            }
            let mut first_err = None;
            let mut opened = false;
            for (r, p) in ladder {
                match self.open_output(OutputFormat::dsd(r.hz(family), p, channels)) {
                    Ok(()) => {
                        if (r, p) != (d.rate, d.packing) {
                            let msg = format!(
                                "{} refused {} {}; playing {} {}",
                                self.output_name,
                                d.rate.label(),
                                d.packing.label(),
                                r.label(),
                                p.label()
                            );
                            log::warn!("{msg}");
                            self.error = Some(msg);
                        }
                        d.rate = r;
                        d.packing = p;
                        opened = true;
                        break;
                    }
                    Err(e) => {
                        log::warn!("{} {}: {e}", r.label(), p.label());
                        first_err.get_or_insert(e);
                    }
                }
            }
            if !opened {
                return Err(format!(
                    "{} (no DSD rate worked; set Output mode to PCM)",
                    first_err.unwrap_or_else(|| "DSD did not open".into())
                ));
            }
        } else {
            self.open_output(wanted)?;
        }
        let got = self.output_format.unwrap();

        let convolution = self.load_convolution();
        let config = pipeline_config(
            &s,
            self.volume,
            convolution,
            got,
            input.rate,
            dsd,
            self.output_name.clone(),
        );
        let pipeline =
            Pipeline::new(input, config, &self.cache).map_err(|e: PipelineError| e.to_string())?;
        if pipeline.channels() as u32 != got.channels {
            return Err(format!(
                "the matrix produces {} channels but the device was opened with {}",
                pipeline.channels(),
                got.channels
            ));
        }
        let path = pipeline.signal_path();
        log::info!("{} ({src_label})", path.label);
        self.chain = Some(Chain::Pipeline(Box::new(pipeline)));
        Ok(())
    }

    /// Open the output in `wanted` unless it already is (or already holds what the device
    /// gave for that request).
    fn open_output(&mut self, wanted: OutputFormat) -> Result<(), String> {
        let out = self.output.as_mut().ok_or("no output")?;
        let open_as_wanted = self.output_format == Some(wanted)
            || (self.output_format.is_some() && self.output_wanted == Some(wanted));
        if !open_as_wanted {
            if self.output_format.is_some() && !self.no_drain {
                let _ = out.drain();
            }
            let got = out.open(wanted).map_err(|e| e.to_string())?;
            if got != wanted {
                log::info!("{} opened {got} for {wanted}", out.name());
            }
            self.output_format = Some(got);
            self.output_wanted = Some(wanted);
            // A backend may fall back to a shared stream, which changes both.
            self.output_name = out.name();
            self.output_exclusive = out.exclusive();
        }
        Ok(())
    }

    /// Write output frames to the device. If the device went away, playback stops, the
    /// device is let go, and the next play picks up from the current position.
    fn write_out(&mut self, frames: &[i32]) {
        if frames.is_empty() {
            return;
        }
        let Some(o) = &mut self.output else { return };
        if let Err(e) = o.write(frames) {
            // The device went away (unplugged, driver reset): stop instead of decoding
            // into nothing, let go of it, and pick up from here on the next play.
            let at = self.position();
            self.stop_playback(false);
            self.drop_output();
            self.resume_at = Some(at);
            self.fail(format!(
                "output lost: {e}. Check the device and press play to carry on."
            ));
        }
    }

    /// Close and forget the output; the next track opens it again.
    fn drop_output(&mut self) {
        if let Some(mut o) = self.output.take() {
            o.close();
        }
        self.output_format = None;
        self.chain = None;
    }

    // ----- playback -------------------------------------------------------------------

    /// Decode one chunk, run it through the chain and write it out; at the end of a track,
    /// move on to the next one.
    fn pump(&mut self) {
        let Some(decoder) = &mut self.decoder else {
            self.state = PlayState::Stopped;
            return;
        };
        let passthrough = self.chain.as_ref().is_some_and(Chain::is_passthrough);
        // Gapless DSD → PCM: when the next track (already opened) is DSD in the same
        // format, this one leaves its converter running for it.
        if let TrackDecoder::Dsd(d) = decoder {
            let next = match &self.prefetched {
                Some((_, TrackDecoder::Dsd(n))) => n.can_continue(d),
                _ => false,
            };
            d.set_continues(next && !passthrough);
        }
        let chunk = if passthrough {
            decoder.next_dsd_chunk().map(|c| c.map(Chunk::Dsd))
        } else {
            decoder.next_chunk().map(|c| c.map(Chunk::Pcm))
        };
        match chunk {
            Ok(Some(chunk)) => {
                let pos = decoder.position_seconds();
                let delta = pos - self.last_pos;
                if delta > 0.0 && delta < 2.0 {
                    self.played += delta;
                }
                self.last_pos = pos;
                let out = match (&mut self.chain, chunk) {
                    (Some(c), Chunk::Pcm(pcm)) => c.process_pcm(&pcm),
                    (Some(c), Chunk::Dsd(bits)) => c.process_dsd(&bits),
                    (None, _) => Vec::new(),
                };
                self.write_out(&out);
                self.report_play();
                self.maybe_prefetch();
            }
            Ok(None) => self.advance(),
            Err(e) => {
                log::error!("decode: {e}");
                self.error = Some(format!("decode: {e}"));
                self.advance();
            }
        }
    }

    /// Half the song or four minutes, whichever comes first: report once.
    fn report_play(&mut self) {
        if self.play_reported {
            return;
        }
        let threshold = match self.info.as_ref().and_then(|i| i.2) {
            Some(d) if d > 0.0 => (d / 2.0).min(240.0),
            _ => 240.0,
        };
        if self.played >= threshold {
            self.play_reported = true;
            if let Some(i) = self.index {
                if let Some(t) = self.queue.get(i) {
                    let track = t.clone();
                    let seconds = self.played;
                    self.emit(Event::TrackPlayed { track, seconds });
                }
            }
        }
    }

    /// Open the next queue entry's decoder ahead of time for gapless playback.
    fn maybe_prefetch(&mut self) {
        let Some(i) = self.index else { return };
        if self.prefetched.is_some() || i + 1 >= self.queue.len() {
            return;
        }
        let remaining = match self.info.as_ref().and_then(|x| x.2) {
            Some(d) => d - self.last_pos,
            None => return,
        };
        if remaining > PREFETCH_SECONDS {
            return;
        }
        match TrackDecoder::open(&self.queue[i + 1].source) {
            Ok(d) => self.prefetched = Some((i + 1, d)),
            Err(e) => {
                log::warn!("prefetch {}: {e}", self.queue[i + 1].source.display());
                // Remember the failure so we do not retry every packet.
                self.prefetched = None;
                self.queue[i + 1].format = Some(format!("error: {e}"));
            }
        }
    }

    /// The prefetched decoder, if it is for queue entry `i`.
    fn take_prefetched(&mut self, i: usize) -> Option<TrackDecoder> {
        match self.prefetched.take() {
            Some((j, d)) if j == i => Some(d),
            _ => None,
        }
    }

    /// The current track ended: continue gaplessly with the next or finish.
    fn advance(&mut self) {
        let Some(i) = self.index else {
            self.stop_playback(true);
            return;
        };
        let next = i + 1;
        if next >= self.queue.len() {
            self.stop_playback(true);
            return;
        }
        let track = self.queue[next].clone();
        let decoder = match self.take_prefetched(next) {
            Some(d) => d,
            None => match TrackDecoder::open(&track.source) {
                Ok(d) => d,
                Err(e) => {
                    self.fail(format!("{}: {e}", track.source.display()));
                    self.index = Some(next);
                    self.advance();
                    return;
                }
            },
        };
        let mut decoder = decoder;
        if let (Some(TrackDecoder::Dsd(old)), TrackDecoder::Dsd(new)) =
            (self.decoder.as_mut(), &mut decoder)
        {
            if new.adopt(old) {
                log::debug!("gapless DSD: the converter runs on into the next track");
            }
        }
        let key = decoder.format_key();
        let label = decoder.source_label();
        let same_format = self.info.as_ref().is_some_and(|old| old.0 == key);
        if !same_format {
            if let Some(c) = &mut self.chain {
                let tail = c.flush();
                self.write_out(&tail);
            }
        }
        self.index = Some(next);
        self.info = Some((key, label.clone(), decoder.duration()));
        self.decoder = Some(decoder);
        self.played = 0.0;
        self.last_pos = 0.0;
        self.play_reported = false;
        if !same_format {
            if let Err(e) = self.build_chain() {
                self.fail(format!("cannot play {label}: {e}"));
                self.decoder = None;
                self.state = PlayState::Stopped;
                self.runs_finished += 1;
                return;
            }
        }
        self.emit(Event::TrackStarted(track));
        self.dirty = true;
    }

    // ----- status ---------------------------------------------------------------------

    /// Playback position in seconds as heard: the decoder position minus the chain latency
    /// and what the device still has queued.
    fn position(&self) -> f64 {
        let Some(d) = &self.decoder else { return 0.0 };
        let mut pos = d.position_seconds();
        if let Some(c) = &self.chain {
            pos -= c.latency_seconds();
        }
        if let (Some(o), Some(f)) = (&self.output, self.output_format) {
            pos -= o.queued_frames() as f64 / f.rate as f64;
        }
        pos.max(0.0)
    }

    /// Record and log a playback error.
    fn fail(&mut self, msg: String) {
        log::error!("{msg}");
        self.error = Some(msg);
        self.dirty = true;
    }

    /// Send an event to the handle's subscribers (dropped when nobody listens).
    fn emit(&self, e: Event) {
        let _ = self.shared.events.send(e);
    }

    /// Snapshot the state into a [`Status`], store it for polling and broadcast it.
    fn publish(&mut self) {
        self.dirty = false;
        self.last_publish = Instant::now();
        let track = self.index.and_then(|i| self.queue.get(i).cloned());
        let duration = self
            .info
            .as_ref()
            .and_then(|i| i.2)
            .or(track.as_ref().and_then(|t| t.duration));
        let status = Status {
            state: self.state,
            track,
            index: self.index,
            position: self.position(),
            duration,
            volume: self.volume,
            volume_linear: self.volume.to_linear(),
            queue: self.queue.clone(),
            signal_path: self
                .chain
                .as_ref()
                .filter(|_| self.decoder.is_some())
                .map(Chain::signal_path),
            source_format: self.info.as_ref().map(|i| i.1.clone()),
            settings: self.settings.clone(),
            output: self.output_name.clone(),
            output_format: self.output_format,
            output_rates: self.output_rates.clone(),
            output_dsd: self.output_dsd.clone(),
            output_exclusive: self.output_exclusive,
            output_frames: self
                .output
                .as_ref()
                .map(|o| o.frames_written())
                .unwrap_or(0),
            runs_finished: self.runs_finished,
            error: self.error.clone(),
        };
        let status = Arc::new(status);
        *self.shared.status.lock().unwrap() = Arc::clone(&status);
        self.emit(Event::Status(status));
    }
}

/// The pipeline settings for playing a `input_rate` source into a device opened as `got`
/// (`dsd`: the DSD rate and packing chosen for it). Shared with the limit test, so it
/// measures exactly what playback runs.
pub(crate) fn pipeline_config(
    s: &Settings,
    volume: Volume,
    convolution: Option<Arc<ConvolutionSpec>>,
    got: OutputFormat,
    input_rate: u32,
    dsd: Option<DsdConfig>,
    device: String,
) -> PipelineConfig {
    let filter_design = cadenza_dsp::design::FilterDesign::parse(&s.dsp.filter);
    let filter = FilterKind::parse(&s.dsp.filter).unwrap_or_else(|| {
        if filter_design.is_none() {
            log::warn!("unknown filter {:?}, using the default", s.dsp.filter);
        }
        FilterKind::default()
    });
    let dither = DitherKind::parse(&s.dsp.dither).unwrap_or_else(|| {
        log::warn!("unknown dither {:?}, using TPDF", s.dsp.dither);
        DitherKind::Tpdf
    });
    PipelineConfig {
        bit_perfect: s.dsp.bit_perfect,
        volume,
        eq: s.dsp.eq.clone(),
        eq_enabled: s.dsp.eq_enabled,
        convolution,
        filter,
        attenuation_db: s.dsp.attenuation_db,
        output_rate: (!got.is_dsd() && got.rate != input_rate).then_some(got.rate),
        output_bits: got.bits,
        dither,
        headroom_db: s.dsp.headroom_db,
        soft_clip: s.dsp.soft_clip,
        output_device: Some(device),
        output_mode: if dsd.is_some() {
            OutputMode::Dsd
        } else {
            OutputMode::Pcm
        },
        dsd: dsd.unwrap_or_else(|| s.dsp.dsd()),
        gpu: s.dsp.gpu,
        matrix: s.matrix.clone(),
        headphone: s.dsp.headphones,
        filter_design,
        dac_noise_floor_db: s.dsp.dac_noise_floor_db,
    }
}
