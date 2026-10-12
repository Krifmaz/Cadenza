//! The playback thread.
//!
//! One loop: take a command if there is one, then (while playing) decode a packet, push it
//! through the pipeline and write the result to the output. The output blocks when its
//! buffer is full, which paces everything to the device clock. Gapless playback falls out
//! of the design: when a track ends, the next decoder is opened and the pipeline keeps
//! running, so the filter history carries across the boundary.

use super::{Command, Event, PlayState, Shared, Status, Track};
use crate::config::{RateSetting, Settings};
use crate::decode::{Decoder, TrackInfo};
use crate::output::{self, Output, OutputFormat};
use cadenza_dsp::pipeline::PipelineError;
use cadenza_dsp::{
    CoefficientCache, ConvolutionSpec, DitherKind, FilterKind, InputFormat, Pipeline,
    PipelineConfig, Volume,
};
use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Open the next track this long before the current one ends, so HTTP streams are ready.
const PREFETCH_SECONDS: f64 = 10.0;

struct Worker {
    rx: Receiver<Command>,
    shared: Arc<Shared>,
    settings: Settings,
    cache: CoefficientCache,
    realtime: bool,
    persist: Option<super::PersistFn>,

    queue: Vec<Track>,
    index: Option<usize>,
    state: PlayState,
    volume: Volume,

    decoder: Option<Decoder>,
    info: Option<TrackInfo>,
    prefetched: Option<(usize, Decoder)>,
    pipeline: Option<Pipeline>,
    output: Option<Box<dyn Output>>,
    output_name: String,
    output_format: Option<OutputFormat>,
    /// `(path, impulse)` so the file is only parsed once.
    convolution: Option<(String, Arc<ConvolutionSpec>)>,

    error: Option<String>,
    runs_finished: u64,
    /// Seconds of the current track actually played (for play reporting).
    played: f64,
    last_pos: f64,
    play_reported: bool,
    last_publish: Instant,
    dirty: bool,
}

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
        pipeline: None,
        output: None,
        output_name: String::new(),
        output_format: None,
        convolution: None,
        error: None,
        runs_finished: 0,
        played: 0.0,
        last_pos: 0.0,
        play_reported: false,
        last_publish: Instant::now(),
        dirty: true,
    };
    w.publish();
    loop {
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
    w.stop_playback(true);
    if let Some(o) = &mut w.output {
        o.close();
    }
}

impl Worker {
    // ----- commands -------------------------------------------------------------------

    fn handle(&mut self, cmd: Command) {
        log::debug!("command {cmd:?}");
        self.dirty = true;
        match cmd {
            Command::Play => match self.state {
                PlayState::Playing => {}
                PlayState::Paused => self.resume(),
                PlayState::Stopped => {
                    if self.queue.is_empty() {
                        return;
                    }
                    let i = self.index.unwrap_or(0).min(self.queue.len() - 1);
                    self.start_index(i, 0.0);
                }
            },
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
                if let Some(p) = &mut self.pipeline {
                    p.set_volume(v);
                }
                self.persist_settings();
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
            Command::Shutdown => {}
        }
    }

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
        let chain_changed = old_dsp != dsp;
        if let Some(p) = &mut self.pipeline {
            p.set_volume(self.volume);
        }
        if (output_changed || chain_changed) && self.decoder.is_some() {
            // Push out what the old chain still holds, then build the new one.
            if let Some(p) = &mut self.pipeline {
                let tail = p.flush();
                self.write_out(&tail);
            }
            if output_changed {
                if let Some(o) = &mut self.output {
                    let _ = o.drain();
                    o.close();
                }
                self.output = None;
                self.output_format = None;
            }
            if let Some(info) = self.info.clone() {
                if let Err(e) = self.build_chain(&info) {
                    self.fail(format!("rebuilding the chain: {e}"));
                }
            }
        } else if output_changed {
            self.output = None;
            self.output_format = None;
        }
        self.persist_settings();
    }

    fn persist_settings(&self) {
        if let Some(p) = &self.persist {
            p(&self.settings);
        }
    }

    // ----- transport ------------------------------------------------------------------

    fn start_index(&mut self, i: usize, position: f64) {
        self.index = Some(i);
        self.error = None;
        let track = self.queue[i].clone();
        let mut decoder = match self.take_prefetched(i) {
            Some(d) => d,
            None => match Decoder::open(&track.source) {
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
        let info = decoder.info().clone();
        self.decoder = Some(decoder);
        self.info = Some(info.clone());
        self.played = 0.0;
        self.last_pos = position;
        self.play_reported = false;
        if let Err(e) = self.build_chain(&info) {
            self.fail(format!("cannot play {}: {e}", info.label()));
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

    fn resume(&mut self) {
        if let Some(o) = &mut self.output {
            if let Err(e) = o.pause(false) {
                log::warn!("resume: {e}");
            }
        }
        self.state = PlayState::Playing;
    }

    fn stop_playback(&mut self, drain: bool) {
        self.decoder = None;
        self.info = None;
        self.prefetched = None;
        if let Some(p) = &mut self.pipeline {
            if drain {
                let tail = p.flush();
                self.write_out(&tail);
            } else {
                p.reset();
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

    fn seek(&mut self, t: f64) {
        let Some(d) = &mut self.decoder else { return };
        match d.seek(t) {
            Ok(actual) => {
                self.last_pos = actual;
                if let Some(p) = &mut self.pipeline {
                    p.reset();
                }
                if let Some(o) = &mut self.output {
                    let _ = o.discard();
                }
            }
            Err(e) => log::warn!("seek: {e}"),
        }
    }

    // ----- the chain ------------------------------------------------------------------

    fn ensure_output(&mut self) -> Result<(), String> {
        if self.output.is_some() {
            return Ok(());
        }
        let (backend, device) = (
            self.settings.output.backend.as_str(),
            self.settings.output.device.as_str(),
        );
        let out: Box<dyn Output> = match backend {
            "null" => Box::new(output::null::NullOutput::new(self.realtime)),
            _ => match output::open_backend(backend, device) {
                Ok(o) => o,
                Err(e) => {
                    log::error!("output {backend} '{device}': {e}; using the null output");
                    self.error = Some(format!(
                        "output {backend} '{device}': {e}; using the null output"
                    ));
                    Box::new(output::null::NullOutput::new(self.realtime))
                }
            },
        };
        self.output_name = out.name();
        self.output = Some(out);
        Ok(())
    }

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

    /// Build the pipeline for `info` and (re)open the output to match.
    fn build_chain(&mut self, info: &TrackInfo) -> Result<(), String> {
        self.ensure_output()?;
        let s = self.settings.clone();
        let source_bits = if info.bits == 0 { 24 } else { info.bits };
        let out = self.output.as_mut().unwrap();

        // Which rate and depth the device gets.
        let (rate, bits) = if s.dsp.bit_perfect {
            let bits = [s.output.bits.max(source_bits), 32, 24, 16]
                .into_iter()
                .find(|b| *b >= source_bits && out.supports(info.rate, *b))
                .unwrap_or(s.output.bits);
            (info.rate, bits)
        } else {
            let bits = [s.output.bits, 32, 24, 16]
                .into_iter()
                .find(|b| out.supports(info.rate, *b))
                .unwrap_or(s.output.bits);
            let rate = match s.output.rate {
                RateSetting::Source => info.rate,
                RateSetting::Auto => {
                    output::choose_rate(out.as_ref(), info.rate, None, s.output.max_rate, bits)
                }
                RateSetting::Fixed(r) => {
                    output::choose_rate(out.as_ref(), info.rate, Some(r), s.output.max_rate, bits)
                }
            };
            (rate, bits)
        };
        let wanted = OutputFormat {
            rate,
            bits,
            channels: info.channels,
        };
        if self.output_format != Some(wanted) {
            if self.output_format.is_some() {
                let _ = out.drain();
            }
            let got = out.open(wanted).map_err(|e| e.to_string())?;
            self.output_format = Some(got);
        }
        let got = self.output_format.unwrap();

        let convolution = self.load_convolution();
        let filter = FilterKind::parse(&s.dsp.filter).unwrap_or_else(|| {
            log::warn!("unknown filter {:?}, using the default", s.dsp.filter);
            FilterKind::default()
        });
        let dither = DitherKind::parse(&s.dsp.dither).unwrap_or_else(|| {
            log::warn!("unknown dither {:?}, using TPDF", s.dsp.dither);
            DitherKind::Tpdf
        });
        let config = PipelineConfig {
            bit_perfect: s.dsp.bit_perfect,
            volume: self.volume,
            eq: s.dsp.eq.clone(),
            eq_enabled: s.dsp.eq_enabled,
            convolution,
            filter,
            attenuation_db: s.dsp.attenuation_db,
            output_rate: (got.rate != info.rate).then_some(got.rate),
            output_bits: got.bits,
            dither,
            output_device: Some(self.output_name.clone()),
        };
        let input = InputFormat {
            rate: info.rate,
            bits: source_bits,
            channels: info.channels,
            codec: info.codec.clone(),
        };
        let pipeline =
            Pipeline::new(input, config, &self.cache).map_err(|e: PipelineError| e.to_string())?;
        log::info!("{}", pipeline.signal_path().label);
        self.pipeline = Some(pipeline);
        Ok(())
    }

    fn write_out(&mut self, frames: &[i32]) {
        if frames.is_empty() {
            return;
        }
        if let Some(o) = &mut self.output {
            if let Err(e) = o.write(frames) {
                log::error!("output write: {e}");
                self.error = Some(format!("output: {e}"));
            }
        }
    }

    // ----- playback -------------------------------------------------------------------

    fn pump(&mut self) {
        let Some(decoder) = &mut self.decoder else {
            self.state = PlayState::Stopped;
            return;
        };
        match decoder.next_chunk() {
            Ok(Some(pcm)) => {
                let pos = decoder.position_seconds();
                let delta = pos - self.last_pos;
                if delta > 0.0 && delta < 2.0 {
                    self.played += delta;
                }
                self.last_pos = pos;
                let out = match &mut self.pipeline {
                    Some(p) => p.process(&pcm),
                    None => Vec::new(),
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
        let threshold = match self.info.as_ref().and_then(|i| i.duration) {
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

    fn maybe_prefetch(&mut self) {
        let Some(i) = self.index else { return };
        if self.prefetched.is_some() || i + 1 >= self.queue.len() {
            return;
        }
        let remaining = match self.info.as_ref().and_then(|x| x.duration) {
            Some(d) => d - self.last_pos,
            None => return,
        };
        if remaining > PREFETCH_SECONDS {
            return;
        }
        match Decoder::open(&self.queue[i + 1].source) {
            Ok(d) => self.prefetched = Some((i + 1, d)),
            Err(e) => {
                log::warn!("prefetch {}: {e}", self.queue[i + 1].source.display());
                // Remember the failure so we do not retry every packet.
                self.prefetched = None;
                self.queue[i + 1].format = Some(format!("error: {e}"));
            }
        }
    }

    fn take_prefetched(&mut self, i: usize) -> Option<Decoder> {
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
            None => match Decoder::open(&track.source) {
                Ok(d) => d,
                Err(e) => {
                    self.fail(format!("{}: {e}", track.source.display()));
                    self.index = Some(next);
                    self.advance();
                    return;
                }
            },
        };
        let info = decoder.info().clone();
        let same_format = self.info.as_ref().is_some_and(|old| {
            (old.rate, old.bits, old.channels) == (info.rate, info.bits, info.channels)
        });
        if !same_format {
            if let Some(p) = &mut self.pipeline {
                let tail = p.flush();
                self.write_out(&tail);
            }
        }
        self.index = Some(next);
        self.decoder = Some(decoder);
        self.info = Some(info.clone());
        self.played = 0.0;
        self.last_pos = 0.0;
        self.play_reported = false;
        if !same_format {
            if let Err(e) = self.build_chain(&info) {
                self.fail(format!("cannot play {}: {e}", info.label()));
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

    fn position(&self) -> f64 {
        let Some(d) = &self.decoder else { return 0.0 };
        let mut pos = d.position_seconds();
        if let Some(p) = &self.pipeline {
            pos -= p.latency_seconds();
        }
        if let (Some(o), Some(f)) = (&self.output, self.output_format) {
            pos -= o.queued_frames() as f64 / f.rate as f64;
        }
        pos.max(0.0)
    }

    fn fail(&mut self, msg: String) {
        log::error!("{msg}");
        self.error = Some(msg);
        self.dirty = true;
    }

    fn emit(&self, e: Event) {
        let _ = self.shared.events.send(e);
    }

    fn publish(&mut self) {
        self.dirty = false;
        self.last_publish = Instant::now();
        let track = self.index.and_then(|i| self.queue.get(i).cloned());
        let duration = self
            .info
            .as_ref()
            .and_then(|i| i.duration)
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
                .pipeline
                .as_ref()
                .filter(|_| self.decoder.is_some())
                .map(|p| p.signal_path()),
            source_format: self.info.as_ref().map(|i| i.label()),
            settings: self.settings.clone(),
            output: self.output_name.clone(),
            output_format: self.output_format,
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
