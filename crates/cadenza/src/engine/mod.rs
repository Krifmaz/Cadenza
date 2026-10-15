//! The player: a queue, transport, settings and a worker thread that decodes, runs the
//! DSP pipeline and writes to the output. [`Engine`] is the cheap-to-clone handle; the
//! HTTP API, the CLI and the Encore client all talk to the same one.

pub mod decoder;
mod worker;

use crate::config::Settings;
use crate::decode::Source;
use crate::output::{DsdCaps, OutputFormat};
use cadenza_dsp::{CoefficientCache, SignalPath, Volume};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

/// One queue entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Track {
    /// Encore track id (or any id the client wants echoed back).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Where the audio is.
    pub source: Source,
    /// Title.
    #[serde(default)]
    pub title: String,
    /// Artist.
    #[serde(default)]
    pub artist: String,
    /// Album.
    #[serde(default)]
    pub album: String,
    /// Encore album id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub album_id: Option<String>,
    /// Duration in seconds, if known before decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// Format label (`FLAC`, `MP3 320`…) if known before decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Cover art URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover: Option<String>,
}

impl Default for Source {
    fn default() -> Self {
        Source::Path {
            path: Default::default(),
        }
    }
}

impl Track {
    /// A disc image or CUE sheet as its tracks (anything else as itself).
    pub fn expand(self) -> Result<Vec<Track>, String> {
        match crate::decode::expand(&self.source) {
            None => Ok(vec![self]),
            Some(entries) => Ok(entries?
                .into_iter()
                .map(|e| Track {
                    source: e.source,
                    title: e.title,
                    artist: e.artist,
                    album: e.album,
                    duration: e.duration,
                    ..Default::default()
                })
                .collect()),
        }
    }

    /// A track from a CLI argument (path or URL), titled after its file name.
    pub fn from_arg(arg: &str) -> Self {
        let source = Source::parse(arg);
        let title = source
            .display()
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(arg)
            .to_string();
        Self {
            source,
            title,
            ..Default::default()
        }
    }
}

/// Transport state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PlayState {
    /// Nothing loaded / finished.
    #[default]
    Stopped,
    /// Playing.
    Playing,
    /// Paused.
    Paused,
}

/// A snapshot of everything the UI shows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Status {
    /// Transport.
    pub state: PlayState,
    /// The current track.
    pub track: Option<Track>,
    /// Its index in the queue.
    pub index: Option<usize>,
    /// Position in seconds.
    pub position: f64,
    /// Duration in seconds, when known.
    pub duration: Option<f64>,
    /// Volume.
    pub volume: Volume,
    /// Volume as a 0…1 fader.
    pub volume_linear: f64,
    /// The queue.
    pub queue: Vec<Track>,
    /// What the DSP does right now (`None` while stopped).
    pub signal_path: Option<SignalPath>,
    /// `FLAC 16/44.1` style label of the current source.
    pub source_format: Option<String>,
    /// Settings in force.
    pub settings: Settings,
    /// Output name (`alsa · hw:1,0`).
    pub output: String,
    /// The open output format.
    pub output_format: Option<OutputFormat>,
    /// Sample rates the current output device accepts (at the configured depth).
    pub output_rates: Vec<u32>,
    /// What the current output device does with DSD (DoP / native layouts / top rate), so
    /// the UI can grey out rates the device stops short of.
    #[serde(default)]
    pub output_dsd: DsdCaps,
    /// Whether the output is exclusive (hog / direct), i.e. bit-perfect capable.
    #[serde(default)]
    pub output_exclusive: bool,
    /// Frames the output has accepted since it was opened (diagnostics).
    pub output_frames: u64,
    /// How many times playback has come to a stop (queue finished, or `Stop`). Sample it
    /// before starting playback and wait for it to grow to know a run has ended.
    pub runs_finished: u64,
    /// The last error, if any.
    pub error: Option<String>,
}

/// Something the worker wants the rest of the program to know.
#[derive(Clone, Debug)]
pub enum Event {
    /// The status changed (or the position moved).
    Status(Arc<Status>),
    /// A track started playing.
    TrackStarted(Track),
    /// A track has been played "enough" (half of it or four minutes).
    TrackPlayed {
        /// The track.
        track: Track,
        /// Seconds played when the threshold was reached.
        seconds: f64,
    },
}

/// Commands to the worker.
#[derive(Debug)]
pub enum Command {
    /// Start / resume.
    Play,
    /// Pause.
    Pause,
    /// Play ↔ pause.
    Toggle,
    /// Stop.
    Stop,
    /// Next track.
    Next,
    /// Previous track (or restart the current one after 3 s).
    Prev,
    /// Seek to seconds in the current track.
    Seek(f64),
    /// Set the volume.
    SetVolume(Volume),
    /// Replace the queue and play `index` from `position`.
    SetQueue {
        /// Tracks.
        tracks: Vec<Track>,
        /// Start index.
        index: usize,
        /// Start position in seconds.
        position: f64,
    },
    /// Play this queue index.
    PlayIndex(usize),
    /// Append to the queue.
    Enqueue(Vec<Track>),
    /// Insert after the current track.
    PlayNext(Vec<Track>),
    /// Remove a queue index.
    Remove(usize),
    /// Replace the settings (the worker rebuilds what changed).
    UpdateSettings(Box<Settings>),
    /// Replace the title, artist, album, cover and duration of the playing entry, if its
    /// source is this track's (live streams announce their metadata as they go).
    SetTrackInfo(Box<Track>),
    /// Exit the worker thread.
    Shutdown,
}

/// Callback that stores changed settings (the config file, usually).
pub type PersistFn = Box<dyn Fn(&Settings) + Send>;

/// Construction options.
pub struct EngineOptions {
    /// Where long filters are cached.
    pub cache: CoefficientCache,
    /// Make the null output pace itself in real time (off for tests).
    pub realtime: bool,
    /// Called with the settings whenever they change, to persist them.
    pub persist: Option<PersistFn>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            cache: CoefficientCache::default_location(),
            realtime: true,
            persist: None,
        }
    }
}

pub(crate) struct Shared {
    pub status: Mutex<Arc<Status>>,
    pub events: broadcast::Sender<Event>,
}

/// Handle to the player.
#[derive(Clone)]
pub struct Engine {
    tx: Sender<Command>,
    shared: Arc<Shared>,
}

impl Engine {
    /// Start the worker thread.
    pub fn start(settings: Settings, options: EngineOptions) -> Engine {
        let (tx, rx) = mpsc::channel();
        let (events, _) = broadcast::channel(256);
        let status = Status {
            settings: settings.clone(),
            volume_linear: 1.0,
            ..Default::default()
        };
        let shared = Arc::new(Shared {
            status: Mutex::new(Arc::new(status)),
            events,
        });
        let worker_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("cadenza-player".into())
            .spawn(move || worker::run(rx, worker_shared, settings, options))
            .expect("spawn player thread");
        Engine { tx, shared }
    }

    /// The latest snapshot.
    pub fn status(&self) -> Arc<Status> {
        Arc::clone(&self.shared.status.lock().unwrap())
    }

    /// Settings in force.
    pub fn settings(&self) -> Settings {
        self.status().settings.clone()
    }

    /// Listen for events.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.shared.events.subscribe()
    }

    /// Send a command (ignored if the worker is gone).
    pub fn send(&self, cmd: Command) {
        let _ = self.tx.send(cmd);
    }

    /// Replace the queue and play.
    pub fn play_queue(&self, tracks: Vec<Track>, index: usize, position: f64) {
        self.send(Command::SetQueue {
            tracks,
            index,
            position,
        });
    }

    /// Replace the settings.
    pub fn update_settings(&self, settings: Settings) {
        self.send(Command::UpdateSettings(Box::new(settings)));
    }

    /// Set the volume from a 0…1 fader.
    pub fn set_volume_linear(&self, level: f64) {
        self.send(Command::SetVolume(Volume::from_linear(level)));
    }

    /// Block until `runs_finished` exceeds `runs_before` (sampled from [`Engine::status`]
    /// before the play command was sent) or `timeout` passes. Returns the final status.
    pub fn wait_until_finished(&self, runs_before: u64, timeout: Duration) -> Arc<Status> {
        let deadline = Instant::now() + timeout;
        loop {
            let st = self.status();
            if st.runs_finished > runs_before || Instant::now() >= deadline {
                return st;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Stop the worker thread.
    pub fn shutdown(&self) {
        self.send(Command::Shutdown);
    }
}
