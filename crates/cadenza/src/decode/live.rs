//! Live PCM pushed in by a receiver (AirPlay) instead of read from a file or URL.
//!
//! The receiver registers a [`LiveStream`] under an id and queues a track whose source is
//! [`Source::Live`](super::Source::Live) with that id. The engine's decoder pulls chunks
//! from it; the output paces the pull. When nothing is due it gets a short block of
//! silence at once, so the output keeps running through network gaps and pauses at the
//! sender (and the worker stays responsive to commands).
//! The sender's clock and the DAC's clock drift apart slowly: when the buffer grows past
//! `max_frames` the oldest audio is dropped, and an empty buffer is bridged with silence.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Frames of silence handed out when the buffer is empty.
const SILENCE_FRAMES: usize = 352;

#[derive(Default)]
struct Inner {
    chunks: VecDeque<Vec<f64>>,
    /// Frames in `chunks`.
    frames: usize,
    /// Playback waits until this many frames are buffered (after start and after a flush).
    primed: bool,
    closed: bool,
    underruns: u64,
    dropped: u64,
}

/// One live stream: interleaved `f64` PCM in, pulled by the engine.
pub struct LiveStream {
    /// Sample rate.
    pub rate: u32,
    /// Channels.
    pub channels: u32,
    /// Source bit depth.
    pub bits: u32,
    /// Codec label (`ALAC`, `PCM`).
    pub codec: String,
    /// Frames to collect before playing.
    pub prebuffer_frames: usize,
    /// Above this the oldest audio is dropped (clock drift, a burst after a stall).
    pub max_frames: usize,
    inner: Mutex<Inner>,
    /// Frames handed to the engine.
    pulled: AtomicU64,
    /// `(frames pulled, track seconds)` at the last position report from the sender.
    mark: Mutex<Option<(u64, f64)>>,
}

impl LiveStream {
    /// A new stream (not yet registered).
    pub fn new(rate: u32, channels: u32, bits: u32, codec: &str, prebuffer_ms: u32) -> Self {
        let prebuffer_frames = (rate as u64 * prebuffer_ms as u64 / 1000) as usize;
        Self {
            rate,
            channels,
            bits,
            codec: codec.to_string(),
            prebuffer_frames,
            max_frames: prebuffer_frames + rate as usize,
            inner: Mutex::new(Inner::default()),
            pulled: AtomicU64::new(0),
            mark: Mutex::new(None),
        }
    }

    /// Append interleaved samples (`-1.0..1.0`).
    pub fn push(&self, samples: Vec<f64>) {
        if samples.is_empty() {
            return;
        }
        let ch = self.channels.max(1) as usize;
        let mut st = self.inner.lock().unwrap();
        if st.closed {
            return;
        }
        st.frames += samples.len() / ch;
        st.chunks.push_back(samples);
        while st.frames > self.max_frames {
            match st.chunks.pop_front() {
                Some(c) => {
                    st.frames -= c.len() / ch;
                    st.dropped += 1;
                }
                None => break,
            }
        }
        if !st.primed && st.frames >= self.prebuffer_frames {
            st.primed = true;
        }
    }

    /// Drop everything buffered and wait for a new prebuffer (sender paused or seeked).
    pub fn flush(&self) {
        let mut st = self.inner.lock().unwrap();
        st.chunks.clear();
        st.frames = 0;
        st.primed = false;
    }

    /// End the stream: the decoder delivers what is left, then reports the end.
    pub fn close(&self) {
        self.inner.lock().unwrap().closed = true;
    }

    /// Whether [`LiveStream::close`] was called.
    pub fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().closed
    }

    /// Frames buffered now.
    pub fn buffered_frames(&self) -> usize {
        self.inner.lock().unwrap().frames
    }

    /// `(underruns, dropped chunks)` so far.
    pub fn counters(&self) -> (u64, u64) {
        let st = self.inner.lock().unwrap();
        (st.underruns, st.dropped)
    }

    /// The next chunk: audio when primed, silence when the buffer is empty, `None` once
    /// closed and drained.
    pub fn pull(&self) -> Option<Vec<f64>> {
        let ch = self.channels.max(1) as usize;
        let mut st = self.inner.lock().unwrap();
        if st.primed || st.closed {
            if let Some(c) = st.chunks.pop_front() {
                st.frames -= c.len() / ch;
                self.pulled
                    .fetch_add((c.len() / ch) as u64, Ordering::Relaxed);
                return Some(c);
            }
            if st.closed {
                return None;
            }
            // Ran dry: count it and collect a new prebuffer.
            st.primed = false;
            st.underruns += 1;
        }
        self.pulled
            .fetch_add(SILENCE_FRAMES as u64, Ordering::Relaxed);
        Some(vec![0.0; SILENCE_FRAMES * ch])
    }

    /// The sender says the track is at `seconds` now (what plays now is roughly what the
    /// engine pulled last).
    pub fn mark_position(&self, seconds: f64) {
        *self.mark.lock().unwrap() = Some((self.pulled.load(Ordering::Relaxed), seconds));
    }

    /// Track position: from the last mark, or seconds since the stream began.
    pub fn position_seconds(&self) -> f64 {
        let pulled = self.pulled.load(Ordering::Relaxed);
        let rate = self.rate.max(1) as f64;
        match *self.mark.lock().unwrap() {
            Some((at, secs)) => secs + pulled.saturating_sub(at) as f64 / rate,
            None => pulled as f64 / rate,
        }
    }
}

fn registry() -> &'static Mutex<HashMap<String, Arc<LiveStream>>> {
    static R: OnceLock<Mutex<HashMap<String, Arc<LiveStream>>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

/// Make `stream` reachable as `Source::Live { live: id }`.
pub fn register(id: &str, stream: Arc<LiveStream>) {
    registry().lock().unwrap().insert(id.to_string(), stream);
}

/// Remove a stream (closing it).
pub fn unregister(id: &str) {
    if let Some(s) = registry().lock().unwrap().remove(id) {
        s.close();
    }
}

/// Look a stream up.
pub fn lookup(id: &str) -> Option<Arc<LiveStream>> {
    registry().lock().unwrap().get(id).cloned()
}

/// The engine side of a live stream.
pub struct LiveDecoder {
    stream: Arc<LiveStream>,
}

impl LiveDecoder {
    /// Attach to a registered stream.
    pub fn open(id: &str) -> std::io::Result<Self> {
        let stream = lookup(id).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("no live stream {id}"))
        })?;
        Ok(Self { stream })
    }

    /// Format.
    pub fn info(&self) -> super::TrackInfo {
        super::TrackInfo {
            codec: self.stream.codec.clone(),
            rate: self.stream.rate,
            bits: self.stream.bits,
            channels: self.stream.channels,
            duration: None,
            tags: Default::default(),
        }
    }

    /// Seconds delivered so far.
    pub fn position_seconds(&self) -> f64 {
        self.stream.position_seconds()
    }

    /// Next chunk (see [`LiveStream::pull`]).
    pub fn next_chunk(&mut self) -> Option<Vec<f64>> {
        self.stream.pull()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes_bridges_drops_and_ends() {
        let s = Arc::new(LiveStream::new(1000, 2, 16, "PCM", 10)); // 10 frames prebuffer
        register("t1", Arc::clone(&s));
        let mut d = LiveDecoder::open("t1").unwrap();
        // Not primed yet: silence.
        s.push(vec![0.5; 8]);
        assert!(d.next_chunk().unwrap().iter().all(|v| *v == 0.0));
        s.push(vec![0.25; 12]); // 10 frames now: primed
        assert_eq!(d.next_chunk().unwrap(), vec![0.5; 8]);
        assert_eq!(d.next_chunk().unwrap(), vec![0.25; 12]);
        // Empty: one underrun, silence, waits for a new prebuffer.
        assert_eq!(d.next_chunk().unwrap().len(), SILENCE_FRAMES * 2);
        assert_eq!(s.counters().0, 1);
        // Overfull: the oldest goes (max = 10 + 1000 frames).
        for _ in 0..600 {
            s.push(vec![0.1; 4]);
        }
        assert!(s.buffered_frames() <= s.max_frames);
        assert!(s.counters().1 > 0);
        s.flush();
        assert_eq!(s.buffered_frames(), 0);
        s.push(vec![0.3; 20]);
        s.close();
        assert_eq!(d.next_chunk().unwrap(), vec![0.3; 20]);
        assert!(d.next_chunk().is_none());
        s.mark_position(30.0);
        assert_eq!(d.position_seconds(), 30.0);
        unregister("t1");
        assert!(LiveDecoder::open("t1").is_err());
    }
}
