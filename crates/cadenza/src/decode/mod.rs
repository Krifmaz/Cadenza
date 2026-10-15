//! Decoding with Symphonia: any supported container/codec, from a file or an HTTP URL,
//! to interleaved `f64` PCM.

pub mod cue;
pub mod dsd;
pub mod dst;
pub mod live;
pub mod sacd;
pub mod source;

pub use source::Source;

use std::time::Duration;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::{Time, TimeBase};

/// What was learned about a track when it was opened.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackInfo {
    /// Codec short name in upper case (`FLAC`, `MP3`, `AAC`, `ALAC`, `PCM`…).
    pub codec: String,
    /// Sample rate in Hz.
    pub rate: u32,
    /// Source bit depth, `0` when the codec has none (lossy).
    pub bits: u32,
    /// Channel count.
    pub channels: u32,
    /// Duration in seconds when the container knows it.
    pub duration: Option<f64>,
    /// Tag metadata found while probing.
    pub tags: Tags,
}

/// Basic tags.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tags {
    /// Title.
    pub title: Option<String>,
    /// Artist.
    pub artist: Option<String>,
    /// Album.
    pub album: Option<String>,
}

impl TrackInfo {
    /// `FLAC 16/44.1` style label.
    pub fn label(&self) -> String {
        if self.bits > 0 {
            format!(
                "{} {}/{}",
                self.codec,
                self.bits,
                cadenza_dsp::signal_path::rate_label(self.rate)
            )
        } else {
            format!(
                "{} {}",
                self.codec,
                cadenza_dsp::signal_path::rate_label(self.rate)
            )
        }
    }
}

/// Decoding errors.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Could not open the source.
    #[error("open: {0}")]
    Io(#[from] std::io::Error),
    /// Symphonia could not handle the stream.
    #[error("decode: {0}")]
    Symphonia(#[from] SymphoniaError),
    /// No audio track in the container.
    #[error("no audio track found")]
    NoTrack,
}

/// A streaming decoder for one track.
pub struct Decoder {
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    info: TrackInfo,
    time_base: Option<TimeBase>,
    buf: Option<SampleBuffer<f64>>,
    /// Timestamp (in `time_base` units) of the next frame to be delivered.
    next_ts: u64,
    finished: bool,
    /// Play only frames `start..end` (a CUE track).
    range: (u64, Option<u64>),
}

impl Decoder {
    /// Open and probe a source.
    pub fn open(source: &Source) -> Result<Self, DecodeError> {
        let (media, mime) = source.open()?;
        let mut hint = Hint::new();
        if let Some(ext) = source.extension() {
            hint.with_extension(&ext);
        }
        if let Some(m) = mime {
            hint.mime_type(&m);
        }
        let mss = MediaSourceStream::new(media, Default::default());
        let fmt_opts = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let mut probed = symphonia::default::get_probe().format(
            &hint,
            mss,
            &fmt_opts,
            &MetadataOptions::default(),
        )?;

        let track = probed
            .format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or(DecodeError::NoTrack)?;
        let params = track.codec_params.clone();
        let track_id = track.id;
        let decoder = symphonia::default::get_codecs().make(&params, &DecoderOptions::default())?;
        let codec = symphonia::default::get_codecs()
            .get_codec(params.codec)
            .map(|d| d.short_name.to_ascii_uppercase())
            .unwrap_or_else(|| "PCM".into());
        let rate = params.sample_rate.unwrap_or(44100);
        let channels = params.channels.map(|c| c.count() as u32).unwrap_or(2);
        let time_base = params.time_base;
        let duration = params.n_frames.map(|n| {
            // n_frames is in time_base units; for PCM codecs that equals sample frames.
            match time_base {
                Some(tb) => {
                    let t = tb.calc_time(n);
                    t.seconds as f64 + t.frac
                }
                None => n as f64 / rate as f64,
            }
        });
        let mut tags = Tags::default();
        let mut read_tags = |rev: &symphonia::core::meta::MetadataRevision| {
            use symphonia::core::meta::StandardTagKey as K;
            for t in rev.tags() {
                match t.std_key {
                    Some(K::TrackTitle) => tags.title = Some(t.value.to_string()),
                    Some(K::Artist) => tags.artist = Some(t.value.to_string()),
                    Some(K::Album) => tags.album = Some(t.value.to_string()),
                    _ => {}
                }
            }
        };
        if let Some(rev) = probed.metadata.get().as_ref().and_then(|m| m.current()) {
            read_tags(rev);
        }
        if let Some(rev) = probed.format.metadata().current() {
            read_tags(rev);
        }
        let info = TrackInfo {
            codec: normalise_codec(&codec),
            rate,
            bits: params.bits_per_sample.unwrap_or(0),
            channels,
            duration,
            tags,
        };
        Ok(Self {
            reader: probed.format,
            decoder,
            track_id,
            info,
            time_base,
            buf: None,
            next_ts: 0,
            finished: false,
            range: (0, None),
        })
    }

    /// Format and metadata.
    pub fn info(&self) -> &TrackInfo {
        &self.info
    }

    /// Where the next returned frame sits in the track, in seconds.
    pub fn position_seconds(&self) -> f64 {
        let abs = match self.time_base {
            Some(tb) => {
                let t = tb.calc_time(self.next_ts);
                t.seconds as f64 + t.frac
            }
            None => self.next_ts as f64 / self.info.rate as f64,
        };
        (abs - self.range.0 as f64 / self.info.rate as f64).max(0.0)
    }

    /// Play only sample frames `start..end` (`None`: to the end), exactly: the first and
    /// last packets are trimmed. Seeks and positions are then relative to `start`.
    pub fn set_range(&mut self, start: u64, end: Option<u64>) -> Result<(), DecodeError> {
        let total = self
            .info
            .duration
            .map(|d| (d * self.info.rate as f64).round() as u64);
        self.range = (start, end);
        if let Some(len) = end.or(total).map(|e| e.saturating_sub(start)) {
            self.info.duration = Some(len as f64 / self.info.rate as f64);
        }
        self.seek(0.0)?;
        Ok(())
    }

    /// Decode the next packet. `Ok(None)` at the end of the stream. Decode errors in the
    /// middle of a stream are skipped (logged); I/O errors end the stream.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f64>>, DecodeError> {
        if self.finished {
            return Ok(None);
        }
        loop {
            let packet = match self.reader.next_packet() {
                Ok(p) => p,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.finished = true;
                    return Ok(None);
                }
                Err(SymphoniaError::ResetRequired) => {
                    self.decoder.reset();
                    continue;
                }
                Err(e) => {
                    self.finished = true;
                    return Err(e.into());
                }
            };
            if packet.track_id() != self.track_id {
                continue;
            }
            let first = ts_frames(self.time_base, self.info.rate, packet.ts());
            match self.decoder.decode(&packet) {
                Ok(audio) => {
                    let spec = *audio.spec();
                    let frames = audio.frames();
                    if frames == 0 {
                        continue;
                    }
                    let buf = match &mut self.buf {
                        Some(b) if b.capacity() >= frames * spec.channels.count() => b,
                        _ => self
                            .buf
                            .insert(SampleBuffer::<f64>::new(audio.capacity() as u64, spec)),
                    };
                    buf.copy_interleaved_ref(audio);
                    self.next_ts = packet.ts() + packet.dur();
                    let (start, end) = self.range;
                    if start == 0 && end.is_none() {
                        return Ok(Some(buf.samples().to_vec()));
                    }
                    let ch = spec.channels.count();
                    if end.is_some_and(|e| first >= e) {
                        self.finished = true;
                        return Ok(None);
                    }
                    let skip = (start.saturating_sub(first) as usize).min(frames);
                    let keep = end.map_or(frames, |e| ((e - first) as usize).min(frames));
                    if skip >= keep {
                        continue;
                    }
                    return Ok(Some(buf.samples()[skip * ch..keep * ch].to_vec()));
                }
                Err(SymphoniaError::DecodeError(e)) => {
                    log::warn!("skipping undecodable packet: {e}");
                    continue;
                }
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.finished = true;
                    return Ok(None);
                }
                Err(e) => {
                    self.finished = true;
                    return Err(e.into());
                }
            }
        }
    }

    /// Seek to `seconds`; returns where the stream actually landed (packet boundary).
    pub fn seek(&mut self, seconds: f64) -> Result<f64, DecodeError> {
        let seconds = seconds.max(0.0) + self.range.0 as f64 / self.info.rate as f64;
        let time = Time::new(seconds.trunc() as u64, seconds.fract());
        let seeked = self.reader.seek(
            SeekMode::Accurate,
            SeekTo::Time {
                time,
                track_id: Some(self.track_id),
            },
        )?;
        self.decoder.reset();
        self.finished = false;
        self.next_ts = seeked.actual_ts;
        Ok(self.position_seconds())
    }

    /// Duration as a `Duration`, when known.
    pub fn duration(&self) -> Option<Duration> {
        self.info.duration.map(Duration::from_secs_f64)
    }
}

/// One playable entry of a disc image or CUE sheet.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// `album.iso#3` / `album.cue#3`.
    pub source: Source,
    /// Title.
    pub title: String,
    /// Artist.
    pub artist: String,
    /// Album.
    pub album: String,
    /// Seconds.
    pub duration: Option<f64>,
}

/// The tracks of a bare `.iso` (SACD image) or `.cue` sheet, in order; `None` for any
/// other source (and for a single track such as `album.iso#3`).
pub fn expand(source: &Source) -> Option<Result<Vec<Entry>, String>> {
    let Source::Path { path } = source else {
        return None;
    };
    let s = path.to_str()?;
    let lower = s.to_ascii_lowercase();
    if lower.ends_with(".iso") {
        let res = (|| {
            let mut f =
                std::io::BufReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
            let disc = sacd::read_disc(&mut f).map_err(|e| e.to_string())?;
            // The stereo area when there is one.
            let area = disc
                .area(false)
                .or(disc.areas.first())
                .ok_or("no audio area")?;
            let prefix = if area.multichannel { "m" } else { "" };
            Ok(area
                .tracks
                .iter()
                .map(|t| Entry {
                    source: Source::Path {
                        path: format!("{s}#{prefix}{}", t.number).into(),
                    },
                    title: t
                        .title
                        .clone()
                        .unwrap_or_else(|| format!("Track {}", t.number)),
                    artist: t
                        .performer
                        .clone()
                        .or(disc.artist.clone())
                        .unwrap_or_default(),
                    album: disc.title.clone().unwrap_or_default(),
                    duration: Some(t.frames as f64 / 75.0),
                })
                .collect())
        })();
        return Some(res);
    }
    if lower.ends_with(".cue") {
        return Some(cue::read(path).map(|sheet| {
            sheet
                .tracks
                .iter()
                .enumerate()
                .map(|(i, t)| Entry {
                    source: Source::Path {
                        path: format!("{s}#{}", i + 1).into(),
                    },
                    title: t
                        .title
                        .clone()
                        .unwrap_or_else(|| format!("Track {}", t.number)),
                    artist: t.performer.clone().unwrap_or_default(),
                    album: sheet.title.clone().unwrap_or_default(),
                    duration: t.end.map(|e| (e - t.start) as f64 / 75.0),
                })
                .collect()
        }));
    }
    None
}

/// Sample frames from a timestamp.
fn ts_frames(tb: Option<TimeBase>, rate: u32, ts: u64) -> u64 {
    match tb {
        Some(tb) => (ts as u128 * rate as u128 * tb.numer as u128 / tb.denom.max(1) as u128) as u64,
        None => ts,
    }
}

/// Symphonia short names → the labels people expect.
fn normalise_codec(short: &str) -> String {
    match short {
        "PCM_S16LE" | "PCM_S24LE" | "PCM_S32LE" | "PCM_S16BE" | "PCM_S24BE" | "PCM_S32BE"
        | "PCM_F32LE" | "PCM_F64LE" | "PCM_U8" | "PCM_S8" => "PCM".into(),
        s if s.starts_with("PCM") => "PCM".into(),
        "MP3" | "MPA" => "MP3".into(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav(path: &std::path::Path, rate: u32, bits: u16, frames: usize) -> Vec<i32> {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: rate,
            bits_per_sample: bits,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        let mut written = Vec::new();
        for i in 0..frames {
            for c in 0..2 {
                let v = (((i * 7 + c * 3) as i64 * 1237) % (1 << (bits - 1)) - (1 << (bits - 2)))
                    as i32;
                written.push(v);
                match bits {
                    16 => w.write_sample(v as i16).unwrap(),
                    _ => w.write_sample(v).unwrap(),
                }
            }
        }
        w.finalize().unwrap();
        written
    }

    #[test]
    fn decodes_a_wav_bit_exactly_and_seeks() {
        let dir = std::env::temp_dir().join(format!("cadenza-dec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        let ints = write_wav(&path, 44100, 24, 20_000);
        let mut d = Decoder::open(&Source::Path { path: path.clone() }).unwrap();
        let info = d.info().clone();
        assert_eq!(
            (info.codec.as_str(), info.rate, info.bits, info.channels),
            ("PCM", 44100, 24, 2)
        );
        assert!((info.duration.unwrap() - 20_000.0 / 44100.0).abs() < 1e-6);
        let mut all = Vec::new();
        while let Some(chunk) = d.next_chunk().unwrap() {
            all.extend(chunk);
        }
        assert_eq!(all.len(), ints.len());
        let scale = (1u64 << 23) as f64;
        for (f, i) in all.iter().zip(&ints) {
            assert_eq!((f * scale).round() as i32, *i);
        }
        let got = d.seek(0.25).unwrap();
        assert!((got - 0.25).abs() < 0.05, "seeked to {got}");
        let chunk = d.next_chunk().unwrap().unwrap();
        let frame = (got * 44100.0).round() as usize;
        assert_eq!((chunk[0] * scale).round() as i32, ints[frame * 2]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
