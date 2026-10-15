//! One decoder type for the worker: Symphonia for PCM codecs, the DSF/DFF reader for DSD.
//!
//! A DSD track can be delivered two ways: as PCM at 352.8 / 384 kHz (through the CIC +
//! FIR decimator, for the normal pipeline – and for DSD → DSD re-modulation at another
//! rate, which is what HQPlayer does too) or as the raw 1-bit stream for bit-perfect
//! pass-through to a DoP / native DSD output.

use crate::decode::dsd::{is_dsd_source, DsdDecodeError, DsdDecoder};
use crate::decode::live::LiveDecoder;
use crate::decode::{DecodeError, Decoder, Source, TrackInfo};
use cadenza_dsp::dsd::DsdRate;
use cadenza_dsp::signal_path::rate_label;
use cadenza_dsp::InputFormat;

/// Errors from either decoder.
#[derive(Debug, thiserror::Error)]
pub enum TrackDecodeError {
    /// The PCM decoder.
    #[error("{0}")]
    Pcm(#[from] DecodeError),
    /// The DSD reader.
    #[error("{0}")]
    Dsd(#[from] DsdDecodeError),
}

/// A PCM or DSD track decoder.
pub enum TrackDecoder {
    /// Symphonia.
    Pcm(Decoder),
    /// DSF / DFF.
    Dsd(DsdDecoder),
    /// PCM pushed in by a receiver (AirPlay).
    Live(LiveDecoder),
}

impl TrackDecoder {
    /// Open a source, picking the reader from the file extension (`.dsf` / `.dff`).
    pub fn open(source: &Source) -> Result<Self, TrackDecodeError> {
        if let Some((sheet, n)) = crate::decode::cue::cue_track(source) {
            return Self::open_cue(&sheet, n);
        }
        if let Source::Live { live } = source {
            return LiveDecoder::open(live)
                .map(TrackDecoder::Live)
                .map_err(|e| TrackDecodeError::Pcm(e.into()));
        }
        let dsd = is_dsd_source(source);
        if dsd {
            Ok(TrackDecoder::Dsd(DsdDecoder::open(source)?))
        } else {
            Ok(TrackDecoder::Pcm(Decoder::open(source)?))
        }
    }

    /// Track `n` (1-based) of the CUE sheet at `sheet`: its file, cut to the track.
    fn open_cue(sheet: &std::path::Path, n: u32) -> Result<Self, TrackDecodeError> {
        use crate::decode::cue;
        let io_err = |m: String| TrackDecodeError::Pcm(std::io::Error::other(m).into());
        let s = cue::read(sheet).map_err(io_err)?;
        let t = s
            .tracks
            .get(n as usize - 1)
            .ok_or_else(|| io_err(format!("{}: no track {n}", sheet.display())))?;
        let file = Source::Path {
            path: cue::resolve(sheet, &t.file),
        };
        // CUE times are 1/75 s: whole samples at the usual rates.
        let at = |frames: u64, rate: u64| frames * rate / 75;
        if is_dsd_source(&file) {
            let mut d = DsdDecoder::open(&file)?;
            let rate = d.dsd_info().rate as u64;
            d.set_range(at(t.start, rate), t.end.map(|e| at(e, rate)));
            Ok(TrackDecoder::Dsd(d))
        } else {
            let mut d = Decoder::open(&file)?;
            let rate = d.info().rate as u64;
            d.set_range(at(t.start, rate), t.end.map(|e| at(e, rate)))?;
            Ok(TrackDecoder::Pcm(d))
        }
    }

    /// `true` for a DSF / DFF source.
    pub fn is_dsd(&self) -> bool {
        matches!(self, TrackDecoder::Dsd(_))
    }

    /// The DSD bit rate of a DSD source.
    pub fn dsd_rate(&self) -> Option<u32> {
        match self {
            TrackDecoder::Dsd(d) => Some(d.dsd_info().rate),
            TrackDecoder::Pcm(_) | TrackDecoder::Live(_) => None,
        }
    }

    /// What the source is (codec `DSF`/`DFF`, 1 bit, the DSD rate for DSD).
    pub fn info(&self) -> TrackInfo {
        match self {
            TrackDecoder::Pcm(d) => d.info().clone(),
            TrackDecoder::Dsd(d) => d.info(),
            TrackDecoder::Live(d) => d.info(),
        }
    }

    /// `FLAC 16/44.1` or `DSF DSD64 (2.8224 MHz)`.
    pub fn source_label(&self) -> String {
        match self {
            TrackDecoder::Pcm(d) => d.info().label(),
            TrackDecoder::Live(d) => d.info().label(),
            TrackDecoder::Dsd(d) => {
                let info = d.dsd_info();
                let name = DsdRate::from_hz(info.rate)
                    .map(|(r, _)| r.label())
                    .unwrap_or_else(|| "DSD".into());
                format!(
                    "{} {name} ({:.4} MHz)",
                    info.codec_label(),
                    info.rate as f64 / 1e6
                )
            }
        }
    }

    /// What [`TrackDecoder::next_chunk`] delivers to the pipeline: the PCM format of the
    /// source, or 352.8 / 384 kHz 32-bit PCM for a DSD source.
    pub fn pcm_input(&self) -> InputFormat {
        match self {
            TrackDecoder::Pcm(d) => {
                let i = d.info();
                InputFormat {
                    rate: i.rate,
                    bits: if i.bits == 0 { 24 } else { i.bits },
                    channels: i.channels,
                    codec: i.codec.clone(),
                }
            }
            TrackDecoder::Dsd(d) => d.pcm_input_format(),
            TrackDecoder::Live(d) => {
                let i = d.info();
                InputFormat {
                    rate: i.rate,
                    bits: i.bits,
                    channels: i.channels,
                    codec: i.codec,
                }
            }
        }
    }

    /// `(rate, bits, channels, dsd)` – what decides whether the chain must be rebuilt.
    pub fn format_key(&self) -> (u32, u32, u32, bool) {
        let i = self.info();
        (i.rate, i.bits, i.channels, self.is_dsd())
    }

    /// Duration in seconds, when known.
    pub fn duration(&self) -> Option<f64> {
        match self {
            TrackDecoder::Pcm(d) => d.info().duration,
            TrackDecoder::Dsd(d) => d.duration().map(|d| d.as_secs_f64()),
            TrackDecoder::Live(_) => None,
        }
    }

    /// Position of the next frame to be delivered, seconds.
    pub fn position_seconds(&self) -> f64 {
        match self {
            TrackDecoder::Pcm(d) => d.position_seconds(),
            TrackDecoder::Dsd(d) => d.position_seconds(),
            TrackDecoder::Live(d) => d.position_seconds(),
        }
    }

    /// Seek; returns where the stream landed.
    pub fn seek(&mut self, seconds: f64) -> Result<f64, TrackDecodeError> {
        Ok(match self {
            TrackDecoder::Pcm(d) => d.seek(seconds)?,
            TrackDecoder::Dsd(d) => d.seek(seconds)?,
            // A live stream plays on from where it is.
            TrackDecoder::Live(d) => d.position_seconds(),
        })
    }

    /// Next interleaved PCM chunk (`Ok(None)` at the end).
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f64>>, TrackDecodeError> {
        Ok(match self {
            TrackDecoder::Pcm(d) => d.next_chunk()?,
            TrackDecoder::Dsd(d) => d.next_chunk()?,
            TrackDecoder::Live(d) => d.next_chunk(),
        })
    }

    /// Next chunk of raw planar DSD bytes (MSB first) for pass-through; `Ok(None)` at the
    /// end, and always `None` for a PCM source.
    pub fn next_dsd_chunk(&mut self) -> Result<Option<Vec<Vec<u8>>>, TrackDecodeError> {
        Ok(match self {
            TrackDecoder::Pcm(_) | TrackDecoder::Live(_) => None,
            TrackDecoder::Dsd(d) => d.next_dsd_chunk()?,
        })
    }

    /// `DSD64` style label of a DSD source rate.
    pub fn dsd_label(rate: u32) -> String {
        DsdRate::from_hz(rate)
            .map(|(r, _)| r.label())
            .unwrap_or_else(|| format!("DSD {} kHz", rate_label(rate)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::dsd::{write_dff, write_dsf};
    use crate::decode::Tags;
    use crate::engine::Track;
    use std::path::{Path, PathBuf};

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cadenza-trk-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn all_pcm(d: &mut TrackDecoder) -> Vec<f64> {
        let mut v = Vec::new();
        while let Some(c) = d.next_chunk().unwrap() {
            v.extend(c);
        }
        v
    }

    fn all_dsd(d: &mut TrackDecoder) -> Vec<Vec<u8>> {
        let mut v: Vec<Vec<u8>> = Vec::new();
        while let Some(c) = d.next_dsd_chunk().unwrap() {
            v.resize(c.len(), Vec::new());
            for (o, p) in v.iter_mut().zip(c) {
                o.extend(p);
            }
        }
        v
    }

    fn interleaved(planar: &[Vec<u8>]) -> Vec<u8> {
        (0..planar[0].len())
            .flat_map(|i| planar.iter().map(move |p| p[i]))
            .collect()
    }

    fn open(path: &Path) -> TrackDecoder {
        TrackDecoder::open(&Source::Path { path: path.into() }).unwrap()
    }

    #[test]
    fn sacd_image_plays_as_tracks() {
        let d = dir("sacd");
        let img = crate::decode::sacd::tests::image(&[5, 7, 3], true);
        let iso = d.join("Disc.iso");
        std::fs::write(&iso, &img.bytes).unwrap();
        let tracks = Track::from_arg(iso.to_str().unwrap()).expand().unwrap();
        assert_eq!(tracks.len(), 3);
        assert_eq!(tracks[1].title, "Track 2");
        assert_eq!(tracks[1].album, "Test Album");
        assert_eq!(tracks[1].artist, "Quartet");
        assert_eq!(tracks[2].duration, Some(3.0 / 75.0));
        for (t, want) in tracks.iter().zip(&img.tracks) {
            let mut dec = TrackDecoder::open(&t.source).unwrap();
            assert!(dec.is_dsd());
            assert!(dec.source_label().starts_with("SACD DST DSD64"));
            assert_eq!(interleaved(&all_dsd(&mut dec)), *want);
        }
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn dsd_tracks_continue_through_the_converter_gaplessly() {
        // Three SACD tracks decoded to PCM with the converter handed on give exactly
        // what the same DSD decoded as one file gives.
        let d = dir("gapless");
        let img = crate::decode::sacd::tests::image(&[5, 7, 3], false);
        let iso = d.join("Disc.iso");
        std::fs::write(&iso, &img.bytes).unwrap();
        let tracks = Track::from_arg(iso.to_str().unwrap()).expand().unwrap();
        let mut joined = Vec::new();
        let mut prev: Option<DsdDecoder> = None;
        for (i, t) in tracks.iter().enumerate() {
            let TrackDecoder::Dsd(mut dec) = TrackDecoder::open(&t.source).unwrap() else {
                panic!()
            };
            if let Some(p) = prev.as_mut() {
                assert!(dec.adopt(p));
            }
            dec.set_continues(i + 1 < tracks.len());
            while let Some(c) = dec.next_chunk().unwrap() {
                joined.extend(c);
            }
            prev = Some(dec);
        }
        let all: Vec<u8> = img.tracks.concat();
        let planar: Vec<Vec<u8>> = (0..2)
            .map(|c| all.iter().skip(c).step_by(2).copied().collect())
            .collect();
        let whole = d.join("whole.dff");
        write_dff(&whole, 2_822_400, &planar, &Tags::default()).unwrap();
        let reference = all_pcm(&mut open(&whole));
        assert_eq!(joined.len(), reference.len());
        let worst = joined
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-12, "{worst}");
        let _ = std::fs::remove_dir_all(d);
    }

    fn write_wav(path: &Path, frames: usize) -> Vec<f64> {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        let mut v = Vec::new();
        for i in 0..frames * 2 {
            let s = ((i as i64 * 7919) % 60000 - 30000) as i16;
            w.write_sample(s).unwrap();
            v.push(s as f64 / 32768.0);
        }
        w.finalize().unwrap();
        v
    }

    const CUE: &str = "TITLE \"Live\"\nPERFORMER \"Band\"\nFILE \"live.wav\" WAVE\n\
        TRACK 01 AUDIO\n TITLE \"One\"\n INDEX 01 00:00:00\n\
        TRACK 02 AUDIO\n TITLE \"Two\"\n INDEX 00 00:00:70\n INDEX 01 00:01:03\n\
        TRACK 03 AUDIO\n TITLE \"Three\"\n INDEX 01 00:02:37\n";

    fn cue_tracks(cue: &Path) -> Vec<Vec<f64>> {
        let tracks = Track::from_arg(cue.to_str().unwrap()).expand().unwrap();
        assert_eq!(tracks.len(), 3);
        assert_eq!(
            (tracks[1].title.as_str(), tracks[1].album.as_str()),
            ("Two", "Live")
        );
        assert_eq!(tracks[0].duration, Some(78.0 / 75.0));
        tracks
            .iter()
            .map(|t| all_pcm(&mut TrackDecoder::open(&t.source).unwrap()))
            .collect()
    }

    #[test]
    fn cue_tracks_join_sample_exact() {
        let d = dir("cue");
        let full = write_wav(&d.join("live.wav"), 44100 * 3 + 123);
        std::fs::write(d.join("live.cue"), CUE).unwrap();
        let parts = cue_tracks(&d.join("live.cue"));
        // Track 2 starts at 1 s + 3/75 s = 45 864 frames.
        assert_eq!(parts[0].len() / 2, 45_864);
        assert_eq!(parts.concat(), full);

        // The same through FLAC, where seeks land on packet boundaries and are trimmed.
        let flac = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(d.join("live.wav"))
            .arg(d.join("live.flac"))
            .status();
        if flac.is_ok_and(|s| s.success()) {
            std::fs::remove_file(d.join("live.wav")).unwrap();
            // The sheet still says live.wav: the converted file is found by its name.
            let parts = cue_tracks(&d.join("live.cue"));
            assert_eq!(parts.concat(), full);
            // Seeking inside a track is relative to the track.
            let tracks = Track::from_arg(d.join("live.cue").to_str().unwrap())
                .expand()
                .unwrap();
            let mut t2 = TrackDecoder::open(&tracks[1].source).unwrap();
            let at = t2.seek(0.5).unwrap();
            assert!((0.4..=0.5).contains(&at), "{at}");
        } else {
            eprintln!("ffmpeg not installed: FLAC CUE check skipped");
        }
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn cue_over_a_dsf_cuts_whole_bytes() {
        let d = dir("cue-dsd");
        let bytes = 2_822_400 / 8 * 3; // 3 s
        let planar: Vec<Vec<u8>> = (0..2)
            .map(|c| (0..bytes).map(|i| (i * 13 + c * 7) as u8).collect())
            .collect();
        write_dsf(
            &d.join("live.dsf"),
            2_822_400,
            &planar,
            true,
            &Tags::default(),
        )
        .unwrap();
        std::fs::write(d.join("live.cue"), CUE.replace("live.wav", "live.dsf")).unwrap();
        let tracks = Track::from_arg(d.join("live.cue").to_str().unwrap())
            .expand()
            .unwrap();
        let mut joined: Vec<Vec<u8>> = vec![Vec::new(), Vec::new()];
        for t in &tracks {
            let mut dec = TrackDecoder::open(&t.source).unwrap();
            assert!(dec.is_dsd());
            for (o, p) in joined.iter_mut().zip(all_dsd(&mut dec)) {
                o.extend(p);
            }
        }
        assert_eq!(joined, planar);
        let _ = std::fs::remove_dir_all(d);
    }
}
