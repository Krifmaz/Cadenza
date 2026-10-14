//! One decoder type for the worker: Symphonia for PCM codecs, the DSF/DFF reader for DSD.
//!
//! A DSD track can be delivered two ways: as PCM at 352.8 / 384 kHz (through the CIC +
//! FIR decimator, for the normal pipeline – and for DSD → DSD re-modulation at another
//! rate, which is what HQPlayer does too) or as the raw 1-bit stream for bit-perfect
//! pass-through to a DoP / native DSD output.

use crate::decode::dsd::{is_dsd_extension, DsdDecodeError, DsdDecoder};
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
}

impl TrackDecoder {
    /// Open a source, picking the reader from the file extension (`.dsf` / `.dff`).
    pub fn open(source: &Source) -> Result<Self, TrackDecodeError> {
        let dsd = source.extension().is_some_and(|e| is_dsd_extension(&e));
        if dsd {
            Ok(TrackDecoder::Dsd(DsdDecoder::open(source)?))
        } else {
            Ok(TrackDecoder::Pcm(Decoder::open(source)?))
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
            TrackDecoder::Pcm(_) => None,
        }
    }

    /// What the source is (codec `DSF`/`DFF`, 1 bit, the DSD rate for DSD).
    pub fn info(&self) -> TrackInfo {
        match self {
            TrackDecoder::Pcm(d) => d.info().clone(),
            TrackDecoder::Dsd(d) => d.info(),
        }
    }

    /// `FLAC 16/44.1` or `DSF DSD64 (2.8224 MHz)`.
    pub fn source_label(&self) -> String {
        match self {
            TrackDecoder::Pcm(d) => d.info().label(),
            TrackDecoder::Dsd(d) => {
                let info = d.dsd_info();
                let name = DsdRate::from_hz(info.rate)
                    .map(|(r, _)| r.label())
                    .unwrap_or_else(|| "DSD".into());
                format!(
                    "{} {name} ({:.4} MHz)",
                    info.container.codec_name(),
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
        }
    }

    /// Position of the next frame to be delivered, seconds.
    pub fn position_seconds(&self) -> f64 {
        match self {
            TrackDecoder::Pcm(d) => d.position_seconds(),
            TrackDecoder::Dsd(d) => d.position_seconds(),
        }
    }

    /// Seek; returns where the stream landed.
    pub fn seek(&mut self, seconds: f64) -> Result<f64, TrackDecodeError> {
        Ok(match self {
            TrackDecoder::Pcm(d) => d.seek(seconds)?,
            TrackDecoder::Dsd(d) => d.seek(seconds)?,
        })
    }

    /// Next interleaved PCM chunk (`Ok(None)` at the end).
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f64>>, TrackDecodeError> {
        Ok(match self {
            TrackDecoder::Pcm(d) => d.next_chunk()?,
            TrackDecoder::Dsd(d) => d.next_chunk()?,
        })
    }

    /// Next chunk of raw planar DSD bytes (MSB first) for pass-through; `Ok(None)` at the
    /// end, and always `None` for a PCM source.
    pub fn next_dsd_chunk(&mut self) -> Result<Option<Vec<Vec<u8>>>, TrackDecodeError> {
        Ok(match self {
            TrackDecoder::Pcm(_) => None,
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
