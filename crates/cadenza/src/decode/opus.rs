//! Opus for symphonia (which has the Ogg and WebM demuxers but no Opus decoder), through
//! the pure-Rust `opus-decoder`. Encore streams some libraries as Opus.

use opus_decoder::OpusDecoder;
use std::sync::OnceLock;
use symphonia::core::audio::{
    AsAudioBufferRef, AudioBuffer, AudioBufferRef, Channels, Signal, SignalSpec,
};
use symphonia::core::codecs::{
    CodecDescriptor, CodecParameters, CodecRegistry, Decoder, DecoderOptions, FinalizeResult,
    CODEC_TYPE_OPUS,
};
use symphonia::core::errors::{Error, Result};
use symphonia::core::formats::Packet;
use symphonia::core::support_codec;

/// Opus always decodes at 48 kHz.
const RATE: u32 = 48_000;
/// The longest Opus packet: 120 ms.
const MAX_FRAMES: usize = 5_760;

/// symphonia's codecs plus Opus.
pub fn codecs() -> &'static CodecRegistry {
    static REG: OnceLock<CodecRegistry> = OnceLock::new();
    REG.get_or_init(|| {
        let mut r = CodecRegistry::new();
        symphonia::default::register_enabled_codecs(&mut r);
        r.register_all::<Opus>();
        r
    })
}

pub struct Opus {
    dec: OpusDecoder,
    params: CodecParameters,
    channels: usize,
    pcm: Vec<i16>,
    buf: AudioBuffer<f32>,
}

impl Decoder for Opus {
    fn try_new(params: &CodecParameters, _: &DecoderOptions) -> Result<Self> {
        let channels = params.channels.map(|c| c.count()).unwrap_or(2);
        let layout = match channels {
            1 => Channels::FRONT_LEFT,
            2 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
            _ => return Err(Error::Unsupported("opus: more than two channels")),
        };
        let dec = OpusDecoder::new(RATE, channels)
            .map_err(|_| Error::DecodeError("opus: cannot start the decoder"))?;
        let mut params = params.clone();
        params.with_sample_rate(RATE);
        Ok(Self {
            dec,
            params,
            channels,
            pcm: vec![0; MAX_FRAMES * channels],
            buf: AudioBuffer::new(MAX_FRAMES as u64, SignalSpec::new(RATE, layout)),
        })
    }

    fn supported_codecs() -> &'static [CodecDescriptor] {
        &[support_codec!(CODEC_TYPE_OPUS, "opus", "Opus")]
    }

    fn reset(&mut self) {
        self.dec.reset();
    }

    fn codec_params(&self) -> &CodecParameters {
        &self.params
    }

    fn decode(&mut self, packet: &Packet) -> Result<AudioBufferRef<'_>> {
        let n = self
            .dec
            .decode(&packet.data, &mut self.pcm, false)
            .map_err(|_| Error::DecodeError("opus: bad packet"))?;
        self.buf.clear();
        self.buf.render_reserved(Some(n));
        for c in 0..self.channels {
            let plane = self.buf.chan_mut(c);
            for (i, s) in plane.iter_mut().enumerate().take(n) {
                *s = self.pcm[i * self.channels + c] as f32 / 32_768.0;
            }
        }
        // The demuxer marks the encoder's pre-skip and the end padding for trimming.
        self.buf
            .trim(packet.trim_start() as usize, packet.trim_end() as usize);
        Ok(self.buf.as_audio_buffer_ref())
    }

    fn finalize(&mut self) -> FinalizeResult {
        FinalizeResult::default()
    }

    fn last_decoded(&self) -> AudioBufferRef<'_> {
        self.buf.as_audio_buffer_ref()
    }
}
