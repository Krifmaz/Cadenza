//! The stream an AirPlay sender announces (SDP) and the packet decoder: Apple Lossless
//! (what iPhones, Macs and iTunes send) or plain 16-bit PCM.

use super::crypto::{self, Cipher};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CodecParameters, Decoder as _, DecoderOptions, CODEC_TYPE_ALAC};
use symphonia::core::formats::Packet;

/// The codec of an announced stream.
#[derive(Clone, Debug, PartialEq)]
pub enum Codec {
    /// ALAC with the parameters of the `fmtp` line.
    Alac {
        /// Frames per packet (352).
        frame_length: u32,
        /// The 24-byte ALAC "magic cookie" built from `fmtp`.
        cookie: Vec<u8>,
    },
    /// Big-endian 16-bit PCM.
    L16,
}

/// What an ANNOUNCE carries.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamFormat {
    /// Codec.
    pub codec: Codec,
    /// Sample rate (44 100).
    pub rate: u32,
    /// Channels (2).
    pub channels: u32,
    /// Bits per sample (16).
    pub bits: u32,
    /// Session key and IV, when the stream is encrypted.
    pub key: Option<([u8; 16], [u8; 16])>,
}

impl StreamFormat {
    /// Frames per packet.
    pub fn frames_per_packet(&self) -> usize {
        match &self.codec {
            Codec::Alac { frame_length, .. } => *frame_length as usize,
            Codec::L16 => 352,
        }
    }

    /// `ALAC` or `PCM`.
    pub fn codec_label(&self) -> &'static str {
        match self.codec {
            Codec::Alac { .. } => "ALAC",
            Codec::L16 => "PCM",
        }
    }
}

/// Parse the ANNOUNCE body. Errors name what is missing or unsupported.
pub fn parse_sdp(sdp: &str) -> Result<StreamFormat, String> {
    let mut rtpmap = None;
    let mut fmtp = None;
    let mut rsaaeskey = None;
    let mut aesiv = None;
    for line in sdp.lines() {
        let line = line.trim();
        let Some(attr) = line.strip_prefix("a=") else {
            continue;
        };
        let (name, value) = attr.split_once(':').unwrap_or((attr, ""));
        match name {
            "rtpmap" => rtpmap = Some(value.to_string()),
            "fmtp" => fmtp = Some(value.to_string()),
            "rsaaeskey" => rsaaeskey = Some(value.to_string()),
            "aesiv" => aesiv = Some(value.to_string()),
            _ => {}
        }
    }
    let rtpmap = rtpmap.ok_or("no a=rtpmap")?;
    // "96 AppleLossless" or "96 L16/44100/2"
    let encoding = rtpmap.split_whitespace().nth(1).unwrap_or("");
    let key = match (rsaaeskey, aesiv) {
        (Some(k), Some(iv)) => Some((
            crypto::aes_key(&k).ok_or("rsaaeskey does not unwrap")?,
            crypto::aes_iv(&iv).ok_or("bad aesiv")?,
        )),
        (None, None) => None,
        _ => return Err("rsaaeskey without aesiv".into()),
    };
    if encoding.eq_ignore_ascii_case("AppleLossless") {
        let fmtp = fmtp.ok_or("ALAC without a=fmtp")?;
        let f: Vec<u32> = fmtp
            .split_whitespace()
            .skip(1)
            .map(|v| v.parse::<u32>())
            .collect::<Result<_, _>>()
            .map_err(|_| format!("bad fmtp {fmtp}"))?;
        // frameLength compatibleVersion bitDepth pb mb kb numChannels maxRun maxFrameBytes
        // avgBitRate sampleRate
        if f.len() != 11 {
            return Err(format!("bad fmtp {fmtp}"));
        }
        let mut cookie = Vec::with_capacity(24);
        cookie.extend_from_slice(&f[0].to_be_bytes());
        for v in &f[1..7] {
            cookie.push(*v as u8);
        }
        cookie.extend_from_slice(&(f[7] as u16).to_be_bytes());
        cookie.extend_from_slice(&f[8].to_be_bytes());
        cookie.extend_from_slice(&f[9].to_be_bytes());
        cookie.extend_from_slice(&f[10].to_be_bytes());
        if f[2] != 16 && f[2] != 24 {
            return Err(format!("{}-bit ALAC is not supported", f[2]));
        }
        Ok(StreamFormat {
            codec: Codec::Alac {
                frame_length: f[0],
                cookie,
            },
            rate: f[10],
            channels: f[6],
            bits: f[2],
            key,
        })
    } else if let Some(rest) = encoding.strip_prefix("L16") {
        let mut it = rest.trim_start_matches('/').split('/');
        let rate = it.next().and_then(|v| v.parse().ok()).unwrap_or(44100);
        let channels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2);
        Ok(StreamFormat {
            codec: Codec::L16,
            rate,
            channels,
            bits: 16,
            key,
        })
    } else {
        Err(format!(
            "unsupported AirPlay codec {encoding:?} (AAC/Opus are AirPlay 2)"
        ))
    }
}

/// Decrypts and decodes RTP payloads to interleaved `f64`.
pub struct PacketDecoder {
    format: StreamFormat,
    cipher: Option<Cipher>,
    alac: Option<Box<dyn symphonia::core::codecs::Decoder>>,
    buf: Option<SampleBuffer<f64>>,
}

impl PacketDecoder {
    /// A decoder for `format`.
    pub fn new(format: &StreamFormat) -> Result<Self, String> {
        let alac = match &format.codec {
            Codec::Alac { cookie, .. } => {
                let mut params = CodecParameters::new();
                params
                    .for_codec(CODEC_TYPE_ALAC)
                    .with_sample_rate(format.rate)
                    .with_extra_data(cookie.clone().into_boxed_slice());
                Some(Box::new(
                    symphonia::default::codecs::AlacDecoder::try_new(
                        &params,
                        &DecoderOptions::default(),
                    )
                    .map_err(|e| format!("ALAC: {e}"))?,
                )
                    as Box<dyn symphonia::core::codecs::Decoder>)
            }
            Codec::L16 => None,
        };
        Ok(Self {
            cipher: format.key.map(|(k, iv)| Cipher::new(k, iv)),
            format: format.clone(),
            alac,
            buf: None,
        })
    }

    /// Silence for one packet.
    pub fn silence(&self) -> Vec<f64> {
        vec![0.0; self.format.frames_per_packet() * self.format.channels as usize]
    }

    /// Decrypt and decode one payload; a packet that does not decode becomes silence.
    pub fn decode(&mut self, payload: &[u8]) -> Vec<f64> {
        let mut data = payload.to_vec();
        if let Some(c) = &self.cipher {
            c.decrypt(&mut data);
        }
        match &mut self.alac {
            None => data
                .chunks_exact(2)
                .map(|b| i16::from_be_bytes([b[0], b[1]]) as f64 / 32768.0)
                .collect(),
            Some(dec) => match dec.decode(&Packet::new_from_slice(0, 0, 0, &data)) {
                Ok(audio) if audio.frames() > 0 => {
                    let spec = *audio.spec();
                    let frames = audio.frames();
                    let buf = match &mut self.buf {
                        Some(b) if b.capacity() >= frames * spec.channels.count() => b,
                        b => b.insert(SampleBuffer::new(frames as u64, spec)),
                    };
                    buf.copy_interleaved_ref(audio);
                    buf.samples().to_vec()
                }
                Ok(_) => self.silence(),
                Err(e) => {
                    log::debug!("AirPlay: ALAC packet: {e}");
                    self.silence()
                }
            },
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An uncompressed ("escape") ALAC frame of 16-bit stereo, as simple senders make.
    pub(crate) fn alac_frame(samples: &[i16], frame_length: usize) -> Vec<u8> {
        struct Bits {
            out: Vec<u8>,
            acc: u64,
            n: u32,
        }
        impl Bits {
            fn put(&mut self, v: u32, bits: u32) {
                self.acc = (self.acc << bits) | (v as u64 & ((1u64 << bits) - 1));
                self.n += bits;
                while self.n >= 8 {
                    self.n -= 8;
                    self.out.push((self.acc >> self.n) as u8);
                }
            }
        }
        let frames = samples.len() / 2;
        let mut b = Bits {
            out: Vec::new(),
            acc: 0,
            n: 0,
        };
        b.put(1, 3); // CPE
        b.put(0, 4); // element instance tag
        b.put(0, 12); // unused
        let partial = frames != frame_length;
        b.put(partial as u32, 1);
        b.put(0, 2); // no shift
        b.put(1, 1); // escape: uncompressed
        if partial {
            b.put(frames as u32, 32);
        }
        for s in samples {
            b.put(*s as u16 as u32, 16);
        }
        b.put(7, 3); // END
        if b.n > 0 {
            let pad = 8 - b.n;
            b.put(0, pad);
        }
        b.out
    }

    pub(crate) const SDP_ALAC: &str = "v=0\r\no=iTunes 1 O IN IP4 192.168.1.5\r\ns=iTunes\r\n\
        c=IN IP4 192.168.1.5\r\nt=0 0\r\nm=audio 0 RTP/AVP 96\r\na=rtpmap:96 AppleLossless\r\n\
        a=fmtp:96 352 0 16 40 10 14 2 255 0 0 44100\r\n";

    #[test]
    fn sdp_alac_plain_and_encrypted() {
        let f = parse_sdp(SDP_ALAC).unwrap();
        assert_eq!((f.rate, f.channels, f.bits), (44100, 2, 16));
        assert_eq!(f.frames_per_packet(), 352);
        assert!(f.key.is_none());
        let Codec::Alac { cookie, .. } = &f.codec else {
            panic!()
        };
        assert_eq!(cookie.len(), 24);
        assert_eq!(&cookie[..4], &352u32.to_be_bytes());
        assert_eq!(&cookie[20..], &44100u32.to_be_bytes());

        let k = [9u8; 16];
        let sdp = format!(
            "{SDP_ALAC}a=rsaaeskey:{}\r\na=aesiv:AAECAwQFBgcICQoLDA0ODw\r\n",
            super::super::crypto::tests::wrap_key(k)
        );
        let f = parse_sdp(&sdp).unwrap();
        let iv: Vec<u8> = (0..16).collect();
        assert_eq!(f.key, Some((k, iv.try_into().unwrap())));

        let l16 = parse_sdp("m=audio 0 RTP/AVP 96\na=rtpmap:96 L16/44100/2\n").unwrap();
        assert_eq!(l16.codec, Codec::L16);
        assert!(parse_sdp("a=rtpmap:96 mpeg4-generic/44100/2\n").is_err());
        assert!(parse_sdp("a=rtpmap:96 AppleLossless\n").is_err());
    }

    #[test]
    fn decodes_encrypted_alac_and_l16() {
        let k = [3u8; 16];
        let iv = [4u8; 16];
        let mut f = parse_sdp(SDP_ALAC).unwrap();
        f.key = Some((k, iv));
        let mut dec = PacketDecoder::new(&f).unwrap();
        let samples: Vec<i16> = (0..352 * 2).map(|i| (i * 37 - 9000) as i16).collect();
        let mut pkt = alac_frame(&samples, 352);
        super::super::crypto::tests::encrypt(k, iv, &mut pkt);
        let out = dec.decode(&pkt);
        let want: Vec<f64> = samples.iter().map(|s| *s as f64 / 32768.0).collect();
        assert_eq!(out, want);
        // A short (partial) frame.
        let mut pkt = alac_frame(&samples[..200], 352);
        super::super::crypto::tests::encrypt(k, iv, &mut pkt);
        assert_eq!(dec.decode(&pkt), want[..200]);
        // Garbage becomes one packet of silence.
        assert_eq!(dec.decode(&[0xff; 7]).len(), 704);

        let l16 = parse_sdp("a=rtpmap:96 L16/44100/2\n").unwrap();
        let mut dec = PacketDecoder::new(&l16).unwrap();
        assert_eq!(dec.decode(&[0x40, 0x00, 0xc0, 0x00]), vec![0.5, -0.5]);
    }
}
