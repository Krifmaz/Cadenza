//! Packing planar 1-bit streams into what a device takes.
//!
//! **DoP** (DSD over PCM, dCS/Playback Designs open standard 1.1): every 24-bit PCM frame
//! carries, per channel, a marker byte in bits 23–16 and 16 DSD bits (oldest in bit 15)
//! below it. The marker alternates `0x05`, `0xFA`, `0x05`, … frame by frame, identical
//! on every channel, which is how the DAC tells DoP from ordinary PCM. The PCM frame
//! rate is `f_dsd / 16` (DSD64 → 176.4 kHz, DSD256 → 705.6 kHz). The words produced
//! here are sign-extended 24-bit values in `i32`, exactly like the pipeline's own
//! 24-bit output, so an output backend handles them as any 24-bit PCM.
//!
//! **Native** words are 32 DSD bits per channel per word, oldest bit in the MSB, as the
//! `i32` whose big-endian bytes are the four bytes of the stream in time order:
//! `i32::from_be_bytes([oldest, …, newest])`. [`words_to_bytes`] turns such words into
//! the byte layout a driver wants ([`NativeFormat`]): ALSA `DSD_U32_BE` is the big-endian
//! bytes, `DSD_U32_LE` the same bytes reversed (ALSA's DSD_U32_LE puts the oldest byte
//! last), `DSD_U8` one byte per channel per frame, channels interleaved.

use super::DsdPacking;

/// The two DoP marker bytes, in order.
pub const DOP_MARKERS: [u8; 2] = [0x05, 0xFA];
/// DSD bits per channel in one DoP frame.
pub const DOP_BITS_PER_FRAME: usize = 16;
/// DSD bits per channel in one native word.
pub const NATIVE_BITS_PER_WORD: usize = 32;

/// Byte layout of native DSD for a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeFormat {
    /// ALSA `DSD_U32_BE`: four bytes per channel per frame, oldest first, MSB oldest bit.
    U32Be,
    /// ALSA `DSD_U32_LE`: four bytes per channel per frame, oldest byte last.
    U32Le,
    /// ALSA `DSD_U8`: one byte per channel per frame, MSB oldest bit.
    U8,
}

impl NativeFormat {
    /// `dsd_u32_be` / `dsd_u32_le` / `dsd_u8`.
    pub fn id(self) -> &'static str {
        match self {
            NativeFormat::U32Be => "dsd_u32_be",
            NativeFormat::U32Le => "dsd_u32_le",
            NativeFormat::U8 => "dsd_u8",
        }
    }
}

/// One DoP word for one channel: `marker` over two DSD bytes (`hi` is the older one).
#[inline]
pub fn dop_word(marker: u8, hi: u8, lo: u8) -> i32 {
    let w = ((marker as u32) << 16) | ((hi as u32) << 8) | lo as u32;
    // Sign-extend from 24 bits, like every other 24-bit sample in the pipeline.
    ((w << 8) as i32) >> 8
}

/// One native word from four DSD bytes in time order.
#[inline]
pub fn native_word(bytes: [u8; 4]) -> i32 {
    i32::from_be_bytes(bytes)
}

/// Streaming packer: planar bytes in, device words out.
#[derive(Clone, Debug)]
pub struct Packer {
    /// How DSD bytes are laid out in device words.
    packing: DsdPacking,
    /// Interleaved channel count (at least 1).
    channels: usize,
    /// Index into `DOP_MARKERS` of the next DoP word's marker (alternates every frame).
    marker: usize,
    /// Per-channel DSD bytes not yet packed into a whole word.
    pending: Vec<Vec<u8>>,
    /// Device frames written so far.
    frames: u64,
}

impl Packer {
    /// A packer for `channels` channels.
    pub fn new(packing: DsdPacking, channels: usize) -> Self {
        Self {
            packing,
            channels: channels.max(1),
            marker: 0,
            pending: vec![Vec::new(); channels.max(1)],
            frames: 0,
        }
    }

    /// The packing.
    pub fn packing(&self) -> DsdPacking {
        self.packing
    }
    /// Channels.
    pub fn channels(&self) -> usize {
        self.channels
    }
    /// DSD bytes per channel consumed by one output frame (2 for DoP, 4 native).
    pub fn bytes_per_frame(&self) -> usize {
        match self.packing {
            DsdPacking::Dop => DOP_BITS_PER_FRAME / 8,
            DsdPacking::Native => NATIVE_BITS_PER_WORD / 8,
        }
    }
    /// Valid bits in each output word (24 for DoP, 32 native).
    pub fn word_bits(&self) -> u32 {
        match self.packing {
            DsdPacking::Dop => 24,
            DsdPacking::Native => 32,
        }
    }
    /// Device frame rate for a DSD rate (`f_dsd / 16` for DoP, `/ 32` native).
    pub fn frame_rate(&self, dsd_rate: u32) -> u32 {
        dsd_rate / (8 * self.bytes_per_frame()) as u32
    }
    /// Frames emitted so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }
    /// Bytes per channel waiting for a complete frame.
    pub fn pending_bytes(&self) -> usize {
        self.pending[0].len()
    }

    /// Pack planar bytes (one slice per channel, equal lengths), appending interleaved
    /// words to `out`. Leftover bytes that do not fill a frame are kept.
    pub fn push(&mut self, planar: &[&[u8]], out: &mut Vec<i32>) {
        assert_eq!(planar.len(), self.channels, "channel count mismatch");
        for (p, bytes) in self.pending.iter_mut().zip(planar) {
            p.extend_from_slice(bytes);
        }
        let per = self.bytes_per_frame();
        let frames = self
            .pending
            .iter()
            .map(|p| p.len() / per)
            .min()
            .unwrap_or(0);
        out.reserve(frames * self.channels);
        for f in 0..frames {
            let i = f * per;
            match self.packing {
                DsdPacking::Dop => {
                    let marker = DOP_MARKERS[self.marker];
                    self.marker ^= 1;
                    for p in &self.pending {
                        out.push(dop_word(marker, p[i], p[i + 1]));
                    }
                }
                DsdPacking::Native => {
                    for p in &self.pending {
                        out.push(native_word([p[i], p[i + 1], p[i + 2], p[i + 3]]));
                    }
                }
            }
        }
        for p in &mut self.pending {
            p.drain(..frames * per);
        }
        self.frames += frames as u64;
    }

    /// Pad the pending bytes with DSD silence (`0x69`) to a whole frame and emit it.
    pub fn flush(&mut self, out: &mut Vec<i32>) {
        let per = self.bytes_per_frame();
        let have = self.pending_bytes();
        if have % per != 0 {
            let pad = vec![0x69u8; per - have % per];
            let planar: Vec<&[u8]> = (0..self.channels).map(|_| pad.as_slice()).collect();
            self.push(&planar, out);
        }
    }

    /// Drop pending bytes and restart the marker sequence.
    pub fn reset(&mut self) {
        for p in &mut self.pending {
            p.clear();
        }
        self.marker = 0;
    }
}

/// Lay native words (interleaved, `channels` per frame) out as bytes for a driver.
pub fn words_to_bytes(words: &[i32], channels: usize, format: NativeFormat) -> Vec<u8> {
    let channels = channels.max(1);
    let mut out = Vec::with_capacity(words.len() * 4);
    match format {
        NativeFormat::U32Be => {
            for w in words {
                out.extend_from_slice(&w.to_be_bytes());
            }
        }
        NativeFormat::U32Le => {
            for w in words {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        NativeFormat::U8 => {
            for frame in words.chunks(channels) {
                let bytes: Vec<[u8; 4]> = frame.iter().map(|w| w.to_be_bytes()).collect();
                for b in 0..4 {
                    for ch in &bytes {
                        out.push(ch[b]);
                    }
                }
            }
        }
    }
    out
}

/// Unpack DoP words back into planar DSD bytes (tests, loopback); returns `None` when a
/// marker is wrong.
pub fn unpack_dop(words: &[i32], channels: usize) -> Option<Vec<Vec<u8>>> {
    let mut planar = vec![Vec::new(); channels];
    for (f, frame) in words.chunks(channels).enumerate() {
        let want = DOP_MARKERS[f % 2];
        for (c, w) in frame.iter().enumerate() {
            let u = (*w as u32) & 0xFF_FFFF;
            if (u >> 16) as u8 != want {
                return None;
            }
            planar[c].push((u >> 8) as u8);
            planar[c].push(u as u8);
        }
    }
    Some(planar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dop_markers_alternate_and_bits_are_in_place() {
        let mut p = Packer::new(DsdPacking::Dop, 2);
        let l: Vec<u8> = (0..8).collect();
        let r: Vec<u8> = (100..108).collect();
        let mut out = Vec::new();
        p.push(&[&l, &r], &mut out);
        assert_eq!(out.len(), 8); // 4 frames × 2 channels
        for (f, frame) in out.chunks(2).enumerate() {
            let marker = DOP_MARKERS[f % 2];
            assert_eq!(((frame[0] as u32) >> 16) as u8, marker);
            assert_eq!(((frame[1] as u32) >> 16) as u8, marker);
            assert_eq!((frame[0] as u32 >> 8) as u8, l[2 * f]);
            assert_eq!(frame[0] as u32 as u8, l[2 * f + 1]);
            assert_eq!((frame[1] as u32 >> 8) as u8, r[2 * f]);
        }
        // 0xFA frames are negative 24-bit values (sign extended).
        assert!(out[2] < 0 && out[0] > 0);
        assert_eq!(unpack_dop(&out, 2).unwrap(), vec![l.clone(), r.clone()]);
        // The marker sequence continues across pushes.
        p.push(&[&l[..2], &r[..2]], &mut out);
        assert_eq!(((out[8] as u32) >> 16) as u8, DOP_MARKERS[0]);
        // A stray marker is detected.
        let mut bad = out.clone();
        bad[0] ^= 0x01_0000;
        assert!(unpack_dop(&bad, 2).is_none());
    }

    #[test]
    fn dop_frame_rate_and_leftovers() {
        let mut p = Packer::new(DsdPacking::Dop, 1);
        assert_eq!(p.frame_rate(2_822_400), 176_400);
        assert_eq!(p.word_bits(), 24);
        let mut out = Vec::new();
        p.push(&[&[1u8, 2, 3]], &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(p.pending_bytes(), 1);
        p.flush(&mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1] as u32 & 0xFFFF, 0x0369);
    }

    #[test]
    fn native_words_and_byte_layouts() {
        let mut p = Packer::new(DsdPacking::Native, 2);
        assert_eq!(p.frame_rate(11_289_600), 352_800);
        let l = [0x11u8, 0x22, 0x33, 0x44];
        let r = [0xAAu8, 0xBB, 0xCC, 0xDD];
        let mut out = Vec::new();
        p.push(&[&l, &r], &mut out);
        assert_eq!(out, vec![0x1122_3344, 0xAABB_CCDDu32 as i32]);
        assert_eq!(
            words_to_bytes(&out, 2, NativeFormat::U32Be),
            vec![0x11, 0x22, 0x33, 0x44, 0xAA, 0xBB, 0xCC, 0xDD]
        );
        assert_eq!(
            words_to_bytes(&out, 2, NativeFormat::U32Le),
            vec![0x44, 0x33, 0x22, 0x11, 0xDD, 0xCC, 0xBB, 0xAA]
        );
        assert_eq!(
            words_to_bytes(&out, 2, NativeFormat::U8),
            vec![0x11, 0xAA, 0x22, 0xBB, 0x33, 0xCC, 0x44, 0xDD]
        );
    }
}
