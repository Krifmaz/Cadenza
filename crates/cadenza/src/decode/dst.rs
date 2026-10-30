//! DST (Direct Stream Transfer) decoding: the lossless compression of DSD used on SACDs and
//! in DST-compressed DSDIFF files (ISO/IEC 14496-3, subpart 10).
//!
//! A frame holds 1/75 s of every channel. It is either stored plain, or coded: per channel
//! and segment a prediction filter (up to 128 taps over the past bits) and a probability
//! table, then one arithmetic-coded bit stream for all channels. Each output bit is the
//! filter's predicted bit corrected by an arithmetic-decoded residual, so decoding is
//! exact – the output is the original DSD, bit for bit.
//!
//! Written from the standard's description and checked against an independent decoder
//! (FFmpeg's, in a test that runs when `ffmpeg` is installed). Unlike FFmpeg this decoder
//! also takes frames with several filter / table segments per channel.

/// DST frames per second.
pub const FRAMES_PER_SECOND: u32 = 75;

/// Most channels a DST stream carries.
const MAX_CHANNELS: usize = 6;
/// Most prediction-filter segments per channel.
const MAX_FILTER_SEGMENTS: usize = 4;
/// Most probability-table segments per channel.
const MAX_PTABLE_SEGMENTS: usize = 8;
/// Shortest filter segment in bits.
const MIN_FILTER_SEGMENT_BITS: usize = 1024;
/// Shortest probability-table segment in bits.
const MIN_PTABLE_SEGMENT_BITS: usize = 32;
/// Prediction of the coded filter coefficients / table entries from the previous ones.
const FILTER_CODE_PRED: [&[i32]; 3] = [&[-8], &[-16, 8], &[-9, -5, 6]];
/// Prediction of the coded probability-table entries from the previous ones.
const PTABLE_CODE_PRED: [&[i32]; 3] = [&[-8], &[-16, 8], &[-24, 24, -8]];

/// Why a frame did not decode.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DstError {
    /// The frame ends inside its header.
    #[error("DST frame truncated")]
    Truncated,
    /// A header field out of range.
    #[error("DST frame invalid: {0}")]
    Invalid(&'static str),
    /// Channel count / rate the format does not allow.
    #[error("DST: {0}")]
    Unsupported(String),
}

/// Bits per channel in a frame at `rate` Hz (37 632 for DSD64).
pub fn bits_per_frame(rate: u32) -> usize {
    (rate / FRAMES_PER_SECOND) as usize
}

/// MSB-first bit reader over one frame.
struct Bits<'a> {
    /// The frame bytes.
    data: &'a [u8],
    /// Next bit to read.
    pos: usize,
}

impl<'a> Bits<'a> {
    /// A reader at the first bit of `data`.
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Bits in the frame.
    fn total(&self) -> usize {
        self.data.len() * 8
    }

    /// One bit; zero past the end (the arithmetic code is defined that way).
    fn bit_or_zero(&mut self) -> u32 {
        let b = self
            .data
            .get(self.pos / 8)
            .map_or(0, |byte| (byte >> (7 - self.pos % 8)) & 1);
        self.pos += 1;
        b as u32
    }

    /// The next `n` bits as an unsigned value; [`DstError::Truncated`] past the end.
    fn get(&mut self, n: usize) -> Result<u32, DstError> {
        if self.pos + n > self.total() {
            return Err(DstError::Truncated);
        }
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit_or_zero();
        }
        Ok(v)
    }

    /// One bit as a flag.
    fn flag(&mut self) -> Result<bool, DstError> {
        Ok(self.get(1)? == 1)
    }

    /// The next `n` bits as a two's-complement value.
    fn signed(&mut self, n: usize) -> Result<i32, DstError> {
        let v = self.get(n)? as i32;
        Ok(if v >= 1 << (n - 1) { v - (1 << n) } else { v })
    }

    /// A Rice code with parameter `m` (unary run, `m` low bits, then a sign bit when
    /// non-zero).
    fn rice(&mut self, m: usize) -> Result<i32, DstError> {
        let mut run = 0i32;
        while !self.flag()? {
            run += 1;
            if run > 1 << 16 {
                return Err(DstError::Invalid("rice run"));
            }
        }
        let v = (run << m) + self.get(m)? as i32;
        Ok(if v != 0 && self.flag()? { -v } else { v })
    }
}

/// Bits needed to write `x` (the standard's "log2, rounded up").
fn bit_len(x: usize) -> usize {
    (usize::BITS - x.leading_zeros()) as usize
}

/// Segmentation of one kind of table: per channel, the segment lengths in bits (the last
/// segment runs to the end of the frame).
#[derive(Clone, Debug, PartialEq)]
struct Segments {
    /// Per channel, segment start bits.
    starts: Vec<Vec<usize>>,
    /// Per channel and segment, the table used.
    table: Vec<Vec<usize>>,
    /// Number of distinct tables.
    tables: usize,
}

/// Read one segmentation (filters or probability tables): per channel the segment start
/// bits, or one shared by every channel.
fn read_segments(
    r: &mut Bits,
    channels: usize,
    frame_bytes: usize,
    max_segs: usize,
    min_bits: usize,
) -> Result<Vec<Vec<usize>>, DstError> {
    let mut resolution = 0usize;
    let mut starts: Vec<Vec<usize>> = Vec::with_capacity(channels);
    let read_channel = |r: &mut Bits, resolution: &mut usize| -> Result<Vec<usize>, DstError> {
        let mut s = vec![0usize];
        let mut defined = 0usize;
        let mut max_size = frame_bytes - min_bits / 8;
        while !r.flag()? {
            if s.len() >= max_segs {
                return Err(DstError::Invalid("too many segments"));
            }
            if *resolution == 0 {
                *resolution = r.get(bit_len(frame_bytes - min_bits / 8))? as usize;
                if *resolution == 0 || *resolution > frame_bytes - min_bits / 8 {
                    return Err(DstError::Invalid("segment resolution"));
                }
            }
            let len = r.get(bit_len(max_size / *resolution))? as usize;
            let bits = *resolution * 8 * len;
            if bits < min_bits || bits > frame_bytes * 8 - defined - min_bits {
                return Err(DstError::Invalid("segment length"));
            }
            defined += bits;
            max_size -= *resolution * len;
            s.push(defined);
        }
        Ok(s)
    };
    if r.flag()? {
        // The same for all channels.
        let s = read_channel(r, &mut resolution)?;
        starts.resize(channels, s);
    } else {
        for _ in 0..channels {
            starts.push(read_channel(r, &mut resolution)?);
        }
    }
    Ok(starts)
}

/// Read which table each segment uses; returns the mapping and the table count.
fn read_mapping(
    r: &mut Bits,
    starts: &[Vec<usize>],
    max_tables: usize,
) -> Result<(Vec<Vec<usize>>, usize), DstError> {
    let mut count = 1usize;
    let next = |r: &mut Bits, count: &mut usize| -> Result<usize, DstError> {
        let t = r.get(bit_len(*count))? as usize;
        if t == *count {
            *count += 1;
        } else if t > *count {
            return Err(DstError::Invalid("table number"));
        }
        Ok(t)
    };
    let mut table: Vec<Vec<usize>> = Vec::with_capacity(starts.len());
    if r.flag()? {
        let n = starts[0].len();
        if starts.iter().any(|s| s.len() != n) {
            return Err(DstError::Invalid("channel mapping"));
        }
        let mut t0 = vec![0usize];
        for _ in 1..n {
            t0.push(next(r, &mut count)?);
        }
        table.resize(starts.len(), t0);
    } else {
        for (ch, s) in starts.iter().enumerate() {
            let mut t = Vec::with_capacity(s.len());
            for seg in 0..s.len() {
                t.push(if ch == 0 && seg == 0 {
                    0
                } else {
                    next(r, &mut count)?
                });
            }
            table.push(t);
        }
    }
    if count > max_tables {
        return Err(DstError::Invalid("too many tables"));
    }
    Ok((table, count))
}

/// Read a predictively coded table of `len` entries: the first entries verbatim, the rest
/// as Rice-coded residuals of `pred`, each checked against `range`.
fn read_coded_table(
    r: &mut Bits,
    len: usize,
    entry_bits: usize,
    signed: bool,
    pred: &[&[i32]; 3],
    range: (i32, i32),
) -> Result<Vec<i32>, DstError> {
    let entry = |r: &mut Bits| -> Result<i32, DstError> {
        if signed {
            r.signed(entry_bits)
        } else {
            Ok(r.get(entry_bits)? as i32 + 1)
        }
    };
    let mut v = Vec::with_capacity(len);
    if !r.flag()? {
        for _ in 0..len {
            v.push(entry(r)?);
        }
        return Ok(v);
    }
    let method = r.get(2)? as usize;
    let p = *pred.get(method).ok_or(DstError::Invalid("coding method"))?;
    if p.len() >= len {
        return Err(DstError::Invalid("coding order"));
    }
    for _ in 0..p.len() {
        v.push(entry(r)?);
    }
    let m = r.get(3)? as usize;
    for i in p.len()..len {
        let x: i32 = p.iter().enumerate().map(|(t, c)| c * v[i - t - 1]).sum();
        let mut c = r.rice(m)?;
        if x >= 0 {
            c -= (x + 4) / 8;
        } else {
            c += (-x + 3) / 8;
        }
        if c < range.0 || c > range.1 {
            return Err(DstError::Invalid("coded value out of range"));
        }
        v.push(c);
    }
    Ok(v)
}

/// The arithmetic decoder (12-bit, with the standard's "partial rounding").
struct Arith {
    /// Interval width.
    a: u32,
    /// Code value.
    c: u32,
}

impl Arith {
    /// A decoder primed with the first 12 bits.
    fn new(r: &mut Bits) -> Self {
        let mut c = 0;
        for _ in 0..12 {
            c = (c << 1) | r.bit_or_zero();
        }
        Self { a: 4095, c }
    }

    /// Decode one bit with probability `p` (of a 0, 8-bit) and renormalise.
    #[inline]
    fn bit(&mut self, r: &mut Bits, p: u32) -> u32 {
        let ap = ((self.a >> 8) | ((self.a >> 7) & 1)) * p;
        let h = self.a - ap;
        let b = if self.c >= h {
            self.c -= h;
            self.a = ap;
            0
        } else {
            self.a = h;
            1
        };
        while self.a < 2048 {
            self.a <<= 1;
            self.c = (self.c << 1) | r.bit_or_zero();
        }
        b
    }
}

/// Per-filter lookup: 16 tables of 256 partial sums, one per byte of history.
type FilterTable = [[i16; 256]; 16];

/// Precompute the partial sums of `coefs` for every byte of history.
fn filter_table(coefs: &[i32]) -> Box<FilterTable> {
    let mut t = Box::new([[0i16; 256]; 16]);
    for (j, row) in t.iter_mut().enumerate() {
        let taps = coefs.len().saturating_sub(j * 8).min(8);
        for (k, v) in row.iter_mut().enumerate() {
            let mut s = 0i32;
            for l in 0..taps {
                s += (((k >> l) & 1) as i32 * 2 - 1) * coefs[j * 8 + l];
            }
            *v = s as i16;
        }
    }
    t
}

/// The standard's mapping of the first filter coefficient to the probability of the
/// reserved first bit: the 7 low bits reversed, plus one.
fn x_bit_probability(c: i32) -> u32 {
    ((c as u32 & 127).reverse_bits() >> 25) + 1
}

/// A DST decoder for one stream.
pub struct DstDecoder {
    /// Channels in the stream.
    channels: usize,
    /// Bits per channel in one frame.
    bits_per_ch: usize,
    /// Frames whose arithmetic code did not end where it should (decoded anyway).
    pub flush_warnings: u64,
}

impl DstDecoder {
    /// A decoder for `channels` channels at DSD rate `rate`.
    pub fn new(channels: u32, rate: u32) -> Result<Self, DstError> {
        let channels = channels as usize;
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(DstError::Unsupported(format!("{channels} channels")));
        }
        let bits_per_ch = bits_per_frame(rate);
        if rate % FRAMES_PER_SECOND != 0 || bits_per_ch % 8 != 0 || bits_per_ch == 0 {
            return Err(DstError::Unsupported(format!("rate {rate} Hz")));
        }
        Ok(Self {
            channels,
            bits_per_ch,
            flush_warnings: 0,
        })
    }

    /// Bytes of one decoded frame (all channels, byte-interleaved).
    pub fn frame_bytes(&self) -> usize {
        self.channels * self.bits_per_ch / 8
    }

    /// Decode one frame into `out` ([`DstDecoder::frame_bytes`] long): DSD bytes, one per
    /// channel in turn, oldest bit in the MSB (the DSDIFF layout).
    pub fn decode(&mut self, frame: &[u8], out: &mut [u8]) -> Result<(), DstError> {
        let ch_n = self.channels;
        let frame_bytes = self.bits_per_ch / 8;
        assert_eq!(out.len(), self.frame_bytes());
        let mut r = Bits::new(frame);
        if !r.flag()? {
            // Stored plain.
            r.get(1)?;
            if r.get(6)? != 0 {
                return Err(DstError::Invalid("stuffing"));
            }
            let raw = &frame[1..];
            let n = raw.len().min(out.len());
            out[..n].copy_from_slice(&raw[..n]);
            out[n..].fill(0x69);
            return Ok(());
        }

        // Segmentation, then mapping, of filters and probability tables.
        let p_same_seg = r.flag()?;
        let f_starts = read_segments(
            &mut r,
            ch_n,
            frame_bytes,
            MAX_FILTER_SEGMENTS,
            MIN_FILTER_SEGMENT_BITS,
        )?;
        let p_starts = if p_same_seg {
            if f_starts.iter().any(|s| s.len() > MAX_PTABLE_SEGMENTS) {
                return Err(DstError::Invalid("too many segments"));
            }
            f_starts.clone()
        } else {
            read_segments(
                &mut r,
                ch_n,
                frame_bytes,
                MAX_PTABLE_SEGMENTS,
                MIN_PTABLE_SEGMENT_BITS,
            )?
        };
        let p_same_map = r.flag()?;
        let (f_table, n_filters) = read_mapping(&mut r, &f_starts, 2 * ch_n)?;
        let (p_table, n_ptables) = if p_same_map {
            if p_starts
                .iter()
                .zip(&f_starts)
                .any(|(p, f)| p.len() != f.len())
            {
                return Err(DstError::Invalid("segment count mismatch"));
            }
            (f_table.clone(), n_filters)
        } else {
            read_mapping(&mut r, &p_starts, 2 * ch_n)?
        };
        let fseg = Segments {
            starts: f_starts,
            table: f_table,
            tables: n_filters,
        };
        let pseg = Segments {
            starts: p_starts,
            table: p_table,
            tables: n_ptables,
        };
        let mut half_prob = [false; MAX_CHANNELS];
        for h in half_prob.iter_mut().take(ch_n) {
            *h = r.flag()?;
        }

        // Filters and probability tables.
        let mut filters = Vec::with_capacity(fseg.tables);
        for _ in 0..fseg.tables {
            let order = r.get(7)? as usize + 1;
            filters.push(read_coded_table(
                &mut r,
                order,
                9,
                true,
                &FILTER_CODE_PRED,
                (-256, 255),
            )?);
        }
        let mut ptables = Vec::with_capacity(pseg.tables);
        for _ in 0..pseg.tables {
            let len = r.get(6)? as usize + 1;
            ptables.push(if len > 1 {
                read_coded_table(&mut r, len, 7, false, &PTABLE_CODE_PRED, (1, 128))?
            } else {
                vec![128]
            });
        }

        // The arithmetic-coded bits.
        if r.pos < r.total() && r.get(1)? != 0 {
            return Err(DstError::Invalid("arithmetic code start"));
        }
        let tables: Vec<Box<FilterTable>> = filters.iter().map(|f| filter_table(f)).collect();
        let half_bits: Vec<usize> = (0..ch_n).map(|c| filters[fseg.table[c][0]].len()).collect();
        let mut ac = Arith::new(&mut r);
        ac.bit(&mut r, x_bit_probability(filters[0][0]));

        let mut status = [u128::from_le_bytes([0xaa; 16]); MAX_CHANNELS];
        // Per channel: the current segment of each kind and where the next one starts.
        let mut fs = [0usize; MAX_CHANNELS];
        let mut ps = [0usize; MAX_CHANNELS];
        let next_start = |seg: &Segments, c: usize, s: usize| -> usize {
            seg.starts[c].get(s + 1).copied().unwrap_or(usize::MAX)
        };
        let mut f_next: Vec<usize> = (0..ch_n).map(|c| next_start(&fseg, c, 0)).collect();
        let mut p_next: Vec<usize> = (0..ch_n).map(|c| next_start(&pseg, c, 0)).collect();
        out.fill(0);
        for bit in 0..self.bits_per_ch {
            for c in 0..ch_n {
                if bit == f_next[c] {
                    fs[c] += 1;
                    f_next[c] = next_start(&fseg, c, fs[c]);
                }
                if bit == p_next[c] {
                    ps[c] += 1;
                    p_next[c] = next_start(&pseg, c, ps[c]);
                }
                let fi = fseg.table[c][fs[c]];
                let t = &tables[fi];
                let taps = filters[fi].len().div_ceil(8);
                let st = status[c];
                let mut predict = 0i16;
                for (j, row) in t.iter().enumerate().take(taps) {
                    predict = predict.wrapping_add(row[((st >> (8 * j)) & 0xff) as usize]);
                }
                let p = if half_prob[c] && bit < half_bits[c] {
                    128
                } else {
                    let pt = &ptables[pseg.table[c][ps[c]]];
                    pt[((predict as i32).unsigned_abs() as usize >> 3).min(pt.len() - 1)] as u32
                };
                let residual = ac.bit(&mut r, p);
                let v = ((predict as u16 >> 15) as u32 ^ residual) & 1;
                out[(bit / 8) * ch_n + c] |= (v as u8) << (7 - bit % 8);
                status[c] = (st << 1) | v as u128;
            }
        }
        // The code should end within the last byte.
        if r.pos + 7 < r.total() {
            self.flush_warnings += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod encode {
    //! A DST encoder for the tests: exercises every part of the bitstream the decoder
    //! reads (segments, mappings, coded and plain tables, half probability, plain
    //! frames). The filters are a simple least-effort predictor, not a good one.

    use super::*;

    struct Writer {
        bits: Vec<u8>,
    }

    impl Writer {
        fn put(&mut self, v: u32, n: usize) {
            for i in (0..n).rev() {
                self.bits.push(((v >> i) & 1) as u8);
            }
        }
        fn signed(&mut self, v: i32, n: usize) {
            self.put((v as u32) & ((1 << n) - 1), n);
        }
        fn rice(&mut self, v: i32, m: usize) {
            let a = v.unsigned_abs();
            for _ in 0..(a >> m) {
                self.bits.push(0);
            }
            self.bits.push(1);
            self.put(a & ((1 << m) - 1), m);
            if a != 0 {
                self.bits.push((v < 0) as u8);
            }
        }
        fn bytes(&self) -> Vec<u8> {
            self.bits
                .chunks(8)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .fold(0u8, |b, (i, v)| b | (v << (7 - i)))
                })
                .collect()
        }
    }

    /// The arithmetic encoder matching [`Arith`].
    struct ArithEnc {
        low: u32,
        a: u32,
        out: Vec<u8>,
    }

    impl ArithEnc {
        fn new() -> Self {
            Self {
                low: 0,
                a: 4095,
                out: Vec::new(),
            }
        }
        fn bit(&mut self, b: u32, p: u32) {
            let ap = ((self.a >> 8) | ((self.a >> 7) & 1)) * p;
            let h = self.a - ap;
            if b == 0 {
                self.low += h;
                self.a = ap;
                if self.low >= 4096 {
                    self.low -= 4096;
                    for x in self.out.iter_mut().rev() {
                        if *x == 1 {
                            *x = 0;
                        } else {
                            *x = 1;
                            break;
                        }
                    }
                }
            } else {
                self.a = h;
            }
            while self.a < 2048 {
                self.a <<= 1;
                self.out.push(((self.low >> 11) & 1) as u8);
                self.low = (self.low << 1) & 4095;
            }
        }
        fn finish(mut self) -> Vec<u8> {
            for i in (0..12).rev() {
                self.out.push(((self.low >> i) & 1) as u8);
            }
            self.out
        }
    }

    /// How to code a frame.
    #[derive(Clone, Debug)]
    pub struct Options {
        /// Filter segment lengths in bytes (all channels), the last runs to the end.
        pub filter_segments: Vec<usize>,
        /// Probability-table segments; `None`: the same as the filters'.
        pub ptable_segments: Option<Vec<usize>>,
        /// One filter / table per channel and segment (else all share table 0 per
        /// segment index).
        pub per_channel_tables: bool,
        /// Half-probability start on channel 0.
        pub half_prob: bool,
        /// Rice-code the tables (method 0..2 by channel).
        pub coded_tables: bool,
        /// Filter order.
        pub order: usize,
    }

    impl Default for Options {
        fn default() -> Self {
            Self {
                filter_segments: Vec::new(),
                ptable_segments: None,
                per_channel_tables: false,
                half_prob: false,
                coded_tables: true,
                order: 32,
            }
        }
    }

    fn write_segments(w: &mut Writer, frame_bytes: usize, segs: &[usize], min_bits: usize) {
        w.put(1, 1); // same for all channels
        let resolution = 8usize;
        let mut max_size = frame_bytes - min_bits / 8;
        for (i, s) in segs.iter().enumerate() {
            w.put(0, 1);
            if i == 0 {
                w.put(resolution as u32, bit_len(frame_bytes - min_bits / 8));
            }
            assert_eq!(s % resolution, 0);
            w.put((s / resolution) as u32, bit_len(max_size / resolution));
            max_size -= s;
        }
        w.put(1, 1);
    }

    fn write_table(w: &mut Writer, v: &[i32], bits: usize, signed: bool, coded: Option<usize>) {
        let put = |w: &mut Writer, x: i32| {
            if signed {
                w.signed(x, bits)
            } else {
                w.put((x - 1) as u32, bits)
            }
        };
        let preds = if signed {
            FILTER_CODE_PRED
        } else {
            PTABLE_CODE_PRED
        };
        match coded {
            Some(method) if preds[method].len() < v.len() => {
                w.put(1, 1);
                w.put(method as u32, 2);
                let p = preds[method];
                for x in &v[..p.len()] {
                    put(w, *x);
                }
                let m = 2usize;
                w.put(m as u32, 3);
                for i in p.len()..v.len() {
                    let x: i32 = p.iter().enumerate().map(|(t, c)| c * v[i - t - 1]).sum();
                    let r = if x >= 0 {
                        v[i] + (x + 4) / 8
                    } else {
                        v[i] - (-x + 3) / 8
                    };
                    w.rice(r, m);
                }
            }
            _ => {
                w.put(0, 1);
                for x in v {
                    put(w, *x);
                }
            }
        }
    }

    /// Encode one frame of byte-interleaved DSD (`channels` × bytes per channel).
    pub fn encode_frame(dsd: &[u8], channels: usize, o: &Options) -> Vec<u8> {
        let frame_bytes = dsd.len() / channels;
        let nbits = frame_bytes * 8;
        let bit_at = |c: usize, i: usize| ((dsd[(i / 8) * channels + c] >> (7 - i % 8)) & 1) as i32;
        let nseg = o.filter_segments.len() + 1;
        let pseg_lens = o
            .ptable_segments
            .clone()
            .unwrap_or_else(|| o.filter_segments.clone());
        let npseg = pseg_lens.len() + 1;
        let starts = |lens: &[usize]| -> Vec<usize> {
            let mut s = vec![0];
            for l in lens {
                s.push(s.last().unwrap() + l * 8);
            }
            s
        };
        let fstarts = starts(&o.filter_segments);
        let pstarts = starts(&pseg_lens);
        let seg_of = |st: &[usize], i: usize| st.iter().rposition(|s| *s <= i).unwrap();
        let ftab = |c: usize, s: usize| {
            if o.per_channel_tables {
                c * nseg + s
            } else {
                s
            }
        };
        let ptab = |c: usize, s: usize| {
            if o.per_channel_tables {
                c * npseg + s
            } else {
                s
            }
        };
        let nf = if o.per_channel_tables {
            channels * nseg
        } else {
            nseg
        };
        let np = if o.per_channel_tables {
            channels * npseg
        } else {
            npseg
        };

        // Filters: least squares on the ±1 signal (Levinson-Durbin), coefficients in
        // 1/64 steps.
        let mut filters = vec![vec![0i32; o.order]; nf];
        for (fi, f) in filters.iter_mut().enumerate() {
            let mut acc = vec![0i64; o.order];
            for c in 0..channels {
                for i in o.order..nbits {
                    let s = seg_of(&fstarts, i);
                    if ftab(c, s) != fi {
                        continue;
                    }
                    let x = bit_at(c, i) * 2 - 1;
                    for (t, a) in acc.iter_mut().enumerate() {
                        *a += (x * (bit_at(c, i - t - 1) * 2 - 1)) as i64;
                    }
                }
            }
            // Levinson-Durbin on the autocorrelation r[0..=order].
            let mut r = vec![0f64; o.order + 1];
            r[0] = 1.0;
            // acc[t] is the lag-(t+1) correlation summed over n; normalise by the count.
            let count = (channels * nbits) as f64;
            for t in 0..o.order {
                r[t + 1] = acc[t] as f64 / count;
            }
            let mut a = vec![0f64; o.order];
            let mut err = r[0];
            for k in 0..o.order {
                let mut acc_k = r[k + 1];
                for j in 0..k {
                    acc_k -= a[j] * r[k - j];
                }
                let refl = if err > 1e-12 { acc_k / err } else { 0.0 };
                let prev = a.clone();
                a[k] = refl;
                for j in 0..k {
                    a[j] = prev[j] - refl * prev[k - 1 - j];
                }
                err *= 1.0 - refl * refl;
            }
            for (t, c) in f.iter_mut().enumerate() {
                *c = (a[t] * 64.0).round().clamp(-256.0, 255.0) as i32;
            }
            if f[0] == 0 {
                f[0] = 1;
            }
        }
        let ftables: Vec<Box<FilterTable>> = filters.iter().map(|f| filter_table(f)).collect();
        let half_bits = filters[ftab(0, 0)].len();

        // Pass 1: predictions and residuals; pass 2 codes them.
        let mut status = vec![u128::from_le_bytes([0xaa; 16]); channels];
        let mut events = Vec::with_capacity(nbits * channels);
        let mut counts = vec![vec![(0u32, 0u32); 64]; np];
        for i in 0..nbits {
            for c in 0..channels {
                let fi = ftab(c, seg_of(&fstarts, i));
                let taps = filters[fi].len().div_ceil(8);
                let mut predict = 0i16;
                for j in 0..taps {
                    predict = predict
                        .wrapping_add(ftables[fi][j][((status[c] >> (8 * j)) & 0xff) as usize]);
                }
                let v = bit_at(c, i) as u32;
                let residual = ((predict as u16 >> 15) as u32 ^ v) & 1;
                let pi = ptab(c, seg_of(&pstarts, i));
                let idx = ((predict as i32).unsigned_abs() as usize >> 3).min(63);
                let half = o.half_prob && c == 0 && i < half_bits;
                if !half {
                    let e = &mut counts[pi][idx];
                    e.0 += (residual == 0) as u32;
                    e.1 += 1;
                }
                events.push((pi, idx, half, residual));
                status[c] = (status[c] << 1) | v as u128;
            }
        }
        let ptables: Vec<Vec<i32>> = counts
            .iter()
            .map(|t| {
                let used = t
                    .iter()
                    .rposition(|e| e.1 > 0)
                    .map_or(2, |i| (i + 1).max(2));
                t[..used]
                    .iter()
                    .map(|(zero, n)| {
                        if *n == 0 {
                            64
                        } else {
                            ((*zero as u64 * 256 + *n as u64 / 2) / *n as u64).clamp(1, 128) as i32
                        }
                    })
                    .collect()
            })
            .collect();

        let mut w = Writer { bits: Vec::new() };
        w.put(1, 1);
        // Segmentation.
        let p_same = o.ptable_segments.is_none();
        w.put(p_same as u32, 1);
        write_segments(
            &mut w,
            frame_bytes,
            &o.filter_segments,
            MIN_FILTER_SEGMENT_BITS,
        );
        if !p_same {
            write_segments(&mut w, frame_bytes, &pseg_lens, MIN_PTABLE_SEGMENT_BITS);
        }
        // Mapping.
        let write_map = |w: &mut Writer, n: usize, tab: &dyn Fn(usize, usize) -> usize| {
            if !o.per_channel_tables {
                w.put(1, 1);
                let mut count = 1;
                for s in 1..n {
                    w.put(tab(0, s) as u32, bit_len(count));
                    count = count.max(tab(0, s) + 1);
                }
            } else {
                w.put(0, 1);
                let mut count = 1;
                for c in 0..channels {
                    for s in 0..n {
                        if c == 0 && s == 0 {
                            continue;
                        }
                        w.put(tab(c, s) as u32, bit_len(count));
                        count = count.max(tab(c, s) + 1);
                    }
                }
            }
        };
        let p_same_map = p_same;
        w.put(p_same_map as u32, 1);
        write_map(&mut w, nseg, &ftab);
        if !p_same_map {
            write_map(&mut w, npseg, &ptab);
        }
        for c in 0..channels {
            w.put((o.half_prob && c == 0) as u32, 1);
        }
        for (i, f) in filters.iter().enumerate() {
            w.put((f.len() - 1) as u32, 7);
            write_table(&mut w, f, 9, true, o.coded_tables.then_some(i % 3));
        }
        for (i, p) in ptables.iter().enumerate() {
            w.put((p.len() - 1) as u32, 6);
            if p.len() > 1 {
                write_table(&mut w, p, 7, false, o.coded_tables.then_some(i % 3));
            }
        }
        w.put(0, 1);
        let mut ac = ArithEnc::new();
        ac.bit(0, x_bit_probability(filters[0][0]));
        for (pi, idx, half, residual) in events {
            let p = if half {
                128
            } else {
                let t = &ptables[pi];
                t[idx.min(t.len() - 1)] as u32
            };
            ac.bit(residual, p);
        }
        w.bits.extend(ac.finish());
        let coded = w.bytes();
        if coded.len() > dsd.len() {
            // Not worth it: store plain.
            let mut plain = vec![0u8];
            plain.extend_from_slice(dsd);
            return plain;
        }
        coded
    }

    /// A second-order sigma-delta stream of a sine (byte-interleaved, MSB first).
    pub fn test_dsd(channels: usize, bytes_per_ch: usize, freq: f64, rate: f64) -> Vec<u8> {
        let mut out = vec![0u8; channels * bytes_per_ch];
        for c in 0..channels {
            let (mut i1, mut i2) = (0.0f64, 0.0f64);
            let f = freq * (c + 1) as f64;
            for n in 0..bytes_per_ch * 8 {
                let x = 0.5 * (2.0 * std::f64::consts::PI * f * n as f64 / rate).sin();
                let y = if i2 >= 0.0 { 1.0 } else { -1.0 };
                i1 += x - y;
                i2 += i1 - y;
                if y > 0.0 {
                    out[(n / 8) * channels + c] |= 1 << (7 - n % 8);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::encode::*;
    use super::*;

    const RATE: u32 = 2_822_400;

    fn round_trip(channels: usize, o: &Options) -> (usize, usize) {
        let mut dec = DstDecoder::new(channels as u32, RATE).unwrap();
        let dsd = test_dsd(channels, bits_per_frame(RATE) / 8, 1000.0, RATE as f64);
        let frame = encode_frame(&dsd, channels, o);
        let mut out = vec![0u8; dec.frame_bytes()];
        dec.decode(&frame, &mut out).unwrap();
        assert!(out == dsd, "{o:?}: decoded frame differs");
        assert_eq!(dec.flush_warnings, 0, "{o:?}");
        (frame.len(), dsd.len())
    }

    #[test]
    fn frames_round_trip_bit_exact_in_every_coding_mode() {
        // One segment, shared tables, coded: what SACDs mostly hold.
        let (coded, raw) = round_trip(2, &Options::default());
        assert!(coded * 2 < raw, "{coded} of {raw} bytes");
        for o in [
            Options {
                coded_tables: false,
                ..Default::default()
            },
            Options {
                half_prob: true,
                order: 128,
                ..Default::default()
            },
            Options {
                per_channel_tables: true,
                order: 9,
                ..Default::default()
            },
            Options {
                filter_segments: vec![1600, 1200],
                ..Default::default()
            },
            Options {
                filter_segments: vec![2000],
                ptable_segments: Some(vec![400, 800, 1200]),
                half_prob: true,
                ..Default::default()
            },
        ] {
            round_trip(2, &o);
        }
        round_trip(6, &Options::default());
        round_trip(
            5,
            &Options {
                per_channel_tables: true,
                filter_segments: vec![1200],
                order: 64,
                ..Default::default()
            },
        );
    }

    #[test]
    fn plain_frames_and_bad_input() {
        let mut dec = DstDecoder::new(2, RATE).unwrap();
        let dsd: Vec<u8> = (0..dec.frame_bytes()).map(|i| (i * 7) as u8).collect();
        let mut frame = vec![0u8];
        frame.extend_from_slice(&dsd);
        let mut out = vec![0u8; dec.frame_bytes()];
        dec.decode(&frame, &mut out).unwrap();
        assert_eq!(out, dsd);
        // Short plain frame: padded with DSD silence.
        dec.decode(&frame[..11], &mut out).unwrap();
        assert_eq!(&out[..10], &dsd[..10]);
        assert!(out[10..].iter().all(|b| *b == 0x69));
        assert_eq!(
            dec.decode(&[0x01], &mut out),
            Err(DstError::Invalid("stuffing"))
        );
        assert_eq!(dec.decode(&[0x80], &mut out), Err(DstError::Truncated));
        assert!(DstDecoder::new(7, RATE).is_err());
        assert!(DstDecoder::new(2, 2_822_401).is_err());
        assert_eq!(x_bit_probability(0), 1);
        assert_eq!(x_bit_probability(1), 65);
        assert_eq!(x_bit_probability(-1), 128);
    }
}

#[cfg(test)]
mod speed {
    use super::encode::*;
    use super::*;

    #[test]
    #[ignore = "timing"]
    fn decode_speed() {
        let rate = 2_822_400;
        for ch in [2usize, 6] {
            let dsd = test_dsd(ch, bits_per_frame(rate) / 8, 1000.0, rate as f64);
            let frame = encode_frame(
                &dsd,
                ch,
                &Options {
                    order: 128,
                    ..Default::default()
                },
            );
            let mut dec = DstDecoder::new(ch as u32, rate).unwrap();
            let mut out = vec![0u8; dec.frame_bytes()];
            let t = std::time::Instant::now();
            for _ in 0..75 {
                dec.decode(&frame, &mut out).unwrap();
            }
            eprintln!("{ch} ch: 1 s of DSD64 in {:?}", t.elapsed());
        }
    }
}
