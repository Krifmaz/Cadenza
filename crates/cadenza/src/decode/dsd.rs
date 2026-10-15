//! DSD input: DSF and DFF (DSDIFF) readers, a decoder that delivers either PCM (through
//! `cadenza_dsp::dsd::DsdToPcm`) or the raw 1-bit stream for bit-perfect pass-through,
//! and writers for both containers (tests, tools).
//!
//! * **DSF** (Sony): little-endian; `DSD ` / `fmt ` / `data` chunks; the data is stored in
//!   blocks of `block_size` (4096) bytes per channel, channel after channel; with
//!   `bits_per_sample = 1` the oldest bit of each byte is the **LSB** (reversed here to
//!   the MSB-first convention the DSP uses), with `8` it is the MSB. Tags live in an
//!   ID3v2 block the header points to.
//! * **DFF** (Philips DSDIFF): big-endian IFF; `FRM8` → `FVER`, `PROP` (`FS  `, `CHNL`,
//!   `CMPR`), `DSD ` data interleaved **one byte per channel per frame**, MSB oldest.
//!   or a `DST ` chunk of DST-compressed frames (1/75 s each, decoded losslessly by
//!   [`super::dst`]; a `DSTI` index, when present, makes seeking direct). Artist/title come
//!   from the `DIIN` chunk.
//!
//! Three ways to play DSD, selected by the player:
//!
//! 1. **DSD → PCM**: [`DsdDecoder::next_chunk`] gives 352.8 / 384 kHz PCM for the normal
//!    pipeline (CIC + compensation + windowed sinc, see `cadenza_dsp::dsd::decimate`).
//! 2. **DSD → DSD at another rate**: the same PCM path feeds a DSD-mode pipeline, which
//!    re-modulates. This is also how HQPlayer does it; a 1-bit stream cannot be
//!    resampled without going through multi-bit.
//! 3. **Pass-through**: [`DsdDecoder::next_dsd_chunk`] gives the raw planar bits for
//!    `cadenza_dsp::dsd::Packer` when the output takes the source rate natively or as DoP.

use super::source::Source;
use super::{Tags, TrackInfo};
use cadenza_dsp::dsd::DsdToPcm;
use cadenza_dsp::InputFormat;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// The two containers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsdContainer {
    /// Sony DSF.
    Dsf,
    /// Philips DSDIFF.
    Dff,
    /// A track of an SACD disc image (`album.iso#3`).
    Sacd,
}

impl DsdContainer {
    /// `DSF` / `DFF` as shown in the signal path.
    pub fn codec_name(self) -> &'static str {
        match self {
            DsdContainer::Dsf => "DSF",
            DsdContainer::Dff => "DFF",
            DsdContainer::Sacd => "SACD",
        }
    }
}

/// `album.iso#3` as a source: the image's own source, multichannel area, track number.
pub fn sacd_track(source: &Source) -> Option<(Source, bool, u32)> {
    match source {
        Source::Path { path } => {
            let s = path.to_str()?;
            let (base, mc, n) = super::sacd::parse_fragment(s)?;
            Some((Source::Path { path: base.into() }, mc, n))
        }
        Source::Url { url, headers } => {
            let (base, mc, n) = super::sacd::parse_fragment(url)?;
            Some((
                Source::Url {
                    url: base.to_string(),
                    headers: headers.clone(),
                },
                mc,
                n,
            ))
        }
        Source::Live { .. } => None,
    }
}

/// Whether a source is DSD: a `.dsf` / `.dff` file or an SACD image track.
pub fn is_dsd_source(source: &Source) -> bool {
    source.extension().is_some_and(|e| is_dsd_extension(&e)) || sacd_track(source).is_some()
}

/// Whether a file extension means DSD.
pub fn is_dsd_extension(ext: &str) -> bool {
    matches!(ext.to_ascii_lowercase().as_str(), "dsf" | "dff")
}

/// Everything the header says.
#[derive(Clone, Debug, PartialEq)]
pub struct DsdInfo {
    /// Container.
    pub container: DsdContainer,
    /// DSD bit rate in Hz (2 822 400 for DSD64).
    pub rate: u32,
    /// Channels.
    pub channels: u32,
    /// Frames (bits per channel).
    pub frames: u64,
    /// Tags.
    pub tags: Tags,
    /// DSF: oldest bit in the LSB of each byte.
    lsb_first: bool,
    /// DSF: bytes per channel per block; DFF: 1.
    block_size: usize,
    /// Byte offset of the sample data.
    data_offset: u64,
    /// Length of the sample data in bytes.
    data_len: u64,
    /// DST-compressed DSDIFF.
    dst: Option<DstLayout>,
    /// A DST-coded SACD area.
    sacd_dst: bool,
}

/// Where the DST frames of a DSDIFF file are.
#[derive(Clone, Debug, PartialEq)]
struct DstLayout {
    /// Frames (1/75 s each).
    frames: u64,
    /// First sub-chunk after `FRTE`, and the end of the `DST ` chunk.
    first: u64,
    end: u64,
}

impl DsdInfo {
    /// `DSF`, `DFF` or `DFF DST`.
    pub fn codec_label(&self) -> &'static str {
        match (self.container, &self.dst, self.sacd_dst) {
            (DsdContainer::Dff, Some(_), _) => "DFF DST",
            (DsdContainer::Sacd, _, true) => "SACD DST",
            (c, _, _) => c.codec_name(),
        }
    }

    /// Whether the data is DST-compressed.
    pub fn is_dst(&self) -> bool {
        self.dst.is_some() || self.sacd_dst
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        self.frames as f64 / self.rate as f64
    }
    /// The decoder's [`TrackInfo`] (codec `DSF`/`DFF`, 1 bit, DSD rate).
    pub fn track_info(&self) -> TrackInfo {
        TrackInfo {
            codec: self.codec_label().into(),
            rate: self.rate,
            bits: 1,
            channels: self.channels,
            duration: Some(self.duration()),
            tags: self.tags.clone(),
        }
    }
}

/// DSD decoding errors.
#[derive(Debug, thiserror::Error)]
pub enum DsdDecodeError {
    /// I/O.
    #[error("{0}")]
    Io(#[from] io::Error),
    /// Not a DSF/DFF file, or a damaged one.
    #[error("not a DSD file: {0}")]
    Format(String),
    /// A DST frame that does not decode.
    #[error("DST: {0}")]
    Dst(#[from] super::dst::DstError),
    /// The DSP refused the rates.
    #[error("{0}")]
    Dsp(#[from] cadenza_dsp::dsd::DsdError),
}

fn bad(msg: impl Into<String>) -> DsdDecodeError {
    DsdDecodeError::Format(msg.into())
}

fn read_exact<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    r.read_exact(&mut v)?;
    Ok(v)
}
fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().unwrap())
}
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}
fn u16be(b: &[u8]) -> u16 {
    u16::from_be_bytes(b[..2].try_into().unwrap())
}
fn u32be(b: &[u8]) -> u32 {
    u32::from_be_bytes(b[..4].try_into().unwrap())
}
fn u64be(b: &[u8]) -> u64 {
    u64::from_be_bytes(b[..8].try_into().unwrap())
}

/// A raw 1-bit reader over any seekable byte source.
pub struct DsdReader<R: Read + Seek> {
    src: R,
    info: DsdInfo,
    /// Next frame (bit index per channel) to deliver; always a multiple of 8.
    pos: u64,
    /// Where the source's cursor is (seeks are only issued when it has to move, because
    /// an HTTP source reopens its connection on every seek).
    file_pos: u64,
    dst: Option<Box<DstState>>,
    sacd: Option<Box<SacdState>>,
    /// First frame of the range played (a CUE track); `pos` counts from here.
    origin: u64,
}

/// An SACD track being read.
struct SacdState {
    track: super::sacd::TrackReader,
    /// The frame in `frame`, if any.
    current: Option<u32>,
    frame: Vec<u8>,
}

/// DST decoding state: the frames found so far and the last one decoded.
struct DstState {
    decoder: super::dst::DstDecoder,
    /// `(data offset, length)` of the `DSTF` chunks found so far, in order.
    found: Vec<(u64, u32)>,
    /// Where scanning for the next `DSTF` continues.
    scan: u64,
    /// The `DSTI` index: `(offset, length)` per frame, as stored.
    index: Option<Vec<(u64, u32)>>,
    /// Its offset convention: `Some(true)` when the entries point at the frame data,
    /// `Some(false)` at the chunk header (worked out from the first frame).
    index_at_data: Option<bool>,
    /// The frame in `frame` (byte-interleaved), if any.
    current: Option<u64>,
    frame: Vec<u8>,
    raw: Vec<u8>,
}

impl<R: Read + Seek> DsdReader<R> {
    /// Parse the header.
    pub fn new(mut src: R) -> Result<Self, DsdDecodeError> {
        src.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; 4];
        src.read_exact(&mut magic)?;
        let mut index = None;
        let info = match &magic {
            b"DSD " => parse_dsf(&mut src)?,
            b"FRM8" => parse_dff(&mut src, &mut index)?,
            _ => return Err(bad("neither DSF nor DSDIFF magic")),
        };
        if info.channels == 0 || info.channels > 32 {
            return Err(bad(format!("{} channels", info.channels)));
        }
        if info.rate < 2_000_000 {
            return Err(bad(format!("implausible DSD rate {} Hz", info.rate)));
        }
        let file_pos = src.stream_position()?;
        let dst = match &info.dst {
            Some(l) => {
                let decoder = super::dst::DstDecoder::new(info.channels, info.rate)?;
                Some(Box::new(DstState {
                    frame: vec![0; decoder.frame_bytes()],
                    decoder,
                    found: Vec::new(),
                    scan: l.first,
                    index: index.take(),
                    index_at_data: None,
                    current: None,
                    raw: Vec::new(),
                }))
            }
            None => None,
        };
        Ok(Self {
            src,
            info,
            pos: 0,
            file_pos,
            dst,
            sacd: None,
            origin: 0,
        })
    }

    /// Track `number` (1-based) of an SACD image, from the stereo area or with
    /// `multichannel` the multichannel one.
    pub fn new_sacd(mut src: R, multichannel: bool, number: u32) -> Result<Self, DsdDecodeError> {
        use super::sacd;
        let disc = sacd::read_disc(&mut src).map_err(|e| bad(e.to_string()))?;
        let area = disc
            .area(multichannel)
            .or_else(|| disc.areas.first().filter(|_| !multichannel))
            .ok_or_else(|| bad("no such area on the disc"))?;
        let reader = sacd::TrackReader::new(area, number).map_err(|e| bad(e.to_string()))?;
        let t = reader.track().clone();
        let info = DsdInfo {
            container: DsdContainer::Sacd,
            rate: sacd::RATE,
            channels: area.channels,
            frames: t.frames as u64 * super::dst::bits_per_frame(sacd::RATE) as u64,
            tags: Tags {
                title: t.title.clone().or_else(|| Some(format!("Track {number}"))),
                artist: t.performer.clone().or(disc.artist.clone()),
                album: disc.title.clone(),
            },
            lsb_first: false,
            block_size: 1,
            data_offset: 0,
            data_len: 0,
            dst: None,
            sacd_dst: area.dst,
        };
        Ok(Self {
            src,
            pos: 0,
            file_pos: u64::MAX,
            dst: None,
            origin: 0,
            sacd: Some(Box::new(SacdState {
                frame: vec![0; reader.frame_bytes()],
                track: reader,
                current: None,
            })),
            info,
        })
    }

    /// Header information.
    pub fn info(&self) -> &DsdInfo {
        &self.info
    }
    /// Next frame to be delivered.
    pub fn position_frames(&self) -> u64 {
        self.pos
    }

    fn ensure_at(&mut self, offset: u64) -> io::Result<()> {
        if self.file_pos != offset {
            self.src.seek(SeekFrom::Start(offset))?;
            self.file_pos = offset;
        }
        Ok(())
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.ensure_at(offset)?;
        let got = read_fully(&mut self.src, buf)?;
        self.file_pos += got as u64;
        Ok(got)
    }

    /// `(offset, length)` of the data of DST frame `n`, scanning the chunk list (or using
    /// the index) as far as needed.
    fn dst_locate(&mut self, n: u64) -> Result<Option<(u64, u32)>, DsdDecodeError> {
        let end = self.info.dst.as_ref().map_or(0, |l| l.end);
        loop {
            let st = self.dst.as_mut().expect("DST state");
            if let Some(found) = st.found.get(n as usize) {
                return Ok(Some(*found));
            }
            // With an index, trust it once the first frame has shown which offset it means.
            if let (Some(index), Some(at_data)) = (&st.index, st.index_at_data) {
                return Ok(index
                    .get(n as usize)
                    .map(|(o, l)| (if at_data { *o } else { o + 12 }, *l)));
            }
            let scan = st.scan;
            if scan + 12 > end {
                return Ok(None);
            }
            let mut hdr = [0u8; 12];
            if self.read_at(scan, &mut hdr)? < 12 {
                return Ok(None);
            }
            let size = u64be(&hdr[4..]);
            let st = self.dst.as_mut().expect("DST state");
            if &hdr[0..4] == b"DSTF" {
                st.found.push((scan + 12, size.min(u32::MAX as u64) as u32));
                if st.found.len() == 1 {
                    st.index_at_data =
                        st.index
                            .as_ref()
                            .and_then(|i| i.first())
                            .and_then(|(o, _)| {
                                if *o == scan + 12 {
                                    Some(true)
                                } else if *o == scan {
                                    Some(false)
                                } else {
                                    None
                                }
                            });
                }
            }
            st.scan = scan + 12 + size + (size & 1);
        }
    }

    /// Decode DST frame `n` into the frame buffer (if not there already).
    fn dst_load(&mut self, n: u64) -> Result<bool, DsdDecodeError> {
        if self.dst.as_ref().is_some_and(|s| s.current == Some(n)) {
            return Ok(true);
        }
        let Some((off, len)) = self.dst_locate(n)? else {
            return Ok(false);
        };
        let mut raw = std::mem::take(&mut self.dst.as_mut().expect("DST state").raw);
        raw.resize(len as usize, 0);
        let got = self.read_at(off, &mut raw)?;
        raw.truncate(got);
        let st = self.dst.as_mut().expect("DST state");
        let res = st.decoder.decode(&raw, &mut st.frame);
        st.raw = raw;
        match res {
            Ok(()) => {}
            Err(e) => {
                // One bad frame: play silence for it rather than stop.
                log::warn!("DST frame {n}: {e}");
                st.frame.fill(0x69);
            }
        }
        st.current = Some(n);
        Ok(true)
    }

    fn read_sacd(&mut self, want: usize, planar: &mut [Vec<u8>]) -> Result<(), DsdDecodeError> {
        let ch = self.info.channels as usize;
        let frame_bytes = super::dst::bits_per_frame(self.info.rate) / 8;
        let mut byte_pos = (self.origin + self.pos) / 8;
        let mut collected = 0;
        // The track reader seeks the source itself.
        self.file_pos = u64::MAX;
        while collected < want {
            let n = (byte_pos / frame_bytes as u64) as u32;
            let st = self.sacd.as_mut().expect("SACD state");
            if st.current != Some(n) {
                if !st
                    .track
                    .read_frame(&mut self.src, n, &mut st.frame)
                    .map_err(|e| bad(e.to_string()))?
                {
                    break;
                }
                st.current = Some(n);
            }
            let off = (byte_pos % frame_bytes as u64) as usize;
            let take = (frame_bytes - off).min(want - collected);
            for i in off..off + take {
                for (c, p) in planar.iter_mut().enumerate() {
                    p.push(st.frame[i * ch + c]);
                }
            }
            collected += take;
            byte_pos += take as u64;
        }
        Ok(())
    }

    fn read_dst(&mut self, want: usize, planar: &mut [Vec<u8>]) -> Result<(), DsdDecodeError> {
        let ch = self.info.channels as usize;
        let frame_bytes = super::dst::bits_per_frame(self.info.rate) / 8;
        let mut byte_pos = (self.origin + self.pos) / 8;
        let mut collected = 0;
        while collected < want {
            let n = byte_pos / frame_bytes as u64;
            if !self.dst_load(n)? {
                break;
            }
            let st = self.dst.as_ref().expect("DST state");
            let off = (byte_pos % frame_bytes as u64) as usize;
            let take = (frame_bytes - off).min(want - collected);
            for i in off..off + take {
                for (c, p) in planar.iter_mut().enumerate() {
                    p.push(st.frame[i * ch + c]);
                }
            }
            collected += take;
            byte_pos += take as u64;
        }
        Ok(())
    }

    /// Read up to `max_frames` frames (whole bytes, at least one) as planar MSB-first
    /// bytes, one `Vec` per channel. `Ok(None)` at the end.
    pub fn read_frames(
        &mut self,
        max_frames: usize,
    ) -> Result<Option<Vec<Vec<u8>>>, DsdDecodeError> {
        let ch = self.info.channels as usize;
        let remaining_bytes = self.info.frames.saturating_sub(self.pos).div_ceil(8);
        if remaining_bytes == 0 {
            return Ok(None);
        }
        let want = ((max_frames / 8).max(1) as u64).min(remaining_bytes) as usize;
        let mut planar: Vec<Vec<u8>> = (0..ch).map(|_| Vec::with_capacity(want)).collect();
        let mut byte_pos = (self.origin + self.pos) / 8;
        match self.info.container {
            DsdContainer::Sacd => self.read_sacd(want, &mut planar)?,
            DsdContainer::Dff if self.info.dst.is_some() => self.read_dst(want, &mut planar)?,
            DsdContainer::Dff => {
                let mut buf = vec![0u8; want * ch];
                let got = self.read_at(self.info.data_offset + byte_pos * ch as u64, &mut buf)?;
                let n = got / ch;
                for (i, b) in buf[..n * ch].iter().enumerate() {
                    planar[i % ch].push(*b);
                }
            }
            DsdContainer::Dsf => {
                let bs = self.info.block_size;
                let mut group = vec![0u8; bs * ch];
                let mut collected = 0usize;
                while collected < want {
                    let g = byte_pos / bs as u64;
                    let off = (byte_pos % bs as u64) as usize;
                    let got =
                        self.read_at(self.info.data_offset + g * (bs * ch) as u64, &mut group)?;
                    if got == 0 {
                        break;
                    }
                    // A truncated last group is padded with DSD silence.
                    group[got..].fill(0x69);
                    let n = (bs - off).min(want - collected);
                    for (c, p) in planar.iter_mut().enumerate() {
                        let block = &group[c * bs + off..c * bs + off + n];
                        p.extend(block.iter().map(|b| {
                            if self.info.lsb_first {
                                b.reverse_bits()
                            } else {
                                *b
                            }
                        }));
                    }
                    collected += n;
                    byte_pos += n as u64;
                }
            }
        }
        let n = planar[0].len();
        if n == 0 {
            return Ok(None);
        }
        self.pos = (self.pos + n as u64 * 8).min(self.info.frames.max(self.pos + 1));
        Ok(Some(planar))
    }

    /// Play only frames `start..end` (`None`: to the end); both are rounded down to whole
    /// bytes. Positions and seeks are then relative to `start`.
    pub fn set_range(&mut self, start: u64, end: Option<u64>) {
        let total = self.origin + self.info.frames;
        let start = (start.min(total) / 8) * 8;
        let end = end.map_or(total, |e| (e.min(total) / 8) * 8).max(start);
        self.origin = start;
        self.info.frames = end - start;
        self.pos = 0;
    }

    /// Seek to a frame (rounded down to a whole byte); returns the frame reached.
    pub fn seek_frames(&mut self, frame: u64) -> Result<u64, DsdDecodeError> {
        self.pos = (frame.min(self.info.frames) / 8) * 8;
        Ok(self.pos)
    }
}

fn read_fully<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

fn parse_dsf<R: Read + Seek>(src: &mut R) -> Result<DsdInfo, DsdDecodeError> {
    // "DSD " already consumed.
    let head = read_exact(src, 24)?;
    let chunk_size = u64le(&head[0..]);
    if chunk_size != 28 {
        return Err(bad(format!("DSF header chunk size {chunk_size}")));
    }
    let metadata_ptr = u64le(&head[16..]);
    let fmt_id = read_exact(src, 4)?;
    if &fmt_id[..] != b"fmt " {
        return Err(bad("DSF: fmt chunk missing"));
    }
    let fmt = read_exact(src, 48)?;
    let fmt_size = u64le(&fmt[0..]);
    let version = u32le(&fmt[8..]);
    let format_id = u32le(&fmt[12..]);
    let channels = u32le(&fmt[20..]);
    let rate = u32le(&fmt[24..]);
    let bits = u32le(&fmt[28..]);
    let frames = u64le(&fmt[32..]);
    let block_size = u32le(&fmt[40..]) as usize;
    if version != 1 || format_id != 0 {
        return Err(bad(format!(
            "DSF format version {version} / id {format_id}"
        )));
    }
    if !matches!(bits, 1 | 8) || block_size == 0 {
        return Err(bad(format!(
            "DSF bits per sample {bits}, block {block_size}"
        )));
    }
    if fmt_size > 52 {
        src.seek(SeekFrom::Current((fmt_size - 52) as i64))?;
    }
    let data_id = read_exact(src, 4)?;
    if &data_id[..] != b"data" {
        return Err(bad("DSF: data chunk missing"));
    }
    let data_size = u64le(&read_exact(src, 8)?);
    let data_offset = src.stream_position()?;
    let data_len = data_size.saturating_sub(12);
    let mut tags = Tags::default();
    if metadata_ptr != 0 {
        if let Ok(t) = read_id3(src, metadata_ptr) {
            tags = t;
        }
    }
    Ok(DsdInfo {
        container: DsdContainer::Dsf,
        rate,
        channels,
        frames,
        tags,
        lsb_first: bits == 1,
        block_size,
        data_offset,
        data_len,
        dst: None,
        sacd_dst: false,
    })
}

fn parse_dff<R: Read + Seek>(
    src: &mut R,
    index: &mut Option<Vec<(u64, u32)>>,
) -> Result<DsdInfo, DsdDecodeError> {
    // "FRM8" already consumed.
    let head = read_exact(src, 12)?;
    let form_size = u64be(&head[0..]);
    if &head[8..12] != b"DSD " {
        return Err(bad("DSDIFF: form type is not DSD"));
    }
    let end = 16 + form_size;
    let mut rate = 0u32;
    let mut channels = 0u32;
    let mut compressed = false;
    let mut data: Option<(u64, u64)> = None;
    let mut dst: Option<DstLayout> = None;
    let mut dst_index: Option<Vec<(u64, u32)>> = None;
    let mut tags = Tags::default();
    loop {
        let pos = src.stream_position()?;
        if pos + 12 > end {
            break;
        }
        let mut hdr = [0u8; 12];
        if src.read_exact(&mut hdr).is_err() {
            break;
        }
        let id = &hdr[0..4];
        let size = u64be(&hdr[4..]);
        let body_start = pos + 12;
        let next = body_start + size + (size & 1);
        match id {
            b"PROP" => {
                let kind = read_exact(src, 4)?;
                if &kind[..] != b"SND " {
                    src.seek(SeekFrom::Start(next))?;
                    continue;
                }
                let prop_end = body_start + size;
                while src.stream_position()? + 12 <= prop_end {
                    let mut h = [0u8; 12];
                    src.read_exact(&mut h)?;
                    let sz = u64be(&h[4..]);
                    let here = src.stream_position()?;
                    match &h[0..4] {
                        b"FS  " => rate = u32be(&read_exact(src, 4)?),
                        b"CHNL" => channels = u16be(&read_exact(src, 2)?) as u32,
                        b"CMPR" => {
                            let c = read_exact(src, 4)?;
                            compressed = &c[..] != b"DSD ";
                        }
                        _ => {}
                    }
                    src.seek(SeekFrom::Start(here + sz + (sz & 1)))?;
                }
            }
            b"DSD " => data = Some((body_start, size)),
            b"DST " => {
                // FRTE (frame count, rate) first, then DSTF / DSTC chunks.
                let mut h = [0u8; 12];
                src.read_exact(&mut h)?;
                if &h[0..4] != b"FRTE" || u64be(&h[4..]) < 6 {
                    return Err(bad("DSDIFF: DST chunk without FRTE"));
                }
                let frte = read_exact(src, 6)?;
                let frames = u32be(&frte[0..]) as u64;
                let fps = u16be(&frte[4..]);
                if fps as u32 != super::dst::FRAMES_PER_SECOND {
                    return Err(bad(format!("DSDIFF: DST at {fps} frames/s")));
                }
                let frte_size = u64be(&h[4..]);
                dst = Some(DstLayout {
                    frames,
                    first: body_start + 12 + frte_size + (frte_size & 1),
                    end: body_start + size,
                });
            }
            b"DSTI" => {
                // Index: (offset u64, length u32) per frame. Up to an hour and a half at
                // 75 frames/s is about 5 MB; larger indexes are ignored.
                if size % 12 == 0 && size <= 12 * 75 * 6000 {
                    let body = read_exact(src, size as usize)?;
                    dst_index = Some(
                        body.chunks_exact(12)
                            .map(|e| (u64be(&e[0..]), u32be(&e[8..])))
                            .collect(),
                    );
                }
            }
            b"DIIN" => {
                let diin_end = body_start + size;
                while src.stream_position()? + 12 <= diin_end {
                    let mut h = [0u8; 12];
                    src.read_exact(&mut h)?;
                    let sz = u64be(&h[4..]);
                    let here = src.stream_position()?;
                    if matches!(&h[0..4], b"DIAR" | b"DITI") && sz >= 4 {
                        let n = u32be(&read_exact(src, 4)?) as usize;
                        let text =
                            String::from_utf8_lossy(&read_exact(src, n.min((sz - 4) as usize))?)
                                .to_string();
                        if &h[0..4] == b"DIAR" {
                            tags.artist = Some(text);
                        } else {
                            tags.title = Some(text);
                        }
                    }
                    src.seek(SeekFrom::Start(here + sz + (sz & 1)))?;
                }
            }
            _ => {}
        }
        src.seek(SeekFrom::Start(next))?;
    }
    if channels == 0 || rate == 0 {
        return Err(bad("DSDIFF: missing FS/CHNL properties"));
    }
    if let Some(layout) = dst {
        if !compressed {
            return Err(bad("DSDIFF: DST chunk in an uncompressed file"));
        }
        if layout.frames == 0 {
            return Err(bad("DSDIFF: no DST frames"));
        }
        *index = dst_index.filter(|i| i.len() as u64 >= layout.frames);
        return Ok(DsdInfo {
            container: DsdContainer::Dff,
            rate,
            channels,
            frames: layout.frames * super::dst::bits_per_frame(rate) as u64,
            tags,
            lsb_first: false,
            block_size: 1,
            data_offset: layout.first,
            data_len: layout.end - layout.first,
            dst: Some(layout),
            sacd_dst: false,
        });
    }
    if compressed {
        return Err(bad("DSDIFF: compressed, but no DST chunk"));
    }
    let (data_offset, data_len) = data.ok_or_else(|| bad("DSDIFF: no DSD data chunk"))?;
    Ok(DsdInfo {
        container: DsdContainer::Dff,
        rate,
        channels,
        frames: data_len / channels as u64 * 8,
        tags,
        lsb_first: false,
        block_size: 1,
        data_offset,
        data_len,
        dst: None,
        sacd_dst: false,
    })
}

/// Minimal ID3v2.3/2.4 text-frame reader (TIT2, TPE1, TALB).
fn read_id3<R: Read + Seek>(src: &mut R, at: u64) -> Result<Tags, DsdDecodeError> {
    src.seek(SeekFrom::Start(at))?;
    let h = read_exact(src, 10)?;
    if &h[0..3] != b"ID3" {
        return Err(bad("no ID3 tag"));
    }
    let major = h[3];
    let size = syncsafe(&h[6..10]) as usize;
    let body = read_exact(src, size.min(1 << 24))?;
    let mut tags = Tags::default();
    let mut i = if h[5] & 0x40 != 0 && body.len() >= 4 {
        // Extended header.
        let ext = if major == 4 {
            syncsafe(&body[0..4]) as usize
        } else {
            u32be(&body[0..4]) as usize + 4
        };
        ext.min(body.len())
    } else {
        0
    };
    while i + 10 <= body.len() {
        let id = &body[i..i + 4];
        if id[0] == 0 {
            break;
        }
        let fsize = if major == 4 {
            syncsafe(&body[i + 4..i + 8]) as usize
        } else {
            u32be(&body[i + 4..i + 8]) as usize
        };
        let start = i + 10;
        let end = (start + fsize).min(body.len());
        if matches!(id, b"TIT2" | b"TPE1" | b"TALB") && end > start {
            let text = decode_id3_text(&body[start..end]);
            match id {
                b"TIT2" => tags.title = Some(text),
                b"TPE1" => tags.artist = Some(text),
                _ => tags.album = Some(text),
            }
        }
        i = end;
    }
    Ok(tags)
}

fn syncsafe(b: &[u8]) -> u32 {
    ((b[0] as u32) << 21) | ((b[1] as u32) << 14) | ((b[2] as u32) << 7) | b[3] as u32
}

fn decode_id3_text(b: &[u8]) -> String {
    let (enc, data) = (b[0], &b[1..]);
    let s = match enc {
        1 | 2 => {
            let (bytes, be) = if enc == 1 && data.len() >= 2 && data[0] == 0xFF && data[1] == 0xFE {
                (&data[2..], false)
            } else if enc == 1 && data.len() >= 2 && data[0] == 0xFE && data[1] == 0xFF {
                (&data[2..], true)
            } else {
                (data, true)
            };
            let units: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| {
                    if be {
                        u16::from_be_bytes([c[0], c[1]])
                    } else {
                        u16::from_le_bytes([c[0], c[1]])
                    }
                })
                .collect();
            String::from_utf16_lossy(&units)
        }
        0 => data.iter().map(|&c| c as char).collect(),
        _ => String::from_utf8_lossy(data).to_string(),
    };
    s.trim_end_matches('\0').to_string()
}

/// A decoder for one DSD track with the same shape as [`super::Decoder`].
pub struct DsdDecoder {
    reader: DsdReader<Box<dyn ReadSeek>>,
    converter: DsdToPcm,
    pcm_rate: u32,
    finished: bool,
    /// Frames pushed into the converter since the last reset.
    pushed: u64,
    /// The next track continues this one: leave the converter's tail for it.
    continues: bool,
    /// Converter output that belonged to the previous track (after [`DsdDecoder::adopt`]).
    out_base: u64,
}

/// `Read + Seek + Send` as one trait object.
pub trait ReadSeek: Read + Seek + Send {}
impl<T: Read + Seek + Send> ReadSeek for T {}

/// Frames per read (bits per channel): 32 768 = 4096 bytes, one DSF block.
const CHUNK_FRAMES: usize = 32_768;

impl DsdDecoder {
    /// Open a local file or URL.
    pub fn open(source: &Source) -> Result<Self, DsdDecodeError> {
        if let Some((base, mc, n)) = sacd_track(source) {
            let src: Box<dyn ReadSeek> = match &base {
                Source::Url { url, headers } => {
                    Box::new(super::source::HttpSource::open(url, headers)?)
                }
                Source::Path { path } => Box::new(io::BufReader::new(std::fs::File::open(path)?)),
                Source::Live { .. } => unreachable!("not an image"),
            };
            return Self::from_dsd_reader(DsdReader::new_sacd(src, mc, n)?);
        }
        let src: Box<dyn ReadSeek> = match source {
            Source::Path { path } => Box::new(std::fs::File::open(path)?),
            Source::Url { url, headers } => {
                Box::new(super::source::HttpSource::open(url, headers)?)
            }
            Source::Live { live } => {
                return Err(std::io::Error::other(format!("live stream {live} is not DSD")).into())
            }
        };
        Self::from_reader(src)
    }

    /// Open from any seekable byte source.
    pub fn from_reader(src: Box<dyn ReadSeek>) -> Result<Self, DsdDecodeError> {
        Self::from_dsd_reader(DsdReader::new(src)?)
    }

    fn from_dsd_reader(reader: DsdReader<Box<dyn ReadSeek>>) -> Result<Self, DsdDecodeError> {
        let pcm_rate = DsdToPcm::natural_pcm_rate(reader.info().rate);
        let converter = DsdToPcm::new(
            reader.info().rate,
            pcm_rate,
            reader.info().channels as usize,
        )?;
        Ok(Self {
            reader,
            converter,
            pcm_rate,
            finished: false,
            pushed: 0,
            continues: false,
            out_base: 0,
        })
    }

    /// Mark that the next track continues this one without a gap (same rate and
    /// channels): at the end the converter is not flushed but handed on.
    pub fn set_continues(&mut self, on: bool) {
        self.continues = on;
    }

    /// Whether [`DsdDecoder::adopt`] can take `prev`'s converter.
    pub fn can_continue(&self, prev: &DsdDecoder) -> bool {
        self.reader.info().rate == prev.reader.info().rate
            && self.reader.info().channels == prev.reader.info().channels
            && self.pcm_rate == prev.pcm_rate
    }

    /// Continue from `prev` (which ended with [`DsdDecoder::set_continues`]): take its
    /// converter, so the DSD → PCM filters run on across the track change instead of
    /// starting from silence. Returns `false` (nothing changed) when the formats differ.
    pub fn adopt(&mut self, prev: &mut DsdDecoder) -> bool {
        if !prev.continues || !self.can_continue(prev) || self.pushed != 0 {
            return false;
        }
        std::mem::swap(&mut self.converter, &mut prev.converter);
        self.out_base = self.converter.frames().1;
        // What the converter still holds of the previous track plays first.
        self.pushed = 0;
        true
    }

    /// Play only DSD frames `start..end` of the file (see [`DsdReader::set_range`]).
    pub fn set_range(&mut self, start: u64, end: Option<u64>) {
        self.reader.set_range(start, end);
        self.converter.reset();
        self.pushed = 0;
        self.finished = false;
    }

    /// Gain applied in the DSD → PCM conversion (e.g. +6 dB for SACD's 50 % modulation).
    pub fn with_pcm_gain_db(mut self, db: f64) -> Self {
        self.converter = self.converter.with_gain_db(db);
        self
    }

    /// The DSD format (codec `DSF`/`DFF`, 1 bit, DSD rate).
    pub fn info(&self) -> TrackInfo {
        self.reader.info().track_info()
    }
    /// Header details.
    pub fn dsd_info(&self) -> &DsdInfo {
        self.reader.info()
    }
    /// The PCM rate [`DsdDecoder::next_chunk`] delivers (352.8 or 384 kHz).
    pub fn pcm_rate(&self) -> u32 {
        self.pcm_rate
    }
    /// What the pipeline should be built for when playing through PCM.
    pub fn pcm_input_format(&self) -> InputFormat {
        InputFormat {
            rate: self.pcm_rate,
            bits: 32,
            channels: self.reader.info().channels,
            codec: self.reader.info().codec_label().into(),
        }
    }
    /// Position of the next PCM frame to be delivered, seconds.
    pub fn position_seconds(&self) -> f64 {
        let out = self.converter.frames().1.saturating_sub(self.out_base);
        let base = self.reader.position_frames().saturating_sub(self.pushed);
        base as f64 / self.reader.info().rate as f64 + out as f64 / self.pcm_rate as f64
    }
    /// Duration.
    pub fn duration(&self) -> Option<std::time::Duration> {
        Some(std::time::Duration::from_secs_f64(
            self.reader.info().duration(),
        ))
    }

    /// Next chunk of interleaved PCM at [`DsdDecoder::pcm_rate`]; `Ok(None)` at the end.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f64>>, DsdDecodeError> {
        if self.finished {
            return Ok(None);
        }
        let mut out = Vec::new();
        match self.reader.read_frames(CHUNK_FRAMES)? {
            Some(planar) => {
                let refs: Vec<&[u8]> = planar.iter().map(|p| p.as_slice()).collect();
                self.pushed += planar[0].len() as u64 * 8;
                self.converter.push(&refs, &mut out);
            }
            None => {
                self.finished = true;
                if self.continues {
                    // The next track takes the converter over, tail and all.
                    return Ok(None);
                }
                self.converter.flush(&mut out);
                self.pushed = 0;
            }
        }
        Ok(Some(out))
    }

    /// Next chunk of raw planar DSD (MSB-first bytes per channel) for pass-through.
    pub fn next_dsd_chunk(&mut self) -> Result<Option<Vec<Vec<u8>>>, DsdDecodeError> {
        if self.finished {
            return Ok(None);
        }
        match self.reader.read_frames(CHUNK_FRAMES)? {
            Some(p) => Ok(Some(p)),
            None => {
                self.finished = true;
                Ok(None)
            }
        }
    }

    /// Seek; returns where the stream landed (a byte boundary: 8 frames).
    pub fn seek(&mut self, seconds: f64) -> Result<f64, DsdDecodeError> {
        let frame = (seconds.max(0.0) * self.reader.info().rate as f64) as u64;
        let got = self.reader.seek_frames(frame)?;
        self.converter.reset();
        self.pushed = 0;
        self.finished = false;
        Ok(got as f64 / self.reader.info().rate as f64)
    }
}

// ----- writers --------------------------------------------------------------------------

/// Write a DSF file from planar MSB-first bytes (equal lengths). `lsb_first` selects the
/// standard 1-bit (LSB-first) layout; `false` writes the 8-bit (MSB-first) variant.
pub fn write_dsf(
    path: &Path,
    rate: u32,
    planar: &[Vec<u8>],
    lsb_first: bool,
    tags: &Tags,
) -> io::Result<()> {
    let ch = planar.len();
    assert!(ch > 0 && planar.iter().all(|p| p.len() == planar[0].len()));
    let bytes = planar[0].len();
    let block = 4096usize;
    let blocks = bytes.div_ceil(block).max(1);
    let data_len = blocks * block * ch;
    let id3 = build_id3(tags);
    let total = 28 + 52 + 12 + data_len + id3.len();
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"DSD ")?;
    f.write_all(&28u64.to_le_bytes())?;
    f.write_all(&(total as u64).to_le_bytes())?;
    f.write_all(
        &(if id3.is_empty() {
            0u64
        } else {
            (28 + 52 + 12 + data_len) as u64
        })
        .to_le_bytes(),
    )?;
    f.write_all(b"fmt ")?;
    f.write_all(&52u64.to_le_bytes())?;
    f.write_all(&1u32.to_le_bytes())?; // version
    f.write_all(&0u32.to_le_bytes())?; // DSD raw
    f.write_all(
        &(match ch {
            1 => 1u32,
            2 => 2,
            3 => 3,
            4 => 5,
            5 => 6,
            _ => 7,
        })
        .to_le_bytes(),
    )?;
    f.write_all(&(ch as u32).to_le_bytes())?;
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(&(if lsb_first { 1u32 } else { 8 }).to_le_bytes())?;
    f.write_all(&(bytes as u64 * 8).to_le_bytes())?;
    f.write_all(&(block as u32).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&((12 + data_len) as u64).to_le_bytes())?;
    let mut buf = vec![0u8; block];
    for b in 0..blocks {
        for p in planar {
            buf.fill(0x69); // DSD silence in the padding
            let start = b * block;
            let end = (start + block).min(bytes);
            if start < end {
                for (dst, src) in buf.iter_mut().zip(&p[start..end]) {
                    *dst = if lsb_first { src.reverse_bits() } else { *src };
                }
            }
            f.write_all(&buf)?;
        }
    }
    f.write_all(&id3)?;
    f.flush()
}

fn build_id3(tags: &Tags) -> Vec<u8> {
    let mut frames = Vec::new();
    for (id, v) in [
        (b"TIT2", &tags.title),
        (b"TPE1", &tags.artist),
        (b"TALB", &tags.album),
    ] {
        if let Some(t) = v {
            let body: Vec<u8> = std::iter::once(3u8).chain(t.bytes()).collect();
            frames.extend_from_slice(id);
            frames.extend_from_slice(&(body.len() as u32).to_be_bytes());
            frames.extend_from_slice(&[0, 0]);
            frames.extend_from_slice(&body);
        }
    }
    if frames.is_empty() {
        return Vec::new();
    }
    let n = frames.len() as u32;
    let mut out = b"ID3\x03\x00\x00".to_vec();
    out.extend_from_slice(&[
        (n >> 21) as u8 & 0x7F,
        (n >> 14) as u8 & 0x7F,
        (n >> 7) as u8 & 0x7F,
        n as u8 & 0x7F,
    ]);
    out.extend(frames);
    out
}

/// Write a DSDIFF file from planar MSB-first bytes (equal lengths).
pub fn write_dff(path: &Path, rate: u32, planar: &[Vec<u8>], tags: &Tags) -> io::Result<()> {
    let ch = planar.len();
    assert!(ch > 0 && planar.iter().all(|p| p.len() == planar[0].len()));
    let bytes = planar[0].len();
    let mut data = Vec::with_capacity(bytes * ch);
    for i in 0..bytes {
        for p in planar {
            data.push(p[i]);
        }
    }
    write_dff_form(path, rate, ch, false, tags, |_| dff_chunk(b"DSD ", &data))
}

/// Write a DST-compressed DSDIFF file from coded DST frames (see [`super::dst`]), with a
/// `DSTI` index when `index` is set.
pub fn write_dff_dst(
    path: &Path,
    rate: u32,
    channels: usize,
    frames: &[Vec<u8>],
    index: bool,
    tags: &Tags,
) -> io::Result<()> {
    write_dff_form(path, rate, channels, true, tags, |at| {
        // `at`: file offset where this chunk starts.
        let mut body = dff_chunk(b"FRTE", &{
            let mut f = (frames.len() as u32).to_be_bytes().to_vec();
            f.extend_from_slice(&(super::dst::FRAMES_PER_SECOND as u16).to_be_bytes());
            f
        });
        let mut entries = Vec::new();
        for f in frames {
            entries.push((at + 12 + body.len() as u64 + 12, f.len() as u32));
            body.extend(dff_chunk(b"DSTF", f));
        }
        let mut out = dff_chunk(b"DST ", &body);
        if index {
            let mut dsti = Vec::with_capacity(entries.len() * 12);
            for (o, l) in entries {
                dsti.extend_from_slice(&o.to_be_bytes());
                dsti.extend_from_slice(&l.to_be_bytes());
            }
            out.extend(dff_chunk(b"DSTI", &dsti));
        }
        out
    })
}

fn dff_chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = id.to_vec();
    v.extend_from_slice(&(body.len() as u64).to_be_bytes());
    v.extend_from_slice(body);
    if body.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// The DSDIFF form around the sound data `sound(offset)` returns.
fn write_dff_form(
    path: &Path,
    rate: u32,
    ch: usize,
    dst: bool,
    tags: &Tags,
    sound: impl FnOnce(u64) -> Vec<u8>,
) -> io::Result<()> {
    let mut chnl = (ch as u16).to_be_bytes().to_vec();
    for i in 0..ch {
        let id: [u8; 4] = match (ch, i) {
            (1, _) => *b"C   ",
            (2, 0) => *b"SLFT",
            (2, _) => *b"SRGT",
            _ => format!("C{:03}", i + 1).as_bytes().try_into().unwrap(),
        };
        chnl.extend_from_slice(&id);
    }
    let mut cmpr = if dst {
        b"DST ".to_vec()
    } else {
        b"DSD ".to_vec()
    };
    let name: &[u8] = if dst {
        b"DST Encoded"
    } else {
        b"not compressed"
    };
    cmpr.push(name.len() as u8);
    cmpr.extend_from_slice(name);
    if cmpr.len() % 2 == 1 {
        cmpr.push(0);
    }
    let mut prop_body = b"SND ".to_vec();
    prop_body.extend(dff_chunk(b"FS  ", &rate.to_be_bytes()));
    prop_body.extend(dff_chunk(b"CHNL", &chnl));
    prop_body.extend(dff_chunk(b"CMPR", &cmpr));
    let mut diin = Vec::new();
    for (id, v) in [(b"DIAR", &tags.artist), (b"DITI", &tags.title)] {
        if let Some(t) = v {
            let mut body = (t.len() as u32).to_be_bytes().to_vec();
            body.extend_from_slice(t.as_bytes());
            diin.extend(dff_chunk(id, &body));
        }
    }
    let mut form = b"DSD ".to_vec();
    form.extend(dff_chunk(b"FVER", &0x0105_0000u32.to_be_bytes()));
    form.extend(dff_chunk(b"PROP", &prop_body));
    // The FRM8 header is 12 bytes before the form body.
    let at = 12 + form.len() as u64;
    form.extend(sound(at));
    if !diin.is_empty() {
        form.extend(dff_chunk(b"DIIN", &diin));
    }
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&dff_chunk(b"FRM8", &form))?;
    f.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cadenza_dsp::dsd::{DsdModulator, SigmaDelta};
    use std::f64::consts::PI;

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("cadenza-dsd-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn pattern(ch: usize, bytes: usize) -> Vec<Vec<u8>> {
        (0..ch)
            .map(|c| {
                (0..bytes)
                    .map(|i| ((i * 7 + c * 131 + 3) % 251) as u8)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn dsf_round_trip_both_bit_orders_with_tags_and_seek() {
        let d = dir("dsf");
        let planar = pattern(2, 10_000); // not a block multiple: exercises padding
        let tags = Tags {
            title: Some("Tone".into()),
            artist: Some("Cadenza".into()),
            album: Some("Tests".into()),
        };
        for lsb in [true, false] {
            let path = d.join(format!("t{lsb}.dsf"));
            write_dsf(&path, 2_822_400, &planar, lsb, &tags).unwrap();
            let mut r = DsdReader::new(std::fs::File::open(&path).unwrap()).unwrap();
            let info = r.info().clone();
            assert_eq!(info.container, DsdContainer::Dsf);
            assert_eq!(
                (info.rate, info.channels, info.frames),
                (2_822_400, 2, 80_000)
            );
            assert_eq!(info.tags, tags);
            assert_eq!(info.track_info().label(), "DSF 1/2822.4");
            let mut got = vec![Vec::new(); 2];
            while let Some(p) = r.read_frames(5000 * 8).unwrap() {
                for (g, x) in got.iter_mut().zip(p) {
                    g.extend(x);
                }
            }
            assert_eq!(got, planar, "lsb_first={lsb}");
            // Seek into the second block and read the rest.
            let f = r.seek_frames(4096 * 8 + 803).unwrap();
            assert_eq!(f, 4096 * 8 + 800); // byte boundary, inside the second block
            let p = r.read_frames(usize::MAX).unwrap().unwrap();
            assert_eq!(p[1], planar[1][4196..]);
            assert!(r.read_frames(8).unwrap().is_none());
        }
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn dff_round_trip_with_tags_and_seek() {
        let d = dir("dff");
        let planar = pattern(2, 7001);
        let tags = Tags {
            title: Some("Tone".into()),
            artist: Some("Cadenza".into()),
            album: None,
        };
        let path = d.join("t.dff");
        write_dff(&path, 5_644_800, &planar, &tags).unwrap();
        let mut r = DsdReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        let info = r.info().clone();
        assert_eq!(info.container, DsdContainer::Dff);
        assert_eq!(
            (info.rate, info.channels, info.frames),
            (5_644_800, 2, 7001 * 8)
        );
        assert_eq!(info.tags, tags);
        let mut got = vec![Vec::new(); 2];
        while let Some(p) = r.read_frames(1000 * 8).unwrap() {
            for (g, x) in got.iter_mut().zip(p) {
                g.extend(x);
            }
        }
        assert_eq!(got, planar);
        assert_eq!(r.seek_frames(100 * 8 + 3).unwrap(), 800);
        let p = r.read_frames(16).unwrap().unwrap();
        assert_eq!(p[0], planar[0][100..102]);
        let _ = std::fs::remove_dir_all(d);
    }

    /// `frames` DST frames of a 2-channel tone: the frames, coded with a mix of options
    /// (one stored plain unless `ffmpeg_safe`), and the original planar DSD.
    fn dst_stream(frames: usize, ffmpeg_safe: bool) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        use crate::decode::dst::{bits_per_frame, encode};
        let rate = 2_822_400u32;
        let fb = bits_per_frame(rate) / 8;
        let all = encode::test_dsd(2, fb * frames, 997.0, rate as f64);
        let mut coded = Vec::new();
        for (i, f) in all.chunks(fb * 2).enumerate() {
            let o = match (i % 5, ffmpeg_safe) {
                (1, _) => encode::Options {
                    coded_tables: false,
                    half_prob: true,
                    ..Default::default()
                },
                (2, _) => encode::Options {
                    per_channel_tables: true,
                    order: 100,
                    ..Default::default()
                },
                (3, false) => encode::Options {
                    filter_segments: vec![1600],
                    ptable_segments: Some(vec![800, 800]),
                    ..Default::default()
                },
                _ => encode::Options::default(),
            };
            // FFmpeg 6.x lays plain frames out wrongly (fixed in later releases).
            coded.push(if i == 4 && !ffmpeg_safe {
                let mut plain = vec![0u8];
                plain.extend_from_slice(f);
                plain
            } else {
                encode::encode_frame(f, 2, &o)
            });
        }
        let mut planar = vec![Vec::new(), Vec::new()];
        for (i, b) in all.iter().enumerate() {
            planar[i % 2].push(*b);
        }
        (coded, planar)
    }

    fn read_all(r: &mut DsdReader<std::fs::File>) -> Vec<Vec<u8>> {
        let mut out = vec![Vec::new(); r.info().channels as usize];
        while let Some(p) = r.read_frames(30_000).unwrap() {
            for (o, c) in out.iter_mut().zip(p) {
                o.extend(c);
            }
        }
        out
    }

    #[test]
    fn dst_dff_decodes_bit_exact_and_seeks_with_and_without_index() {
        let d = dir("dst");
        let (coded, planar) = dst_stream(12, false);
        let tags = Tags {
            title: Some("DST".into()),
            artist: Some("Philips".into()),
            ..Default::default()
        };
        for index in [false, true] {
            let path = d.join(format!("t{index}.dff"));
            write_dff_dst(&path, 2_822_400, 2, &coded, index, &tags).unwrap();
            let mut r = DsdReader::new(std::fs::File::open(&path).unwrap()).unwrap();
            assert!(r.info().is_dst());
            assert_eq!(r.info().codec_label(), "DFF DST");
            assert_eq!(r.info().frames, 12 * 37_632);
            assert_eq!(r.info().tags.title.as_deref(), Some("DST"));
            assert_eq!(read_all(&mut r), planar, "index {index}");
            // Seek into frame 9 from a fresh reader.
            let mut r = DsdReader::new(std::fs::File::open(&path).unwrap()).unwrap();
            let at = 9 * 37_632 + 800;
            assert_eq!(r.seek_frames(at).unwrap(), at);
            let got = read_all(&mut r);
            assert_eq!(got[0], planar[0][at as usize / 8..]);
            assert_eq!(got[1], planar[1][at as usize / 8..]);
            // With the index only the first frame is scanned for; without, all up to 9.
            let found = r.dst.as_ref().unwrap().found.len();
            if index {
                assert_eq!(found, 1);
            } else {
                assert!(found >= 10);
            }
        }
        // The PCM path gives what the same DSD gives uncompressed.
        let plain = d.join("plain.dff");
        write_dff(&plain, 2_822_400, &planar, &tags).unwrap();
        let pcm = |path: &Path| {
            let mut dec =
                DsdDecoder::from_reader(Box::new(std::fs::File::open(path).unwrap())).unwrap();
            let mut all = Vec::new();
            while let Some(c) = dec.next_chunk().unwrap() {
                all.extend(c);
            }
            all
        };
        assert_eq!(pcm(&d.join("ttrue.dff")), pcm(&plain));
        let _ = std::fs::remove_dir_all(d);
    }

    /// FFmpeg's DST decoder (written independently) must decode our frames to exactly
    /// what it decodes from the same DSD stored plain. Skipped when ffmpeg is missing.
    #[test]
    fn dst_frames_agree_with_ffmpeg() {
        let ok = std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success());
        if !ok {
            eprintln!("ffmpeg not installed: DST cross-check skipped");
            return;
        }
        let d = dir("dst-ffmpeg");
        // FFmpeg reads one segment per channel only, and before 7.0 mishandles frames
        // stored plain: those two are covered by the round-trip test instead.
        let (coded, planar) = dst_stream(30, true);
        let dst = d.join("dst.dff");
        let plain = d.join("plain.dff");
        write_dff_dst(&dst, 2_822_400, 2, &coded, false, &Tags::default()).unwrap();
        write_dff(&plain, 2_822_400, &planar, &Tags::default()).unwrap();
        let decode = |p: &Path| {
            let o = std::process::Command::new("ffmpeg")
                .args(["-v", "error", "-i"])
                .arg(p)
                .args(["-f", "f32le", "-"])
                .output()
                .unwrap();
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
            o.stdout
        };
        let a = decode(&dst);
        let b = decode(&plain);
        assert!(!a.is_empty());
        assert!(a == b, "ffmpeg decodes our DST frames differently");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn dst_and_garbage_are_refused() {
        let d = dir("bad");
        let path = d.join("x.dff");
        std::fs::write(&path, b"RIFF....WAVE").unwrap();
        assert!(matches!(
            DsdReader::new(std::fs::File::open(&path).unwrap()),
            Err(DsdDecodeError::Format(_))
        ));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn decodes_a_known_tone_to_pcm_within_0_01_db() {
        let d = dir("tone");
        let rate = 2_822_400u32;
        let hz = 1000.0;
        let amp = 0.5;
        let n = rate as usize / 2; // 0.5 s
        let mut planar = Vec::new();
        for c in 0..2 {
            let mut m = SigmaDelta::new(DsdModulator::Asdm7, rate);
            let x: Vec<f64> = (0..n)
                .map(|i| {
                    amp * (2.0 * PI * hz * i as f64 / rate as f64).sin()
                        * if c == 0 { 1.0 } else { -1.0 }
                })
                .collect();
            let mut bits = Vec::new();
            m.process(&x, &mut bits);
            planar.push(bits);
        }
        let path = d.join("tone.dsf");
        write_dsf(&path, rate, &planar, true, &Tags::default()).unwrap();
        let mut dec = DsdDecoder::open(&Source::Path { path: path.clone() }).unwrap();
        assert_eq!(dec.pcm_rate(), 352_800);
        assert_eq!(dec.info().codec, "DSF");
        assert_eq!(dec.pcm_input_format().rate, 352_800);
        let mut pcm = Vec::new();
        while let Some(c) = dec.next_chunk().unwrap() {
            pcm.extend(c);
        }
        assert_eq!(pcm.len(), n / 8 * 2);
        let frames = pcm.len() / 2;
        let mid = &pcm[frames / 2..frames * 3 / 2];
        for ch in 0..2 {
            let v: Vec<f64> = mid.iter().skip(ch).step_by(2).copied().collect();
            let rms = (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
            let db = 20.0 * (rms * 2f64.sqrt() / amp).log10();
            assert!(db.abs() < 0.01, "channel {ch} level error {db:.4} dB");
        }
        // Channels are anti-phase, as written.
        let corr: f64 = mid.chunks(2).map(|f| f[0] * f[1]).sum::<f64>();
        assert!(corr < 0.0);
        // Seek lands on a byte boundary and decoding resumes.
        let got = dec.seek(0.25).unwrap();
        assert!((got - 0.25).abs() < 1e-5, "seeked to {got}");
        let c = dec.next_chunk().unwrap().unwrap();
        assert!(!c.is_empty() || dec.next_chunk().unwrap().is_some());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn pass_through_chunks_are_the_raw_bits() {
        let d = dir("raw");
        let planar = pattern(1, 4096 * 2);
        let path = d.join("raw.dff");
        write_dff(&path, 2_822_400, &planar, &Tags::default()).unwrap();
        let mut dec = DsdDecoder::open(&Source::Path { path }).unwrap();
        let mut got = Vec::new();
        while let Some(p) = dec.next_dsd_chunk().unwrap() {
            got.extend(p.into_iter().next().unwrap());
        }
        assert_eq!(got, planar[0]);
        assert!(is_dsd_extension("DSF") && is_dsd_extension("dff") && !is_dsd_extension("flac"));
        let _ = std::fs::remove_dir_all(d);
    }
}
