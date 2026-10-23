//! SACD disc images (`.iso`): the Scarlet Book layout of a Super Audio CD.
//!
//! - **Master TOC** at sector 510 (`SACDMTOC`): where the stereo and multichannel areas
//!   are; album / disc text follows it (`SACDText`).
//! - **Area TOC** (`TWOCHTOC` / `MULCHTOC`): channel count, plain or DST, the track list
//!   (`SACDTRL1`: start sector and length; `SACDTRL2`: start time and duration in 1/75 s
//!   frames) and track text (`SACDTTxt`: title, performer …).
//! - **Audio sectors** (2048 bytes): a header byte, up to 7 packet descriptors and the
//!   frame descriptors (time code, and for DST the number of sectors the frame spans),
//!   then the packets. A 1/75 s frame – DST-coded, or plain DSD bytes interleaved by
//!   channel – is split over packets in consecutive sectors.
//!
//! Tracks are cut by the frames' time codes, not by sectors, so consecutive tracks join
//! without a gap or an overlap. A track is addressed as `album.iso#3` (stereo area) or
//! `album.iso#m3` (multichannel area).

use super::dst;
use std::io::{self, Read, Seek, SeekFrom};

/// Bytes per sector.
pub const SECTOR: u64 = 2048;
/// Sector number of the master TOC (`SACDMTOC`).
const MASTER_TOC: u64 = 510;
/// DSD rate of every SACD.
pub const RATE: u32 = 2_822_400;
/// Bytes per channel in one plain frame.
const PLAIN_FRAME_BYTES: usize = 4704;

/// Errors from the disc structures.
#[derive(Debug, thiserror::Error)]
pub enum SacdError {
    /// I/O.
    #[error("{0}")]
    Io(#[from] io::Error),
    /// Not an SACD image, or a damaged one.
    #[error("not an SACD image: {0}")]
    Format(String),
    /// No such track / area.
    #[error("{0}")]
    NoTrack(String),
}

/// A format error with `m`.
fn bad(m: impl Into<String>) -> SacdError {
    SacdError::Format(m.into())
}

/// Big-endian `u16` at byte `at`.
fn be16(b: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([b[at], b[at + 1]])
}
/// Big-endian `u32` at byte `at`.
fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Frames (1/75 s) from a minutes / seconds / frames time code.
fn tc_frames(b: &[u8]) -> u32 {
    b[0] as u32 * 60 * 75 + b[1] as u32 * 75 + b[2] as u32
}

/// Text in the disc's character set: ISO 646 / 8859-1 as Latin-1; the Asian sets are not
/// converted (shown as UTF-8 where they happen to be ASCII).
fn text(bytes: &[u8], charset: u8) -> String {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    let b = &bytes[..end];
    let s: String = match charset & 7 {
        1 | 2 | 7 => b.iter().map(|c| *c as char).collect(),
        _ => String::from_utf8_lossy(b).into_owned(),
    };
    s.trim().to_string()
}

/// Read `n` whole sectors starting at sector `lsn`.
fn read_sectors<R: Read + Seek>(r: &mut R, lsn: u64, n: usize) -> io::Result<Vec<u8>> {
    r.seek(SeekFrom::Start(lsn * SECTOR))?;
    let mut buf = vec![0u8; n * SECTOR as usize];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// One track.
#[derive(Clone, Debug, PartialEq)]
pub struct SacdTrack {
    /// 1-based number in its area.
    pub number: u32,
    /// Title.
    pub title: Option<String>,
    /// Performer.
    pub performer: Option<String>,
    /// First sector.
    pub start_lsn: u32,
    /// Sectors.
    pub length_lsn: u32,
    /// Start time on the disc, frames.
    pub start: u32,
    /// Duration, frames.
    pub frames: u32,
}

/// The stereo or the multichannel area.
#[derive(Clone, Debug, PartialEq)]
pub struct SacdArea {
    /// Multichannel area.
    pub multichannel: bool,
    /// Channels (2, 5 or 6).
    pub channels: u32,
    /// DST-coded (otherwise plain DSD).
    pub dst: bool,
    /// First and last audio sector of the area.
    pub start_lsn: u32,
    /// Last audio sector.
    pub end_lsn: u32,
    /// Tracks.
    pub tracks: Vec<SacdTrack>,
}

/// What the disc holds.
#[derive(Clone, Debug, PartialEq)]
pub struct SacdDisc {
    /// Album / disc title.
    pub title: Option<String>,
    /// Album / disc artist.
    pub artist: Option<String>,
    /// Areas found (stereo first).
    pub areas: Vec<SacdArea>,
}

impl SacdDisc {
    /// The stereo area (or, with `multichannel`, the multichannel one).
    pub fn area(&self, multichannel: bool) -> Option<&SacdArea> {
        self.areas.iter().find(|a| a.multichannel == multichannel)
    }
}

/// Read the master and area TOCs.
pub fn read_disc<R: Read + Seek>(r: &mut R) -> Result<SacdDisc, SacdError> {
    let m = read_sectors(r, MASTER_TOC, 1).map_err(|_| bad("too short for a master TOC"))?;
    if &m[0..8] != b"SACDMTOC" {
        return Err(bad("no SACDMTOC at sector 510"));
    }
    let charset = m[138];
    let mut disc = SacdDisc {
        title: None,
        artist: None,
        areas: Vec::new(),
    };
    // Album / disc text of the first language.
    if let Ok(t) = read_sectors(r, MASTER_TOC + 1, 1) {
        if &t[0..8] == b"SACDText" {
            let field = |pos: usize| {
                let off = be16(&t, pos) as usize;
                (off > 0 && off < t.len())
                    .then(|| text(&t[off..], charset))
                    .filter(|s| !s.is_empty())
            };
            disc.title = field(16).or_else(|| field(32));
            disc.artist = field(18).or_else(|| field(34));
        }
    }
    for (toc_at, size_at) in [(64usize, 84usize), (72, 86)] {
        let lsn = be32(&m, toc_at) as u64;
        let size = be16(&m, size_at) as usize;
        if lsn == 0 || size == 0 {
            continue;
        }
        match read_area(r, lsn, size) {
            Ok(a) => disc.areas.push(a),
            Err(e) => log::warn!("SACD: area TOC at sector {lsn}: {e}"),
        }
    }
    if disc.areas.is_empty() {
        return Err(bad("no audio area"));
    }
    disc.areas.sort_by_key(|a| a.multichannel);
    Ok(disc)
}

/// Parse the area TOC at sector `lsn` (`TWOCHTOC` or `MULCHTOC`): format, channels,
/// track list and the area's sector range.
fn read_area<R: Read + Seek>(r: &mut R, lsn: u64, size: usize) -> Result<SacdArea, SacdError> {
    let d = read_sectors(r, lsn, size.min(96))?;
    let id = &d[0..8];
    if id != b"TWOCHTOC" && id != b"MULCHTOC" {
        return Err(bad("no area TOC"));
    }
    let frame_format = d[21] & 0x0f;
    let channels = d[32] as u32;
    let loudspeaker = d[33] >> 3;
    let track_count = d[69] as usize;
    let charset = d[90];
    let mut area = SacdArea {
        multichannel: !(channels == 2 && loudspeaker == 0),
        channels,
        dst: frame_format == 0,
        start_lsn: be32(&d, 72),
        end_lsn: be32(&d, 76),
        tracks: (0..track_count)
            .map(|i| SacdTrack {
                number: i as u32 + 1,
                title: None,
                performer: None,
                start_lsn: 0,
                length_lsn: 0,
                start: 0,
                frames: 0,
            })
            .collect(),
    };
    if !matches!(channels, 1..=6) {
        return Err(bad(format!("{channels} channels")));
    }
    let mut seen_text = false;
    for sector in d.chunks_exact(SECTOR as usize).skip(1) {
        match &sector[0..8] {
            b"SACDTRL1" => {
                for (i, t) in area.tracks.iter_mut().enumerate() {
                    t.start_lsn = be32(sector, 8 + 4 * i);
                    t.length_lsn = be32(sector, 8 + 4 * 255 + 4 * i);
                }
            }
            b"SACDTRL2" => {
                for (i, t) in area.tracks.iter_mut().enumerate() {
                    t.start = tc_frames(&sector[8 + 4 * i..]);
                    t.frames = tc_frames(&sector[8 + 4 * 255 + 4 * i..]);
                }
            }
            b"SACDTTxt" if !seen_text => {
                seen_text = true;
                for (i, t) in area.tracks.iter_mut().enumerate() {
                    let pos = be16(sector, 8 + 2 * i) as usize;
                    if pos == 0 || pos + 4 >= sector.len() {
                        continue;
                    }
                    let count = sector[pos] as usize;
                    let mut p = pos + 4;
                    for _ in 0..count {
                        if p + 2 >= sector.len() {
                            break;
                        }
                        let kind = sector[p];
                        p += 2;
                        let s = text(&sector[p..], charset);
                        match kind {
                            1 if !s.is_empty() => t.title = Some(s),
                            2 if !s.is_empty() => t.performer = Some(s),
                            _ => {}
                        }
                        while p < sector.len() && sector[p] != 0 {
                            p += 1;
                        }
                        while p < sector.len() && sector[p] == 0 {
                            p += 1;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if area
        .tracks
        .iter()
        .any(|t| t.frames == 0 || t.start_lsn == 0)
    {
        return Err(bad("track list incomplete"));
    }
    Ok(area)
}

/// One frame as stored: its disc time and its bytes (DST-coded or plain).
#[derive(Clone, Debug, PartialEq)]
pub struct RawFrame {
    /// Time code, frames from the start of the disc area.
    pub time: u32,
    /// Bytes.
    pub data: Vec<u8>,
}

/// Reassembles frames from audio sectors.
pub struct FrameScanner {
    /// Whether the area's frames are DST-compressed.
    dst: bool,
    /// Channels in the area.
    channels: usize,
    /// Next sector to read.
    lsn: u64,
    /// Last sector of the area.
    end_lsn: u64,
    /// The frame being assembled from packets.
    current: Option<RawFrame>,
    /// Frames complete and waiting to be taken.
    ready: std::collections::VecDeque<RawFrame>,
}

impl FrameScanner {
    /// Scan from sector `lsn` to `end_lsn` (inclusive).
    pub fn new(area: &SacdArea, lsn: u64) -> Self {
        Self {
            dst: area.dst,
            channels: area.channels as usize,
            lsn,
            end_lsn: area.end_lsn as u64,
            current: None,
            ready: Default::default(),
        }
    }

    /// The sector the next read comes from.
    pub fn lsn(&self) -> u64 {
        self.lsn
    }

    /// Parse one sector into frames.
    pub fn push_sector(&mut self, s: &[u8]) {
        let h = s[0];
        let dst = h & 1 == 1;
        let n_frames = ((h >> 2) & 7) as usize;
        let n_packets = ((h >> 5) & 7) as usize;
        let mut p = 1;
        let mut packets = Vec::with_capacity(n_packets);
        for _ in 0..n_packets {
            let b0 = s[p];
            packets.push((
                b0 >> 7 == 1,
                (b0 >> 3) & 7,
                ((b0 as usize & 7) << 8) | s[p + 1] as usize,
            ));
            p += 2;
        }
        let info_size = if dst { 4 } else { 3 };
        let mut times = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            times.push(tc_frames(&s[p..]));
            p += info_size;
        }
        let mut next_time = times.into_iter();
        for (start, kind, len) in packets {
            if p + len > s.len() {
                break;
            }
            let data = &s[p..p + len];
            p += len;
            if kind != 2 {
                continue; // supplementary data or padding
            }
            if start {
                if let Some(f) = self.current.take() {
                    self.finish(f);
                }
                self.current = Some(RawFrame {
                    time: next_time.next().unwrap_or(0),
                    data: Vec::with_capacity(if self.dst { 8192 } else { 9408 }),
                });
            }
            if let Some(f) = &mut self.current {
                f.data.extend_from_slice(data);
            }
        }
    }

    /// Queue a completed frame (a plain frame only when whole).
    fn finish(&mut self, f: RawFrame) {
        // A plain frame must be whole; a DST frame decodes or is reported later.
        if self.dst || f.data.len() == self.channels * PLAIN_FRAME_BYTES {
            self.ready.push_back(f);
        }
    }

    /// The next whole frame, reading sectors as needed; `None` past the area's end.
    pub fn next<R: Read + Seek>(&mut self, r: &mut R) -> Result<Option<RawFrame>, SacdError> {
        while self.ready.is_empty() {
            if self.lsn > self.end_lsn {
                if let Some(f) = self.current.take() {
                    self.finish(f);
                    continue;
                }
                return Ok(None);
            }
            // Read a few sectors at a time (one seek per 16 for an HTTP source).
            let n = (self.end_lsn + 1 - self.lsn).min(16) as usize;
            let buf = read_sectors(r, self.lsn, n)?;
            for s in buf.chunks_exact(SECTOR as usize) {
                self.push_sector(s);
            }
            self.lsn += n as u64;
        }
        Ok(self.ready.pop_front())
    }
}

/// Plays one track: decoded DSD frames (byte-interleaved, MSB first), cut by time code.
pub struct TrackReader {
    /// The area the track belongs to.
    area: SacdArea,
    /// The track's TOC entry (start, length).
    track: SacdTrack,
    /// Reassembles frames from the area's sectors.
    scanner: FrameScanner,
    /// DST decoder, when the area is compressed.
    dst: Option<dst::DstDecoder>,
    /// Track-relative index of the frame the scanner delivers next (`None`: unknown).
    next: Option<u32>,
}

impl TrackReader {
    /// A reader for track `number` (1-based) of `area`.
    pub fn new(area: &SacdArea, number: u32) -> Result<Self, SacdError> {
        let track = area
            .tracks
            .get((number as usize).wrapping_sub(1))
            .cloned()
            .ok_or_else(|| SacdError::NoTrack(format!("no track {number}")))?;
        let dst = if area.dst {
            Some(dst::DstDecoder::new(area.channels, RATE).map_err(|e| bad(e.to_string()))?)
        } else {
            None
        };
        let mut r = Self {
            area: area.clone(),
            scanner: FrameScanner::new(area, 0),
            track,
            dst,
            next: None,
        };
        r.restart(0);
        Ok(r)
    }

    /// The track.
    pub fn track(&self) -> &SacdTrack {
        &self.track
    }

    /// Bytes of a decoded frame.
    pub fn frame_bytes(&self) -> usize {
        self.area.channels as usize * PLAIN_FRAME_BYTES
    }

    /// Restart scanning a little before where frame `n` should be.
    fn restart(&mut self, n: u32) {
        let t = &self.track;
        let est = t.start_lsn as u64 + t.length_lsn as u64 * n as u64 / t.frames.max(1) as u64;
        // A frame can begin up to 31 sectors before its successor's start.
        let lsn = est
            .saturating_sub(if n == 0 { 4 } else { 40 })
            .max(self.area.start_lsn as u64);
        self.scanner = FrameScanner::new(&self.area, lsn);
        self.next = None;
    }

    /// Decode frame `n` of the track into `out`; `false` past the end.
    pub fn read_frame<R: Read + Seek>(
        &mut self,
        r: &mut R,
        n: u32,
        out: &mut [u8],
    ) -> Result<bool, SacdError> {
        if n >= self.track.frames {
            return Ok(false);
        }
        if self.next.is_some_and(|x| x > n) || self.next.is_some_and(|x| x + 400 < n) {
            self.restart(n);
        }
        let mut backoffs = 0;
        loop {
            let Some(f) = self.scanner.next(r)? else {
                return Ok(false);
            };
            let rel = f.time as i64 - self.track.start as i64;
            if rel > n as i64 {
                // Overshot (the estimate was too far on): go further back.
                backoffs += 1;
                if backoffs > 8 {
                    return Err(bad(format!(
                        "frame {n} of track {} not found",
                        self.track.number
                    )));
                }
                let lsn = self.scanner.lsn().saturating_sub(64 * backoffs);
                self.scanner = FrameScanner::new(&self.area, lsn.max(self.area.start_lsn as u64));
                continue;
            }
            self.next = Some((rel + 1).max(0) as u32);
            if rel < n as i64 {
                continue;
            }
            match &mut self.dst {
                Some(d) => {
                    if let Err(e) = d.decode(&f.data, out) {
                        log::warn!("SACD track {} frame {n}: {e}", self.track.number);
                        out.fill(0x69);
                    }
                }
                None => out.copy_from_slice(&f.data[..out.len()]),
            }
            return Ok(true);
        }
    }
}

/// `album.iso#3` / `album.iso#m3`: the image path and (multichannel, track).
pub fn parse_fragment(s: &str) -> Option<(&str, bool, u32)> {
    let (base, frag) = s.rsplit_once('#')?;
    if !base.to_ascii_lowercase().ends_with(".iso") {
        return None;
    }
    let (mc, num) = match frag.strip_prefix('m') {
        Some(n) => (true, n),
        None => (false, frag),
    };
    Some((base, mc, num.parse().ok().filter(|n| *n > 0)?))
}

#[cfg(test)]
pub(crate) mod tests {
    //! A synthetic disc image: master TOC, one area TOC with track list and text, and
    //! audio sectors packed the way an SACD is.

    use super::*;
    use crate::decode::dst::encode;

    fn tc(f: u32) -> [u8; 3] {
        [(f / 75 / 60) as u8, (f / 75 % 60) as u8, (f % 75) as u8]
    }

    /// Pack frames into audio sectors; returns the sectors and each frame's first sector.
    fn pack(frames: &[Vec<u8>], first_tc: u32, dst: bool, channels: usize) -> (Vec<u8>, Vec<u32>) {
        let mut sectors: Vec<u8> = Vec::new();
        let mut starts = vec![0u32; frames.len()];
        let mut fi = 0usize; // frame being written
        let mut off = 0usize; // bytes of it written
        let info = if dst { 4 } else { 3 };
        while fi < frames.len() {
            let sector_no = sectors.len() / 2048;
            let mut packets: Vec<(bool, usize, usize, usize)> = Vec::new(); // start, frame, from, len
            let mut infos: Vec<(u32, u8)> = Vec::new();
            let mut used = 1usize;
            while fi < frames.len() && packets.len() < 7 {
                let start = off == 0;
                let overhead = 2 + if start { info } else { 0 };
                if used + overhead >= 2048 || (start && infos.len() >= 7) {
                    break;
                }
                let room = (2048 - used - overhead).min(2045);
                let len = (frames[fi].len() - off).min(room);
                if len == 0 {
                    break;
                }
                if start {
                    starts[fi] = sector_no as u32;
                    // Sectors this frame will span (approximate is fine for the reader).
                    let span = (frames[fi].len() / 2000 + 1).min(31) as u8;
                    let chan = match channels {
                        6 => 0b10,
                        5 => 0b01,
                        _ => 0,
                    };
                    infos.push((first_tc + fi as u32, (span << 2) | chan));
                }
                packets.push((start, fi, off, len));
                used += overhead + len;
                off += len;
                if off == frames[fi].len() {
                    fi += 1;
                    off = 0;
                }
            }
            let mut s = vec![0u8; 2048];
            s[0] = ((packets.len() as u8) << 5) | ((infos.len() as u8) << 2) | dst as u8;
            let mut p = 1;
            for (start, _, _, len) in &packets {
                s[p] = ((*start as u8) << 7) | (2 << 3) | ((len >> 8) as u8 & 7);
                s[p + 1] = *len as u8;
                p += 2;
            }
            for (t, b) in &infos {
                s[p..p + 3].copy_from_slice(&tc(*t));
                if dst {
                    s[p + 3] = *b;
                }
                p += info;
            }
            for (_, f, from, len) in &packets {
                s[p..p + len].copy_from_slice(&frames[*f][*from..from + len]);
                p += len;
            }
            sectors.extend(s);
        }
        (sectors, starts)
    }

    pub(crate) struct Image {
        pub bytes: Vec<u8>,
        /// Per track: the original DSD, byte-interleaved.
        pub tracks: Vec<Vec<u8>>,
    }

    /// An image with `track_frames` tracks (frame counts), stereo, DST or plain.
    pub(crate) fn image(track_frames: &[u32], dst: bool) -> Image {
        let ch = 2usize;
        let fb = PLAIN_FRAME_BYTES * ch;
        let total: u32 = track_frames.iter().sum();
        let all = encode::test_dsd(ch, PLAIN_FRAME_BYTES * total as usize, 441.0, RATE as f64);
        let frames: Vec<Vec<u8>> = all
            .chunks(fb)
            .map(|f| {
                if dst {
                    encode::encode_frame(f, ch, &Default::default())
                } else {
                    f.to_vec()
                }
            })
            .collect();
        let audio_start = 600u32;
        let (audio, starts) = pack(&frames, 150, dst, ch);
        let audio_sectors = (audio.len() / 2048) as u32;
        let area_toc = 540u32;
        let mut img = vec![0u8; (audio_start as usize) * 2048];
        // Master TOC + text.
        let m = &mut img[510 * 2048..511 * 2048];
        m[0..8].copy_from_slice(b"SACDMTOC");
        m[8] = 1;
        m[9] = 20;
        m[64..68].copy_from_slice(&area_toc.to_be_bytes());
        m[84..86].copy_from_slice(&5u16.to_be_bytes());
        m[138] = 2; // ISO 8859-1
        let t = &mut img[511 * 2048..512 * 2048];
        t[0..8].copy_from_slice(b"SACDText");
        t[16..18].copy_from_slice(&64u16.to_be_bytes());
        t[18..20].copy_from_slice(&96u16.to_be_bytes());
        t[64..76].copy_from_slice(b"Test Album\0\0");
        t[96..105].copy_from_slice(b"Caf\xe9 Trio");
        // Area TOC.
        let a0 = area_toc as usize * 2048;
        let a = &mut img[a0..a0 + 5 * 2048];
        a[0..8].copy_from_slice(b"TWOCHTOC");
        a[8] = 1;
        a[9] = 20;
        a[10..12].copy_from_slice(&5u16.to_be_bytes());
        a[20] = 4;
        a[21] = if dst { 0 } else { 2 };
        a[32] = 2;
        a[69] = track_frames.len() as u8;
        a[72..76].copy_from_slice(&audio_start.to_be_bytes());
        a[76..80].copy_from_slice(&(audio_start + audio_sectors - 1).to_be_bytes());
        a[90] = 2;
        let trl1 = &mut a[2048..4096];
        trl1[0..8].copy_from_slice(b"SACDTRL1");
        let trl2_start = 4096;
        let mut first = 0usize;
        for (i, n) in track_frames.iter().enumerate() {
            let s = audio_start + starts[first];
            let end_frame = first + *n as usize;
            let e = if end_frame < frames.len() {
                audio_start + starts[end_frame]
            } else {
                audio_start + audio_sectors
            };
            trl1[8 + 4 * i..12 + 4 * i].copy_from_slice(&s.to_be_bytes());
            trl1[8 + 1020 + 4 * i..12 + 1020 + 4 * i].copy_from_slice(&(e - s).to_be_bytes());
            first = end_frame;
        }
        let trl2 = &mut a[trl2_start..trl2_start + 2048];
        trl2[0..8].copy_from_slice(b"SACDTRL2");
        let mut t0 = 150;
        for (i, n) in track_frames.iter().enumerate() {
            trl2[8 + 4 * i..11 + 4 * i].copy_from_slice(&tc(t0));
            trl2[8 + 1020 + 4 * i..11 + 1020 + 4 * i].copy_from_slice(&tc(*n));
            t0 += n;
        }
        let txt = &mut a[3 * 2048..4 * 2048];
        txt[0..8].copy_from_slice(b"SACDTTxt");
        let mut p = 600usize;
        for i in 0..track_frames.len() {
            txt[8 + 2 * i..10 + 2 * i].copy_from_slice(&(p as u16).to_be_bytes());
            txt[p] = 2;
            p += 4;
            let title = format!("Track {}", i + 1);
            for (kind, s) in [(1u8, title.as_str()), (2, "Quartet")] {
                txt[p] = kind;
                txt[p + 1] = 0x20;
                p += 2;
                txt[p..p + s.len()].copy_from_slice(s.as_bytes());
                p += s.len() + 2;
            }
        }
        img.extend(audio);
        let mut tracks = Vec::new();
        let mut at = 0usize;
        for n in track_frames {
            tracks.push(all[at * fb..(at + *n as usize) * fb].to_vec());
            at += *n as usize;
        }
        Image { bytes: img, tracks }
    }

    #[test]
    fn reads_the_tocs() {
        let img = image(&[5, 7, 3], true);
        let disc = read_disc(&mut io::Cursor::new(&img.bytes)).unwrap();
        assert_eq!(disc.title.as_deref(), Some("Test Album"));
        assert_eq!(disc.artist.as_deref(), Some("Café Trio"));
        let a = disc.area(false).unwrap();
        assert!(a.dst && !a.multichannel && a.channels == 2);
        assert_eq!(a.tracks.len(), 3);
        assert_eq!(a.tracks[1].title.as_deref(), Some("Track 2"));
        assert_eq!(a.tracks[1].performer.as_deref(), Some("Quartet"));
        assert_eq!((a.tracks[1].start, a.tracks[1].frames), (155, 7));
        assert!(disc.area(true).is_none());
        assert!(read_disc(&mut io::Cursor::new(vec![0u8; 600 * 2048])).is_err());
        assert_eq!(parse_fragment("/x/A.ISO#m3"), Some(("/x/A.ISO", true, 3)));
        assert_eq!(parse_fragment("a.iso#2"), Some(("a.iso", false, 2)));
        assert_eq!(parse_fragment("a.iso#0"), None);
        assert_eq!(parse_fragment("a.flac#2"), None);
    }

    #[test]
    fn tracks_decode_bit_exact_cut_by_time_code_and_seek() {
        for dst in [true, false] {
            let img = image(&[5, 7, 3], dst);
            let mut cur = io::Cursor::new(&img.bytes);
            let disc = read_disc(&mut cur).unwrap();
            let area = disc.area(false).unwrap();
            for (i, want) in img.tracks.iter().enumerate() {
                let mut tr = TrackReader::new(area, i as u32 + 1).unwrap();
                let mut got = Vec::new();
                let mut out = vec![0u8; tr.frame_bytes()];
                let mut n = 0;
                while tr.read_frame(&mut cur, n, &mut out).unwrap() {
                    got.extend_from_slice(&out);
                    n += 1;
                }
                assert!(got == *want, "dst {dst} track {}", i + 1);
                // Random access: a later frame, then an earlier one.
                let last = tr.track().frames - 1;
                assert!(tr.read_frame(&mut cur, last, &mut out).unwrap());
                assert_eq!(out[..], want[last as usize * out.len()..]);
                assert!(tr.read_frame(&mut cur, 0, &mut out).unwrap());
                assert_eq!(out[..], want[..out.len()]);
            }
        }
    }
}

#[cfg(test)]
mod seek_tests {
    use super::tests::image;
    use super::*;

    #[test]
    fn long_tracks_seek_anywhere() {
        let img = image(&[300, 450, 200], false);
        let mut cur = io::Cursor::new(&img.bytes);
        let disc = read_disc(&mut cur).unwrap();
        let area = disc.area(false).unwrap();
        let mut tr = TrackReader::new(area, 2).unwrap();
        let mut out = vec![0u8; tr.frame_bytes()];
        let fb = out.len();
        for n in [400u32, 10, 449, 0, 225, 226, 30] {
            assert!(tr.read_frame(&mut cur, n, &mut out).unwrap(), "frame {n}");
            assert!(
                out[..] == img.tracks[1][n as usize * fb..(n as usize + 1) * fb],
                "frame {n}"
            );
        }
        assert!(!tr.read_frame(&mut cur, 450, &mut out).unwrap());
    }
}
