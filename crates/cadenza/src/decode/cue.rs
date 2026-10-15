//! CUE sheets: one audio file (or several) cut into tracks by `INDEX 01` times.
//!
//! A track is addressed as `album.cue#3`. It plays from its `INDEX 01` to the next
//! track's `INDEX 01` in the same file (so a pregap belongs to the track before, and
//! consecutive tracks join sample for sample), or to the end of the file. Times are CD
//! frames (1/75 s), which are whole samples at every common rate and whole bytes in DSD.

use std::path::{Path, PathBuf};

/// One track.
#[derive(Clone, Debug, PartialEq)]
pub struct CueTrack {
    /// Track number as written.
    pub number: u32,
    /// Title.
    pub title: Option<String>,
    /// Performer (the sheet's when the track has none).
    pub performer: Option<String>,
    /// The audio file, as written in the sheet.
    pub file: String,
    /// `INDEX 01`, in 1/75 s.
    pub start: u64,
    /// The next track's `INDEX 01` in the same file, in 1/75 s; `None`: the file's end.
    pub end: Option<u64>,
}

/// A parsed sheet.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CueSheet {
    /// Album title.
    pub title: Option<String>,
    /// Album performer.
    pub performer: Option<String>,
    /// Tracks in order.
    pub tracks: Vec<CueTrack>,
}

/// `mm:ss:ff` → 1/75 s.
fn msf(s: &str) -> Option<u64> {
    let mut it = s.split(':');
    let m: u64 = it.next()?.trim().parse().ok()?;
    let sec: u64 = it.next()?.trim().parse().ok()?;
    let f: u64 = it.next()?.trim().parse().ok()?;
    (it.next().is_none() && sec < 60 && f < 75).then_some((m * 60 + sec) * 75 + f)
}

/// A quoted or bare argument, and what follows it.
fn arg(rest: &str) -> (String, &str) {
    let rest = rest.trim_start();
    if let Some(r) = rest.strip_prefix('"') {
        match r.find('"') {
            Some(end) => (r[..end].to_string(), &r[end + 1..]),
            None => (r.to_string(), ""),
        }
    } else {
        match rest.find(char::is_whitespace) {
            Some(end) => (rest[..end].to_string(), &rest[end..]),
            None => (rest.to_string(), ""),
        }
    }
}

/// Decode a sheet's bytes: UTF-8 (with or without BOM), else Windows-1252 / Latin-1.
pub fn text_of(bytes: &[u8]) -> String {
    let b = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    match std::str::from_utf8(b) {
        Ok(s) => s.to_string(),
        Err(_) => b.iter().map(|c| *c as char).collect(),
    }
}

/// Parse a sheet.
pub fn parse(text: &str) -> Result<CueSheet, String> {
    let mut sheet = CueSheet::default();
    let mut file: Option<String> = None;
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        let (cmd, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        match cmd.to_ascii_uppercase().as_str() {
            "FILE" => file = Some(arg(rest).0),
            "TRACK" => {
                let (num, kind) = arg(rest);
                let number = num
                    .parse()
                    .map_err(|_| format!("line {}: TRACK {num}", n + 1))?;
                let f = file
                    .clone()
                    .ok_or_else(|| format!("line {}: TRACK before FILE", n + 1))?;
                // Data tracks are listed but not played.
                let audio = kind.trim().eq_ignore_ascii_case("AUDIO");
                sheet.tracks.push(CueTrack {
                    number,
                    title: None,
                    performer: None,
                    file: if audio { f } else { String::new() },
                    start: u64::MAX,
                    end: None,
                });
            }
            "TITLE" | "PERFORMER" => {
                let v = Some(arg(rest).0).filter(|s| !s.trim().is_empty());
                let upper = cmd.to_ascii_uppercase();
                match (sheet.tracks.last_mut(), upper.as_str()) {
                    (Some(t), "TITLE") => t.title = v,
                    (Some(t), _) => t.performer = v,
                    (None, "TITLE") => sheet.title = v,
                    (None, _) => sheet.performer = v,
                }
            }
            "INDEX" => {
                let (idx, rest) = arg(rest);
                if idx.parse::<u32>().ok() == Some(1) {
                    let t =
                        msf(&arg(rest).0).ok_or_else(|| format!("line {}: INDEX time", n + 1))?;
                    if let Some(tr) = sheet.tracks.last_mut() {
                        tr.start = t;
                    }
                }
            }
            _ => {}
        }
    }
    sheet.tracks.retain(|t| !t.file.is_empty());
    if sheet.tracks.iter().any(|t| t.start == u64::MAX) {
        return Err("a track has no INDEX 01".into());
    }
    if sheet.tracks.is_empty() {
        return Err("no audio tracks".into());
    }
    let n = sheet.tracks.len();
    for i in 0..n {
        let next = sheet
            .tracks
            .get(i + 1)
            .filter(|t| t.file == sheet.tracks[i].file);
        sheet.tracks[i].end = next.map(|t| t.start);
        if sheet.tracks[i].performer.is_none() {
            sheet.tracks[i].performer = sheet.performer.clone();
        }
    }
    Ok(sheet)
}

/// Read and parse the sheet at `path`.
pub fn read(path: &Path) -> Result<CueSheet, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text_of(&bytes))
}

/// Where a track's audio file is, next to the sheet. A sheet that names a `.wav` when
/// the files were converted afterwards (a common case) finds the same name with another
/// audio extension.
pub fn resolve(sheet_path: &Path, file: &str) -> PathBuf {
    let dir = sheet_path.parent().unwrap_or(Path::new("."));
    let direct = dir.join(file.replace('\\', "/"));
    if direct.exists() {
        return direct;
    }
    for ext in [
        "flac", "wav", "ape", "wv", "dsf", "dff", "m4a", "aiff", "mp3",
    ] {
        let p = direct.with_extension(ext);
        if p.exists() {
            return p;
        }
    }
    direct
}

/// `album.cue#3` as a source: the sheet path and the track (1-based). Local files only.
pub fn cue_track(source: &super::Source) -> Option<(PathBuf, u32)> {
    match source {
        super::Source::Path { path } => {
            let (base, n) = parse_fragment(path.to_str()?)?;
            Some((PathBuf::from(base), n))
        }
        _ => None,
    }
}

/// `album.cue#3`: the sheet path and the track's position in the list (1-based).
pub fn parse_fragment(s: &str) -> Option<(&str, u32)> {
    let (base, frag) = s.rsplit_once('#')?;
    if !base.to_ascii_lowercase().ends_with(".cue") {
        return None;
    }
    Some((base, frag.parse().ok().filter(|n| *n > 0)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHEET: &str = "\u{feff}REM GENRE Jazz\r\nPERFORMER \"Miles Davis\"\r\nTITLE \"Kind of Blue\"\r\n\
FILE \"Kind of Blue.flac\" WAVE\r\n  TRACK 01 AUDIO\r\n    TITLE \"So What\"\r\n    INDEX 01 00:00:00\r\n\
  TRACK 02 AUDIO\r\n    TITLE \"Freddie Freeloader\"\r\n    PERFORMER \"Miles Davis Sextet\"\r\n\
    INDEX 00 09:20:50\r\n    INDEX 01 09:22:10\r\n  TRACK 03 AUDIO\r\n    TITLE Blue\r\n    INDEX 01 19:00:74\r\n\
FILE \"bonus.dsf\" WAVE\r\n  TRACK 04 AUDIO\r\n    INDEX 01 00:02:00\r\n";

    #[test]
    fn parses_tracks_times_and_names() {
        let s = parse(&text_of(SHEET.as_bytes())).unwrap();
        assert_eq!(s.title.as_deref(), Some("Kind of Blue"));
        assert_eq!(s.tracks.len(), 4);
        let t = &s.tracks;
        assert_eq!((t[0].start, t[0].end), (0, Some((9 * 60 + 22) * 75 + 10)));
        assert_eq!(t[1].title.as_deref(), Some("Freddie Freeloader"));
        assert_eq!(t[1].performer.as_deref(), Some("Miles Davis Sextet"));
        assert_eq!(t[0].performer.as_deref(), Some("Miles Davis"));
        // The last track of a file runs to its end; the next file starts anew.
        assert_eq!(t[2].end, None);
        assert_eq!(t[2].title.as_deref(), Some("Blue"));
        assert_eq!((t[3].file.as_str(), t[3].start), ("bonus.dsf", 150));
        // Latin-1 sheets.
        let s = parse(&text_of(
            b"FILE a.wav WAVE\nTRACK 1 AUDIO\nTITLE \"Caf\xe9\"\nINDEX 01 0:0:0\n",
        ))
        .unwrap();
        assert_eq!(s.tracks[0].title.as_deref(), Some("Café"));
        assert!(parse("TRACK 01 AUDIO\nINDEX 01 00:00:00").is_err());
        assert!(parse("FILE a WAVE\nTRACK 01 AUDIO\n").is_err());
        assert_eq!(parse_fragment("/m/a.CUE#12"), Some(("/m/a.CUE", 12)));
        assert_eq!(parse_fragment("/m/a.cue"), None);
    }
}
