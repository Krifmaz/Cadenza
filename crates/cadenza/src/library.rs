//! The local library: saved music folders, a folder browser and recursive scans, for the
//! web UI's Library page and the desktop app's drag-and-drop.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// `[library]` in config.toml.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LibraryConfig {
    /// Saved music folders, shown as shortcuts.
    pub folders: Vec<String>,
}

/// File extensions the player opens (decoded audio, DSD, SACD images and CUE sheets).
pub const AUDIO_EXTS: &[&str] = &[
    "flac", "wav", "wave", "w64", "aif", "aiff", "aifc", "caf", "alac", "m4a", "mp4", "mp3", "ogg",
    "oga", "opus", "wv", "ape", "mka", "dsf", "dff", "iso", "cue",
];

/// At most this many files come back from one scan.
pub const SCAN_LIMIT: usize = 20_000;
const SCAN_DEPTH: usize = 16;

type Persist = Box<dyn Fn(&LibraryConfig) + Send>;
static FOLDERS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static PERSIST: Mutex<Option<Persist>> = Mutex::new(None);

/// The saved folders from config.toml (called once at startup).
pub fn init(folders: Vec<String>) {
    *FOLDERS.lock().unwrap() = clean(folders);
}

/// Register the function that saves the library configuration.
pub fn set_persist(f: impl Fn(&LibraryConfig) + Send + 'static) {
    *PERSIST.lock().unwrap() = Some(Box::new(f));
}

/// The saved folders.
pub fn folders() -> Vec<String> {
    FOLDERS.lock().unwrap().clone()
}

/// Replace the saved folders (trimmed, without blanks or repeats) and save them.
pub fn set_folders(folders: Vec<String>) {
    let folders = clean(folders);
    *FOLDERS.lock().unwrap() = folders.clone();
    if let Some(f) = PERSIST.lock().unwrap().as_ref() {
        f(&LibraryConfig { folders });
    }
}

fn clean(folders: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for f in folders {
        let f = f.trim().to_string();
        if !f.is_empty() && !out.contains(&f) {
            out.push(f);
        }
    }
    out
}

/// Whether the player opens this file (by its extension).
pub fn is_audio(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| AUDIO_EXTS.iter().any(|a| a.eq_ignore_ascii_case(e)))
}

fn hidden(name: &str) -> bool {
    name.starts_with('.')
        || name.eq_ignore_ascii_case("$RECYCLE.BIN")
        || name == "System Volume Information"
}

fn sort_key(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// Where browsing starts: the drives on Windows, `/` and the home folder elsewhere.
fn roots() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        (b'C'..=b'Z')
            .map(|d| PathBuf::from(format!("{}:\\", d as char)))
            .filter(|p| p.exists())
            .collect()
    }
    #[cfg(not(windows))]
    {
        let mut v = vec![PathBuf::from("/")];
        if let Some(h) = dirs::home_dir() {
            v.insert(0, h);
        }
        v
    }
}

/// A folder's sub-folders and audio files; an empty `path` lists the starting points.
pub fn browse(path: &str) -> Result<Value, String> {
    let entry = |p: &Path| {
        json!({
            "name": p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string()),
            "path": p.display().to_string(),
        })
    };
    if path.trim().is_empty() {
        let dirs: Vec<Value> = roots()
            .iter()
            .map(|p| json!({ "name": p.display().to_string(), "path": p.display().to_string() }))
            .collect();
        return Ok(json!({ "path": "", "parent": null, "dirs": dirs, "files": [] }));
    }
    let dir = PathBuf::from(path.trim());
    let rd = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (mut dirs, mut files) = (Vec::new(), Vec::new());
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if hidden(&name) {
            continue;
        }
        let p = e.path();
        match e.file_type() {
            Ok(t) if t.is_dir() || (t.is_symlink() && p.is_dir()) => dirs.push(p),
            Ok(_) if is_audio(&p) => files.push(p),
            _ => {}
        }
    }
    dirs.sort_by_key(|p| sort_key(p));
    files.sort_by_key(|p| sort_key(p));
    let parent = dir
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    Ok(json!({
        "path": dir.display().to_string(),
        "parent": parent,
        "dirs": dirs.iter().map(|p| entry(p)).collect::<Vec<_>>(),
        "files": files.iter().map(|p| entry(p)).collect::<Vec<_>>(),
    }))
}

/// Every audio file under `root` (or `root` itself when it is a file), folder by folder in
/// name order. A CUE sheet stands in for the audio file it describes (same name), so an
/// album ripped as one file plus a CUE plays as its tracks once, not twice.
pub fn scan(root: &Path) -> Result<Vec<String>, String> {
    if root.is_file() {
        return Ok(if is_audio(root) {
            vec![root.display().to_string()]
        } else {
            Vec::new()
        });
    }
    if !root.is_dir() {
        return Err(format!("{}: not a folder", root.display()));
    }
    let mut out = Vec::new();
    walk(root, 0, &mut out);
    Ok(out)
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<String>) {
    if depth > SCAN_DEPTH || out.len() >= SCAN_LIMIT {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let (mut dirs, mut files) = (Vec::new(), Vec::new());
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if hidden(&name) {
            continue;
        }
        let p = e.path();
        match e.file_type() {
            Ok(t) if t.is_dir() => dirs.push(p),
            Ok(_) if is_audio(&p) => files.push(p),
            _ => {}
        }
    }
    files.sort_by_key(|p| sort_key(p));
    dirs.sort_by_key(|p| sort_key(p));
    let cue_stems: Vec<String> = files
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("cue")))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_lowercase()))
        .collect();
    for f in files {
        let is_cue = f.extension().is_some_and(|e| e.eq_ignore_ascii_case("cue"));
        let stem = f
            .file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if !is_cue && cue_stems.contains(&stem) {
            continue;
        }
        if out.len() >= SCAN_LIMIT {
            return;
        }
        out.push(f.display().to_string());
    }
    for d in dirs {
        walk(&d, depth + 1, out);
    }
}

/// A queue entry for a local file, titled by its file name (as `POST /api/queue` does).
pub fn track_for(path: &str) -> crate::engine::Track {
    let title = path.rsplit(['/', '\\']).next().unwrap_or(path).to_string();
    crate::engine::Track {
        id: None,
        source: crate::decode::Source::Path { path: path.into() },
        title,
        artist: String::new(),
        album: String::new(),
        album_id: None,
        duration: None,
        format: None,
        cover: None,
    }
}

/// Files and folders dropped on the desktop window as queue entries: folders are scanned,
/// CUE sheets and disc images become their tracks, anything unplayable is skipped.
pub fn dropped(paths: &[PathBuf]) -> Vec<crate::engine::Track> {
    let mut out = Vec::new();
    for p in paths {
        for f in scan(p).unwrap_or_default() {
            match track_for(&f).expand() {
                Ok(t) => out.extend(t),
                Err(e) => log::warn!("{f}: {e}"),
            }
        }
    }
    out
}

/// Play dropped files: start them when nothing is playing, otherwise add them to the queue.
pub fn play_dropped(engine: &crate::engine::Engine, paths: &[PathBuf]) -> usize {
    let tracks = dropped(paths);
    let n = tracks.len();
    if n == 0 {
        return 0;
    }
    if engine.status().state == crate::engine::PlayState::Stopped {
        engine.play_queue(tracks, 0, 0.0);
    } else {
        engine.send(crate::engine::Command::Enqueue(tracks));
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    #[test]
    fn scan_orders_folders_skips_junk_and_prefers_cue_sheets() {
        let t = tempfile::tempdir().unwrap();
        let r = t.path();
        touch(&r.join("B Album/02 two.flac"));
        touch(&r.join("B Album/01 one.flac"));
        touch(&r.join("B Album/cover.jpg"));
        touch(&r.join("A Album/disc.flac"));
        touch(&r.join("A Album/disc.cue"));
        touch(&r.join(".hidden/x.flac"));
        touch(&r.join("top.mp3"));
        let got: Vec<String> = scan(r)
            .unwrap()
            .iter()
            .map(|p| {
                Path::new(p)
                    .strip_prefix(r)
                    .unwrap()
                    .display()
                    .to_string()
                    .replace('\\', "/")
            })
            .collect();
        assert_eq!(
            got,
            [
                "top.mp3",
                "A Album/disc.cue",
                "B Album/01 one.flac",
                "B Album/02 two.flac"
            ]
        );
        // A single file scans as itself; a missing folder is an error.
        assert_eq!(scan(&r.join("top.mp3")).unwrap().len(), 1);
        assert!(scan(&r.join("nope")).is_err());
    }

    #[test]
    fn browse_lists_folders_then_audio_files() {
        let t = tempfile::tempdir().unwrap();
        let r = t.path();
        touch(&r.join("Sub/a.flac"));
        touch(&r.join("b.wav"));
        touch(&r.join("notes.txt"));
        let v = browse(&r.display().to_string()).unwrap();
        assert_eq!(v["dirs"][0]["name"], "Sub");
        assert_eq!(v["files"].as_array().unwrap().len(), 1);
        assert_eq!(v["files"][0]["name"], "b.wav");
        assert!(!browse("").unwrap()["dirs"].as_array().unwrap().is_empty());
    }

    #[test]
    fn dropped_folders_become_tracks_named_by_file() {
        let t = tempfile::tempdir().unwrap();
        touch(&t.path().join("Album/01 one.flac"));
        touch(&t.path().join("Album/02 two.flac"));
        let tracks = dropped(&[t.path().join("Album"), t.path().join("missing")]);
        let titles: Vec<&str> = tracks.iter().map(|t| t.title.as_str()).collect();
        assert_eq!(titles, ["01 one.flac", "02 two.flac"]);
    }

    #[test]
    fn folders_are_cleaned_and_saved() {
        let saved = std::sync::Arc::new(Mutex::new(None));
        let s = saved.clone();
        set_persist(move |c: &LibraryConfig| *s.lock().unwrap() = Some(c.clone()));
        set_folders(vec![" /m ".into(), "".into(), "/m".into(), "/n".into()]);
        assert_eq!(folders(), ["/m", "/n"]);
        assert_eq!(
            saved.lock().unwrap().as_ref().unwrap().folders,
            ["/m", "/n"]
        );
    }
}
