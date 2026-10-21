//! What the app remembers between runs, next to the config file: the look, and the
//! window's size and position.

use crate::theme::Appearance;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn dir() -> PathBuf {
    cadenza::config::Config::default_path()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The app's log (the one from the run before is kept as `cadenza.previous.log`).
pub fn log_path() -> PathBuf {
    dir().join("cadenza.log")
}

/// Show `path` in the file manager (Explorer, Finder, the desktop's own).
pub fn reveal(path: &std::path::Path) {
    #[cfg(windows)]
    let r = std::process::Command::new("explorer")
        .arg(format!("/select,{}", path.display()))
        .spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open")
        .arg("-R")
        .arg(path)
        .spawn();
    #[cfg(not(any(windows, target_os = "macos")))]
    let r = std::process::Command::new("xdg-open")
        .arg(path.parent().unwrap_or(path))
        .spawn();
    if let Err(e) = r {
        log::warn!("cannot show {}: {e}", path.display());
    }
}

fn read<T: for<'de> Deserialize<'de> + Default>(name: &str) -> T {
    std::fs::read_to_string(dir().join(name))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write<T: Serialize>(name: &str, v: &T) {
    let d = dir();
    let _ = std::fs::create_dir_all(&d);
    if let Ok(s) = serde_json::to_string_pretty(v) {
        if let Err(e) = std::fs::write(d.join(name), s) {
            log::warn!("could not save {name}: {e}");
        }
    }
}

pub fn load_appearance() -> Appearance {
    read("appearance.json")
}

pub fn save_appearance(a: &Appearance) {
    write("appearance.json", a);
}

/// Whether the update check also offers hotfix builds (every push to main).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Updates {
    #[serde(default)]
    pub hotfix: bool,
}

pub fn load_updates() -> Updates {
    read("updates.json")
}

pub fn save_updates(u: &Updates) {
    write("updates.json", u);
}

/// The window's last size (logical pixels) and position.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WindowState {
    pub width: f32,
    pub height: f32,
    pub x: Option<f32>,
    pub y: Option<f32>,
}

pub fn load_window() -> WindowState {
    read("window.json")
}

pub fn save_window(app: &crate::app::App) {
    let mut w = load_window();
    w.width = app.width;
    w.height = app.height;
    write("window.json", &w);
}
