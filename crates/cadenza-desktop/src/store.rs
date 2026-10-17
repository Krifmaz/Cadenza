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
