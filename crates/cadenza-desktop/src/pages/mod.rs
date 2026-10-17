//! The pages of the window.

pub mod appearance;
pub mod encore;
pub mod filters;
pub mod guide;
pub mod library;
pub mod now;
pub mod output;
pub mod room;
pub mod sound;
pub mod system;

use crate::app::{App, Message};
use iced::Task;

/// A text box's value applied (Enter, or leaving a field that commits on its own).
pub fn commit_text(app: &mut App, key: &'static str, v: String) -> Task<Message> {
    match key {
        "gpu_device" => output::commit_gpu_device(app, v),
        "ap_custom" => appearance::commit_custom(app, v),
        k if k.starts_with("snd_") || k.starts_with("eq_") => return sound::commit(app, k, v),
        k if k.starts_with("enc_") => return encore::commit(app, k, v),
        k if k.starts_with("lib_") => return library::commit(app, k, v),
        k if k.starts_with("fd_") => return filters::commit(app, k, v),
        k if k.starts_with("room_") => return room::commit(app, k, v),
        _ => {}
    }
    Task::none()
}
