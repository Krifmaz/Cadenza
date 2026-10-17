//! The window and the parts of the desktop around it.

use crate::app::{App, Message};
use iced::{window, Size, Subscription, Task};

/// Open the main window at its remembered size.
pub fn open_window() -> (window::Id, Task<window::Id>) {
    let saved = crate::store::load_window();
    let size = if saved.width >= 420.0 && saved.height >= 480.0 {
        Size::new(saved.width, saved.height)
    } else {
        Size::new(1280.0, 820.0)
    };
    window::open(window::Settings {
        size,
        min_size: Some(Size::new(420.0, 520.0)),
        icon: icon(),
        exit_on_close_request: false,
        ..Default::default()
    })
}

fn icon() -> Option<window::Icon> {
    window::icon::from_file_data(include_bytes!("../icons/128x128.png"), None).ok()
}

/// Events from outside the window (tray, media keys, a second launch).
pub fn events() -> Subscription<Message> {
    Subscription::none()
}

/// Keep the tray and the OS media controls in step with the engine.
pub fn sync(_app: &App) -> Task<Message> {
    Task::none()
}

/// Whether closing the window can leave the app running in the tray.
pub fn tray_available() -> bool {
    false
}
