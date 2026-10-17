//! Screenshots for checking the pages without a person at the screen:
//! `CADENZA_SHOT=<dir>` (and optionally `CADENZA_SHOT_PAGES=now,sound,…`) opens each page,
//! saves `<dir>/<page>.rgba` plus `<page>.size` (`width height`), then quits.

use crate::app::{App, Message, Page};
use iced::Task;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum Msg {
    Step,
    Taken(String, iced::window::Screenshot),
}

pub struct State {
    dir: PathBuf,
    pages: Vec<Page>,
    next: usize,
    /// Ticks to wait on a page before taking it (lists load, covers arrive).
    wait: u32,
}

impl State {
    pub fn from_env() -> Option<Self> {
        let dir = PathBuf::from(std::env::var_os("CADENZA_SHOT")?);
        let pages = match std::env::var("CADENZA_SHOT_PAGES") {
            Ok(list) => list
                .split(',')
                .filter_map(|id| Page::ALL.into_iter().find(|p| p.id() == id.trim()))
                .collect(),
            Err(_) => Page::ALL.to_vec(),
        };
        Some(Self {
            dir,
            pages,
            next: 0,
            wait: 3,
        })
    }
}

pub fn update(app: &mut App, msg: Msg) -> Task<Message> {
    let Some(s) = app.shot.as_mut() else {
        return Task::none();
    };
    match msg {
        Msg::Step => {
            if s.wait > 0 {
                s.wait -= 1;
                return Task::none();
            }
            let Some(&page) = s.pages.get(s.next) else {
                return iced::exit();
            };
            if app.page != page {
                s.wait = 2;
                return app.update(Message::Nav(page));
            }
            s.next += 1;
            s.wait = 1;
            let Some(id) = app.window else {
                return Task::none();
            };
            let name = page.id().to_string();
            iced::window::screenshot(id).map(move |shot| Message::Shot(Msg::Taken(name.clone(), shot)))
        }
        Msg::Taken(name, shot) => {
            let _ = std::fs::create_dir_all(&s.dir);
            let _ = std::fs::write(s.dir.join(format!("{name}.rgba")), &shot.rgba[..]);
            let _ = std::fs::write(
                s.dir.join(format!("{name}.size")),
                format!("{} {}", shot.size.width, shot.size.height),
            );
            Task::none()
        }
    }
}
