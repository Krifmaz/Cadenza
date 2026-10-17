//! Screenshots for checking the pages without a person at the screen:
//! `CADENZA_SHOT=<dir>` (and optionally `CADENZA_SHOT_PAGES=now,sound,sound:1,…`, where
//! `:f` scrolls that fraction of the way down) opens each page, saves `<dir>/<name>.rgba`
//! plus `<name>.size` (`width height`), then quits.

use crate::app::{App, Message, Page, MAIN_SCROLL};
use iced::Task;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum Msg {
    Step,
    Taken(String, iced::window::Screenshot),
}

pub struct State {
    dir: PathBuf,
    shots: Vec<(Page, f32, String)>,
    next: usize,
    /// Ticks to wait before the next step (lists load, covers arrive).
    wait: u32,
    /// The current shot's page is open and scrolled.
    ready: bool,
}

impl State {
    pub fn from_env() -> Option<Self> {
        let dir = PathBuf::from(std::env::var_os("CADENZA_SHOT")?);
        let spec = std::env::var("CADENZA_SHOT_PAGES").unwrap_or_else(|_| {
            Page::ALL
                .iter()
                .map(|p| p.id())
                .collect::<Vec<_>>()
                .join(",")
        });
        let shots = spec
            .split(',')
            .filter_map(|s| {
                let (id, frac) = s.trim().split_once(':').unwrap_or((s.trim(), "0"));
                let page = Page::ALL.into_iter().find(|p| p.id() == id)?;
                let f: f32 = frac.parse().unwrap_or(0.0);
                let name = if f > 0.0 {
                    format!("{id}-{}", (f * 100.0) as u32)
                } else {
                    id.to_string()
                };
                Some((page, f, name))
            })
            .collect();
        Some(Self {
            dir,
            shots,
            next: 0,
            wait: 3,
            ready: false,
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
            let Some((page, frac, name)) = s.shots.get(s.next).cloned() else {
                return iced::exit();
            };
            if !s.ready {
                s.ready = true;
                s.wait = 2;
                let nav = app.update(Message::Nav(page));
                let scroll = iced::widget::operation::snap_to(
                    MAIN_SCROLL,
                    iced::widget::scrollable::RelativeOffset { x: 0.0, y: frac },
                );
                return nav.chain(scroll);
            }
            s.next += 1;
            s.ready = false;
            s.wait = 0;
            let Some(id) = app.window else {
                return Task::none();
            };
            iced::window::screenshot(id)
                .map(move |shot| Message::Shot(Msg::Taken(name.clone(), shot)))
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
