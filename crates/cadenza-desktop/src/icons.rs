//! Line icons drawn as SVG and tinted to the theme, so they look the same on every
//! system (plain symbol characters turn into colour emoji on Windows).

use crate::app::Message;
use iced::widget::svg::{self, Handle, Svg};
use iced::{Color, Element};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Icon {
    Play,
    Pause,
    Prev,
    Next,
    Stop,
    Volume,
    Mute,
    Music,
    Library,
    Guide,
    Sliders,
    Wave,
    Speaker,
    Room,
    Monitor,
    Cloud,
    Palette,
}

const ALL: [Icon; 17] = [
    Icon::Play,
    Icon::Pause,
    Icon::Prev,
    Icon::Next,
    Icon::Stop,
    Icon::Volume,
    Icon::Mute,
    Icon::Music,
    Icon::Library,
    Icon::Guide,
    Icon::Sliders,
    Icon::Wave,
    Icon::Speaker,
    Icon::Room,
    Icon::Monitor,
    Icon::Cloud,
    Icon::Palette,
];

/// The shapes, on a 24×24 grid. Filled shapes for the transport, strokes for the rest.
fn body(i: Icon) -> &'static str {
    match i {
        Icon::Play => {
            r#"<path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5z"/>"#
        }
        Icon::Pause => {
            r#"<rect x="5.5" y="4" width="4.5" height="16" rx="1.2"/><rect x="14" y="4" width="4.5" height="16" rx="1.2"/>"#
        }
        Icon::Prev => {
            r#"<rect x="4" y="5" width="2.6" height="14" rx="1"/><path d="M20 6.2v11.6a1 1 0 0 1-1.55.83L9.6 12.83a1 1 0 0 1 0-1.66l8.85-5.8A1 1 0 0 1 20 6.2z"/>"#
        }
        Icon::Next => {
            r#"<rect x="17.4" y="5" width="2.6" height="14" rx="1"/><path d="M4 6.2v11.6a1 1 0 0 0 1.55.83l8.85-5.8a1 1 0 0 0 0-1.66L5.55 5.37A1 1 0 0 0 4 6.2z"/>"#
        }
        Icon::Stop => r#"<rect x="6" y="6" width="12" height="12" rx="2"/>"#,
        Icon::Volume => {
            r#"<path d="M4 9.5h3l4.5-4v13l-4.5-4H4z" fill="black" stroke="black" stroke-width="1.6" stroke-linejoin="round"/><path d="M15.5 9a4 4 0 0 1 0 6M18.5 6.5a7.5 7.5 0 0 1 0 11" fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round"/>"#
        }
        Icon::Mute => {
            r#"<path d="M4 9.5h3l4.5-4v13l-4.5-4H4z" fill="black" stroke="black" stroke-width="1.6" stroke-linejoin="round"/><path d="M16 9.5l5 5M21 9.5l-5 5" fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round"/>"#
        }
        Icon::Music => {
            r#"<g fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l11-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/></g>"#
        }
        Icon::Library => {
            r#"<g fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4v16M8 4v16"/><path d="M12 5l4-1 4 15.5-4 1z"/></g>"#
        }
        Icon::Guide => {
            r#"<g fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.3a2.6 2.6 0 0 1 5 .9c0 1.7-2.5 2.3-2.5 4"/><path d="M12 17.3h.01"/></g>"#
        }
        Icon::Sliders => {
            r#"<g fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round"><path d="M6 4v5M6 13v7M12 4v9M12 17v3M18 4v3M18 11v9"/><path d="M4 11h4M10 15h4M16 9h4"/></g>"#
        }
        Icon::Wave => {
            r#"<path d="M2 12h3l2.5-7 4 14 4-11 2.5 4H22" fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>"#
        }
        Icon::Speaker => {
            r#"<g fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="2.5" width="14" height="19" rx="2.5"/><circle cx="12" cy="14.5" r="3.5"/><path d="M12 7h.01"/></g>"#
        }
        Icon::Room => {
            r#"<g fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2.5" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7"/></g>"#
        }
        Icon::Monitor => {
            r#"<g fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2.5" y="4" width="19" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></g>"#
        }
        Icon::Cloud => {
            r#"<path d="M7 18.5h10.5a4 4 0 0 0 .4-7.98A6 6 0 0 0 6.3 9.6 4.5 4.5 0 0 0 7 18.5z" fill="none" stroke="black" stroke-width="1.8" stroke-linejoin="round"/>"#
        }
        Icon::Palette => {
            r#"<g fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a9 9 0 0 0 0 18c1.2 0 1.8-.8 1.8-1.7 0-1.3-1-1.5-1-2.6 0-1 .8-1.7 1.8-1.7H17a4 4 0 0 0 4-4C21 6.6 17 3 12 3z"/><circle cx="7.5" cy="11" r="1"/><circle cx="10" cy="7" r="1"/><circle cx="15" cy="7.3" r="1"/></g>"#
        }
    }
}

fn handle(i: Icon) -> Handle {
    static CACHE: OnceLock<Vec<Handle>> = OnceLock::new();
    let all = CACHE.get_or_init(|| {
        ALL.iter()
            .map(|&i| {
                let svg = format!(
                    r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="black">{}</svg>"#,
                    body(i)
                );
                Handle::from_memory(svg.into_bytes())
            })
            .collect()
    });
    all[ALL.iter().position(|&x| x == i).unwrap_or(0)].clone()
}

/// The icon `i`, `size` pixels square, painted in `color`.
pub fn icon<'a>(i: Icon, size: f32, color: Color) -> Element<'a, Message> {
    Svg::new(handle(i))
        .width(size)
        .height(size)
        .style(move |_, _| svg::Style { color: Some(color) })
        .into()
}
