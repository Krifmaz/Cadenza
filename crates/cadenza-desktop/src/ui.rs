//! Small building blocks shared by the pages: cards, labels, badges, fields, choices.

use crate::app::Message;
use crate::theme::{self, Badge, Btn};
use iced::widget::{button, column, container, pick_list, row, text, text_input, toggler, Column};
use iced::{Alignment, Element, Length, Padding};

pub type El<'a> = Element<'a, Message>;

/// A choice in a pick list: shows `label`, compares by `id`.
#[derive(Clone, Debug)]
pub struct Choice {
    pub id: String,
    pub label: String,
}

impl Choice {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
        }
    }
}

impl PartialEq for Choice {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id
    }
}

impl std::fmt::Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label)
    }
}

/// `[{id, label}]` (or any key pair) from the engine's option lists.
pub fn choices(v: &serde_json::Value, id: &str, label: &str) -> Vec<Choice> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|x| {
                    let i = x[id]
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| x[id].to_string());
                    let l = x[label].as_str().map(str::to_string).unwrap_or_else(|| i.clone());
                    Choice::new(i, l)
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn pick<'a>(
    options: Vec<Choice>,
    selected: &str,
    on: impl Fn(String) -> Message + 'a,
) -> El<'a> {
    let sel = options.iter().find(|c| c.id == selected).cloned();
    pick_list(options, sel, move |c: Choice| on(c.id))
        .style(theme::pick)
        .menu_style(theme::menu)
        .padding([9, 12])
        .width(Length::Fill)
        .into()
}

pub fn h1<'a>(title: &'a str, sub: &'a str) -> El<'a> {
    let l = theme::look();
    column![
        text(title).size(30).font(bold()),
        text(sub).size(14).color(l.muted),
    ]
    .spacing(4)
    .padding(Padding::default().bottom(18))
    .into()
}

pub fn h2<'a>(t: impl text::IntoFragment<'a>) -> El<'a> {
    text(t).size(17).font(bold()).into()
}

pub fn bold() -> iced::Font {
    iced::Font {
        weight: iced::font::Weight::Semibold,
        ..iced::Font::DEFAULT
    }
}

pub fn card<'a>(content: impl Into<El<'a>>) -> El<'a> {
    container(content)
        .padding(20)
        .width(Length::Fill)
        .style(theme::card)
        .into()
}

pub fn label<'a>(t: impl text::IntoFragment<'a>) -> El<'a> {
    text(t).size(12.5).color(theme::look().muted).font(bold()).into()
}

pub fn hint<'a>(t: impl text::IntoFragment<'a>) -> El<'a> {
    text(t).size(12.5).color(theme::look().muted).into()
}

pub fn small<'a>(t: impl text::IntoFragment<'a>) -> El<'a> {
    text(t).size(12.5).into()
}

pub fn err<'a>(t: impl text::IntoFragment<'a>) -> El<'a> {
    text(t).size(12.5).color(theme::look().err).into()
}

/// The "In plain words" box.
pub fn plain<'a>(t: impl text::IntoFragment<'a>) -> El<'a> {
    container(text(t).size(12.5))
        .padding([8, 10])
        .width(Length::Fill)
        .style(theme::inset)
        .into()
}

pub fn badge<'a>(t: impl text::IntoFragment<'a>, kind: Badge) -> El<'a> {
    container(text(t).size(11.5).font(bold()))
        .padding([2, 9])
        .style(theme::badge(kind))
        .into()
}

/// Label above a control.
pub fn field<'a>(name: &'a str, control: impl Into<El<'a>>) -> El<'a> {
    column![label(name), control.into()].spacing(6).into()
}

pub fn input<'a>(placeholder: &str, value: &str, on: impl Fn(String) -> Message + 'a) -> iced::widget::TextInput<'a, Message> {
    text_input(placeholder, value)
        .on_input(on)
        .padding([9, 12])
        .style(theme::input)
}

pub fn btn<'a>(t: impl text::IntoFragment<'a>, kind: Btn, msg: Option<Message>) -> El<'a> {
    button(text(t).size(13.5).font(bold()))
        .padding([8, 14])
        .style(theme::btn(kind))
        .on_press_maybe(msg)
        .into()
}

pub fn switch<'a>(on: bool, name: &'a str, msg: impl Fn(bool) -> Message + 'a) -> El<'a> {
    toggler(on)
        .label(name)
        .on_toggle(msg)
        .size(20)
        .text_size(14)
        .style(theme::toggle)
        .into()
}

/// Two columns that fold into one when narrow (cards on a page).
pub fn grid<'a>(items: Vec<El<'a>>, wide: bool) -> El<'a> {
    if !wide {
        return Column::with_children(items).spacing(18).into();
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (i, it) in items.into_iter().enumerate() {
        if i % 2 == 0 {
            left.push(it)
        } else {
            right.push(it)
        }
    }
    row![
        Column::with_children(left).spacing(18).width(Length::Fill),
        Column::with_children(right).spacing(18).width(Length::Fill),
    ]
    .spacing(18)
    .into()
}

/// Segmented buttons: `(id, label)`; the chosen one is highlighted.
pub fn seg<'a>(items: &[(&'a str, &'a str)], on: &str, msg: impl Fn(String) -> Message) -> El<'a> {
    let mut r = row![].spacing(4);
    for (id, name) in items {
        let kind = if *id == on { Btn::SegOn } else { Btn::Seg };
        r = r.push(
            button(text(*name).size(13.5).font(bold()))
                .padding([7, 14])
                .style(theme::btn(kind))
                .on_press(msg(id.to_string())),
        );
    }
    container(r.align_y(Alignment::Center))
        .padding(4)
        .style(theme::inset)
        .into()
}

pub fn khz(rate: u32) -> String {
    let k = rate as f64 / 1000.0;
    if k.fract() == 0.0 {
        format!("{k:.0} kHz")
    } else {
        format!("{k} kHz")
    }
}

pub fn mmss(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}
