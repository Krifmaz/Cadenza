//! Placeholder while the page is being ported.

use crate::app::{App, Message};
use crate::ui::{self, El};
use iced::Task;

#[derive(Default)]
pub struct State {}

#[derive(Clone, Debug)]
pub enum Msg {}

#[allow(dead_code)]
pub fn init(_app: &mut App) -> Task<Message> {
    Task::none()
}

pub fn update(_app: &mut App, m: Msg) -> Task<Message> {
    match m {}
}

#[allow(dead_code)]
pub fn commit(_app: &mut App, _k: &'static str, _v: String) -> Task<Message> {
    Task::none()
}

#[allow(dead_code)]
pub fn tick(_app: &mut App) -> Task<Message> {
    Task::none()
}

#[allow(dead_code)]
pub fn view<'a>(_app: &'a App, _wide: bool) -> El<'a> {
    ui::h1("filters", "Coming next.")
}
