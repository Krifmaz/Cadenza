//! Encore: sign in, so Cadenza is a playback device in Encore and its library is here.

use crate::app::{App, Message};
use crate::theme::{Badge, Btn};
use crate::ui::{self, El};
use cadenza::config::EncoreConfig;
use iced::widget::{column, row};
use iced::Task;
use serde_json::Value;

#[derive(Default)]
pub struct State {
    pub status: Value,
    url: String,
    user: String,
    pass: String,
    name: String,
    msg: String,
    busy: bool,
    filled: bool,
}

impl State {
    pub fn badge(&self) -> (String, Badge) {
        if self.status["connected"].as_bool() == Some(true) {
            ("on".into(), Badge::Ok)
        } else if self.status["configured"].as_bool() == Some(true) {
            ("error".into(), Badge::Bad)
        } else {
            ("off".into(), Badge::Plain)
        }
    }
}

#[derive(Clone, Debug)]
pub enum Msg {
    Status(Value),
    Url(String),
    User(String),
    Pass(String),
    Name(String),
    SignIn,
    SignOut,
}

pub fn init(_app: &mut App) -> Task<Message> {
    poll()
}

pub fn poll() -> Task<Message> {
    Task::perform(
        async {
            tokio::task::spawn_blocking(cadenza::encore::status)
                .await
                .unwrap_or_default()
        },
        |s| Message::Encore(Msg::Status(s)),
    )
}

fn set(app: &mut App, cfg: EncoreConfig) -> Task<Message> {
    app.encore.busy = true;
    let engine = app.engine.clone();
    Task::perform(
        async move {
            tokio::task::spawn_blocking(move || {
                cadenza::encore::set_config(engine, &cfg);
                // The link signs in on its own thread; give it a moment.
                for _ in 0..20 {
                    std::thread::sleep(std::time::Duration::from_millis(250));
                    let s = cadenza::encore::status();
                    if s["connected"].as_bool() == Some(true)
                        || s["error"].is_string()
                        || s["configured"].as_bool() != Some(true)
                    {
                        return s;
                    }
                }
                cadenza::encore::status()
            })
            .await
            .unwrap_or_default()
        },
        |s| Message::Encore(Msg::Status(s)),
    )
}

pub fn update(app: &mut App, m: Msg) -> Task<Message> {
    let st = &mut app.encore;
    match m {
        Msg::Status(s) => {
            if !st.filled {
                st.url = s["server"].as_str().unwrap_or("").into();
                st.user = s["username"].as_str().unwrap_or("").into();
                st.name = s["device_name"].as_str().unwrap_or("").into();
                st.filled = true;
            }
            if st.busy {
                st.pass.clear();
            }
            st.busy = false;
            st.msg.clear();
            st.status = s;
        }
        Msg::Url(v) => st.url = v,
        Msg::User(v) => st.user = v,
        Msg::Pass(v) => st.pass = v,
        Msg::Name(v) => st.name = v,
        Msg::SignIn => {
            // A blank password keeps the saved one (see `encore::set_config`).
            let cfg = EncoreConfig {
                url: st.url.trim().into(),
                username: st.user.trim().into(),
                device_name: st.name.trim().into(),
                password: st.pass.clone(),
                ..Default::default()
            };
            st.msg = "Signing in…".into();
            return set(app, cfg);
        }
        Msg::SignOut => {
            st.url.clear();
            st.user.clear();
            st.pass.clear();
            st.name.clear();
            st.msg = "Signing out…".into();
            return set(app, EncoreConfig::default());
        }
    }
    Task::none()
}

pub fn commit(_app: &mut App, _k: &str, _v: String) -> Task<Message> {
    Task::none()
}

pub fn view(app: &App, _wide: bool) -> El<'_> {
    let st = &app.encore;
    let j = &st.status;
    let info = if !st.msg.is_empty() {
        st.msg.clone()
    } else if j["configured"].as_bool() != Some(true) {
        "Not signed in.".into()
    } else if let Some(e) = j["error"].as_str() {
        format!("Problem talking to Encore: {e}")
    } else {
        format!(
            "Signed in as \"{}\" → {}. You can now pick Cadenza in Encore, or browse your library on the Library page.",
            j["device_name"].as_str().unwrap_or(""),
            j["server"].as_str().unwrap_or("")
        )
    };
    let (b, k) = st.badge();
    let card = column![
        row![ui::h2("Encore account"), ui::badge(b, k)].spacing(8),
        ui::field("Server address", ui::input("http://192.168.1.10:8080", &st.url, |v| Message::Encore(Msg::Url(v)))),
        row![
            ui::field("Username", ui::input("", &st.user, |v| Message::Encore(Msg::User(v)))),
            ui::field(
                "Password",
                ui::input("(blank keeps the saved one)", &st.pass, |v| Message::Encore(Msg::Pass(v)))
                    .secure(true)
                    .on_submit(Message::Encore(Msg::SignIn))
            ),
        ]
        .spacing(12),
        ui::field("Device name (shown in Encore)", ui::input("Cadenza on this computer", &st.name, |v| Message::Encore(Msg::Name(v)))),
        row![
            ui::btn(if st.busy { "Signing in…" } else { "Sign in" }, Btn::Primary, (!st.busy).then_some(Message::Encore(Msg::SignIn))),
            ui::btn("Sign out", Btn::Ghost, (!st.busy).then_some(Message::Encore(Msg::SignOut))),
        ]
        .spacing(8),
        ui::small(info),
        ui::hint("Signing in makes Cadenza a playback device in Encore (pick it in Encore's device list) and shows your Encore library on the Library page. The login stays on this computer."),
    ]
    .spacing(12);
    column![
        ui::h1(
            "Encore",
            "Use Cadenza as Encore's player, and play your Encore library here."
        ),
        ui::card(card)
    ]
    .into()
}
