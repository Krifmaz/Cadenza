//! System input: Cadenza as the upsampler for everything the computer plays.

use crate::app::{App, Message};
use crate::ui::{self, Choice, El};
use cadenza::system::{SystemConfig, SystemMode};
use iced::widget::{column, row, Column};
use iced::{Length, Task};
use serde_json::Value;

#[derive(Default)]
pub struct State {
    status: Value,
    apps: Vec<String>,
    app_text: String,
    last_poll: Option<std::time::Instant>,
}

impl State {
    pub fn on(&self) -> bool {
        self.status["running"].as_bool() == Some(true)
    }
}

#[derive(Clone, Debug)]
pub enum Msg {
    Loaded(Value, Vec<String>),
    Enabled(bool),
    Mode(String),
    AppText(String),
    AppCommit,
    PickApp(String),
}

pub fn init(_app: &mut App) -> Task<Message> {
    Task::perform(
        async {
            tokio::task::spawn_blocking(|| (cadenza::system::status(), cadenza::system::apps()))
                .await
                .unwrap_or_default()
        },
        |(s, a)| Message::System(Msg::Loaded(s, a)),
    )
}

fn config(app: &App) -> SystemConfig {
    serde_json::from_value(app.system.status["config"].clone()).unwrap_or_default()
}

fn apply(app: &mut App, cfg: SystemConfig) -> Task<Message> {
    if !cadenza::system::AVAILABLE {
        return Task::none();
    }
    if cfg.enabled && cfg.mode == SystemMode::App && cfg.app.trim().is_empty() {
        // Wait for an app to be picked.
        app.system.status["config"] = serde_json::to_value(&cfg).unwrap_or_default();
        return Task::none();
    }
    let engine = app.engine.clone();
    Task::perform(
        async move {
            tokio::task::spawn_blocking(move || {
                cadenza::system::set_config(engine, &cfg);
                (cadenza::system::status(), cadenza::system::apps())
            })
            .await
            .unwrap_or_default()
        },
        |(s, a)| Message::System(Msg::Loaded(s, a)),
    )
}

pub fn update(app: &mut App, m: Msg) -> Task<Message> {
    match m {
        Msg::Loaded(s, a) => {
            if app.system.app_text.is_empty() {
                app.system.app_text = s["config"]["app"].as_str().unwrap_or("").to_string();
            }
            app.system.status = s;
            app.system.apps = a;
            app.system.last_poll = Some(std::time::Instant::now());
            Task::none()
        }
        Msg::Enabled(on) => {
            let mut c = config(app);
            c.enabled = on;
            apply(app, c)
        }
        Msg::Mode(m) => {
            let mut c = config(app);
            c.mode = if m == "app" {
                SystemMode::App
            } else {
                SystemMode::Sink
            };
            apply(app, c)
        }
        Msg::AppText(t) => {
            app.system.app_text = t;
            Task::none()
        }
        Msg::AppCommit => {
            let mut c = config(app);
            c.app = app.system.app_text.trim().to_string();
            apply(app, c)
        }
        Msg::PickApp(a) => {
            app.system.app_text = a.clone();
            let mut c = config(app);
            c.app = a;
            apply(app, c)
        }
    }
}

pub fn view(app: &App, wide: bool) -> El<'_> {
    let j = &app.system.status;
    let c = config(app);
    let info = if !cadenza::system::AVAILABLE {
        "This build has no system input for this platform.".to_string()
    } else if let Some(e) = j["error"].as_str() {
        e.to_string()
    } else if j["running"].as_bool() != Some(true) {
        "Off.".into()
    } else {
        let st = &j["stats"];
        let what = if c.mode == SystemMode::App {
            format!("Capturing {}", c.app)
        } else {
            format!("Device \"{}\"", c.name)
        };
        let rate = st["rate"].as_u64().unwrap_or(0);
        if rate > 0 {
            let peak = st["peak"].as_f64().unwrap_or(0.0);
            format!(
                "{what} · {} ch at {} · peak {}",
                st["channels"],
                ui::khz(rate as u32),
                if peak > 0.0 {
                    format!("{:.1} dBFS", 20.0 * peak.log10())
                } else {
                    "silent".into()
                }
            )
        } else {
            format!("{what} · waiting for sound")
        }
    };
    let status = column![
        ui::h2("Status"),
        ui::small(info),
        ui::hint("Linux: a PipeWire device \"Cadenza\" any app can play to, or one app's sound before the mixer. Windows 10 2004+: process loopback (\"whole system\" means all apps except Cadenza; Windows has no virtual device without a signed driver). macOS 14.2+: process taps, muted, so the apps are heard only through Cadenza."),
    ]
    .spacing(12);
    let mode = if c.mode == SystemMode::App {
        "app"
    } else {
        "sink"
    };
    let mut ctl = Column::new()
        .spacing(12)
        .push(row![
            ui::h2("Upsample system audio"),
            iced::widget::space().width(Length::Fill),
            ui::switch(c.enabled, "On", |v| Message::System(Msg::Enabled(v))),
        ])
        .push(ui::field(
            "Source",
            ui::pick(
                vec![
                    Choice::new("sink", "Whole system (all apps)"),
                    Choice::new("app", "One application"),
                ],
                mode,
                |v| Message::System(Msg::Mode(v)),
            ),
        ));
    if c.mode == SystemMode::App {
        ctl = ctl.push(ui::field(
            "Application",
            ui::input("e.g. spotify, chrome, firefox", &app.system.app_text, |v| {
                Message::System(Msg::AppText(v))
            })
            .on_submit(Message::System(Msg::AppCommit)),
        ));
        if !app.system.apps.is_empty() {
            let apps: Vec<Choice> = app
                .system
                .apps
                .iter()
                .map(|a| Choice::new(a.clone(), a.clone()))
                .collect();
            ctl = ctl.push(ui::field(
                "Running programs",
                ui::pick(apps, &app.system.app_text, |v| {
                    Message::System(Msg::PickApp(v))
                }),
            ));
        }
    }
    ctl = ctl.push(ui::hint("Whole system captures everything except Cadenza. Set a different device as your system default so you do not hear it twice, and let Cadenza output to your DAC in exclusive mode. Changes apply right away."));
    column![
        ui::h1(
            "System input",
            "Cadenza as the upsampler for everything your computer plays."
        ),
        ui::grid(vec![ui::card(status), ui::card(ctl)], wide)
    ]
    .into()
}
