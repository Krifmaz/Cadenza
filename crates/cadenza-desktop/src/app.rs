//! The window: state, messages, the shell (sidebar, player bar) and page switching. The
//! engine runs in this process; pages read its status and call it directly.

use crate::icons::{icon, Icon};
use crate::pages;
use crate::theme::{self, Appearance, Btn};
use crate::ui::{self, El};
use cadenza::config::Settings;
use cadenza::engine::{Command, Engine, PlayState, Status};
use cadenza::output::DeviceInfo;
use iced::widget::{button, column, container, image, row, scrollable, slider, space, text};
use iced::{window, Alignment, Element, Length, Size, Subscription, Task};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// The pages, in sidebar order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Page {
    Now,
    Library,
    Guide,
    Sound,
    Filters,
    Output,
    Room,
    System,
    Encore,
    Appearance,
}

impl Page {
    pub const ALL: [Page; 10] = [
        Page::Now,
        Page::Library,
        Page::Guide,
        Page::Sound,
        Page::Filters,
        Page::Output,
        Page::Room,
        Page::System,
        Page::Encore,
        Page::Appearance,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Page::Now => "Now playing",
            Page::Library => "Library",
            Page::Guide => "Guide",
            Page::Sound => "Sound",
            Page::Filters => "Filters",
            Page::Output => "Output",
            Page::Room => "Room & measure",
            Page::System => "System input",
            Page::Encore => "Encore",
            Page::Appearance => "Appearance",
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Page::Now => "now",
            Page::Library => "library",
            Page::Guide => "guide",
            Page::Sound => "sound",
            Page::Filters => "filters",
            Page::Output => "output",
            Page::Room => "room",
            Page::System => "system",
            Page::Encore => "encore",
            Page::Appearance => "appearance",
        }
    }

    fn icon(self) -> Icon {
        match self {
            Page::Now => Icon::Play,
            Page::Library => Icon::Library,
            Page::Guide => Icon::Guide,
            Page::Sound => Icon::Sliders,
            Page::Filters => Icon::Wave,
            Page::Output => Icon::Speaker,
            Page::Room => Icon::Room,
            Page::System => Icon::Monitor,
            Page::Encore => Icon::Cloud,
            Page::Appearance => Icon::Palette,
        }
    }
}

/// Transport, from the player bar, the queue, the tray or the media keys.
#[derive(Clone, Copy, Debug)]
pub enum Transport {
    Play,
    Pause,
    Toggle,
    Stop,
    Next,
    Prev,
    Volume(f64),
    PlayIndex(usize),
    Remove(usize),
}

impl Transport {
    pub fn command(self) -> Command {
        match self {
            Transport::Play => Command::Play,
            Transport::Pause => Command::Pause,
            Transport::Toggle => Command::Toggle,
            Transport::Stop => Command::Stop,
            Transport::Next => Command::Next,
            Transport::Prev => Command::Prev,
            Transport::Volume(v) => Command::SetVolume(cadenza_dsp_volume(v)),
            Transport::PlayIndex(i) => Command::PlayIndex(i),
            Transport::Remove(i) => Command::Remove(i),
        }
    }
}

fn cadenza_dsp_volume(v: f64) -> cadenza_dsp::Volume {
    cadenza_dsp::Volume::from_linear(v)
}

/// A change to the settings, applied to a copy and handed to the engine.
#[derive(Clone)]
pub struct Edit(pub Arc<dyn Fn(&mut Settings) + Send + Sync>);

impl std::fmt::Debug for Edit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Edit")
    }
}

/// `edit(|s| s.dsp.soft_clip = on)`.
pub fn edit(f: impl Fn(&mut Settings) + Send + Sync + 'static) -> Message {
    Message::Edit(Edit(Arc::new(f)))
}

/// A settings field set from its JSON id (`{"dsp": {"dither": "auto"}}`), as the web page
/// does: the engine's option lists hand out ids, so the pick lists need no enum mapping.
pub fn patch(v: Value) -> Message {
    Message::Patch(v)
}

#[derive(Clone, Debug)]
pub enum Message {
    Tick,
    Nav(Page),
    Transport(Transport),
    SeekDrag(f64),
    SeekDone,
    VolumeDrag(f64),
    VolumeDone,
    Edit(Edit),
    Patch(Value),
    /// A text box changed (`key`, text); `TextCommit` applies it.
    Text(String, String),
    TextCommit(String),
    RefreshDevices(bool),
    Devices(Vec<DeviceInfo>),
    Cover(String, Option<Vec<u8>>),
    SetAppearance(Appearance),
    WindowOpened(window::Id),
    WindowClosed(window::Id),
    CloseRequested(window::Id),
    Resized(Size),
    Raise,
    Quit,
    FilesDropped(Vec<std::path::PathBuf>),
    Library(pages::library::Msg),
    Sound(pages::sound::Msg),
    Filters(pages::filters::Msg),
    Room(pages::room::Msg),
    System(pages::system::Msg),
    Encore(pages::encore::Msg),
    Appearance(pages::appearance::Msg),
    Shot(crate::shot::Msg),
    Nothing,
}

pub struct App {
    pub engine: Engine,
    pub status: Arc<Status>,
    pub settings: Settings,
    pub page: Page,
    pub window: Option<window::Id>,
    pub width: f32,
    pub height: f32,
    pub appearance: Appearance,
    pub options: Value,
    pub filters: Vec<Value>,
    pub dithers: Vec<Value>,
    pub devices: Vec<DeviceInfo>,
    pub devices_busy: bool,
    pub covers: HashMap<String, Option<image::Handle>>,
    pub seek_drag: Option<f64>,
    /// The volume slider's value while it is held (the engine's status lags a little).
    pub vol_drag: Option<f64>,
    /// The tick the volume slider was let go on (it shows `vol_drag` until the engine agrees).
    vol_released: Option<u64>,
    /// Text boxes being typed in, by key (applied on Enter / leaving).
    pub texts: HashMap<String, String>,
    pub error: Option<String>,
    pub library: pages::library::State,
    pub sound: pages::sound::State,
    pub filters_page: pages::filters::State,
    pub room: pages::room::State,
    pub system: pages::system::State,
    pub encore: pages::encore::State,
    pub picker: pages::appearance::Picker,
    pub shot: Option<crate::shot::State>,
    pub quitting: bool,
    pub ticks: u64,
}

impl App {
    pub fn new(engine: Engine, page: Page) -> (Self, Task<Message>) {
        let status = engine.status();
        let settings = engine.settings();
        let appearance = crate::store::load_appearance();
        theme::set(&appearance);
        let options = cadenza::api::options(&settings);
        let filters = cadenza::api::filter_list(&Default::default());
        let dithers = cadenza_dsp::DitherKind::catalog()
            .iter()
            .map(|d| serde_json::json!({ "id": d.id(), "label": d.label() }))
            .collect();
        let mut app = Self {
            engine,
            status,
            settings,
            page,
            window: None,
            width: 1280.0,
            height: 820.0,
            appearance,
            options,
            filters,
            dithers,
            devices: Vec::new(),
            devices_busy: false,
            covers: HashMap::new(),
            seek_drag: None,
            vol_drag: None,
            vol_released: None,
            texts: HashMap::new(),
            error: None,
            library: Default::default(),
            sound: Default::default(),
            filters_page: Default::default(),
            room: Default::default(),
            system: Default::default(),
            encore: Default::default(),
            picker: Default::default(),
            shot: crate::shot::State::from_env(),
            quitting: false,
            ticks: 0,
        };
        let (open, task) = crate::platform::open_window();
        app.window = Some(open);
        let first = Task::batch([
            task.map(Message::WindowOpened),
            app.load_devices(false),
            pages::library::init(&mut app),
            pages::system::init(&mut app),
            pages::encore::init(&mut app),
            pages::room::init(&mut app),
        ]);
        (app, first)
    }

    pub fn title(&self, _: window::Id) -> String {
        match &self.status.track {
            Some(t) if self.status.state == PlayState::Playing && !t.title.is_empty() => {
                format!("{} – Cadenza", t.title)
            }
            _ => "Cadenza".into(),
        }
    }

    pub fn theme(&self, _: window::Id) -> iced::Theme {
        theme::look().theme()
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let mut subs = vec![
            iced::time::every(Duration::from_millis(200)).map(|_| Message::Tick),
            window::close_requests().map(Message::CloseRequested),
            window::close_events().map(Message::WindowClosed),
            window::resize_events().map(|(_, s)| Message::Resized(s)),
            iced::event::listen_with(|e, _, _| match e {
                iced::Event::Window(window::Event::FileDropped(p)) => {
                    Some(Message::FilesDropped(vec![p]))
                }
                _ => None,
            }),
            crate::platform::events(),
        ];
        if self.shot.is_some() {
            subs.push(
                iced::time::every(Duration::from_millis(900))
                    .map(|_| Message::Shot(crate::shot::Msg::Step)),
            );
        }
        Subscription::batch(subs)
    }

    pub fn load_devices(&mut self, reconnect: bool) -> Task<Message> {
        self.devices_busy = true;
        let engine = self.engine.clone();
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    let list = cadenza::output::list_devices();
                    if reconnect {
                        engine.send(Command::Reconnect);
                    }
                    list
                })
                .await
                .unwrap_or_default()
            },
            Message::Devices,
        )
    }

    /// Ask for a cover once; it shows when it arrives.
    pub fn want_cover(&mut self, url: &str) -> Task<Message> {
        if url.is_empty() || self.covers.contains_key(url) {
            return Task::none();
        }
        self.covers.insert(url.to_string(), None);
        let u = url.to_string();
        Task::perform(
            async move {
                let k = u.clone();
                let bytes = tokio::task::spawn_blocking(move || crate::covers::fetch(&k))
                    .await
                    .ok()
                    .flatten();
                (u, bytes)
            },
            |(u, b)| Message::Cover(u, b),
        )
    }

    pub fn cover(&self, url: Option<&str>) -> Option<image::Handle> {
        url.and_then(|u| self.covers.get(u).cloned().flatten())
    }

    pub fn text(&self, key: &str, current: impl FnOnce() -> String) -> String {
        self.texts.get(key).cloned().unwrap_or_else(current)
    }

    fn apply(&mut self, s: Settings) {
        match cadenza::api::validate(&s) {
            Ok(()) => {
                self.error = None;
                self.settings = s.clone();
                self.engine.update_settings(s);
            }
            Err(e) => self.error = Some(e),
        }
    }

    pub fn update(&mut self, msg: Message) -> Task<Message> {
        match msg {
            Message::Tick => {
                self.status = self.engine.status();
                // Settings changed elsewhere (phone, Encore) show up here; while a change
                // of ours is on its way the engine's copy catches up within a tick.
                self.settings = self.status.settings.clone();
                if let (Some(v), Some(t)) = (self.vol_drag, self.vol_released) {
                    if (self.status.volume_linear - v).abs() < 0.02 || self.ticks > t + 10 {
                        self.vol_drag = None;
                        self.vol_released = None;
                    }
                }
                let mut tasks = Vec::new();
                if let Some(c) = self.status.track.as_ref().and_then(|t| t.cover.clone()) {
                    tasks.push(self.want_cover(&c));
                }
                tasks.push(pages::sound::tick(self));
                self.ticks += 1;
                if self.ticks % 25 == 0 {
                    tasks.push(pages::encore::poll());
                }
                if self.page == Page::System && self.ticks % 10 == 0 {
                    tasks.push(pages::system::init(self));
                }
                tasks.push(crate::platform::sync(self));
                Task::batch(tasks)
            }
            Message::Nav(p) => {
                self.page = p;
                let top = iced::widget::operation::snap_to(
                    MAIN_SCROLL,
                    iced::widget::scrollable::RelativeOffset::START,
                );
                let t = match p {
                    Page::Output => self.load_devices(false),
                    Page::System => pages::system::init(self),
                    _ => Task::none(),
                };
                Task::batch([top, t])
            }
            Message::Transport(t) => {
                self.engine.send(t.command());
                Task::none()
            }
            Message::SeekDrag(v) => {
                self.seek_drag = Some(v);
                Task::none()
            }
            Message::SeekDone => {
                if let Some(v) = self.seek_drag.take() {
                    self.engine.send(Command::Seek(v));
                }
                Task::none()
            }
            Message::VolumeDrag(v) => {
                self.vol_drag = Some(v);
                self.vol_released = None;
                self.engine.send(Transport::Volume(v).command());
                Task::none()
            }
            Message::VolumeDone => {
                self.vol_released = Some(self.ticks);
                Task::none()
            }
            Message::Edit(Edit(f)) => {
                let mut s = self.settings.clone();
                f(&mut s);
                self.apply(s);
                Task::none()
            }
            Message::Patch(p) => {
                let mut v = serde_json::to_value(&self.settings).unwrap_or(Value::Null);
                merge(&mut v, &p);
                match serde_json::from_value::<Settings>(v) {
                    Ok(s) => self.apply(s),
                    Err(e) => self.error = Some(format!("bad setting: {e}")),
                }
                Task::none()
            }
            Message::Text(k, v) => {
                self.texts.insert(k, v);
                Task::none()
            }
            Message::TextCommit(k) => {
                let Some(v) = self.texts.remove(&k) else {
                    return Task::none();
                };
                pages::commit_text(self, &k, v)
            }
            Message::RefreshDevices(reconnect) => self.load_devices(reconnect),
            Message::Devices(d) => {
                self.devices = d;
                self.devices_busy = false;
                Task::none()
            }
            Message::Cover(u, b) => {
                self.covers.insert(u, b.map(image::Handle::from_bytes));
                Task::none()
            }
            Message::SetAppearance(a) => {
                theme::set(&a);
                crate::store::save_appearance(&a);
                self.appearance = a;
                Task::none()
            }
            Message::WindowOpened(id) => {
                self.window = Some(id);
                crate::platform::start(self)
            }
            Message::CloseRequested(id) => {
                // Closing the window keeps the music playing; Quit is in the tray.
                crate::store::save_window(self);
                if self.quitting || !crate::platform::tray_available() {
                    return iced::exit();
                }
                self.window = None;
                window::close(id)
            }
            Message::WindowClosed(id) => {
                if self.window == Some(id) {
                    self.window = None;
                }
                Task::none()
            }
            Message::Resized(s) => {
                self.width = s.width;
                self.height = s.height;
                Task::none()
            }
            Message::Raise => match self.window {
                Some(id) => Task::batch([window::minimize(id, false), window::gain_focus(id)]),
                None => {
                    let (id, task) = crate::platform::open_window();
                    self.window = Some(id);
                    task.map(Message::WindowOpened)
                }
            },
            Message::Quit => {
                self.quitting = true;
                crate::store::save_window(self);
                self.engine.send(Command::Stop);
                iced::exit()
            }
            Message::FilesDropped(paths) => {
                let engine = self.engine.clone();
                Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            cadenza::library::play_dropped(&engine, &paths)
                        })
                        .await
                        .ok();
                    },
                    |_| Message::Nothing,
                )
            }
            Message::Library(m) => pages::library::update(self, m),
            Message::Sound(m) => pages::sound::update(self, m),
            Message::Filters(m) => pages::filters::update(self, m),
            Message::Room(m) => pages::room::update(self, m),
            Message::Appearance(m) => {
                pages::appearance::update(self, m);
                Task::none()
            }
            Message::System(m) => pages::system::update(self, m),
            Message::Encore(m) => pages::encore::update(self, m),
            Message::Shot(m) => crate::shot::update(self, m),
            Message::Nothing => Task::none(),
        }
    }

    pub fn view(&self, _: window::Id) -> Element<'_, Message> {
        let wide = self.width >= 1100.0;
        let page: El = match self.page {
            Page::Now => pages::now::view(self),
            Page::Library => pages::library::view(self, wide),
            Page::Guide => pages::guide::view(self, wide),
            Page::Sound => pages::sound::view(self, wide),
            Page::Filters => pages::filters::view(self, wide),
            Page::Output => pages::output::view(self, wide),
            Page::Room => pages::room::view(self, wide),
            Page::System => pages::system::view(self, wide),
            Page::Encore => pages::encore::view(self, wide),
            Page::Appearance => pages::appearance::view(self, wide),
        };
        let main = scrollable(container(page).padding([26, 32]).width(Length::Fill))
            .id(MAIN_SCROLL)
            .style(theme::scroll)
            .height(Length::Fill);
        let body: El = if self.width >= 760.0 {
            row![self.sidebar(), main].into()
        } else {
            column![self.topnav(), main].into()
        };
        container(column![container(body).height(Length::Fill), self.player()])
            .style(theme::app_bg)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn nav_badge(&self, p: Page) -> Option<(String, theme::Badge)> {
        match p {
            Page::System => Some(match self.system.on() {
                true => ("on".into(), theme::Badge::Ok),
                false => ("off".into(), theme::Badge::Plain),
            }),
            Page::Encore => Some(self.encore.badge()),
            _ => None,
        }
    }

    fn nav_button(&self, p: Page, compact: bool) -> El<'_> {
        let l = theme::look();
        let on = self.page == p;
        let mut r = row![
            icon(p.icon(), 18.0, if on { l.accent } else { l.muted }),
            text(p.name()).size(14.5),
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        if !compact {
            if let Some((t, k)) = self.nav_badge(p) {
                r = r.push(space().width(Length::Fill)).push(ui::badge(t, k));
            }
        }
        button(r)
            .padding([9, 12])
            .width(if compact {
                Length::Shrink
            } else {
                Length::Fill
            })
            .style(theme::btn(if on { Btn::NavOn } else { Btn::Nav }))
            .on_press(Message::Nav(p))
            .into()
    }

    fn sidebar(&self) -> El<'_> {
        let l = theme::look();
        let section = |t: &'static str| -> El<'_> {
            container(text(t).size(11).color(l.faint).font(ui::bold()))
                .padding([10, 12])
                .into()
        };
        let mut nav = column![].spacing(2);
        for p in [Page::Now, Page::Library, Page::Guide] {
            nav = nav.push(self.nav_button(p, false));
        }
        nav = nav.push(section("ENGINE"));
        for p in [Page::Sound, Page::Filters, Page::Output] {
            nav = nav.push(self.nav_button(p, false));
        }
        nav = nav.push(section("TOOLS"));
        for p in [Page::Room, Page::System, Page::Encore, Page::Appearance] {
            nav = nav.push(self.nav_button(p, false));
        }
        let brand = row![
            container(icon(Icon::Music, 18.0, l.on_accent))
                .width(34)
                .height(34)
                .center(34)
                .style(theme::accent_tile),
            text("Cadenza").size(19).font(ui::bold()),
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        let gpu = self.options["gpu"]["label"]
            .as_str()
            .unwrap_or("CPU")
            .to_string();
        let out = if self.status.output.is_empty() {
            String::new()
        } else {
            format!(
                "{}{}",
                self.status.output,
                if self.status.output_exclusive {
                    " · exclusive"
                } else {
                    ""
                }
            )
        };
        let foot = column![
            ui::badge(gpu, theme::Badge::Accent),
            text(out).size(12).color(l.muted),
            text(concat!("Cadenza ", env!("CARGO_PKG_VERSION")))
                .size(11.5)
                .color(l.muted),
        ]
        .spacing(6);
        container(
            column![
                container(brand).padding([6, 10]),
                scrollable(nav).style(theme::scroll).height(Length::Fill),
                foot
            ]
            .spacing(18),
        )
        .padding(14)
        .width(236)
        .height(Length::Fill)
        .style(theme::sidebar)
        .into()
    }

    fn topnav(&self) -> El<'_> {
        let mut r = row![].spacing(4);
        for p in Page::ALL {
            r = r.push(self.nav_button(p, true));
        }
        container(scrollable(r).direction(scrollable::Direction::Horizontal(
            scrollable::Scrollbar::new().width(0).scroller_width(0),
        )))
        .padding(8)
        .style(theme::sidebar)
        .width(Length::Fill)
        .into()
    }

    fn player(&self) -> El<'_> {
        let l = theme::look();
        let st = &self.status;
        let t = st.track.as_ref();
        let thumb: El = match self.cover(t.and_then(|t| t.cover.as_deref())) {
            Some(h) => container(image(h).width(52).height(52))
                .style(theme::inset)
                .into(),
            None => container(icon(Icon::Music, 24.0, l.on_accent))
                .width(52)
                .height(52)
                .center(52)
                .style(theme::accent_tile)
                .into(),
        };
        let title = t
            .map(|t| {
                if t.title.is_empty() {
                    "Untitled".to_string()
                } else {
                    t.title.clone()
                }
            })
            .unwrap_or_else(|| "Nothing playing".into());
        let sub = t
            .map(|t| {
                [t.artist.as_str(), t.album.as_str()]
                    .iter()
                    .filter(|x| !x.is_empty())
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" · ")
            })
            .unwrap_or_default();
        let mut info = column![text(title).size(15).font(ui::bold())].spacing(2);
        if !sub.is_empty() {
            info = info.push(text(sub).size(12.5).color(l.muted));
        }
        if let Some(e) = &st.error {
            info = info.push(text(e.clone()).size(12).color(l.err));
        }
        let now = row![thumb, info]
            .spacing(12)
            .align_y(Alignment::Center)
            .width(Length::FillPortion(3));

        let icon_btn = |i: Icon, tr: Transport| -> El<'_> {
            button(icon(i, 20.0, l.text))
                .padding([8, 10])
                .style(theme::btn(Btn::Ghost))
                .on_press(Message::Transport(tr))
                .into()
        };
        let playing = st.state == PlayState::Playing;
        let toggle = button(
            container(icon(
                if playing { Icon::Pause } else { Icon::Play },
                20.0,
                l.on_accent,
            ))
            .center_x(44)
            .center_y(44),
        )
        .padding(0)
        .style(theme::play_btn)
        .on_press(Message::Transport(Transport::Toggle));
        let dur = st.duration.unwrap_or(0.0);
        let pos = self.seek_drag.unwrap_or(st.position);
        let seek = row![
            text(ui::mmss(pos)).size(12).color(l.muted),
            slider(
                0.0..=dur.max(0.001),
                pos.min(dur.max(0.001)),
                Message::SeekDrag
            )
            .on_release(Message::SeekDone)
            .step(0.1)
            .style(theme::range),
            text(ui::mmss(dur)).size(12).color(l.muted),
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        let ctrls = column![
            row![
                icon_btn(Icon::Prev, Transport::Prev),
                toggle,
                icon_btn(Icon::Next, Transport::Next),
                icon_btn(Icon::Stop, Transport::Stop)
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            seek
        ]
        .spacing(6)
        .align_x(Alignment::Center)
        .width(Length::FillPortion(4));
        let vol = self.vol_drag.unwrap_or(st.volume_linear);
        let vol = row![
            icon(
                if st.volume_linear <= 0.0 {
                    Icon::Mute
                } else {
                    Icon::Volume
                },
                20.0,
                l.muted,
            ),
            slider(0.0..=1.0, vol, Message::VolumeDrag)
                .on_release(Message::VolumeDone)
                .step(0.01)
                .style(theme::range)
                .width(150)
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        let vol = container(vol).align_right(Length::FillPortion(2));
        container(row![now, ctrls, vol].spacing(18).align_y(Alignment::Center))
            .padding([12, 18])
            .width(Length::Fill)
            .style(theme::player)
            .into()
    }
}

/// The page's scroll area (scrolled back to the top on a page change).
pub const MAIN_SCROLL: &str = "main";

/// Deep-merge `patch` into `v` (objects merge, everything else replaces).
pub fn merge(v: &mut Value, patch: &Value) {
    match (v, patch) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, pv) in b {
                merge(a.entry(k.clone()).or_insert(Value::Null), pv);
            }
        }
        (a, b) => *a = b.clone(),
    }
}
