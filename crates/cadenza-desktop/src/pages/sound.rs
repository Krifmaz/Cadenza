//! Sound: the upsampling filter, Find my limit, noise shaping, DSD, headphones and
//! convolution, the parametric EQ (graph + band boxes) and the matrix.

use crate::app::{edit, patch, App, Message};
use crate::theme::{self, Badge, Btn};
use crate::ui::{self, Choice, El};
use cadenza::engine::Command;
use cadenza::limit::Progress;
use cadenza_dsp::eq::{Band, BandKind, Equalizer};
use iced::mouse;
use iced::widget::canvas::{self, Frame, Geometry, Path, Stroke, Text};
use iced::widget::{button, column, container, row, scrollable, text, Column, Row};
use iced::{Alignment, Color, Length, Point, Rectangle, Renderer, Size, Task, Theme};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct State {
    limit: Arc<Mutex<Progress>>,
    limit_cancel: Arc<AtomicBool>,
    pub limit_view: Progress,
    eq_sel: Option<usize>,
    eq_range: f32,
    /// While dragging a band: the bands as dragged, and when the engine last got them.
    eq_live: Option<(Vec<Band>, Instant)>,
    noise_busy: bool,
    noise_msg: String,
}

impl Default for State {
    fn default() -> Self {
        Self {
            limit: Arc::default(),
            limit_cancel: Arc::default(),
            limit_view: Progress::default(),
            eq_sel: None,
            eq_range: 12.0,
            eq_live: None,
            noise_busy: false,
            noise_msg: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Msg {
    LimitStart,
    LimitCancel,
    LimitUse(u32, String),
    EqSelect(Option<usize>),
    EqDrag(usize, f64, f64),
    EqDragEnd,
    EqWheel(usize, f32),
    EqAdd(f64, f64),
    EqRemove(usize),
    EqFlat,
    EqClear,
    EqRange(f32),
    EqKind(usize, String),
    EqSlope(usize, u32),
    EqEnabled(usize, bool),
    NoiseMeasure,
    NoiseDone(Result<serde_json::Value, String>),
}

const EQ_COLORS: [&str; 8] = [
    "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6", "#a855f7", "#ec4899", "#ef4444",
];
const EQ_KINDS: [(&str, &str); 6] = [
    ("high-pass", "Low cut"),
    ("low-shelf", "Low shelf"),
    ("peaking", "Bell"),
    ("notch", "Notch"),
    ("high-shelf", "High shelf"),
    ("low-pass", "High cut"),
];
/// The EQ graph is drawn at this rate (the engine runs the EQ at the source's rate; the
/// curves only differ near its top).
const EQ_RATE: u32 = 48_000;
const FMIN: f64 = 20.0;
const FMAX: f64 = 20_000.0;

fn eq_color(i: usize) -> Color {
    theme::hex(EQ_COLORS[i % EQ_COLORS.len()])
}

fn kind_id(k: BandKind) -> &'static str {
    match k {
        BandKind::Peaking => "peaking",
        BandKind::LowShelf => "low-shelf",
        BandKind::HighShelf => "high-shelf",
        BandKind::LowPass => "low-pass",
        BandKind::HighPass => "high-pass",
        BandKind::Notch => "notch",
    }
}

fn kind_from(id: &str) -> BandKind {
    match id {
        "low-shelf" => BandKind::LowShelf,
        "high-shelf" => BandKind::HighShelf,
        "low-pass" => BandKind::LowPass,
        "high-pass" => BandKind::HighPass,
        "notch" => BandKind::Notch,
        _ => BandKind::Peaking,
    }
}

fn has_gain(k: BandKind) -> bool {
    matches!(
        k,
        BandKind::Peaking | BandKind::LowShelf | BandKind::HighShelf
    )
}

fn hz_label(f: f64) -> String {
    if f >= 1000.0 {
        let k = f / 1000.0;
        let s = if f >= 10_000.0 {
            format!("{k:.1}")
        } else {
            format!("{k:.2}")
        };
        format!("{}k", s.trim_end_matches('0').trim_end_matches('.'))
    } else {
        format!("{}", f.round())
    }
}

/// `1.2k`, `1200`, `1200 Hz` → 1200.
fn parse_hz(s: &str) -> Option<f64> {
    let t = s.trim().to_lowercase();
    let t = t.trim_end_matches("hz").trim();
    let (num, mul) = match t.strip_suffix('k') {
        Some(n) => (n.trim(), 1000.0),
        None => (t, 1.0),
    };
    num.parse::<f64>()
        .ok()
        .map(|v| v * mul)
        .filter(|v| v.is_finite() && *v > 0.0)
}

fn num(s: &str) -> Option<f64> {
    s.trim()
        .trim_end_matches("dB")
        .trim_end_matches("db")
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
}

/// The bands the page shows: the dragged copy while a drag runs, else the settings'.
fn bands(app: &App) -> Vec<Band> {
    match &app.sound.eq_live {
        Some((b, _)) => b.clone(),
        None => app.settings.dsp.eq.clone(),
    }
}

fn set_bands(app: &mut App, b: Vec<Band>) {
    let _ = app.update(edit(move |s| s.dsp.eq = b.clone()));
}

pub fn tick(app: &mut App) -> Task<Message> {
    let p = app
        .sound
        .limit
        .lock()
        .map(|p| p.clone())
        .unwrap_or_default();
    app.sound.limit_view = p;
    Task::none()
}

pub fn update(app: &mut App, m: Msg) -> Task<Message> {
    match m {
        Msg::LimitStart => {
            if app.sound.limit_view.state == "running" {
                return Task::none();
            }
            // The test needs the whole machine and must not hold the device half-open.
            app.engine.send(Command::Release);
            app.sound.limit_cancel.store(false, Ordering::Relaxed);
            *app.sound.limit.lock().unwrap() = Progress {
                state: "running".into(),
                current: "looking at the device".into(),
                ..Default::default()
            };
            let (engine, limit, cancel) = (
                app.engine.clone(),
                Arc::clone(&app.sound.limit),
                Arc::clone(&app.sound.limit_cancel),
            );
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                let settings = engine.settings();
                let device =
                    cadenza::limit::find_device(&settings, &cadenza::output::list_devices());
                let r = cadenza::limit::run(&settings, device, &engine.cache(), &cancel, |p| {
                    *limit.lock().unwrap() = p.clone();
                });
                let mut p = limit.lock().unwrap();
                match r {
                    Ok(report) => {
                        p.done = p.total;
                        p.current.clear();
                        p.report = report;
                        p.state = "done".into();
                    }
                    Err(e) if e == "cancelled" => p.state = "cancelled".into(),
                    Err(e) => {
                        p.state = "error".into();
                        p.error = Some(e);
                    }
                }
            });
            tick(app)
        }
        Msg::LimitCancel => {
            app.sound.limit_cancel.store(true, Ordering::Relaxed);
            Task::none()
        }
        Msg::LimitUse(rate, filter) => app.update(edit(move |s| {
            s.dsp.filter = filter.clone();
            if rate > 0 {
                s.output.rate = cadenza::config::RateSetting::Fixed(rate);
            }
        })),
        Msg::EqSelect(i) => {
            app.sound.eq_sel = i;
            Task::none()
        }
        Msg::EqDrag(i, f, g) => {
            let mut b = bands(app);
            if let Some(band) = b.get_mut(i) {
                band.freq = (f * 10.0).round() / 10.0;
                if has_gain(band.kind) {
                    band.gain_db = (g * 10.0).round() / 10.0;
                }
            }
            app.sound.eq_sel = Some(i);
            // At most ~15 updates a second to the engine while dragging.
            let send = app
                .sound
                .eq_live
                .as_ref()
                .is_none_or(|(_, t)| t.elapsed() > Duration::from_millis(66));
            if send {
                let s = b.clone();
                let _ = app.update(edit(move |x| x.dsp.eq = s.clone()));
                app.sound.eq_live = Some((b, Instant::now()));
            } else if let Some((live, _)) = app.sound.eq_live.as_mut() {
                *live = b;
            }
            Task::none()
        }
        Msg::EqDragEnd => {
            if let Some((b, _)) = app.sound.eq_live.take() {
                set_bands(app, b);
            }
            Task::none()
        }
        Msg::EqWheel(i, dy) => {
            let mut b = bands(app);
            if let Some(band) = b.get_mut(i) {
                let f = if dy > 0.0 { 1.1 } else { 1.0 / 1.1 };
                band.q = ((band.q * f).clamp(0.1, 20.0) * 100.0).round() / 100.0;
            }
            app.sound.eq_sel = Some(i);
            set_bands(app, b);
            Task::none()
        }
        Msg::EqAdd(f, g) => {
            let mut b = bands(app);
            if b.len() < cadenza_dsp::eq::MAX_BANDS {
                b.push(Band {
                    kind: BandKind::Peaking,
                    freq: f.round(),
                    gain_db: (g * 10.0).round() / 10.0,
                    q: 1.0,
                    enabled: true,
                    slope_db: 12,
                });
                app.sound.eq_sel = Some(b.len() - 1);
                set_bands(app, b);
            }
            Task::none()
        }
        Msg::EqRemove(i) => {
            let mut b = bands(app);
            if i < b.len() {
                b.remove(i);
            }
            app.sound.eq_sel = None;
            set_bands(app, b);
            Task::none()
        }
        Msg::EqFlat => {
            let mut b = bands(app);
            for band in &mut b {
                if has_gain(band.kind) {
                    band.gain_db = 0.0;
                }
            }
            set_bands(app, b);
            Task::none()
        }
        Msg::EqClear => {
            app.sound.eq_sel = None;
            set_bands(app, Vec::new());
            Task::none()
        }
        Msg::EqRange(r) => {
            app.sound.eq_range = r;
            Task::none()
        }
        Msg::EqKind(i, k) => {
            let mut b = bands(app);
            if let Some(band) = b.get_mut(i) {
                band.kind = kind_from(&k);
            }
            set_bands(app, b);
            Task::none()
        }
        Msg::EqSlope(i, s) => {
            let mut b = bands(app);
            if let Some(band) = b.get_mut(i) {
                band.slope_db = s;
            }
            set_bands(app, b);
            Task::none()
        }
        Msg::EqEnabled(i, on) => {
            let mut b = bands(app);
            if let Some(band) = b.get_mut(i) {
                band.enabled = on;
            }
            set_bands(app, b);
            Task::none()
        }
        Msg::NoiseMeasure => {
            app.sound.noise_busy = true;
            app.sound.noise_msg = "Measuring: a −20 dBFS tone, then silence…".into();
            let (room, engine) = (Arc::clone(&app.room.room), app.engine.clone());
            let input = app.room.input.clone();
            Task::perform(
                async move {
                    tokio::task::spawn_blocking(move || {
                        let mut r = room.lock().unwrap_or_else(|e| e.into_inner());
                        r.measure_noise(&engine, &input)
                            .map(|m| serde_json::to_value(m).unwrap_or_default())
                    })
                    .await
                    .unwrap_or_else(|e| Err(e.to_string()))
                },
                |r| Message::Sound(Msg::NoiseDone(r)),
            )
        }
        Msg::NoiseDone(r) => {
            app.sound.noise_busy = false;
            app.sound.noise_msg = match r {
                Ok(m) => format!(
                    "Measured: noise floor {:.1} dBFS (A-weighted). Saved for Auto.",
                    m["floor_db"].as_f64().unwrap_or(f64::NAN)
                ),
                Err(e) => e,
            };
            Task::none()
        }
    }
}

/// Text boxes on this page: `snd_*` settings and `eq_{f,g,q}{i}` band values.
pub fn commit(app: &mut App, k: &str, v: String) -> Task<Message> {
    if let Some(rest) = k.strip_prefix("eq_") {
        let (what, idx) = rest.split_at(1);
        let Ok(i) = idx.parse::<usize>() else {
            return Task::none();
        };
        let mut b = bands(app);
        if let Some(band) = b.get_mut(i) {
            match what {
                "f" => {
                    if let Some(f) = parse_hz(&v) {
                        band.freq = f.clamp(10.0, 24_000.0);
                    }
                }
                "g" => {
                    if let Some(g) = num(&v) {
                        band.gain_db = g.clamp(-24.0, 24.0);
                    }
                }
                "q" => {
                    if let Some(q) = num(&v) {
                        band.q = q.clamp(0.1, 20.0);
                    }
                }
                _ => {}
            }
        }
        app.sound.eq_sel = Some(i);
        set_bands(app, b);
        return Task::none();
    }
    let v = v.trim().to_string();
    match k {
        "snd_headroom" => {
            let h = if v.is_empty() {
                Some(None)
            } else {
                num(&v).map(Some)
            };
            if let Some(h) = h {
                return app.update(edit(move |s| {
                    s.dsp.headroom_db = h.map(|x| x.clamp(-30.0, 0.0))
                }));
            }
        }
        "snd_dac_floor" => {
            let h = if v.is_empty() {
                Some(None)
            } else {
                num(&v).map(Some)
            };
            if let Some(h) = h {
                return app.update(edit(move |s| s.dsp.dac_noise_floor_db = h));
            }
        }
        "snd_dsd_gain" => {
            if let Some(g) = num(&v) {
                return app.update(edit(move |s| s.dsp.dsd_gain_db = g.clamp(-30.0, 6.0)));
            }
        }
        "snd_convolution" => return app.update(edit(move |s| s.dsp.convolution = v.clone())),
        "snd_balance" => {
            if let Some(b) = num(&v) {
                return app.update(patch(
                    json!({ "matrix": { "stereo": { "balance_db": b } } }),
                ));
            }
        }
        "snd_matrix" => match serde_json::from_str::<serde_json::Value>(&v) {
            Ok(outputs) => return app.update(patch(json!({ "matrix": { "outputs": outputs } }))),
            Err(e) => app.error = Some(format!("matrix outputs: {e}")),
        },
        _ => {}
    }
    Task::none()
}

fn txt(app: &App, key: &str, current: impl FnOnce() -> String) -> String {
    app.text(key, current)
}

fn text_box<'a>(
    app: &App,
    key: impl Into<String>,
    placeholder: &str,
    current: impl FnOnce() -> String,
) -> El<'a> {
    let key: String = key.into();
    let v = txt(app, &key, current);
    let k2 = key.clone();
    ui::input(placeholder, &v, move |s| Message::Text(k2.clone(), s))
        .on_submit(Message::TextCommit(key))
        .into()
}

/// The filter families in plain words, in the order the picker shows them.
const FAMILIES: [(&str, &str); 6] = [
    ("sinc", "The classic upsampling filter"),
    (
        "poly-sinc-gauss",
        "The gentlest filter, with the least ringing",
    ),
    ("poly-sinc-ext3", "A short, light filter for slower PCs"),
    ("closed-form", "The sharpest cut-off for its length"),
    (
        "sinc-mx",
        "A long filter that keeps the same shape at every rate",
    ),
    ("sinc-xtr", "An extremely long filter for strong PCs"),
];

fn filter_plain(id: &str) -> String {
    let fam_of = FAMILIES;
    if id.starts_with("custom") {
        return "A filter you made in the filter designer.".into();
    }
    let Some((pre, fam)) = fam_of
        .iter()
        .find(|(p, _)| id.starts_with(&format!("{p}-")))
    else {
        return String::new();
    };
    let rest = &id[pre.len() + 1..];
    let (ph, len) = rest.split_once('-').unwrap_or((rest, ""));
    let phase = match ph {
        "l" => "linear phase (most accurate timing; the usual choice)",
        "m" => "minimum phase (no ringing before sounds)",
        "la" => "linear phase, apodizing (also cleans up ringing baked into older recordings)",
        "ma" => "minimum phase, apodizing (no pre-ringing, cleans up older recordings)",
        _ => "",
    };
    let big = len.ends_with('m') || len.contains("long") || len.contains("xla");
    format!(
        "In plain words: {fam}, {phase}. {} Run \"Find my limit\" to see how long a filter yours handles. More in the Guide.",
        if big { "Very long: needs a capable PC, and waits a while before the first sound." } else { "Light on the PC." }
    )
}

/// The DSD card (on the Output page, next to the output mode that turns DSD on).
pub fn dsd_card(app: &App) -> El<'_> {
    let d = &app.settings.dsp;
    let caps = &app.status.output_dsd;
    let mut rates = Vec::new();
    for r in app.options["dsd_rates"]
        .as_array()
        .cloned()
        .unwrap_or_default()
    {
        let id = r["id"].as_str().unwrap_or("").to_string();
        let hz = r["hz_44"].as_u64().unwrap_or(0) as u32;
        let ok = caps.max_rate.is_none_or(|m| hz <= m) || app.status.output.is_empty();
        let label = format!(
            "{} · {:.4} MHz{}",
            r["label"].as_str().unwrap_or(""),
            hz as f64 / 1e6,
            if ok { "" } else { " (above this device)" }
        );
        rates.push(Choice::new(id, label));
    }
    let dsd_on = d.output_mode == cadenza_dsp::dsd::OutputMode::Dsd;
    let dsd = column![
        row![ui::h2("DSD"), ui::badge(if dsd_on { "DSD mode" } else { "PCM mode" }, if dsd_on { Badge::Accent } else { Badge::Plain })]
            .spacing(8)
            .align_y(Alignment::Center),
        row![
            ui::field("Rate", ui::pick(rates, &d.dsd_rate.id(), |v| patch(json!({ "dsp": { "dsd_rate": v } })))),
            ui::field(
                "Modulator",
                ui::pick(
                    app.options["dsd_modulators"]
                        .as_array()
                        .map(|a| a
                            .iter()
                            .map(|m| Choice::new(
                                m["id"].as_str().unwrap_or(""),
                                format!(
                                    "{} · order {}{}",
                                    m["label"].as_str().unwrap_or(""),
                                    m["order"].as_u64().unwrap_or(0),
                                    if m["dithered"].as_bool() == Some(true) { " · dithered" } else { "" }
                                )
                            ))
                            .collect())
                        .unwrap_or_default(),
                    d.dsd_modulator.id(),
                    |v| patch(json!({ "dsp": { "dsd_modulator": v } }))
                )
            ),
        ]
        .spacing(12),
        row![
            ui::field(
                "Packing",
                ui::pick(ui::choices(&app.options["dsd_packings"], "id", "label"), d.dsd_packing.id(), |v| patch(json!({ "dsp": { "dsd_packing": v } })))
            ),
            ui::field("Gain before the modulator (dB)", text_box(app, "snd_dsd_gain", "-3", || d.dsd_gain_db.to_string())),
        ]
        .spacing(12),
        ui::field(
            "Oversampling",
            ui::pick(ui::choices(&app.options["dsd_oversampling"], "id", "id"), d.dsd_oversampling.id(), |v| patch(json!({ "dsp": { "dsd_oversampling": v } })))
        ),
        ui::switch(d.dsd_passthrough, "DSD pass-through", |on| edit(move |s| s.dsp.dsd_passthrough = on)),
        ui::hint("DSF/DFF bits go to a DSD output untouched when the output mode is DSD (or bit-perfect) and the rate matches; in PCM mode they are converted to 352.8/384 kHz PCM. Packing: a driver that takes native DSD (ASIO, most FiiO / iFi / Topping) always gets native; DoP is for the rest, and only up to DSD256."),
    ]
    .spacing(12);
    ui::card(dsd)
}

fn limit_card(app: &App) -> El<'_> {
    let l = theme::look();
    let p = &app.sound.limit_view;
    let running = p.state == "running";
    let r = &p.report;
    let mut c = Column::new()
        .spacing(10)
        .push(ui::h2("Find my limit"))
        .push(ui::hint("Tests how long a filter this computer can run at every rate your device takes, without the sound breaking up. Takes a few minutes (very long filters are designed the first time). The music pauses and the device is let go during the test; press play afterwards to carry on."));
    let mut buttons = row![ui::btn(
        "Run the test",
        Btn::Primary,
        (!running).then_some(Message::Sound(Msg::LimitStart))
    )]
    .spacing(8);
    if running {
        buttons = buttons.push(ui::btn(
            "Cancel",
            Btn::Secondary,
            Some(Message::Sound(Msg::LimitCancel)),
        ));
    }
    c = c.push(buttons);
    if running {
        let frac = if p.total > 0 {
            p.done as f32 / p.total as f32
        } else {
            0.03
        };
        c = c
            .push(
                iced::widget::progress_bar(0.0..=1.0, frac)
                    .girth(6)
                    .style(move |_| iced::widget::progress_bar::Style {
                        background: iced::Background::Color(l.panel2),
                        bar: iced::Background::Color(l.accent),
                        border: iced::Border::default().rounded(3),
                    }),
            )
            .push(ui::hint(format!(
                "Testing {}… ({}/{})",
                p.current,
                p.done,
                p.total.max(1)
            )));
    } else if !r.output.is_empty() {
        c = c.push(ui::hint(format!(
            "{} · {} · {}",
            r.output, r.conversion, r.compute
        )));
    }
    match p.state.as_str() {
        "error" => {
            c = c.push(ui::err(
                p.error.clone().unwrap_or_else(|| "The test failed.".into()),
            ))
        }
        "cancelled" => c = c.push(ui::hint("Cancelled.")),
        "done" => c = c.push(ui::small(r.summary.clone())),
        _ => {}
    }
    if !r.columns.is_empty() {
        let mut t = Column::new().spacing(4).push(
            row![
                text("Rate")
                    .size(12)
                    .color(l.muted)
                    .width(Length::FillPortion(2)),
                text("Longest safe filter")
                    .size(12)
                    .color(l.muted)
                    .width(Length::FillPortion(4)),
                text("Speed")
                    .size(12)
                    .color(l.muted)
                    .width(Length::FillPortion(1)),
                text("").width(70)
            ]
            .spacing(8),
        );
        for col in &r.columns {
            let best = col
                .rows
                .iter()
                .find(|x| Some(&x.filter) == col.best.as_ref());
            let mut rate = row![text(col.label.clone()).size(13.5)]
                .spacing(6)
                .align_y(Alignment::Center);
            if col.current {
                rate = rate.push(ui::badge("now", Badge::Accent));
            }
            let cell: El = match (&col.best, best) {
                (Some(b), Some(x)) => row![
                    text(b.clone()).size(13.5).font(ui::bold()),
                    text(format!("{} taps", taps_label(x.taps)))
                        .size(12)
                        .color(l.muted)
                ]
                .spacing(6)
                .into(),
                _ if p.state == "done" => ui::badge("none is safe", Badge::Bad),
                _ => ui::hint("testing…"),
            };
            let use_btn: El = match (&col.best, p.state.as_str()) {
                (Some(b), "done") => ui::btn(
                    "Use",
                    Btn::Secondary,
                    Some(Message::Sound(Msg::LimitUse(col.rate, b.clone()))),
                ),
                _ => text("").into(),
            };
            t = t.push(
                container(
                    row![
                        container(rate).width(Length::FillPortion(2)),
                        container(cell).width(Length::FillPortion(4)),
                        text(best.map(|x| format!("{:.1}×", x.speed)).unwrap_or_default())
                            .size(13)
                            .width(Length::FillPortion(1)),
                        container(use_btn).width(70),
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                )
                .padding([6, 8])
                .style(theme::inset),
            );
        }
        c = c.push(t);
    }
    ui::card(c)
}

fn taps_label(t: usize) -> String {
    if t >= 1 << 20 {
        format!("{}M", t >> 20)
    } else {
        format!("{}k", t >> 10)
    }
}

pub fn view(app: &App, wide: bool) -> El<'_> {
    let s = &app.settings;
    let d = &s.dsp;

    // Upsampling: the family first, then that family's filters (a hundred in one list
    // is too many to read).
    let family_of = |id: &str| {
        app.filters
            .iter()
            .find(|f| f["id"] == json!(id))
            .and_then(|f| f["family"].as_str())
            .unwrap_or(if id.starts_with("custom") {
                "custom"
            } else {
                ""
            })
            .to_string()
    };
    let fam = family_of(&d.filter);
    let mut fams: Vec<String> = Vec::new();
    for f in &app.filters {
        let f = f["family"].as_str().unwrap_or("").to_string();
        if !f.is_empty() && !fams.contains(&f) {
            fams.push(f);
        }
    }
    fams.sort_by_key(|f| FAMILIES.iter().position(|(id, _)| id == f).unwrap_or(99));
    if !fams.contains(&fam) {
        fams.push(fam.clone());
    }
    let fam_choices: Vec<Choice> = fams
        .iter()
        .map(|f| {
            let what = FAMILIES
                .iter()
                .find(|(id, _)| id == f)
                .map(|(_, w)| w.to_lowercase())
                .unwrap_or_else(|| {
                    if f == "custom" {
                        "made in the filter designer".into()
                    } else {
                        String::new()
                    }
                });
            Choice::new(
                f.clone(),
                if what.is_empty() {
                    f.clone()
                } else {
                    format!("{f}: {what}")
                },
            )
        })
        .collect();
    // Switching family keeps the phase (and apodizing) when the new family has it.
    let kind = |label: &str| (label.contains("minimum"), label.contains("apodizing"));
    let current_kind = app
        .filters
        .iter()
        .find(|f| f["id"] == json!(d.filter))
        .map(|f| kind(f["label"].as_str().unwrap_or("")))
        .unwrap_or((false, false));
    let catalogue = app.filters.clone();
    let family_pick = ui::pick(fam_choices, &fam, move |f| {
        let in_fam: Vec<&serde_json::Value> = catalogue
            .iter()
            .filter(|x| x["family"] == json!(f))
            .collect();
        let to = in_fam
            .iter()
            .find(|x| kind(x["label"].as_str().unwrap_or("")) == current_kind)
            .or(in_fam.first())
            .and_then(|x| x["id"].as_str())
            .map(str::to_string);
        match to {
            Some(id) => edit(move |s| s.dsp.filter = id.clone()),
            None => Message::Nothing,
        }
    });
    let mut filters: Vec<Choice> = app
        .filters
        .iter()
        .filter(|f| f["family"] == json!(fam))
        .map(|f| {
            Choice::new(
                f["id"].as_str().unwrap_or(""),
                f["label"].as_str().unwrap_or(""),
            )
        })
        .collect();
    if !filters.iter().any(|f| f.id == d.filter) {
        filters.insert(0, Choice::new(d.filter.clone(), d.filter.clone()));
    }
    let info = app
        .filters
        .iter()
        .find(|f| f["id"] == json!(d.filter))
        .map(|f| {
            format!(
                "{} · {} taps at ×{} · transition {:.0} Hz ({:.3} %) · stopband {:.0} dB · latency {:.1} ms · {} window",
                f["family"].as_str().unwrap_or(""),
                f["taps_at_rate"].as_u64().unwrap_or(0),
                (f["out_rate"].as_f64().unwrap_or(1.0) / f["in_rate"].as_f64().unwrap_or(1.0)).round(),
                f["transition_hz"].as_f64().unwrap_or(0.0),
                f["transition_percent"].as_f64().unwrap_or(0.0),
                f["stopband_db"].as_f64().unwrap_or(0.0),
                f["latency_seconds"].as_f64().unwrap_or(0.0) * 1000.0,
                f["window"].as_str().unwrap_or("")
            )
        })
        .unwrap_or_default();
    let up = column![
        ui::h2("Upsampling"),
        ui::field("Filter family", family_pick),
        ui::field(
            "Filter",
            ui::pick(filters, &d.filter, |v| edit(
                move |s| s.dsp.filter = v.clone()
            ))
        ),
        ui::plain(filter_plain(&d.filter)),
        ui::hint(info),
        row![
            ui::field(
                "Headroom (dB; empty = auto)",
                text_box(
                    app,
                    "snd_headroom",
                    "auto (−3 dB when upsampling)",
                    || d.headroom_db.map(|v| v.to_string()).unwrap_or_default()
                )
            ),
            container(ui::switch(d.soft_clip, "Soft clipper", |on| edit(
                move |s| s.dsp.soft_clip = on
            )))
            .padding(iced::Padding::default().top(24)),
        ]
        .spacing(14),
    ]
    .spacing(12);

    // Noise shaping.
    let dithers = ui::choices(
        &serde_json::Value::Array(app.dithers.clone()),
        "id",
        "label",
    );
    let dplain = match d.dither.as_str() {
        "none" => "In plain words: no dither. Rounding to the bit depth can leave faint distortion on quiet passages; only for testing.",
        "auto" => "In plain words: Cadenza picks the best way to hide rounding noise for your output. Leave this on Auto if unsure.",
        _ => "In plain words: hides the tiny rounding error as a faint hiss, pushed where you can't hear it. Auto is a safe choice.",
    };
    let mut noise = column![
        ui::h2("Noise shaping"),
        ui::field("Dither / noise shaper (PCM output)", ui::pick(dithers, &d.dither, |v| edit(move |s| s.dsp.dither = v.clone()))),
        ui::plain(dplain),
        row![
            container(ui::field(
                "DAC noise floor (dBFS, A-weighted; empty = −120)",
                text_box(app, "snd_dac_floor", "-120", || d.dac_noise_floor_db.map(|v| v.to_string()).unwrap_or_default())
            ))
            .width(Length::Fill),
            ui::btn(
                if app.sound.noise_busy { "Measuring…" } else { "Measure" },
                Btn::Secondary,
                (!app.sound.noise_busy).then_some(Message::Sound(Msg::NoiseMeasure))
            ),
        ]
        .spacing(10)
        .align_y(Alignment::End),
        ui::hint("Used by Auto only: it picks the gentlest shaper whose audible noise stays 10 dB under this. Measure plays a −20 dBFS tone, then silence, and records them with the input chosen under Room & measure."),
    ]
    .spacing(12);
    if !app.sound.noise_msg.is_empty() {
        noise = noise.push(ui::small(app.sound.noise_msg.clone()));
    }

    // Headphones & convolution.
    let hp = column![
        ui::h2("Headphones & convolution"),
        ui::field(
            "Headphones (stereo sources)",
            ui::pick(
                ui::choices(&app.options["headphone_modes"], "id", "label"),
                &serde_json::to_value(d.headphones)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default(),
                |v| patch(json!({ "dsp": { "headphones": v } }))
            )
        ),
        ui::field(
            "Convolution (impulse response file, WAV or text; empty = off)",
            text_box(app, "snd_convolution", "/path/to/room.wav", || d
                .convolution
                .clone())
        ),
        ui::hint("Room correction from the Room & measure page loads here."),
    ]
    .spacing(12);

    let mut page = column![ui::h1(
        "Sound",
        "Filters, dither, EQ and the rest of the processing."
    )];
    if let Some(e) = &app.error {
        page = page.push(ui::err(e.clone()));
    }
    page.push(ui::grid(
        vec![ui::card(up), limit_card(app), ui::card(noise), ui::card(hp)],
        wide,
    ))
    .push(iced::widget::space().height(18))
    .push(eq_card(app))
    .push(iced::widget::space().height(18))
    .push(matrix_card(app))
    .into()
}

// ----- EQ ------------------------------------------------------------------------------

fn eq_card(app: &App) -> El<'_> {
    let l = theme::look();
    let b = bands(app);
    let on = app.settings.dsp.eq_enabled;
    let head = row![
        ui::h2("Parametric EQ"),
        iced::widget::space().width(Length::Fill),
        ui::switch(on, "On", |v| edit(move |s| s.dsp.eq_enabled = v)),
    ]
    .align_y(Alignment::Center);
    let tools = row![
        ui::btn(
            "+ Band",
            Btn::Secondary,
            (b.len() < cadenza_dsp::eq::MAX_BANDS)
                .then_some(Message::Sound(Msg::EqAdd(1000.0, 0.0)))
        ),
        ui::btn("Flat", Btn::Secondary, Some(Message::Sound(Msg::EqFlat))),
        ui::btn(
            "Clear",
            Btn::Secondary,
            (!b.is_empty()).then_some(Message::Sound(Msg::EqClear))
        ),
        iced::widget::space().width(Length::Fill),
        ui::small("Range"),
        ui::seg(
            &[("6", "±6"), ("12", "±12"), ("24", "±24")],
            &format!("{}", app.sound.eq_range as u32),
            |v| { Message::Sound(Msg::EqRange(v.parse().unwrap_or(12.0))) }
        ),
    ]
    .spacing(8)
    .align_y(Alignment::Center);
    let graph = container(
        canvas::Canvas::new(EqGraph {
            bands: b.clone(),
            on,
            range: app.sound.eq_range as f64,
            sel: app.sound.eq_sel,
        })
        .width(Length::Fill)
        .height(270),
    )
    .style(move |_| iced::widget::container::Style {
        background: Some(iced::Background::Color(l.bg2)),
        border: iced::Border {
            color: l.border,
            width: 1.0,
            radius: 12.0.into(),
        },
        ..Default::default()
    });

    // One box per band.
    let mut strip = Row::new().spacing(8);
    for (i, band) in b.iter().enumerate() {
        let c = eq_color(i);
        let sel = app.sound.eq_sel == Some(i);
        let (f_key, g_key, q_key) = (format!("eq_f{i}"), format!("eq_g{i}"), format!("eq_q{i}"));
        let mut col = column![
            row![
                iced::widget::checkbox(band.enabled)
                    .on_toggle(move |v| Message::Sound(Msg::EqEnabled(i, v)))
                    .size(16),
                text(format!("{}", i + 1))
                    .font(ui::bold())
                    .size(13)
                    .color(c),
                iced::widget::space().width(Length::Fill),
                button(text("✕").size(11))
                    .padding([2, 6])
                    .style(theme::btn(Btn::Ghost))
                    .on_press(Message::Sound(Msg::EqRemove(i))),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
            ui::pick(
                EQ_KINDS
                    .iter()
                    .map(|(id, n)| Choice::new(*id, *n))
                    .collect(),
                kind_id(band.kind),
                move |v| Message::Sound(Msg::EqKind(i, v))
            ),
            ui::label("FREQUENCY"),
            text_box(app, f_key, "1000", || format!("{} Hz", hz_label(band.freq))),
        ]
        .spacing(4);
        if has_gain(band.kind) {
            col = col
                .push(ui::label("GAIN"))
                .push(text_box(app, g_key, "0", || {
                    format!("{:.1} dB", band.gain_db)
                }));
        }
        col = col
            .push(ui::label("Q (WIDTH)"))
            .push(text_box(app, q_key, "0.71", || format!("{:.2}", band.q)));
        if matches!(band.kind, BandKind::LowPass | BandKind::HighPass) {
            col = col.push(ui::label("SLOPE")).push(ui::pick(
                [12u32, 24, 36, 48]
                    .iter()
                    .map(|v| Choice::new(v.to_string(), format!("{v} dB/oct")))
                    .collect(),
                &band.slope_db.to_string(),
                move |v| Message::Sound(Msg::EqSlope(i, v.parse().unwrap_or(12))),
            ));
        }
        let enabled = band.enabled;
        strip = strip.push(
            button(container(col).width(150))
                .padding(8)
                .style(move |t, st| {
                    let mut s = theme::btn(Btn::Secondary)(t, st);
                    s.background = Some(iced::Background::Color(l.panel2));
                    s.border = iced::Border {
                        color: if sel { c } else { l.border },
                        width: if sel { 2.0 } else { 1.0 },
                        radius: 10.0.into(),
                    };
                    if !enabled {
                        s.text_color = theme::alpha(l.text, 0.5);
                    }
                    s
                })
                .on_press(Message::Sound(Msg::EqSelect(Some(i)))),
        );
    }
    let strip: El = if b.is_empty() {
        ui::hint("No bands yet. Press + Band, or double-click the graph.")
    } else {
        scrollable(strip)
            .direction(scrollable::Direction::Horizontal(
                scrollable::Scrollbar::new(),
            ))
            .style(theme::scroll)
            .into()
    };
    ui::card(
        column![
            head,
            tools,
            graph,
            strip,
            ui::hint("Drag a point to move it (left/right = frequency, up/down = gain), scroll on it to change the width (Q), double-click the graph to add a bell there. Or type the numbers in the boxes and press Enter: frequency in Hz (1.2k works), gain in dB. Changes play straight away. Up to 16 bands."),
        ]
        .spacing(12),
    )
}

struct EqGraph {
    bands: Vec<Band>,
    on: bool,
    range: f64,
    sel: Option<usize>,
}

#[derive(Default)]
struct GraphState {
    drag: Option<usize>,
    last_click: Option<Instant>,
}

impl EqGraph {
    fn fx(&self, f: f64, w: f32) -> f32 {
        ((f / FMIN).ln() / (FMAX / FMIN).ln()) as f32 * w
    }
    fn xf(&self, x: f32, w: f32) -> f64 {
        FMIN * (FMAX / FMIN).powf((x / w).clamp(0.0, 1.0) as f64)
    }
    fn gy(&self, g: f64, h: f32) -> f32 {
        h / 2.0 - (g / self.range) as f32 * (h / 2.0 - 14.0)
    }
    fn yg(&self, y: f32, h: f32) -> f64 {
        ((h / 2.0 - y) / (h / 2.0 - 14.0)) as f64 * self.range
    }
    fn band_db(&self, b: &Band, f: f64) -> f64 {
        Equalizer::new(
            EQ_RATE,
            1,
            &[Band {
                enabled: true,
                ..*b
            }],
        )
        .response_db(f)
    }
    /// Where band `i`'s handle sits.
    fn handle(&self, i: usize, size: Size) -> Point {
        let b = &self.bands[i];
        let g = if has_gain(b.kind) {
            b.gain_db
        } else {
            self.band_db(b, b.freq).clamp(-self.range, self.range)
        };
        Point::new(self.fx(b.freq, size.width), self.gy(g, size.height))
    }
    fn hit(&self, p: Point, size: Size) -> Option<usize> {
        (0..self.bands.len())
            .rev()
            .find(|&i| self.handle(i, size).distance(p) <= 12.0)
    }
}

impl canvas::Program<Message> for EqGraph {
    type State = GraphState;

    fn update(
        &self,
        state: &mut GraphState,
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<canvas::Action<Message>> {
        let size = bounds.size();
        let pos = cursor.position_in(bounds);
        match event {
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let p = pos?;
                if let Some(i) = self.hit(p, size) {
                    state.drag = Some(i);
                    return Some(
                        canvas::Action::publish(Message::Sound(Msg::EqSelect(Some(i))))
                            .and_capture(),
                    );
                }
                // Double-click on empty graph: a bell there.
                let now = Instant::now();
                let double = state
                    .last_click
                    .is_some_and(|t| now.duration_since(t) < Duration::from_millis(400));
                state.last_click = Some(now);
                if double {
                    state.last_click = None;
                    let f = self.xf(p.x, size.width);
                    let g = self.yg(p.y, size.height).clamp(-self.range, self.range);
                    return Some(
                        canvas::Action::publish(Message::Sound(Msg::EqAdd(f, g))).and_capture(),
                    );
                }
                None
            }
            iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let i = state.drag?;
                let p = cursor.position_from(bounds.position())?;
                let f = self.xf(p.x, size.width);
                let g = self.yg(p.y, size.height).clamp(-self.range, self.range);
                Some(canvas::Action::publish(Message::Sound(Msg::EqDrag(i, f, g))).and_capture())
            }
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                state.drag.take()?;
                Some(canvas::Action::publish(Message::Sound(Msg::EqDragEnd)).and_capture())
            }
            iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                let p = pos?;
                let i = self.hit(p, size)?;
                let dy = match delta {
                    mouse::ScrollDelta::Lines { y, .. } | mouse::ScrollDelta::Pixels { y, .. } => {
                        *y
                    }
                };
                Some(canvas::Action::publish(Message::Sound(Msg::EqWheel(i, dy))).and_capture())
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        _: &GraphState,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        _: mouse::Cursor,
    ) -> Vec<Geometry> {
        let l = theme::look();
        let mut frame = Frame::new(renderer, bounds.size());
        let (w, h) = (bounds.width, bounds.height);
        let grid = Stroke::default().with_color(l.border).with_width(1.0);
        for f in [
            20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
        ] {
            let x = self.fx(f, w);
            frame.stroke(&Path::line(Point::new(x, 0.0), Point::new(x, h)), grid);
            frame.fill_text(Text {
                content: hz_label(f),
                position: Point::new(if f >= 20000.0 { x - 26.0 } else { x + 3.0 }, h - 16.0),
                color: l.faint,
                size: 11.0.into(),
                ..Default::default()
            });
        }
        let r = self.range;
        for g in [-r, -r / 2.0, 0.0, r / 2.0, r] {
            let y = self.gy(g, h);
            frame.stroke(
                &Path::line(Point::new(0.0, y), Point::new(w, y)),
                Stroke::default()
                    .with_color(l.border)
                    .with_width(if g == 0.0 { 1.6 } else { 1.0 }),
            );
            frame.fill_text(Text {
                content: format!("{}{}", if g > 0.0 { "+" } else { "" }, g),
                position: Point::new(4.0, y - 14.0),
                color: l.faint,
                size: 11.0.into(),
                ..Default::default()
            });
        }
        // Each band's own curve, faint; the sum bold, filled to 0 dB.
        let steps = 240;
        if self.on {
            for (i, b) in self.bands.iter().enumerate().filter(|(_, b)| b.enabled) {
                let path = Path::new(|p| {
                    for k in 0..=steps {
                        let x = k as f32 / steps as f32 * w;
                        let y = self
                            .gy(self.band_db(b, self.xf(x, w)), h)
                            .clamp(-50.0, h + 50.0);
                        if k == 0 {
                            p.move_to(Point::new(x, y))
                        } else {
                            p.line_to(Point::new(x, y))
                        }
                    }
                });
                let a = if self.sel == Some(i) { 0.9 } else { 0.35 };
                frame.stroke(
                    &path,
                    Stroke::default()
                        .with_color(theme::alpha(eq_color(i), a))
                        .with_width(1.2),
                );
            }
        }
        let eq = Equalizer::new(EQ_RATE, 1, &self.bands);
        let total: Vec<Point> = (0..=steps * 2)
            .map(|k| {
                let x = k as f32 / (steps * 2) as f32 * w;
                let g = if self.on {
                    eq.response_db(self.xf(x, w))
                } else {
                    0.0
                };
                Point::new(x, self.gy(g, h).clamp(-50.0, h + 50.0))
            })
            .collect();
        let zero = self.gy(0.0, h);
        let fill = Path::new(|p| {
            p.move_to(Point::new(0.0, zero));
            for pt in &total {
                p.line_to(*pt);
            }
            p.line_to(Point::new(w, zero));
            p.close();
        });
        frame.fill(&fill, theme::alpha(l.accent, 0.18));
        let line = Path::new(|p| {
            for (k, pt) in total.iter().enumerate() {
                if k == 0 {
                    p.move_to(*pt)
                } else {
                    p.line_to(*pt)
                }
            }
        });
        frame.stroke(
            &line,
            Stroke::default().with_color(l.accent).with_width(2.4),
        );
        for i in 0..self.bands.len() {
            let c = self.handle(i, bounds.size());
            let r = if self.sel == Some(i) { 10.0 } else { 8.0 };
            frame.fill(&Path::circle(c, r + 2.0), l.bg2);
            frame.fill(
                &Path::circle(c, r),
                theme::alpha(eq_color(i), if self.bands[i].enabled { 0.95 } else { 0.3 }),
            );
            frame.fill_text(Text {
                content: format!("{}", i + 1),
                position: Point::new(c.x - 3.5, c.y - 7.0),
                color: l.bg,
                size: 11.0.into(),
                font: ui::bold(),
                ..Default::default()
            });
        }
        vec![frame.into_geometry()]
    }

    fn mouse_interaction(
        &self,
        state: &GraphState,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.drag.is_some() {
            return mouse::Interaction::Grabbing;
        }
        match cursor.position_in(bounds) {
            Some(p) if self.hit(p, bounds.size()).is_some() => mouse::Interaction::Grab,
            Some(_) => mouse::Interaction::Crosshair,
            None => mouse::Interaction::default(),
        }
    }
}

// ----- Matrix --------------------------------------------------------------------------

fn matrix_card(app: &App) -> El<'_> {
    let m = &app.settings.matrix;
    let presets = {
        let mut p = ui::choices(&app.options["matrix_presets"], "id", "label");
        p.push(Choice::new("custom", "custom"));
        p
    };
    let preset_id = serde_json::to_value(&m.preset)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "stereo".into());
    let desc = app.options["matrix_presets"]
        .as_array()
        .and_then(|a| a.iter().find(|p| p["id"] == json!(preset_id)))
        .map(|p| {
            format!(
                "{} ({} → {} ch)",
                p["description"].as_str().unwrap_or(""),
                p["input_channels"],
                p["output_channels"]
            )
        })
        .unwrap_or_default();
    let st = &m.stereo;
    let ms = serde_json::to_value(st.mid_side)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "off".into());
    let mut c = column![
        row![
            ui::h2("Matrix"),
            iced::widget::space().width(Length::Fill),
            ui::switch(m.enabled, "On", |v| edit(move |s| s.matrix.enabled = v)),
        ]
        .align_y(Alignment::Center),
        row![
            column![
                ui::field(
                    "Preset",
                    ui::pick(presets, &preset_id, |v| patch(
                        json!({ "matrix": { "preset": v } })
                    ))
                ),
                ui::hint(desc),
            ]
            .spacing(6)
            .width(Length::Fill),
            column![
                row![
                    ui::switch(st.mono, "Mono", |v| patch(
                        json!({ "matrix": { "stereo": { "mono": v } } })
                    )),
                    ui::switch(st.swap, "Swap L/R", |v| patch(
                        json!({ "matrix": { "stereo": { "swap": v } } })
                    )),
                ]
                .spacing(20),
                row![
                    ui::field(
                        "Balance (dB, + attenuates left)",
                        text_box(app, "snd_balance", "0", || st.balance_db.to_string())
                    ),
                    ui::field(
                        "Mid/side",
                        ui::pick(
                            ["off", "encode", "decode"]
                                .iter()
                                .map(|v| Choice::new(*v, *v))
                                .collect(),
                            &ms,
                            |v| patch(json!({ "matrix": { "stereo": { "mid_side": v } } }))
                        )
                    ),
                ]
                .spacing(12),
            ]
            .spacing(12)
            .width(Length::Fill),
        ]
        .spacing(18),
    ]
    .spacing(12);
    if preset_id == "custom" {
        let json = serde_json::to_string_pretty(&m.outputs).unwrap_or_default();
        c = c
            .push(ui::label("Custom outputs (JSON: name, inputs dB, gain_db, delay_ms, delay_samples, invert, lowpass/highpass {kind, freq, slope_db}, eq); press Enter to apply"))
            .push(text_box(app, "snd_matrix", "[]", || json.replace('\n', " ")));
    }
    ui::card(c)
}
