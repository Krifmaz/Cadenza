//! Filters: design an upsampling filter and look at any filter's response.

use crate::app::{edit, App, Message};
use crate::theme::{self, Btn};
use crate::ui::{self, Choice, El};
use iced::mouse;
use iced::widget::canvas::{self, Frame, Geometry, Path, Stroke, Text};
use iced::widget::{column, container, row, Column};
use iced::{Color, Length, Point, Rectangle, Renderer, Task, Theme};
use serde_json::{json, Value};

pub struct State {
    window: String,
    phase: String,
    rates: String,
    cutoff: String,
    transition: String,
    stopband: String,
    info: String,
    response: Option<Value>,
    busy: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            window: "kaiser".into(),
            phase: "l".into(),
            rates: "44100,705600".into(),
            cutoff: "100".into(),
            transition: "5".into(),
            stopband: "150".into(),
            info: String::new(),
            response: None,
            busy: false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Msg {
    Window(String),
    Phase(String),
    Rates(String),
    Cutoff(String),
    Transition(String),
    Stopband(String),
    PlotDesign,
    PlotCurrent,
    Use,
    Plotted(Result<Value, String>),
}

fn num_id(s: &str, dflt: f64) -> String {
    let v: f64 = s.trim().parse().unwrap_or(dflt);
    let t = format!("{v:.4}");
    t.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn design_id(st: &State) -> String {
    format!(
        "custom-{}-{}-c{}-t{}-s{}",
        st.window,
        st.phase,
        num_id(&st.cutoff, 100.0),
        num_id(&st.transition, 5.0),
        num_id(&st.stopband, 150.0)
    )
}

fn plot(app: &mut App, id: String) -> Task<Message> {
    app.filters_page.busy = true;
    app.filters_page.info = "Designing…".into();
    let (ir, or) = app
        .filters_page
        .rates
        .split_once(',')
        .unwrap_or(("44100", "705600"));
    let (ir, or): (u32, u32) = (ir.parse().unwrap_or(44_100), or.parse().unwrap_or(705_600));
    let att = app.settings.dsp.attenuation_db;
    let cache = app.engine.cache();
    Task::perform(
        async move {
            tokio::task::spawn_blocking(move || {
                use cadenza_dsp::design::{response, Filter, MAX_PLOT_TAPS};
                let f = Filter::parse(&id).ok_or_else(|| format!("unknown filter {id:?}"))?;
                let (spec, rate) = f.primary(ir, or, att);
                let info = f.describe(ir, or, att);
                if spec.taps > MAX_PLOT_TAPS {
                    return Err(format!(
                        "{} taps at this rate: too long to plot (up to {} here)",
                        spec.taps,
                        MAX_PLOT_TAPS - 1
                    ));
                }
                let h = cache.get_or_design(&spec);
                let (p, s) = f.edges(ir, or, att);
                let r = response(&h, rate, ir as f64 / 2.0, p, s);
                Ok(json!({ "info": info, "response": r }))
            })
            .await
            .unwrap_or_else(|e| Err(e.to_string()))
        },
        |r| Message::Filters(Msg::Plotted(r)),
    )
}

pub fn update(app: &mut App, m: Msg) -> Task<Message> {
    let st = &mut app.filters_page;
    match m {
        Msg::Window(v) => st.window = v,
        Msg::Phase(v) => st.phase = v,
        Msg::Rates(v) => st.rates = v,
        Msg::Cutoff(v) => st.cutoff = v,
        Msg::Transition(v) => st.transition = v,
        Msg::Stopband(v) => st.stopband = v,
        Msg::PlotDesign => {
            let id = design_id(st);
            return plot(app, id);
        }
        Msg::PlotCurrent => {
            let id = app.settings.dsp.filter.clone();
            return plot(app, id);
        }
        Msg::Use => {
            let id = design_id(st);
            return app.update(edit(move |s| s.dsp.filter = id.clone()));
        }
        Msg::Plotted(r) => {
            st.busy = false;
            match r {
                Ok(v) => {
                    let i = &v["info"];
                    let r = &v["response"];
                    st.info = format!(
                        "{} · {} taps at {} · transition {:.0} Hz · measured stopband {:.1} dB (from {:.2} kHz) · passband ripple {:.1e} dB (to {:.2} kHz) · latency {:.2} ms",
                        i["label"].as_str().unwrap_or(""),
                        i["taps_at_rate"].as_u64().unwrap_or(0),
                        ui::khz(r["filter_rate"].as_f64().unwrap_or(0.0) as u32),
                        i["transition_hz"].as_f64().unwrap_or(0.0),
                        r["stopband_db"].as_f64().unwrap_or(0.0),
                        r["stopband_hz"].as_f64().unwrap_or(0.0) / 1000.0,
                        r["ripple_db"].as_f64().unwrap_or(0.0),
                        r["passband_hz"].as_f64().unwrap_or(0.0) / 1000.0,
                        i["latency_seconds"].as_f64().unwrap_or(0.0) * 1000.0
                    );
                    st.response = Some(v["response"].clone());
                }
                Err(e) => {
                    st.info = e;
                    st.response = None;
                }
            }
        }
    }
    Task::none()
}

pub fn commit(_app: &mut App, _k: &str, _v: String) -> Task<Message> {
    Task::none()
}

fn points(v: &Value) -> Vec<(f64, f64)> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| Some((p[0].as_f64()?, p[1].as_f64()?)))
                .collect()
        })
        .unwrap_or_default()
}

pub fn view(app: &App, _wide: bool) -> El<'_> {
    let st = &app.filters_page;
    let windows = vec![
        Choice::new("kaiser", "Kaiser"),
        Choice::new("dolph-chebyshev", "Dolph–Chebyshev (closed-form)"),
        Choice::new("gaussian", "Gaussian (no sidelobes)"),
        Choice::new("cosine3", "3-term cosine (71.5 dB)"),
    ];
    let rates = [
        ("44100,705600", "44.1 → 705.6 kHz (×16)"),
        ("44100,176400", "44.1 → 176.4 kHz (×4)"),
        ("44100,352800", "44.1 → 352.8 kHz (×8)"),
        ("48000,768000", "48 → 768 kHz (×16)"),
        ("48000,192000", "48 → 192 kHz (×4)"),
        ("96000,384000", "96 → 384 kHz (×4)"),
        ("44100,48000", "44.1 → 48 kHz"),
    ]
    .iter()
    .map(|(a, b)| Choice::new(*a, *b))
    .collect();
    let mut c = Column::new()
        .spacing(12)
        .push(
            row![
                ui::field(
                    "Window",
                    ui::pick(windows, &st.window, |v| Message::Filters(Msg::Window(v)))
                ),
                ui::field(
                    "Phase",
                    ui::pick(
                        vec![Choice::new("l", "Linear"), Choice::new("m", "Minimum")],
                        &st.phase,
                        |v| Message::Filters(Msg::Phase(v))
                    )
                ),
                ui::field(
                    "Plot for",
                    ui::pick(rates, &st.rates, |v| Message::Filters(Msg::Rates(v)))
                ),
            ]
            .spacing(12),
        )
        .push(
            row![
                ui::field(
                    "Cutoff (% of source Nyquist)",
                    ui::input("100", &st.cutoff, |v| Message::Filters(Msg::Cutoff(v)))
                ),
                ui::field(
                    "Transition (% of source Nyquist)",
                    ui::input("5", &st.transition, |v| Message::Filters(Msg::Transition(
                        v
                    )))
                ),
                ui::field(
                    "Stopband (dB)",
                    ui::input("150", &st.stopband, |v| Message::Filters(Msg::Stopband(v)))
                ),
            ]
            .spacing(12),
        )
        .push(
            row![
                ui::btn(
                    "Plot design",
                    Btn::Primary,
                    (!st.busy).then_some(Message::Filters(Msg::PlotDesign))
                ),
                ui::btn(
                    "Plot selected filter",
                    Btn::Secondary,
                    (!st.busy).then_some(Message::Filters(Msg::PlotCurrent))
                ),
                ui::btn(
                    "Use this design",
                    Btn::Secondary,
                    Some(Message::Filters(Msg::Use))
                ),
            ]
            .spacing(8),
        );
    if !st.info.is_empty() {
        c = c.push(ui::small(st.info.clone()));
    }
    if let Some(r) = &st.response {
        let nyq = r["nyquist"].as_f64().unwrap_or(22050.0);
        let tr = points(&r["transition"]);
        let floor = ((r["stopband_db"].as_f64().unwrap_or(-150.0) - 30.0) / 20.0).floor() * 20.0;
        let floor = floor.max(-300.0);
        let x1 = tr.last().map(|p| p.0).unwrap_or(2.0 * nyq);
        c = c.push(plot_box(
            Plot {
                series: vec![(tr, theme::hex("#6366f1"))],
                x: (0.0, x1),
                y: (floor, 5.0),
                marks: vec![
                    (nyq, theme::look().muted),
                    (
                        r["passband_hz"].as_f64().unwrap_or(0.0),
                        theme::hex("#22c55e"),
                    ),
                    (
                        r["stopband_hz"].as_f64().unwrap_or(0.0),
                        theme::hex("#ef4444"),
                    ),
                ],
                title:
                    "Magnitude (dB) · grey: source Nyquist · green / red: passband / stopband edge"
                        .into(),
                xk: true,
            },
            240.0,
        ));
        let pb = points(&r["passband"]);
        let (lo, hi) = pb
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
        let pad = ((hi - lo) * 0.2).max(1e-9);
        let imp = points(&r["impulse"]);
        let ilo = imp.iter().map(|p| p.1).fold(f64::MAX, f64::min);
        c = c.push(
            row![
                plot_box(
                    Plot {
                        series: vec![(pb, theme::hex("#22c55e"))],
                        x: (0.0, r["passband_hz"].as_f64().unwrap_or(20000.0)),
                        y: (lo - pad, hi + pad),
                        marks: vec![],
                        title: "Passband (dB)".into(),
                        xk: true,
                    },
                    200.0
                ),
                plot_box(
                    Plot {
                        x: (
                            imp.first().map(|p| p.0).unwrap_or(0.0),
                            imp.last().map(|p| p.0).unwrap_or(1.0)
                        ),
                        series: vec![(imp, theme::hex("#f59e0b"))],
                        y: (ilo - 0.05, 1.05),
                        marks: vec![],
                        title: "Impulse (ms)".into(),
                        xk: false,
                    },
                    200.0
                ),
            ]
            .spacing(12),
        );
    }
    c = c.push(ui::hint("Cadenza works out the length (it grows with the ratio, so the filter has the same shape in Hz at every output rate), designs it in 64-bit and caches it. Top: the response up to twice the source Nyquist. Left: the passband ripple. Right: the impulse around its peak. Filters over 2M taps at the chosen rate are not plotted."));
    column![
        ui::h1(
            "Filter designer",
            "Make an upsampling filter to measure, and look at any filter's response."
        ),
        ui::card(c)
    ]
    .into()
}

fn plot_box<'a>(p: Plot, h: f32) -> El<'a> {
    container(canvas::Canvas::new(p).width(Length::Fill).height(h))
        .style(theme::inset)
        .width(Length::Fill)
        .into()
}

struct Plot {
    series: Vec<(Vec<(f64, f64)>, Color)>,
    x: (f64, f64),
    y: (f64, f64),
    marks: Vec<(f64, Color)>,
    title: String,
    /// x labels in kHz.
    xk: bool,
}

fn ticks(a: f64, b: f64, n: f64) -> Vec<f64> {
    if b.partial_cmp(&a) != Some(std::cmp::Ordering::Greater) {
        return vec![a];
    }
    let st = 10f64.powf(((b - a) / n).log10().floor());
    let m = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|k| k * st)
        .find(|m| (b - a) / m <= n)
        .unwrap_or(st * 10.0);
    let mut out = Vec::new();
    let mut v = (a / m).ceil() * m;
    while v <= b + 1e-12 && out.len() < 50 {
        out.push(v);
        v += m;
    }
    out
}

impl canvas::Program<Message> for Plot {
    type State = ();

    fn draw(
        &self,
        _: &(),
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        _: mouse::Cursor,
    ) -> Vec<Geometry> {
        let l = theme::look();
        let mut frame = Frame::new(renderer, bounds.size());
        let (w, h) = (bounds.width, bounds.height);
        let (x0, x1, y0, y1) = (self.x.0, self.x.1, self.y.0, self.y.1);
        let sx = |v: f64| 44.0 + ((v - x0) / (x1 - x0).max(1e-12)) as f32 * (w - 50.0);
        let sy = |v: f64| {
            22.0 + (1.0 - ((v.clamp(y0, y1) - y0) / (y1 - y0).max(1e-300)) as f32) * (h - 40.0)
        };
        let grid = Stroke::default().with_color(l.border).with_width(1.0);
        let label = |frame: &mut Frame, s: String, p: Point| {
            frame.fill_text(Text {
                content: s,
                position: p,
                color: l.faint,
                size: 10.0.into(),
                ..Default::default()
            });
        };
        for v in ticks(x0, x1, 6.0) {
            frame.stroke(
                &Path::line(Point::new(sx(v), 22.0), Point::new(sx(v), h - 18.0)),
                grid,
            );
            let t = if self.xk {
                format!("{:.1}k", v / 1000.0)
            } else {
                format!("{v:.2}")
            };
            label(&mut frame, t, Point::new(sx(v) - 10.0, h - 15.0));
        }
        for v in ticks(y0, y1, 5.0) {
            frame.stroke(
                &Path::line(Point::new(44.0, sy(v)), Point::new(w - 6.0, sy(v))),
                grid,
            );
            let t = if (y1 - y0) < 0.01 {
                format!("{v:.0e}")
            } else {
                format!("{v:.0}")
            };
            label(&mut frame, t, Point::new(2.0, sy(v) - 6.0));
        }
        for (v, c) in &self.marks {
            frame.stroke(
                &Path::line(Point::new(sx(*v), 22.0), Point::new(sx(*v), h - 18.0)),
                Stroke::default().with_color(*c).with_width(1.0),
            );
        }
        for (pts, c) in &self.series {
            let path = Path::new(|p| {
                for (i, (x, y)) in pts.iter().enumerate() {
                    let pt = Point::new(sx(*x), sy(*y));
                    if i == 0 {
                        p.move_to(pt)
                    } else {
                        p.line_to(pt)
                    }
                }
            });
            frame.stroke(&path, Stroke::default().with_color(*c).with_width(1.3));
        }
        label(&mut frame, self.title.clone(), Point::new(44.0, 4.0));
        vec![frame.into_geometry()]
    }
}
