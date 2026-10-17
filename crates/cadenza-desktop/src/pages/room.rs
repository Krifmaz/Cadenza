//! Room & measure: measure the room with a microphone, design and apply a correction, and
//! check the chain through a loopback (bit-perfect, THD+N).

use crate::app::{App, Message};
use crate::theme::{self, Badge, Btn};
use crate::ui::{self, Choice, El};
use cadenza::room::Room;
use iced::mouse;
use iced::widget::canvas::{self, Frame, Geometry, Path, Stroke, Text};
use iced::widget::{column, container, row, text, Column};
use iced::{Alignment, Color, Length, Point, Rectangle, Renderer, Task, Theme};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

pub struct State {
    pub room: Arc<Mutex<Room>>,
    /// Microphone / input ("" = default).
    pub input: String,
    report: Value,
    busy: bool,
    status: String,
    level: String,
    seconds: String,
    vf_bits: bool,
    vf_thd: bool,
    vf_level: String,
    vf_result: String,
}

impl Default for State {
    fn default() -> Self {
        Self {
            room: Arc::new(Mutex::new(Room::new(
                Room::default_dir(),
                cadenza::room::capture::system(),
            ))),
            input: String::new(),
            report: Value::Null,
            busy: false,
            status: String::new(),
            level: "-12".into(),
            seconds: "6".into(),
            vf_bits: true,
            vf_thd: true,
            vf_level: "-3".into(),
            vf_result: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Msg {
    Report(Result<Value, String>),
    Input(String),
    Level(String),
    Seconds(String),
    Measure(&'static str),
    Forget(&'static str),
    Apply,
    Remove,
    VfBits(bool),
    VfThd(bool),
    VfLevel(String),
    Verify,
    Verified(Result<Value, String>),
}

/// Run `f` on the room off the UI thread; the new report comes back.
fn call(
    app: &mut App,
    busy: Option<&str>,
    f: impl FnOnce(&mut Room, &cadenza::engine::Engine) -> Result<(), String> + Send + 'static,
) -> Task<Message> {
    if let Some(b) = busy {
        app.room.busy = true;
        app.room.status = b.into();
    }
    let (room, engine) = (Arc::clone(&app.room.room), app.engine.clone());
    Task::perform(
        async move {
            tokio::task::spawn_blocking(move || {
                let mut r = room.lock().unwrap_or_else(|e| e.into_inner());
                f(&mut r, &engine)?;
                Ok(r.report(&engine))
            })
            .await
            .unwrap_or_else(|e| Err(e.to_string()))
        },
        |r| Message::Room(Msg::Report(r)),
    )
}

pub fn init(app: &mut App) -> Task<Message> {
    call(app, None, |_, _| Ok(()))
}

pub fn update(app: &mut App, m: Msg) -> Task<Message> {
    let st = &mut app.room;
    match m {
        Msg::Report(r) => {
            match r {
                Ok(v) => {
                    st.report = v;
                    if st.busy {
                        st.status = "Done.".into();
                    }
                }
                Err(e) => st.status = e,
            }
            st.busy = false;
        }
        Msg::Input(i) => st.input = i,
        Msg::Level(v) => st.level = v,
        Msg::Seconds(v) => st.seconds = v,
        Msg::Measure(ch) => {
            let req = cadenza::room::MeasureRequest {
                channel: ch.into(),
                input: st.input.clone(),
                level_db: st.level.trim().parse().unwrap_or(-12.0),
                seconds: st.seconds.trim().parse().unwrap_or(6.0),
            };
            return call(
                app,
                Some(&format!("Measuring {ch}… keep quiet.")),
                move |r, e| r.measure(e, &req),
            );
        }
        Msg::Forget(ch) => return call(app, None, move |r, _| r.forget(ch)),
        Msg::Apply => {
            return call(app, Some("Writing the filter…"), |r, e| {
                r.apply(e).map(|_| ())
            })
        }
        Msg::Remove => {
            return call(app, Some("Removing…"), |r, e| {
                r.remove(e);
                Ok(())
            })
        }
        Msg::VfBits(v) => st.vf_bits = v,
        Msg::VfThd(v) => st.vf_thd = v,
        Msg::VfLevel(v) => st.vf_level = v,
        Msg::Verify => {
            st.busy = true;
            st.vf_result = "Playing and recording… keep quiet (about 10 s).".into();
            let req = cadenza::room::verify::VerifyRequest {
                input: st.input.clone(),
                bit_perfect: st.vf_bits,
                thd_n: st.vf_thd,
                level_db: st.vf_level.trim().parse().unwrap_or(-3.0),
                ..Default::default()
            };
            let (room, engine) = (Arc::clone(&st.room), app.engine.clone());
            return Task::perform(
                async move {
                    tokio::task::spawn_blocking(move || {
                        let mut r = room.lock().unwrap_or_else(|e| e.into_inner());
                        r.verify(&engine, &req)
                            .and_then(|rep| serde_json::to_value(rep).map_err(|e| e.to_string()))
                    })
                    .await
                    .unwrap_or_else(|e| Err(e.to_string()))
                },
                |r| Message::Room(Msg::Verified(r)),
            );
        }
        Msg::Verified(r) => {
            st.busy = false;
            st.vf_result = match r {
                Err(e) => e,
                Ok(r) => {
                    let mut out = Vec::new();
                    if let Some(b) = r.get("bit_perfect").filter(|v| !v.is_null()) {
                        out.push(format!(
                            "{}{}",
                            b["verdict"].as_str().unwrap_or(""),
                            b["signal_path"]
                                .as_str()
                                .map(|p| format!(" · {p}"))
                                .unwrap_or_default()
                        ));
                    }
                    if let Some(t) = r.get("thd_n").filter(|v| !v.is_null()) {
                        let f = |k: &str| t[k].as_f64().unwrap_or(f64::NAN);
                        let h: Vec<String> = t["harmonics_db"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .take(4)
                                    .map(|h| format!("{:.0}", h.as_f64().unwrap_or(0.0)))
                                    .collect()
                            })
                            .unwrap_or_default();
                        out.push(format!(
                            "THD+N {:.1} dB ({:.2} %) · THD {:.1} dB · noise {:.1} dB · tone {:.1} dBFS at {:.1} Hz · H2…: {} dB",
                            f("thd_n_db"),
                            f("thd_n_percent"),
                            f("thd_db"),
                            f("noise_db"),
                            f("level_db"),
                            f("frequency"),
                            h.join(" / ")
                        ));
                    }
                    if out.is_empty() {
                        "Nothing selected.".into()
                    } else {
                        out.join("\n")
                    }
                }
            };
        }
    }
    Task::none()
}

/// The design-option boxes (`room_high_hz` …).
pub fn commit(app: &mut App, k: &str, v: String) -> Task<Message> {
    let key = k.trim_start_matches("room_").to_string();
    let Ok(n) = v.trim().parse::<f64>() else {
        return Task::none();
    };
    call(app, None, move |r, _| {
        let mut cur = serde_json::to_value(&r.options).map_err(|e| e.to_string())?;
        cur[key.as_str()] = json!(n);
        let o: cadenza_dsp::room::DesignOptions =
            serde_json::from_value(cur).map_err(|e| e.to_string())?;
        if !(o.low_hz > 0.0 && o.high_hz > o.low_hz && o.taps >= 1024 && o.taps <= 1 << 18) {
            return Err("options out of range".into());
        }
        r.options = o;
        Ok(())
    })
}

pub fn view(app: &App, _wide: bool) -> El<'_> {
    let st = &app.room;
    let r = &st.report;
    let mut inputs = vec![Choice::new("", "Default input")];
    for n in r["inputs"].as_array().cloned().unwrap_or_default() {
        if let Some(n) = n.as_str() {
            inputs.push(Choice::new(n, n));
        }
    }
    let chans: Vec<Value> = r["channels"]
        .as_array()
        .map(|a| a.iter().filter(|c| !c.is_null()).cloned().collect())
        .unwrap_or_default();
    let badge: El = if r["applied"].as_bool() == Some(true) {
        ui::badge("applied", Badge::Ok)
    } else if !chans.is_empty() {
        ui::badge("measured", Badge::Accent)
    } else {
        text("").into()
    };
    let opt = |k: &'static str, name: &'static str| -> El<'_> {
        let key = format!("room_{k}");
        let v = app.text(&key, || {
            r["options"][k]
                .as_f64()
                .map(|v| v.to_string())
                .unwrap_or_default()
        });
        let k2 = key.clone();
        ui::field(
            name,
            ui::input("", &v, move |s| Message::Text(k2.clone(), s))
                .on_submit(Message::TextCommit(key)),
        )
    };
    let free = !st.busy;
    let legend = if chans.is_empty() {
        "No measurements yet.".to_string()
    } else {
        format!(
            "Solid: measured · dashed: corrected · grey: target. {}",
            chans
                .iter()
                .map(|c| format!(
                    "{} peak {} dBFS at {}",
                    c["channel"].as_str().unwrap_or(""),
                    c["peak_dbfs"]
                        .as_f64()
                        .map(|v| format!("{v:.1}"))
                        .unwrap_or("?".into()),
                    ui::khz(c["rate"].as_u64().unwrap_or(0) as u32)
                ))
                .collect::<Vec<_>>()
                .join(" · ")
        )
    };
    let room_card = column![
        row![ui::h2("Room correction"), badge].spacing(8).align_y(Alignment::Center),
        row![
            ui::field("Microphone / input", ui::pick(inputs, &st.input, |v| Message::Room(Msg::Input(v)))),
            ui::field("Level (dBFS)", ui::input("-12", &st.level, |v| Message::Room(Msg::Level(v)))),
            ui::field("Sweep (s)", ui::input("6", &st.seconds, |v| Message::Room(Msg::Seconds(v)))),
        ]
        .spacing(12),
        row![
            ui::btn("Measure left", Btn::Secondary, free.then_some(Message::Room(Msg::Measure("left")))),
            ui::btn("Measure right", Btn::Secondary, free.then_some(Message::Room(Msg::Measure("right")))),
            ui::btn("Forget left", Btn::Ghost, free.then_some(Message::Room(Msg::Forget("left")))),
            ui::btn("Forget right", Btn::Ghost, free.then_some(Message::Room(Msg::Forget("right")))),
        ]
        .spacing(8),
        row![
            opt("high_hz", "Correct up to (Hz)"),
            opt("max_boost_db", "Max boost (dB)"),
            opt("max_cut_db", "Max cut (dB)"),
            opt("tilt_db_per_octave", "Tilt (dB/octave)"),
        ]
        .spacing(12),
        container(canvas::Canvas::new(RoomPlot { report: r.clone() }).width(Length::Fill).height(260)).style(theme::inset),
        ui::small(legend),
        row![
            ui::btn("Apply correction", Btn::Primary, free.then_some(Message::Room(Msg::Apply))),
            ui::btn("Remove", Btn::Secondary, free.then_some(Message::Room(Msg::Remove))),
        ]
        .spacing(8),
        ui::hint(if st.status.is_empty() {
            "Put a measurement microphone at the listening seat. Each measurement plays a sine sweep from one speaker (turn the volume down first). The correction is a minimum-phase FIR that only cuts peaks and fills dips up to the set limits, then loads as the convolution filter.".to_string()
        } else {
            st.status.clone()
        }),
    ]
    .spacing(12);

    let verify = column![
        ui::h2("Loopback verification"),
        row![
            ui::switch(st.vf_bits, "Bit-perfect check", |v| Message::Room(Msg::VfBits(v))),
            ui::switch(st.vf_thd, "THD+N", |v| Message::Room(Msg::VfThd(v))),
            iced::widget::space().width(Length::Fill),
            container(ui::field("Tone (dBFS)", ui::input("-3", &st.vf_level, |v| Message::Room(Msg::VfLevel(v))))).width(130),
            ui::btn("Run", Btn::Primary, free.then_some(Message::Room(Msg::Verify))),
        ]
        .spacing(16)
        .align_y(Alignment::End),
        ui::small(st.vf_result.clone()),
        ui::hint("Connect the DAC's output (or a digital output) back to the input above, turn the volume to 100 %, and run. The bit-perfect check plays a random 24-bit pattern with the engine in bit-perfect mode and looks for it, sample for sample, in the recording – only a digital loopback can pass. THD+N plays a 997 Hz tone through the chain as configured; through an analog loopback it measures the DAC and the ADC together."),
    ]
    .spacing(12);

    Column::new()
        .push(ui::h1(
            "Room & measure",
            "Measure the room, correct it, and check what the chain really does.",
        ))
        .push(ui::card(room_card))
        .push(iced::widget::space().height(18))
        .push(ui::card(verify))
        .into()
}

struct RoomPlot {
    report: Value,
}

impl canvas::Program<Message> for RoomPlot {
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
        let (f0, f1, db) = (20.0f64, 20000.0f64, 20.0f64);
        let x = |f: f64| ((f / f0).ln() / (f1 / f0).ln()) as f32 * w;
        let y = |v: f64| h / 2.0 - (v.clamp(-db, db) / db) as f32 * (h / 2.0 - 4.0);
        let grid = Stroke::default().with_color(l.border).with_width(1.0);
        for f in [
            20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
        ] {
            frame.stroke(
                &Path::line(Point::new(x(f), 0.0), Point::new(x(f), h)),
                grid,
            );
            if f > 20.0 && f < 20000.0 {
                frame.fill_text(Text {
                    content: if f >= 1000.0 {
                        format!("{}k", f / 1000.0)
                    } else {
                        format!("{f}")
                    },
                    position: Point::new(x(f) + 2.0, h - 15.0),
                    color: l.faint,
                    size: 10.0.into(),
                    ..Default::default()
                });
            }
        }
        for v in [-20.0, -10.0, 0.0, 10.0, 20.0] {
            frame.stroke(
                &Path::line(Point::new(0.0, y(v)), Point::new(w, y(v))),
                grid,
            );
            frame.fill_text(Text {
                content: format!("{v} dB"),
                position: Point::new(3.0, (y(v) - 13.0).max(1.0)),
                color: l.faint,
                size: 10.0.into(),
                ..Default::default()
            });
        }
        let freqs: Vec<f64> = self.report["freqs"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let mut line = |vals: &Value, color: Color, dashed: bool| {
            let vals: Vec<f64> = vals
                .as_array()
                .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                .unwrap_or_default();
            if vals.len() != freqs.len() || vals.is_empty() {
                return;
            }
            let path = Path::new(|p| {
                for (i, (f, v)) in freqs.iter().zip(&vals).enumerate() {
                    let pt = Point::new(x(*f), y(*v));
                    if i == 0 || (dashed && i % 4 == 2) {
                        p.move_to(pt)
                    } else {
                        p.line_to(pt)
                    }
                }
            });
            frame.stroke(&path, Stroke::default().with_color(color).with_width(1.5));
        };
        line(&self.report["target"], theme::alpha(l.muted, 0.8), false);
        for c in self.report["channels"]
            .as_array()
            .cloned()
            .unwrap_or_default()
        {
            if c.is_null() {
                continue;
            }
            let color = if c["channel"] == json!("left") {
                theme::hex("#3b82f6")
            } else {
                theme::hex("#ef4444")
            };
            line(&c["measured"], color, false);
            line(&c["corrected"], color, true);
        }
        vec![frame.into_geometry()]
    }
}
