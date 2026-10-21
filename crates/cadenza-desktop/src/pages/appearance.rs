//! Appearance: the same looks as Encore, remembered on this computer.

use crate::app::{App, Message};
use crate::theme::{self, Appearance, Btn, ACCENTS, LOOKS};
use crate::ui::{self, El};
use iced::mouse;
use iced::widget::canvas::{self as canvas_mod, gradient, Frame, Geometry, Path, Stroke};
use iced::widget::{button, canvas, column, container, row, text, Row};
use iced::{Alignment, Color, Length, Point, Rectangle, Renderer, Size, Theme};

/// The custom colour picker, opened from the "+" swatch.
#[derive(Default)]
pub struct Picker {
    open: bool,
    h: f32,
    s: f32,
    v: f32,
}

#[derive(Clone, Debug)]
pub enum Msg {
    Toggle,
    /// Hue, saturation, value; `done` once the mouse is let go (then it is saved).
    Pick(f32, f32, f32, bool),
}

pub fn update(app: &mut App, m: Msg) {
    match m {
        Msg::Toggle => {
            let p = &mut app.picker;
            p.open = !p.open;
            if p.open {
                (p.h, p.s, p.v) = to_hsv(theme::look().accent);
            }
        }
        Msg::Pick(h, s, v, done) => {
            app.picker.h = h;
            app.picker.s = s;
            app.picker.v = v;
            let custom = to_hex(from_hsv(h, s, v));
            app.texts.remove("ap_custom");
            let a = Appearance {
                accent: "custom".into(),
                custom,
                ..app.appearance.clone()
            };
            if done {
                let _ = app.update(Message::SetAppearance(a));
            } else {
                // Live while dragging; written to disk when the mouse is let go.
                theme::set(&a);
                app.appearance = a;
            }
        }
    }
}

fn from_hsv(h: f32, s: f32, v: f32) -> Color {
    let c = v * s;
    let hp = (h.rem_euclid(1.0)) * 6.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    Color::from_rgb(r + m, g + m, b + m)
}

fn to_hsv(c: Color) -> (f32, f32, f32) {
    let max = c.r.max(c.g).max(c.b);
    let min = c.r.min(c.g).min(c.b);
    let d = max - min;
    let h = if d <= f32::EPSILON {
        0.0
    } else if max == c.r {
        ((c.g - c.b) / d).rem_euclid(6.0) / 6.0
    } else if max == c.g {
        ((c.b - c.r) / d + 2.0) / 6.0
    } else {
        ((c.r - c.g) / d + 4.0) / 6.0
    };
    let s = if max <= f32::EPSILON { 0.0 } else { d / max };
    (h, s, max)
}

fn to_hex(c: Color) -> String {
    let b = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c.r), b(c.g), b(c.b))
}

const SQ: f32 = 220.0;
const BAR_X: f32 = 234.0;
const PICK_W: f32 = 260.0;
const PICK_H: f32 = 170.0;

#[derive(Clone, Copy, PartialEq)]
enum Zone {
    Square,
    Hue,
}

struct Wheel {
    h: f32,
    s: f32,
    v: f32,
}

impl Wheel {
    fn at(&self, z: Zone, p: Point, done: bool) -> Message {
        let (mut h, mut s, mut v) = (self.h, self.s, self.v);
        match z {
            Zone::Square => {
                s = (p.x / SQ).clamp(0.0, 1.0);
                v = 1.0 - (p.y / PICK_H).clamp(0.0, 1.0);
            }
            Zone::Hue => h = (p.y / PICK_H).clamp(0.0, 0.9999),
        }
        Message::Appearance(Msg::Pick(h, s, v, done))
    }
}

impl canvas_mod::Program<Message> for Wheel {
    type State = Option<Zone>;

    fn update(
        &self,
        drag: &mut Option<Zone>,
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<canvas_mod::Action<Message>> {
        match event {
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let p = cursor.position_in(bounds)?;
                let z = if p.x <= SQ {
                    Zone::Square
                } else if p.x >= BAR_X {
                    Zone::Hue
                } else {
                    return None;
                };
                *drag = Some(z);
                Some(canvas_mod::Action::publish(self.at(z, p, false)).and_capture())
            }
            iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let z = (*drag)?;
                let p = cursor.position_from(bounds.position())?;
                Some(canvas_mod::Action::publish(self.at(z, p, false)).and_capture())
            }
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                let z = drag.take()?;
                let p = cursor.position_from(bounds.position())?;
                Some(canvas_mod::Action::publish(self.at(z, p, true)).and_capture())
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        _: &Option<Zone>,
        renderer: &Renderer,
        _: &Theme,
        bounds: Rectangle,
        _: mouse::Cursor,
    ) -> Vec<Geometry> {
        let mut f = Frame::new(renderer, bounds.size());
        let l = theme::look();
        let sq = Path::rounded_rectangle(Point::ORIGIN, Size::new(SQ, PICK_H), 8.0.into());
        f.fill(
            &sq,
            gradient::Linear::new(Point::ORIGIN, Point::new(SQ, 0.0))
                .add_stop(0.0, Color::WHITE)
                .add_stop(1.0, from_hsv(self.h, 1.0, 1.0)),
        );
        f.fill(
            &sq,
            gradient::Linear::new(Point::ORIGIN, Point::new(0.0, PICK_H))
                .add_stop(0.0, Color::TRANSPARENT)
                .add_stop(1.0, Color::BLACK),
        );
        f.stroke(&sq, Stroke::default().with_color(l.border).with_width(1.0));

        let bar = Path::rounded_rectangle(
            Point::new(BAR_X, 0.0),
            Size::new(PICK_W - BAR_X, PICK_H),
            6.0.into(),
        );
        let mut g = gradient::Linear::new(Point::new(0.0, 0.0), Point::new(0.0, PICK_H));
        for i in 0..=6 {
            let t = i as f32 / 6.0;
            g = g.add_stop(t, from_hsv(t, 1.0, 1.0));
        }
        f.fill(&bar, g);
        f.stroke(&bar, Stroke::default().with_color(l.border).with_width(1.0));

        // Markers: a ring on the square, a line across the bar.
        let at = Point::new(self.s * SQ, (1.0 - self.v) * PICK_H);
        let ring = Path::circle(at, 7.0);
        f.stroke(
            &ring,
            Stroke::default().with_color(Color::BLACK).with_width(3.5),
        );
        f.stroke(
            &ring,
            Stroke::default().with_color(Color::WHITE).with_width(2.0),
        );
        let y = self.h * PICK_H;
        let mark = Path::rounded_rectangle(
            Point::new(BAR_X - 3.0, y - 3.0),
            Size::new(PICK_W - BAR_X + 6.0, 6.0),
            3.0.into(),
        );
        f.stroke(
            &mark,
            Stroke::default().with_color(Color::BLACK).with_width(3.0),
        );
        f.stroke(
            &mark,
            Stroke::default().with_color(Color::WHITE).with_width(1.5),
        );
        vec![f.into_geometry()]
    }

    fn mouse_interaction(
        &self,
        drag: &Option<Zone>,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if drag.is_some() || cursor.is_over(bounds) {
            mouse::Interaction::Crosshair
        } else {
            mouse::Interaction::default()
        }
    }
}

pub fn view(app: &App, wide: bool) -> El<'_> {
    let a = &app.appearance;
    let mut looks = Row::new().spacing(10);
    for (id, name, th, accent, custom) in LOOKS {
        let on = a.theme == th && a.accent == accent && (accent != "custom" || a.custom == custom);
        let mut next = Appearance {
            theme: th.into(),
            accent: accent.into(),
            ..a.clone()
        };
        if !custom.is_empty() {
            next.custom = custom.into();
        }
        let sw = theme::Look::from(&next);
        let chip = row![
            container(text("")).width(22).height(22).style(move |_| {
                iced::widget::container::Style {
                    background: Some(iced::Background::Gradient(
                        iced::gradient::Linear::new(iced::Radians(2.35))
                            .add_stop(0.0, sw.bg)
                            .add_stop(0.5, sw.bg)
                            .add_stop(0.5, sw.accent)
                            .add_stop(1.0, sw.accent2)
                            .into(),
                    )),
                    border: iced::Border {
                        color: sw.border,
                        width: 1.0,
                        radius: 99.0.into(),
                    },
                    ..Default::default()
                }
            }),
            text(name).size(13.5).font(ui::bold()),
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        let _ = id;
        looks = looks.push(
            button(chip)
                .padding([8, 12])
                .style(theme::btn(if on { Btn::SegOn } else { Btn::Secondary }))
                .on_press(Message::SetAppearance(next)),
        );
    }
    let themes = ui::seg(
        &[
            ("encore", "Encore"),
            ("dark", "Dark"),
            ("oled", "OLED"),
            ("light", "Light"),
        ],
        &a.theme,
        {
            let a = a.clone();
            move |t| {
                Message::SetAppearance(Appearance {
                    theme: t,
                    ..a.clone()
                })
            }
        },
    );
    let mut sw = Row::new().spacing(8);
    for (id, _name, c1, _c2) in ACCENTS {
        sw = sw.push(
            button(text(""))
                .width(30)
                .height(30)
                .style(theme::swatch(theme::hex(c1), a.accent == id))
                .on_press(Message::SetAppearance(Appearance {
                    accent: id.into(),
                    ..a.clone()
                })),
        );
    }
    let pk = &app.picker;
    let custom_on = a.accent == "custom";
    sw = sw.push(
        button(
            container(
                text(if pk.open { "×" } else { "+" })
                    .size(17)
                    .color(theme::look().text),
            )
            .center(30),
        )
        .width(30)
        .height(30)
        .padding(0)
        .style(theme::swatch(
            if custom_on {
                theme::hex(&a.custom)
            } else {
                theme::look().panel
            },
            custom_on || pk.open,
        ))
        .on_press(Message::Appearance(Msg::Toggle)),
    );
    let custom = app.text("ap_custom", || a.custom.clone());
    let custom_row = row![
        ui::small("Custom colour (#rrggbb)"),
        ui::input("#f0ead6", &custom, |v| Message::Text("ap_custom".into(), v))
            .on_submit(Message::TextCommit("ap_custom".into()))
            .width(140),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    column![
        ui::h1(
            "Appearance",
            "The same looks as Encore. Remembered on this computer."
        ),
        ui::card(
            column![
                ui::h2("Looks"),
                iced::widget::scrollable(looks)
                    .direction(iced::widget::scrollable::Direction::Horizontal(
                        Default::default()
                    ))
                    .style(theme::scroll)
            ]
            .spacing(12)
        ),
        iced::widget::space().height(18),
        ui::grid(
            vec![
                ui::card(column![ui::h2("Theme"), themes].spacing(12)),
                ui::card({
                    let mut c = column![ui::h2("Colour"), sw.wrap()].spacing(12);
                    if pk.open {
                        c = c.push(
                            row![
                                canvas(Wheel {
                                    h: pk.h,
                                    s: pk.s,
                                    v: pk.v
                                })
                                .width(PICK_W)
                                .height(PICK_H),
                                column![
                                    container(text(""))
                                        .width(56)
                                        .height(56)
                                        .style(theme::swatch_box(from_hsv(pk.h, pk.s, pk.v))),
                                    ui::small(to_hex(from_hsv(pk.h, pk.s, pk.v))),
                                ]
                                .spacing(8)
                                .align_x(Alignment::Center),
                            ]
                            .spacing(16)
                            .align_y(Alignment::Center),
                        );
                        c = c.push(ui::hint(
                            "Drag in the square for shade and brightness, and on the bar for the colour. It is saved when you let go.",
                        ));
                    }
                    c.push(custom_row)
                }),
                ui::card(
                    column![
                        ui::h2("Updates"),
                        ui::small(format!("This is Cadenza {}.", crate::update::current())),
                        row![
                            ui::btn(
                                "Check for updates",
                                Btn::Secondary,
                                (!app.update_busy).then_some(Message::UpdateCheck(true))
                            ),
                            ui::btn(
                                "Update now",
                                Btn::Primary,
                                (app.update.is_some() && !app.update_busy)
                                    .then_some(Message::UpdateInstall)
                            ),
                        ]
                        .spacing(8),
                        ui::small(app.update_msg.clone()),
                        ui::switch(app.update_hotfix, "Hotfixes", Message::UpdateHotfix),
                        ui::hint("Cadenza looks for a new version when it starts and every few hours. On Windows, Update installs it and restarts Cadenza; elsewhere it opens the download page. Hotfixes: also the build of every change on GitHub (Linux and Windows); a full release is still offered when it is newer."),
                    ]
                    .spacing(12)
                ),
            ],
            wide
        ),
    ]
    .width(Length::Fill)
    .into()
}

pub fn commit_custom(app: &mut App, v: String) {
    let v = v.trim().to_string();
    let ok = v.len() == 7 && v.starts_with('#') && u32::from_str_radix(&v[1..], 16).is_ok();
    if ok {
        let a = Appearance {
            accent: "custom".into(),
            custom: v,
            ..app.appearance.clone()
        };
        let _ = app.update(Message::SetAppearance(a));
    }
}
