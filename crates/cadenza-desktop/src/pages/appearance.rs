//! Appearance: the same looks as Encore, remembered on this computer.

use crate::app::{App, Message};
use crate::theme::{self, Appearance, Btn, ACCENTS, LOOKS};
use crate::ui::{self, El};
use iced::widget::{button, column, container, row, text, Row};
use iced::{Alignment, Length};

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
            container(text(""))
                .width(22)
                .height(22)
                .style(move |_| iced::widget::container::Style {
                    background: Some(iced::Background::Gradient(
                        iced::gradient::Linear::new(iced::Radians(2.35))
                            .add_stop(0.0, sw.bg)
                            .add_stop(0.5, sw.bg)
                            .add_stop(0.5, sw.accent)
                            .add_stop(1.0, sw.accent2)
                            .into()
                    )),
                    border: iced::Border {
                        color: sw.border,
                        width: 1.0,
                        radius: 99.0.into()
                    },
                    ..Default::default()
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
        &[("encore", "Encore"), ("dark", "Dark"), ("oled", "OLED"), ("light", "Light")],
        &a.theme,
        {
            let a = a.clone();
            move |t| Message::SetAppearance(Appearance { theme: t, ..a.clone() })
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
    let custom = app.text("ap_custom", || a.custom.clone());
    let custom_row = row![
        ui::small("Custom colour (#rrggbb)"),
        ui::input("#f0ead6", &custom, |v| Message::Text("ap_custom", v))
            .on_submit(Message::TextCommit("ap_custom"))
            .width(140),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    column![
        ui::h1("Appearance", "The same looks as Encore. Remembered on this computer."),
        ui::card(column![ui::h2("Looks"), iced::widget::scrollable(looks).direction(
            iced::widget::scrollable::Direction::Horizontal(Default::default())
        ).style(theme::scroll)].spacing(12)),
        iced::widget::space().height(18),
        ui::grid(
            vec![
                ui::card(column![ui::h2("Theme"), themes].spacing(12)),
                ui::card(column![ui::h2("Colour"), sw.wrap(), custom_row].spacing(12)),
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
