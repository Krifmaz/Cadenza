//! Now playing: the track, what the engine does to it (signal path) and the queue.

use crate::app::{App, Message, Transport};
use crate::theme::{self, Badge, Btn};
use crate::ui::{self, El};
use cadenza_dsp::signal_path::Endpoint;
use iced::widget::{button, column, container, image, row, space, text, Column, Row};
use iced::{Alignment, Length};

pub fn view(app: &App) -> El<'_> {
    let l = theme::look();
    let st = &app.status;
    let t = st.track.as_ref();
    let art: El = match app.cover(t.and_then(|t| t.cover.as_deref())) {
        Some(h) => container(image(h).width(180).height(180)).into(),
        None => container(text("♪").size(64).color(l.on_accent))
            .width(180)
            .height(180)
            .center(180)
            .style(theme::accent_tile)
            .into(),
    };
    let title = t
        .map(|t| if t.title.is_empty() { "Untitled".into() } else { t.title.clone() })
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
    let mut chips = row![].spacing(6);
    if let Some(f) = &st.source_format {
        chips = chips.push(ui::badge(f.clone(), Badge::Plain));
    }
    if let Some(p) = &st.signal_path {
        chips = chips.push(if p.bit_perfect {
            ui::badge("bit-perfect", Badge::Ok)
        } else {
            ui::badge("processed", Badge::Warn)
        });
    }
    let mut info = column![
        text("NOW PLAYING").size(12).color(l.accent).font(ui::bold()),
        text(title).size(32).font(ui::bold()),
    ]
    .spacing(6);
    if !sub.is_empty() {
        info = info.push(text(sub).size(16).color(l.muted));
    }
    info = info.push(chips);
    if let Some(e) = &st.error {
        info = info.push(ui::err(e.clone()));
    }
    let hero = row![art, info].spacing(26).align_y(Alignment::Center);

    // The signal path, as boxes in processing order.
    let chain: El = match &st.signal_path {
        None => ui::hint("Stopped."),
        Some(p) => {
            let ep = |e: &Endpoint, name: String| -> El<'_> {
                let detail = [
                    e.format.clone(),
                    if e.rate > 0 { ui::khz(e.rate) } else { String::new() },
                    if e.bits > 0 { format!("{}-bit", e.bits) } else { String::new() },
                    if e.channels > 0 { format!("{} ch", e.channels) } else { String::new() },
                ]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" · ");
                stage(name, detail, true, false)
            };
            let mut boxes: Vec<El> = vec![ep(&p.input, "Source".into())];
            for s in &p.stages {
                let hot = s.active
                    && (s.id == "modulator"
                        || s.id == "gpu"
                        || (s.id == "headroom" && s.detail.contains("engaged")));
                boxes.push(stage(
                    format!("{}{}", s.name, if s.active { "" } else { " (bypassed)" }),
                    s.detail.clone(),
                    s.active,
                    hot,
                ));
            }
            boxes.push(ep(&p.output, format!("Output · {}", st.output)));
            let mut col = Column::new().spacing(8);
            // Rows of up to three, with arrows between.
            let mut it = boxes.into_iter().peekable();
            while it.peek().is_some() {
                let mut r = Row::new().spacing(8).align_y(Alignment::Center);
                for i in 0..3 {
                    let Some(b) = it.next() else { break };
                    if i > 0 {
                        r = r.push(text("→").color(l.faint));
                    }
                    r = r.push(b);
                }
                col = col.push(r);
            }
            col.into()
        }
    };

    let queue: El = if st.queue.is_empty() {
        ui::hint("Empty. Pick music on the Library page, drop files on the window, or send music from Encore.")
    } else {
        let mut c = Column::new().spacing(2);
        for (i, q) in st.queue.iter().enumerate() {
            let cur = st.index == Some(i);
            let label = format!(
                "{}. {}{}",
                i + 1,
                if q.title.is_empty() { "Untitled" } else { &q.title },
                if q.artist.is_empty() { String::new() } else { format!(" – {}", q.artist) }
            );
            let line = button(
                text(label)
                    .size(13.5)
                    .color(if cur { l.accent } else { l.text })
                    .font(if cur { ui::bold() } else { iced::Font::DEFAULT }),
            )
            .padding([6, 10])
            .width(Length::Fill)
            .style(theme::btn(if cur { Btn::NavOn } else { Btn::Ghost }))
            .on_press(Message::Transport(Transport::PlayIndex(i)));
            let rm = button(text("✕").size(12))
                .padding([4, 8])
                .style(theme::btn(Btn::Ghost))
                .on_press(Message::Transport(Transport::Remove(i)));
            c = c.push(row![line, rm].align_y(Alignment::Center));
        }
        c.into()
    };

    column![
        hero,
        space().height(22),
        ui::card(column![ui::h2("Signal path"), chain].spacing(12)),
        space().height(18),
        ui::card(column![ui::h2("Queue"), queue].spacing(12)),
    ]
    .into()
}

fn stage<'a>(name: String, detail: String, active: bool, hot: bool) -> El<'a> {
    let l = theme::look();
    let c = container(
        column![
            text(name).size(13).font(ui::bold()),
            text(detail).size(12).color(l.muted)
        ]
        .spacing(2),
    )
    .padding([10, 12])
    .width(Length::Fill)
    .style(move |t| {
        let mut s = if hot { theme::inset_selected(t) } else { theme::inset(t) };
        if !active {
            s.text_color = Some(theme::alpha(l.text, 0.45));
        }
        s
    });
    c.into()
}
