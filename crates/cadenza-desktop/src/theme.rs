//! The look: the same themes and accents as the web page (and Encore), as colour tokens
//! the widget styles read. One look is active at a time; changing it repaints everything.

use iced::border::Radius;
use iced::widget::{button, container, pick_list, scrollable, slider, text_input, toggler};
use iced::{Background, Border, Color, Shadow, Theme, Vector};
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

/// Colour tokens (the web page's CSS variables).
#[derive(Clone, Copy, Debug)]
pub struct Look {
    pub bg: Color,
    pub bg2: Color,
    pub panel: Color,
    pub panel2: Color,
    pub border: Color,
    pub text: Color,
    pub muted: Color,
    pub faint: Color,
    pub accent: Color,
    pub accent2: Color,
    pub on_accent: Color,
    pub ok: Color,
    pub warn: Color,
    pub err: Color,
    pub dark: bool,
}

/// What the Appearance page stores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Appearance {
    /// `encore`, `dark`, `oled`, `light`.
    pub theme: String,
    /// An [`ACCENTS`] id or `custom`.
    pub accent: String,
    /// `#rrggbb` for `custom`.
    pub custom: String,
}

impl Default for Appearance {
    fn default() -> Self {
        Self {
            theme: "encore".into(),
            accent: "logo".into(),
            custom: "#f0ead6".into(),
        }
    }
}

/// `(id, name, colour, second colour)`.
pub const ACCENTS: [(&str, &str, &str, &str); 10] = [
    ("logo", "Logo", "#f0ead6", "#d6c49a"),
    ("violet", "Violet", "#8b5cf6", "#ec4899"),
    ("encore", "Encore", "#f2a7e4", "#9fd2fb"),
    ("lilac", "Lilac", "#b48cf2", "#7cc4fa"),
    ("pink", "Pink", "#ec4899", "#f97316"),
    ("sky", "Sky", "#0ea5e9", "#8b5cf6"),
    ("teal", "Teal", "#14b8a6", "#0ea5e9"),
    ("green", "Green", "#22c55e", "#14b8a6"),
    ("amber", "Amber", "#f59e0b", "#ef4444"),
    ("red", "Red", "#ef4444", "#ec4899"),
];

/// `(id, name, theme, accent, custom colour)`.
pub const LOOKS: [(&str, &str, &str, &str, &str); 6] = [
    ("encore", "Encore", "encore", "logo", ""),
    ("default", "Violet dark", "dark", "violet", ""),
    ("oled", "OLED black", "oled", "violet", ""),
    ("pastel", "Pastel", "dark", "encore", ""),
    ("light", "Daylight", "light", "sky", ""),
    ("mono", "Mono", "oled", "custom", "#e5e5e5"),
];

/// `#rrggbb` (or `#rgb`) to a colour; grey when it does not parse.
pub fn hex(s: &str) -> Color {
    let h = s.trim().trim_start_matches('#');
    let h: String = if h.len() == 3 {
        h.chars().flat_map(|c| [c, c]).collect()
    } else {
        h.to_string()
    };
    match u32::from_str_radix(&h, 16) {
        Ok(n) if h.len() == 6 => Color::from_rgb8((n >> 16) as u8, (n >> 8) as u8, n as u8),
        _ => Color::from_rgb8(128, 128, 128),
    }
}

fn luminance(c: Color) -> f32 {
    let f = |v: f32| {
        if v <= 0.03928 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b)
}

/// A second colour for a custom accent: the hue turned a little (as the web page does).
fn partner(c: Color) -> Color {
    let (r, g, b) = (c.r, c.g, c.b);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    let (mut h, mut s) = (0.0, 0.0);
    if max != min {
        let d = max - min;
        s = if l > 0.5 {
            d / (2.0 - max - min)
        } else {
            d / (max + min)
        };
        h = if max == r {
            (g - b) / d + if g < b { 6.0 } else { 0.0 }
        } else if max == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        } / 6.0;
    }
    h = (h + 0.1) % 1.0;
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let f = |t: f32| {
        let t = (t + 1.0) % 1.0;
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    Color::from_rgb(f(h + 1.0 / 3.0), f(h), f(h - 1.0 / 3.0))
}

impl Look {
    pub fn from(a: &Appearance) -> Self {
        let (bg, bg2, panel, panel2, border, text, muted, faint, dark) = match a.theme.as_str() {
            "dark" => (
                "#0b0b12", "#12121c", "#161624", "#1d1d2e", "#2a2a40", "#ececf4", "#9a9ab3",
                "#6b6b85", true,
            ),
            "oled" => (
                "#000000", "#000000", "#0b0b0d", "#17171b", "#232329", "#ececf4", "#9a9aa8",
                "#66666f", true,
            ),
            "light" => (
                "#f6f6fb", "#ffffff", "#ffffff", "#f0f0f7", "#e2e2ee", "#16161f", "#5d5d75",
                "#8d8da5", false,
            ),
            _ => (
                "#161617", "#1c1c1e", "#222224", "#2a2a2d", "#353538", "#f0ead6", "#aaa493",
                "#77725f", true,
            ),
        };
        let (accent, accent2) = match ACCENTS.iter().find(|x| x.0 == a.accent) {
            Some(x) => (hex(x.2), hex(x.3)),
            None => {
                let c = hex(&a.custom);
                (c, partner(c))
            }
        };
        let on_accent = if (luminance(accent) + luminance(accent2)) / 2.0 > 0.45 {
            hex("#16161f")
        } else {
            Color::WHITE
        };
        Self {
            bg: hex(bg),
            bg2: hex(bg2),
            panel: hex(panel),
            panel2: hex(panel2),
            border: hex(border),
            text: hex(text),
            muted: hex(muted),
            faint: hex(faint),
            accent,
            accent2,
            on_accent,
            ok: hex("#22c55e"),
            warn: hex("#f59e0b"),
            err: hex("#ef4444"),
            dark,
        }
    }

    /// The iced theme for the built-in widgets.
    pub fn theme(&self) -> Theme {
        Theme::custom(
            "Cadenza",
            iced::theme::Palette {
                background: self.bg,
                text: self.text,
                primary: self.accent,
                success: self.ok,
                warning: self.warn,
                danger: self.err,
            },
        )
    }
}

static LOOK: RwLock<Option<Look>> = RwLock::new(None);

/// The active look.
pub fn look() -> Look {
    LOOK.read()
        .ok()
        .and_then(|l| *l)
        .unwrap_or_else(|| Look::from(&Appearance::default()))
}

/// Make `a` the active look.
pub fn set(a: &Appearance) {
    if let Ok(mut l) = LOOK.write() {
        *l = Some(Look::from(a));
    }
}

pub fn alpha(c: Color, a: f32) -> Color {
    Color { a, ..c }
}

fn mix(a: Color, b: Color, t: f32) -> Color {
    Color::from_rgba(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
        a.a + (b.a - a.a) * t,
    )
}

fn border(color: Color, width: f32, radius: f32) -> Border {
    Border {
        color,
        width,
        radius: Radius::from(radius),
    }
}

// ----- containers ----------------------------------------------------------------------

pub fn app_bg(_: &Theme) -> container::Style {
    let l = look();
    container::Style {
        background: Some(Background::Color(l.bg)),
        text_color: Some(l.text),
        ..Default::default()
    }
}

pub fn sidebar(_: &Theme) -> container::Style {
    let l = look();
    container::Style {
        background: Some(Background::Color(l.bg2)),
        border: Border {
            color: l.border,
            width: 0.0,
            radius: 0.0.into(),
        },
        ..Default::default()
    }
}

pub fn player(_: &Theme) -> container::Style {
    let l = look();
    container::Style {
        background: Some(Background::Color(alpha(l.bg2, 0.97))),
        border: border(l.border, 1.0, 0.0),
        ..Default::default()
    }
}

pub fn card(_: &Theme) -> container::Style {
    let l = look();
    container::Style {
        background: Some(Background::Color(l.panel)),
        border: border(l.border, 1.0, 14.0),
        shadow: if l.dark && l.bg != Color::BLACK {
            Shadow {
                color: alpha(Color::BLACK, 0.25),
                offset: Vector::new(0.0, 6.0),
                blur_radius: 18.0,
            }
        } else {
            Shadow::default()
        },
        ..Default::default()
    }
}

/// The lighter inset box (plain-words notes, list rows).
pub fn inset(_: &Theme) -> container::Style {
    let l = look();
    container::Style {
        background: Some(Background::Color(l.panel2)),
        border: border(Color::TRANSPARENT, 0.0, 10.0),
        ..Default::default()
    }
}

pub fn inset_selected(_: &Theme) -> container::Style {
    let l = look();
    container::Style {
        background: Some(Background::Color(l.panel2)),
        border: border(l.accent, 2.0, 10.0),
        ..Default::default()
    }
}

/// The logo square / cover placeholder.
pub fn accent_tile(_: &Theme) -> container::Style {
    let l = look();
    container::Style {
        background: Some(Background::Gradient(
            iced::gradient::Linear::new(iced::Radians(2.35))
                .add_stop(0.0, l.accent)
                .add_stop(1.0, l.accent2)
                .into(),
        )),
        border: border(Color::TRANSPARENT, 0.0, 10.0),
        text_color: Some(l.on_accent),
        ..Default::default()
    }
}

/// A small pill (`on`, `exclusive`, …); `kind` picks the colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Badge {
    Plain,
    Ok,
    Warn,
    Bad,
    Accent,
}

pub fn badge(kind: Badge) -> impl Fn(&Theme) -> container::Style {
    move |_| {
        let l = look();
        let c = match kind {
            Badge::Plain => l.muted,
            Badge::Ok => l.ok,
            Badge::Warn => l.warn,
            Badge::Bad => l.err,
            Badge::Accent => mix(l.accent, l.text, 0.45),
        };
        let (bg, line) = match kind {
            Badge::Plain => (l.panel2, l.border),
            Badge::Accent => (alpha(l.accent, 0.14), alpha(l.accent, 0.4)),
            _ => (alpha(c, 0.1), alpha(c, 0.35)),
        };
        container::Style {
            background: Some(Background::Color(bg)),
            border: border(line, 1.0, 99.0),
            text_color: Some(c),
            ..Default::default()
        }
    }
}

// ----- buttons -------------------------------------------------------------------------

/// `Primary`: the accent; `Secondary`: a panel button; `Ghost`: no frame; `Nav` / `NavOn`:
/// the sidebar; `Seg` / `SegOn`: segmented choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Btn {
    Primary,
    Secondary,
    Ghost,
    Nav,
    NavOn,
    Seg,
    SegOn,
}

pub fn btn(kind: Btn) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let l = look();
        let hover = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let disabled = matches!(status, button::Status::Disabled);
        let mut s = match kind {
            Btn::Primary => button::Style {
                background: Some(Background::Gradient(
                    iced::gradient::Linear::new(iced::Radians(2.35))
                        .add_stop(0.0, l.accent)
                        .add_stop(1.0, if hover { l.accent } else { l.accent2 })
                        .into(),
                )),
                text_color: l.on_accent,
                border: border(Color::TRANSPARENT, 0.0, 10.0),
                ..Default::default()
            },
            Btn::Secondary => button::Style {
                background: Some(Background::Color(if hover { l.border } else { l.panel2 })),
                text_color: l.text,
                border: border(l.border, 1.0, 10.0),
                ..Default::default()
            },
            Btn::Ghost => button::Style {
                background: hover.then_some(Background::Color(l.panel2)),
                text_color: l.text,
                border: border(Color::TRANSPARENT, 0.0, 10.0),
                ..Default::default()
            },
            Btn::Nav => button::Style {
                background: hover.then_some(Background::Color(l.panel2)),
                text_color: l.muted,
                border: border(Color::TRANSPARENT, 0.0, 10.0),
                ..Default::default()
            },
            Btn::NavOn => button::Style {
                background: Some(Background::Color(l.panel2)),
                text_color: l.text,
                border: Border {
                    color: l.accent,
                    width: 0.0,
                    radius: 10.0.into(),
                },
                ..Default::default()
            },
            Btn::Seg => button::Style {
                background: hover.then_some(Background::Color(l.panel)),
                text_color: l.muted,
                border: border(Color::TRANSPARENT, 0.0, 8.0),
                ..Default::default()
            },
            Btn::SegOn => button::Style {
                background: Some(Background::Color(l.panel)),
                text_color: l.text,
                border: border(l.border, 1.0, 8.0),
                ..Default::default()
            },
        };
        if disabled {
            s.text_color = alpha(s.text_color, 0.45);
            if let Some(Background::Color(c)) = s.background {
                s.background = Some(Background::Color(alpha(c, c.a * 0.5)));
            }
        }
        s
    }
}

/// The round play button.
pub fn play_btn(_: &Theme, status: button::Status) -> button::Style {
    let l = look();
    let hover = matches!(status, button::Status::Hovered | button::Status::Pressed);
    button::Style {
        background: Some(Background::Color(if hover {
            mix(l.accent, l.text, 0.2)
        } else {
            l.accent
        })),
        text_color: l.on_accent,
        border: border(Color::TRANSPARENT, 0.0, 99.0),
        ..Default::default()
    }
}

/// A swatch button filled with `c` (ringed when chosen).
pub fn swatch(c: Color, on: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, _| {
        let l = look();
        button::Style {
            background: Some(Background::Color(c)),
            text_color: l.text,
            border: border(
                if on { l.text } else { l.border },
                if on { 3.0 } else { 1.0 },
                99.0,
            ),
            ..Default::default()
        }
    }
}

// ----- inputs --------------------------------------------------------------------------

pub fn input(_: &Theme, status: text_input::Status) -> text_input::Style {
    let l = look();
    let focused = matches!(status, text_input::Status::Focused { .. });
    text_input::Style {
        background: Background::Color(l.panel2),
        border: border(if focused { l.accent } else { l.border }, 1.0, 10.0),
        icon: l.muted,
        placeholder: l.faint,
        value: l.text,
        selection: alpha(l.accent, 0.35),
    }
}

pub fn pick(_: &Theme, status: pick_list::Status) -> pick_list::Style {
    let l = look();
    let hot = matches!(
        status,
        pick_list::Status::Hovered | pick_list::Status::Opened { .. }
    );
    pick_list::Style {
        text_color: l.text,
        placeholder_color: l.faint,
        handle_color: l.muted,
        background: Background::Color(l.panel2),
        border: border(if hot { l.accent } else { l.border }, 1.0, 10.0),
    }
}

pub fn menu(_: &Theme) -> iced::overlay::menu::Style {
    let l = look();
    iced::overlay::menu::Style {
        background: Background::Color(l.panel),
        border: border(l.border, 1.0, 10.0),
        text_color: l.text,
        selected_text_color: l.on_accent,
        selected_background: Background::Color(l.accent),
        shadow: Shadow {
            color: alpha(Color::BLACK, 0.35),
            offset: Vector::new(0.0, 8.0),
            blur_radius: 24.0,
        },
    }
}

pub fn toggle(_: &Theme, status: toggler::Status) -> toggler::Style {
    let l = look();
    let on = matches!(
        status,
        toggler::Status::Active { is_toggled: true }
            | toggler::Status::Hovered { is_toggled: true }
            | toggler::Status::Disabled { is_toggled: true }
    );
    toggler::Style {
        background: Background::Color(if on { l.accent } else { l.panel2 }),
        background_border_width: 1.0,
        background_border_color: if on { l.accent } else { l.border },
        foreground: Background::Color(if on { l.on_accent } else { l.muted }),
        foreground_border_width: 0.0,
        foreground_border_color: Color::TRANSPARENT,
        text_color: Some(l.text),
        border_radius: None,
        padding_ratio: 0.1,
    }
}

pub fn range(_: &Theme, status: slider::Status) -> slider::Style {
    let l = look();
    let hot = matches!(status, slider::Status::Hovered | slider::Status::Dragged);
    slider::Style {
        rail: slider::Rail {
            backgrounds: (Background::Color(l.accent), Background::Color(l.panel2)),
            width: 6.0,
            border: border(Color::TRANSPARENT, 0.0, 3.0),
        },
        handle: slider::Handle {
            shape: slider::HandleShape::Circle {
                radius: if hot { 8.0 } else { 7.0 },
            },
            background: Background::Color(l.text),
            border_width: 0.0,
            border_color: Color::TRANSPARENT,
        },
    }
}

pub fn scroll(theme: &Theme, status: scrollable::Status) -> scrollable::Style {
    let l = look();
    let mut s = scrollable::default(theme, status);
    let rail = |_: scrollable::Rail| scrollable::Rail {
        background: None,
        border: border(Color::TRANSPARENT, 0.0, 4.0),
        scroller: scrollable::Scroller {
            background: Background::Color(alpha(l.muted, 0.35)),
            border: border(Color::TRANSPARENT, 0.0, 4.0),
        },
    };
    s.vertical_rail = rail(s.vertical_rail);
    s.horizontal_rail = rail(s.horizontal_rail);
    s.container = container::Style::default();
    s
}

/// A plain box filled with `c`, for colour previews.
pub fn swatch_box(c: Color) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(c)),
        border: border(look().border, 1.0, 12.0),
        ..Default::default()
    }
}
