//! Output: the device, the format it gets, GPU offload, network endpoints.

use crate::app::{patch, App, Message};
use crate::theme::{self, Btn};
use crate::ui::{self, Choice, El};
use cadenza::config::RateSetting;
use iced::widget::{column, row, text, Column};
use iced::{Alignment, Length};
use serde_json::json;

fn device_label(d: &cadenza::output::DeviceInfo) -> String {
    let dsd = d.backend != "null" && (d.dsd.dop || !d.dsd.native.is_empty());
    format!(
        "{} · {}{}{}",
        d.backend,
        d.name,
        if d.exclusive { " · exclusive" } else { "" },
        if dsd { " · DSD" } else { "" }
    )
}

fn plain_for(backend: &str) -> &'static str {
    match backend {
        "asio" => "In plain words: ASIO talks to the device through its maker's driver, with Cadenza in sole control. Best choice when it works.",
        "wasapi" => "In plain words: Windows' own exclusive mode: Cadenza has the device to itself, nothing in between. As good as ASIO.",
        "cpal" => "In plain words: shared mode. Windows mixes all apps and may resample the sound. Pick an \"exclusive\" entry for the full effect.",
        "coreaudio" => "In plain words: macOS's own audio path, with Cadenza in control of the device.",
        "alsa" => "In plain words: Linux's direct path to the device (hw: entries are untouched by the system mixer).",
        "endpoint" => "In plain words: another computer running \"cadenza endpoint\" next to the DAC; the finished sound streams to it.",
        "null" => "In plain words: no sound. For testing only.",
        _ => "",
    }
}

pub fn view(app: &App, wide: bool) -> El<'_> {
    let s = &app.settings;
    let st = &app.status;
    // An empty device means the backend's default one.
    let want = match app.devices.iter().find(|d| {
        d.backend == s.output.backend
            && (d.id == s.output.device
                || (s.output.device.is_empty() && (d.default || d.backend == "null")))
    }) {
        Some(d) => format!("{}|{}", d.backend, d.id),
        None => format!("{}|{}", s.output.backend, s.output.device),
    };
    let mut devices: Vec<Choice> = app
        .devices
        .iter()
        .map(|d| Choice::new(format!("{}|{}", d.backend, d.id), device_label(d)))
        .collect();
    let connected = devices.iter().any(|c| c.id == want);
    if !connected && s.output.backend != "endpoint" && !app.devices_busy {
        devices.insert(
            0,
            Choice::new(
                want.clone(),
                format!("{} · {} · not connected", s.output.backend, s.output.device),
            ),
        );
    }
    let device_pick = ui::pick(devices, &want, |v| {
        let (b, d) = v.split_once('|').unwrap_or((&v, ""));
        patch(json!({ "output": { "backend": b, "device": d } }))
    });
    let refresh = ui::btn(
        if app.devices_busy {
            "Looking…"
        } else {
            "Refresh"
        },
        Btn::Secondary,
        (!app.devices_busy).then_some(Message::RefreshDevices(true)),
    );

    // What the open device can do.
    let mut caps = Vec::new();
    if !st.output.is_empty() {
        caps.push(if st.output_exclusive {
            "exclusive".to_string()
        } else {
            "shared mode".into()
        });
        if let Some(m) = st.output_rates.iter().max() {
            caps.push(format!("PCM up to {}", ui::khz(*m)));
        }
        let d = &st.output_dsd;
        caps.push(if !d.dop && d.native.is_empty() {
            "no DSD".into()
        } else {
            d.summary()
        });
    }

    let modes = ui::choices(&app.options["output_modes"], "id", "label");
    let mode = ui::pick(modes, s.dsp.output_mode.id(), |v| {
        patch(json!({ "dsp": { "output_mode": v } }))
    });
    let mut rates = vec![
        Choice::new("auto", "Auto (highest multiple of the source)"),
        Choice::new("source", "Same as the source"),
    ];
    let mut all: Vec<u32> = app.options["pcm_rates"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect()
        })
        .unwrap_or_default();
    all.extend(st.output_rates.iter().copied());
    all.sort_unstable();
    all.dedup();
    for r in all {
        let ok = st.output_rates.is_empty() || st.output_rates.contains(&r);
        rates.push(Choice::new(
            r.to_string(),
            if ok {
                ui::khz(r)
            } else {
                format!("{} — not supported by this device", ui::khz(r))
            },
        ));
    }
    let rate_now = match s.output.rate {
        RateSetting::Auto => "auto".to_string(),
        RateSetting::Source => "source".into(),
        RateSetting::Fixed(r) => r.to_string(),
    };
    let rate = ui::pick(rates, &rate_now, |v| {
        let val = v.parse::<u64>().map(|n| json!(n)).unwrap_or(json!(v));
        patch(json!({ "output": { "rate": val } }))
    });
    let bits = ui::pick(
        ["16", "24", "32"]
            .iter()
            .map(|b| Choice::new(*b, *b))
            .collect(),
        &s.output.bits.to_string(),
        |v| patch(json!({ "output": { "bits": v.parse::<u32>().unwrap_or(24) } })),
    );

    let mut dev = Column::new()
        .spacing(12)
        .push(ui::h2("Device"))
        .push(ui::field(
            "Device",
            row![device_pick, refresh]
                .spacing(8)
                .align_y(Alignment::Center),
        ));
    if !caps.is_empty() {
        dev = dev.push(ui::hint(caps.join(" · ")));
    }
    let p = plain_for(&s.output.backend);
    if !p.is_empty() && connected {
        dev = dev.push(ui::plain(p));
    }
    dev = dev
        .push(ui::field("Output mode", mode))
        .push(row![ui::field("PCM rate", rate), ui::field("PCM bits", bits)].spacing(12))
        .push(ui::plain("PCM is the normal format; DSD only if your DAC lists it. PCM rate: Auto picks the highest your device takes (the filter computes the extra points). Bits: 24 for any DAC, 16 for Bluetooth aptX Lossless (set 44.1 kHz too). The Guide page explains more."))
        .push(ui::switch(s.dsp.bit_perfect, "Bit-perfect", |on| {
            crate::app::edit(move |s| s.dsp.bit_perfect = on)
        }))
        .push(ui::hint("Send the file untouched: no upsampling, dither, EQ, matrix or volume."));

    // GPU.
    let gpu_modes = ui::choices(&app.options["gpu_modes"], "id", "label");
    let gpu_mode = serde_json::to_value(s.dsp.gpu.mode)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    let precs = ui::choices(&app.options["gpu_precisions"], "id", "label");
    let prec = serde_json::to_value(s.dsp.gpu.precision)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    let g = &app.options["gpu"];
    let info = format!(
        "{}{}{} · precision {}{}",
        g["label"].as_str().unwrap_or(""),
        g["reason"]
            .as_str()
            .map(|r| format!(" · {r}"))
            .unwrap_or_default(),
        g["compute_capability"]
            .as_str()
            .map(|r| format!(" · compute {r}"))
            .unwrap_or_default(),
        g["precision"].as_str().unwrap_or(""),
        if app.options["gpu_compiled_in"].as_bool() == Some(false) {
            " · this build has no GPU support"
        } else {
            ""
        }
    );
    let gpu_dev = app.text("gpu_device", || s.dsp.gpu.device.to_string());
    let gpu = column![
        ui::h2("GPU"),
        row![
            ui::field(
                "Offload",
                ui::pick(gpu_modes, &gpu_mode, |v| patch(
                    json!({ "dsp": { "gpu": v } })
                ))
            ),
            ui::field(
                "Device",
                ui::input("0", &gpu_dev, |v| Message::Text("gpu_device".into(), v))
                    .on_submit(Message::TextCommit("gpu_device".into()))
            ),
        ]
        .spacing(12),
        ui::field(
            "Precision (Vulkan / Metal / DX12)",
            ui::pick(precs, &prec, |v| patch(
                json!({ "dsp": { "gpu_precision": v } })
            ))
        ),
        ui::hint(info),
    ]
    .spacing(12);

    // Endpoints.
    let eps: Vec<_> = app
        .devices
        .iter()
        .filter(|d| d.backend == "endpoint")
        .collect();
    let mut ep = Column::new().spacing(8).push(ui::h2("Endpoints"));
    if eps.is_empty() {
        ep = ep.push(ui::hint("No endpoint found on the network yet."));
    }
    for d in eps {
        let id = d.id.clone();
        ep = ep.push(
            row![
                text(d.name.clone()).font(ui::bold()).size(14),
                text(d.id.clone())
                    .size(12.5)
                    .color(theme::look().muted)
                    .width(Length::Fill),
                ui::btn(
                    "Use as output",
                    Btn::Secondary,
                    Some(patch(
                        json!({ "output": { "backend": "endpoint", "device": id } })
                    ))
                ),
            ]
            .spacing(10)
            .align_y(Alignment::Center),
        );
    }
    ep = ep.push(ui::hint("Run `cadenza endpoint` on the box next to the DAC; it appears here and as a device. The final PCM or DSD words stream to it over TCP, paced by its DAC clock."));

    let mut page = column![ui::h1("Output", "Where the sound goes, and in what form.")];
    if let Some(e) = &app.error {
        page = page.push(ui::err(e.clone()));
    }
    page.push(ui::grid(vec![ui::card(dev), ui::card(gpu)], wide))
        .push(iced::widget::space().height(18))
        .push(ui::card(ep))
        .into()
}

/// The GPU device box.
pub fn commit_gpu_device(app: &mut App, v: String) {
    if let Ok(n) = v.trim().parse::<usize>() {
        let _ = app.update(crate::app::edit(move |s| s.dsp.gpu.device = n));
    }
}
