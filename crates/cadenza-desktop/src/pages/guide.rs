//! Guide: what Cadenza does and what to pick, in plain words.

use crate::app::App;
use crate::theme;
use crate::ui::{self, El};
use iced::widget::{column, rich_text, span, text, Column};

fn p<'a>(t: &'a str) -> El<'a> {
    text(t).size(14).line_height(1.45).into()
}

/// A line with a bold start: `("Output → Device:", " pick your DAC…")`.
fn lead<'a>(b: &'a str, rest: &'a str) -> El<'a> {
    let r: iced::widget::text::Rich<'a, (), crate::app::Message> =
        rich_text([span(b).font(ui::bold()), span(rest)]);
    r.size(14).line_height(1.45).into()
}

fn h3<'a>(t: &'a str) -> El<'a> {
    text(t).size(15).font(ui::bold()).into()
}

/// A glossary entry.
fn term<'a>(name: &'a str, what: &'a str) -> El<'a> {
    column![
        text(name).size(14).font(ui::bold()),
        text(what).size(13.5).color(theme::look().muted).line_height(1.45)
    ]
    .spacing(2)
    .into()
}

fn card<'a>(title: &'a str, items: Vec<El<'a>>) -> El<'a> {
    ui::card(column![ui::h2(title), Column::with_children(items).spacing(10)].spacing(12))
}

pub fn view(_app: &App, wide: bool) -> El<'static> {
    let what = card(
        "What Cadenza does",
        vec![
            p("Your music files store sound as numbers: usually 44,100 numbers a second per channel (CD quality). Your DAC (the thing your headphones plug into) turns those numbers back into sound."),
            p("Before the numbers reach the DAC, Cadenza can upsample them: calculate the in-between points very precisely, so the DAC gets 8, 16 or 32 times more points. DACs do this themselves too, but with small, cheap filters built into the chip. Your computer can run a far bigger, more precise filter."),
            p("Whether you hear a difference depends on your DAC and your ears. Nothing here makes the sound worse, and you can always turn on Bit-perfect (Output page) to send the music untouched."),
        ],
    );
    let quick = card(
        "Quick setup: 4 steps",
        vec![
            lead("1. Output → Device: ", "pick your DAC. Prefer asio or wasapi … exclusive: Cadenza then has the device to itself and Windows can't change the sound."),
            lead("2. Output → PCM rate: ", "leave it on Auto. It picks the highest rate your device takes. (Bluetooth: see below.)"),
            lead("3. Sound → Find my limit: ", "press Run the test, then Use next to a rate. That picks the longest filter your PC handles there without stutter."),
            lead("4. Play something. ", "The Now playing page shows the whole path the sound takes."),
            text("Something wrong? Output → Refresh looks for devices again (after plugging something in), and errors show under the song title at the bottom.")
                .size(12.5)
                .color(theme::look().muted)
                .into(),
        ],
    );
    let recipes = card(
        "Your setup: which recipe?",
        vec![
            h3("USB DAC or dongle (cable to headphones)"),
            p("Device: asio or wasapi exclusive. PCM rate: Auto. Bits: 24 or 32. Filter: from Find my limit. If the device name says DSD, you can also try Output mode → DSD (see DSD below)."),
            h3("Bluetooth headphones (e.g. through a FiiO Air Link)"),
            p("Bluetooth squeezes the sound to fit the radio link. The best mode, aptX Lossless, is only lossless at 44.1 kHz / 16-bit. So set PCM rate 44.1 kHz and bits 16. At that rate no upsampling happens, so the filter choice doesn't matter; Cadenza still adds EQ, volume and clean dither. Higher rates make the link switch to a lossy mode, which is worse."),
            h3("Laptop speakers or \"Speakers (…)\" without \"exclusive\""),
            p("That's Windows' shared mode: Windows mixes and resamples everything itself. Fine for casual listening; pick an exclusive device for the full effect."),
        ],
    );
    let filters = card(
        "Filters, explained",
        vec![
            p("Upsampling needs a filter: the recipe for computing the in-between points. Filter names look like sinc-l-1m:"),
            term("The length (4k … 16m)", "How many points of the music the filter looks at for each new point (\"taps\"). Longer = sharper cut-off above the audible range and more precise timing, but more work for your PC. 64k is already far beyond what DAC chips do; 1M and up is \"as good as it gets\". Find my limit tells you how far your PC goes."),
            term("l = linear phase", "Treats every frequency the same in time. The faint \"ringing\" of the filter is spread before and after each sound. The usual, most accurate choice."),
            term("m = minimum phase", "All the ringing comes after the sound, none before. Some people find it sounds more natural on drums; timing between frequencies shifts slightly."),
            term("a = apodizing (la, ma)", "Cuts a little earlier, just below the top of the range, which also cleans up ringing that was already baked into the recording. Good for older CDs."),
            term("Families", "sinc: the classic, good default. closed-form: sharpest cut for its length. poly-sinc-gauss: gentlest, least ringing. sinc-xtr: extremely long (8M/16M), for strong PCs. sinc-mx: keeps the same shape at every output rate. poly-sinc-ext3: short and light, for weak PCs."),
            term("Delay", "Very long filters hold back the music while they fill: 16M taps is about 12 seconds before the first sound, and after every jump in the track. Shorter filters start almost at once."),
            text("Not sure? sinc-l at the longest length Find my limit marks safe.")
                .size(12.5)
                .color(theme::look().muted)
                .into(),
        ],
    );
    let words = card(
        "Words you'll see",
        vec![
            term("Exclusive mode", "Cadenza has the device to itself: no Windows mixer, no other app's sounds, no hidden resampling. Other apps go silent on that device while Cadenza plays."),
            term("ASIO / WASAPI", "Two ways Windows talks to sound devices. Both can be exclusive. ASIO comes from the device maker's driver (e.g. \"FiiO ASIO Driver\"); WASAPI is Windows' own. Sound is the same; use whichever works."),
            term("Sample rate (kHz)", "How many points per second. 44.1 kHz is CD. 705.6 kHz is 16× that. Higher only helps if a good filter computes the extra points."),
            term("Bit depth (16 / 24 / 32)", "How fine each point is. 24-bit is more than enough for any DAC; 16 only for Bluetooth or old devices."),
            term("Dither / noise shaping", "When the numbers are rounded to fit the bit depth, dither turns the rounding error into a faint, even hiss far below hearing. Noise shapers push that hiss up to frequencies you can't hear. \"Auto\" picks for you."),
            term("DSD", "A different format: one bit per point, millions of points per second. Some DACs sound their best on it. A modulator converts the music to DSD; higher DSD rates (DSD256, 512…) need much more computing."),
            term("Bit-perfect", "Sends the file exactly as it is: no upsampling, EQ or volume change. Useful to compare."),
            term("Headroom / soft clipper", "Filters can briefly overshoot on very loud recordings. Headroom lowers the level a little (−3 dB) so nothing distorts; the soft clipper rounds off anything that still would."),
            term("EQ, room correction, crossfeed", "EQ changes the tone. Room correction (Room & measure page) fixes your room's echoes for speakers. Crossfeed (Headphones) makes headphones sound more like speakers in front of you."),
            term("GPU", "Your graphics card can run very long filters faster than the CPU. \"Automatic\" uses it when it is there."),
            term("Speed 7.5× (Find my limit)", "Your PC computes 7.5 seconds of music per second. Anything under 1× stutters; Cadenza calls 1.5× and up safe."),
        ],
    );
    column![
        ui::h1("Guide", "What Cadenza does and what to pick, in plain words. No experience needed."),
        ui::grid(vec![what, quick, recipes, filters, words], wide)
    ]
    .into()
}
