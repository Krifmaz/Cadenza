//! End-to-end: generate WAV files, play them through the real engine with the null output
//! and check what came out the other side.

use cadenza::config::{RateSetting, Settings};
use cadenza::engine::{Command, Engine, EngineOptions, Event, PlayState, Track};
use cadenza_dsp::CoefficientCache;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("cadenza-it-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_sine_wav(path: &Path, rate: u32, bits: u16, seconds: f64, freq: f64) -> usize {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: bits,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    let frames = (rate as f64 * seconds) as usize;
    let amp = ((1u64 << (bits - 1)) as f64 - 1.0) * 0.5;
    for i in 0..frames {
        let v = (amp * (2.0 * PI * freq * i as f64 / rate as f64).sin()).round() as i32;
        if bits == 16 {
            w.write_sample(v as i16).unwrap();
            w.write_sample(v as i16).unwrap();
        } else {
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
    }
    w.finalize().unwrap();
    frames
}

fn settings(
    rate: RateSetting,
    bits: u32,
    filter: &str,
    dither: &str,
    bit_perfect: bool,
) -> Settings {
    let mut s = Settings::default();
    s.output.backend = "null".into();
    s.output.rate = rate;
    s.output.bits = bits;
    s.dsp.filter = filter.into();
    s.dsp.dither = dither.into();
    s.dsp.bit_perfect = bit_perfect;
    s
}

fn engine(s: Settings) -> Engine {
    Engine::start(
        s,
        EngineOptions {
            cache: CoefficientCache::disabled(),
            realtime: false,
            persist: None,
        },
    )
}

#[test]
fn plays_a_wav_upsampled_x4_with_tpdf_to_the_null_output() {
    let dir = temp_dir("x4");
    let wav = dir.join("tone.wav");
    let frames = write_sine_wav(&wav, 44100, 16, 1.5, 1000.0);

    let e = engine(settings(
        RateSetting::Fixed(176_400),
        24,
        "sinc-l-4k",
        "tpdf",
        false,
    ));
    let mut events = e.subscribe();
    let runs = e.status().runs_finished;
    e.play_queue(vec![Track::from_arg(wav.to_str().unwrap())], 0, 0.0);
    let final_status = e.wait_until_finished(runs, Duration::from_secs(60));
    assert_eq!(
        final_status.runs_finished,
        runs + 1,
        "playback did not finish in time"
    );

    assert_eq!(
        final_status.state,
        PlayState::Stopped,
        "{:?}",
        final_status.error
    );
    assert_eq!(final_status.error, None);
    assert_eq!(
        final_status.output_frames,
        frames as u64 * 4,
        "every input frame became four output frames"
    );
    let fmt = final_status.output_format.unwrap();
    assert_eq!((fmt.rate, fmt.bits, fmt.channels), (176_400, 24, 2));

    // Somewhere along the way the status carried the signal path of a running chain.
    let mut saw_path = false;
    let mut started = 0;
    while let Ok(ev) = events.try_recv() {
        match ev {
            Event::Status(s) => {
                if let Some(p) = &s.signal_path {
                    saw_path = true;
                    assert_eq!(p.label, "PCM 16/44.1 → 24/176.4");
                    assert!(!p.bit_perfect);
                    assert!(p.stages.iter().any(|st| st.id == "resampler" && st.active));
                    assert!(p.stages.iter().any(|st| st.id == "dither" && st.active));
                    assert_eq!(s.source_format.as_deref(), Some("PCM 16/44.1"));
                }
            }
            Event::TrackStarted(t) => {
                started += 1;
                assert_eq!(t.title, "tone.wav");
            }
            Event::TrackPlayed { seconds, .. } => assert!(seconds >= 0.74),
        }
    }
    assert!(saw_path, "no status with a signal path was published");
    assert_eq!(started, 1);
    e.shutdown();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn bit_perfect_queue_is_gapless_and_untouched() {
    let dir = temp_dir("bp");
    let a = dir.join("a.wav");
    let b = dir.join("b.wav");
    let fa = write_sine_wav(&a, 48000, 24, 0.7, 440.0);
    let fb = write_sine_wav(&b, 48000, 24, 0.4, 880.0);

    let e = engine(settings(
        RateSetting::Auto,
        16,
        "sinc-l-64k",
        "lipshitz",
        true,
    ));
    let mut events = e.subscribe();
    let runs = e.status().runs_finished;
    e.play_queue(
        vec![
            Track::from_arg(a.to_str().unwrap()),
            Track::from_arg(b.to_str().unwrap()),
        ],
        0,
        0.0,
    );
    let st = e.wait_until_finished(runs, Duration::from_secs(60));
    assert_eq!(
        st.runs_finished,
        runs + 1,
        "playback did not finish in time"
    );
    assert_eq!(st.error, None);
    assert_eq!(
        st.output_frames,
        (fa + fb) as u64,
        "bit-perfect: one output frame per input frame, no gaps"
    );
    let fmt = st.output_format.unwrap();
    assert_eq!(
        (fmt.rate, fmt.bits),
        (48000, 24),
        "24-bit source is never truncated to the configured 16 bits"
    );

    let mut started = Vec::new();
    while let Ok(ev) = events.try_recv() {
        match ev {
            Event::TrackStarted(t) => started.push(t.title),
            Event::Status(s) => {
                if let Some(p) = &s.signal_path {
                    assert!(p.bit_perfect, "{p:?}");
                    assert_eq!(p.label, "PCM 24/48 · bit-perfect");
                }
            }
            _ => {}
        }
    }
    assert_eq!(started, ["a.wav", "b.wav"]);
    e.shutdown();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn format_change_between_tracks_rebuilds_the_chain() {
    let dir = temp_dir("fmt");
    let a = dir.join("a.wav");
    let b = dir.join("b.wav");
    let fa = write_sine_wav(&a, 44100, 16, 0.5, 440.0);
    let fb = write_sine_wav(&b, 48000, 24, 0.5, 440.0);

    let e = engine(settings(RateSetting::Auto, 24, "sinc-l-4k", "tpdf", false));
    let mut s = e.settings();
    s.output.max_rate = 192_000;
    e.update_settings(s);
    let runs = e.status().runs_finished;
    e.play_queue(
        vec![
            Track::from_arg(a.to_str().unwrap()),
            Track::from_arg(b.to_str().unwrap()),
        ],
        0,
        0.0,
    );
    let st = e.wait_until_finished(runs, Duration::from_secs(60));
    assert_eq!(
        st.runs_finished,
        runs + 1,
        "playback did not finish in time"
    );
    assert_eq!(st.error, None);
    // 44.1k ×4 → 176.4k, then 48k ×4 → 192k: the counter spans both opens.
    assert_eq!(st.output_frames, (fa * 4 + fb * 4) as u64);
    assert_eq!(st.output_format.unwrap().rate, 192_000);
    e.shutdown();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn transport_commands_pause_seek_and_stop() {
    let dir = temp_dir("transport");
    let a = dir.join("a.wav");
    write_sine_wav(&a, 44100, 16, 30.0, 440.0);

    let mut s = settings(RateSetting::Source, 16, "sinc-l-4k", "none", false);
    s.output.backend = "null".into();
    // A realtime null output so the track does not finish instantly.
    let e = Engine::start(
        s,
        EngineOptions {
            cache: CoefficientCache::disabled(),
            realtime: true,
            persist: None,
        },
    );
    e.play_queue(vec![Track::from_arg(a.to_str().unwrap())], 0, 5.0);
    std::thread::sleep(Duration::from_millis(600));
    let st = e.status();
    assert_eq!(st.state, PlayState::Playing, "{:?}", st.error);
    assert!(
        st.position > 4.0 && st.position < 7.0,
        "position {}",
        st.position
    );
    assert!(
        st.signal_path.as_ref().unwrap().bit_perfect,
        "16-bit source, same rate, no dither, 0 dB"
    );

    e.send(Command::Pause);
    std::thread::sleep(Duration::from_millis(300));
    let p1 = e.status();
    assert_eq!(p1.state, PlayState::Paused);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(
        e.status().position,
        p1.position,
        "paused position does not move"
    );

    e.send(Command::Seek(20.0));
    e.send(Command::Play);
    std::thread::sleep(Duration::from_millis(600));
    let st = e.status();
    assert_eq!(st.state, PlayState::Playing);
    assert!(
        st.position > 19.0 && st.position < 22.0,
        "position after seek {}",
        st.position
    );

    e.send(Command::SetVolume(cadenza_dsp::Volume {
        gain_db: -6.0,
        muted: false,
    }));
    std::thread::sleep(Duration::from_millis(300));
    let st = e.status();
    assert_eq!(st.volume.gain_db, -6.0);
    assert!(
        !st.signal_path.as_ref().unwrap().bit_perfect,
        "volume below 0 dB is not bit-perfect"
    );

    e.send(Command::Stop);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(e.status().state, PlayState::Stopped);
    e.shutdown();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_unplugged_device_stops_playback_and_play_picks_up_where_it_was() {
    let dir = temp_dir("unplug");
    let a = dir.join("a.wav");
    write_sine_wav(&a, 44100, 16, 30.0, 440.0);
    let gone = dir.join("gone");
    let _ = std::fs::remove_file(&gone);

    let mut s = settings(RateSetting::Source, 16, "sinc-l-4k", "none", false);
    s.output.device = format!("unplug={}", gone.display());
    let e = Engine::start(
        s,
        EngineOptions {
            cache: CoefficientCache::disabled(),
            realtime: true,
            persist: None,
        },
    );
    e.play_queue(vec![Track::from_arg(a.to_str().unwrap())], 0, 5.0);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(
        e.status().state,
        PlayState::Playing,
        "{:?}",
        e.status().error
    );

    // Unplugged: playback stops with a message instead of running on into nothing.
    std::fs::write(&gone, b"").unwrap();
    std::thread::sleep(Duration::from_millis(400));
    let st = e.status();
    assert_eq!(st.state, PlayState::Stopped);
    let err = st.error.clone().unwrap_or_default();
    assert!(err.contains("output lost"), "{err}");

    // Play while still unplugged: it stops again, still saying why.
    e.send(Command::Play);
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(e.status().state, PlayState::Stopped);

    // Plugged back in: play carries on from where the device went away.
    std::fs::remove_file(&gone).unwrap();
    e.send(Command::Play);
    std::thread::sleep(Duration::from_millis(500));
    let st = e.status();
    assert_eq!(st.state, PlayState::Playing, "{:?}", st.error);
    assert!(st.error.is_none(), "{:?}", st.error);
    assert!(
        st.position > 5.0 && st.position < 8.0,
        "position {}",
        st.position
    );

    // Release (the limit test): the device is let go, play carries on from the same place.
    let before = e.status().position;
    e.send(Command::Release);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(e.status().state, PlayState::Stopped);
    e.send(Command::Play);
    std::thread::sleep(Duration::from_millis(400));
    let st = e.status();
    assert_eq!(st.state, PlayState::Playing, "{:?}", st.error);
    assert!(
        (st.position - before).abs() < 1.5,
        "{} vs {before}",
        st.position
    );

    // Reconnect while playing keeps playing from about the same place.
    let before = e.status().position;
    e.send(Command::Reconnect);
    std::thread::sleep(Duration::from_millis(400));
    let st = e.status();
    assert_eq!(st.state, PlayState::Playing, "{:?}", st.error);
    assert!(
        (st.position - before).abs() < 1.5,
        "{} vs {before}",
        st.position
    );
    e.shutdown();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn eq_changes_while_playing_retune_without_a_rebuild() {
    let dir = temp_dir("eq-live");
    let a = dir.join("a.wav");
    write_sine_wav(&a, 44100, 16, 30.0, 440.0);
    let mut s = settings(RateSetting::Fixed(88200), 24, "sinc-l-4k", "tpdf", false);
    s.dsp.eq = vec![cadenza_dsp::Band::peaking(1000.0, -3.0, 1.0)];
    let e = Engine::start(
        s.clone(),
        EngineOptions {
            cache: CoefficientCache::disabled(),
            realtime: true,
            persist: None,
        },
    );
    e.play_queue(vec![Track::from_arg(a.to_str().unwrap())], 0, 0.0);
    std::thread::sleep(Duration::from_millis(500));
    let detail = |st: &cadenza::engine::Status| {
        st.signal_path
            .as_ref()
            .and_then(|p| p.stages.iter().find(|x| x.id == "eq").cloned())
            .map(|x| x.params["bands"][0]["gain_db"].as_f64().unwrap())
    };
    assert_eq!(detail(&e.status()), Some(-3.0));
    let before = e.status().position;

    // Drag the band: many small changes, the track keeps playing and nothing restarts.
    for g in 1..=10 {
        s.dsp.eq[0].gain_db = -3.0 + g as f64 * 0.5;
        e.update_settings(s.clone());
        std::thread::sleep(Duration::from_millis(30));
    }
    std::thread::sleep(Duration::from_millis(300));
    let st = e.status();
    assert_eq!(st.state, PlayState::Playing, "{:?}", st.error);
    assert_eq!(detail(&st), Some(2.0));
    assert!(
        st.position > before + 0.4,
        "playback carried on: {} after {before}",
        st.position
    );
    e.shutdown();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn dsd_output_mode_modulates_to_dop_and_reports_the_chain() {
    use cadenza_dsp::dsd::{DsdModulator, DsdRate, OutputMode};
    let dir = temp_dir("dsd");
    let wav = dir.join("tone.wav");
    let frames = write_sine_wav(&wav, 44100, 16, 0.25, 1000.0);

    let mut s = settings(RateSetting::Auto, 24, "sinc-l-4k", "tpdf", false);
    s.dsp.output_mode = OutputMode::Dsd;
    s.dsp.dsd_rate = DsdRate::Dsd64;
    s.dsp.dsd_modulator = DsdModulator::Asdm5;
    // A DoP-only DAC: one that takes native DSD gets native whatever the packing says.
    s.output.device = "dsd=dop".into();
    let e = engine(s);
    let runs = e.status().runs_finished;
    e.play_queue(vec![Track::from_arg(wav.to_str().unwrap())], 0, 0.0);

    // Catch the signal path while the track plays.
    let mut path = None;
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
        let st = e.status();
        if let Some(p) = &st.signal_path {
            path = Some(p.clone());
        }
        if st.runs_finished > runs {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let st = e.wait_until_finished(runs, Duration::from_secs(60));
    assert_eq!(
        st.runs_finished,
        runs + 1,
        "playback did not finish in time"
    );
    assert_eq!(st.error, None);

    // The null output was opened for DSD64 over DoP: 24-bit frames at 2.8224 MHz / 16.
    let f = st.output_format.expect("output opened");
    assert!(f.is_dsd(), "{f:?}");
    assert_eq!(f.rate, 2_822_400 / 16);
    assert_eq!(f.bits, 24);
    assert_eq!(f.channels, 2);
    // 64 DSD bits per source sample, 16 per DoP frame: 4 DoP frames per source frame.
    assert!(
        st.output_frames >= (frames * 4) as u64,
        "{} DoP frames for {frames} source frames",
        st.output_frames
    );
    let p = path.expect("signal path published while playing");
    assert!(!p.bit_perfect);
    let ids: Vec<&str> = p
        .stages
        .iter()
        .filter(|s| s.active)
        .map(|s| s.id.as_str())
        .collect();
    assert!(ids.contains(&"modulator"), "{ids:?}");
    assert!(ids.contains(&"resampler"), "{ids:?}");
    assert!(
        !ids.contains(&"dither"),
        "the quantiser is bypassed in DSD mode: {ids:?}"
    );
    assert!(
        p.label.contains("ASDM5") && p.label.contains("DSD64"),
        "{}",
        p.label
    );
    assert_eq!(p.output.bits, 1);
    assert_eq!(p.output.rate, 2_822_400);
    e.shutdown();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn matrix_preset_changes_the_output_channel_count() {
    let dir = temp_dir("matrix");
    let wav = dir.join("tone.wav");
    let frames = write_sine_wav(&wav, 48000, 16, 0.25, 440.0);

    let mut s = settings(RateSetting::Source, 24, "sinc-l-4k", "tpdf", false);
    s.matrix.enabled = true;
    s.matrix.preset = "2.1".into();
    let e = engine(s);
    let runs = e.status().runs_finished;
    e.play_queue(vec![Track::from_arg(wav.to_str().unwrap())], 0, 0.0);
    let st = e.wait_until_finished(runs, Duration::from_secs(60));
    assert_eq!(st.runs_finished, runs + 1);
    assert_eq!(st.error, None);
    let f = st.output_format.unwrap();
    assert_eq!(f.channels, 3, "2.1 = L, R and a sub");
    assert_eq!(f.rate, 48_000);
    assert_eq!(st.output_frames, frames as u64);
    e.shutdown();
    let _ = std::fs::remove_dir_all(dir);
}

/// Two DSD files that are one recording cut in two play to PCM exactly like the whole:
/// the DSD → PCM converter runs on across the change instead of flushing and restarting.
#[test]
fn consecutive_dsd_tracks_play_gaplessly_to_pcm() {
    use cadenza::decode::dsd::write_dsf;
    use cadenza::decode::Tags;
    let dir = temp_dir("dsd-gapless");
    let bytes = 2_822_400 / 8; // 1 s per channel
    let planar: Vec<Vec<u8>> = (0..2)
        .map(|c| {
            (0..bytes * 2)
                .map(|i| if (i * 5 + c) % 3 == 0 { 0x96 } else { 0x69 })
                .collect()
        })
        .collect();
    let half = |r: std::ops::Range<usize>| -> Vec<Vec<u8>> {
        planar.iter().map(|p| p[r.clone()].to_vec()).collect()
    };
    let whole = dir.join("whole.dsf");
    let a = dir.join("a.dsf");
    let b = dir.join("b.dsf");
    write_dsf(&whole, 2_822_400, &planar, true, &Tags::default()).unwrap();
    write_dsf(&a, 2_822_400, &half(0..bytes), true, &Tags::default()).unwrap();
    write_dsf(
        &b,
        2_822_400,
        &half(bytes..bytes * 2),
        true,
        &Tags::default(),
    )
    .unwrap();

    let run = |files: &[&Path], capture: &Path| {
        let mut s = settings(RateSetting::Auto, 24, "sinc-m", "none", false);
        s.dsp.dsd_passthrough = false;
        s.output.device = format!("capture:{}", capture.display());
        let e = engine(s);
        let runs = e.status().runs_finished;
        e.play_queue(
            files
                .iter()
                .map(|f| Track::from_arg(f.to_str().unwrap()))
                .collect(),
            0,
            0.0,
        );
        let st = e.wait_until_finished(runs, Duration::from_secs(60));
        assert_eq!(st.runs_finished, runs + 1, "did not finish");
        assert_eq!(st.error, None);
        e.shutdown();
        std::fs::read(capture).unwrap()
    };
    let joined = run(&[&a, &b], &dir.join("joined.raw"));
    let reference = run(&[&whole], &dir.join("whole.raw"));
    assert!(reference.len() > 300_000 * 8);
    assert_eq!(joined.len(), reference.len());
    assert!(
        joined == reference,
        "the samples differ at the track change (converter restarted)"
    );
    let _ = std::fs::remove_dir_all(dir);
}
