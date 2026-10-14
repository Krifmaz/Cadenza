//! The `cadenza` command.

mod cli;

use cadenza::config::{Config, RateSetting, Settings};
use cadenza::engine::{Engine, EngineOptions, Event, Track};
use cadenza_dsp::dsd::{DsdModulator, DsdOversampling, DsdPacking, DsdRate, OutputMode};
use cadenza_dsp::gpu::GpuMode;
use cadenza_dsp::{CoefficientCache, FilterKind};
use clap::Parser;
use cli::{Cli, Commands, Overrides};
use std::sync::Arc;
use std::time::Duration;

fn main() {
    let cli = Cli::parse();
    if let Commands::Render(args) = &cli.command {
        // a pure filter: stderr stays quiet (errors only) unless --progress or RUST_LOG
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("error")).init();
        std::process::exit(render(args));
    }
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(&cli.log))
        .format_timestamp_millis()
        .init();
    let path = cli.config.clone().unwrap_or_else(Config::default_path);
    let mut config = match Config::load(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cannot read {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let result = match cli.command {
        Commands::Play {
            items,
            seek,
            overrides,
        } => {
            apply_overrides(&mut config, &overrides);
            play(&config, items, seek)
        }
        Commands::Serve {
            listen,
            no_encore,
            overrides,
        } => {
            apply_overrides(&mut config, &overrides);
            if let Some(l) = listen {
                config.server.listen = l;
            }
            if no_encore {
                config.encore.url.clear();
            }
            serve(config, path)
        }
        Commands::Devices => {
            devices();
            Ok(())
        }
        Commands::Filters { in_rate, out_rate } => {
            let in_rate = in_rate.unwrap_or(44_100);
            let out_rate = out_rate.unwrap_or(705_600);
            println!(
                "{:<24} {:>9} {:>10} {:>9} {:>10}  FILTER   ({} Hz -> {} Hz)",
                "ID", "TAPS", "TRANS Hz", "STOP dB", "LATENCY s", in_rate, out_rate
            );
            for k in cadenza_dsp::FilterKind::catalog() {
                let i = k.describe(in_rate, out_rate, cadenza_dsp::fir::DEFAULT_ATTENUATION_DB);
                println!(
                    "{:<24} {:>9} {:>10.0} {:>9.0} {:>10.4}  {}",
                    i.id,
                    i.taps_at_rate,
                    i.transition_hz,
                    i.stopband_db,
                    i.latency_seconds,
                    i.label
                );
            }
            println!("\nAny tap count works too: sinc-l-300000, sinc-ma-2m. Phase: l=linear m=minimum, a=apodizing.");
            Ok(())
        }
        Commands::Dithers => {
            println!("{:<14} DITHER", "ID");
            for d in cadenza_dsp::DitherKind::catalog() {
                println!("{:<14} {}", d.id(), d.label());
            }
            Ok(())
        }
        Commands::Config => {
            println!(
                "# {}{}",
                path.display(),
                if path.exists() {
                    ""
                } else {
                    " (not written yet; these are the defaults)"
                }
            );
            print!("{}", toml::to_string_pretty(&config).unwrap_or_default());
            Ok(())
        }
        Commands::Bench {
            filter,
            rate,
            in_rate,
            seconds,
            channels,
            no_dsd,
            dsd_upsampling,
            dsd_full,
            json,
        } => bench(
            &config,
            filter,
            rate,
            in_rate,
            seconds,
            channels,
            !no_dsd,
            dsd_upsampling,
            dsd_full,
            json,
        ),
        Commands::Render(_) => unreachable!("handled above"),
        Commands::Endpoint {
            name,
            listen,
            backend,
            device,
            buffer_ms,
            no_mdns,
        } => {
            let mut ep = config.endpoint.clone();
            if let Some(n) = name {
                ep.name = n;
            }
            if let Some(l) = listen {
                ep.listen = l;
            }
            if let Some(b) = backend {
                ep.backend = b;
            }
            if let Some(d) = device {
                ep.device = d;
            }
            if let Some(ms) = buffer_ms {
                ep.buffer_ms = ms;
            }
            if no_mdns {
                ep.mdns = false;
            }
            cadenza::endpoint::run(ep)
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

/// `cadenza render`: exit code 0, or 2 with the message on stderr.
fn render(args: &cli::RenderArgs) -> i32 {
    use cadenza::render::{self, EngineSettings, InFormat, RenderOptions};
    use std::io::Write;
    if args.describe {
        println!("{}", render::describe());
        return 0;
    }
    let fail = |msg: String| {
        eprintln!("cadenza render: {msg}");
        2
    };
    let Some(in_rate) = args.in_rate else {
        return fail("--in-rate is required".into());
    };
    let Some(in_format) = InFormat::parse(&args.in_format) else {
        return fail(format!(
            "unknown input format {:?} (f32le or f64le)",
            args.in_format
        ));
    };
    if !render::OUT_FORMATS.contains(&args.out_format.trim().to_ascii_lowercase().as_str()) {
        return fail(format!(
            "unknown output format {:?} (s32le)",
            args.out_format
        ));
    }
    let mut settings = EngineSettings::default();
    // the old flags are aliases of --set; --set comes last and wins
    let mut sets: Vec<String> = Vec::new();
    for (key, v) in [
        ("filter", &args.filter),
        ("dither", &args.dither),
        ("headroom_db", &args.headroom_db),
        ("gpu", &args.gpu),
    ] {
        if let Some(v) = v {
            sets.push(format!("{key}={v}"));
        }
    }
    sets.extend(args.set.iter().cloned());
    if let Err(e) = settings.apply_sets(&sets, |w| eprintln!("cadenza render: {w}")) {
        return fail(e);
    }
    let opts = RenderOptions {
        in_rate,
        channels: args.channels,
        in_format,
        rate: args.rate.unwrap_or(in_rate),
        bits: args.bits,
        settings,
        cache: match &args.cache_dir {
            Some(d) => CoefficientCache::in_dir(d),
            None => CoefficientCache::default_location(),
        },
    };
    let stdin = std::io::stdin().lock();
    let stdout = std::io::stdout().lock();
    let input = std::io::BufReader::with_capacity(render::IO_BUFFER, stdin);
    let output = std::io::BufWriter::with_capacity(render::IO_BUFFER, stdout);
    let mut stderr = std::io::stderr();
    let progress: Option<&mut dyn Write> = if args.progress {
        Some(&mut stderr)
    } else {
        None
    };
    match render::run(&opts, input, output, progress) {
        Ok(_) => 0,
        Err(e) => fail(e),
    }
}

fn apply_overrides(config: &mut Config, o: &Overrides) {
    if o.bitperfect {
        config.dsp.bit_perfect = true;
    }
    if let Some(f) = &o.filter {
        config.dsp.filter = f.clone();
    }
    if let Some(d) = &o.dither {
        config.dsp.dither = d.clone();
    }
    if let Some(r) = &o.rate {
        match RateSetting::parse(r) {
            Some(rs) => config.output.rate = rs,
            None => eprintln!("ignoring unknown rate {r:?}"),
        }
    }
    if let Some(b) = o.bits {
        config.output.bits = b;
    }
    if let Some(b) = &o.backend {
        config.output.backend = b.clone();
    }
    if let Some(d) = &o.device {
        config.output.device = d.clone();
    }
    if let Some(v) = o.volume {
        config.dsp.volume_db = v;
    }
    if let Some(c) = &o.convolution {
        config.dsp.convolution = c.clone();
    }
    if let Some(m) = &o.output_mode {
        match OutputMode::parse(m) {
            Some(mode) => config.dsp.output_mode = mode,
            None => eprintln!("ignoring unknown output mode {m:?} (pcm | dsd)"),
        }
    }
    if let Some(r) = &o.dsd_rate {
        match DsdRate::parse(r) {
            Some(rate) => config.dsp.dsd_rate = rate,
            None => eprintln!("ignoring unknown DSD rate {r:?} (dsd64 … dsd2048)"),
        }
    }
    if let Some(m) = &o.dsd_modulator {
        match DsdModulator::parse(m) {
            Some(k) => config.dsp.dsd_modulator = k,
            None => {
                eprintln!("ignoring unknown DSD modulator {m:?} (asdm5, asdm7, dsd5v2, dsd7v2, asdm5ec, asdm7ec, asdm7ec8)")
            }
        }
    }
    if let Some(p) = &o.dsd_packing {
        match DsdPacking::parse(p) {
            Some(k) => config.dsp.dsd_packing = k,
            None => eprintln!("ignoring unknown DSD packing {p:?} (dop | native)"),
        }
    }
    if let Some(g) = o.dsd_gain {
        config.dsp.dsd_gain_db = g;
    }
    if let Some(m) = &o.dsd_oversampling {
        match DsdOversampling::parse(m) {
            Some(k) => config.dsp.dsd_oversampling = k,
            None => eprintln!("ignoring unknown DSD oversampling {m:?} (staged | full)"),
        }
    }
    if let Some(h) = &o.headroom {
        let h = h.trim();
        if h.eq_ignore_ascii_case("auto") || h.is_empty() {
            config.dsp.headroom_db = None;
        } else {
            match h.parse::<f64>() {
                Ok(db) => config.dsp.headroom_db = Some(db),
                Err(_) => eprintln!("ignoring headroom {h:?} (dB or auto)"),
            }
        }
    }
    if let Some(g) = &o.gpu {
        match g.parse::<GpuMode>() {
            Ok(mode) => config.dsp.gpu.mode = mode,
            Err(_) => eprintln!("ignoring unknown gpu mode {g:?} (auto | on | off | cuda | wgpu)"),
        }
    }
    if let Some(p) = &o.gpu_precision {
        match p.parse::<cadenza_dsp::gpu::GpuPrecision>() {
            Ok(p) => config.dsp.gpu.precision = p,
            Err(e) => eprintln!("ignoring {e}"),
        }
    }
    if let Some(m) = &o.matrix {
        let m = m.trim().to_ascii_lowercase();
        if m == "off" || m == "none" {
            config.matrix.enabled = false;
        } else if m == "custom" || cadenza_dsp::matrix::preset_ids().contains(&m.as_str()) {
            config.matrix.enabled = true;
            config.matrix.preset = m;
        } else {
            eprintln!(
                "ignoring unknown matrix preset {m:?} ({}, custom, off)",
                cadenza_dsp::matrix::preset_ids().join(", ")
            );
        }
    }
}

/// Parse `705600`, `705.6` (kHz) or `44.1k`.
fn parse_rate(s: &str) -> Option<u32> {
    let t = s.trim().to_ascii_lowercase();
    let t = t.trim_end_matches("hz").trim_end_matches('k').trim();
    let v: f64 = t.parse().ok()?;
    let hz = if v < 1000.0 { v * 1000.0 } else { v };
    (hz > 0.0).then_some(hz.round() as u32)
}

#[allow(clippy::too_many_arguments)]
fn bench(
    config: &Config,
    filter: Option<String>,
    rate: Option<String>,
    in_rate: Option<u32>,
    seconds: f64,
    channels: usize,
    dsd: bool,
    dsd_upsampling: bool,
    dsd_full: bool,
    json: bool,
) -> anyhow::Result<()> {
    let filter_id = filter.unwrap_or_else(|| config.dsp.filter.clone());
    let filter = FilterKind::parse(&filter_id)
        .ok_or_else(|| anyhow::anyhow!("unknown filter {filter_id:?} (see `cadenza filters`)"))?;
    let out_rate = match rate {
        Some(r) => parse_rate(&r).ok_or_else(|| anyhow::anyhow!("bad rate {r:?}"))?,
        None => 705_600,
    };
    let in_rate = in_rate.unwrap_or_else(|| {
        cadenza_dsp::resampler::RateFamily::of(out_rate)
            .map(cadenza_dsp::resampler::RateFamily::base)
            .unwrap_or(44_100)
    });
    anyhow::ensure!(
        out_rate > in_rate,
        "the output rate must be above the source rate"
    );
    let options = cadenza::bench::BenchOptions {
        filter,
        in_rate,
        out_rate,
        channels: channels.clamp(1, 32),
        seconds: seconds.clamp(0.1, 60.0),
        attenuation_db: config.dsp.attenuation_db,
        gpu: config.dsp.gpu,
        dsd,
        modulator: config.dsp.dsd_modulator,
        dsd_upsampling,
        dsd_full,
        cache: CoefficientCache::default_location(),
    };
    if !json {
        eprintln!(
            "Benchmarking {} {} → {} Hz, {} ch, {:.1} s of audio per row ({})…",
            filter.id(),
            in_rate,
            out_rate,
            options.channels,
            options.seconds,
            cadenza_dsp::gpu::Backend::detect(&config.dsp.gpu).label()
        );
    }
    let rows = cadenza::bench::run(&options);
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else {
        print!("{}", cadenza::bench::table(&rows));
    }
    Ok(())
}

fn devices() {
    println!(
        "{:<9} {:<40} {:<32} RATES / BITS · CHANNELS · DSD · MODE",
        "BACKEND", "ID", "NAME"
    );
    for d in cadenza::output::list_devices() {
        let rates: Vec<String> = d
            .rates
            .iter()
            .map(|r| cadenza_dsp::signal_path::rate_label(*r))
            .collect();
        println!(
            "{:<9} {:<40} {:<32} {} kHz · {} bit · {} · {}{}{}",
            d.backend,
            d.id.chars().take(40).collect::<String>(),
            d.name.chars().take(32).collect::<String>(),
            if rates.is_empty() {
                "?".to_string()
            } else {
                rates.join("/")
            },
            if d.bits.is_empty() {
                "?".to_string()
            } else {
                d.bits
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join("/")
            },
            match d.channels {
                Some(c) => format!("{c} ch"),
                None => "? ch".to_string(),
            },
            d.dsd.summary(),
            if d.exclusive {
                " · exclusive"
            } else {
                " · shared"
            },
            if d.default { " · default" } else { "" },
        );
    }
    println!(
        "\nCompiled backends: {} (endpoint://<name> streams to a `cadenza endpoint` on the network)",
        cadenza::output::backends().join(", ")
    );
}

fn play(config: &Config, items: Vec<String>, seek: f64) -> anyhow::Result<()> {
    let tracks: Vec<Track> = items.iter().map(|a| Track::from_arg(a)).collect();
    let engine = Engine::start(
        config.settings(),
        EngineOptions {
            cache: CoefficientCache::default_location(),
            realtime: true,
            persist: None,
        },
    );
    let mut events = engine.subscribe();
    let runs = engine.status().runs_finished;
    engine.play_queue(tracks, 0, seek);
    let mut last_label = String::new();
    loop {
        let st = engine.status();
        if st.runs_finished > runs {
            if let Some(e) = &st.error {
                eprintln!("{e}");
            }
            break;
        }
        while let Ok(ev) = events.try_recv() {
            match ev {
                Event::TrackStarted(t) => println!(
                    "▶ {}{}",
                    t.title,
                    if t.artist.is_empty() {
                        String::new()
                    } else {
                        format!(" — {}", t.artist)
                    }
                ),
                Event::Status(s) => {
                    if let Some(p) = &s.signal_path {
                        if p.label != last_label {
                            last_label = p.label.clone();
                            println!("  {} → {}", p.label, s.output);
                            for stage in p.stages.iter().filter(|s| s.active) {
                                println!("    · {}: {}", stage.name, stage.detail);
                            }
                        }
                    }
                }
                Event::TrackPlayed { .. } => {}
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    engine.shutdown();
    Ok(())
}

fn serve(config: Config, path: std::path::PathBuf) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    let persist_path = path.clone();
    let persist_cfg = std::sync::Mutex::new(config.clone());
    let persist = Box::new(move |s: &Settings| {
        let mut c = persist_cfg.lock().unwrap();
        c.apply(s);
        if let Err(e) = c.save(&persist_path) {
            log::warn!("could not save {}: {e}", persist_path.display());
        }
    });
    let engine = Engine::start(
        config.settings(),
        EngineOptions {
            cache: CoefficientCache::default_location(),
            realtime: true,
            persist: Some(persist),
        },
    );
    // The link runs for the life of the process; the handle is only needed to stop it.
    let _link = config
        .encore
        .enabled()
        .then(|| cadenza::encore::start(Arc::new(config.encore.clone()), engine.clone()));
    // The UPnP renderer, when enabled, for the life of the process.
    let _upnp = if config.upnp.enabled {
        match cadenza::upnp::start(engine.clone(), &config.upnp, true) {
            Ok(r) => Some(r),
            Err(e) => {
                log::error!("UPnP renderer: {e}");
                None
            }
        }
    } else {
        None
    };
    runtime.block_on(cadenza::api::serve(&config.server.listen, engine))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_cover_every_v2_setting() {
        let mut c = Config::default();
        let o = Overrides {
            output_mode: Some("dsd".into()),
            dsd_rate: Some("dsd512".into()),
            dsd_modulator: Some("dsd5v2".into()),
            dsd_packing: Some("native".into()),
            dsd_gain: Some(-4.0),
            dsd_oversampling: Some("full".into()),
            headroom: Some("-2.5".into()),
            gpu: Some("off".into()),
            matrix: Some("2.1".into()),
            dither: Some("ns9".into()),
            ..Default::default()
        };
        apply_overrides(&mut c, &o);
        assert_eq!(c.dsp.output_mode, OutputMode::Dsd);
        assert_eq!(c.dsp.dsd_rate, DsdRate::Dsd512);
        assert_eq!(c.dsp.dsd_modulator, DsdModulator::Dsd5v2);
        assert_eq!(c.dsp.dsd_packing, DsdPacking::Native);
        assert_eq!(c.dsp.dsd_gain_db, -4.0);
        assert_eq!(c.dsp.dsd_oversampling, DsdOversampling::Full);
        assert_eq!(c.dsp.headroom_db, Some(-2.5));
        assert_eq!(c.dsp.gpu.mode, GpuMode::Off);
        assert!(c.matrix.enabled);
        assert_eq!(c.matrix.preset, "2.1");
        assert_eq!(c.dsp.dither, "ns9");
        let o = Overrides {
            headroom: Some("auto".into()),
            matrix: Some("off".into()),
            dsd_rate: Some("dsd9000".into()),
            ..Default::default()
        };
        apply_overrides(&mut c, &o);
        assert_eq!(c.dsp.headroom_db, None);
        assert!(!c.matrix.enabled);
        assert_eq!(
            c.dsp.dsd_rate,
            DsdRate::Dsd512,
            "unknown values are ignored"
        );
        assert_eq!(parse_rate("705.6"), Some(705_600));
        assert_eq!(parse_rate("1536000"), Some(1_536_000));
        assert_eq!(parse_rate("44.1k"), Some(44_100));
        assert_eq!(parse_rate("x"), None);
    }
}
