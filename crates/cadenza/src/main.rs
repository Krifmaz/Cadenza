//! The `cadenza` command.

mod cli;

use cadenza::config::{Config, RateSetting, Settings};
use cadenza::engine::{Engine, EngineOptions, Event, Track};
use cadenza_dsp::CoefficientCache;
use clap::Parser;
use cli::{Cli, Commands, Overrides};
use std::sync::Arc;
use std::time::Duration;

fn main() {
    let cli = Cli::parse();
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
                    i.id, i.taps_at_rate, i.transition_hz, i.stopband_db, i.latency_seconds, i.label
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
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
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
}

fn devices() {
    println!("{:<8} {:<40} {:<36} RATES / BITS", "BACKEND", "ID", "NAME");
    for d in cadenza::output::list_devices() {
        let rates: Vec<String> = d
            .rates
            .iter()
            .map(|r| cadenza_dsp::signal_path::rate_label(*r))
            .collect();
        println!(
            "{:<8} {:<40} {:<36} {} kHz · {} bit{}{}",
            d.backend,
            d.id,
            d.name.chars().take(36).collect::<String>(),
            if rates.is_empty() {
                "?".to_string()
            } else {
                rates.join("/")
            },
            d.bits
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join("/"),
            if d.exclusive { " · exclusive" } else { "" },
            if d.default { " · default" } else { "" },
        );
    }
    println!(
        "\nCompiled backends: {}",
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
    if config.encore.enabled() {
        cadenza::encore::start(Arc::new(config.encore.clone()), engine.clone());
    }
    runtime.block_on(cadenza::api::serve(&config.server.listen, engine))
}
