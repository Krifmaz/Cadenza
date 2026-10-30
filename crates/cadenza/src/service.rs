//! Everything `cadenza serve` runs, as a library call: the engine (with its settings saved
//! back to the config file) and, as configured, the Encore link and the system input.
//! Used by `cadenza serve`.

use crate::config::{Config, Settings};
use crate::engine::{Engine, EngineOptions};
use cadenza_dsp::CoefficientCache;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The running services (the engine runs on while any clone of it lives).
pub struct Services {
    /// The engine.
    pub engine: Engine,
}

/// Start the engine and the configured inputs. `path` is the config file the settings
/// are saved to when they change.
pub fn start(config: &Config, path: PathBuf) -> Services {
    // One shared copy of the full config and one path, so engine settings and the system
    // input (changed live from the web UI) both save to the same file.
    let shared = Arc::new(Mutex::new(config.clone()));
    let path = Arc::new(path);
    let (sc, sp) = (shared.clone(), path.clone());
    let persist = Box::new(move |s: &Settings| {
        let mut c = sc.lock().unwrap_or_else(|e| e.into_inner());
        c.apply(s);
        if let Err(e) = c.save(&sp) {
            log::warn!("could not save {}: {e}", sp.display());
        }
    });
    let (sc, sp) = (shared.clone(), path.clone());
    crate::system::set_persist(move |sys: &crate::system::SystemConfig| {
        let mut c = sc.lock().unwrap_or_else(|e| e.into_inner());
        c.system = sys.clone();
        if let Err(e) = c.save(&sp) {
            log::warn!("could not save {}: {e}", sp.display());
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
    let (sc, sp) = (shared, path);
    crate::encore::set_persist(move |enc: &crate::config::EncoreConfig| {
        let mut c = sc.lock().unwrap_or_else(|e| e.into_inner());
        c.encore = enc.clone();
        if let Err(e) = c.save(&sp) {
            log::warn!("could not save {}: {e}", sp.display());
        }
    });
    crate::encore::start_global(engine.clone(), &config.encore);
    if config.system.enabled {
        crate::system::start_global(engine.clone(), &config.system);
    }
    Services { engine }
}
