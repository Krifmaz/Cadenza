//! Everything `cadenza serve` runs, as a library call: the engine (with its settings saved
//! back to the config file) and, as configured, the Encore link, the UPnP renderer, the
//! AirPlay receiver and the system input. Used by `cadenza serve` and the desktop app.

use crate::config::{Config, Settings};
use crate::engine::{Engine, EngineOptions};
use cadenza_dsp::CoefficientCache;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The running services; dropping it stops the receivers (the engine runs on while any
/// clone of it lives).
pub struct Services {
    /// The engine.
    pub engine: Engine,
    _link: Option<crate::encore::EncoreLink>,
    _upnp: Option<crate::upnp::UpnpRenderer>,
    _airplay: Option<crate::airplay::AirplayReceiver>,
}

/// Start the engine and the configured receivers. `path` is the config file the settings
/// are saved to when they change.
pub fn start(config: &Config, path: PathBuf) -> Services {
    let persist_cfg = Mutex::new(config.clone());
    let persist = Box::new(move |s: &Settings| {
        let mut c = persist_cfg.lock().unwrap_or_else(|e| e.into_inner());
        c.apply(s);
        if let Err(e) = c.save(&path) {
            log::warn!("could not save {}: {e}", path.display());
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
    let link = config
        .encore
        .enabled()
        .then(|| crate::encore::start(Arc::new(config.encore.clone()), engine.clone()));
    let upnp = config
        .upnp
        .enabled
        .then(|| crate::upnp::start(engine.clone(), &config.upnp, true))
        .and_then(|r| r.map_err(|e| log::error!("UPnP renderer: {e}")).ok());
    let airplay = config
        .airplay
        .enabled
        .then(|| crate::airplay::start(engine.clone(), &config.airplay, true))
        .and_then(|r| r.map_err(|e| log::error!("AirPlay receiver: {e}")).ok());
    if config.system.enabled {
        crate::system::start_global(engine.clone(), &config.system);
    }
    Services {
        engine,
        _link: link,
        _upnp: upnp,
        _airplay: airplay,
    }
}
