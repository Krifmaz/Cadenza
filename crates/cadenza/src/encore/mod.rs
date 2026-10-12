//! Encore client placeholder (filled in a later milestone).

use crate::config::EncoreConfig;
use crate::engine::Engine;
use std::sync::Arc;

/// Start the Encore device threads.
pub fn start(config: Arc<EncoreConfig>, _engine: Engine) {
    log::warn!("Encore integration not implemented yet ({})", config.url);
}
