//! Control API placeholder (filled in the next milestone).

use crate::engine::Engine;

/// Run the HTTP server until the process ends.
pub async fn serve(listen: &str, _engine: Engine) -> anyhow::Result<()> {
    anyhow::bail!("the control API is not implemented yet (listen={listen})")
}
