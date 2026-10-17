//! Cadenza as a desktop app: the engine runs in this process (the same services as
//! `cadenza serve`: Encore link, UPnP, AirPlay, system input, as configured) behind a native
//! window built with iced, plus a tray icon and the OS media controls. Closing the window
//! keeps the music playing; Quit in the tray ends it. The web control page is still served
//! (on the configured `[server] listen`, when free) so a phone can drive the engine too.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod covers;
mod icons;
mod pages;
mod platform;
mod shot;
mod store;
mod theme;
mod ui;

use cadenza::config::Config;

fn main() -> iced::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    if !platform::init() {
        log::info!("Cadenza is already running: brought its window forward");
        return Ok(());
    }
    let path = std::env::var_os("CADENZA_CONFIG")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(Config::default_path);
    let config = Config::load(&path).unwrap_or_else(|e| {
        log::error!("cannot read {}: {e}; using the defaults", path.display());
        Config::default()
    });
    let services = cadenza::service::start(&config, path);
    let engine = services.engine.clone();
    serve_web(&config, engine.clone());
    let start_page = std::env::var("CADENZA_PAGE")
        .ok()
        .and_then(|id| app::Page::ALL.into_iter().find(|p| p.id() == id))
        .unwrap_or(app::Page::Now);

    let r = iced::daemon(
        move || app::App::new(engine.clone(), start_page),
        app::App::update,
        app::App::view,
    )
    .title(app::App::title)
    .theme(app::App::theme)
    .subscription(app::App::subscription)
    .antialiasing(true)
    .run();
    drop(services);
    r
}

/// The web control page and API for phones and Encore, on its own thread: the configured
/// address when it is free, otherwise nothing (a `cadenza serve` already answers there).
fn serve_web(config: &Config, engine: cadenza::engine::Engine) {
    let listen = config.server.listen.clone();
    std::thread::Builder::new()
        .name("cadenza-web".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => return log::error!("web page: {e}"),
            };
            rt.block_on(async move {
                match tokio::net::TcpListener::bind(&listen).await {
                    Ok(l) => {
                        log::info!("web control page on http://{listen}/");
                        if let Err(e) = cadenza::api::serve_on(l, engine).await {
                            log::error!("web page: {e}");
                        }
                    }
                    Err(e) => log::info!("web control page not started ({listen}: {e})"),
                }
            });
        })
        .expect("start the web page thread");
}
