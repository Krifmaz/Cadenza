//! Cadenza as a desktop app: the engine runs in this process (the same services as
//! `cadenza serve`: Encore link, UPnP, AirPlay, system input, as configured), its control
//! page opens in a native window, and a tray icon and the OS media controls (media keys,
//! SMTC on Windows, Now Playing on macOS, MPRIS on Linux) drive it. Closing the window keeps
//! playing; Quit in the tray ends it. The page is the same one `cadenza serve` serves, from
//! a port on 127.0.0.1 (the configured `[server] listen` when it is free, any free port
//! otherwise), so a phone pointed at the configured address still works.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use cadenza::config::Config;
use cadenza::engine::{Command, Engine, PlayState};
use souvlaki::{MediaControlEvent, MediaControls, MediaMetadata, MediaPlayback, PlatformConfig};
use std::cell::RefCell;
use std::time::Duration;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder};

/// The tray's changing items and the OS media controls: UI objects, kept on the main thread.
struct Ui {
    now_playing: MenuItem<tauri::Wry>,
    toggle: MenuItem<tauri::Wry>,
    media: Option<MediaControls>,
    last: Option<(String, String, bool)>,
}

thread_local! {
    static UI: RefCell<Option<Ui>> = const { RefCell::new(None) };
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    let app = tauri::Builder::default()
        // A second launch only brings this window forward; it exits before `setup`, so it
        // never starts a second engine or grabs the receivers' ports.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_window(app)
        }))
        .plugin(
            tauri_plugin_window_state::Builder::new()
                .with_state_flags(
                    tauri_plugin_window_state::StateFlags::SIZE
                        | tauri_plugin_window_state::StateFlags::POSITION
                        | tauri_plugin_window_state::StateFlags::MAXIMIZED,
                )
                .build(),
        )
        .setup(|app| {
            let path = Config::default_path();
            let config = Config::load(&path).unwrap_or_else(|e| {
                log::error!("cannot read {}: {e}; using the defaults", path.display());
                Config::default()
            });
            let services = cadenza::service::start(&config, path);
            let engine = services.engine.clone();
            // The control page: the configured address when it is free, any free local
            // port if not (a `cadenza serve` may already be running there).
            let listener = std::net::TcpListener::bind(&config.server.listen)
                .or_else(|_| std::net::TcpListener::bind("127.0.0.1:0"))?;
            listener.set_nonblocking(true)?;
            let port = listener.local_addr()?.port();
            let url: tauri::Url = format!("http://127.0.0.1:{port}/").parse()?;
            let server_engine = engine.clone();
            tauri::async_runtime::spawn(async move {
                match tokio::net::TcpListener::from_std(listener) {
                    Ok(l) => {
                        if let Err(e) = cadenza::api::serve_on(l, server_engine).await {
                            log::error!("control page: {e}");
                        }
                    }
                    Err(e) => log::error!("control page: {e}"),
                }
            });
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("Cadenza")
                .inner_size(1280.0, 820.0)
                .min_inner_size(420.0, 560.0)
                .build()?;
            setup_tray(app.handle(), &engine)?;
            watch(app.handle().clone(), engine);
            app.manage(services);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("start the desktop app");

    app.run(move |app, event| match event {
        // Closing the window hides it; the music goes on (Quit is in the tray).
        tauri::RunEvent::WindowEvent {
            event: tauri::WindowEvent::CloseRequested { api, .. },
            label,
            ..
        } if label == "main" => {
            api.prevent_close();
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.hide();
            }
        }
        #[cfg(target_os = "macos")]
        tauri::RunEvent::Reopen { .. } => show_window(app),
        tauri::RunEvent::Exit => {
            if let Some(services) = app.try_state::<cadenza::service::Services>() {
                services.engine.send(Command::Stop);
            }
            drop(UI.with(|u| u.borrow_mut().take()));
        }
        _ => {}
    });
}

fn show_window(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

fn setup_tray(app: &AppHandle, engine: &Engine) -> tauri::Result<()> {
    let now_playing = MenuItem::with_id(app, "np", "Cadenza", false, None::<&str>)?;
    let toggle = MenuItem::with_id(app, "toggle", "Play", true, None::<&str>)?;
    let menu = Menu::with_items(
        app,
        &[
            &now_playing,
            &PredefinedMenuItem::separator(app)?,
            &toggle,
            &MenuItem::with_id(app, "next", "Next", true, None::<&str>)?,
            &MenuItem::with_id(app, "prev", "Previous", true, None::<&str>)?,
            &PredefinedMenuItem::separator(app)?,
            &MenuItem::with_id(app, "show", "Show Cadenza", true, None::<&str>)?,
            &PredefinedMenuItem::separator(app)?,
            &MenuItem::with_id(app, "quit", "Quit Cadenza", true, None::<&str>)?,
        ],
    )?;
    let e = engine.clone();
    let mut tray = TrayIconBuilder::with_id("main")
        .tooltip("Cadenza")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(move |app, ev| match ev.id().as_ref() {
            "toggle" => e.send(Command::Toggle),
            "next" => e.send(Command::Next),
            "prev" => e.send(Command::Prev),
            "show" => show_window(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, ev| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = ev
            {
                show_window(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    // Kept alive by the app (looked up by id); the handle itself is not needed.
    let _tray = tray.build(app)?;
    let media = media_controls(app, engine);
    UI.with(|u| {
        *u.borrow_mut() = Some(Ui {
            now_playing,
            toggle,
            media,
            last: None,
        })
    });
    Ok(())
}

/// The OS media controls: media keys, the lock screen / media overlay, MPRIS.
fn media_controls(app: &AppHandle, engine: &Engine) -> Option<MediaControls> {
    if !session_bus_reachable() {
        log::info!("no session bus: media keys and MPRIS are off");
        return None;
    }
    #[allow(unused_mut, unused_assignments)]
    let mut hwnd: Option<*mut std::ffi::c_void> = None;
    #[cfg(target_os = "windows")]
    {
        hwnd = app
            .get_webview_window("main")
            .and_then(|w| w.hwnd().ok())
            .map(|h| h.0 as *mut std::ffi::c_void);
    }
    let mut mc = match MediaControls::new(PlatformConfig {
        display_name: "Cadenza",
        dbus_name: "cadenza",
        hwnd,
    }) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("media controls unavailable: {e:?}");
            return None;
        }
    };
    let e = engine.clone();
    let handle = app.clone();
    if let Err(err) = mc.attach(move |ev| match ev {
        MediaControlEvent::Play => e.send(Command::Play),
        MediaControlEvent::Pause => e.send(Command::Pause),
        MediaControlEvent::Toggle => e.send(Command::Toggle),
        MediaControlEvent::Next => e.send(Command::Next),
        MediaControlEvent::Previous => e.send(Command::Prev),
        MediaControlEvent::Stop => e.send(Command::Stop),
        MediaControlEvent::Raise => {
            let h = handle.clone();
            let _ = handle.run_on_main_thread(move || show_window(&h));
        }
        MediaControlEvent::Quit => handle.exit(0),
        _ => {}
    }) {
        log::warn!("media controls: {err:?}");
        return None;
    }
    Some(mc)
}

/// MPRIS needs the session D-Bus; souvlaki panics in its own thread without one (headless
/// machines, some minimal desktops), so look first. Other platforms always have their APIs.
fn session_bus_reachable() -> bool {
    #[cfg(target_os = "linux")]
    {
        let Ok(addr) = std::env::var("DBUS_SESSION_BUS_ADDRESS") else {
            // The default location when the variable is missing.
            return std::env::var("XDG_RUNTIME_DIR").is_ok_and(|d| {
                std::os::unix::net::UnixStream::connect(std::path::Path::new(&d).join("bus"))
                    .is_ok()
            });
        };
        addr.split(';').any(|a| match a.strip_prefix("unix:path=") {
            Some(rest) => {
                let path = rest.split(',').next().unwrap_or(rest);
                std::os::unix::net::UnixStream::connect(path).is_ok()
            }
            // Abstract sockets and other transports: let zbus try.
            None => a.starts_with("unix:"),
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

/// Follow the engine: what is playing goes to the tray and the OS media controls.
fn watch(app: AppHandle, engine: Engine) {
    std::thread::Builder::new()
        .name("cadenza-desktop-watch".into())
        .spawn(move || loop {
            let st = engine.status();
            let playing = st.state == PlayState::Playing;
            let (title, artist) = st
                .track
                .as_ref()
                .map(|t| (t.title.clone(), t.artist.clone()))
                .unwrap_or_default();
            let album = st
                .track
                .as_ref()
                .map(|t| t.album.clone())
                .unwrap_or_default();
            let state = st.state;
            let h = app.clone();
            let _ = app.run_on_main_thread(move || {
                UI.with(|u| {
                    let mut u = u.borrow_mut();
                    let Some(ui) = u.as_mut() else { return };
                    let now = (title.clone(), artist.clone(), playing);
                    if ui.last.as_ref() == Some(&now) {
                        return;
                    }
                    ui.last = Some(now);
                    let label = match (title.is_empty(), artist.is_empty()) {
                        (true, _) => "Cadenza".to_string(),
                        (false, true) => title.clone(),
                        (false, false) => format!("{title} – {artist}"),
                    };
                    let _ = ui.now_playing.set_text(&label);
                    let _ = ui.toggle.set_text(if playing { "Pause" } else { "Play" });
                    if let Some(tray) = h.tray_by_id("main") {
                        let _ = tray.set_tooltip(Some(format!("Cadenza · {label}")));
                    }
                    if let Some(mc) = ui.media.as_mut() {
                        let _ = mc.set_metadata(MediaMetadata {
                            title: (!title.is_empty()).then_some(title.as_str()),
                            artist: (!artist.is_empty()).then_some(artist.as_str()),
                            album: (!album.is_empty()).then_some(album.as_str()),
                            ..Default::default()
                        });
                        let _ = mc.set_playback(match state {
                            PlayState::Playing => MediaPlayback::Playing { progress: None },
                            PlayState::Paused => MediaPlayback::Paused { progress: None },
                            PlayState::Stopped => MediaPlayback::Stopped,
                        });
                    }
                });
            });
            std::thread::sleep(Duration::from_millis(500));
        })
        .expect("start the watcher");
}
