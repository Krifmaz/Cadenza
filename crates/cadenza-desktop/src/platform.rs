//! The window and the parts of the desktop around it: the tray icon and its menu, the OS
//! media controls (media keys, SMTC on Windows, Now Playing on macOS, MPRIS on Linux), one
//! instance at a time (a second launch brings this window forward), and closing to the tray.
//!
//! Everything outside iced reports through one channel ([`send`]) that [`events`] turns
//! into messages.

use crate::app::{App, Message};
use cadenza::engine::PlayState;
use iced::{window, Size, Subscription, Task};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Mutex, OnceLock};

static TX: OnceLock<Mutex<Sender<Message>>> = OnceLock::new();
static RX: Mutex<Option<Receiver<Message>>> = Mutex::new(None);
static TRAY: AtomicBool = AtomicBool::new(false);

/// Hand a message to the window from any thread.
pub fn send(m: Message) {
    if let Some(tx) = TX.get() {
        let _ = tx.lock().unwrap_or_else(|e| e.into_inner()).send(m);
    }
}

/// Set up the channel; returns `false` when another Cadenza window already runs (it has
/// been asked to come forward, and this process should exit).
pub fn init() -> bool {
    let (tx, rx) = channel();
    let _ = TX.set(Mutex::new(tx));
    *RX.lock().unwrap() = Some(rx);
    single_instance()
}

// ----- one instance --------------------------------------------------------------------

/// A local port only Cadenza's window listens on: the first instance holds it; a second
/// launch connects and says `hello <version>`. The running one answers `quit` when the
/// newcomer is newer (just updated: it hands over and exits) and `ok` otherwise (it comes
/// forward and the newcomer exits). Before 3.0.5 the listener only knew `raise`.
const INSTANCE_PORT: u16 = 47_731;

fn single_instance() -> bool {
    use std::io::{Read, Write};
    use std::time::{Duration, Instant};
    let listen = || std::net::TcpListener::bind(("127.0.0.1", INSTANCE_PORT));
    let serve = |l: std::net::TcpListener| {
        std::thread::Builder::new()
            .name("cadenza-instance".into())
            .spawn(move || {
                for s in l.incoming().flatten() {
                    let mut s = s;
                    let _ = s.set_read_timeout(Some(Duration::from_secs(1)));
                    let mut msg = Vec::new();
                    let mut b = [0u8; 1];
                    while msg.len() < 64 && matches!(s.read(&mut b), Ok(1)) && b[0] != b'\n' {
                        msg.push(b[0]);
                    }
                    let msg = String::from_utf8_lossy(&msg).to_string();
                    let newer = msg.strip_prefix("hello ").is_some_and(|v| {
                        crate::update::parse_version(v)
                            > crate::update::parse_version(crate::update::current())
                    });
                    if newer {
                        log::info!("a newer Cadenza ({msg}) is starting: handing over");
                        let _ = s.write_all(b"quit\n");
                        send(Message::Quit);
                        return;
                    }
                    let _ = s.write_all(b"ok\n");
                    send(Message::Raise);
                }
            })
            .ok();
    };
    if let Ok(l) = listen() {
        serve(l);
        return true;
    }
    let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", INSTANCE_PORT)) else {
        // Something else holds the port: run anyway.
        return true;
    };
    let _ = s.write_all(format!("hello {}\n", crate::update::current()).as_bytes());
    let _ = s.set_read_timeout(Some(Duration::from_secs(3)));
    let mut reply = String::new();
    let _ = s.read_to_string(&mut reply);
    match reply.trim() {
        "quit" => {
            // The old one is closing: wait for the port, then carry on as the only one.
            let until = Instant::now() + Duration::from_secs(15);
            while Instant::now() < until {
                if let Ok(l) = listen() {
                    serve(l);
                    return true;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            true
        }
        "ok" => false,
        _ => {
            // A version before 3.0.5: it only knows "raise".
            if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", INSTANCE_PORT)) {
                let _ = s.write_all(b"raise");
            }
            false
        }
    }
}

// ----- the window ----------------------------------------------------------------------

/// Open the main window at its remembered size.
pub fn open_window() -> (window::Id, Task<window::Id>) {
    let saved = crate::store::load_window();
    let size = if saved.width >= 420.0 && saved.height >= 480.0 {
        Size::new(saved.width, saved.height)
    } else {
        Size::new(1280.0, 820.0)
    };
    window::open(window::Settings {
        size,
        min_size: Some(Size::new(420.0, 520.0)),
        icon: icon(),
        exit_on_close_request: false,
        ..Default::default()
    })
}

fn icon() -> Option<window::Icon> {
    window::icon::from_file_data(include_bytes!("../icons/128x128.png"), None).ok()
}

/// Events from outside the window (tray, media keys, a second launch).
pub fn events() -> Subscription<Message> {
    Subscription::run(|| {
        iced::stream::channel(
            64,
            |mut out: iced::futures::channel::mpsc::Sender<Message>| async move {
                let Some(rx) = RX.lock().unwrap().take() else {
                    return;
                };
                let (atx, mut arx) = iced::futures::channel::mpsc::unbounded();
                std::thread::spawn(move || {
                    while let Ok(m) = rx.recv() {
                        if atx.unbounded_send(m).is_err() {
                            break;
                        }
                    }
                });
                use iced::futures::{SinkExt, StreamExt};
                while let Some(m) = arx.next().await {
                    if out.send(m).await.is_err() {
                        break;
                    }
                }
            },
        )
    })
}

/// Whether closing the window can leave the app running in the tray.
pub fn tray_available() -> bool {
    TRAY.load(Ordering::Relaxed)
}

// ----- tray and media controls ---------------------------------------------------------

/// What the tray and the media controls show.
#[derive(Clone, Default, PartialEq)]
struct NowPlaying {
    title: String,
    artist: String,
    album: String,
    state: u8,
}

static NOW: Mutex<Option<NowPlaying>> = Mutex::new(None);

fn now_of(app: &App) -> NowPlaying {
    let t = app.status.track.as_ref();
    NowPlaying {
        title: t.map(|t| t.title.clone()).unwrap_or_default(),
        artist: t.map(|t| t.artist.clone()).unwrap_or_default(),
        album: t.map(|t| t.album.clone()).unwrap_or_default(),
        state: match app.status.state {
            PlayState::Playing => 2,
            PlayState::Paused => 1,
            PlayState::Stopped => 0,
        },
    }
}

fn label(n: &NowPlaying) -> String {
    match (n.title.is_empty(), n.artist.is_empty()) {
        (true, _) => "Cadenza".into(),
        (false, true) => n.title.clone(),
        (false, false) => format!("{} – {}", n.title, n.artist),
    }
}

/// Start the tray and the media controls (once, after the window opened).
pub fn start(app: &App) -> Task<Message> {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::Relaxed) {
        return Task::none();
    }
    tray::start();
    let Some(id) = app.window else {
        return Task::none();
    };
    // The media controls need the window's native handle on Windows.
    window::run(id, |w| {
        #[cfg(target_os = "windows")]
        {
            use raw_window_handle::RawWindowHandle;
            if let Ok(RawWindowHandle::Win32(h)) = w.window_handle().map(|h| h.as_raw()) {
                return Some(h.hwnd.get());
            }
        }
        let _ = w;
        None::<isize>
    })
    .map(|hwnd| {
        media::start(hwnd);
        Message::Nothing
    })
}

/// Keep the tray and the OS media controls in step with the engine.
pub fn sync(app: &App) -> Task<Message> {
    let n = now_of(app);
    let mut g = NOW.lock().unwrap();
    if g.as_ref() == Some(&n) {
        return Task::none();
    }
    *g = Some(n.clone());
    drop(g);
    tray::update(&n);
    media::update(&n);
    Task::none()
}

mod tray {
    //! The tray icon. On Linux it lives on its own GTK thread (libappindicator); on Windows
    //! and macOS on the main thread, next to the window.

    use super::{label, send, NowPlaying, TRAY};
    use crate::app::{Message, Transport};
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;
    use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
    use tray_icon::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};

    /// The latest "now playing" for the tray thread to pick up.
    static PENDING: Mutex<Option<NowPlaying>> = Mutex::new(None);

    struct Items {
        _tray: TrayIcon,
        now: MenuItem,
        toggle: MenuItem,
    }

    thread_local! {
        static ITEMS: std::cell::RefCell<Option<Items>> = const { std::cell::RefCell::new(None) };
    }

    fn icon() -> Option<tray_icon::Icon> {
        let img = image_rgba(include_bytes!("../icons/32x32.png"))?;
        tray_icon::Icon::from_rgba(img.0, img.1, img.2).ok()
    }

    /// Decode the PNG with iced's own image support.
    fn image_rgba(png: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
        let icon = iced::window::icon::from_file_data(png, None).ok()?;
        let (rgba, size) = icon.into_raw();
        Some((rgba, size.width, size.height))
    }

    fn build() -> Option<Items> {
        let now = MenuItem::with_id("np", "Cadenza", false, None);
        let toggle = MenuItem::with_id("toggle", "Play", true, None);
        let menu = Menu::new();
        let _ = menu.append_items(&[
            &now,
            &PredefinedMenuItem::separator(),
            &toggle,
            &MenuItem::with_id("next", "Next", true, None),
            &MenuItem::with_id("prev", "Previous", true, None),
            &PredefinedMenuItem::separator(),
            &MenuItem::with_id("show", "Show Cadenza", true, None),
            &PredefinedMenuItem::separator(),
            &MenuItem::with_id("quit", "Quit Cadenza", true, None),
        ]);
        let mut b = TrayIconBuilder::new()
            .with_tooltip("Cadenza")
            .with_menu(Box::new(menu))
            .with_menu_on_left_click(false);
        if let Some(i) = icon() {
            b = b.with_icon(i);
        }
        match b.build() {
            Ok(t) => Some(Items {
                _tray: t,
                now,
                toggle,
            }),
            Err(e) => {
                log::info!("no tray icon: {e}");
                None
            }
        }
    }

    fn handlers() {
        MenuEvent::set_event_handler(Some(|e: MenuEvent| {
            send(match e.id().as_ref() {
                "toggle" => Message::Transport(Transport::Toggle),
                "next" => Message::Transport(Transport::Next),
                "prev" => Message::Transport(Transport::Prev),
                "show" => Message::Raise,
                "quit" => Message::Quit,
                _ => return,
            })
        }));
        TrayIconEvent::set_event_handler(Some(|e: TrayIconEvent| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = e
            {
                send(Message::Raise);
            }
        }));
    }

    fn apply(n: &NowPlaying) {
        ITEMS.with(|i| {
            if let Some(items) = i.borrow().as_ref() {
                items.now.set_text(label(n));
                items
                    .toggle
                    .set_text(if n.state == 2 { "Pause" } else { "Play" });
                let _ = items
                    ._tray
                    .set_tooltip(Some(format!("Cadenza · {}", label(n))));
            }
        });
    }

    #[cfg(target_os = "linux")]
    pub fn start() {
        std::thread::Builder::new()
            .name("cadenza-tray".into())
            .spawn(|| {
                if gtk::init().is_err() {
                    log::info!("no GTK: no tray icon");
                    return;
                }
                let Some(items) = build() else { return };
                ITEMS.with(|i| *i.borrow_mut() = Some(items));
                handlers();
                TRAY.store(true, Ordering::Relaxed);
                gtk::glib::timeout_add_local(std::time::Duration::from_millis(400), || {
                    if let Some(n) = PENDING.lock().unwrap().take() {
                        apply(&n);
                    }
                    gtk::glib::ControlFlow::Continue
                });
                gtk::main();
            })
            .ok();
    }

    #[cfg(not(target_os = "linux"))]
    pub fn start() {
        let Some(items) = build() else { return };
        ITEMS.with(|i| *i.borrow_mut() = Some(items));
        handlers();
        TRAY.store(true, Ordering::Relaxed);
    }

    pub fn update(n: &NowPlaying) {
        if cfg!(target_os = "linux") {
            *PENDING.lock().unwrap() = Some(n.clone());
        } else {
            apply(n);
        }
    }
}

mod media {
    //! The OS media controls (souvlaki). Kept on the main thread.

    use super::{send, NowPlaying};
    use crate::app::{Message, Transport};
    use souvlaki::{
        MediaControlEvent, MediaControls, MediaMetadata, MediaPlayback, PlatformConfig,
    };

    thread_local! {
        static MC: std::cell::RefCell<Option<MediaControls>> = const { std::cell::RefCell::new(None) };
    }

    pub fn start(hwnd: Option<isize>) {
        if !session_bus_reachable() {
            log::info!("no session bus: media keys and MPRIS are off");
            return;
        }
        if cfg!(target_os = "windows") && hwnd.is_none() {
            return;
        }
        let mut mc = match MediaControls::new(PlatformConfig {
            display_name: "Cadenza",
            dbus_name: "cadenza",
            hwnd: hwnd.map(|h| h as *mut std::ffi::c_void),
        }) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("media controls unavailable: {e:?}");
                return;
            }
        };
        let r = mc.attach(|ev| {
            send(match ev {
                MediaControlEvent::Play => Message::Transport(Transport::Play),
                MediaControlEvent::Pause => Message::Transport(Transport::Pause),
                MediaControlEvent::Toggle => Message::Transport(Transport::Toggle),
                MediaControlEvent::Next => Message::Transport(Transport::Next),
                MediaControlEvent::Previous => Message::Transport(Transport::Prev),
                MediaControlEvent::Stop => Message::Transport(Transport::Stop),
                MediaControlEvent::Raise => Message::Raise,
                MediaControlEvent::Quit => Message::Quit,
                _ => return,
            })
        });
        if let Err(e) = r {
            log::warn!("media controls: {e:?}");
            return;
        }
        MC.with(|m| *m.borrow_mut() = Some(mc));
    }

    pub fn update(n: &NowPlaying) {
        MC.with(|m| {
            if let Some(mc) = m.borrow_mut().as_mut() {
                let _ = mc.set_metadata(MediaMetadata {
                    title: (!n.title.is_empty()).then_some(n.title.as_str()),
                    artist: (!n.artist.is_empty()).then_some(n.artist.as_str()),
                    album: (!n.album.is_empty()).then_some(n.album.as_str()),
                    ..Default::default()
                });
                let _ = mc.set_playback(match n.state {
                    2 => MediaPlayback::Playing { progress: None },
                    1 => MediaPlayback::Paused { progress: None },
                    _ => MediaPlayback::Stopped,
                });
            }
        });
    }

    /// MPRIS needs the session D-Bus; souvlaki panics in its own thread without one, so
    /// look first. Other platforms always have their APIs.
    fn session_bus_reachable() -> bool {
        #[cfg(target_os = "linux")]
        {
            let Ok(addr) = std::env::var("DBUS_SESSION_BUS_ADDRESS") else {
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
                None => a.starts_with("unix:"),
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            true
        }
    }
}
