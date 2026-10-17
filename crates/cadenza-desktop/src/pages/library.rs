//! Library: local folders and, once signed in, the Encore library. Everything plays through
//! this engine.

use crate::app::{App, Message};
use crate::icons::{icon, Icon};
use crate::theme::{self, Btn};
use crate::ui::{self, El};
use cadenza::engine::{Command, Track};
use iced::widget::{button, column, container, image, row, text, Column, Row};
use iced::{Alignment, Length, Task};
use serde_json::Value;

#[derive(Default)]
pub struct State {
    pub tab: Tab,
    path: String,
    path_text: String,
    listing: Value,
    folders: Vec<String>,
    msg: String,
    // Encore
    query: String,
    albums: Vec<Value>,
    total: u64,
    songs: Vec<Value>,
    album: Option<Value>,
    emsg: String,
    loaded: bool,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    #[default]
    Local,
    Encore,
}

#[derive(Clone, Debug)]
pub enum Msg {
    Tab(Tab),
    Open(String),
    Opened(Result<Value, String>),
    PathText(String),
    PlayFolder(String, bool),
    Queued(Result<usize, String>, bool),
    PlayFile(usize),
    AddFile(usize),
    SaveFolder,
    ForgetFolder(usize),
    Query(String),
    Search,
    More,
    Albums(Result<(Value, Option<Value>), String>, bool),
    Album(String),
    AlbumLoaded(Result<Value, String>),
    Back,
    PlayEncore(Vec<Value>, usize, &'static str),
    Played(Result<usize, String>),
}

pub fn init(app: &mut App) -> Task<Message> {
    app.library.folders = cadenza::library::folders();
    let first = app.library.folders.first().cloned().unwrap_or_default();
    open(first)
}

fn open(path: String) -> Task<Message> {
    Task::perform(
        async move {
            tokio::task::spawn_blocking(move || cadenza::library::browse(&path))
                .await
                .unwrap_or_else(|e| Err(e.to_string()))
        },
        |r| Message::Library(Msg::Opened(r)),
    )
}

fn files(app: &App) -> Vec<String> {
    app.library.listing["files"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|f| f["path"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Paths to tracks (CUE sheets become their tracks).
fn tracks_for(paths: &[String]) -> Vec<Track> {
    paths
        .iter()
        .flat_map(|p| cadenza::library::track_for(p).expand().unwrap_or_default())
        .collect()
}

fn encore_load(app: &mut App, more: bool) -> Task<Message> {
    let Some(c) = cadenza::encore::client() else {
        app.library.emsg =
            "Not signed in. Sign in on the Encore page to browse your Encore library here.".into();
        app.library.albums.clear();
        app.library.songs.clear();
        return Task::none();
    };
    app.library.emsg = "Loading…".into();
    let q = app.library.query.clone();
    let offset = if more {
        app.library.albums.len() as u32
    } else {
        0
    };
    Task::perform(
        async move {
            tokio::task::spawn_blocking(move || {
                let al = cadenza::encore::library::albums(&c, &q, offset, 60)?;
                let tr = if !q.is_empty() && offset == 0 {
                    Some(cadenza::encore::library::tracks(&c, &q, 50)?)
                } else {
                    None
                };
                Ok((al, tr))
            })
            .await
            .unwrap_or_else(|e| Err(e.to_string()))
        },
        move |r| Message::Library(Msg::Albums(r, more)),
    )
}

pub fn update(app: &mut App, m: Msg) -> Task<Message> {
    match m {
        Msg::Tab(t) => {
            app.library.tab = t;
            if t == Tab::Encore && !app.library.loaded {
                app.library.loaded = true;
                return encore_load(app, false);
            }
        }
        Msg::Open(p) => return open(p),
        Msg::Opened(r) => match r {
            Ok(v) => {
                app.library.path = v["path"].as_str().unwrap_or("").into();
                app.library.path_text = app.library.path.clone();
                app.library.listing = v;
                app.library.msg.clear();
            }
            Err(e) => app.library.msg = e,
        },
        Msg::PathText(t) => app.library.path_text = t,
        Msg::PlayFolder(path, append) => {
            app.library.msg = "Reading the folder…".into();
            let engine = app.engine.clone();
            return Task::perform(
                async move {
                    tokio::task::spawn_blocking(move || {
                        let paths = cadenza::library::scan(std::path::Path::new(&path))?;
                        let t = tracks_for(&paths);
                        if t.is_empty() {
                            return Err("No music files in that folder.".to_string());
                        }
                        let n = t.len();
                        if append {
                            engine.send(Command::Enqueue(t));
                        } else {
                            engine.play_queue(t, 0, 0.0);
                        }
                        Ok(n)
                    })
                    .await
                    .unwrap_or_else(|e| Err(e.to_string()))
                },
                move |r| Message::Library(Msg::Queued(r, append)),
            );
        }
        Msg::Queued(r, append) => {
            app.library.msg = match r {
                Ok(n) => format!(
                    "{} {n} file{}.",
                    if append { "Added" } else { "Playing" },
                    if n == 1 { "" } else { "s" }
                ),
                Err(e) => e,
            }
        }
        Msg::PlayFile(i) => {
            let t = tracks_for(&files(app));
            if !t.is_empty() {
                app.engine.play_queue(t, i, 0.0);
            }
        }
        Msg::AddFile(i) => {
            if let Some(p) = files(app).get(i) {
                app.engine
                    .send(Command::Enqueue(tracks_for(std::slice::from_ref(p))));
                app.library.msg = "Added to the queue.".into();
            }
        }
        Msg::SaveFolder => {
            let mut f = app.library.folders.clone();
            if !app.library.path.is_empty() && !f.contains(&app.library.path) {
                f.push(app.library.path.clone());
            }
            cadenza::library::set_folders(f);
            app.library.folders = cadenza::library::folders();
        }
        Msg::ForgetFolder(i) => {
            let mut f = app.library.folders.clone();
            if i < f.len() {
                f.remove(i);
            }
            cadenza::library::set_folders(f);
            app.library.folders = cadenza::library::folders();
        }
        Msg::Query(q) => app.library.query = q,
        Msg::Search => {
            app.library.album = None;
            return encore_load(app, false);
        }
        Msg::More => return encore_load(app, true),
        Msg::Albums(r, more) => match r {
            Ok((al, tr)) => {
                let items = al["items"].as_array().cloned().unwrap_or_default();
                if more {
                    app.library.albums.extend(items);
                } else {
                    app.library.albums = items;
                }
                app.library.total = al["total"].as_u64().unwrap_or(0);
                if !more {
                    app.library.songs = tr.and_then(|t| t.as_array().cloned()).unwrap_or_default();
                }
                let q = &app.library.query;
                app.library.emsg = if app.library.total > 0 {
                    format!(
                        "{} album{}{}",
                        app.library.total,
                        if app.library.total == 1 { "" } else { "s" },
                        if q.is_empty() {
                            String::new()
                        } else {
                            format!(" matching \"{q}\"")
                        }
                    )
                } else if q.is_empty() {
                    "Your Encore library is empty.".into()
                } else {
                    "Nothing matches.".into()
                };
                let urls: Vec<String> = app
                    .library
                    .albums
                    .iter()
                    .filter_map(|a| a["id"].as_str().map(|id| format!("/api/encore/cover/{id}")))
                    .collect();
                return Task::batch(urls.iter().map(|u| app.want_cover(u)).collect::<Vec<_>>());
            }
            Err(e) => app.library.emsg = e,
        },
        Msg::Album(id) => {
            let Some(c) = cadenza::encore::client() else {
                return Task::none();
            };
            return Task::perform(
                async move {
                    tokio::task::spawn_blocking(move || cadenza::encore::library::album(&c, &id))
                        .await
                        .unwrap_or_else(|e| Err(e.to_string()))
                },
                |r| Message::Library(Msg::AlbumLoaded(r)),
            );
        }
        Msg::AlbumLoaded(r) => match r {
            Ok(a) => app.library.album = Some(a),
            Err(e) => app.library.emsg = e,
        },
        Msg::Back => app.library.album = None,
        Msg::PlayEncore(items, index, mode) => {
            let Some(c) = cadenza::encore::client() else {
                return Task::none();
            };
            let engine = app.engine.clone();
            return Task::perform(
                async move {
                    tokio::task::spawn_blocking(move || {
                        cadenza::encore::library::play(&c, &engine, &items, index, mode)
                    })
                    .await
                    .unwrap_or_else(|e| Err(e.to_string()))
                },
                |r| Message::Library(Msg::Played(r)),
            );
        }
        Msg::Played(r) => {
            if let Err(e) = r {
                app.library.emsg = e;
            }
        }
    }
    Task::none()
}

pub fn commit(_app: &mut App, _k: &str, _v: String) -> Task<Message> {
    Task::none()
}

fn list_row<'a>(icon: &'a str, name: String, on: Message, acts: Vec<El<'a>>) -> El<'a> {
    let l = theme::look();
    let mut r = Row::new().spacing(8).align_y(Alignment::Center).push(
        button(
            row![text(icon).color(l.muted).width(20), text(name).size(14)]
                .spacing(8)
                .align_y(Alignment::Center),
        )
        .padding([7, 10])
        .width(Length::Fill)
        .style(theme::btn(Btn::Ghost))
        .on_press(on),
    );
    for a in acts {
        r = r.push(a);
    }
    r.into()
}

fn small_btn<'a>(t: &'a str, m: Message) -> El<'a> {
    button(text(t).size(12).font(ui::bold()))
        .padding([4, 10])
        .style(theme::btn(Btn::Secondary))
        .on_press(m)
        .into()
}

fn local(app: &App) -> El<'_> {
    let st = &app.library;
    let l = theme::look();
    let mut c = Column::new().spacing(10);
    if !st.folders.is_empty() {
        let mut chips = Row::new().spacing(6);
        for (i, f) in st.folders.iter().enumerate() {
            let name = std::path::Path::new(f)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| f.clone());
            chips = chips.push(
                container(
                    row![
                        button(text(name).size(12.5).font(ui::bold()))
                            .padding([3, 8])
                            .style(theme::btn(Btn::Ghost))
                            .on_press(Message::Library(Msg::Open(f.clone()))),
                        button(text("×").size(12))
                            .padding([3, 6])
                            .style(theme::btn(Btn::Ghost))
                            .on_press(Message::Library(Msg::ForgetFolder(i))),
                    ]
                    .align_y(Alignment::Center),
                )
                .style(theme::inset),
            );
        }
        c = c.push(chips.wrap());
    }
    c = c.push(
        row![
            ui::input(
                "A folder, e.g. D:\\Music or /home/me/Music",
                &st.path_text,
                |v| Message::Library(Msg::PathText(v))
            )
            .on_submit(Message::Library(Msg::Open(st.path_text.clone()))),
            ui::btn(
                "Open",
                Btn::Secondary,
                Some(Message::Library(Msg::Open(st.path_text.clone())))
            ),
            ui::btn(
                "Drives",
                Btn::Ghost,
                Some(Message::Library(Msg::Open(String::new())))
            ),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    );
    if !st.path.is_empty() {
        let mut acts = row![
            ui::btn(
                "Play this folder",
                Btn::Primary,
                Some(Message::Library(Msg::PlayFolder(st.path.clone(), false)))
            ),
            ui::btn(
                "Add to queue",
                Btn::Secondary,
                Some(Message::Library(Msg::PlayFolder(st.path.clone(), true)))
            ),
        ]
        .spacing(8);
        if !st.folders.contains(&st.path) {
            acts = acts.push(ui::btn(
                "Save folder",
                Btn::Ghost,
                Some(Message::Library(Msg::SaveFolder)),
            ));
        }
        c = c.push(acts);
    }
    if !st.msg.is_empty() {
        c = c.push(text(st.msg.clone()).size(12.5).color(l.muted));
    }
    let v = &st.listing;
    let mut list = Column::new().spacing(2);
    if !st.path.is_empty() {
        list = list.push(list_row(
            "↑",
            "Up".into(),
            Message::Library(Msg::Open(v["parent"].as_str().unwrap_or("").into())),
            vec![],
        ));
    }
    for d in v["dirs"].as_array().cloned().unwrap_or_default() {
        let p = d["path"].as_str().unwrap_or("").to_string();
        let acts = if st.path.is_empty() {
            vec![]
        } else {
            vec![
                small_btn("Play", Message::Library(Msg::PlayFolder(p.clone(), false))),
                small_btn("Add", Message::Library(Msg::PlayFolder(p.clone(), true))),
            ]
        };
        list = list.push(list_row(
            "▸",
            d["name"].as_str().unwrap_or("").into(),
            Message::Library(Msg::Open(p)),
            acts,
        ));
    }
    for (i, f) in v["files"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        list = list.push(list_row(
            "♪",
            f["name"].as_str().unwrap_or("").into(),
            Message::Library(Msg::PlayFile(i)),
            vec![small_btn("Add", Message::Library(Msg::AddFile(i)))],
        ));
    }
    c = c.push(list);
    c = c.push(ui::hint(
        "You can also drop files or folders onto the window to add them to the queue.",
    ));
    ui::card(c)
}

fn song_rows<'a>(tracks: &[Value]) -> El<'a> {
    let l = theme::look();
    let mut list = Column::new().spacing(2);
    let all: Vec<Value> = tracks.to_vec();
    for (i, t) in tracks.iter().enumerate() {
        let name = format!(
            "{} — {}",
            t["title"]
                .as_str()
                .filter(|s| !s.is_empty())
                .unwrap_or("Untitled"),
            t["artist"].as_str().unwrap_or("")
        );
        let meta = format!(
            "{} {}",
            t["format"].as_str().unwrap_or("").to_uppercase(),
            t["duration"].as_f64().map(ui::mmss).unwrap_or_default()
        );
        list = list.push(list_row(
            "♪",
            name,
            Message::Library(Msg::PlayEncore(all.clone(), i, "replace")),
            vec![
                text(meta).size(12).color(l.muted).into(),
                small_btn(
                    "Add",
                    Message::Library(Msg::PlayEncore(vec![t.clone()], 0, "append")),
                ),
            ],
        ));
    }
    list.into()
}

fn encore(app: &App) -> El<'_> {
    let st = &app.library;
    let l = theme::look();
    let mut c = Column::new().spacing(12);
    c = c.push(
        row![
            ui::input("Search albums, artists and songs", &st.query, |v| {
                Message::Library(Msg::Query(v))
            })
            .on_submit(Message::Library(Msg::Search)),
            ui::btn(
                "Search",
                Btn::Secondary,
                Some(Message::Library(Msg::Search))
            ),
        ]
        .spacing(8),
    );
    if !st.emsg.is_empty() {
        c = c.push(text(st.emsg.clone()).size(12.5).color(l.muted));
    }
    if let Some(a) = &st.album {
        let id = a["id"].as_str().unwrap_or("").to_string();
        let tracks = a["tracks"].as_array().cloned().unwrap_or_default();
        let art: El = match app.cover(Some(&format!("/api/encore/cover/{id}"))) {
            Some(h) => image(h).width(160).height(160).into(),
            None => container(icon(Icon::Music, 48.0, l.on_accent))
                .width(160)
                .height(160)
                .center(160)
                .style(theme::accent_tile)
                .into(),
        };
        let sub = [
            a["albumartist"].as_str().map(str::to_string),
            a["year"].as_u64().map(|y| y.to_string()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" · ");
        c = c
            .push(ui::btn(
                "← Albums",
                Btn::Ghost,
                Some(Message::Library(Msg::Back)),
            ))
            .push(
                row![
                    art,
                    column![
                        text(a["album"].as_str().unwrap_or("Unknown album").to_string())
                            .size(22)
                            .font(ui::bold()),
                        text(sub).color(l.muted),
                        row![
                            ui::btn(
                                "Play album",
                                Btn::Primary,
                                Some(Message::Library(Msg::PlayEncore(
                                    tracks.clone(),
                                    0,
                                    "replace"
                                )))
                            ),
                            ui::btn(
                                "Add to queue",
                                Btn::Secondary,
                                Some(Message::Library(Msg::PlayEncore(
                                    tracks.clone(),
                                    0,
                                    "append"
                                )))
                            ),
                        ]
                        .spacing(8)
                    ]
                    .spacing(8)
                ]
                .spacing(18)
                .align_y(Alignment::Center),
            )
            .push(song_rows(&tracks));
        return ui::card(c);
    }
    if !st.songs.is_empty() {
        c = c
            .push(ui::h2("Songs"))
            .push(song_rows(&st.songs))
            .push(ui::h2("Albums"));
    }
    // Album covers in rows that fill the width.
    let per_row = ((app.width - 320.0) / 180.0).floor().max(2.0) as usize;
    let mut grid = Column::new().spacing(14);
    for chunk in st.albums.chunks(per_row) {
        let mut r = Row::new().spacing(14);
        for a in chunk {
            let id = a["id"].as_str().unwrap_or("").to_string();
            let art: El = match app.cover(Some(&format!("/api/encore/cover/{id}"))) {
                Some(h) => image(h).width(160).height(160).into(),
                None => container(text(""))
                    .width(160)
                    .height(160)
                    .style(theme::inset)
                    .into(),
            };
            let sub = [
                a["albumartist"].as_str().map(str::to_string),
                a["year"].as_u64().map(|y| y.to_string()),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" · ");
            r = r.push(
                button(
                    column![
                        art,
                        text(a["album"].as_str().unwrap_or("Unknown album").to_string())
                            .size(13.5)
                            .font(ui::bold()),
                        text(sub).size(12).color(l.muted)
                    ]
                    .spacing(4)
                    .width(160),
                )
                .padding(4)
                .style(theme::btn(Btn::Ghost))
                .on_press(Message::Library(Msg::Album(id))),
            );
        }
        grid = grid.push(r);
    }
    c = c.push(grid);
    if (st.albums.len() as u64) < st.total {
        c = c.push(ui::btn(
            "More albums",
            Btn::Secondary,
            Some(Message::Library(Msg::More)),
        ));
    }
    ui::card(c)
}

pub fn view(app: &App, _wide: bool) -> El<'_> {
    let tab = match app.library.tab {
        Tab::Local => "local",
        Tab::Encore => "encore",
    };
    column![
        ui::h1("Library", "Your music folders and, once you sign in on the Encore page, your Encore library. Everything plays through this engine to your DAC."),
        ui::seg(&[("local", "Local"), ("encore", "Encore")], tab, |t| Message::Library(Msg::Tab(if t == "encore" { Tab::Encore } else { Tab::Local }))),
        iced::widget::space().height(14),
        match app.library.tab {
            Tab::Local => local(app),
            Tab::Encore => encore(app),
        },
    ]
    .into()
}
