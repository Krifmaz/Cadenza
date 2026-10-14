//! OpenHome services on the renderer – Product, Playlist, Volume, Time and Info – for
//! the control points built on them (Linn Kazoo, Lumin, BubbleUPnP's OpenHome mode …).
//!
//! The playlist lives here, not in the engine: the engine is given the current entry and
//! the one after it, and [`OpenHome::tick`] queues the following entry each time the
//! engine moves on, so playback stays gapless while the controller edits the list freely.
//! Only `http(s)` URIs are accepted, as for AVTransport.

use super::renderer::{track_for, Fault, Out};
use super::xml::{self, escape};
use crate::engine::{Command, Engine, PlayState, Status};
use cadenza_dsp::Volume;
use std::sync::Mutex;

/// Most entries the playlist holds.
pub const TRACKS_MAX: usize = 1000;

impl Fault {
    /// OpenHome 800: the id or index does not exist.
    pub const OH_INVALID: Fault = Fault {
        code: 800,
        description: "Id not found",
    };
    /// OpenHome 801: the playlist is full.
    pub const OH_FULL: Fault = Fault {
        code: 801,
        description: "Playlist full",
    };
}

#[derive(Clone, Debug, PartialEq)]
struct Entry {
    id: u32,
    uri: String,
    meta: String,
}

/// The two sources Product lists.
pub const SOURCES: [(&str, &str); 2] = [("Playlist", "Playlist"), ("UPnP AV", "UpnpAv")];

#[derive(Debug)]
struct State {
    entries: Vec<Entry>,
    next_id: u32,
    /// Changes of the id array (`IdArrayToken`).
    token: u32,
    /// The entry the engine plays (0 = none).
    current: u32,
    /// The entry queued behind it in the engine (0 = none).
    queued: u32,
    repeat: bool,
    shuffle: bool,
    rng: u64,
    source_index: u32,
    standby: bool,
    /// Tracks started (Time / Info `TrackCount`).
    track_count: u32,
    /// The engine's track the last count was taken for.
    counted: Option<String>,
}

/// The OpenHome side of the renderer.
pub struct OpenHome {
    engine: Engine,
    st: Mutex<State>,
    /// Room / product name.
    pub name: String,
}

fn tag(id: u32) -> String {
    format!("oh:{id}")
}

fn id_of(status: &Status) -> Option<u32> {
    status
        .track
        .as_ref()?
        .id
        .as_deref()?
        .strip_prefix("oh:")?
        .parse()
        .ok()
}

fn arg(body: &str, name: &str) -> Result<String, Fault> {
    xml::text(body, name).ok_or(Fault::INVALID_ARGS)
}

fn num<T: std::str::FromStr>(body: &str, name: &str) -> Result<T, Fault> {
    arg(body, name)?
        .trim()
        .parse()
        .map_err(|_| Fault::INVALID_ARGS)
}

fn boolean(body: &str, name: &str) -> Result<bool, Fault> {
    match arg(body, name)?.trim() {
        "1" | "true" | "yes" => Ok(true),
        "0" | "false" | "no" => Ok(false),
        _ => Err(Fault::INVALID_ARGS),
    }
}

fn b(v: bool) -> String {
    if v { "true" } else { "false" }.into()
}

/// Standard base64 of `bytes`.
pub fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// `(codec, bit depth, sample rate)` from a `FLAC 16/44.1` source label.
fn details(label: Option<&str>) -> (String, u32, u32) {
    let Some(label) = label else {
        return (String::new(), 0, 0);
    };
    let mut parts = label.split_whitespace();
    let codec = parts.next().unwrap_or("").to_string();
    let (mut bits, mut rate) = (0, 0);
    for p in parts {
        if let Some((bd, r)) = p.split_once('/') {
            bits = bd.parse().unwrap_or(0);
            rate = r
                .parse::<f64>()
                .map(|k| (k * 1000.0).round() as u32)
                .unwrap_or(0);
        }
    }
    (codec, bits, rate)
}

impl OpenHome {
    /// The services for `engine`, with product name `name`.
    pub fn new(engine: Engine, name: String) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1)
            | 1;
        Self {
            engine,
            st: Mutex::new(State {
                entries: Vec::new(),
                next_id: 1,
                token: 1,
                current: 0,
                queued: 0,
                repeat: false,
                shuffle: false,
                rng: seed,
                source_index: 0,
                standby: false,
                track_count: 0,
                counted: None,
            }),
            name,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// AVTransport took over: Product shows the UPnP AV source.
    pub fn avtransport_active(&self) {
        let mut st = self.lock();
        st.source_index = 1;
        st.current = 0;
        st.queued = 0;
    }

    fn pos(st: &State, id: u32) -> Option<usize> {
        st.entries.iter().position(|e| e.id == id)
    }

    /// The entry to play after `id` (repeat wraps, shuffle picks another at random).
    fn after(st: &mut State, id: u32) -> Option<u32> {
        let i = Self::pos(st, id)?;
        let n = st.entries.len();
        if st.shuffle && n > 1 {
            st.rng ^= st.rng << 13;
            st.rng ^= st.rng >> 7;
            st.rng ^= st.rng << 17;
            let mut j = (st.rng % (n as u64 - 1)) as usize;
            if j >= i {
                j += 1;
            }
            return Some(st.entries[j].id);
        }
        match st.entries.get(i + 1) {
            Some(e) => Some(e.id),
            None if st.repeat => st.entries.first().map(|e| e.id),
            None => None,
        }
    }

    fn track(e: &Entry) -> crate::engine::Track {
        let mut t = track_for(&e.uri, &e.meta);
        t.id = Some(tag(e.id));
        t
    }

    /// Start playing entry `id` (and queue the one after it).
    fn play_id(&self, st: &mut State, id: u32) -> Result<(), Fault> {
        let i = Self::pos(st, id).ok_or(Fault::OH_INVALID)?;
        let mut tracks = vec![Self::track(&st.entries[i])];
        st.current = id;
        st.queued = 0;
        if let Some(n) = Self::after(st, id) {
            let p = Self::pos(st, n).expect("after() returns a listed id");
            tracks.push(Self::track(&st.entries[p]));
            st.queued = n;
        }
        st.source_index = 0;
        st.standby = false;
        self.engine.play_queue(tracks, 0, 0.0);
        Ok(())
    }

    /// The engine plays one of ours.
    fn ours(st: &State, status: &Status) -> bool {
        st.source_index == 0 && status.state != PlayState::Stopped && id_of(status).is_some()
    }

    /// Re-queue the entry after the current one if it changed (after an edit).
    fn requeue(&self, st: &mut State, status: &Status) {
        if !Self::ours(st, status) {
            return;
        }
        let want = Self::after(st, st.current).unwrap_or(0);
        if want == st.queued {
            return;
        }
        if st.queued != 0 {
            if let Some(i) = status.index {
                self.engine.send(Command::Remove(i + 1));
            }
        }
        st.queued = 0;
        if want != 0 {
            let p = Self::pos(st, want).expect("listed");
            self.engine
                .send(Command::Enqueue(vec![Self::track(&st.entries[p])]));
            st.queued = want;
        }
    }

    /// Follow the engine (call regularly): when it has moved on to the queued entry,
    /// that is current and the one after it is queued.
    pub fn tick(&self) {
        let status = self.engine.status();
        let mut st = self.lock();
        let key = status.track.as_ref().map(|t| {
            format!(
                "{}|{}",
                t.id.as_deref().unwrap_or(""),
                status.index.unwrap_or(0)
            )
        });
        if key.is_some() && key != st.counted && status.state != PlayState::Stopped {
            st.track_count = st.track_count.wrapping_add(1);
            st.counted = key;
        }
        if st.source_index != 0 {
            return;
        }
        match id_of(&status) {
            Some(id) if id == st.queued && id != st.current => {
                st.current = id;
                st.queued = 0;
                if let Some(n) = Self::after(&mut st, id) {
                    let p = Self::pos(&st, n).expect("listed");
                    self.engine
                        .send(Command::Enqueue(vec![Self::track(&st.entries[p])]));
                    st.queued = n;
                }
            }
            _ => {}
        }
    }

    fn transport_state(st: &State, status: &Status) -> &'static str {
        if !Self::ours(st, status) {
            return "Stopped";
        }
        match status.state {
            PlayState::Playing => "Playing",
            PlayState::Paused => "Paused",
            PlayState::Stopped => "Stopped",
        }
    }

    fn id_array(st: &State) -> String {
        let bytes: Vec<u8> = st.entries.iter().flat_map(|e| e.id.to_be_bytes()).collect();
        base64(&bytes)
    }

    fn source_xml() -> String {
        let sources: String = SOURCES
            .iter()
            .map(|(name, ty)| {
                format!(
                    "<Source><Name>{name}</Name><Type>{ty}</Type><Visible>true</Visible></Source>"
                )
            })
            .collect();
        format!("<SourceList>{sources}</SourceList>")
    }

    fn volume(status: &Status) -> u32 {
        (status.volume_linear * 100.0).round().clamp(0.0, 100.0) as u32
    }

    /// Run `action` of OpenHome service `service`.
    pub fn handle(&self, service: &str, action: &str, body: &str) -> Result<Out, Fault> {
        let status = self.engine.status();
        let mut st = self.lock();
        let out: Out = match (service, action) {
            // ---- Product ----
            ("Product", "Manufacturer") => vec![
                ("Name", "Cadenza".into()),
                ("Info", "Open-source upsampling player".into()),
                ("Url", "https://github.com/Krifmaz/cadenza".into()),
                ("ImageUri", String::new()),
            ],
            ("Product", "Model") => vec![
                ("Name", "Cadenza".into()),
                ("Info", env!("CARGO_PKG_VERSION").into()),
                ("Url", "https://github.com/Krifmaz/cadenza".into()),
                ("ImageUri", String::new()),
            ],
            ("Product", "Product") => vec![
                ("Room", self.name.clone()),
                ("Name", "Cadenza".into()),
                ("Info", String::new()),
                ("Url", String::new()),
                ("ImageUri", String::new()),
            ],
            ("Product", "Standby") => vec![("Value", b(st.standby))],
            ("Product", "SetStandby") => {
                st.standby = boolean(body, "Value")?;
                if st.standby {
                    self.engine.send(Command::Stop);
                }
                vec![]
            }
            ("Product", "SourceCount") => vec![("Value", SOURCES.len().to_string())],
            ("Product", "SourceXml") => vec![("Value", Self::source_xml())],
            ("Product", "SourceIndex") => vec![("Value", st.source_index.to_string())],
            ("Product", "SetSourceIndex") => {
                let i: u32 = num(body, "Value")?;
                if i as usize >= SOURCES.len() {
                    return Err(Fault::OH_INVALID);
                }
                if i != st.source_index {
                    self.engine.send(Command::Stop);
                    st.source_index = i;
                }
                vec![]
            }
            ("Product", "SetSourceIndexByName" | "SetSourceBySystemName") => {
                let name = arg(body, "Value")?;
                let i = SOURCES
                    .iter()
                    .position(|(n, t)| *n == name.trim() || *t == name.trim())
                    .ok_or(Fault::OH_INVALID)?;
                if i as u32 != st.source_index {
                    self.engine.send(Command::Stop);
                    st.source_index = i as u32;
                }
                vec![]
            }
            ("Product", "Source") => {
                let i: usize = num(body, "Index")?;
                let (name, ty) = SOURCES.get(i).ok_or(Fault::OH_INVALID)?;
                vec![
                    ("SystemName", (*name).into()),
                    ("Type", (*ty).into()),
                    ("Name", (*name).into()),
                    ("Visible", b(true)),
                ]
            }
            ("Product", "Attributes") => vec![("Value", "Info Time Volume".into())],
            ("Product", "SourceXmlChangeCount") => vec![("Value", "0".into())],

            // ---- Playlist ----
            ("Playlist", "Play") => {
                if Self::ours(&st, &status) {
                    self.engine.send(Command::Play);
                } else {
                    let id = if Self::pos(&st, st.current).is_some() {
                        st.current
                    } else {
                        st.entries
                            .first()
                            .map(|e| e.id)
                            .ok_or(Fault::TRANSITION_NOT_AVAILABLE)?
                    };
                    self.play_id(&mut st, id)?;
                }
                vec![]
            }
            ("Playlist", "Pause") => {
                if Self::ours(&st, &status) {
                    self.engine.send(Command::Pause);
                }
                vec![]
            }
            ("Playlist", "Stop") => {
                if Self::ours(&st, &status) {
                    self.engine.send(Command::Stop);
                }
                vec![]
            }
            ("Playlist", "Next") => {
                let cur = st.current;
                if let Some(n) = Self::after(&mut st, cur) {
                    self.play_id(&mut st, n)?;
                }
                vec![]
            }
            ("Playlist", "Previous") => {
                if let Some(i) = Self::pos(&st, st.current).filter(|&i| i > 0) {
                    let id = st.entries[i - 1].id;
                    self.play_id(&mut st, id)?;
                } else if st.repeat {
                    if let Some(id) = st.entries.last().map(|e| e.id) {
                        self.play_id(&mut st, id)?;
                    }
                }
                vec![]
            }
            ("Playlist", "SetRepeat") => {
                st.repeat = boolean(body, "Value")?;
                self.requeue(&mut st, &status);
                vec![]
            }
            ("Playlist", "Repeat") => vec![("Value", b(st.repeat))],
            ("Playlist", "SetShuffle") => {
                st.shuffle = boolean(body, "Value")?;
                self.requeue(&mut st, &status);
                vec![]
            }
            ("Playlist", "Shuffle") => vec![("Value", b(st.shuffle))],
            ("Playlist", "SeekSecondAbsolute") => {
                let s: u32 = num(body, "Value")?;
                if Self::ours(&st, &status) {
                    self.engine.send(Command::Seek(s as f64));
                }
                vec![]
            }
            ("Playlist", "SeekSecondRelative") => {
                let s: i32 = num(body, "Value")?;
                if Self::ours(&st, &status) {
                    self.engine
                        .send(Command::Seek((status.position + s as f64).max(0.0)));
                }
                vec![]
            }
            ("Playlist", "SeekId") => {
                let id: u32 = num(body, "Value")?;
                self.play_id(&mut st, id)?;
                vec![]
            }
            ("Playlist", "SeekIndex") => {
                let i: usize = num(body, "Value")?;
                let id = st.entries.get(i).map(|e| e.id).ok_or(Fault::OH_INVALID)?;
                self.play_id(&mut st, id)?;
                vec![]
            }
            ("Playlist", "TransportState") => {
                vec![("Value", Self::transport_state(&st, &status).into())]
            }
            ("Playlist", "Id") => vec![(
                "Value",
                if Self::pos(&st, st.current).is_some() {
                    st.current
                } else {
                    0
                }
                .to_string(),
            )],
            ("Playlist", "Read") => {
                let id: u32 = num(body, "Id")?;
                let e = &st.entries[Self::pos(&st, id).ok_or(Fault::OH_INVALID)?];
                vec![("Uri", e.uri.clone()), ("Metadata", e.meta.clone())]
            }
            ("Playlist", "ReadList") => {
                let list: String = arg(body, "IdList")?
                    .split_whitespace()
                    .filter_map(|t| t.parse::<u32>().ok())
                    .filter_map(|id| Self::pos(&st, id).map(|i| &st.entries[i]))
                    .map(|e| {
                        format!(
                            "<Entry><Id>{}</Id><Uri>{}</Uri><Metadata>{}</Metadata></Entry>",
                            e.id,
                            escape(&e.uri),
                            escape(&e.meta)
                        )
                    })
                    .collect();
                vec![("TrackList", format!("<TrackList>{list}</TrackList>"))]
            }
            ("Playlist", "Insert") => {
                let after: u32 = num(body, "AfterId")?;
                let uri = arg(body, "Uri")?.trim().to_string();
                let meta = xml::text(body, "Metadata").unwrap_or_default();
                let lower = uri.to_ascii_lowercase();
                if !(lower.starts_with("http://") || lower.starts_with("https://")) {
                    return Err(Fault::RESOURCE_NOT_FOUND);
                }
                if st.entries.len() >= TRACKS_MAX {
                    return Err(Fault::OH_FULL);
                }
                let at = if after == 0 {
                    0
                } else {
                    Self::pos(&st, after).ok_or(Fault::OH_INVALID)? + 1
                };
                let id = st.next_id;
                st.next_id = st.next_id.wrapping_add(1).max(1);
                st.entries.insert(at, Entry { id, uri, meta });
                st.token = st.token.wrapping_add(1);
                self.requeue(&mut st, &status);
                vec![("NewId", id.to_string())]
            }
            ("Playlist", "DeleteId") => {
                let id: u32 = num(body, "Value")?;
                let Some(i) = Self::pos(&st, id) else {
                    // Already gone: not an error for OpenHome.
                    return Ok(vec![]);
                };
                if id == st.current && Self::ours(&st, &status) {
                    // The playing entry goes: play on with the one after it (or stop).
                    let next = Self::after(&mut st, id).filter(|&n| n != id);
                    st.entries.remove(i);
                    st.token = st.token.wrapping_add(1);
                    match next {
                        Some(n) if status.state == PlayState::Playing => {
                            self.play_id(&mut st, n)?
                        }
                        _ => {
                            self.engine.send(Command::Stop);
                            st.current = next.unwrap_or(0);
                            st.queued = 0;
                        }
                    }
                } else {
                    st.entries.remove(i);
                    st.token = st.token.wrapping_add(1);
                    if id == st.current {
                        st.current = 0;
                    }
                    self.requeue(&mut st, &status);
                }
                vec![]
            }
            ("Playlist", "DeleteAll") => {
                if Self::ours(&st, &status) {
                    self.engine.send(Command::Stop);
                }
                st.entries.clear();
                st.current = 0;
                st.queued = 0;
                st.token = st.token.wrapping_add(1);
                vec![]
            }
            ("Playlist", "TracksMax") => vec![("Value", TRACKS_MAX.to_string())],
            ("Playlist", "IdArray") => vec![
                ("Token", st.token.to_string()),
                ("Array", Self::id_array(&st)),
            ],
            ("Playlist", "IdArrayChanged") => {
                let t: u32 = num(body, "Token")?;
                vec![("Value", b(t != st.token))]
            }
            ("Playlist", "ProtocolInfo") => {
                vec![("Value", super::description::sink_protocol_info())]
            }

            // ---- Volume ----
            ("Volume", "Characteristics") => vec![
                ("VolumeMax", "100".into()),
                ("VolumeUnity", "100".into()),
                ("VolumeSteps", "100".into()),
                ("VolumeMilliDbPerStep", "1000".into()),
                ("BalanceMax", "0".into()),
                ("FadeMax", "0".into()),
            ],
            ("Volume", "SetVolume") => {
                let v: u32 = num(body, "Value")?;
                if v > 100 {
                    return Err(Fault::INVALID_ARGS);
                }
                self.set_volume(&status, v);
                vec![]
            }
            ("Volume", "VolumeInc") => {
                self.set_volume(&status, (Self::volume(&status) + 1).min(100));
                vec![]
            }
            ("Volume", "VolumeDec") => {
                self.set_volume(&status, Self::volume(&status).saturating_sub(1));
                vec![]
            }
            ("Volume", "Volume") => vec![("Value", Self::volume(&status).to_string())],
            ("Volume", "SetBalance" | "SetFade") => {
                let v: i32 = num(body, "Value")?;
                if v != 0 {
                    return Err(Fault::INVALID_ARGS);
                }
                vec![]
            }
            ("Volume", "BalanceInc" | "BalanceDec" | "FadeInc" | "FadeDec") => vec![],
            ("Volume", "Balance" | "Fade") => vec![("Value", "0".into())],
            ("Volume", "SetMute") => {
                let m = boolean(body, "Value")?;
                self.engine.send(Command::SetVolume(Volume {
                    muted: m,
                    ..status.volume
                }));
                vec![]
            }
            ("Volume", "Mute") => vec![("Value", b(status.volume.muted))],
            ("Volume", "VolumeLimit") => vec![("Value", "100".into())],
            ("Volume", "UnityGain") => vec![("Value", b(false))],

            // ---- Time ----
            ("Time", "Time") => vec![
                ("TrackCount", st.track_count.to_string()),
                (
                    "Duration",
                    status.duration.unwrap_or(0.0).round().to_string(),
                ),
                ("Seconds", status.position.floor().max(0.0).to_string()),
            ],

            // ---- Info ----
            ("Info", "Counters") => vec![
                ("TrackCount", st.track_count.to_string()),
                ("DetailsCount", st.track_count.to_string()),
                ("MetatextCount", "0".into()),
            ],
            ("Info", "Track") => {
                let (uri, meta) = self.now_playing(&st, &status);
                vec![("Uri", uri), ("Metadata", meta)]
            }
            ("Info", "Details") => {
                let (codec, bits, rate) = details(status.source_format.as_deref());
                vec![
                    (
                        "Duration",
                        status.duration.unwrap_or(0.0).round().to_string(),
                    ),
                    ("BitRate", (bits as u64 * rate as u64 * 2).to_string()),
                    ("BitDepth", bits.to_string()),
                    ("SampleRate", rate.to_string()),
                    (
                        "Lossless",
                        b(!matches!(codec.as_str(), "MP3" | "AAC" | "Vorbis" | "Opus")),
                    ),
                    ("CodecName", codec),
                ]
            }
            ("Info", "Metatext") => vec![("Value", String::new())],
            _ => return Err(Fault::INVALID_ACTION),
        };
        Ok(out)
    }

    fn set_volume(&self, status: &Status, v: u32) {
        let mut vol = Volume::from_linear(v as f64 / 100.0);
        vol.muted = status.volume.muted;
        self.engine.send(Command::SetVolume(vol));
    }

    fn now_playing(&self, st: &State, status: &Status) -> (String, String) {
        if let Some(e) = id_of(status)
            .and_then(|id| Self::pos(st, id))
            .map(|i| &st.entries[i])
        {
            return (e.uri.clone(), e.meta.clone());
        }
        match status.track.as_ref() {
            Some(t) => match &t.source {
                crate::decode::Source::Url { url, .. } => {
                    (url.clone(), super::renderer::didl_for(t, url))
                }
                _ => (String::new(), String::new()),
            },
            None => (String::new(), String::new()),
        }
    }

    /// The evented variables of OpenHome service `service`.
    pub fn evented(&self, service: &str) -> Vec<(&'static str, String)> {
        let status = self.engine.status();
        let st = self.lock();
        match service {
            "Product" => vec![
                ("ManufacturerName", "Cadenza".into()),
                ("ManufacturerInfo", "Open-source upsampling player".into()),
                (
                    "ManufacturerUrl",
                    "https://github.com/Krifmaz/cadenza".into(),
                ),
                ("ManufacturerImageUri", String::new()),
                ("ModelName", "Cadenza".into()),
                ("ModelInfo", env!("CARGO_PKG_VERSION").into()),
                ("ModelUrl", "https://github.com/Krifmaz/cadenza".into()),
                ("ModelImageUri", String::new()),
                ("ProductRoom", self.name.clone()),
                ("ProductName", "Cadenza".into()),
                ("ProductInfo", String::new()),
                ("ProductUrl", String::new()),
                ("ProductImageUri", String::new()),
                ("Standby", b(st.standby)),
                ("SourceIndex", st.source_index.to_string()),
                ("SourceCount", SOURCES.len().to_string()),
                ("SourceXml", Self::source_xml()),
                ("Attributes", "Info Time Volume".into()),
            ],
            "Playlist" => vec![
                ("TransportState", Self::transport_state(&st, &status).into()),
                ("Repeat", b(st.repeat)),
                ("Shuffle", b(st.shuffle)),
                (
                    "Id",
                    if Self::pos(&st, st.current).is_some() {
                        st.current
                    } else {
                        0
                    }
                    .to_string(),
                ),
                ("IdArray", Self::id_array(&st)),
                ("TracksMax", TRACKS_MAX.to_string()),
                ("ProtocolInfo", super::description::sink_protocol_info()),
            ],
            "Volume" => vec![
                ("Volume", Self::volume(&status).to_string()),
                ("Mute", b(status.volume.muted)),
                ("Balance", "0".into()),
                ("Fade", "0".into()),
                ("VolumeLimit", "100".into()),
                ("VolumeMax", "100".into()),
                ("VolumeUnity", "100".into()),
                ("VolumeSteps", "100".into()),
                ("VolumeMilliDbPerStep", "1000".into()),
                ("BalanceMax", "0".into()),
                ("FadeMax", "0".into()),
                ("UnityGain", b(false)),
            ],
            "Time" => vec![
                ("TrackCount", st.track_count.to_string()),
                (
                    "Duration",
                    status.duration.unwrap_or(0.0).round().to_string(),
                ),
                ("Seconds", status.position.floor().max(0.0).to_string()),
            ],
            "Info" => {
                let (uri, meta) = self.now_playing(&st, &status);
                let (codec, bits, rate) = details(status.source_format.as_deref());
                vec![
                    ("TrackCount", st.track_count.to_string()),
                    ("DetailsCount", st.track_count.to_string()),
                    ("MetatextCount", "0".into()),
                    ("Uri", uri),
                    ("Metadata", meta),
                    (
                        "Duration",
                        status.duration.unwrap_or(0.0).round().to_string(),
                    ),
                    ("BitRate", (bits as u64 * rate as u64 * 2).to_string()),
                    ("BitDepth", bits.to_string()),
                    ("SampleRate", rate.to_string()),
                    (
                        "Lossless",
                        b(!matches!(codec.as_str(), "MP3" | "AAC" | "Vorbis" | "Opus")),
                    ),
                    ("CodecName", codec),
                    ("Metatext", String::new()),
                ]
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_and_details() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(&[0, 0, 0, 1, 0, 0, 0, 2]), "AAAAAQAAAAI=");
        assert_eq!(details(Some("FLAC 24/96")), ("FLAC".into(), 24, 96_000));
        assert_eq!(details(Some("WAV 16/44.1")), ("WAV".into(), 16, 44_100));
        assert_eq!(details(None), (String::new(), 0, 0));
    }
}
