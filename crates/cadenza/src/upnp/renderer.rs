//! The MediaRenderer's behaviour: the AVTransport state machine, RenderingControl volume
//! and mute, and ConnectionManager, on top of the player [`Engine`].
//!
//! A controller sets a URI (`SetAVTransportURI`), optionally the next one
//! (`SetNextAVTransportURI`, which we queue in the engine so the change is gapless), then
//! sends `Play`, `Pause`, `Seek`, `Stop`. Only `http(s)` URIs are accepted: anyone on the
//! network can send these commands, and a renderer must not be a way to open local files.

use super::description;
use super::xml::{self, escape};
use crate::decode::Source;
use crate::engine::{Command, Engine, PlayState, Status, Track};
use cadenza_dsp::Volume;
use std::sync::Mutex;

/// A UPnP error: code and description (sent back as a SOAP fault).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fault {
    /// UPnP error code.
    pub code: u16,
    /// Description.
    pub description: &'static str,
}

impl Fault {
    /// 401: no such action.
    pub const INVALID_ACTION: Fault = Fault {
        code: 401,
        description: "Invalid Action",
    };
    /// 402: missing or bad arguments.
    pub const INVALID_ARGS: Fault = Fault {
        code: 402,
        description: "Invalid Args",
    };
    /// 701: the transition is not possible now.
    pub const TRANSITION_NOT_AVAILABLE: Fault = Fault {
        code: 701,
        description: "Transition not available",
    };
    /// 706 (ConnectionManager): unknown connection id.
    pub const INVALID_CONNECTION: Fault = Fault {
        code: 706,
        description: "Invalid connection reference",
    };
    /// 710: unsupported seek mode.
    pub const SEEK_MODE: Fault = Fault {
        code: 710,
        description: "Seek mode not supported",
    };
    /// 711: bad seek target.
    pub const SEEK_TARGET: Fault = Fault {
        code: 711,
        description: "Illegal seek target",
    };
    /// 716: the URI cannot be played (not http/https).
    pub const RESOURCE_NOT_FOUND: Fault = Fault {
        code: 716,
        description: "Resource not found",
    };
    /// 718: bad instance id.
    pub const INVALID_INSTANCE: Fault = Fault {
        code: 718,
        description: "Invalid InstanceID",
    };
}

/// Output arguments of an action, in SCPD order.
pub type Out = Vec<(&'static str, String)>;

#[derive(Clone, Debug, Default, PartialEq)]
struct Media {
    uri: String,
    meta: String,
}

#[derive(Default)]
struct State {
    current: Media,
    next: Media,
    /// The next URI has been queued in the engine behind the current one.
    next_queued: bool,
}

/// The renderer (one per process).
pub struct Renderer {
    engine: Engine,
    state: Mutex<State>,
    oh: super::openhome::OpenHome,
}

fn url_of(t: &Track) -> Option<&str> {
    match &t.source {
        Source::Url { url, .. } => Some(url),
        Source::Path { .. } => None,
    }
}

/// A queue entry for `uri` with what DIDL-Lite `meta` says about it.
pub fn track_for(uri: &str, meta: &str) -> Track {
    let title = xml::text(meta, "title")
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| {
            let path = uri.split(['?', '#']).next().unwrap_or(uri);
            path.rsplit('/').next().unwrap_or(path).to_string()
        });
    let artist = xml::text(meta, "artist")
        .or_else(|| xml::text(meta, "creator"))
        .unwrap_or_default();
    Track {
        id: None,
        source: Source::Url {
            url: uri.to_string(),
            headers: Vec::new(),
        },
        title,
        artist,
        album: xml::text(meta, "album").unwrap_or_default(),
        album_id: None,
        duration: xml::attr(meta, "res", "duration").and_then(|d| xml::parse_time(&d)),
        format: None,
        cover: xml::text(meta, "albumArtURI").filter(|u| !u.is_empty()),
    }
}

/// DIDL-Lite for a track the controller gave no metadata for.
pub fn didl_for(t: &Track, uri: &str) -> String {
    let duration = t
        .duration
        .map(|d| format!(r#" duration="{}""#, xml::format_time(d)))
        .unwrap_or_default();
    format!(
        r#"<DIDL-Lite xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/"><item id="0" parentID="-1" restricted="1"><dc:title>{}</dc:title><dc:creator>{}</dc:creator><upnp:artist>{}</upnp:artist><upnp:album>{}</upnp:album><upnp:class>object.item.audioItem.musicTrack</upnp:class><res protocolInfo="http-get:*:*:*"{duration}>{}</res></item></DIDL-Lite>"#,
        escape(&t.title),
        escape(&t.artist),
        escape(&t.artist),
        escape(&t.album),
        escape(uri)
    )
}

fn arg(body: &str, name: &str) -> Result<String, Fault> {
    xml::text(body, name).ok_or(Fault::INVALID_ARGS)
}

fn check_instance(body: &str) -> Result<(), Fault> {
    match xml::text(body, "InstanceID").as_deref().map(str::trim) {
        Some("0") => Ok(()),
        Some(_) => Err(Fault::INVALID_INSTANCE),
        None => Err(Fault::INVALID_ARGS),
    }
}

fn http_only(uri: &str) -> Result<(), Fault> {
    let lower = uri.trim().to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        Ok(())
    } else {
        Err(Fault::RESOURCE_NOT_FOUND)
    }
}

impl Renderer {
    /// A renderer driving `engine`, named `name` (OpenHome's room).
    pub fn new(engine: Engine, name: String) -> Self {
        Self {
            oh: super::openhome::OpenHome::new(engine.clone(), name),
            engine,
            state: Mutex::new(State::default()),
        }
    }

    /// Follow the engine (called regularly by the event loop).
    pub fn tick(&self) {
        self.oh.tick();
    }

    /// Follow the engine: when it has moved on to the queued next URI (gapless), that is
    /// now the current one.
    fn sync(&self, st: &mut State, status: &Status) {
        let playing = status.track.as_ref().and_then(url_of);
        if st.next_queued && !st.next.uri.is_empty() && playing == Some(st.next.uri.as_str()) {
            st.current = std::mem::take(&mut st.next);
            st.next_queued = false;
        }
    }

    /// The engine has our current URI loaded.
    fn loaded(st: &State, status: &Status) -> bool {
        !st.current.uri.is_empty()
            && status.track.as_ref().and_then(url_of) == Some(st.current.uri.as_str())
            && status.state != PlayState::Stopped
    }

    /// `PLAYING`, `PAUSED_PLAYBACK`, `STOPPED` or `NO_MEDIA_PRESENT`.
    pub fn transport_state(&self) -> &'static str {
        let status = self.engine.status();
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.sync(&mut st, &status);
        if st.current.uri.is_empty() {
            return "NO_MEDIA_PRESENT";
        }
        if !Self::loaded(&st, &status) {
            return "STOPPED";
        }
        match status.state {
            PlayState::Playing => "PLAYING",
            PlayState::Paused => "PAUSED_PLAYBACK",
            PlayState::Stopped => "STOPPED",
        }
    }

    fn tracks(st: &State) -> Vec<Track> {
        let mut v = vec![track_for(&st.current.uri, &st.current.meta)];
        if !st.next.uri.is_empty() {
            v.push(track_for(&st.next.uri, &st.next.meta));
        }
        v
    }

    fn start(&self, st: &mut State) {
        st.next_queued = !st.next.uri.is_empty();
        self.engine.play_queue(Self::tracks(st), 0, 0.0);
    }

    /// Run `action` of service `service` with the SOAP `body`.
    pub fn handle(&self, service: &str, action: &str, body: &str) -> Result<Out, Fault> {
        match service {
            "AVTransport" => self.av_transport(action, body),
            "RenderingControl" => self.rendering_control(action, body),
            "ConnectionManager" => Self::connection_manager(action, body),
            "Product" | "Playlist" | "Volume" | "Time" | "Info" => {
                self.oh.handle(service, action, body)
            }
            _ => Err(Fault::INVALID_ACTION),
        }
    }

    fn av_transport(&self, action: &str, body: &str) -> Result<Out, Fault> {
        check_instance(body)?;
        let status = self.engine.status();
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.sync(&mut st, &status);
        match action {
            "SetAVTransportURI" => {
                let uri = arg(body, "CurrentURI")?.trim().to_string();
                http_only(&uri)?;
                let meta = xml::text(body, "CurrentURIMetaData").unwrap_or_default();
                let was_playing = status.state == PlayState::Playing;
                st.current = Media { uri, meta };
                st.next = Media::default();
                st.next_queued = false;
                self.oh.avtransport_active();
                if was_playing {
                    // Most controllers expect a playing renderer to switch at once.
                    self.start(&mut st);
                } else {
                    self.engine.send(Command::Stop);
                }
                Ok(Vec::new())
            }
            "SetNextAVTransportURI" => {
                let uri = arg(body, "NextURI")?.trim().to_string();
                if !uri.is_empty() {
                    http_only(&uri)?;
                }
                let meta = xml::text(body, "NextURIMetaData").unwrap_or_default();
                let loaded = Self::loaded(&st, &status);
                if loaded && st.next_queued {
                    // Replace the next entry we queued earlier.
                    if let Some(i) = status.index {
                        self.engine.send(Command::Remove(i + 1));
                    }
                    st.next_queued = false;
                }
                st.next = Media { uri, meta };
                if loaded && !st.next.uri.is_empty() {
                    self.engine.send(Command::Enqueue(vec![track_for(
                        &st.next.uri,
                        &st.next.meta,
                    )]));
                    st.next_queued = true;
                }
                Ok(Vec::new())
            }
            "Play" => {
                if st.current.uri.is_empty() {
                    return Err(Fault::TRANSITION_NOT_AVAILABLE);
                }
                if Self::loaded(&st, &status) {
                    self.engine.send(Command::Play);
                } else {
                    self.oh.avtransport_active();
                    self.start(&mut st);
                }
                Ok(Vec::new())
            }
            "Pause" => {
                if !Self::loaded(&st, &status) {
                    return Err(Fault::TRANSITION_NOT_AVAILABLE);
                }
                self.engine.send(Command::Pause);
                Ok(Vec::new())
            }
            "Stop" => {
                self.engine.send(Command::Stop);
                st.next_queued = false;
                Ok(Vec::new())
            }
            "Seek" => {
                let unit = arg(body, "Unit")?;
                let target = arg(body, "Target")?;
                let secs = match unit.trim() {
                    "REL_TIME" | "ABS_TIME" => {
                        xml::parse_time(&target).ok_or(Fault::SEEK_TARGET)?
                    }
                    _ => return Err(Fault::SEEK_MODE),
                };
                if !Self::loaded(&st, &status) {
                    return Err(Fault::TRANSITION_NOT_AVAILABLE);
                }
                self.engine.send(Command::Seek(secs));
                Ok(Vec::new())
            }
            "Next" => {
                if st.next.uri.is_empty() || !Self::loaded(&st, &status) {
                    return Err(Fault::TRANSITION_NOT_AVAILABLE);
                }
                self.engine.send(Command::Next);
                Ok(Vec::new())
            }
            "Previous" => {
                if !Self::loaded(&st, &status) {
                    return Err(Fault::TRANSITION_NOT_AVAILABLE);
                }
                self.engine.send(Command::Seek(0.0));
                Ok(Vec::new())
            }
            "GetTransportInfo" => {
                drop(st);
                Ok(vec![
                    ("CurrentTransportState", self.transport_state().into()),
                    ("CurrentTransportStatus", "OK".into()),
                    ("CurrentSpeed", "1".into()),
                ])
            }
            "GetPositionInfo" => {
                let loaded = Self::loaded(&st, &status);
                let (duration, rel) = if loaded {
                    (
                        status.duration.map(xml::format_time),
                        xml::format_time(status.position),
                    )
                } else {
                    (None, "0:00:00".to_string())
                };
                let meta = self.meta(&st.current);
                Ok(vec![
                    (
                        "Track",
                        if st.current.uri.is_empty() { "0" } else { "1" }.into(),
                    ),
                    (
                        "TrackDuration",
                        duration.unwrap_or_else(|| "0:00:00".into()),
                    ),
                    ("TrackMetaData", meta),
                    ("TrackURI", st.current.uri.clone()),
                    ("RelTime", rel.clone()),
                    ("AbsTime", rel),
                    ("RelCount", "2147483647".into()),
                    ("AbsCount", "2147483647".into()),
                ])
            }
            "GetMediaInfo" => {
                let loaded = Self::loaded(&st, &status);
                Ok(vec![
                    (
                        "NrTracks",
                        if st.current.uri.is_empty() { "0" } else { "1" }.into(),
                    ),
                    (
                        "MediaDuration",
                        status
                            .duration
                            .filter(|_| loaded)
                            .map(xml::format_time)
                            .unwrap_or_else(|| "0:00:00".into()),
                    ),
                    ("CurrentURI", st.current.uri.clone()),
                    ("CurrentURIMetaData", self.meta(&st.current)),
                    ("NextURI", st.next.uri.clone()),
                    ("NextURIMetaData", st.next.meta.clone()),
                    ("PlayMedium", "NETWORK".into()),
                    ("RecordMedium", "NOT_IMPLEMENTED".into()),
                    ("WriteStatus", "NOT_IMPLEMENTED".into()),
                ])
            }
            "GetDeviceCapabilities" => Ok(vec![
                ("PlayMedia", "NETWORK".into()),
                ("RecMedia", "NOT_IMPLEMENTED".into()),
                ("RecQualityModes", "NOT_IMPLEMENTED".into()),
            ]),
            "GetTransportSettings" => Ok(vec![
                ("PlayMode", "NORMAL".into()),
                ("RecQualityMode", "NOT_IMPLEMENTED".into()),
            ]),
            "GetCurrentTransportActions" => {
                drop(st);
                Ok(vec![("Actions", self.actions())])
            }
            _ => Err(Fault::INVALID_ACTION),
        }
    }

    fn meta(&self, m: &Media) -> String {
        if m.uri.is_empty() || !m.meta.trim().is_empty() {
            m.meta.clone()
        } else {
            didl_for(&track_for(&m.uri, ""), &m.uri)
        }
    }

    fn actions(&self) -> String {
        let has_next = !self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .next
            .uri
            .is_empty();
        let mut a: Vec<&str> = match self.transport_state() {
            "PLAYING" => vec!["Pause", "Stop", "Seek"],
            "PAUSED_PLAYBACK" => vec!["Play", "Stop", "Seek"],
            "STOPPED" => vec!["Play"],
            _ => vec![],
        };
        if has_next && !a.is_empty() {
            a.push("Next");
        }
        a.join(",")
    }

    fn volume_percent(&self) -> u32 {
        (self.engine.status().volume_linear * 100.0)
            .round()
            .clamp(0.0, 100.0) as u32
    }

    fn rendering_control(&self, action: &str, body: &str) -> Result<Out, Fault> {
        check_instance(body)?;
        let status = self.engine.status();
        match action {
            "GetVolume" => Ok(vec![("CurrentVolume", self.volume_percent().to_string())]),
            "SetVolume" => {
                let v: f64 = arg(body, "DesiredVolume")?
                    .trim()
                    .parse()
                    .map_err(|_| Fault::INVALID_ARGS)?;
                if !(0.0..=100.0).contains(&v) {
                    return Err(Fault::INVALID_ARGS);
                }
                let mut vol = Volume::from_linear(v / 100.0);
                vol.muted = status.volume.muted;
                self.engine.send(Command::SetVolume(vol));
                Ok(Vec::new())
            }
            "GetMute" => Ok(vec![(
                "CurrentMute",
                if status.volume.muted { "1" } else { "0" }.into(),
            )]),
            "SetMute" => {
                let m = match arg(body, "DesiredMute")?.trim() {
                    "1" | "true" | "yes" => true,
                    "0" | "false" | "no" => false,
                    _ => return Err(Fault::INVALID_ARGS),
                };
                self.engine.send(Command::SetVolume(Volume {
                    muted: m,
                    ..status.volume
                }));
                Ok(Vec::new())
            }
            "ListPresets" => Ok(vec![("CurrentPresetNameList", "FactoryDefaults".into())]),
            "SelectPreset" => Ok(Vec::new()),
            _ => Err(Fault::INVALID_ACTION),
        }
    }

    fn connection_manager(action: &str, body: &str) -> Result<Out, Fault> {
        match action {
            "GetProtocolInfo" => Ok(vec![
                ("Source", String::new()),
                ("Sink", description::sink_protocol_info()),
            ]),
            "GetCurrentConnectionIDs" => Ok(vec![("ConnectionIDs", "0".into())]),
            "GetCurrentConnectionInfo" => {
                if arg(body, "ConnectionID")?.trim() != "0" {
                    return Err(Fault::INVALID_CONNECTION);
                }
                Ok(vec![
                    ("RcsID", "0".into()),
                    ("AVTransportID", "0".into()),
                    ("ProtocolInfo", String::new()),
                    ("PeerConnectionManager", String::new()),
                    ("PeerConnectionID", "-1".into()),
                    ("Direction", "Input".into()),
                    ("Status", "OK".into()),
                ])
            }
            _ => Err(Fault::INVALID_ACTION),
        }
    }

    /// The `LastChange` value of `service` (the evented state, as an escaped XML
    /// document), or `None` for services without one.
    pub fn last_change(&self, service: &str) -> Option<String> {
        let inner = match service {
            "AVTransport" => {
                let state = self.transport_state();
                let actions = self.actions();
                let status = self.engine.status();
                let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                let loaded = Self::loaded(&st, &status);
                let duration = status
                    .duration
                    .filter(|_| loaded)
                    .map(xml::format_time)
                    .unwrap_or_else(|| "0:00:00".into());
                let meta = self.meta(&st.current);
                format!(
                    r#"<Event xmlns="urn:schemas-upnp-org:metadata-1-0/AVT/"><InstanceID val="0"><TransportState val="{state}"/><TransportStatus val="OK"/><TransportPlaySpeed val="1"/><CurrentPlayMode val="NORMAL"/><NumberOfTracks val="{n}"/><CurrentTrack val="{n}"/><CurrentTrackDuration val="{duration}"/><CurrentMediaDuration val="{duration}"/><AVTransportURI val="{uri}"/><CurrentTrackURI val="{uri}"/><AVTransportURIMetaData val="{meta}"/><CurrentTrackMetaData val="{meta}"/><NextAVTransportURI val="{next}"/><NextAVTransportURIMetaData val="{next_meta}"/><CurrentTransportActions val="{actions}"/></InstanceID></Event>"#,
                    n = if st.current.uri.is_empty() { 0 } else { 1 },
                    uri = escape(&st.current.uri),
                    meta = escape(&meta),
                    next = escape(&st.next.uri),
                    next_meta = escape(&st.next.meta),
                )
            }
            "RenderingControl" => {
                let muted = self.engine.status().volume.muted;
                format!(
                    r#"<Event xmlns="urn:schemas-upnp-org:metadata-1-0/RCS/"><InstanceID val="0"><Volume channel="Master" val="{}"/><Mute channel="Master" val="{}"/><PresetNameList val="FactoryDefaults"/></InstanceID></Event>"#,
                    self.volume_percent(),
                    muted as u8
                )
            }
            _ => return None,
        };
        Some(inner)
    }

    /// The evented variables of `service` as `(name, value)` (the GENA property set).
    pub fn evented(&self, service: &str) -> Vec<(&'static str, String)> {
        match service {
            "AVTransport" | "RenderingControl" => self
                .last_change(service)
                .map(|v| vec![("LastChange", v)])
                .unwrap_or_default(),
            "Product" | "Playlist" | "Volume" | "Time" | "Info" => self.oh.evented(service),
            "ConnectionManager" => vec![
                ("SourceProtocolInfo", String::new()),
                ("SinkProtocolInfo", description::sink_protocol_info()),
                ("CurrentConnectionIDs", "0".into()),
            ],
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn didl_metadata_becomes_a_track() {
        let meta = r#"<DIDL-Lite><item><dc:title>Blue in Green</dc:title><upnp:artist>Miles Davis</upnp:artist><upnp:album>Kind of Blue</upnp:album><upnp:albumArtURI>http://s/a.jpg</upnp:albumArtURI><res duration="0:05:37">http://s/t.flac</res></item></DIDL-Lite>"#;
        let t = track_for("http://s/t.flac", meta);
        assert_eq!(t.title, "Blue in Green");
        assert_eq!(t.artist, "Miles Davis");
        assert_eq!(t.album, "Kind of Blue");
        assert_eq!(t.cover.as_deref(), Some("http://s/a.jpg"));
        assert_eq!(t.duration, Some(337.0));
        let bare = track_for("http://s/music/01%20Song.flac?token=1", "");
        assert_eq!(bare.title, "01%20Song.flac");
        let didl = didl_for(&t, "http://s/t.flac");
        assert_eq!(track_for("http://s/t.flac", &didl).title, "Blue in Green");
        assert!(http_only("file:///etc/passwd").is_err());
        assert!(http_only("/home/user/x.flac").is_err());
        assert!(http_only(" HTTPS://x/y").is_ok());
    }
}
