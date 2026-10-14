//! The device description and the three service descriptions (SCPD) of a UPnP AV
//! MediaRenderer:1 – AVTransport:1, RenderingControl:1 and ConnectionManager:1.

use super::xml::escape;

/// The device type controllers search for.
pub const DEVICE_TYPE: &str = "urn:schemas-upnp-org:device:MediaRenderer:1";

/// One service: short id (used in our URLs), service type and service id.
pub struct Service {
    /// `AVTransport`, `RenderingControl`, `ConnectionManager`.
    pub name: &'static str,
    /// `urn:schemas-upnp-org:service:AVTransport:1`.
    pub service_type: &'static str,
    /// `urn:upnp-org:serviceId:AVTransport`.
    pub service_id: &'static str,
    /// The SCPD document.
    pub scpd: &'static str,
}

/// The services, in description order.
pub const SERVICES: [Service; 3] = [
    Service {
        name: "AVTransport",
        service_type: "urn:schemas-upnp-org:service:AVTransport:1",
        service_id: "urn:upnp-org:serviceId:AVTransport",
        scpd: AV_TRANSPORT_SCPD,
    },
    Service {
        name: "RenderingControl",
        service_type: "urn:schemas-upnp-org:service:RenderingControl:1",
        service_id: "urn:upnp-org:serviceId:RenderingControl",
        scpd: RENDERING_CONTROL_SCPD,
    },
    Service {
        name: "ConnectionManager",
        service_type: "urn:schemas-upnp-org:service:ConnectionManager:1",
        service_id: "urn:upnp-org:serviceId:ConnectionManager",
        scpd: CONNECTION_MANAGER_SCPD,
    },
];

/// The service with short name `name`.
pub fn service(name: &str) -> Option<&'static Service> {
    SERVICES.iter().find(|s| s.name == name)
}

/// The root device description.
pub fn device_xml(friendly_name: &str, udn: &str) -> String {
    let services: String = SERVICES
        .iter()
        .map(|s| {
            format!(
                "<service><serviceType>{}</serviceType><serviceId>{}</serviceId>\
<SCPDURL>/upnp/{n}.xml</SCPDURL><controlURL>/upnp/control/{n}</controlURL>\
<eventSubURL>/upnp/event/{n}</eventSubURL></service>",
                s.service_type,
                s.service_id,
                n = s.name
            )
        })
        .collect();
    format!(
        r#"<?xml version="1.0" encoding="utf-8"?>
<root xmlns="urn:schemas-upnp-org:device-1-0" xmlns:dlna="urn:schemas-dlna-org:device-1-0">
<specVersion><major>1</major><minor>0</minor></specVersion>
<device>
<deviceType>{DEVICE_TYPE}</deviceType>
<friendlyName>{}</friendlyName>
<manufacturer>Cadenza</manufacturer>
<manufacturerURL>https://github.com/Krifmaz/cadenza</manufacturerURL>
<modelDescription>Cadenza upsampling renderer</modelDescription>
<modelName>Cadenza</modelName>
<modelNumber>{}</modelNumber>
<UDN>{}</UDN>
<dlna:X_DLNADOC>DMR-1.50</dlna:X_DLNADOC>
<serviceList>{services}</serviceList>
</device>
</root>
"#,
        escape(friendly_name),
        env!("CARGO_PKG_VERSION"),
        escape(udn),
    )
}

macro_rules! scpd {
    ($actions:expr, $vars:expr) => {
        concat!(
            r#"<?xml version="1.0" encoding="utf-8"?>
<scpd xmlns="urn:schemas-upnp-org:service-1-0"><specVersion><major>1</major><minor>0</minor></specVersion>
<actionList>"#,
            $actions,
            "</actionList><serviceStateTable>",
            $vars,
            "</serviceStateTable></scpd>\n"
        )
    };
}

macro_rules! action {
    ($name:literal $(, $dir:literal $arg:literal $var:literal)*) => {
        concat!(
            "<action><name>", $name, "</name><argumentList>",
            $("<argument><name>", $arg, "</name><direction>", $dir,
              "</direction><relatedStateVariable>", $var, "</relatedStateVariable></argument>",)*
            "</argumentList></action>"
        )
    };
}

macro_rules! var {
    ($name:literal, $ty:literal) => {
        concat!(
            r#"<stateVariable sendEvents="no"><name>"#,
            $name,
            "</name><dataType>",
            $ty,
            "</dataType></stateVariable>"
        )
    };
    (events $name:literal, $ty:literal) => {
        concat!(
            r#"<stateVariable sendEvents="yes"><name>"#,
            $name,
            "</name><dataType>",
            $ty,
            "</dataType></stateVariable>"
        )
    };
}

/// AVTransport:1.
pub const AV_TRANSPORT_SCPD: &str = scpd!(
    concat!(
        action!("SetAVTransportURI", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "in" "CurrentURI" "AVTransportURI", "in" "CurrentURIMetaData" "AVTransportURIMetaData"),
        action!("SetNextAVTransportURI", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "in" "NextURI" "NextAVTransportURI", "in" "NextURIMetaData" "NextAVTransportURIMetaData"),
        action!("GetMediaInfo", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "out" "NrTracks" "NumberOfTracks", "out" "MediaDuration" "CurrentMediaDuration",
            "out" "CurrentURI" "AVTransportURI", "out" "CurrentURIMetaData" "AVTransportURIMetaData",
            "out" "NextURI" "NextAVTransportURI", "out" "NextURIMetaData" "NextAVTransportURIMetaData",
            "out" "PlayMedium" "PlaybackStorageMedium", "out" "RecordMedium" "RecordStorageMedium",
            "out" "WriteStatus" "RecordMediumWriteStatus"),
        action!("GetTransportInfo", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "out" "CurrentTransportState" "TransportState",
            "out" "CurrentTransportStatus" "TransportStatus", "out" "CurrentSpeed" "TransportPlaySpeed"),
        action!("GetPositionInfo", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "out" "Track" "CurrentTrack", "out" "TrackDuration" "CurrentTrackDuration",
            "out" "TrackMetaData" "CurrentTrackMetaData", "out" "TrackURI" "CurrentTrackURI",
            "out" "RelTime" "RelativeTimePosition", "out" "AbsTime" "AbsoluteTimePosition",
            "out" "RelCount" "RelativeCounterPosition", "out" "AbsCount" "AbsoluteCounterPosition"),
        action!("GetDeviceCapabilities", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "out" "PlayMedia" "PossiblePlaybackStorageMedia", "out" "RecMedia" "PossibleRecordStorageMedia",
            "out" "RecQualityModes" "PossibleRecordQualityModes"),
        action!("GetTransportSettings", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "out" "PlayMode" "CurrentPlayMode", "out" "RecQualityMode" "CurrentRecordQualityMode"),
        action!("GetCurrentTransportActions", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "out" "Actions" "CurrentTransportActions"),
        action!("Stop", "in" "InstanceID" "A_ARG_TYPE_InstanceID"),
        action!("Play", "in" "InstanceID" "A_ARG_TYPE_InstanceID", "in" "Speed" "TransportPlaySpeed"),
        action!("Pause", "in" "InstanceID" "A_ARG_TYPE_InstanceID"),
        action!("Seek", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "in" "Unit" "A_ARG_TYPE_SeekMode", "in" "Target" "A_ARG_TYPE_SeekTarget"),
        action!("Next", "in" "InstanceID" "A_ARG_TYPE_InstanceID"),
        action!("Previous", "in" "InstanceID" "A_ARG_TYPE_InstanceID")
    ),
    concat!(
        var!(events "LastChange", "string"),
        var!("TransportState", "string"),
        var!("TransportStatus", "string"),
        var!("TransportPlaySpeed", "string"),
        var!("PlaybackStorageMedium", "string"),
        var!("RecordStorageMedium", "string"),
        var!("PossiblePlaybackStorageMedia", "string"),
        var!("PossibleRecordStorageMedia", "string"),
        var!("PossibleRecordQualityModes", "string"),
        var!("RecordMediumWriteStatus", "string"),
        var!("CurrentPlayMode", "string"),
        var!("CurrentRecordQualityMode", "string"),
        var!("NumberOfTracks", "ui4"),
        var!("CurrentTrack", "ui4"),
        var!("CurrentTrackDuration", "string"),
        var!("CurrentMediaDuration", "string"),
        var!("CurrentTrackMetaData", "string"),
        var!("CurrentTrackURI", "string"),
        var!("AVTransportURI", "string"),
        var!("AVTransportURIMetaData", "string"),
        var!("NextAVTransportURI", "string"),
        var!("NextAVTransportURIMetaData", "string"),
        var!("RelativeTimePosition", "string"),
        var!("AbsoluteTimePosition", "string"),
        var!("RelativeCounterPosition", "i4"),
        var!("AbsoluteCounterPosition", "i4"),
        var!("CurrentTransportActions", "string"),
        var!("A_ARG_TYPE_SeekMode", "string"),
        var!("A_ARG_TYPE_SeekTarget", "string"),
        var!("A_ARG_TYPE_InstanceID", "ui4")
    )
);

/// RenderingControl:1 (master channel volume and mute).
pub const RENDERING_CONTROL_SCPD: &str = scpd!(
    concat!(
        action!("ListPresets", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "out" "CurrentPresetNameList" "PresetNameList"),
        action!("SelectPreset", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "in" "PresetName" "A_ARG_TYPE_PresetName"),
        action!("GetVolume", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "in" "Channel" "A_ARG_TYPE_Channel", "out" "CurrentVolume" "Volume"),
        action!("SetVolume", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "in" "Channel" "A_ARG_TYPE_Channel", "in" "DesiredVolume" "Volume"),
        action!("GetMute", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "in" "Channel" "A_ARG_TYPE_Channel", "out" "CurrentMute" "Mute"),
        action!("SetMute", "in" "InstanceID" "A_ARG_TYPE_InstanceID",
            "in" "Channel" "A_ARG_TYPE_Channel", "in" "DesiredMute" "Mute")
    ),
    concat!(
        var!(events "LastChange", "string"),
        var!("PresetNameList", "string"),
        r#"<stateVariable sendEvents="no"><name>Volume</name><dataType>ui2</dataType><allowedValueRange><minimum>0</minimum><maximum>100</maximum><step>1</step></allowedValueRange></stateVariable>"#,
        var!("Mute", "boolean"),
        var!("A_ARG_TYPE_Channel", "string"),
        var!("A_ARG_TYPE_PresetName", "string"),
        var!("A_ARG_TYPE_InstanceID", "ui4")
    )
);

/// ConnectionManager:1.
pub const CONNECTION_MANAGER_SCPD: &str = scpd!(
    concat!(
        action!("GetProtocolInfo", "out" "Source" "SourceProtocolInfo",
            "out" "Sink" "SinkProtocolInfo"),
        action!("GetCurrentConnectionIDs", "out" "ConnectionIDs" "CurrentConnectionIDs"),
        action!("GetCurrentConnectionInfo", "in" "ConnectionID" "A_ARG_TYPE_ConnectionID",
            "out" "RcsID" "A_ARG_TYPE_RcsID", "out" "AVTransportID" "A_ARG_TYPE_AVTransportID",
            "out" "ProtocolInfo" "A_ARG_TYPE_ProtocolInfo",
            "out" "PeerConnectionManager" "A_ARG_TYPE_ConnectionManager",
            "out" "PeerConnectionID" "A_ARG_TYPE_ConnectionID",
            "out" "Direction" "A_ARG_TYPE_Direction", "out" "Status" "A_ARG_TYPE_ConnectionStatus")
    ),
    concat!(
        var!(events "SourceProtocolInfo", "string"),
        var!(events "SinkProtocolInfo", "string"),
        var!(events "CurrentConnectionIDs", "string"),
        var!("A_ARG_TYPE_ConnectionStatus", "string"),
        var!("A_ARG_TYPE_ConnectionManager", "string"),
        var!("A_ARG_TYPE_Direction", "string"),
        var!("A_ARG_TYPE_ProtocolInfo", "string"),
        var!("A_ARG_TYPE_ConnectionID", "i4"),
        var!("A_ARG_TYPE_AVTransportID", "i4"),
        var!("A_ARG_TYPE_RcsID", "i4")
    )
);

/// What the renderer can play (`SinkProtocolInfo`): every format the decoder reads, over
/// HTTP.
pub const SINK_PROTOCOLS: &[&str] = &[
    "audio/flac",
    "audio/x-flac",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/L16",
    "audio/L24",
    "audio/aiff",
    "audio/x-aiff",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
    "audio/ogg",
    "audio/x-ogg",
    "audio/vorbis",
    "audio/opus",
    "audio/x-dsf",
    "audio/dsf",
    "audio/x-dff",
    "audio/dff",
    "audio/x-alac",
];

/// The `SinkProtocolInfo` string.
pub fn sink_protocol_info() -> String {
    SINK_PROTOCOLS
        .iter()
        .map(|m| format!("http-get:*:{m}:*"))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upnp::xml;

    #[test]
    fn descriptions_are_well_formed_enough_and_complete() {
        let d = device_xml("Living <room>", "uuid:1234");
        assert_eq!(
            xml::text(&d, "friendlyName").as_deref(),
            Some("Living <room>")
        );
        assert_eq!(xml::text(&d, "UDN").as_deref(), Some("uuid:1234"));
        assert_eq!(xml::text(&d, "deviceType").as_deref(), Some(DEVICE_TYPE));
        for s in &SERVICES {
            assert!(d.contains(&format!(
                "<controlURL>/upnp/control/{}</controlURL>",
                s.name
            )));
            // Balanced tags (a rough well-formedness check).
            for tag in ["action", "argument", "stateVariable", "argumentList"] {
                assert_eq!(
                    s.scpd.matches(&format!("<{tag}>")).count()
                        + s.scpd.matches(&format!("<{tag} ")).count(),
                    s.scpd.matches(&format!("</{tag}>")).count(),
                    "{} {tag}",
                    s.name
                );
            }
            // Every related state variable is declared.
            let mut rest = s.scpd;
            while let Some(i) = rest.find("<relatedStateVariable>") {
                rest = &rest[i + 22..];
                let end = rest.find('<').unwrap();
                let var = &rest[..end];
                assert!(
                    s.scpd.contains(&format!("<name>{var}</name><dataType>")),
                    "{}: {var} not declared",
                    s.name
                );
            }
        }
        assert!(sink_protocol_info().starts_with("http-get:*:audio/flac:*,"));
        assert!(service("AVTransport").is_some() && service("Nope").is_none());
    }
}
