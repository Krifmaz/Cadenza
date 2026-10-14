//! The renderer's HTTP side: descriptions, SOAP control and GENA event subscriptions.
//!
//! Events: a subscriber gets the full evented state right after subscribing (`SEQ 0`),
//! then a `NOTIFY` whenever it changes (checked twice a second). Subscriptions expire
//! after their timeout unless renewed.

use super::description::{self, SERVICES};
use super::renderer::{Fault, Out, Renderer};
use super::xml::escape;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Longest subscription we grant.
const MAX_TIMEOUT: u64 = 1800;

/// One event subscription.
#[derive(Clone, Debug)]
struct Subscription {
    service: &'static str,
    callbacks: Vec<String>,
    seq: u32,
    expires: Instant,
}

/// Shared state of the HTTP handlers.
pub struct Shared {
    /// The renderer.
    pub renderer: Renderer,
    /// Friendly name.
    pub name: String,
    /// `uuid:…`.
    pub udn: String,
    subs: Mutex<HashMap<String, Subscription>>,
    /// The last evented values sent per service.
    sent: Mutex<HashMap<&'static str, Vec<(&'static str, String)>>>,
}

impl Shared {
    /// State for `renderer`.
    pub fn new(renderer: Renderer, name: String, udn: String) -> Arc<Self> {
        Arc::new(Self {
            renderer,
            name,
            udn,
            subs: Mutex::new(HashMap::new()),
            sent: Mutex::new(HashMap::new()),
        })
    }
}

/// The router (`/upnp/…`).
pub fn router(shared: Arc<Shared>) -> Router {
    Router::new()
        .route("/upnp/device.xml", get(device))
        .route("/upnp/{file}", get(scpd))
        .route("/upnp/control/{service}", post(control))
        .route("/upnp/event/{service}", any(event))
        .with_state(shared)
}

fn xml_response(body: String) -> Response {
    ([("content-type", r#"text/xml; charset="utf-8""#)], body).into_response()
}

async fn device(State(s): State<Arc<Shared>>) -> Response {
    xml_response(description::device_xml(&s.name, &s.udn))
}

async fn scpd(Path(file): Path<String>) -> Response {
    match file.strip_suffix(".xml").and_then(description::service) {
        Some(svc) => xml_response((svc.scpd)()),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// A SOAP envelope around `inner`.
pub fn envelope(inner: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body>{inner}</s:Body></s:Envelope>
"#
    )
}

/// The response to `action` of `service_type`.
pub fn action_response(service_type: &str, action: &str, out: &Out) -> String {
    let args: String = out
        .iter()
        .map(|(k, v)| format!("<{k}>{}</{k}>", escape(v)))
        .collect();
    envelope(&format!(
        r#"<u:{action}Response xmlns:u="{service_type}">{args}</u:{action}Response>"#
    ))
}

/// A SOAP fault for `fault`.
pub fn fault_response(fault: &Fault) -> String {
    envelope(&format!(
        r#"<s:Fault><faultcode>s:Client</faultcode><faultstring>UPnPError</faultstring><detail><UPnPError xmlns="urn:schemas-upnp-org:control-1-0"><errorCode>{}</errorCode><errorDescription>{}</errorDescription></UPnPError></detail></s:Fault>"#,
        fault.code, fault.description
    ))
}

/// The action name from a `SOAPACTION: "urn:…:service:X:1#Action"` header.
pub fn soap_action(headers: &HeaderMap) -> Option<String> {
    let v = headers.get("soapaction")?.to_str().ok()?;
    let v = v.trim().trim_matches('"');
    v.rsplit_once('#').map(|(_, a)| a.to_string())
}

async fn control(
    State(s): State<Arc<Shared>>,
    Path(service): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(svc) = description::service(&service) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let body = String::from_utf8_lossy(&body);
    let result = match soap_action(&headers) {
        Some(action) => s
            .renderer
            .handle(svc.name, &action, &body)
            .map(|out| action_response(svc.service_type, &action, &out)),
        None => Err(Fault::INVALID_ACTION),
    };
    match result {
        Ok(xml) => xml_response(xml),
        Err(f) => {
            log::debug!("UPnP {service}: fault {} {}", f.code, f.description);
            let mut r = xml_response(fault_response(&f));
            *r.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            r
        }
    }
}

fn timeout_of(headers: &HeaderMap) -> u64 {
    headers
        .get("timeout")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().strip_prefix("Second-"))
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(MAX_TIMEOUT)
        .clamp(60, MAX_TIMEOUT)
}

/// `<url1><url2>` → the URLs.
fn callbacks_of(headers: &HeaderMap) -> Vec<String> {
    let Some(v) = headers.get("callback").and_then(|v| v.to_str().ok()) else {
        return Vec::new();
    };
    v.split('>')
        .filter_map(|p| p.trim().strip_prefix('<'))
        .filter(|u| u.starts_with("http://"))
        .map(str::to_string)
        .collect()
}

fn new_sid() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(1);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let h = super::fnv(&[&t.to_le_bytes()[..], &n.to_le_bytes()[..]]);
    format!("uuid:{}", super::uuid_from(h, n))
}

async fn event(
    State(s): State<Arc<Shared>>,
    Path(service): Path<String>,
    method: Method,
    headers: HeaderMap,
) -> Response {
    let Some(svc) = description::service(&service) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let sid = headers
        .get("sid")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string());
    match method.as_str() {
        "SUBSCRIBE" => {
            let timeout = timeout_of(&headers);
            let mut subs = s.subs.lock().unwrap_or_else(|e| e.into_inner());
            let sid = match sid {
                // Renewal.
                Some(sid) => match subs.get_mut(&sid) {
                    Some(sub) => {
                        sub.expires = Instant::now() + Duration::from_secs(timeout);
                        sid
                    }
                    None => return StatusCode::PRECONDITION_FAILED.into_response(),
                },
                None => {
                    let callbacks = callbacks_of(&headers);
                    if callbacks.is_empty() {
                        return StatusCode::PRECONDITION_FAILED.into_response();
                    }
                    let sid = new_sid();
                    subs.insert(
                        sid.clone(),
                        Subscription {
                            service: svc.name,
                            callbacks,
                            seq: 0,
                            expires: Instant::now() + Duration::from_secs(timeout),
                        },
                    );
                    // The initial event follows the response.
                    let (s2, sid2) = (Arc::clone(&s), sid.clone());
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(100));
                        initial_event(&s2, &sid2);
                    });
                    sid
                }
            };
            (
                StatusCode::OK,
                [
                    ("sid", sid),
                    ("timeout", format!("Second-{timeout}")),
                    (
                        "server",
                        format!(
                            "{}/1.0 UPnP/1.0 Cadenza/{}",
                            std::env::consts::OS,
                            env!("CARGO_PKG_VERSION")
                        ),
                    ),
                ],
            )
                .into_response()
        }
        "UNSUBSCRIBE" => {
            let Some(sid) = sid else {
                return StatusCode::PRECONDITION_FAILED.into_response();
            };
            match s
                .subs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&sid)
            {
                Some(_) => StatusCode::OK.into_response(),
                None => StatusCode::PRECONDITION_FAILED.into_response(),
            }
        }
        _ => StatusCode::METHOD_NOT_ALLOWED.into_response(),
    }
}

/// A GENA property set.
pub fn propertyset(vars: &[(&'static str, String)]) -> String {
    let props: String = vars
        .iter()
        .map(|(k, v)| format!("<e:property><{k}>{}</{k}></e:property>", escape(v)))
        .collect();
    format!(
        r#"<?xml version="1.0" encoding="utf-8"?><e:propertyset xmlns:e="urn:schemas-upnp-org:event-1-0">{props}</e:propertyset>"#
    )
}

fn send_notify(callbacks: &[String], sid: &str, seq: u32, body: &str) -> bool {
    for url in callbacks {
        let r = ureq::request("NOTIFY", url)
            .timeout(Duration::from_secs(3))
            .set("Content-Type", r#"text/xml; charset="utf-8""#)
            .set("NT", "upnp:event")
            .set("NTS", "upnp:propchange")
            .set("SID", sid)
            .set("SEQ", &seq.to_string())
            .send_string(body);
        if r.is_ok() {
            return true;
        }
    }
    false
}

fn initial_event(s: &Arc<Shared>, sid: &str) {
    let Some(sub) = s
        .subs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(sid)
        .cloned()
    else {
        return;
    };
    let body = propertyset(&s.renderer.evented(sub.service));
    send_notify(&sub.callbacks, sid, 0, &body);
    if let Some(sub) = s
        .subs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_mut(sid)
    {
        sub.seq = sub.seq.max(1);
    }
}

/// Check for state changes and notify the subscribers; drop expired subscriptions.
pub fn tick(s: &Arc<Shared>) {
    s.renderer.tick();
    let now = Instant::now();
    s.subs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|_, sub| sub.expires > now);
    for svc in SERVICES.iter().map(|x| x.name) {
        let vars = s.renderer.evented(svc);
        let changed = {
            let mut sent = s.sent.lock().unwrap_or_else(|e| e.into_inner());
            let changed = sent.get(svc) != Some(&vars);
            if changed {
                sent.insert(svc, vars.clone());
            }
            changed
        };
        if !changed {
            continue;
        }
        let targets: Vec<(String, Vec<String>, u32)> = {
            let mut subs = s.subs.lock().unwrap_or_else(|e| e.into_inner());
            subs.iter_mut()
                .filter(|(_, sub)| sub.service == svc && sub.seq > 0)
                .map(|(sid, sub)| {
                    let seq = sub.seq;
                    sub.seq = sub.seq.checked_add(1).unwrap_or(1);
                    (sid.clone(), sub.callbacks.clone(), seq)
                })
                .collect()
        };
        let body = propertyset(&vars);
        for (sid, callbacks, seq) in targets {
            send_notify(&callbacks, &sid, seq, &body);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upnp::xml;

    #[test]
    fn soap_envelopes_and_headers() {
        let mut h = HeaderMap::new();
        h.insert(
            "soapaction",
            r#""urn:schemas-upnp-org:service:AVTransport:1#Play""#
                .parse()
                .unwrap(),
        );
        assert_eq!(soap_action(&h).as_deref(), Some("Play"));
        let r = action_response(
            "urn:schemas-upnp-org:service:AVTransport:1",
            "GetTransportInfo",
            &vec![("CurrentTransportState", "PLAYING".into())],
        );
        assert!(r.contains(
            "<u:GetTransportInfoResponse xmlns:u=\"urn:schemas-upnp-org:service:AVTransport:1\">"
        ));
        assert_eq!(
            xml::text(&r, "CurrentTransportState").as_deref(),
            Some("PLAYING")
        );
        let f = fault_response(&Fault::INVALID_ARGS);
        assert_eq!(xml::text(&f, "errorCode").as_deref(), Some("402"));
        let mut h = HeaderMap::new();
        h.insert(
            "callback",
            "<http://10.0.0.2:4000/ev> <ftp://x/>".parse().unwrap(),
        );
        h.insert("timeout", "Second-60".parse().unwrap());
        assert_eq!(callbacks_of(&h), vec!["http://10.0.0.2:4000/ev"]);
        assert_eq!(timeout_of(&h), 60);
        assert_eq!(timeout_of(&HeaderMap::new()), MAX_TIMEOUT);
        let p = propertyset(&[("LastChange", "<Event a=\"1\"/>".into())]);
        assert_eq!(
            xml::text(&p, "LastChange").as_deref(),
            Some("<Event a=\"1\"/>")
        );
        assert_ne!(new_sid(), new_sid());
    }
}
