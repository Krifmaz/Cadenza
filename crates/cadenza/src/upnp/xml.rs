//! Just enough XML for UPnP: escaping, reading the text of an element by its local name
//! (namespace prefixes ignored), attributes of the first such element, and the time
//! format of AVTransport (`H:MM:SS`).
//!
//! SOAP requests and DIDL-Lite metadata are small, flat documents; a full parser would
//! buy nothing here, and these helpers never panic on malformed input – they return
//! `None`.

/// Escape text for an XML element or attribute.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

/// Resolve the five predefined entities and numeric character references.
pub fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let Some(end) = rest.find(';').filter(|&e| e <= 10) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let ent = &rest[1..end];
        let c = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                u32::from_str_radix(&ent[2..], 16)
                    .ok()
                    .and_then(char::from_u32)
            }
            _ if ent.starts_with('#') => ent[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match c {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Find the first start tag whose local name is `name`: returns `(attributes text, the
/// index just past the start tag, self-closing)`.
fn find_start<'a>(doc: &'a str, name: &str, from: usize) -> Option<(&'a str, usize, bool)> {
    let mut at = from;
    while let Some(off) = doc.get(at..)?.find('<') {
        let start = at + off + 1;
        let tail = &doc[start..];
        let end = tail.find('>')?;
        let tag = &tail[..end];
        at = start + end + 1;
        if tag.starts_with('/') || tag.starts_with('?') || tag.starts_with('!') {
            continue;
        }
        let tag_name = tag
            .split(|c: char| c.is_whitespace() || c == '/')
            .next()
            .unwrap_or("");
        let local = tag_name.rsplit(':').next().unwrap_or(tag_name);
        if local == name {
            let attrs = tag[tag_name.len()..].trim_end_matches('/');
            return Some((attrs, at, tag.ends_with('/')));
        }
    }
    None
}

/// The unescaped text of the first element with local name `name` (`""` for an empty
/// element), or `None` when there is none.
pub fn text(doc: &str, name: &str) -> Option<String> {
    let (_, body, closed) = find_start(doc, name, 0)?;
    if closed {
        return Some(String::new());
    }
    // The matching end tag: the next `</prefix:name>` or `</name>`.
    let mut at = body;
    loop {
        let off = doc.get(at..)?.find("</")?;
        let start = at + off + 2;
        let end = doc[start..].find('>')? + start;
        let tag = doc[start..end].trim();
        if tag.rsplit(':').next() == Some(name) {
            return Some(unescape(&doc[body..at + off]));
        }
        at = end + 1;
    }
}

/// The unescaped value of attribute `attr` on the first element with local name `name`.
pub fn attr(doc: &str, name: &str, attr: &str) -> Option<String> {
    let (attrs, _, _) = find_start(doc, name, 0)?;
    let mut rest = attrs;
    while let Some(eq) = rest.find('=') {
        let key = rest[..eq].trim();
        let after = rest[eq + 1..].trim_start();
        let quote = after.chars().next()?;
        if quote != '"' && quote != '\'' {
            return None;
        }
        let close = after[1..].find(quote)? + 1;
        let value = &after[1..close];
        if key.rsplit(':').next() == Some(attr) {
            return Some(unescape(value));
        }
        rest = &after[close + 1..];
    }
    None
}

/// `H:MM:SS` (AVTransport's time format; hours unpadded).
pub fn format_time(seconds: f64) -> String {
    let s = seconds.max(0.0).floor() as u64;
    format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// Parse `H+:MM:SS[.F+]` or `H+:MM:SS[.F0/F1]` into seconds.
pub fn parse_time(t: &str) -> Option<f64> {
    let t = t.trim().trim_start_matches('+');
    let mut parts = t.split(':');
    let h: f64 = parts.next()?.parse().ok()?;
    let m: f64 = parts.next()?.parse().ok()?;
    let s_part = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let s: f64 = match s_part.split_once('.') {
        Some((whole, frac)) if frac.contains('/') => {
            let (a, b) = frac.split_once('/')?;
            let (a, b): (f64, f64) = (a.parse().ok()?, b.parse().ok()?);
            whole.parse::<f64>().ok()? + if b > 0.0 { a / b } else { 0.0 }
        }
        _ => s_part.parse().ok()?,
    };
    (h >= 0.0 && (0.0..60.0).contains(&m) && (0.0..60.0).contains(&s))
        .then_some(h * 3600.0 + m * 60.0 + s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_round_trip() {
        let s = r#"Tom & Jerry <"live"> 'it's'"#;
        assert_eq!(unescape(&escape(s)), s);
        assert_eq!(unescape("&#65;&#x42;&amp;&bogus;&"), "AB&&bogus;&");
    }

    #[test]
    fn elements_and_attributes_by_local_name() {
        let doc = r#"<?xml version="1.0"?><s:Envelope><s:Body><u:SetAVTransportURI xmlns:u="x">
            <InstanceID>0</InstanceID><CurrentURI>http://h/a.flac?x=1&amp;y=2</CurrentURI>
            <CurrentURIMetaData/></u:SetAVTransportURI></s:Body></s:Envelope>"#;
        assert_eq!(text(doc, "InstanceID").as_deref(), Some("0"));
        assert_eq!(
            text(doc, "CurrentURI").as_deref(),
            Some("http://h/a.flac?x=1&y=2")
        );
        assert_eq!(text(doc, "CurrentURIMetaData").as_deref(), Some(""));
        assert_eq!(text(doc, "Missing"), None);
        let didl = r#"<DIDL-Lite><item id="1"><dc:title>Song &amp; Dance</dc:title>
            <res protocolInfo="http-get:*:audio/flac:*" duration="0:03:21.500">http://h/s.flac</res>
            </item></DIDL-Lite>"#;
        assert_eq!(text(didl, "title").as_deref(), Some("Song & Dance"));
        assert_eq!(
            attr(didl, "res", "duration").as_deref(),
            Some("0:03:21.500")
        );
        assert_eq!(attr(didl, "item", "id").as_deref(), Some("1"));
        assert_eq!(attr(didl, "res", "size"), None);
        // Malformed input gives None, never a panic.
        assert_eq!(text("<a><b>unclosed", "b"), None);
        assert_eq!(attr("<a x=unquoted>", "a", "x"), None);
    }

    #[test]
    fn times() {
        assert_eq!(format_time(201.9), "0:03:21");
        assert_eq!(format_time(3725.0), "1:02:05");
        assert_eq!(parse_time("0:03:21"), Some(201.0));
        assert_eq!(parse_time("1:02:05.5"), Some(3725.5));
        assert_eq!(parse_time("0:00:10.1/4"), Some(10.25));
        assert_eq!(parse_time("10:61:00"), None);
        assert_eq!(parse_time("NOT_IMPLEMENTED"), None);
    }
}
