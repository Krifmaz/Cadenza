//! The DMAP metadata AirPlay senders attach to a track (`application/x-dmap-tagged`):
//! four-letter code, 32-bit big-endian length, value; `mlit` is a container.

use crate::decode::Tags;

/// Title (`minm`), artist (`asar`) and album (`asal`).
pub fn parse(data: &[u8]) -> Tags {
    let mut tags = Tags::default();
    walk(data, &mut tags, 0);
    tags
}

fn walk(mut data: &[u8], tags: &mut Tags, depth: u32) {
    while data.len() >= 8 {
        let code = &data[..4];
        let len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let Some(value) = data.get(8..8 + len) else {
            return;
        };
        let text = || String::from_utf8_lossy(value).trim().to_string();
        match code {
            b"mlit" if depth < 4 => walk(value, tags, depth + 1),
            b"minm" => tags.title = Some(text()),
            b"asar" => tags.artist = Some(text()),
            b"asal" => tags.album = Some(text()),
            _ => {}
        }
        data = &data[8 + len..];
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn item(code: &[u8; 4], value: &[u8]) -> Vec<u8> {
        let mut v = code.to_vec();
        v.extend_from_slice(&(value.len() as u32).to_be_bytes());
        v.extend_from_slice(value);
        v
    }

    #[test]
    fn reads_title_artist_album() {
        let mut inner = item(b"minm", "Blue in Green".as_bytes());
        inner.extend(item(b"asar", b"Miles Davis"));
        inner.extend(item(b"astm", &[0, 0, 0x14, 0x00]));
        inner.extend(item(b"asal", "Kind of Blue".as_bytes()));
        let t = parse(&item(b"mlit", &inner));
        assert_eq!(t.title.as_deref(), Some("Blue in Green"));
        assert_eq!(t.artist.as_deref(), Some("Miles Davis"));
        assert_eq!(t.album.as_deref(), Some("Kind of Blue"));
        // Truncated data stops cleanly.
        let t = parse(&item(b"mlit", &inner)[..20]);
        assert!(t.album.is_none());
    }
}
