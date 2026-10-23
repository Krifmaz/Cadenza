//! Wire protocol of the Cadenza endpoint: length-prefixed messages over one TCP stream.
//!
//! ```text
//! +----------------+---------+----------------------------+
//! | length u32 LE  | type u8 | payload (length − 1 bytes) |
//! +----------------+---------+----------------------------+
//! ```
//!
//! Control messages carry JSON; the hot ones ([`Message::Frames`], [`Message::Ack`]) are
//! fixed little-endian binary. Sequence of a session:
//!
//! ```text
//! client → Hello{version,name}        server → Caps{name,version,devices}
//! client → Open{device,format}        server → Opened{format}   (or Error)
//! client → Frames{seq,position,words}* server → Ack{…}  (one per Frames)
//! client → Pause/Discard/Drain/Close  server → Drained after a Drain
//! ```
//!
//! Version 2 adds multi-room sync: `Sync` pings (answered at once with the endpoint's
//! clock), `Schedule` (when a stream position must reach the DAC, in the endpoint's
//! clock, and whether to follow that timeline or just start on it) and
//! [`Ack::presenting`]. A version 1 peer simply never sees them.
//!
//! `position` is the stream position (frames) of the first frame of the block, `seq`
//! counts blocks; both let the endpoint spot gaps. [`Ack::played`] is the endpoint's
//! device clock (frames the local output has consumed), [`Ack::queued`] what still waits
//! in its jitter buffer and device queue; the sender uses them to pace itself.

use crate::output::{DeviceInfo, OutputFormat};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

/// Largest message we accept (a second of 32-channel 1.536 MHz words is ~200 MB; nobody
/// sends that, but 64 MB leaves room for big blocks at 2048× rates).
pub const MAX_MESSAGE: usize = 64 << 20;

/// `Hello` payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hello {
    /// [`super::PROTOCOL_VERSION`] of the sender.
    pub version: u32,
    /// Human readable name of the sending instance.
    pub name: String,
}

/// `Caps` payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Caps {
    /// Endpoint name.
    pub name: String,
    /// Protocol version of the endpoint.
    pub version: u32,
    /// Output devices on the endpoint (the first is its configured default).
    pub devices: Vec<DeviceInfo>,
    /// Jitter buffer in milliseconds.
    pub buffer_ms: u32,
}

/// `Open` payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Open {
    /// Device id on the endpoint (empty = its configured default).
    pub device: String,
    /// The format to open.
    pub format: OutputFormat,
}

/// `Ack` payload: the endpoint's view after a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Ack {
    /// Sequence number of the block being acknowledged.
    pub seq: u64,
    /// Frames received in total since `Open`.
    pub received: u64,
    /// Frames the local output has consumed (its clock), since `Open`.
    pub played: u64,
    /// Frames waiting in the jitter buffer plus the device queue.
    pub queued: u64,
    /// Underruns (jitter buffer ran dry) since `Open`.
    pub underruns: u64,
    /// Endpoint monotonic clock, nanoseconds (for latency estimates).
    pub mono_ns: u64,
    /// Stream position the DAC presents at `mono_ns` ([`Ack::UNKNOWN`] when not measured;
    /// version 2).
    pub presenting: u64,
}

impl Ack {
    /// `presenting` when the ack carries no measurement.
    pub const UNKNOWN: u64 = u64::MAX;
}

/// `Schedule` payload (version 2): stream `position` reaches the DAC at `at_ns` on the
/// endpoint's clock, and later positions every `frame_ns` after it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Schedule {
    /// Stream position.
    pub position: u64,
    /// Endpoint clock, ns.
    pub at_ns: i64,
    /// Nanoseconds per frame (the group leader's real rate).
    pub frame_ns: f64,
    /// Steer to this timeline (a follower); otherwise only start on it (the leader).
    pub follow: bool,
}

/// Every message.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    /// Client → server, first message.
    Hello(Hello),
    /// Server → client, answer to `Hello`.
    Caps(Caps),
    /// Client → server.
    Open(Open),
    /// Server → client: the format actually opened.
    Opened(OutputFormat),
    /// Client → server: a block of interleaved words.
    Frames {
        /// Block counter.
        seq: u64,
        /// Stream position in frames of the first frame.
        position: u64,
        /// Interleaved words in the open format.
        words: Vec<i32>,
    },
    /// Server → client.
    Ack(Ack),
    /// Client → server: hold (`true`) or resume.
    Pause(bool),
    /// Client → server: drop everything queued.
    Discard,
    /// Client → server: play out everything queued, then answer `Drained`.
    Drain,
    /// Server → client.
    Drained,
    /// Client → server: close the stream (the connection stays for another `Open`).
    Close,
    /// Either direction.
    Error(String),
    /// Client → server (v2): a clock ping sent at the client's `t0`.
    Sync {
        /// Ping id.
        id: u64,
        /// Client clock, ns.
        t0: u64,
    },
    /// Server → client (v2): the answer, with the endpoint's clock when it arrived.
    SyncReply {
        /// Ping id.
        id: u64,
        /// The client's `t0`, echoed.
        t0: u64,
        /// Endpoint clock, ns (the same clock as [`Ack::mono_ns`]).
        server: u64,
    },
    /// Client → server (v2): the timeline to play on.
    Schedule(Schedule),
}

/// Message type byte of [`Message::Hello`].
const T_HELLO: u8 = 0x01;
/// Message type byte of [`Message::Caps`].
const T_CAPS: u8 = 0x02;
/// Message type byte of [`Message::Open`].
const T_OPEN: u8 = 0x03;
/// Message type byte of [`Message::Opened`].
const T_OPENED: u8 = 0x04;
/// Message type byte of [`Message::Frames`].
const T_FRAMES: u8 = 0x05;
/// Message type byte of [`Message::Ack`].
const T_ACK: u8 = 0x06;
/// Message type byte of [`Message::Pause`].
const T_PAUSE: u8 = 0x07;
/// Message type byte of [`Message::Discard`].
const T_DISCARD: u8 = 0x08;
/// Message type byte of [`Message::Drain`].
const T_DRAIN: u8 = 0x09;
/// Message type byte of [`Message::Drained`].
const T_DRAINED: u8 = 0x0A;
/// Message type byte of [`Message::Close`].
const T_CLOSE: u8 = 0x0B;
/// Message type byte of [`Message::Error`].
const T_ERROR: u8 = 0x0C;
/// Message type byte of [`Message::Sync`].
const T_SYNC: u8 = 0x0D;
/// Message type byte of [`Message::SyncReply`].
const T_SYNC_REPLY: u8 = 0x0E;
/// Message type byte of [`Message::Schedule`].
const T_SCHEDULE: u8 = 0x0F;

/// An `InvalidData` error for a malformed message.
fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Serialise a JSON payload.
fn json<T: Serialize>(v: &T) -> io::Result<Vec<u8>> {
    serde_json::to_vec(v).map_err(|e| bad(e.to_string()))
}

/// Parse a JSON payload.
fn from_json<T: for<'de> Deserialize<'de>>(b: &[u8]) -> io::Result<T> {
    serde_json::from_slice(b).map_err(|e| bad(e.to_string()))
}

/// The little-endian `u64` at byte `at` of a payload.
fn u64_at(b: &[u8], at: usize) -> io::Result<u64> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| bad("short message"))
}

impl Message {
    /// Serialise to `type + payload` (without the length prefix).
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        match self {
            Message::Hello(h) => {
                out.push(T_HELLO);
                out.extend(json(h)?);
            }
            Message::Caps(c) => {
                out.push(T_CAPS);
                out.extend(json(c)?);
            }
            Message::Open(o) => {
                out.push(T_OPEN);
                out.extend(json(o)?);
            }
            Message::Opened(f) => {
                out.push(T_OPENED);
                out.extend(json(f)?);
            }
            Message::Frames {
                seq,
                position,
                words,
            } => {
                out.reserve(17 + words.len() * 4);
                out.push(T_FRAMES);
                out.extend(seq.to_le_bytes());
                out.extend(position.to_le_bytes());
                for w in words {
                    out.extend(w.to_le_bytes());
                }
            }
            Message::Ack(a) => {
                out.push(T_ACK);
                for v in [
                    a.seq,
                    a.received,
                    a.played,
                    a.queued,
                    a.underruns,
                    a.mono_ns,
                    a.presenting,
                ] {
                    out.extend(v.to_le_bytes());
                }
            }
            Message::Pause(p) => {
                out.push(T_PAUSE);
                out.push(*p as u8);
            }
            Message::Discard => out.push(T_DISCARD),
            Message::Drain => out.push(T_DRAIN),
            Message::Drained => out.push(T_DRAINED),
            Message::Close => out.push(T_CLOSE),
            Message::Error(e) => {
                out.push(T_ERROR);
                out.extend(e.as_bytes());
            }
            Message::Sync { id, t0 } => {
                out.push(T_SYNC);
                out.extend(id.to_le_bytes());
                out.extend(t0.to_le_bytes());
            }
            Message::SyncReply { id, t0, server } => {
                out.push(T_SYNC_REPLY);
                for v in [id, t0, server] {
                    out.extend(v.to_le_bytes());
                }
            }
            Message::Schedule(s) => {
                out.push(T_SCHEDULE);
                out.extend(json(s)?);
            }
        }
        Ok(out)
    }

    /// Parse `type + payload`.
    pub fn decode(b: &[u8]) -> io::Result<Message> {
        let (&t, p) = b.split_first().ok_or_else(|| bad("empty message"))?;
        Ok(match t {
            T_HELLO => Message::Hello(from_json(p)?),
            T_CAPS => Message::Caps(from_json(p)?),
            T_OPEN => Message::Open(from_json(p)?),
            T_OPENED => Message::Opened(from_json(p)?),
            T_FRAMES => {
                if p.len() < 16 || (p.len() - 16) % 4 != 0 {
                    return Err(bad("bad frames block"));
                }
                let words = p[16..]
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Message::Frames {
                    seq: u64_at(p, 0)?,
                    position: u64_at(p, 8)?,
                    words,
                }
            }
            T_ACK => Message::Ack(Ack {
                seq: u64_at(p, 0)?,
                received: u64_at(p, 8)?,
                played: u64_at(p, 16)?,
                queued: u64_at(p, 24)?,
                underruns: u64_at(p, 32)?,
                mono_ns: u64_at(p, 40)?,
                // Version 1 acks end before it.
                presenting: u64_at(p, 48).unwrap_or(Ack::UNKNOWN),
            }),
            T_SYNC => Message::Sync {
                id: u64_at(p, 0)?,
                t0: u64_at(p, 8)?,
            },
            T_SYNC_REPLY => Message::SyncReply {
                id: u64_at(p, 0)?,
                t0: u64_at(p, 8)?,
                server: u64_at(p, 16)?,
            },
            T_SCHEDULE => Message::Schedule(from_json(p)?),
            T_PAUSE => Message::Pause(p.first().copied().unwrap_or(0) != 0),
            T_DISCARD => Message::Discard,
            T_DRAIN => Message::Drain,
            T_DRAINED => Message::Drained,
            T_CLOSE => Message::Close,
            T_ERROR => Message::Error(String::from_utf8_lossy(p).into_owned()),
            other => return Err(bad(format!("unknown message type {other:#x}"))),
        })
    }

    /// Write one framed message.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let body = self.encode()?;
        let mut buf = Vec::with_capacity(4 + body.len());
        buf.extend((body.len() as u32).to_le_bytes());
        buf.extend(body);
        w.write_all(&buf)?;
        w.flush()
    }

    /// Read one framed message (blocking).
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Message> {
        let mut len = [0u8; 4];
        r.read_exact(&mut len)?;
        let len = u32::from_le_bytes(len) as usize;
        if len == 0 || len > MAX_MESSAGE {
            return Err(bad(format!("bad message length {len}")));
        }
        let mut body = vec![0u8; len];
        r.read_exact(&mut body)?;
        Message::decode(&body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::DsdCaps;
    use cadenza_dsp::dsd::DsdPacking;

    fn round_trip(m: Message) {
        let mut buf = Vec::new();
        m.write_to(&mut buf).unwrap();
        assert_eq!(
            u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize,
            buf.len() - 4
        );
        let back = Message::read_from(&mut &buf[..]).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn every_message_round_trips() {
        round_trip(Message::Hello(Hello {
            version: 1,
            name: "main".into(),
        }));
        round_trip(Message::Caps(Caps {
            name: "kitchen".into(),
            version: 1,
            devices: vec![DeviceInfo {
                backend: "alsa".into(),
                id: "hw:1,0".into(),
                name: "DAC".into(),
                rates: vec![44100, 352800],
                bits: vec![24, 32],
                channels: Some(2),
                default: true,
                exclusive: true,
                dsd: DsdCaps::dop_only(Some(11_289_600)),
            }],
            buffer_ms: 150,
        }));
        round_trip(Message::Open(Open {
            device: String::new(),
            format: OutputFormat::dsd(11_289_600, DsdPacking::Native, 2),
        }));
        round_trip(Message::Opened(OutputFormat::pcm(705_600, 24, 2)));
        round_trip(Message::Frames {
            seq: 7,
            position: 123_456,
            words: vec![i32::MIN, -1, 0, 1, i32::MAX],
        });
        round_trip(Message::Ack(Ack {
            seq: 7,
            received: 100,
            played: 60,
            queued: 40,
            underruns: 1,
            mono_ns: 99,
            presenting: 55,
        }));
        round_trip(Message::Sync { id: 3, t0: 1_000 });
        round_trip(Message::SyncReply {
            id: 3,
            t0: 1_000,
            server: 2_000,
        });
        round_trip(Message::Schedule(Schedule {
            position: 4096,
            at_ns: -12,
            frame_ns: 22675.7,
            follow: true,
        }));
        // A version 1 ack (no `presenting`) still parses.
        let mut v1 = vec![T_ACK];
        for v in [1u64, 2, 3, 4, 5, 6] {
            v1.extend(v.to_le_bytes());
        }
        match Message::decode(&v1).unwrap() {
            Message::Ack(a) => assert_eq!((a.mono_ns, a.presenting), (6, Ack::UNKNOWN)),
            m => panic!("{m:?}"),
        }
        round_trip(Message::Pause(true));
        round_trip(Message::Pause(false));
        round_trip(Message::Discard);
        round_trip(Message::Drain);
        round_trip(Message::Drained);
        round_trip(Message::Close);
        round_trip(Message::Error("nope".into()));
    }

    #[test]
    fn frames_are_little_endian_binary_and_garbage_is_rejected() {
        let m = Message::Frames {
            seq: 1,
            position: 2,
            words: vec![0x0102_0304],
        };
        let b = m.encode().unwrap();
        assert_eq!(b[0], T_FRAMES);
        assert_eq!(&b[1..9], &1u64.to_le_bytes());
        assert_eq!(&b[17..21], &[0x04, 0x03, 0x02, 0x01]);
        assert!(Message::decode(&[T_FRAMES, 1, 2, 3]).is_err());
        assert!(Message::decode(&[0x7F]).is_err());
        assert!(Message::decode(&[]).is_err());
        let huge = [0xFFu8, 0xFF, 0xFF, 0x7F, T_CLOSE];
        assert!(Message::read_from(&mut &huge[..]).is_err());
    }
}
