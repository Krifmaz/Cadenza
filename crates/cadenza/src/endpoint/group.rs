//! [`GroupOutput`]: one stream played in sync on several endpoints (`endpoint://a+b+c`).
//!
//! The first endpoint is the clock leader: it plays bit-perfect on its own DAC clock. The
//! sender measures every endpoint's clock offset (pings), fits the leader's timeline from
//! its acks (which stream position its DAC presents when) and sends the other endpoints
//! that timeline in their own clocks. Each follower then starts on the same instant and
//! steers its DAC onto the leader's (see [`super::sync`]). All endpoints receive the very
//! same words; only a follower's playback ratio differs by a few ppm.

use super::client::EndpointOutput;
use super::protocol::Schedule;
use super::sync::{Anchor, Timeline};
use crate::output::{DsdCaps, Output, OutputError, OutputFormat};
use std::time::{Duration, Instant};

/// How far ahead of now a new stream is scheduled to start: enough for every endpoint to
/// fill its jitter buffer and device queue.
const START_DELAY: Duration = Duration::from_millis(700);
const PING_EVERY: Duration = Duration::from_millis(200);
const SCHEDULE_EVERY: Duration = Duration::from_millis(250);

/// See the module documentation.
pub struct GroupOutput {
    members: Vec<EndpointOutput>,
    label: String,
    epoch: Instant,
    format: Option<OutputFormat>,
    timeline: Timeline,
    /// The start schedule (our clock) of the current stream, once sent.
    start: Option<Anchor>,
    /// Followers steer (false while paused, until the leader's timeline is back).
    following: bool,
    /// The start schedule still describes the stream (no pause since).
    start_valid: bool,
    last_ping: Instant,
    last_schedule: Instant,
    ping_id: u64,
}

/// `a+b+c` → `["a", "b", "c"]`.
pub fn split_group(spec: &str) -> Vec<String> {
    spec.trim()
        .trim_start_matches("endpoint://")
        .split('+')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

impl GroupOutput {
    /// Connect to every endpoint of `spec` (`a+b+c`, each as `EndpointOutput` takes it).
    pub fn connect(spec: &str) -> Result<Self, OutputError> {
        let names = split_group(spec);
        if names.len() < 2 {
            return Err(OutputError::NoDevice(format!(
                "a sync group needs two endpoints or more: {spec}"
            )));
        }
        let epoch = Instant::now();
        let mut members = Vec::with_capacity(names.len());
        for n in &names {
            let m = EndpointOutput::connect_with(n, epoch)?;
            if m.version() < 2 {
                return Err(OutputError::Backend(format!(
                    "endpoint {n} is too old for sync groups (update it)"
                )));
            }
            members.push(m);
        }
        let label = format!(
            "sync group · {}",
            members
                .iter()
                .map(|m| m.caps().name.clone())
                .collect::<Vec<_>>()
                .join(" + ")
        );
        Ok(Self {
            members,
            label,
            epoch,
            format: None,
            timeline: Timeline::default(),
            start: None,
            following: true,
            start_valid: false,
            last_ping: Instant::now() - PING_EVERY,
            last_schedule: Instant::now(),
            ping_id: 0,
        })
    }

    fn now_ns(&self) -> i64 {
        self.epoch.elapsed().as_nanos() as i64
    }

    fn ping_all(&mut self) {
        self.ping_id += 1;
        for m in &self.members {
            let _ = m.ping(self.ping_id);
        }
        self.last_ping = Instant::now();
    }

    /// Clock offsets of all members (pinging until every one has answered, ≤ 2 s).
    fn offsets(&mut self) -> Result<Vec<i64>, OutputError> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut rounds = 0;
        loop {
            let got: Vec<Option<i64>> = self.members.iter().map(|m| m.clock_offset()).collect();
            if rounds >= 8 && got.iter().all(Option::is_some) {
                return Ok(got.into_iter().flatten().collect());
            }
            if Instant::now() > deadline {
                return Err(OutputError::Backend(
                    "sync group: an endpoint does not answer clock pings".into(),
                ));
            }
            self.ping_all();
            rounds += 1;
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn nominal_ns(&self) -> f64 {
        1e9 / self.format.map_or(48000, |f| f.rate) as f64
    }

    /// Send every member the timeline `a` (our clock), converted to its clock.
    fn send_schedule(&self, a: Anchor, offsets: &[i64], leader_too: bool) {
        for (i, m) in self.members.iter().enumerate() {
            if i == 0 && !leader_too {
                continue;
            }
            let _ = m.schedule(Schedule {
                position: a.position,
                at_ns: a.at + offsets[i],
                frame_ns: a.frame_ns,
                follow: i > 0 && self.following,
            });
        }
    }

    /// Start a stream at the leader's next position, `START_DELAY` from now.
    fn start(&mut self) -> Result<(), OutputError> {
        let offsets = self.offsets()?;
        let a = Anchor {
            position: self.members[0].position(),
            at: self.now_ns() + START_DELAY.as_nanos() as i64,
            frame_ns: self.nominal_ns(),
        };
        self.timeline.clear();
        self.following = true;
        self.send_schedule(a, &offsets, true);
        self.start = Some(a);
        self.start_valid = true;
        self.last_schedule = Instant::now();
        Ok(())
    }

    /// Housekeeping between blocks: pings, the leader's timeline, followers' schedules.
    fn tick(&mut self) {
        if self.last_ping.elapsed() >= PING_EVERY {
            self.ping_all();
        }
        let offset0 = self.members[0].clock_offset();
        if let (Some((pos, mono)), Some(off)) = (self.members[0].presenting(), offset0) {
            // Reports still in flight from before a seek are older than this stream.
            if self.start.is_some_and(|s| pos >= s.position) {
                self.timeline.add(pos, mono as i64 - off);
            }
        }
        if self.last_schedule.elapsed() < SCHEDULE_EVERY {
            return;
        }
        self.last_schedule = Instant::now();
        let offsets: Option<Vec<i64>> = self.members.iter().map(|m| m.clock_offset()).collect();
        let Some(offsets) = offsets else { return };
        // After a pause only the leader's new reports describe the stream.
        let fallback = self.start.filter(|_| self.start_valid);
        let anchor = self.timeline.anchor(self.nominal_ns()).or(fallback);
        if let Some(a) = anchor {
            self.following = true;
            self.send_schedule(a, &offsets, false);
        }
    }
}

impl Output for GroupOutput {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn open(&mut self, format: OutputFormat) -> Result<OutputFormat, OutputError> {
        let mut got = None;
        for m in &mut self.members {
            let f = m.open(format)?;
            if got.is_some_and(|g| g != f) {
                return Err(OutputError::Unsupported {
                    device: m.name(),
                    format,
                });
            }
            got = Some(f);
        }
        self.format = got;
        self.start = None;
        Ok(got.unwrap_or(format))
    }

    fn supports(&self, rate: u32, bits: u32) -> bool {
        self.members.iter().all(|m| m.supports(rate, bits))
    }

    fn dsd_caps(&self) -> DsdCaps {
        let mut caps = self.members[0].dsd_caps();
        for m in &self.members[1..] {
            let c = m.dsd_caps();
            caps.dop &= c.dop;
            caps.native.retain(|n| c.native.contains(n));
            caps.max_rate = match (caps.max_rate, c.max_rate) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
        caps
    }

    fn exclusive(&self) -> bool {
        self.members.iter().all(|m| m.exclusive())
    }

    fn write(&mut self, frames: &[i32]) -> Result<(), OutputError> {
        if self.start.is_none() {
            self.start()?;
        }
        self.tick();
        for m in &mut self.members {
            m.write(frames)?;
        }
        Ok(())
    }

    fn pause(&mut self, paused: bool) -> Result<(), OutputError> {
        if paused {
            // Followers free-run while paused; the leader's timeline restarts on resume.
            self.following = false;
            if let (Some(a), Some(offsets)) = (
                self.timeline.anchor(self.nominal_ns()).or(self.start),
                self.members
                    .iter()
                    .map(|m| m.clock_offset())
                    .collect::<Option<Vec<i64>>>(),
            ) {
                self.send_schedule(a, &offsets, false);
            }
        } else {
            self.timeline.clear();
            self.start_valid = false;
        }
        for m in &mut self.members {
            m.pause(paused)?;
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), OutputError> {
        for m in &mut self.members {
            m.drain()?;
        }
        self.start = None;
        Ok(())
    }

    fn discard(&mut self) -> Result<(), OutputError> {
        for m in &mut self.members {
            m.discard()?;
        }
        self.start = None;
        Ok(())
    }

    fn close(&mut self) {
        for m in &mut self.members {
            m.close();
        }
        self.format = None;
        self.start = None;
    }

    fn queued_frames(&self) -> usize {
        self.members[0].queued_frames()
    }

    fn frames_written(&self) -> u64 {
        self.members[0].frames_written()
    }

    fn format(&self) -> Option<OutputFormat> {
        self.format
    }
}

#[cfg(test)]
mod tests_support {
    use crate::endpoint::{EndpointConfig, EndpointServer};
    use std::path::{Path, PathBuf};

    /// Frames as played: (time ns since the null outputs' epoch, ch0 word).
    pub fn played(path: &Path, rate: u32, ppm: f64, channels: usize) -> Vec<(f64, i32)> {
        let b = std::fs::read(path).unwrap();
        let frame_ns = 1e9 / (rate as f64 * (1.0 + ppm * 1e-6));
        let mut out = Vec::new();
        let mut p = 0;
        while p + 12 <= b.len() {
            let at = u64::from_le_bytes(b[p..p + 8].try_into().unwrap()) as f64;
            let n = u32::from_le_bytes(b[p + 8..p + 12].try_into().unwrap()) as usize;
            p += 12;
            for k in 0..n {
                let w = i32::from_le_bytes(b[p..p + 4].try_into().unwrap());
                out.push((at + k as f64 * frame_ns, w));
                p += 4 * channels;
            }
        }
        out
    }

    /// The (fractional) ramp value presented at `t`, if the ramp is playing then.
    pub fn value_at(frames: &[(f64, i32)], t: f64) -> Option<f64> {
        let i = frames.partition_point(|f| f.0 <= t);
        if i == 0 || i >= frames.len() {
            return None;
        }
        let (a, b) = (frames[i - 1], frames[i]);
        if a.1 <= 0 || b.1 <= 0 {
            return None; // silence
        }
        Some(a.1 as f64 + (b.1 - a.1) as f64 * (t - a.0) / (b.0 - a.0))
    }

    pub fn endpoint(dir: &Path, name: &str, ppm: f64) -> (EndpointServer, PathBuf) {
        let tl = dir.join(format!("{name}.tl"));
        let cfg = EndpointConfig {
            name: name.into(),
            listen: "127.0.0.1:0".into(),
            backend: "null".into(),
            device: format!("ppm={ppm};timeline={}", tl.display()),
            buffer_ms: 100,
            mdns: false,
        };
        (EndpointServer::start(cfg).unwrap(), tl)
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;

    #[test]
    fn group_specs_split() {
        assert_eq!(split_group("endpoint://a+b + c"), ["a", "b", "c"]);
        assert!(GroupOutput::connect("only-one").is_err());
    }

    /// Three endpoints whose "DACs" run at +0, +150 and −120 ppm play one ramp; at every
    /// moment they must present the same stream position.
    #[test]
    fn three_drifting_endpoints_play_in_step() {
        let dir = std::env::temp_dir().join(format!("cadenza-group-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rate = 48_000u32;
        let ppms = [0.0, 150.0, -120.0];
        let eps: Vec<_> = ppms
            .iter()
            .enumerate()
            .map(|(i, p)| endpoint(&dir, &format!("room{i}"), *p))
            .collect();
        let spec = eps
            .iter()
            .map(|(s, _)| s.local_addr().to_string())
            .collect::<Vec<_>>()
            .join("+");
        let mut g = GroupOutput::connect(&spec).unwrap();
        assert!(g.name().starts_with("sync group · room0 + room1 + room2"));
        let f = g.open(OutputFormat::pcm(rate, 24, 2)).unwrap();
        assert_eq!(f.rate, rate);
        let seconds = 12;
        let block = 480usize;
        let mut pos = 0i32;
        for _ in 0..(rate as usize * seconds / block) {
            let mut words = Vec::with_capacity(block * 2);
            for _ in 0..block {
                pos += 1;
                words.push(pos);
                words.push(-pos);
            }
            g.write(&words).unwrap();
        }
        g.drain().unwrap();
        g.close();
        drop(g);
        drop(eps);

        let runs: Vec<Vec<(f64, i32)>> = ppms
            .iter()
            .enumerate()
            .map(|(i, p)| played(&dir.join(format!("room{i}.tl")), rate, *p, 2))
            .collect();
        // The leader plays the ramp bit-perfect, without a gap.
        let lead: Vec<i32> = runs[0].iter().map(|f| f.1).filter(|v| *v > 0).collect();
        assert_eq!(lead.len(), rate as usize * seconds);
        assert!(lead.windows(2).all(|w| w[1] == w[0] + 1));
        // From the start on, the followers present what the leader presents.
        let t0 = runs[0].iter().find(|f| f.1 > 0).unwrap().0;
        let mut worst = [0.0f64; 3];
        // After two seconds the rate estimates have settled.
        let mut settled = [0.0f64; 3];
        let mut checked = 0;
        let mut t = t0 + 0.05e9;
        while t < t0 + (seconds as f64 - 0.5) * 1e9 {
            if let Some(l) = value_at(&runs[0], t) {
                for i in 1..3 {
                    if let Some(v) = value_at(&runs[i], t) {
                        worst[i] = worst[i].max((v - l).abs());
                        if t > t0 + 2e9 {
                            settled[i] = settled[i].max((v - l).abs());
                        }
                        checked += 1;
                    }
                }
            }
            t += 0.02e9;
        }
        eprintln!(
            "sync error, frames at 48 kHz: +150 ppm {:.2} (after 2 s {:.2}), \
             -120 ppm {:.2} (after 2 s {:.2}), {checked} checks",
            worst[1], settled[1], worst[2], settled[2]
        );
        assert!(checked > 900);
        // Unsynchronised, 150 ppm would be 86 frames apart by the end. The bounds leave
        // room for a loaded test machine; idle, it settles within about a frame.
        assert!(worst[1] < 48.0 && worst[2] < 48.0, "{worst:?}");
        assert!(settled[1] < 12.0 && settled[2] < 12.0, "{settled:?}");
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod dsd_tests {
    use super::tests_support::*;
    use super::*;
    use cadenza_dsp::dsd::DsdPacking;

    /// DoP to two endpoints (the follower's DAC 100 ppm fast), a seek in the middle: the
    /// follower slips whole DoP frame pairs, keeps the markers alternating and stays in
    /// step, before and after the seek.
    #[test]
    fn dop_group_slips_in_step_and_survives_a_seek() {
        let dir = std::env::temp_dir().join(format!("cadenza-group-dop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ppms = [0.0, 100.0];
        let eps: Vec<_> = ppms
            .iter()
            .enumerate()
            .map(|(i, p)| endpoint(&dir, &format!("dop{i}"), *p))
            .collect();
        let spec = eps
            .iter()
            .map(|(s, _)| s.local_addr().to_string())
            .collect::<Vec<_>>()
            .join("+");
        let mut g = GroupOutput::connect(&spec).unwrap();
        let f = g
            .open(OutputFormat::dsd(2_822_400, DsdPacking::Dop, 2))
            .unwrap();
        let rate = f.rate as usize; // 176 400 DoP frames/s
        let block = rate / 100;
        let markers = [0x05u32, 0xfa];
        let mut n = 0usize;
        let mut play = |g: &mut GroupOutput, seconds: usize, from: usize| {
            n = from;
            for _ in 0..seconds * 100 {
                let mut words = Vec::with_capacity(block * 2);
                for _ in 0..block {
                    // The payload counts frames (skipping the DSD idle value).
                    let mut c = (n as u32) & 0xffff;
                    if c == 0x6969 {
                        c = 0x6968;
                    }
                    let w = ((markers[n % 2] << 16) | c) as i32;
                    words.push(w);
                    words.push(w);
                    n += 1;
                }
                g.write(&words).unwrap();
            }
        };
        play(&mut g, 6, 0);
        g.discard().unwrap();
        play(&mut g, 5, 1_000_000);
        g.drain().unwrap();
        g.close();
        drop(g);
        drop(eps);

        let runs: Vec<Vec<(f64, i32)>> = ppms
            .iter()
            .enumerate()
            .map(|(i, p)| played(&dir.join(format!("dop{i}.tl")), rate as u32, *p, 2))
            .collect();
        // Markers alternate through everything the follower played, slips included.
        let mut bad = 0;
        for w in runs[1].windows(2) {
            let (a, b) = ((w[0].1 as u32 >> 16) & 0xff, (w[1].1 as u32 >> 16) & 0xff);
            if (a == 0x05 || a == 0xfa) && (b == 0x05 || b == 0xfa) && a == b {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "DoP markers broken by the slips");
        // The counters line up (mod 65 536) after each start has settled.
        let counter = |frames: &[(f64, i32)], t: f64| -> Option<i64> {
            let i = frames.partition_point(|f| f.0 <= t);
            let w = frames.get(i)?.1 as u32;
            let c = w & 0xffff;
            (c != 0x6969 && (w >> 16) & 0xff != 0x69).then_some(c as i64)
        };
        // The seek: where the leader's counter jumps (silence in between skipped).
        let valid: Vec<(f64, u32)> = runs[0]
            .iter()
            .map(|f| (f.0, f.1 as u32 & 0xffff))
            .filter(|f| f.1 != 0x6969)
            .collect();
        let t_seek = valid
            .windows(2)
            .find(|w| {
                let d = w[1].1.wrapping_sub(w[0].1) & 0xffff;
                d > 4 && d < 0xffff - 4
            })
            .map(|w| w[1].0)
            .unwrap();
        let t0 = runs[0]
            .iter()
            .find(|f| f.1 as u32 & 0xffff != 0x6969)
            .unwrap()
            .0;
        let mut worst = 0i64;
        let mut checked = 0;
        for (from, to) in [(t0 + 2e9, t_seek - 0.2e9), (t_seek + 2e9, t_seek + 4.5e9)] {
            let mut t = from;
            while t < to {
                if let (Some(a), Some(b)) = (counter(&runs[0], t), counter(&runs[1], t)) {
                    let d = (b - a + 32768).rem_euclid(65536) - 32768;
                    worst = worst.max(d.abs());
                    checked += 1;
                }
                t += 0.02e9;
            }
        }
        eprintln!("DoP sync error: {worst} frames at 176.4 kHz ({checked} checks)");
        assert!(checked > 200);
        // A slip unit is two frames; the clock offset adds a few more.
        assert!(worst <= 24, "{worst}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
