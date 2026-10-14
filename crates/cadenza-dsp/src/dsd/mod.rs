//! DSD: 1-bit output through sigma-delta modulation, DSD input decimation, and packing.
//!
//! ```text
//! PCM 44.1k → long FIR → fs_dsd/8 (or /16) → short polyphase ×8/×16 → ASDM → 1-bit → DoP | native
//!          └─ `full` oversampling: the long FIR runs straight to fs_dsd and the modulator takes it ─┘
//! DSD (DSF/DFF) → CIC ÷R → droop compensation → windowed sinc ÷4 → PCM 352.8 / 384 kHz
//! ```
//!
//! * [`ntf`] designs the noise transfer functions, [`modulator`] runs them as 1-bit loops.
//! * [`interpolate`] is the short last-octave polyphase stage of the `staged` path.
//! * [`packing`] turns planar 1-bit streams into DoP frames or native 32-bit words.
//! * [`decimate`] converts DSD to PCM (the DSD-input path and the test instrument).
//! * [`stage`] chains gain → interpolator → modulators → packer for the [`crate::Pipeline`].
//! * [`measure`] measures SNR and spectra of 1-bit streams (tests, bench).
//!
//! **DSD → DSD at another rate** goes through PCM: the DSD input is decimated to
//! 352.8/384 kHz PCM, the normal pipeline upsamples it again and the modulator produces
//! the new rate. That is also what HQPlayer does ("DSD → PCM → DSD"); there is no
//! lossless way to change the rate of a 1-bit stream. When the source rate equals the
//! output rate and no DSP is wanted, the stream can be packed bit-perfectly with
//! [`packing::Packer`] instead (pass-through).
//!
//! All conventions in this module: bits are **planar per channel, 8 per byte, oldest bit
//! in the MSB**, `1` = positive. `+1.0` in PCM maps to an all-ones stream; SACD's
//! 0 dB (50 % modulation) is therefore −6 dBFS here.

pub mod decimate;
pub mod interpolate;
pub mod measure;
pub mod modulator;
pub mod ntf;
pub mod packing;
pub mod stage;

pub use decimate::DsdToPcm;
pub use modulator::SigmaDelta;
pub use ntf::Ntf;
pub use packing::{NativeFormat, Packer};
pub use stage::DsdStage;

use serde::{Deserialize, Serialize};

/// What the pipeline produces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    /// Multi-bit PCM through the quantiser.
    #[default]
    Pcm,
    /// 1-bit DSD through the sigma-delta modulator.
    Dsd,
}

impl OutputMode {
    /// `pcm` / `dsd`.
    pub fn id(self) -> &'static str {
        match self {
            OutputMode::Pcm => "pcm",
            OutputMode::Dsd => "dsd",
        }
    }
    /// Parse an id.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pcm" => Some(OutputMode::Pcm),
            "dsd" => Some(OutputMode::Dsd),
            _ => None,
        }
    }
}

/// DSD output rate as a multiple of the 44.1 or 48 kHz family base.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum DsdRate {
    /// 2.8224 / 3.072 MHz.
    Dsd64,
    /// 5.6448 / 6.144 MHz.
    Dsd128,
    /// 11.2896 / 12.288 MHz.
    #[default]
    Dsd256,
    /// 22.5792 / 24.576 MHz.
    Dsd512,
    /// 45.1584 / 49.152 MHz.
    Dsd1024,
    /// 90.3168 / 98.304 MHz (native packing only – DoP cannot carry it).
    Dsd2048,
}

impl DsdRate {
    /// Every rate, lowest first.
    pub fn catalog() -> &'static [DsdRate] {
        &[
            DsdRate::Dsd64,
            DsdRate::Dsd128,
            DsdRate::Dsd256,
            DsdRate::Dsd512,
            DsdRate::Dsd1024,
            DsdRate::Dsd2048,
        ]
    }
    /// The oversampling multiple relative to 44.1/48 kHz.
    pub fn multiple(self) -> u32 {
        match self {
            DsdRate::Dsd64 => 64,
            DsdRate::Dsd128 => 128,
            DsdRate::Dsd256 => 256,
            DsdRate::Dsd512 => 512,
            DsdRate::Dsd1024 => 1024,
            DsdRate::Dsd2048 => 2048,
        }
    }
    /// `dsd64` … `dsd2048`.
    pub fn id(self) -> String {
        format!("dsd{}", self.multiple())
    }
    /// `DSD64` … `DSD2048`.
    pub fn label(self) -> String {
        format!("DSD{}", self.multiple())
    }
    /// Parse `dsd256`, `DSD256` or `256`.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        let n: u32 = s.strip_prefix("dsd").unwrap_or(&s).parse().ok()?;
        Self::catalog().iter().copied().find(|r| r.multiple() == n)
    }
    /// The rate for a multiple (`256 → DSD256`).
    pub fn from_multiple(n: u32) -> Option<Self> {
        Self::catalog().iter().copied().find(|r| r.multiple() == n)
    }
    /// Bit rate in Hz for a family base of 44 100 or 48 000.
    pub fn hz(self, family_base: u32) -> u32 {
        family_base * self.multiple()
    }
    /// The rate that a DSD bit rate in Hz corresponds to, with its family base.
    pub fn from_hz(hz: u32) -> Option<(Self, u32)> {
        for base in [44_100, 48_000] {
            if hz % base == 0 {
                if let Some(r) = Self::from_multiple(hz / base) {
                    return Some((r, base));
                }
            }
        }
        None
    }
    /// Whether DoP can carry this rate at all (it needs a PCM frame rate ≤ fs/16, which
    /// no device offers above DSD1024's 2.8224 MHz; DSD2048 is native only).
    pub fn dop_possible(self) -> bool {
        self != DsdRate::Dsd2048
    }
    /// PCM frame rate a DoP stream of this rate needs (16 bits per channel per frame).
    pub fn dop_frame_rate(self, family_base: u32) -> u32 {
        self.hz(family_base) / 16
    }
}

/// Which modulator loop runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DsdModulator {
    /// 5th-order, plain 1-bit quantiser.
    Asdm5,
    /// 7th-order, plain 1-bit quantiser.
    #[default]
    Asdm7,
    /// 5th-order with adaptive dithered quantiser (lower idle tones).
    Dsd5v2,
    /// 7th-order with adaptive dithered quantiser.
    Dsd7v2,
}

impl DsdModulator {
    /// Every modulator, in display order.
    pub fn catalog() -> &'static [DsdModulator] {
        &[
            DsdModulator::Asdm5,
            DsdModulator::Asdm7,
            DsdModulator::Dsd5v2,
            DsdModulator::Dsd7v2,
        ]
    }
    /// `asdm5`, `asdm7`, `dsd5v2`, `dsd7v2`.
    pub fn id(self) -> &'static str {
        match self {
            DsdModulator::Asdm5 => "asdm5",
            DsdModulator::Asdm7 => "asdm7",
            DsdModulator::Dsd5v2 => "dsd5v2",
            DsdModulator::Dsd7v2 => "dsd7v2",
        }
    }
    /// `ASDM5`, `ASDM7`, `DSD5v2`, `DSD7v2`.
    pub fn label(self) -> &'static str {
        match self {
            DsdModulator::Asdm5 => "ASDM5",
            DsdModulator::Asdm7 => "ASDM7",
            DsdModulator::Dsd5v2 => "DSD5v2",
            DsdModulator::Dsd7v2 => "DSD7v2",
        }
    }
    /// Parse an id (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        Self::catalog().iter().copied().find(|m| m.id() == s)
    }
    /// Loop order.
    pub fn order(self) -> usize {
        match self {
            DsdModulator::Asdm5 | DsdModulator::Dsd5v2 => 5,
            DsdModulator::Asdm7 | DsdModulator::Dsd7v2 => 7,
        }
    }
    /// Whether the quantiser is dithered.
    pub fn dithered(self) -> bool {
        matches!(self, DsdModulator::Dsd5v2 | DsdModulator::Dsd7v2)
    }
    /// Maximum out-of-band NTF gain the loop is designed to at the oversampling ratio
    /// `osr` (64 for DSD64 … 2048 for DSD2048; see [`modulator`]). The design point is
    /// chosen per rate: the aggressive value only where the SNR margin needs it (DSD64),
    /// a more conservative loop above it where sustained full-scale DC would otherwise
    /// trip the reset guard and the SNR is far above the requirement anyway.
    pub fn h_inf(self, osr: u32) -> f64 {
        match (self.order(), osr) {
            (5, ..=64) => 1.35,
            (5, _) => 1.28,
            (_, ..=64) => 1.28,
            (_, _) => 1.25,
        }
    }
    /// Input level (full scale = 1) above which the loop would lose stability and the
    /// clipper guard takes over (measured on DC and a 1 kHz tone, see [`modulator`]).
    pub fn input_limit(self) -> f64 {
        match self.order() {
            5 => 0.71,
            _ => 0.71,
        }
    }
}

/// How the 1-bit stream is handed to the device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DsdPacking {
    /// DSD over PCM: 24-bit frames, 16 DSD bits under an alternating `0x05`/`0xFA` marker.
    #[default]
    Dop,
    /// Native DSD words (`DSD_U32_BE` layout in the pipeline; see [`packing`]).
    Native,
}

impl DsdPacking {
    /// `dop` / `native`.
    pub fn id(self) -> &'static str {
        match self {
            DsdPacking::Dop => "dop",
            DsdPacking::Native => "native",
        }
    }
    /// `DoP` / `native`.
    pub fn label(self) -> &'static str {
        match self {
            DsdPacking::Dop => "DoP",
            DsdPacking::Native => "native",
        }
    }
    /// Parse an id.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "dop" => Some(DsdPacking::Dop),
            "native" => Some(DsdPacking::Native),
            _ => None,
        }
    }
}

/// How the PCM reaches the modulator rate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DsdOversampling {
    /// Long FIR to `fs_dsd/8` (≤ DSD256) or `fs_dsd/16`, then a short polyphase stage.
    #[default]
    Staged,
    /// The long FIR runs all the way to the DSD rate (a true 1024×/2048× 64-bit PCM
    /// stream that only ever exists inside the pipeline). Highest quality, very heavy.
    Full,
}

impl DsdOversampling {
    /// `staged` / `full`.
    pub fn id(self) -> &'static str {
        match self {
            DsdOversampling::Staged => "staged",
            DsdOversampling::Full => "full",
        }
    }
    /// Parse an id.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "staged" => Some(DsdOversampling::Staged),
            "full" => Some(DsdOversampling::Full),
            _ => None,
        }
    }
}

/// Default modulator headroom, like HQPlayer's `DSD gain` (−3 dB).
pub const DEFAULT_DSD_GAIN_DB: f64 = -3.0;

/// Everything about the DSD output path (`[dsp]` keys `dsd_rate`, `dsd_modulator`,
/// `dsd_packing`, `dsd_gain_db`, `dsd_oversampling` in config.toml).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DsdConfig {
    /// Output rate (the family – 44.1 or 48 kHz – follows the source).
    pub rate: DsdRate,
    /// Modulator loop.
    pub modulator: DsdModulator,
    /// DoP or native words.
    pub packing: DsdPacking,
    /// Gain applied before the modulator, dB (headroom; default −3).
    pub gain_db: f64,
    /// Staged or full-rate oversampling.
    pub oversampling: DsdOversampling,
}

impl Default for DsdConfig {
    fn default() -> Self {
        Self {
            rate: DsdRate::default(),
            modulator: DsdModulator::default(),
            packing: DsdPacking::default(),
            gain_db: DEFAULT_DSD_GAIN_DB,
            oversampling: DsdOversampling::default(),
        }
    }
}

/// Errors from the DSD building blocks.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DsdError {
    /// Rates do not fit together.
    #[error("{0}")]
    Rate(String),
    /// Parameters that make no sense.
    #[error("invalid parameters: {0}")]
    Invalid(String),
}

/// `44 100` for the 44.1 kHz family, `48 000` for the 48 kHz family (anything else is
/// treated as 48 kHz family).
pub fn family_base(pcm_rate: u32) -> u32 {
    if pcm_rate % 11_025 == 0 {
        44_100
    } else {
        48_000
    }
}

/// The oversampling ratio of a DSD bit rate in Hz relative to its family base
/// (`2_822_400 → 64`, `98_304_000 → 2048`).
pub fn oversampling_ratio(dsd_rate_hz: u32) -> u32 {
    dsd_rate_hz / family_base(dsd_rate_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h_inf_depends_on_the_oversampling_ratio() {
        assert_eq!(oversampling_ratio(2_822_400), 64);
        assert_eq!(oversampling_ratio(3_072_000), 64);
        assert_eq!(oversampling_ratio(98_304_000), 2048);
        for rate in DsdRate::catalog() {
            for base in [44_100, 48_000] {
                assert_eq!(oversampling_ratio(rate.hz(base)), rate.multiple());
            }
        }
        assert_eq!(DsdModulator::Asdm5.h_inf(64), 1.35);
        assert_eq!(DsdModulator::Dsd5v2.h_inf(128), 1.28);
        assert_eq!(DsdModulator::Asdm7.h_inf(64), 1.28);
        assert_eq!(DsdModulator::Dsd7v2.h_inf(512), 1.25);
        // The design point never gets more aggressive as the rate goes up.
        for kind in DsdModulator::catalog() {
            let h: Vec<f64> = DsdRate::catalog()
                .iter()
                .map(|r| kind.h_inf(r.multiple()))
                .collect();
            assert!(h.windows(2).all(|w| w[1] <= w[0]), "{}: {h:?}", kind.id());
        }
    }

    #[test]
    fn rates_ids_and_families() {
        assert_eq!(DsdRate::Dsd64.hz(44_100), 2_822_400);
        assert_eq!(DsdRate::Dsd64.hz(48_000), 3_072_000);
        assert_eq!(DsdRate::Dsd1024.hz(44_100), 45_158_400);
        assert_eq!(DsdRate::Dsd2048.hz(48_000), 98_304_000);
        assert_eq!(DsdRate::parse("DSD512"), Some(DsdRate::Dsd512));
        assert_eq!(DsdRate::parse("128"), Some(DsdRate::Dsd128));
        assert_eq!(
            DsdRate::from_hz(11_289_600),
            Some((DsdRate::Dsd256, 44_100))
        );
        assert_eq!(DsdRate::Dsd256.dop_frame_rate(44_100), 705_600);
        assert!(!DsdRate::Dsd2048.dop_possible());
        assert_eq!(family_base(176_400), 44_100);
        assert_eq!(family_base(96_000), 48_000);
        assert_eq!(DsdModulator::parse("ASDM7"), Some(DsdModulator::Asdm7));
        assert_eq!(DsdModulator::Dsd7v2.order(), 7);
        assert_eq!(OutputMode::parse("dsd"), Some(OutputMode::Dsd));
        assert_eq!(DsdPacking::parse("DoP"), Some(DsdPacking::Dop));
        assert_eq!(DsdOversampling::parse("full"), Some(DsdOversampling::Full));
    }

    #[test]
    fn config_round_trips_through_serde_with_defaults() {
        let c: DsdConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c, DsdConfig::default());
        assert_eq!(c.gain_db, -3.0);
        let c: DsdConfig = serde_json::from_str(
            r#"{"rate":"dsd512","modulator":"dsd7v2","packing":"native","oversampling":"full"}"#,
        )
        .unwrap();
        assert_eq!(c.rate, DsdRate::Dsd512);
        assert_eq!(c.modulator, DsdModulator::Dsd7v2);
        assert_eq!(c.packing, DsdPacking::Native);
        assert_eq!(c.oversampling, DsdOversampling::Full);
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.contains("\"dsd512\""));
    }
}
