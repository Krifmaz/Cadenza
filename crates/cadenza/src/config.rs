//! The TOML configuration file (`~/.config/cadenza/config.toml` or the platform
//! equivalent) and the runtime [`Settings`] the API exposes.

use crate::endpoint::EndpointConfig;
use cadenza_dsp::dsd::{DsdConfig, DsdModulator, DsdOversampling, DsdPacking, DsdRate, OutputMode};
use cadenza_dsp::eq::Band;
use cadenza_dsp::{GpuConfig, MatrixConfig};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

/// Preferred output sample rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateSetting {
    /// The highest rate the device supports in the source's family.
    Auto,
    /// Never resample.
    Source,
    /// Always this rate.
    Fixed(u32),
}

impl RateSetting {
    /// Parse `auto`, `source` or a number (Hz, or kHz if below 1000).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" | "max" => Some(RateSetting::Auto),
            "source" | "none" | "native" => Some(RateSetting::Source),
            n => {
                let v: f64 = n
                    .trim_end_matches("khz")
                    .trim_end_matches("hz")
                    .trim()
                    .parse()
                    .ok()?;
                let hz = if v < 1000.0 {
                    (v * 1000.0).round() as u32
                } else {
                    v as u32
                };
                (hz >= 8000).then_some(RateSetting::Fixed(hz))
            }
        }
    }
}

impl fmt::Display for RateSetting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RateSetting::Auto => write!(f, "auto"),
            RateSetting::Source => write!(f, "source"),
            RateSetting::Fixed(r) => write!(f, "{r}"),
        }
    }
}

impl Serialize for RateSetting {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            RateSetting::Fixed(r) => s.serialize_u32(*r),
            other => s.serialize_str(&other.to_string()),
        }
    }
}

impl<'de> Deserialize<'de> for RateSetting {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Num(u32),
            Str(String),
        }
        match Raw::deserialize(d)? {
            Raw::Num(n) => Ok(RateSetting::Fixed(n)),
            Raw::Str(s) => RateSetting::parse(&s)
                .ok_or_else(|| serde::de::Error::custom(format!("bad rate {s:?}"))),
        }
    }
}

/// `[output]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// `null`, `cpal`, `alsa`, `wasapi`, `coreaudio` or `endpoint://<name>`.
    pub backend: String,
    /// Device id as listed by `cadenza devices` (empty = default).
    pub device: String,
    /// Preferred rate.
    pub rate: RateSetting,
    /// 16, 24 or 32.
    pub bits: u32,
    /// Ceiling for `auto`.
    pub max_rate: u32,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            backend: crate::output::default_backend().to_string(),
            device: String::new(),
            rate: RateSetting::Auto,
            bits: 24,
            max_rate: crate::output::MAX_PCM_RATE,
        }
    }
}

/// `[dsp]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DspConfig {
    /// Bypass everything.
    pub bit_perfect: bool,
    /// Filter id, see `cadenza filters`.
    pub filter: String,
    /// Stopband attenuation for the upsampling filter.
    pub attenuation_db: f64,
    /// Dither id, see `/api/dithers`.
    pub dither: String,
    /// Volume in dB.
    pub volume_db: f64,
    /// Mute.
    pub muted: bool,
    /// Impulse response file (WAV or text); empty = none.
    pub convolution: String,
    /// EQ bands.
    pub eq: Vec<Band>,
    /// EQ master switch.
    pub eq_enabled: bool,
    /// Intersample-peak headroom in dB after the upsampler; `None` (absent in TOML, `null`
    /// in JSON) = automatic: −3 dB when upsampling, 0 dB otherwise.
    pub headroom_db: Option<f64>,
    /// Keep the transparent soft clipper as the last guard before quantisation.
    pub soft_clip: bool,
    /// `pcm` (quantiser) or `dsd` (sigma-delta modulator).
    pub output_mode: OutputMode,
    /// DSD output rate (`dsd64` … `dsd2048`; the family follows the source).
    pub dsd_rate: DsdRate,
    /// `asdm5`, `asdm7`, `dsd5v2`, `dsd7v2`, or the error-correction loops `asdm5ec`,
    /// `asdm7ec`, `asdm7ec8`.
    pub dsd_modulator: DsdModulator,
    /// `dop` or `native`.
    pub dsd_packing: DsdPacking,
    /// Gain before the modulator in dB (headroom; default −3).
    pub dsd_gain_db: f64,
    /// `staged` (long FIR to f_dsd/8 or /16, then a short polyphase stage) or `full`
    /// (the long FIR runs to the DSD rate itself: 1024×/2048× PCM inside the pipeline).
    pub dsd_oversampling: DsdOversampling,
    /// Headphones: `off`, `crossfeed-bs2b`, `crossfeed-chu-moy`, `crossfeed-meier`,
    /// `hrtf-30`, `hrtf-45`, `hrtf-60`.
    pub headphones: cadenza_dsp::HeadphoneMode,
    /// The DAC's own noise floor in dBFS, A-weighted (measured from the web UI or taken
    /// from its spec); the `auto` dither chooses from it. Absent: −120.
    pub dac_noise_floor_db: Option<f64>,
    /// DSD sources (DSF/DFF) go to a DSD output untouched when the rates match and
    /// nothing else needs to touch the samples; off = always re-modulate through PCM.
    pub dsd_passthrough: bool,
    /// `gpu = auto | on | off | cuda | wgpu`, `gpu_device = 0` and
    /// `gpu_precision = auto | f64 | double-f32`.
    #[serde(flatten)]
    pub gpu: GpuConfig,
}

impl Default for DspConfig {
    fn default() -> Self {
        let dsd = DsdConfig::default();
        Self {
            bit_perfect: false,
            filter: cadenza_dsp::FilterKind::default().id(),
            attenuation_db: cadenza_dsp::fir::DEFAULT_ATTENUATION_DB,
            dither: cadenza_dsp::DitherKind::default().id().to_string(),
            volume_db: 0.0,
            muted: false,
            convolution: String::new(),
            eq: Vec::new(),
            eq_enabled: true,
            headroom_db: None,
            soft_clip: true,
            headphones: cadenza_dsp::HeadphoneMode::Off,
            dac_noise_floor_db: None,
            output_mode: OutputMode::Pcm,
            dsd_rate: dsd.rate,
            dsd_modulator: dsd.modulator,
            dsd_packing: dsd.packing,
            dsd_gain_db: dsd.gain_db,
            dsd_oversampling: dsd.oversampling,
            dsd_passthrough: true,
            gpu: GpuConfig::default(),
        }
    }
}

impl DspConfig {
    /// The DSD part of the pipeline configuration.
    pub fn dsd(&self) -> DsdConfig {
        DsdConfig {
            rate: self.dsd_rate,
            modulator: self.dsd_modulator,
            packing: self.dsd_packing,
            gain_db: self.dsd_gain_db,
            oversampling: self.dsd_oversampling,
        }
    }

    /// Whether the chain is asked for DSD output.
    pub fn dsd_output(&self) -> bool {
        self.output_mode == OutputMode::Dsd && !self.bit_perfect
    }
}

/// `[server]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// `host:port` for the control API and web UI.
    pub listen: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8777".into(),
        }
    }
}

/// `[encore]`.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EncoreConfig {
    /// Encore base URL (`http://host:8080`); empty disables the integration.
    pub url: String,
    /// Login name.
    pub username: String,
    /// Password.
    pub password: String,
    /// An existing `encore_session` cookie value, instead of username/password.
    pub cookie: String,
    /// Shown in Encore's device list (default `Cadenza on <hostname>`).
    pub device_name: String,
    /// An Encore app token (`POST /api/auth/app-token`): sent as `Authorization: Bearer`
    /// instead of a cookie. The desktop app uses this.
    pub token: String,
    /// Device id override; empty = `cadenza-<hash of hostname|url>`.
    pub device_id: String,
    /// `true` when this player runs inside the Encore desktop app (reported to Encore as
    /// `engine.embedded`).
    pub embedded: bool,
}

impl EncoreConfig {
    /// Whether the integration should run: a URL and one way to authenticate.
    pub fn enabled(&self) -> bool {
        !self.url.trim().is_empty()
            && (!self.token.trim().is_empty()
                || !self.cookie.trim().is_empty()
                || !self.username.is_empty())
    }
}

/// The runtime settings: everything the API and Encore can change.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Output section.
    pub output: OutputConfig,
    /// DSP section.
    pub dsp: DspConfig,
    /// Pipeline matrix (`[matrix]`): routing, per-output chains, crossovers.
    pub matrix: MatrixConfig,
}

/// The whole file.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Output.
    pub output: OutputConfig,
    /// DSP.
    pub dsp: DspConfig,
    /// Pipeline matrix.
    pub matrix: MatrixConfig,
    /// HTTP server.
    pub server: ServerConfig,
    /// Encore.
    pub encore: EncoreConfig,
    /// `cadenza endpoint` (the DAC-side network endpoint).
    pub endpoint: EndpointConfig,
    /// System-wide input: a virtual output device or one app (`cadenza serve`).
    pub system: crate::system::SystemConfig,
}

impl Config {
    /// The default location.
    pub fn default_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("cadenza")
            .join("config.toml")
    }

    /// Read `path`; a missing file yields the defaults.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(toml::from_str(&text)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Write `path` (creating directories).
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }

    /// The runtime view.
    pub fn settings(&self) -> Settings {
        Settings {
            output: self.output.clone(),
            dsp: self.dsp.clone(),
            matrix: self.matrix.clone(),
        }
    }

    /// Store changed runtime settings.
    pub fn apply(&mut self, s: &Settings) {
        self.output = s.output.clone();
        self.dsp = s.dsp.clone();
        self.matrix = s.matrix.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_setting_round_trips_in_toml_and_json() {
        for (text, want) in [
            ("\"auto\"", RateSetting::Auto),
            ("\"source\"", RateSetting::Source),
            ("352800", RateSetting::Fixed(352800)),
        ] {
            let got: RateSetting = serde_json::from_str(text).unwrap();
            assert_eq!(got, want);
            assert_eq!(serde_json::to_string(&got).unwrap(), text);
        }
        assert_eq!(RateSetting::parse("192"), Some(RateSetting::Fixed(192_000)));
        assert_eq!(
            RateSetting::parse("96000"),
            Some(RateSetting::Fixed(96_000))
        );
        assert_eq!(RateSetting::parse("x"), None);
    }

    #[test]
    fn config_round_trips_and_defaults_fill_gaps() {
        let c: Config = toml::from_str(
            "[dsp]\nfilter = \"sinc-m-1m\"\n[encore]\nurl = \"http://x\"\nusername = \"u\"\n",
        )
        .unwrap();
        assert_eq!(c.dsp.filter, "sinc-m-1m");
        assert_eq!(c.output.bits, 24);
        assert!(c.encore.enabled());
        let text = toml::to_string_pretty(&c).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(c, back);
        let dir = std::env::temp_dir().join(format!("cadenza-cfg-{}", std::process::id()));
        let p = dir.join("sub").join("config.toml");
        c.save(&p).unwrap();
        assert_eq!(Config::load(&p).unwrap(), c);
        assert_eq!(
            Config::load(&dir.join("missing.toml")).unwrap(),
            Config::default()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn v2_keys_round_trip_and_default_sensibly() {
        let d = Config::default();
        assert_eq!(d.dsp.output_mode, OutputMode::Pcm);
        assert_eq!(d.dsp.dsd_rate, DsdRate::Dsd256);
        assert_eq!(d.dsp.dsd_modulator, DsdModulator::Asdm7);
        assert_eq!(d.dsp.dsd_gain_db, -3.0);
        assert_eq!(d.dsp.headroom_db, None);
        assert!(d.dsp.soft_clip && d.dsp.dsd_passthrough);
        assert_eq!(d.dsp.gpu.mode, cadenza_dsp::GpuMode::Auto);
        assert!(!d.matrix.enabled);
        assert_eq!(d.endpoint.buffer_ms, 150);
        // Every key the README documents parses, flat `gpu` / `gpu_device` included.
        let c: Config = toml::from_str(
            r#"
[output]
backend = "endpoint://Living room"
[dsp]
output_mode = "dsd"
dsd_rate = "dsd512"
dsd_modulator = "dsd7v2"
dsd_packing = "native"
dsd_gain_db = -4.5
dsd_oversampling = "full"
dsd_passthrough = false
headroom_db = -2.0
soft_clip = false
dither = "lns15"
filter = "poly-sinc-gauss-l-long"
gpu = "cuda"
gpu_device = 1
[matrix]
enabled = true
preset = "2.1"
[matrix.stereo]
balance_db = 1.5
[endpoint]
name = "Kitchen"
buffer_ms = 300
"#,
        )
        .unwrap();
        assert_eq!(c.dsp.output_mode, OutputMode::Dsd);
        assert_eq!(c.dsp.dsd().rate, DsdRate::Dsd512);
        assert_eq!(c.dsp.dsd().modulator, DsdModulator::Dsd7v2);
        assert_eq!(c.dsp.dsd().packing, DsdPacking::Native);
        assert_eq!(c.dsp.dsd().gain_db, -4.5);
        assert_eq!(c.dsp.dsd().oversampling, DsdOversampling::Full);
        assert!(!c.dsp.dsd_passthrough && !c.dsp.soft_clip);
        assert_eq!(c.dsp.headroom_db, Some(-2.0));
        assert_eq!(c.dsp.gpu.mode, cadenza_dsp::GpuMode::Cuda);
        assert_eq!(c.dsp.gpu.device, 1);
        assert_eq!(c.matrix.preset, "2.1");
        assert_eq!(c.matrix.stereo.balance_db, 1.5);
        assert_eq!(c.endpoint.name, "Kitchen");
        assert!(c.dsp.dsd_output());
        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.contains("gpu = \"cuda\""), "{text}");
        assert!(text.contains("output_mode = \"dsd\""), "{text}");
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(c, back);
        // JSON (the API) carries the same shape, `headroom_db: null` for automatic.
        let j = serde_json::to_value(d.settings()).unwrap();
        assert_eq!(j["dsp"]["headroom_db"], serde_json::Value::Null);
        assert_eq!(j["dsp"]["gpu"], "auto");
        assert_eq!(j["dsp"]["gpu_device"], 0);
        assert_eq!(j["matrix"]["preset"], "stereo");
        let s: Settings = serde_json::from_value(j).unwrap();
        assert_eq!(s, d.settings());
    }
}
