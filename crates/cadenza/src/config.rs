//! The TOML configuration file (`~/.config/cadenza/config.toml` or the platform
//! equivalent) and the runtime [`Settings`] the API exposes.

use cadenza_dsp::eq::Band;
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
    /// `null`, `cpal`, `alsa` or `wasapi`.
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
            max_rate: 768_000,
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
}

impl Default for DspConfig {
    fn default() -> Self {
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
        }
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
}

impl EncoreConfig {
    /// Whether the integration should run.
    pub fn enabled(&self) -> bool {
        !self.url.trim().is_empty() && (!self.cookie.is_empty() || !self.username.is_empty())
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
}

/// The whole file.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Output.
    pub output: OutputConfig,
    /// DSP.
    pub dsp: DspConfig,
    /// HTTP server.
    pub server: ServerConfig,
    /// Encore.
    pub encore: EncoreConfig,
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
        }
    }

    /// Store changed runtime settings.
    pub fn apply(&mut self, s: &Settings) {
        self.output = s.output.clone();
        self.dsp = s.dsp.clone();
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
}
