//! A serialisable description of everything that happens to the audio between the
//! source and the device: shown by the web UI as a chain of boxes and sent to Encore.

use serde::{Deserialize, Serialize};

/// One end of the chain: what comes in, or what goes out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Endpoint {
    /// Codec or container name for the input (`FLAC`, `MP3`…), `PCM` for the output.
    pub format: String,
    /// Sample rate in Hz.
    pub rate: u32,
    /// Bit depth (for lossy input, the decoder's output depth).
    pub bits: u32,
    /// Channel count.
    pub channels: u32,
    /// Output only: the device/backend name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

impl Endpoint {
    /// `16/44.1` style short form.
    pub fn short(&self) -> String {
        format!("{}/{}", self.bits, rate_label(self.rate))
    }
}

/// One processing stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    /// Stable id: `volume`, `eq`, `convolution`, `resampler`, `dither`.
    pub id: String,
    /// Display name.
    pub name: String,
    /// One-line summary of the parameters, e.g. `sinc-L 1m taps · ×4 → 176.4 kHz`.
    pub detail: String,
    /// Full parameters.
    #[serde(default)]
    pub params: serde_json::Value,
    /// `false` when the stage is present in the configuration but bypassed.
    pub active: bool,
}

/// The whole path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct SignalPath {
    /// What the decoder delivers.
    pub input: Endpoint,
    /// Active stages, in processing order (bypassed stages are listed with `active: false`).
    pub stages: Vec<Stage>,
    /// What the device receives.
    pub output: Endpoint,
    /// `true` only when no stage altered any sample and the format is unchanged.
    pub bit_perfect: bool,
    /// `FLAC 16/44.1 → 24/176.4` style summary.
    pub label: String,
    /// Processing latency the player must hide, in seconds.
    pub latency_seconds: f64,
}

impl SignalPath {
    /// Build the summary label from the endpoints.
    pub fn make_label(input: &Endpoint, output: &Endpoint, bit_perfect: bool) -> String {
        if bit_perfect {
            format!("{} {} · bit-perfect", input.format, input.short())
        } else {
            format!("{} {} → {}", input.format, input.short(), output.short())
        }
    }
}

/// `44100 → "44.1"`, `176400 → "176.4"`, `48000 → "48"`.
pub fn rate_label(rate: u32) -> String {
    let k = rate as f64 / 1000.0;
    if (k - k.round()).abs() < 1e-9 {
        format!("{}", k.round() as u32)
    } else {
        format!("{k:.1}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        assert_eq!(rate_label(44100), "44.1");
        assert_eq!(rate_label(48000), "48");
        assert_eq!(rate_label(352800), "352.8");
        let i = Endpoint {
            format: "FLAC".into(),
            rate: 44100,
            bits: 16,
            channels: 2,
            device: None,
        };
        let o = Endpoint {
            format: "PCM".into(),
            rate: 176400,
            bits: 24,
            channels: 2,
            device: None,
        };
        assert_eq!(
            SignalPath::make_label(&i, &o, false),
            "FLAC 16/44.1 → 24/176.4"
        );
        assert_eq!(
            SignalPath::make_label(&i, &o, true),
            "FLAC 16/44.1 · bit-perfect"
        );
    }

    #[test]
    fn serialises() {
        let p = SignalPath::default();
        let s = serde_json::to_string(&p).unwrap();
        let back: SignalPath = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }
}
