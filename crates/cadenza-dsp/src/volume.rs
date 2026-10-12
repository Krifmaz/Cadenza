//! Volume in dB and mute: a single 64-bit multiply per sample.

use serde::{Deserialize, Serialize};

/// Volume setting.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Volume {
    /// Gain in dB (`0.0` = unity, negative attenuates). Clamped to `[-120, +20]`.
    pub gain_db: f64,
    /// Silence regardless of `gain_db`.
    pub muted: bool,
}

impl Default for Volume {
    fn default() -> Self {
        Self {
            gain_db: 0.0,
            muted: false,
        }
    }
}

impl Volume {
    /// Unity gain.
    pub const UNITY: Volume = Volume {
        gain_db: 0.0,
        muted: false,
    };

    /// Lowest gain considered audible; below this the volume is treated as silence.
    pub const MIN_DB: f64 = -120.0;

    /// Linear multiplier (0 when muted).
    pub fn gain(&self) -> f64 {
        if self.muted || self.gain_db <= Self::MIN_DB {
            0.0
        } else {
            10f64.powf(self.gain_db.min(20.0) / 20.0)
        }
    }

    /// `true` when applying this volume would not change a single sample.
    pub fn is_unity(&self) -> bool {
        !self.muted && self.gain_db == 0.0
    }

    /// Multiply `samples` in place (a no-op at unity).
    pub fn apply(&self, samples: &mut [f64]) {
        if self.is_unity() {
            return;
        }
        let g = self.gain();
        for s in samples {
            *s *= g;
        }
    }

    /// Convert a linear 0…1 UI fader to dB with a gentle curve (`1.0` → 0 dB, `0.0` → mute).
    pub fn from_linear(level: f64) -> Volume {
        let level = level.clamp(0.0, 1.0);
        if level <= 0.0 {
            return Volume {
                gain_db: Self::MIN_DB,
                muted: false,
            };
        }
        // 60 dB of travel, cubic feel like most players.
        Volume {
            gain_db: (20.0 * level.log10() * 3.0).max(Self::MIN_DB),
            muted: false,
        }
    }

    /// The inverse of [`Volume::from_linear`].
    pub fn to_linear(&self) -> f64 {
        if self.muted || self.gain_db <= Self::MIN_DB {
            0.0
        } else {
            10f64.powf(self.gain_db.min(0.0) / 60.0).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gains() {
        assert_eq!(Volume::UNITY.gain(), 1.0);
        assert!(
            (Volume {
                gain_db: -6.0206,
                muted: false
            }
            .gain()
                - 0.5)
                .abs()
                < 1e-4
        );
        assert_eq!(
            Volume {
                gain_db: 0.0,
                muted: true
            }
            .gain(),
            0.0
        );
        let mut s = [1.0, -0.5];
        Volume {
            gain_db: -20.0,
            muted: false,
        }
        .apply(&mut s);
        assert!((s[0] - 0.1).abs() < 1e-12 && (s[1] + 0.05).abs() < 1e-12);
    }

    #[test]
    fn linear_round_trip() {
        for l in [0.1, 0.5, 0.75, 1.0] {
            let v = Volume::from_linear(l);
            assert!((v.to_linear() - l).abs() < 1e-9, "{l} -> {v:?}");
        }
        assert_eq!(Volume::from_linear(0.0).to_linear(), 0.0);
    }
}
