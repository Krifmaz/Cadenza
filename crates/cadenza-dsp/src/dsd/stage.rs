//! The DSD output stage of the pipeline: gain → (short interpolator) → modulators → packer.
//!
//! It takes interleaved PCM at [`DsdStage::pcm_input_rate`] – `f_dsd / 8` or `/ 16` in
//! `staged` mode, `f_dsd` itself in `full` mode – and returns device words (DoP frames or
//! native 32-bit words). The channels are modulated in parallel (rayon, whole blocks).

use super::interpolate::ShortInterpolator;
use super::modulator::SigmaDelta;
use super::ntf::Ntf;
use super::packing::Packer;
use super::{DsdConfig, DsdError, DsdOversampling, DsdPacking, DsdRate};
use crate::signal_path::{rate_label, Stage};
use rayon::prelude::*;
use serde_json::json;
use std::sync::Arc;

/// Factor of the short polyphase stage for a rate (×8 up to DSD256, ×16 above).
pub fn short_stage_factor(rate: DsdRate) -> u32 {
    if rate <= DsdRate::Dsd256 {
        8
    } else {
        16
    }
}

/// The running DSD output stage.
pub struct DsdStage {
    config: DsdConfig,
    channels: usize,
    pcm_rate: u32,
    dsd_rate: u32,
    gain: f64,
    interpolator: Option<ShortInterpolator>,
    modulators: Vec<SigmaDelta>,
    ntf: Arc<Ntf>,
    packer: Packer,
    planar: Vec<Vec<f64>>,
    upsampled: Vec<Vec<f64>>,
    bits: Vec<Vec<u8>>,
    frames_in: u64,
}

impl DsdStage {
    /// The PCM rate this stage wants at its input for `config` and a source in the
    /// `family_base` (44 100 / 48 000) family.
    pub fn pcm_input_rate(config: &DsdConfig, family_base: u32) -> u32 {
        let dsd = config.rate.hz(family_base);
        match config.oversampling {
            DsdOversampling::Staged => dsd / short_stage_factor(config.rate),
            DsdOversampling::Full => dsd,
        }
    }

    /// Build the stage for `channels` of PCM arriving at `pcm_rate` (which must be what
    /// [`DsdStage::pcm_input_rate`] says). `source_rate` is the original material's
    /// rate, used to place the short stage's passband edge at the source Nyquist.
    pub fn new(
        pcm_rate: u32,
        source_rate: u32,
        channels: usize,
        config: DsdConfig,
    ) -> Result<Self, DsdError> {
        if channels == 0 || pcm_rate == 0 {
            return Err(DsdError::Invalid("channel count or rate is zero".into()));
        }
        if config.packing == DsdPacking::Dop && !config.rate.dop_possible() {
            return Err(DsdError::Rate(format!(
                "{} cannot be carried over DoP; use native packing",
                config.rate.label()
            )));
        }
        let family = super::family_base(pcm_rate);
        let dsd_rate = config.rate.hz(family);
        let want = Self::pcm_input_rate(&config, family);
        if pcm_rate != want {
            return Err(DsdError::Rate(format!(
                "DSD stage for {} ({}) needs {want} Hz PCM at its input, got {pcm_rate} Hz",
                config.rate.label(),
                config.oversampling.id()
            )));
        }
        let interpolator = match config.oversampling {
            DsdOversampling::Staged => Some(ShortInterpolator::new(
                pcm_rate,
                short_stage_factor(config.rate) as usize,
                source_rate as f64 / 2.0,
                channels,
            )?),
            DsdOversampling::Full => None,
        };
        let kind = config.modulator;
        let ntf = Arc::new(Ntf::for_rate(
            kind.order(),
            dsd_rate,
            super::modulator::NTF_BAND_HZ,
            kind.h_inf(),
        ));
        let modulators = (0..channels)
            .map(|_| SigmaDelta::with_ntf(kind, dsd_rate, Arc::clone(&ntf)))
            .collect();
        Ok(Self {
            gain: 10f64.powf(config.gain_db / 20.0),
            config,
            channels,
            pcm_rate,
            dsd_rate,
            interpolator,
            modulators,
            ntf,
            packer: Packer::new(config.packing, channels),
            planar: vec![Vec::new(); channels],
            upsampled: vec![Vec::new(); channels],
            bits: vec![Vec::new(); channels],
            frames_in: 0,
        })
    }

    /// Configuration.
    pub fn config(&self) -> &DsdConfig {
        &self.config
    }
    /// Channels.
    pub fn channels(&self) -> usize {
        self.channels
    }
    /// PCM input rate, Hz.
    pub fn pcm_rate(&self) -> u32 {
        self.pcm_rate
    }
    /// DSD bit rate, Hz.
    pub fn dsd_rate(&self) -> u32 {
        self.dsd_rate
    }
    /// The designed NTF.
    pub fn ntf(&self) -> &Ntf {
        &self.ntf
    }
    /// Rate of the words the device sees (DoP: `f_dsd/16`; native: `f_dsd/32`).
    pub fn device_frame_rate(&self) -> u32 {
        self.packer.frame_rate(self.dsd_rate)
    }
    /// Valid bits per device word (24 for DoP, 32 native).
    pub fn device_word_bits(&self) -> u32 {
        self.packer.word_bits()
    }
    /// `DSD256 (DoP)` style label of what leaves the stage.
    pub fn output_label(&self) -> String {
        format!("{} ({})", self.config.rate.label(), self.config.packing.label())
    }
    /// Guard engagements (clips, state clamps, resets) over all channels so far.
    pub fn guard_engagements(&self) -> u64 {
        self.modulators.iter().map(SigmaDelta::guard_engagements).sum()
    }
    /// Latency in seconds: the short stage's group delay (the modulator adds none).
    pub fn latency_seconds(&self) -> f64 {
        self.interpolator
            .as_ref()
            .map(|i| i.delay_samples() as f64 / self.dsd_rate as f64)
            .unwrap_or(0.0)
    }
    /// PCM frames pushed so far.
    pub fn frames_in(&self) -> u64 {
        self.frames_in
    }

    /// Run interleaved PCM frames at the input rate; returns interleaved device words.
    pub fn process(&mut self, interleaved: &[f64]) -> Vec<i32> {
        let ch = self.channels;
        let frames = interleaved.len() / ch;
        self.frames_in += frames as u64;
        let gain = self.gain;
        for (c, buf) in self.planar.iter_mut().enumerate() {
            buf.clear();
            buf.extend(interleaved.iter().skip(c).step_by(ch).take(frames).map(|v| v * gain));
        }
        let Self {
            interpolator,
            modulators,
            planar,
            upsampled,
            bits,
            ..
        } = self;
        match interpolator {
            Some(it) => {
                for (c, (inp, up)) in planar.iter().zip(upsampled.iter_mut()).enumerate() {
                    up.clear();
                    it.process_channel(c, inp, up);
                }
            }
            None => {
                for (inp, up) in planar.iter().zip(upsampled.iter_mut()) {
                    up.clear();
                    up.extend_from_slice(inp);
                }
            }
        }
        modulators
            .par_iter_mut()
            .zip(upsampled.par_iter())
            .zip(bits.par_iter_mut())
            .for_each(|((m, inp), out)| {
                out.clear();
                m.process(inp, out);
            });
        let planar_bits: Vec<&[u8]> = bits.iter().map(|b| b.as_slice()).collect();
        let mut out = Vec::new();
        self.packer.push(&planar_bits, &mut out);
        out
    }

    /// End of stream: complete the last byte and frame (with modulated silence) and
    /// reset.
    pub fn flush(&mut self) -> Vec<i32> {
        for (m, b) in self.modulators.iter_mut().zip(self.bits.iter_mut()) {
            b.clear();
            m.complete_byte(b);
        }
        let planar_bits: Vec<&[u8]> = self.bits.iter().map(|b| b.as_slice()).collect();
        let mut out = Vec::new();
        self.packer.push(&planar_bits, &mut out);
        self.packer.flush(&mut out);
        self.reset();
        out
    }

    /// Forget all state (seek).
    pub fn reset(&mut self) {
        if let Some(it) = &mut self.interpolator {
            it.reset();
        }
        for m in &mut self.modulators {
            m.reset();
        }
        self.packer.reset();
        self.frames_in = 0;
    }

    /// The signal-path stages this part of the chain contributes (`dsd-oversampler`,
    /// `modulator`).
    pub fn stages(&self) -> Vec<Stage> {
        let mut stages = Vec::new();
        match &self.interpolator {
            Some(it) => stages.push(Stage {
                id: "dsd-oversampler".into(),
                name: "DSD oversampler".into(),
                detail: format!(
                    "×{} short polyphase · {} taps → {} kHz",
                    it.factor(),
                    it.taps(),
                    rate_label(self.dsd_rate)
                ),
                params: json!({
                    "mode": "staged",
                    "factor": it.factor(),
                    "taps": it.taps(),
                    "passband_hz": it.passband_hz(),
                    "attenuation_db": super::interpolate::ATTENUATION_DB,
                    "out_rate": self.dsd_rate,
                    "latency_seconds": self.latency_seconds(),
                }),
                active: true,
            }),
            None => stages.push(Stage {
                id: "dsd-oversampler".into(),
                name: "DSD oversampler".into(),
                detail: format!(
                    "{}× (full) · long FIR to {} kHz",
                    self.config.rate.multiple(),
                    rate_label(self.dsd_rate)
                ),
                params: json!({ "mode": "full", "factor": 1, "out_rate": self.dsd_rate }),
                active: false,
            }),
        }
        let guards = self.guard_engagements();
        let m = self.config.modulator;
        stages.push(Stage {
            id: "modulator".into(),
            name: "Sigma-delta modulator".into(),
            detail: format!(
                "{} · {}th order · {:+.1} dB gain → {}{}",
                m.label(),
                m.order(),
                self.config.gain_db,
                self.output_label(),
                if guards > 0 {
                    format!(" · guard engaged ×{guards}")
                } else {
                    String::new()
                }
            ),
            params: json!({
                "modulator": m.id(),
                "label": m.label(),
                "order": m.order(),
                "dithered": m.dithered(),
                "h_inf": self.ntf.h_inf,
                "predicted_noise_dbfs": self.ntf.predicted_noise_dbfs(),
                "gain_db": self.config.gain_db,
                "input_limit": m.input_limit(),
                "guard_engagements": guards,
                "rate": self.config.rate.id(),
                "rate_hz": self.dsd_rate,
                "packing": self.config.packing.id(),
                "device_frame_rate": self.device_frame_rate(),
                "device_word_bits": self.device_word_bits(),
                "oversampling": self.config.oversampling.id(),
            }),
            active: true,
        });
        stages
    }
}

/// Realtime factor of one modulator (seconds of audio per second of CPU time, one
/// channel) at `rate`, measured over `samples` samples. Used by `cadenza bench` for the
/// per-DSD-rate table.
pub fn bench_modulator(kind: super::DsdModulator, rate: u32, samples: usize) -> f64 {
    let mut m = SigmaDelta::new(kind, rate);
    let input: Vec<f64> = (0..samples)
        .map(|i| 0.5 * (i as f64 * 0.001).sin())
        .collect();
    let mut out = Vec::with_capacity(samples / 8 + 1);
    let start = std::time::Instant::now();
    m.process(&input, &mut out);
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    (samples as f64 / rate as f64) / elapsed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsd::measure::{coherent_tone_hz, tone_snr_db};
    use crate::dsd::packing::unpack_dop;
    use crate::dsd::DsdModulator;
    use std::f64::consts::PI;

    #[test]
    fn input_rate_rules() {
        let c = DsdConfig::default(); // DSD256 staged
        assert_eq!(DsdStage::pcm_input_rate(&c, 44_100), 1_411_200);
        let c = DsdConfig { rate: DsdRate::Dsd1024, ..Default::default() };
        assert_eq!(DsdStage::pcm_input_rate(&c, 44_100), 2_822_400);
        let c = DsdConfig { rate: DsdRate::Dsd1024, oversampling: DsdOversampling::Full, ..Default::default() };
        assert_eq!(DsdStage::pcm_input_rate(&c, 48_000), 49_152_000);
        assert!(DsdStage::new(352_800, 44_100, 2, DsdConfig::default()).is_err());
        let c = DsdConfig { rate: DsdRate::Dsd2048, ..Default::default() };
        assert!(DsdStage::new(DsdStage::pcm_input_rate(&c, 44_100), 44_100, 2, c).is_err(), "DSD2048 over DoP");
        let c = DsdConfig { rate: DsdRate::Dsd2048, packing: DsdPacking::Native, ..Default::default() };
        let s = DsdStage::new(DsdStage::pcm_input_rate(&c, 44_100), 44_100, 2, c).unwrap();
        assert_eq!(s.dsd_rate(), 90_316_800);
        assert_eq!(s.device_frame_rate(), 2_822_400);
    }

    #[test]
    fn staged_dsd64_dop_stream_is_valid_and_clean() {
        let c = DsdConfig { rate: DsdRate::Dsd64, modulator: DsdModulator::Asdm7, gain_db: 0.0, ..Default::default() };
        let pcm_rate = DsdStage::pcm_input_rate(&c, 44_100);
        assert_eq!(pcm_rate, 352_800);
        let mut s = DsdStage::new(pcm_rate, 44_100, 2, c).unwrap();
        assert_eq!(s.device_frame_rate(), 176_400);
        assert_eq!(s.device_word_bits(), 24);
        let nfft = 1 << 16;
        let hz = coherent_tone_hz(1000.0, pcm_rate, nfft);
        let frames = nfft + 8192;
        let pcm: Vec<f64> = (0..frames)
            .flat_map(|i| {
                let v = 0.5 * (2.0 * PI * hz * i as f64 / pcm_rate as f64).sin();
                [v, -v]
            })
            .collect();
        let mut words = Vec::new();
        for chunk in pcm.chunks(2 * 1000) {
            words.extend(s.process(chunk));
        }
        words.extend(s.flush());
        // 8 DSD bits per PCM frame, 16 per DoP frame, 2 channels.
        assert_eq!(words.len(), frames * 8 / 16 * 2);
        let planar = unpack_dop(&words, 2).expect("DoP markers alternate");
        let (snr_l, level_l) = tone_snr_db(&planar[0], 2_822_400, hz, 20_000.0, nfft);
        let (snr_r, _) = tone_snr_db(&planar[1], 2_822_400, hz, 20_000.0, nfft);
        assert!(snr_l > 110.0 && snr_r > 110.0, "SNR {snr_l:.1} / {snr_r:.1} dB");
        assert!((level_l + 6.02).abs() < 0.05, "level {level_l:.2} dBFS");
        assert_eq!(s.guard_engagements(), 0);
        let stages = s.stages();
        assert_eq!(stages[0].id, "dsd-oversampler");
        assert_eq!(stages[1].id, "modulator");
        assert!(stages[1].detail.contains("ASDM7") && stages[1].detail.contains("DSD64 (DoP)"));
    }

    #[test]
    fn full_mode_takes_the_dsd_rate_directly_and_native_words_pack() {
        let c = DsdConfig {
            rate: DsdRate::Dsd64,
            modulator: DsdModulator::Dsd5v2,
            packing: DsdPacking::Native,
            oversampling: DsdOversampling::Full,
            gain_db: -3.0,
        };
        let rate = DsdStage::pcm_input_rate(&c, 48_000);
        assert_eq!(rate, 3_072_000);
        let mut s = DsdStage::new(rate, 48_000, 1, c).unwrap();
        assert!(s.interpolator.is_none());
        assert_eq!(s.latency_seconds(), 0.0);
        let pcm: Vec<f64> = (0..rate as usize / 10)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / rate as f64).sin())
            .collect();
        let mut words = s.process(&pcm[..1000]); // 1000 bits: 31 words + 8 pending bits
        assert_eq!(words.len(), 31);
        words.extend(s.process(&pcm[1000..]));
        words.extend(s.flush());
        assert_eq!(words.len(), pcm.len().div_ceil(32));
        // Average bit density ≈ 50 % for a symmetric tone.
        let ones: u32 = words.iter().map(|w| w.count_ones()).sum();
        assert!((ones as f64 / (words.len() * 32) as f64 - 0.5).abs() < 0.02);
        assert_eq!(s.guard_engagements(), 0, "0 dBFS at −3 dB gain is inside the stable range");
        assert!(s.stages()[0].detail.contains("64× (full)"));
    }

    #[test]
    fn bench_runs() {
        let f = bench_modulator(DsdModulator::Asdm7, 2_822_400, 100_000);
        assert!(f > 0.0);
    }
}
