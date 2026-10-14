//! DSD → PCM: CIC decimation, droop compensation and a windowed-sinc decimator.
//!
//! ```text
//! 1-bit (±1) → CIC order 5, ÷R₁ (integer, i64) → ×1/R₁⁵ → [comp ⊗ sinc] FIR, ÷4 → PCM
//! ```
//!
//! * The CIC takes the stream from the DSD rate to `4 × f_pcm` with five cascaded
//!   integrators in wrapping 64-bit integers (growth `R₁⁵ ≤ 2⁴⁰`) and five combs after
//!   the decimation. Its aliases of the 0–22 kHz band sit at `4·f_pcm ± 22 kHz` where the
//!   fifth-order `sinc⁵` response is below −160 dB.
//! * The CIC's passband droop (`−0.014 dB at 20 kHz` for R₁ = 2, more for larger R₁) is
//!   cancelled by the 3-tap compensator `[−α, 1 + 2α, −α]` whose `1 + 2α(1 − cos ω)`
//!   response matches the inverse `sinc⁵` series to fourth order (residual < 0.001 dB).
//! * The final linear-phase Kaiser windowed sinc (≥ 150 dB stopband) runs at `4 × f_pcm`,
//!   passes everything to `0.0625 f_pcm` (22.05 kHz at 352.8 kHz) and stops at
//!   `0.1077 f_pcm` (38 kHz): the modulator's shaped noise above the audio band is gone
//!   from the PCM.
//!
//! The output is time-aligned: the group delay is skipped at the start and
//! [`DsdToPcm::flush`] pushes the tail out, so `n` DSD frames give exactly `n / R` PCM
//! frames.

use super::DsdError;
use crate::fir::{kaiser_beta, windowed_sinc};

/// CIC order.
pub const CIC_ORDER: usize = 5;
/// Decimation of the final FIR stage.
const FIR_DECIMATION: usize = 4;
/// Passband edge of the final FIR relative to the PCM rate (22.05 kHz at 352.8 kHz).
pub const PASSBAND: f64 = 0.0625;
/// Stopband edge of the final FIR relative to the PCM rate (38 kHz at 352.8 kHz).
pub const STOPBAND: f64 = 0.1077;
/// Stopband attenuation of the final FIR, dB.
pub const ATTENUATION_DB: f64 = 155.0;

#[derive(Clone, Debug, Default)]
struct Cic {
    integrators: [i64; CIC_ORDER],
    combs: [i64; CIC_ORDER],
    phase: usize,
}

/// Streaming DSD → PCM converter for `channels` planar 1-bit streams.
#[derive(Clone, Debug)]
pub struct DsdToPcm {
    dsd_rate: u32,
    pcm_rate: u32,
    channels: usize,
    r_cic: usize,
    cic: Vec<Cic>,
    cic_scale: f64,
    /// Compensated decimation filter at the CIC output rate.
    h: Vec<f64>,
    /// Per channel FIFO of CIC outputs awaiting the FIR.
    fifo: Vec<Vec<f64>>,
    /// Zero CIC samples the FIFO starts with (group-delay alignment, see [`Self::prime`]).
    prime: usize,
    gain: f64,
    input_frames: u64,
    output_frames: u64,
    pcm: Vec<Vec<f64>>,
}

impl DsdToPcm {
    /// A converter from `dsd_rate` to `pcm_rate` (the DSD rate must be a multiple of
    /// `4 × pcm_rate`, at least 8×).
    pub fn new(dsd_rate: u32, pcm_rate: u32, channels: usize) -> Result<Self, DsdError> {
        if channels == 0 {
            return Err(DsdError::Invalid("channel count is zero".into()));
        }
        if pcm_rate == 0 || dsd_rate % pcm_rate != 0 {
            return Err(DsdError::Rate(format!(
                "{dsd_rate} Hz DSD is not an integer multiple of {pcm_rate} Hz PCM"
            )));
        }
        let r = (dsd_rate / pcm_rate) as usize;
        if r < 8 || r % FIR_DECIMATION != 0 {
            return Err(DsdError::Rate(format!(
                "DSD/PCM ratio {r} must be a multiple of 4 and at least 8"
            )));
        }
        let r_cic = r / FIR_DECIMATION;
        if r_cic > 1 << 12 {
            return Err(DsdError::Rate(format!("CIC decimation {r_cic} is too large")));
        }
        let h = design_filter(r_cic);
        let prime = Self::prime(r_cic, h.len());
        Ok(Self {
            dsd_rate,
            pcm_rate,
            channels,
            r_cic,
            cic: vec![Cic::default(); channels],
            cic_scale: 1.0 / (r_cic as f64).powi(CIC_ORDER as i32),
            prime,
            h,
            fifo: vec![vec![0.0; prime]; channels],
            gain: 1.0,
            input_frames: 0,
            output_frames: 0,
            pcm: vec![Vec::new(); channels],
        })
    }

    /// The natural PCM rate for a DSD rate: 352.8 kHz for the 44.1 family, 384 kHz for 48.
    pub fn natural_pcm_rate(dsd_rate: u32) -> u32 {
        super::family_base(dsd_rate) * 8
    }

    /// Gain applied to the PCM, dB (e.g. +6 dB to map SACD's 50 % modulation to 0 dBFS).
    pub fn with_gain_db(mut self, db: f64) -> Self {
        self.gain = 10f64.powf(db / 20.0);
        self
    }

    /// The FIR output `m` is computed over CIC samples `4m … 4m + N`, i.e. it is centred
    /// `(N−1)/2` CIC samples *ahead* of `4m`; a CIC output `j` itself represents the input
    /// time `jR + (R−1) − K(R−1)/2`, i.e. `1.5(R−1)/R` CIC samples early. Starting the
    /// FIFO with this many zeros makes PCM frame `i` land exactly on DSD time `i·R`.
    fn prime(r_cic: usize, taps: usize) -> usize {
        let r = r_cic as f64;
        let cic_early = (CIC_ORDER as f64 / 2.0 - 1.0) * (r - 1.0) / r;
        ((taps as f64 - 1.0) / 2.0 - cic_early).round().max(0.0) as usize
    }

    /// DSD rate in Hz.
    pub fn dsd_rate(&self) -> u32 {
        self.dsd_rate
    }
    /// PCM rate in Hz.
    pub fn pcm_rate(&self) -> u32 {
        self.pcm_rate
    }
    /// Channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }
    /// Total decimation ratio.
    pub fn ratio(&self) -> usize {
        self.r_cic * FIR_DECIMATION
    }
    /// CIC decimation ratio.
    pub fn cic_ratio(&self) -> usize {
        self.r_cic
    }
    /// Length of the final FIR.
    pub fn taps(&self) -> usize {
        self.h.len()
    }
    /// Processing latency in seconds (what has to arrive before a PCM frame can exist).
    pub fn latency_seconds(&self) -> f64 {
        (self.h.len() as f64 / 2.0) / (FIR_DECIMATION as f64 * self.pcm_rate as f64)
    }
    /// `(DSD frames in, PCM frames out)` so far.
    pub fn frames(&self) -> (u64, u64) {
        (self.input_frames, self.output_frames)
    }

    /// Convert planar bytes (one slice per channel, equal lengths, MSB = oldest bit),
    /// appending interleaved PCM to `out`.
    pub fn push(&mut self, planar: &[&[u8]], out: &mut Vec<f64>) {
        assert_eq!(planar.len(), self.channels, "channel count mismatch");
        let n = planar[0].len();
        assert!(planar.iter().all(|p| p.len() == n), "channels differ in length");
        self.input_frames += 8 * n as u64;
        let r = self.r_cic;
        let scale = self.cic_scale;
        for (c, bytes) in planar.iter().enumerate() {
            let cic = &mut self.cic[c];
            let fifo = &mut self.fifo[c];
            for &byte in bytes.iter() {
                for bit in (0..8).rev() {
                    let x: i64 = if (byte >> bit) & 1 == 1 { 1 } else { -1 };
                    let mut v = x;
                    for acc in cic.integrators.iter_mut() {
                        *acc = acc.wrapping_add(v);
                        v = *acc;
                    }
                    cic.phase += 1;
                    if cic.phase == r {
                        cic.phase = 0;
                        let mut y = v;
                        for d in cic.combs.iter_mut() {
                            let prev = *d;
                            *d = y;
                            y = y.wrapping_sub(prev);
                        }
                        fifo.push(y as f64 * scale);
                    }
                }
            }
        }
        self.run_fir(out);
    }

    fn run_fir(&mut self, out: &mut Vec<f64>) {
        let taps = self.h.len();
        let mut produced = usize::MAX;
        for (c, fifo) in self.fifo.iter_mut().enumerate() {
            let pcm = &mut self.pcm[c];
            pcm.clear();
            let mut pos = 0;
            while pos + taps <= fifo.len() {
                let window = &fifo[pos..pos + taps];
                let mut acc = 0.0;
                for (h, x) in self.h.iter().zip(window) {
                    acc += h * x;
                }
                pcm.push(acc * self.gain);
                pos += FIR_DECIMATION;
            }
            fifo.drain(..pos);
            produced = produced.min(pcm.len());
        }
        if produced == usize::MAX {
            produced = 0;
        }
        for f in 0..produced {
            for pcm in &self.pcm {
                out.push(pcm[f]);
            }
        }
        self.output_frames += produced as u64;
    }

    /// Push silence until every input frame has produced its PCM frame, then reset.
    pub fn flush(&mut self, out: &mut Vec<f64>) {
        let expected = self.input_frames / self.ratio() as u64;
        let zeros = vec![0x69u8; 64]; // DSD "silence": alternating 01101001
        let planar: Vec<&[u8]> = (0..self.channels).map(|_| zeros.as_slice()).collect();
        let before = out.len();
        let mut guard = 0;
        while self.output_frames < expected {
            let frames = self.input_frames;
            self.push(&planar, out);
            self.input_frames = frames;
            guard += 1;
            assert!(guard < 1_000_000, "decimator flush did not converge");
        }
        let extra = (self.output_frames - expected) as usize * self.channels;
        out.truncate(out.len().max(before) - extra);
        self.reset();
    }

    /// Forget all state (seek).
    pub fn reset(&mut self) {
        for c in &mut self.cic {
            *c = Cic::default();
        }
        for f in &mut self.fifo {
            f.clear();
            f.resize(self.prime, 0.0);
        }
        self.input_frames = 0;
        self.output_frames = 0;
    }
}

/// The droop-compensated decimation filter at the CIC output rate (`4 × f_pcm`).
fn design_filter(r_cic: usize) -> Vec<f64> {
    // Frequencies relative to the CIC output rate.
    let pass = PASSBAND / FIR_DECIMATION as f64;
    let stop = STOPBAND / FIR_DECIMATION as f64;
    let cutoff = 0.5 * (pass + stop);
    let transition = stop - pass;
    let taps = (((ATTENUATION_DB - 8.0) / (2.285 * 2.0 * std::f64::consts::PI * transition)).ceil()
        as usize
        + 1)
        | 1;
    let sinc = windowed_sinc(taps, cutoff, kaiser_beta(ATTENUATION_DB));
    // Inverse sinc⁵ droop: 1/CIC(ω) ≈ 1 + (5 (R² − 1) / (24 R²)) ω², ω at the CIC output rate.
    let r = r_cic as f64;
    let alpha = 5.0 * (r * r - 1.0) / (24.0 * r * r);
    let comp = [-alpha, 1.0 + 2.0 * alpha, -alpha];
    let mut h = vec![0.0; taps + 2];
    for (i, s) in sinc.iter().enumerate() {
        for (j, c) in comp.iter().enumerate() {
            h[i + j] += s * c;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsd::modulator::SigmaDelta;
    use crate::dsd::DsdModulator;
    use crate::fft::magnitude_db;
    use std::f64::consts::PI;

    #[test]
    fn rejects_bad_ratios() {
        assert!(DsdToPcm::new(2_822_400, 352_800, 2).is_ok());
        assert!(DsdToPcm::new(2_822_400, 705_600, 2).is_err()); // ratio 4 < 8
        assert!(DsdToPcm::new(2_822_400, 384_000, 2).is_err());
        assert!(DsdToPcm::new(2_822_400, 352_800, 0).is_err());
        assert_eq!(DsdToPcm::natural_pcm_rate(3_072_000), 384_000);
        assert_eq!(DsdToPcm::natural_pcm_rate(45_158_400), 352_800);
    }

    #[test]
    fn filter_meets_the_passband_and_stopband_spec() {
        for r_cic in [2usize, 8, 64] {
            let h = design_filter(r_cic);
            let nfft = 1 << 16;
            let db = magnitude_db(&h, nfft);
            let bin = |f: f64| (f * nfft as f64).round() as usize; // f relative to CIC rate
            // Combined CIC + compensation + FIR response in the passband, to 20 kHz at 352.8k.
            let r = r_cic as f64;
            for (b, d) in db.iter().enumerate().take(bin(20_000.0 / (4.0 * 352_800.0)) + 1).skip(1) {
                let f = b as f64 / nfft as f64; // cycles per CIC-output sample
                let fd = f / r; // cycles per DSD sample
                let cic = ((PI * fd * r).sin() / (r * (PI * fd).sin())).abs().powi(5);
                let total = d + 20.0 * cic.log10();
                assert!(total.abs() < 0.01, "R={r_cic} bin {b}: {total:.4} dB");
            }
            // Stopband from 38 kHz (relative to 352.8k) up to the CIC-rate Nyquist.
            for (b, d) in db.iter().enumerate().take(nfft / 2 + 1).skip(bin(STOPBAND / 4.0)) {
                assert!(*d < -150.0, "R={r_cic} bin {b}: {d:.1} dB");
            }
        }
    }

    #[test]
    fn known_tone_comes_back_within_0_01_db_and_aligned() {
        let dsd_rate = 2_822_400;
        let pcm_rate = 352_800;
        let hz = 1000.0;
        let amp = 0.5;
        let n = dsd_rate as usize / 2;
        let mut m = SigmaDelta::new(DsdModulator::Asdm7, dsd_rate);
        let pcm_in: Vec<f64> = (0..n)
            .map(|i| amp * (2.0 * PI * hz * i as f64 / dsd_rate as f64).sin())
            .collect();
        let mut bits = Vec::new();
        m.process(&pcm_in, &mut bits);
        let mut dec = DsdToPcm::new(dsd_rate, pcm_rate, 1).unwrap();
        let mut out = Vec::new();
        for chunk in bits.chunks(1000) {
            dec.push(&[chunk], &mut out);
        }
        dec.flush(&mut out);
        assert_eq!(out.len(), n / 8);
        // Level in the steady middle part.
        let mid = &out[out.len() / 4..out.len() * 3 / 4];
        let rms = (mid.iter().map(|v| v * v).sum::<f64>() / mid.len() as f64).sqrt();
        let db = 20.0 * (rms * 2f64.sqrt() / amp).log10();
        assert!(db.abs() < 0.01, "level error {db:.4} dB");
        // Time alignment: the output tone has the input's phase (max error one PCM sample).
        for (i, v) in mid.iter().enumerate().step_by(997) {
            let t = (out.len() / 4 + i) as f64 / pcm_rate as f64;
            let want = amp * (2.0 * PI * hz * t).sin();
            assert!((v - want).abs() < 0.02, "sample {i}: {v} vs {want}");
        }
    }

    #[test]
    fn twenty_khz_tone_level_is_flat() {
        let dsd_rate = 2_822_400;
        let pcm_rate = 352_800;
        let hz = 20_000.0;
        let amp = 0.3;
        let n = dsd_rate as usize / 4;
        let mut m = SigmaDelta::new(DsdModulator::Asdm5, dsd_rate);
        let pcm_in: Vec<f64> = (0..n)
            .map(|i| amp * (2.0 * PI * hz * i as f64 / dsd_rate as f64).sin())
            .collect();
        let mut bits = Vec::new();
        m.process(&pcm_in, &mut bits);
        let mut dec = DsdToPcm::new(dsd_rate, pcm_rate, 1).unwrap();
        let mut out = Vec::new();
        dec.push(&[&bits], &mut out);
        dec.flush(&mut out);
        let mid = &out[out.len() / 4..out.len() * 3 / 4];
        let rms = (mid.iter().map(|v| v * v).sum::<f64>() / mid.len() as f64).sqrt();
        let db = 20.0 * (rms * 2f64.sqrt() / amp).log10();
        assert!(db.abs() < 0.01, "20 kHz level error {db:.4} dB");
    }

    #[test]
    fn stereo_channels_stay_separate() {
        let mut dec = DsdToPcm::new(2_822_400, 352_800, 2).unwrap();
        let ones = vec![0xFFu8; 4096];
        let zeros = vec![0x00u8; 4096];
        let mut out = Vec::new();
        dec.push(&[&ones, &zeros], &mut out);
        dec.flush(&mut out);
        assert_eq!(out.len(), 4096 * 8 / 8 * 2);
        let mid = &out[out.len() / 2 - 200..out.len() / 2 + 200];
        assert!(mid.iter().step_by(2).all(|v| (v - 1.0).abs() < 1e-6));
        assert!(mid.iter().skip(1).step_by(2).all(|v| (v + 1.0).abs() < 1e-6));
    }
}
