//! Measurements on 1-bit streams: decimate to PCM, FFT, read off tone and noise power.
//!
//! The measurements use coherent sampling (the tone sits exactly on an FFT bin, see
//! [`coherent_tone_hz`]) so no window is needed and the leakage floor is at rounding
//! level – a Blackman-Harris window's −92 dB sidelobes would otherwise hide a −120 dB
//! noise floor.

use super::decimate::DsdToPcm;
use crate::fft::RealFft;

/// The frequency nearest `target_hz` that completes a whole number of cycles in `nfft`
/// samples at `rate`.
pub fn coherent_tone_hz(target_hz: f64, rate: u32, nfft: usize) -> f64 {
    let k = (target_hz * nfft as f64 / rate as f64).round().max(1.0);
    k * rate as f64 / nfft as f64
}

/// Decimate one channel of DSD to `rate / 8` PCM and return the power spectrum (linear,
/// per bin, `nfft/2 + 1` bins) of the last `nfft` samples, plus the PCM rate.
pub fn power_spectrum(bits: &[u8], dsd_rate: u32, nfft: usize) -> (Vec<f64>, u32) {
    let pcm_rate = dsd_rate / 8;
    let mut dec = DsdToPcm::new(dsd_rate, pcm_rate, 1).expect("rate/8 is always valid");
    let mut pcm = Vec::new();
    // No flush: the padded tail would taper the tone and leak all over the spectrum.
    dec.push(&[bits], &mut pcm);
    assert!(
        pcm.len() >= nfft,
        "need at least {nfft} PCM samples, got {} (feed more DSD)",
        pcm.len()
    );
    let mut buf = pcm[pcm.len() - nfft..].to_vec();
    let mut fft = RealFft::new(nfft);
    let mut spec = fft.spectrum();
    fft.forward(&mut buf, &mut spec);
    // Scale so a full-scale sine on a bin reads power 0.5 (its RMS²).
    let scale = 2.0 / (nfft as f64 * nfft as f64);
    let power = spec.iter().map(|c| c.norm_sqr() * scale).collect();
    (power, pcm_rate)
}

/// SNR (dB) of a coherent tone at `tone_hz` against everything else below `band_hz`, and
/// the tone's level in dBFS (0 dBFS = full-scale sine). `nfft` PCM samples (at
/// `dsd_rate / 8`) from the end of the stream are analysed, so feed a little more than
/// `8 × nfft` bits to let the filters settle.
pub fn tone_snr_db(
    bits: &[u8],
    dsd_rate: u32,
    tone_hz: f64,
    band_hz: f64,
    nfft: usize,
) -> (f64, f64) {
    let (power, pcm_rate) = power_spectrum(bits, dsd_rate, nfft);
    let k = (tone_hz * nfft as f64 / pcm_rate as f64).round() as usize;
    let top = ((band_hz * nfft as f64 / pcm_rate as f64) as usize).min(power.len() - 1);
    let tone: f64 = power[k.saturating_sub(1)..=(k + 1).min(top)].iter().sum();
    let noise: f64 = (2..=top)
        .filter(|&b| b + 1 < k || b > k + 1)
        .map(|b| power[b])
        .sum();
    (
        10.0 * (tone / noise.max(1e-300)).log10(),
        10.0 * (tone / 0.5).log10(),
    )
}

/// In-band spectrum peak over mean (dB), ignoring DC: a figure for idle tones.
pub fn spectrum_peak_to_mean_db(bits: &[u8], dsd_rate: u32, band_hz: f64, nfft: usize) -> f64 {
    let (power, pcm_rate) = power_spectrum(bits, dsd_rate, nfft);
    let top = ((band_hz * nfft as f64 / pcm_rate as f64) as usize).min(power.len() - 1);
    let band = &power[4..=top];
    let peak = band.iter().cloned().fold(0.0, f64::max);
    let mean = band.iter().sum::<f64>() / band.len() as f64;
    10.0 * (peak / mean.max(1e-300)).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherent_frequency_lands_on_a_bin() {
        let f = coherent_tone_hz(1000.0, 352_800, 1 << 16);
        let cycles = f * (1 << 16) as f64 / 352_800.0;
        assert!((cycles - cycles.round()).abs() < 1e-9);
        assert!((f - 1000.0).abs() < 352_800.0 / (1 << 16) as f64);
    }
}
