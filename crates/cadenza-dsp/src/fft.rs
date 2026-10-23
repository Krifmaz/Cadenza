//! Thin wrappers around `realfft` so the rest of the crate never has to think about
//! scratch buffers, scaling or the "DC and Nyquist bins must be real" rule.

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// A real-input FFT pair (forward and inverse) of one fixed length.
///
/// The inverse is scaled by `1/len`, so `inverse(forward(x)) == x`.
pub struct RealFft {
    /// Transform length in real samples.
    len: usize,
    /// Planned forward (real → half spectrum) transform.
    fwd: Arc<dyn RealToComplex<f64>>,
    /// Planned inverse (half spectrum → real) transform.
    inv: Arc<dyn ComplexToReal<f64>>,
    /// Scratch for the forward transform, sized once at planning time.
    scratch_fwd: Vec<Complex<f64>>,
    /// Scratch for the inverse transform, sized once at planning time.
    scratch_inv: Vec<Complex<f64>>,
}

impl std::fmt::Debug for RealFft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealFft").field("len", &self.len).finish()
    }
}

impl RealFft {
    /// Plan transforms of `len` real samples (`len` must be even).
    pub fn new(len: usize) -> Self {
        assert!(
            len >= 2 && len % 2 == 0,
            "RealFft length must be even, got {len}"
        );
        let mut planner = RealFftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(len);
        let inv = planner.plan_fft_inverse(len);
        let scratch_fwd = fwd.make_scratch_vec();
        let scratch_inv = inv.make_scratch_vec();
        Self {
            len,
            fwd,
            inv,
            scratch_fwd,
            scratch_inv,
        }
    }

    /// Number of real samples per transform.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Number of complex bins per spectrum (`len / 2 + 1`).
    pub fn bins(&self) -> usize {
        self.len / 2 + 1
    }

    /// `true` when the transform length is zero (never, kept for the `len`/`is_empty` pair).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A zeroed spectrum buffer of the right size.
    pub fn spectrum(&self) -> Vec<Complex<f64>> {
        vec![Complex::new(0.0, 0.0); self.bins()]
    }

    /// Forward transform. `input` is used as scratch and is left modified.
    pub fn forward(&mut self, input: &mut [f64], output: &mut [Complex<f64>]) {
        self.fwd
            .process_with_scratch(input, output, &mut self.scratch_fwd)
            .expect("realfft forward: buffer sizes are fixed by construction");
    }

    /// Inverse transform, scaled by `1/len`. `spectrum` is used as scratch and is left modified.
    pub fn inverse(&mut self, spectrum: &mut [Complex<f64>], output: &mut [f64]) {
        // Rounding leaves tiny imaginary parts in the DC and Nyquist bins after products
        // of spectra; realfft treats those as an error, so clear them first.
        spectrum[0].im = 0.0;
        let last = spectrum.len() - 1;
        spectrum[last].im = 0.0;
        self.inv
            .process_with_scratch(spectrum, output, &mut self.scratch_inv)
            .expect("realfft inverse: buffer sizes are fixed by construction");
        let scale = 1.0 / self.len as f64;
        for v in output.iter_mut() {
            *v *= scale;
        }
    }
}

/// Smallest power of two that is `>= n` (and at least 2).
pub fn next_pow2(n: usize) -> usize {
    n.max(2).next_power_of_two()
}

/// Magnitude spectrum in dB of a real signal zero-padded to `nfft` points (bins `0..=nfft/2`).
///
/// Used by the tests and by the filter designer to measure what was built.
pub fn magnitude_db(signal: &[f64], nfft: usize) -> Vec<f64> {
    let nfft = next_pow2(nfft.max(signal.len()));
    let mut fft = RealFft::new(nfft);
    let mut buf = vec![0.0; nfft];
    buf[..signal.len()].copy_from_slice(signal);
    let mut spec = fft.spectrum();
    fft.forward(&mut buf, &mut spec);
    spec.iter()
        .map(|c| 20.0 * c.norm().max(1e-300).log10())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut fft = RealFft::new(64);
        let x: Vec<f64> = (0..64).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut buf = x.clone();
        let mut spec = fft.spectrum();
        fft.forward(&mut buf, &mut spec);
        let mut out = vec![0.0; 64];
        fft.inverse(&mut spec, &mut out);
        for (a, b) in x.iter().zip(&out) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn magnitude_of_impulse_is_flat() {
        let db = magnitude_db(&[1.0], 256);
        assert!(db.iter().all(|d| d.abs() < 1e-9));
    }
}
