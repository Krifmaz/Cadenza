//! Noise-transfer-function design for the sigma-delta modulators.
//!
//! The NTF is `B(z) / A(z)` with both polynomials monic in `z⁻¹` (so `NTF(∞) = 1`, the
//! realisability condition of a single-loop modulator):
//!
//! * **Zeros** sit on the unit circle inside the signal band at the Schreier-optimised
//!   positions: the roots of the Legendre polynomial of the modulator order, scaled to
//!   the band edge. Those positions minimise the integrated in-band noise power for a
//!   given order (the `opt = 1` zeros of the Delta-Sigma Toolbox `synthesizeNTF`).
//! * **Poles** follow a Butterworth high-pass pattern (bilinear transform of the analog
//!   Butterworth poles) whose corner frequency is found by bisection so that the maximum
//!   out-of-band gain equals `h_inf` (Lee's rule: 1.5 for a stable 1-bit loop).
//!
//! Everything is computed and stored in `f64`; the polynomial coefficients are exact to
//! rounding, and at the extreme oversampling ratios of DSD1024/DSD2048 the zeros are so
//! close to `z = 1` that coefficient rounding moves them – which is harmless, because the
//! noise floor that rounding sets is still below −250 dBFS.

use rustfft::num_complex::Complex;
use std::f64::consts::PI;

/// A designed noise transfer function.
#[derive(Clone, Debug, PartialEq)]
pub struct Ntf {
    /// Modulator order (number of zeros and poles).
    pub order: usize,
    /// Zeros in the `z` plane (on the unit circle).
    pub zeros: Vec<Complex<f64>>,
    /// Poles in the `z` plane (inside the unit circle).
    pub poles: Vec<Complex<f64>>,
    /// Numerator coefficients in `z⁻¹`, `b[0] == 1`.
    pub b: Vec<f64>,
    /// Denominator coefficients in `z⁻¹`, `a[0] == 1`.
    pub a: Vec<f64>,
    /// Maximum out-of-band gain the poles were tuned to.
    pub h_inf: f64,
    /// Signal band edge, in cycles per sample of the modulator rate.
    pub band: f64,
}

/// Roots of the Legendre polynomial `P_n` in `(-1, 1)` (Newton iteration from the
/// classic cosine initial guesses; converges to machine precision in a few steps).
pub fn legendre_roots(n: usize) -> Vec<f64> {
    let mut roots = Vec::with_capacity(n);
    for i in 1..=n {
        let mut x = (PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            // Evaluate P_n and P_n' with the three-term recurrence.
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let (pn, pn_1) = if n == 0 { (1.0, 0.0) } else { (p1, p0) };
            let dp = n as f64 * (x * pn - pn_1) / (x * x - 1.0);
            let dx = pn / dp;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        if x.abs() < 1e-12 {
            x = 0.0;
        }
        roots.push(x);
    }
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    roots
}

/// Multiply out `Π (1 − r_i z⁻¹)` into real coefficients (the roots come in conjugate
/// pairs or are real, so the imaginary parts vanish up to rounding).
fn poly_from_roots(roots: &[Complex<f64>]) -> Vec<f64> {
    let mut p = vec![Complex::new(1.0, 0.0)];
    for r in roots {
        let mut next = vec![Complex::new(0.0, 0.0); p.len() + 1];
        for (i, c) in p.iter().enumerate() {
            next[i] += c;
            next[i + 1] -= c * r;
        }
        p = next;
    }
    p.iter().map(|c| c.re).collect()
}

/// Evaluate a polynomial in `z⁻¹` at `z = e^{jω}`.
fn eval(poly: &[f64], omega: f64) -> Complex<f64> {
    let zinv = Complex::from_polar(1.0, -omega);
    let mut acc = Complex::new(0.0, 0.0);
    for &c in poly.iter().rev() {
        acc = acc * zinv + c;
    }
    acc
}

impl Ntf {
    /// Design an `order`-th order NTF for a signal band of `band` cycles per sample
    /// (`f_band / f_modulator`) and maximum out-of-band gain `h_inf`.
    pub fn design(order: usize, band: f64, h_inf: f64) -> Self {
        assert!((1..=8).contains(&order), "order must be 1..=8, got {order}");
        assert!(
            band > 0.0 && band < 0.25,
            "band must be in (0, 0.25), got {band}"
        );
        assert!(h_inf > 1.0, "h_inf must exceed 1, got {h_inf}");
        let zeros: Vec<Complex<f64>> = legendre_roots(order)
            .iter()
            .map(|x| Complex::from_polar(1.0, 2.0 * PI * band * x))
            .collect();
        let b = poly_from_roots(&zeros);

        // Bisection on the Butterworth corner frequency so the peak gain hits h_inf.
        let poles_for = |wc: f64| -> Vec<Complex<f64>> {
            let omega_c = (wc / 2.0).tan();
            (1..=order)
                .map(|k| {
                    let theta = PI / 2.0 + PI * (2 * k - 1) as f64 / (2 * order) as f64;
                    let s = Complex::from_polar(omega_c, theta);
                    (Complex::new(1.0, 0.0) + s) / (Complex::new(1.0, 0.0) - s)
                })
                .collect()
        };
        let (mut lo, mut hi) = (1e-6, PI - 1e-6);
        let mut poles = poles_for(PI / 2.0);
        let mut a = poly_from_roots(&poles);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            poles = poles_for(mid);
            a = poly_from_roots(&poles);
            let g = max_gain(&b, &a);
            if g > h_inf {
                hi = mid;
            } else {
                lo = mid;
            }
            if hi - lo < 1e-13 {
                break;
            }
        }
        Self {
            order,
            zeros,
            poles,
            b,
            a,
            h_inf,
            band,
        }
    }

    /// Design for a DSD rate in Hz with a `band_hz` signal band (typically 20 kHz).
    pub fn for_rate(order: usize, dsd_rate_hz: u32, band_hz: f64, h_inf: f64) -> Self {
        Self::design(order, band_hz / dsd_rate_hz as f64, h_inf)
    }

    /// `|NTF(e^{jω})|` at `f` cycles per sample.
    pub fn magnitude(&self, f: f64) -> f64 {
        let w = 2.0 * PI * f;
        eval(&self.b, w).norm() / eval(&self.a, w).norm()
    }

    /// Peak gain over frequency.
    pub fn max_gain(&self) -> f64 {
        max_gain(&self.b, &self.a)
    }

    /// Predicted in-band quantisation noise for a 1-bit quantiser (error variance `1/3`),
    /// in dB relative to a full-scale (±1) sine, assuming the linear model. Real
    /// modulators land a few dB away from this; it is a design sanity figure.
    pub fn predicted_noise_dbfs(&self) -> f64 {
        let steps = 4096;
        let mut power = 0.0;
        for i in 0..steps {
            let f = self.band * (i as f64 + 0.5) / steps as f64;
            let m = self.magnitude(f);
            power += m * m;
        }
        // Two-sided spectrum: noise density σ²·|NTF|² per cycle/sample, integrated over ±band.
        let noise = (1.0 / 3.0) * 2.0 * self.band * power / steps as f64;
        10.0 * (noise / 0.5).log10()
    }

    /// `(A − B)` coefficients `1..=order`: the strictly causal error-feedback filter
    /// numerator that realises `H = 1 − NTF`.
    pub fn feedback_numerator(&self) -> Vec<f64> {
        (1..=self.order).map(|k| self.a[k] - self.b[k]).collect()
    }
}

fn max_gain(b: &[f64], a: &[f64]) -> f64 {
    let mut best = eval(b, PI).norm() / eval(a, PI).norm();
    let n = 1024;
    for i in 1..n {
        let w = PI * i as f64 / n as f64;
        let g = eval(b, w).norm() / eval(a, w).norm();
        if g > best {
            best = g;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legendre_roots_match_the_tables() {
        let r5 = legendre_roots(5);
        let want = [
            -0.906_179_845_9,
            -0.538_469_310_1,
            0.0,
            0.538_469_310_1,
            0.906_179_845_9,
        ];
        for (a, b) in r5.iter().zip(want) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        let r7 = legendre_roots(7);
        assert!((r7[6] - 0.949_107_912_3).abs() < 1e-9);
        assert!((r7[4] - 0.405_845_151_4).abs() < 1e-9);
    }

    #[test]
    fn design_hits_h_inf_and_is_monic() {
        for order in [5, 7] {
            let ntf = Ntf::for_rate(order, 2_822_400, 20_000.0, 1.5);
            assert_eq!(ntf.b.len(), order + 1);
            assert_eq!(ntf.a.len(), order + 1);
            assert!((ntf.b[0] - 1.0).abs() < 1e-12 && (ntf.a[0] - 1.0).abs() < 1e-12);
            assert!(
                (ntf.max_gain() - 1.5).abs() < 1e-6,
                "gain {}",
                ntf.max_gain()
            );
            assert!(ntf.poles.iter().all(|p| p.norm() < 1.0));
            // Deep in-band attenuation and a real DC null.
            assert!(
                ntf.magnitude(0.0) < 1e-6,
                "DC null {:e}",
                ntf.magnitude(0.0)
            );
            assert!(ntf.magnitude(1000.0 / 2_822_400.0) < 1e-5);
            let noise = ntf.predicted_noise_dbfs();
            assert!(
                noise < -110.0,
                "order {order}: predicted noise {noise:.1} dBFS"
            );
        }
    }

    #[test]
    fn higher_oversampling_lowers_the_noise() {
        let n64 = Ntf::for_rate(5, 2_822_400, 20_000.0, 1.5).predicted_noise_dbfs();
        let n256 = Ntf::for_rate(5, 4 * 2_822_400, 20_000.0, 1.5).predicted_noise_dbfs();
        let n2048 = Ntf::for_rate(7, 32 * 2_822_400, 20_000.0, 1.5).predicted_noise_dbfs();
        assert!(n256 < n64 - 50.0, "{n64} {n256}");
        assert!(n2048 < n256, "{n256} {n2048}");
    }
}
