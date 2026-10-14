//! A CPU model of the GPU convolver: the same Stockham passes, the same interleaved
//! complex layout, the same Hermitian expansion of the partition spectra, the same
//! frequency-delay-line ring and the same MAC order, executed one "thread" at a time.
//!
//! It exists so the algorithm the kernels implement can be verified on a machine without
//! an NVIDIA card (and so a GPU result can be compared with something other than the
//! `realfft`-based CPU engine). It is not meant to be fast.

use super::kernels::pass_schedule;
use crate::convolution::{BlockConvolver, SpectralFilter};
use rustfft::num_complex::Complex;

/// One Stockham radix-2 pass (mirror of the `stockham_radix2` kernel) over `n` complex
/// values stored interleaved.
pub fn stockham_radix2(input: &[f64], out: &mut [f64], n: usize, ns: usize, dir: i32) {
    let half = n >> 1;
    for j in 0..half {
        let k = j & (ns - 1);
        let ang = dir as f64 * std::f64::consts::PI * k as f64 / ns as f64;
        let (wi, wr) = ang.sin_cos();
        let (ar, ai) = (input[2 * j], input[2 * j + 1]);
        let (br, bi) = (input[2 * (j + half)], input[2 * (j + half) + 1]);
        let tr = br * wr - bi * wi;
        let ti = br * wi + bi * wr;
        let base = ((j - k) << 1) + k;
        out[2 * base] = ar + tr;
        out[2 * base + 1] = ai + ti;
        out[2 * (base + ns)] = ar - tr;
        out[2 * (base + ns) + 1] = ai - ti;
    }
}

/// One Stockham radix-4 pass (mirror of the `stockham_radix4` kernel).
pub fn stockham_radix4(input: &[f64], out: &mut [f64], n: usize, ns: usize, dir: i32) {
    let quarter = n >> 2;
    for j in 0..quarter {
        let k = j & (ns - 1);
        let ang = dir as f64 * std::f64::consts::PI * k as f64 / (2.0 * ns as f64);
        let (w1i, w1r) = ang.sin_cos();
        let (w2i, w2r) = (2.0 * ang).sin_cos();
        let (w3i, w3r) = (3.0 * ang).sin_cos();
        let (v0r, v0i) = (input[2 * j], input[2 * j + 1]);
        let (x1r, x1i) = (input[2 * (j + quarter)], input[2 * (j + quarter) + 1]);
        let (x2r, x2i) = (
            input[2 * (j + 2 * quarter)],
            input[2 * (j + 2 * quarter) + 1],
        );
        let (x3r, x3i) = (
            input[2 * (j + 3 * quarter)],
            input[2 * (j + 3 * quarter) + 1],
        );
        let (v1r, v1i) = (x1r * w1r - x1i * w1i, x1r * w1i + x1i * w1r);
        let (v2r, v2i) = (x2r * w2r - x2i * w2i, x2r * w2i + x2i * w2r);
        let (v3r, v3i) = (x3r * w3r - x3i * w3i, x3r * w3i + x3i * w3r);
        let s = dir as f64;
        let (a_r, a_i) = (v0r + v2r, v0i + v2i);
        let (b_r, b_i) = (v0r - v2r, v0i - v2i);
        let (c_r, c_i) = (v1r + v3r, v1i + v3i);
        let (d_r, d_i) = (v1r - v3r, v1i - v3i);
        let (dj_r, dj_i) = (-s * d_i, s * d_r);
        let base = ((j - k) << 2) + k;
        out[2 * base] = a_r + c_r;
        out[2 * base + 1] = a_i + c_i;
        out[2 * (base + ns)] = b_r + dj_r;
        out[2 * (base + ns) + 1] = b_i + dj_i;
        out[2 * (base + 2 * ns)] = a_r - c_r;
        out[2 * (base + 2 * ns) + 1] = a_i - c_i;
        out[2 * (base + 3 * ns)] = b_r - dj_r;
        out[2 * (base + 3 * ns) + 1] = b_i - dj_i;
    }
}

/// Full complex FFT of `n` interleaved values in `a` (unscaled; `dir = -1` forward,
/// `+1` inverse), exactly as the GPU runs it: the pass schedule of
/// [`pass_schedule`], ping-ponging between `a` and `b`. The result is left in `a`.
pub fn fft(a: &mut Vec<f64>, b: &mut Vec<f64>, n: usize, dir: i32) {
    debug_assert_eq!(a.len(), 2 * n);
    debug_assert_eq!(b.len(), 2 * n);
    for (radix, ns) in pass_schedule(n) {
        if radix == 4 {
            stockham_radix4(a, b, n, ns, dir);
        } else {
            stockham_radix2(a, b, n, ns, dir);
        }
        std::mem::swap(a, b);
    }
}

/// Expand a `realfft` half spectrum (`n/2 + 1` bins) to the full `n`-point spectrum the
/// complex kernels work on, interleaved: `X[n − k] = conj(X[k])`.
pub fn expand_hermitian(half: &[Complex<f64>], n: usize) -> Vec<f64> {
    assert_eq!(
        half.len(),
        n / 2 + 1,
        "half spectrum must have n/2 + 1 bins"
    );
    let mut full = vec![0.0; 2 * n];
    for k in 0..n {
        let c = if k <= n / 2 {
            half[k]
        } else {
            half[n - k].conj()
        };
        full[2 * k] = c.re;
        full[2 * k + 1] = c.im;
    }
    full
}

/// All partition spectra of `filter`, expanded and concatenated (`partitions × 2n`
/// doubles): exactly the buffer the GPU holds.
pub fn filter_spectra(filter: &SpectralFilter) -> Vec<f64> {
    let n = 2 * filter.block();
    let mut all = Vec::with_capacity(filter.partitions() * 2 * n);
    for h in filter.partition_spectra() {
        all.extend(expand_hermitian(h, n));
    }
    all
}

/// The GPU data flow on the CPU (see the module docs).
#[derive(Debug)]
pub struct ReferenceConvolver {
    block: usize,
    n: usize,
    depth: usize,
    head: usize,
    count: usize,
    frame: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    fdl: Vec<f64>,
    h: Vec<f64>,
}

impl ReferenceConvolver {
    /// Model the GPU convolver for `filter`.
    pub fn from_filter(filter: &SpectralFilter) -> Self {
        let block = filter.block();
        let n = 2 * block;
        let depth = filter.partitions().max(1);
        Self {
            block,
            n,
            depth,
            head: 0,
            count: 0,
            frame: vec![0.0; n],
            a: vec![0.0; 2 * n],
            b: vec![0.0; 2 * n],
            fdl: vec![0.0; depth * 2 * n],
            h: filter_spectra(filter),
        }
    }

    /// Model the GPU convolver for the impulse `h` at `block`.
    pub fn new(h: &[f64], block: usize) -> Self {
        Self::from_filter(&SpectralFilter::new(h, block))
    }
}

impl BlockConvolver for ReferenceConvolver {
    fn process_block(&mut self, input: &[f64], out: &mut [f64]) {
        let (block, n) = (self.block, self.n);
        assert_eq!(input.len(), block);
        assert_eq!(out.len(), block);
        self.frame[block..].copy_from_slice(input);
        // pack_real
        for i in 0..n {
            self.a[2 * i] = self.frame[i];
            self.a[2 * i + 1] = 0.0;
        }
        fft(&mut self.a, &mut self.b, n, -1);
        // copy_complex into the newest slot
        self.head = (self.head + self.depth - 1) % self.depth;
        self.fdl[self.head * 2 * n..(self.head + 1) * 2 * n].copy_from_slice(&self.a);
        self.count = (self.count + 1).min(self.depth);
        // fdl_mac
        for k in 0..n {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            let mut slot = self.head;
            for p in 0..self.count {
                let xr = self.fdl[2 * (slot * n + k)];
                let xi = self.fdl[2 * (slot * n + k) + 1];
                let hr = self.h[2 * (p * n + k)];
                let hi = self.h[2 * (p * n + k) + 1];
                re = xr.mul_add(hr, re);
                re = (-xi).mul_add(hi, re);
                im = xr.mul_add(hi, im);
                im = xi.mul_add(hr, im);
                slot += 1;
                if slot == self.depth {
                    slot = 0;
                }
            }
            self.a[2 * k] = re;
            self.a[2 * k + 1] = im;
        }
        fft(&mut self.a, &mut self.b, n, 1);
        // unpack_real_scaled
        let scale = 1.0 / n as f64;
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.a[2 * (i + block)] * scale;
        }
        self.frame.copy_within(block.., 0);
    }

    fn latency(&self) -> usize {
        self.block
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolution::Convolver;
    use crate::fft::RealFft;

    #[test]
    fn stockham_matches_realfft_for_every_power_of_two() {
        for log2 in 1..=12 {
            let n = 1usize << log2;
            let x: Vec<f64> = (0..n)
                .map(|i| ((i as f64) * 0.37).sin() + 0.1 * i as f64)
                .collect();
            let mut want_fft = RealFft::new(n.max(2));
            let mut buf = x.clone();
            let mut want = want_fft.spectrum();
            want_fft.forward(&mut buf, &mut want);

            let mut a = vec![0.0; 2 * n];
            for (i, v) in x.iter().enumerate() {
                a[2 * i] = *v;
            }
            let mut b = vec![0.0; 2 * n];
            fft(&mut a, &mut b, n, -1);
            for (k, w) in want.iter().enumerate() {
                assert!((a[2 * k] - w.re).abs() < 1e-9 * n as f64, "n={n} k={k}");
                assert!((a[2 * k + 1] - w.im).abs() < 1e-9 * n as f64, "n={n} k={k}");
            }
            // Hermitian half above n/2, and the inverse restores the input.
            for k in n / 2 + 1..n {
                assert!((a[2 * k] - a[2 * (n - k)]).abs() < 1e-9 * n as f64);
                assert!((a[2 * k + 1] + a[2 * (n - k) + 1]).abs() < 1e-9 * n as f64);
            }
            fft(&mut a, &mut b, n, 1);
            for (i, v) in x.iter().enumerate() {
                assert!((a[2 * i] / n as f64 - v).abs() < 1e-11, "n={n} i={i}");
                assert!((a[2 * i + 1] / n as f64).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn reference_convolver_matches_cpu_engine() {
        let h: Vec<f64> = (0..5 * 256 + 17)
            .map(|i| ((i as f64) * 0.013).sin() / (1.0 + i as f64 * 0.02))
            .collect();
        let x: Vec<f64> = (0..12 * 256).map(|i| ((i as f64) * 0.37).cos()).collect();
        let mut cpu = Convolver::with_block(&h, 256);
        let mut model = ReferenceConvolver::new(&h, 256);
        assert_eq!(cpu.latency(), model.latency());
        let mut want = vec![0.0; 256];
        let mut got = vec![0.0; 256];
        for chunk in x.chunks(256) {
            cpu.process_block(chunk, &mut want);
            model.process_block(chunk, &mut got);
            for (a, b) in want.iter().zip(&got) {
                assert!((a - b).abs() < 1e-10, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn hermitian_expansion_round_trips() {
        let n = 16;
        let mut f = RealFft::new(n);
        let mut x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.9).cos()).collect();
        let mut half = f.spectrum();
        f.forward(&mut x, &mut half);
        let full = expand_hermitian(&half, n);
        assert_eq!(full.len(), 2 * n);
        assert_eq!(full[2 * 3], half[3].re);
        assert_eq!(full[2 * (n - 3)], half[3].re);
        assert_eq!(full[2 * (n - 3) + 1], -half[3].im);
    }
}
