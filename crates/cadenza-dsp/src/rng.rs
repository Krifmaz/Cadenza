//! A small, fast, deterministic pseudo-random generator (xoshiro256**) for dither.
//!
//! Dither only needs good statistics, not cryptographic quality, and keeping the
//! generator in-crate avoids pulling in a `rand` version and keeps tests reproducible.

/// xoshiro256** by Blackman & Vigna (public domain reference implementation).
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Seed the generator; the seed is expanded with SplitMix64 so any value is fine.
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut v = z;
            v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            v ^ (v >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[-0.5, 0.5)`.
    #[inline]
    pub fn centred(&mut self) -> f64 {
        self.uniform() - 0.5
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(0x5EED_CADE_72A1_0001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_has_the_right_mean_and_variance() {
        let mut r = Rng::new(1);
        let n = 200_000;
        let (mut sum, mut sq) = (0.0, 0.0);
        for _ in 0..n {
            let v = r.centred();
            sum += v;
            sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.005, "mean {mean}");
        assert!((var - 1.0 / 12.0).abs() < 0.002, "variance {var}");
    }
}
