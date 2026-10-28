//! Platform-independent half of the WASAPI backend: which sample layouts to offer a
//! device, in which order, and how the engine's `i32` words become the bytes each layout
//! wants.
//!
//! Kept apart from `wasapi_backend` (which only builds on Windows) so the negotiation
//! order and the byte packing are unit-tested on every platform.
//!
//! **Why a ladder.** Exclusive-mode drivers are picky about the container, not just the
//! bit depth: many Realtek HD Audio endpoints refuse 24 valid bits in a 32-bit word but
//! take packed 24-bit (three bytes per sample), others only take 16 or 32 bits. Asking
//! for exactly one layout turned "the device can play this track" into "format not
//! supported", so the backend walks [`exclusive_candidates`] and opens the first layout
//! the driver accepts. When exclusive mode is refused altogether (disabled in the Windows
//! sound settings, or another app holds the device) the backend falls back to a shared
//! stream of [`SampleLayout::Float32`] that the Windows mixer converts for us.

use std::collections::VecDeque;

/// How one sample word is laid out in the device buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleLayout {
    /// Little-endian signed integers: `valid` meaningful bits, MSB-aligned in a `store`-bit
    /// container (16/16, 24/24 packed, 24/32 or 32/32).
    Int {
        /// Container size in bits (16, 24 or 32).
        store: u16,
        /// Meaningful bits in the container (16, 24 or 32).
        valid: u16,
    },
    /// Little-endian IEEE 754 single precision in `[-1.0, 1.0)`: the shared-mode mixer's
    /// own sample type, which holds 24-bit integers losslessly.
    Float32,
}

impl SampleLayout {
    /// The integer layout the engine's `bits` map to by default: 16 in 16-bit words,
    /// 24 left-justified in 32-bit words, anything else as full 32-bit words.
    pub fn int_for_bits(bits: u32) -> Self {
        match bits {
            16 => Self::Int {
                store: 16,
                valid: 16,
            },
            24 => Self::Int {
                store: 32,
                valid: 24,
            },
            _ => Self::Int {
                store: 32,
                valid: 32,
            },
        }
    }

    /// Bytes one sample occupies in the device buffer.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Int { store, .. } => store as usize / 8,
            Self::Float32 => 4,
        }
    }

    /// The valid bit depth this layout carries (what the engine must produce for it);
    /// `None` for float, which carries whatever depth it is fed.
    pub fn valid_bits(self) -> Option<u32> {
        match self {
            Self::Int { valid, .. } => Some(valid as u32),
            Self::Float32 => None,
        }
    }
}

/// Integer layouts to try in exclusive mode for a PCM stream the engine would like at
/// `wanted` bits, best first: the wanted depth in its usual container, then the same
/// depth in the other container (24-bit only), then the remaining depths from deepest to
/// shallowest so quality degrades as little as possible. No layout appears twice.
pub fn exclusive_candidates(wanted: u32) -> Vec<SampleLayout> {
    /// Every container a depth can travel in, preferred one first.
    fn containers(bits: u32) -> Vec<SampleLayout> {
        match bits {
            16 => vec![SampleLayout::int_for_bits(16)],
            // 24-in-32 is what most USB DACs want; packed 24 is what many HDA codecs want.
            24 => vec![
                SampleLayout::int_for_bits(24),
                SampleLayout::Int {
                    store: 24,
                    valid: 24,
                },
            ],
            _ => vec![SampleLayout::int_for_bits(32)],
        }
    }
    let mut out: Vec<SampleLayout> = Vec::new();
    for bits in [wanted, 32, 24, 16] {
        for layout in containers(bits) {
            if !out.contains(&layout) {
                out.push(layout);
            }
        }
    }
    out
}

/// Append `samples` (engine words with `bits` valid bits, right-justified in an `i32`) to
/// `out` as `layout` bytes.
///
/// Integer layouts whose `valid` equals `bits` are bit-exact. For float the word is
/// scaled by `2^-(bits-1)`, which is exact for up to 24 bits.
pub fn encode(samples: &[i32], bits: u32, layout: SampleLayout, out: &mut VecDeque<u8>) {
    match layout {
        SampleLayout::Int { store: 16, .. } => {
            for &v in samples {
                out.extend((v as i16).to_le_bytes());
            }
        }
        SampleLayout::Int { store: 24, .. } => {
            // Packed: the low three bytes of the little-endian word, sign included.
            for &v in samples {
                out.extend(&v.to_le_bytes()[..3]);
            }
        }
        SampleLayout::Int { valid: 24, .. } => {
            // 24 valid bits in a 32-bit container must be left-justified (MSB aligned).
            for &v in samples {
                out.extend((v << 8).to_le_bytes());
            }
        }
        SampleLayout::Int { .. } => {
            for &v in samples {
                out.extend(v.to_le_bytes());
            }
        }
        SampleLayout::Float32 => {
            // Full scale of a `bits`-bit word maps to 1.0. Computed in f64 so 32-bit words
            // only lose what f32 itself cannot hold.
            let scale = 1.0 / (1u64 << (bits.clamp(1, 32) - 1)) as f64;
            for &v in samples {
                out.extend(((v as f64 * scale) as f32).to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shorthand for an integer layout in tests.
    fn int(store: u16, valid: u16) -> SampleLayout {
        SampleLayout::Int { store, valid }
    }

    #[test]
    fn candidates_start_with_the_wanted_depth_and_cover_every_container() {
        // The Realtek case: 24 bits wanted, the 32-bit container refused, packed 24 next.
        assert_eq!(
            exclusive_candidates(24),
            vec![int(32, 24), int(24, 24), int(32, 32), int(16, 16)]
        );
        assert_eq!(
            exclusive_candidates(16),
            vec![int(16, 16), int(32, 32), int(32, 24), int(24, 24)]
        );
        assert_eq!(
            exclusive_candidates(32),
            vec![int(32, 32), int(32, 24), int(24, 24), int(16, 16)]
        );
    }

    #[test]
    fn integer_layouts_pack_bit_exactly() {
        let mut b = VecDeque::new();
        encode(&[-2], 16, int(16, 16), &mut b);
        assert_eq!(b.drain(..).collect::<Vec<_>>(), [0xFE, 0xFF]);
        // Packed 24: three bytes, sign carried in the top byte.
        encode(&[0x12_3456, -1], 24, int(24, 24), &mut b);
        assert_eq!(
            b.drain(..).collect::<Vec<_>>(),
            [0x56, 0x34, 0x12, 0xFF, 0xFF, 0xFF]
        );
        // 24 in 32: shifted up a byte.
        encode(&[0x12_3456], 24, int(32, 24), &mut b);
        assert_eq!(b.drain(..).collect::<Vec<_>>(), [0x00, 0x56, 0x34, 0x12]);
        encode(&[i32::MIN], 32, int(32, 32), &mut b);
        assert_eq!(b.drain(..).collect::<Vec<_>>(), [0, 0, 0, 0x80]);
    }

    #[test]
    fn float_scales_full_scale_to_one() {
        let mut b = VecDeque::new();
        encode(&[-32768, 16384], 16, SampleLayout::Float32, &mut b);
        let bytes: Vec<u8> = b.drain(..).collect();
        let f: Vec<f32> = bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(f, [-1.0, 0.5]);
        // 24-bit words survive the round trip exactly.
        encode(&[8_388_607], 24, SampleLayout::Float32, &mut b);
        let bytes: Vec<u8> = b.drain(..).collect();
        let f = f32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert_eq!((f as f64 * 8_388_608.0) as i32, 8_388_607);
    }

    #[test]
    fn layout_sizes() {
        assert_eq!(int(24, 24).bytes_per_sample(), 3);
        assert_eq!(SampleLayout::int_for_bits(24).bytes_per_sample(), 4);
        assert_eq!(SampleLayout::Float32.bytes_per_sample(), 4);
        assert_eq!(int(32, 24).valid_bits(), Some(24));
        assert_eq!(SampleLayout::Float32.valid_bits(), None);
    }
}
