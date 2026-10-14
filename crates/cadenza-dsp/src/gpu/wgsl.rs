//! The portable GPU kernels (WGSL, run through `wgpu` on Vulkan, Metal and DX12) and a CPU
//! model of them.
//!
//! The kernels are the CUDA polyphase upsampler's, rewritten for WGSL: one forward
//! Stockham transform per input block into a frequency-delay line, one multiply-accumulate
//! over every packed pair of sub-filters, the batched inverse transforms and the
//! interleaving into output order. Two differences from `kernels.cu`:
//!
//! * **Twiddles from a table.** WGSL has no `f64` `sin`/`cos`, so the twiddle factors
//!   `exp(−2πi·m/n)` are computed once on the CPU in `f64` and uploaded.
//! * **Two arithmetics.** Many GPUs have no `f64` at all (Apple Silicon, Intel Arc) and
//!   WGSL only offers it as an optional feature. The kernels are written against a scalar
//!   type `R` and its operations (`r_add`, `r_mul`, …); the prelude defines them either as
//!   native `f64` or as **double-f32**: each value is an unevaluated sum `hi + lo` of two
//!   `f32` (Dekker / Knuth error-free transformations, the classic "double-single"
//!   arithmetic), which carries about 48 bits of mantissa – rounding noise around
//!   −280 dB, far below any DAC, though not bit-identical to the CPU's `f64`. Both take 8
//!   bytes per value, so the buffers have the same size and layout either way.
//!
//! Double-f32 relies on the GPU computing `a + b` and `a · b` in IEEE `f32` without
//! reassociating them (the exact product uses Dekker's split, not `fma`, because some
//! drivers – Mesa's lavapipe for one – do not fuse `fma`). Shader compilers that optimise algebraically (Metal's
//! default fast-math) would cancel the error terms, so every error term goes through
//! `opq(x) = x · one`, with `one` a uniform the compiler cannot see is 1. The backend
//! still runs a self-test against the CPU when it opens a device and refuses a device
//! whose result is not accurate enough.
//!
//! [`WgslPolyphase`] runs the same kernels on the CPU, one "thread" at a time, in either
//! arithmetic ([`Df64`] mirrors the WGSL double-f32 operation by operation), so the
//! algorithm and its accuracy are tested on machines without a GPU.

use super::reference::{check_polyphase, pair_spectra};
use super::{GpuError, PolyphaseFactory};
use crate::convolution::{PolyphaseBlockConvolver, SpectralFilter};
use std::fmt::Debug;
use std::sync::Arc;

/// Threads per workgroup of every kernel.
pub const WORKGROUP: u32 = 256;

/// Most workgroups per dispatch dimension (the WebGPU default limit is 65535); larger
/// launches use a second dimension.
pub const MAX_GROUPS_X: u32 = 32_768;

/// Every entry point, in pipeline order.
pub const ENTRY_POINTS: [&str; 6] = [
    "pack_real",
    "radix2",
    "radix4",
    "copy_complex",
    "poly_fdl_mac",
    "poly_interleave",
];

/// How the kernels compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arith {
    /// Native `f64` (`SHADER_F64`): the same 64-bit maths as the CPU.
    F64,
    /// Two `f32` per value, about 48 bits of mantissa.
    DoubleF32,
}

impl Arith {
    /// `64-bit` or `double-f32 (≈48-bit)`, for the signal path.
    pub fn label(self) -> &'static str {
        match self {
            Arith::F64 => "64-bit",
            Arith::DoubleF32 => "double-f32 (≈48-bit)",
        }
    }

    /// `f64` or `double-f32`, for reports.
    pub fn id(self) -> &'static str {
        match self {
            Arith::F64 => "f64",
            Arith::DoubleF32 => "double-f32",
        }
    }
}

/// `(groups_x, groups_y)` for `items` work items.
pub fn dispatch_size(items: usize) -> (u32, u32) {
    let groups = (items as u64).div_ceil(WORKGROUP as u64).max(1);
    let x = groups.min(MAX_GROUPS_X as u64);
    (x as u32, groups.div_ceil(x) as u32)
}

/// Per-dispatch parameters, exactly the WGSL `Params` struct (16 × 4 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Params {
    /// Transform size (complex points).
    pub n: u32,
    /// Sub-transform size completed before this pass.
    pub ns: u32,
    /// −1 forward, +1 inverse.
    pub dir: i32,
    /// Transforms in this launch.
    pub batch: u32,
    /// Work items of this launch (threads beyond it return at once).
    pub items: u32,
    /// Twiddle stride: `n / (radix · ns)`.
    pub tws: u32,
    /// Spectra in the delay line so far.
    pub depth: u32,
    /// Newest delay-line slot.
    pub head: u32,
    /// Delay-line slots.
    pub slots: u32,
    /// Sub-filter pairs.
    pub pairs: u32,
    /// Input block (frames).
    pub block: u32,
    /// Phases `L`.
    pub l: u32,
    /// Output scale (`1/n`, a power of two: exact in `f32`).
    pub scale: f32,
    /// Always 1.0: the opaque factor of the double-f32 error terms.
    pub one: f32,
    /// Delay-line slot `copy_complex` writes.
    pub slot: u32,
    /// Partitions per sub-filter (stride of the spectra).
    pub parts: u32,
}

impl Params {
    /// Size of the struct on the device.
    pub const BYTES: usize = 64;

    /// The 64 bytes the uniform buffer holds.
    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let words = [
            self.n,
            self.ns,
            self.dir as u32,
            self.batch,
            self.items,
            self.tws,
            self.depth,
            self.head,
            self.slots,
            self.pairs,
            self.block,
            self.l,
            self.scale.to_bits(),
            self.one.to_bits(),
            self.slot,
            self.parts,
        ];
        let mut out = [0u8; Self::BYTES];
        for (chunk, w) in out.chunks_exact_mut(4).zip(words) {
            chunk.copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

/// One dispatch of the per-block program: which entry point, its parameters, its size,
/// and whether it reads buffer A (`x = A, y = B`) or buffer B (`x = B, y = A`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dispatch {
    /// Index into [`ENTRY_POINTS`].
    pub entry: usize,
    /// Parameters.
    pub params: Params,
    /// `true`: `x = A, y = B`; `false`: `x = B, y = A`.
    pub x_is_a: bool,
}

/// The program for one input block of a polyphase core, as the device runs it (and as
/// [`WgslPolyphase`] runs it on the CPU): `head` is the newest delay-line slot and
/// `depth` the spectra in it, after this block was added.
pub fn block_program(
    block: usize,
    l: usize,
    parts: usize,
    head: usize,
    depth: usize,
) -> Vec<Dispatch> {
    let n = 2 * block;
    let pairs = l.div_ceil(2);
    let base = Params {
        n: n as u32,
        dir: -1,
        batch: 1,
        depth: depth as u32,
        head: head as u32,
        slots: parts as u32,
        pairs: pairs as u32,
        block: block as u32,
        l: l as u32,
        scale: 1.0 / n as f32,
        one: 1.0,
        slot: head as u32,
        parts: parts as u32,
        ..Params::default()
    };
    let mut prog = Vec::new();
    // pack_real: IO → y (B); the transform input is then in B.
    prog.push(Dispatch {
        entry: 0,
        params: Params {
            items: n as u32,
            ..base
        },
        x_is_a: true,
    });
    let mut in_a = false;
    let passes = |prog: &mut Vec<Dispatch>, in_a: &mut bool, dir: i32, batch: usize| {
        for (radix, ns) in super::kernels::pass_schedule(n) {
            let r = radix as usize;
            prog.push(Dispatch {
                entry: if radix == 4 { 2 } else { 1 },
                params: Params {
                    ns: ns as u32,
                    dir,
                    batch: batch as u32,
                    items: (batch * n / r) as u32,
                    tws: (n / (r * ns)) as u32,
                    ..base
                },
                x_is_a: *in_a,
            });
            *in_a = !*in_a;
        }
    };
    passes(&mut prog, &mut in_a, -1, 1);
    // copy_complex: x → the newest delay-line slot.
    prog.push(Dispatch {
        entry: 3,
        params: Params {
            items: n as u32,
            ..base
        },
        x_is_a: in_a,
    });
    // poly_fdl_mac: delay line × spectra → y (into A).
    prog.push(Dispatch {
        entry: 4,
        params: Params {
            items: (pairs * n) as u32,
            ..base
        },
        x_is_a: false,
    });
    in_a = true;
    passes(&mut prog, &mut in_a, 1, pairs);
    // poly_interleave: x → IO.
    prog.push(Dispatch {
        entry: 5,
        params: Params {
            items: (block * l) as u32,
            ..base
        },
        x_is_a: in_a,
    });
    prog
}

/// The scalar the kernels compute with: `f64`, or [`Df64`] (double-f32). The operations
/// are exactly those of the WGSL prelude of the same arithmetic.
pub trait Real: Copy + Default + Debug + Send + Sync + 'static {
    /// The arithmetic this type models.
    const ARITH: Arith;
    /// Round a CPU value to this type (how the host uploads it).
    fn from_f64(x: f64) -> Self;
    /// The value as `f64` (how the host reads it back).
    fn to_f64(self) -> f64;
    /// `r_add`.
    fn add(self, o: Self) -> Self;
    /// `r_sub`.
    fn sub(self, o: Self) -> Self;
    /// `r_mul`.
    fn mul(self, o: Self) -> Self;
    /// `r_neg`.
    fn neg(self) -> Self;
    /// `r_fma(a, b, c) = a·b + c`.
    fn fma(self, b: Self, c: Self) -> Self;
}

impl Real for f64 {
    const ARITH: Arith = Arith::F64;
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn sub(self, o: Self) -> Self {
        self - o
    }
    fn mul(self, o: Self) -> Self {
        self * o
    }
    fn neg(self) -> Self {
        -self
    }
    fn fma(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
}

/// A double-f32 value: the unevaluated sum `hi + lo` with `|lo| ≤ ulp(hi)/2`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Df64 {
    /// Leading part.
    pub hi: f32,
    /// Trailing error.
    pub lo: f32,
}

/// `s + e = a + b` exactly (Knuth).
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// `s + e = a + b` exactly, for `|a| ≥ |b|` (Dekker).
fn quick_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    (s, b - (s - a))
}

/// Veltkamp split: `a = hi + lo`, 12 significant bits each.
fn split(a: f32) -> (f32, f32) {
    let t = 4097.0 * a;
    let hi = t - (t - a);
    (hi, a - hi)
}

/// `p + e = a · b` exactly (Dekker; needs no fused multiply-add).
fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

impl Real for Df64 {
    const ARITH: Arith = Arith::DoubleF32;
    fn from_f64(x: f64) -> Self {
        let hi = x as f32;
        Df64 {
            hi,
            lo: (x - hi as f64) as f32,
        }
    }
    fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }
    fn add(self, o: Self) -> Self {
        let (s, e) = two_sum(self.hi, o.hi);
        let (t, f) = two_sum(self.lo, o.lo);
        let (s, e) = quick_two_sum(s, e + t);
        let (hi, lo) = quick_two_sum(s, e + f);
        Df64 { hi, lo }
    }
    fn sub(self, o: Self) -> Self {
        self.add(o.neg())
    }
    fn mul(self, o: Self) -> Self {
        let (p, e) = two_prod(self.hi, o.hi);
        let (hi, lo) = quick_two_sum(p, e + (self.hi * o.lo + self.lo * o.hi));
        Df64 { hi, lo }
    }
    fn neg(self) -> Self {
        Df64 {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
    fn fma(self, b: Self, c: Self) -> Self {
        self.mul(b).add(c)
    }
}

/// `values` as the device stores them in `arith` (8 bytes each, little endian).
pub fn to_device_bytes(values: &[f64], arith: Arith) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 * values.len());
    for &v in values {
        match arith {
            Arith::F64 => out.extend_from_slice(&v.to_le_bytes()),
            Arith::DoubleF32 => {
                let d = Df64::from_f64(v);
                out.extend_from_slice(&d.hi.to_le_bytes());
                out.extend_from_slice(&d.lo.to_le_bytes());
            }
        }
    }
    out
}

/// Device bytes (8 per value) back to `f64`, into `out`.
pub fn from_device_bytes(bytes: &[u8], arith: Arith, out: &mut [f64]) {
    debug_assert_eq!(bytes.len(), 8 * out.len());
    for (o, b) in out.iter_mut().zip(bytes.chunks_exact(8)) {
        *o = match arith {
            Arith::F64 => f64::from_le_bytes(b.try_into().expect("8 bytes")),
            Arith::DoubleF32 => {
                let hi = f32::from_le_bytes(b[..4].try_into().expect("4 bytes"));
                let lo = f32::from_le_bytes(b[4..].try_into().expect("4 bytes"));
                hi as f64 + lo as f64
            }
        };
    }
}

/// The twiddle table of an `n`-point transform, interleaved: `exp(−2πi·m/n)` for
/// `m = 0 … n−1`, computed in `f64`.
pub fn twiddles(n: usize) -> Vec<f64> {
    let mut tw = Vec::with_capacity(2 * n);
    for m in 0..n {
        let (s, c) = (2.0 * std::f64::consts::PI * m as f64 / n as f64).sin_cos();
        tw.push(c);
        tw.push(-s);
    }
    tw
}

/// The WGSL source in `arith`: the prelude defining `R` and its operations, then the
/// shared kernels.
pub fn source(arith: Arith) -> String {
    let prelude = match arith {
        Arith::F64 => PRELUDE_F64,
        Arith::DoubleF32 => PRELUDE_DF64,
    };
    format!("{HEADER}{prelude}{KERNELS}")
}

const HEADER: &str = r#"// Cadenza WGSL kernels: the polyphase FFT upsampler (generated by gpu/wgsl.rs).

struct Params {
    n: u32, ns: u32, dir: i32, batch: u32,
    items: u32, tws: u32, depth: u32, head: u32,
    slots: u32, pairs: u32, block: u32, l: u32,
    scale: f32, one: f32, slot: u32, parts: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
"#;

const PRELUDE_F64: &str = r#"
// Native 64-bit arithmetic.
alias R = f64;
alias RArr = array<f64>;
fn r_zero() -> R { return f64(0.0); }
fn r_from(x: f32) -> R { return f64(x); }
fn r_add(a: R, b: R) -> R { return a + b; }
fn r_sub(a: R, b: R) -> R { return a - b; }
fn r_mul(a: R, b: R) -> R { return a * b; }
fn r_neg(a: R) -> R { return -a; }
fn r_fma(a: R, b: R, c: R) -> R { return fma(a, b, c); }
"#;

const PRELUDE_DF64: &str = r#"
// Double-f32 arithmetic: value = x + y, about 48 bits of mantissa.
alias R = vec2<f32>;
alias RArr = array<vec2<f32>>;
// Opaque identity: keeps an optimising compiler from cancelling the error terms.
fn opq(x: f32) -> f32 { return x * P.one; }
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = opq(s) - a;
    return vec2<f32>(s, (a - (opq(s) - v)) + (b - v));
}
fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    return vec2<f32>(s, b - (opq(s) - a));
}
// Veltkamp split: a = hi + lo with 12 significant bits each, so their products are exact.
fn split(a: f32) -> vec2<f32> {
    let t = opq(4097.0 * a);
    let hi = t - opq(t - a);
    return vec2<f32>(hi, a - hi);
}
// Dekker's product: exact with or without a fused fma (some drivers do not fuse it).
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = opq(a * b);
    let x = split(a);
    let y = split(b);
    let e = ((x.x * y.x - p) + x.x * y.y + x.y * y.x) + x.y * y.y;
    return vec2<f32>(p, e);
}
fn r_zero() -> R { return vec2<f32>(0.0, 0.0); }
fn r_from(x: f32) -> R { return vec2<f32>(x, 0.0); }
fn r_add(a: R, b: R) -> R {
    let s = two_sum(a.x, b.x);
    let t = two_sum(a.y, b.y);
    let u = quick_two_sum(s.x, s.y + t.x);
    return quick_two_sum(u.x, u.y + t.y);
}
fn r_neg(a: R) -> R { return -a; }
fn r_sub(a: R, b: R) -> R { return r_add(a, -b); }
fn r_mul(a: R, b: R) -> R {
    let p = two_prod(a.x, b.x);
    return quick_two_sum(p.x, p.y + (a.x * b.y + a.y * b.x));
}
fn r_fma(a: R, b: R, c: R) -> R { return r_add(r_mul(a, b), c); }
"#;

const KERNELS: &str = r#"
// x: the buffer read, y: the buffer written (A/B swap between bind groups), fdl: the
// frequency-delay line, g: the packed sub-filter spectra, tw: twiddles, io: input frame
// in, interleaved output out. Complex values are interleaved (re, im) pairs of R.
@group(0) @binding(1) var<storage, read> x: RArr;
@group(0) @binding(2) var<storage, read_write> y: RArr;
@group(0) @binding(3) var<storage, read_write> fdl: RArr;
@group(0) @binding(4) var<storage, read> g: RArr;
@group(0) @binding(5) var<storage, read> tw: RArr;
@group(0) @binding(6) var<storage, read_write> io: RArr;

const WG: u32 = 256u;

struct C { re: R, im: R }

fn thread(w: vec3<u32>, nw: vec3<u32>, l: vec3<u32>) -> u32 {
    return (w.y * nw.x + w.x) * WG + l.x;
}
fn ldx(i: u32) -> C { return C(x[2u * i], x[2u * i + 1u]); }
fn sty(i: u32, c: C) { y[2u * i] = c.re; y[2u * i + 1u] = c.im; }
fn cadd(a: C, b: C) -> C { return C(r_add(a.re, b.re), r_add(a.im, b.im)); }
fn csub(a: C, b: C) -> C { return C(r_sub(a.re, b.re), r_sub(a.im, b.im)); }
// a · b, in the order of kernels.cu: (ar·br − ai·bi, ar·bi + ai·br).
fn cmul(a: C, b: C) -> C {
    return C(r_sub(r_mul(a.re, b.re), r_mul(a.im, b.im)), r_add(r_mul(a.re, b.im), r_mul(a.im, b.re)));
}
// exp(dir · 2πi · m / n) from the table of exp(−2πi · m / n).
fn twid(m: u32, dir: i32) -> C {
    let t = C(tw[2u * m], tw[2u * m + 1u]);
    if (dir > 0) {
        return C(t.re, r_neg(t.im));
    }
    return t;
}

// Real frame (n values in io) → complex (y).
@compute @workgroup_size(256)
fn pack_real(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
             @builtin(local_invocation_id) li: vec3<u32>) {
    let t = thread(w, nw, li);
    if (t >= P.items) { return; }
    y[2u * t] = io[t];
    y[2u * t + 1u] = r_zero();
}

// One Stockham radix-2 pass over `batch` n-point transforms (x → y).
@compute @workgroup_size(256)
fn radix2(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
          @builtin(local_invocation_id) li: vec3<u32>) {
    let t = thread(w, nw, li);
    if (t >= P.items) { return; }
    let half = P.n >> 1u;
    let b = t / half;
    let j = t - b * half;
    let off = b * P.n;
    let k = j & (P.ns - 1u);
    let wv = twid(k * P.tws, P.dir);
    let a = ldx(off + j);
    let tt = cmul(ldx(off + j + half), wv);
    let base = ((j - k) << 1u) + k;
    sty(off + base, cadd(a, tt));
    sty(off + base + P.ns, csub(a, tt));
}

// One Stockham radix-4 pass over `batch` n-point transforms (x → y).
@compute @workgroup_size(256)
fn radix4(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
          @builtin(local_invocation_id) li: vec3<u32>) {
    let t = thread(w, nw, li);
    if (t >= P.items) { return; }
    let quarter = P.n >> 2u;
    let b = t / quarter;
    let j = t - b * quarter;
    let off = b * P.n;
    let k = j & (P.ns - 1u);
    let m = k * P.tws;
    let v0 = ldx(off + j);
    let v1 = cmul(ldx(off + j + quarter), twid(m, P.dir));
    let v2 = cmul(ldx(off + j + 2u * quarter), twid(2u * m, P.dir));
    let v3 = cmul(ldx(off + j + 3u * quarter), twid(3u * m, P.dir));
    let a = cadd(v0, v2);
    let bb = csub(v0, v2);
    let c = cadd(v1, v3);
    let d = csub(v1, v3);
    // (dir · i) · d
    var dj = C(d.im, r_neg(d.re));
    if (P.dir > 0) {
        dj = C(r_neg(d.im), d.re);
    }
    let base = ((j - k) << 2u) + k;
    sty(off + base, cadd(a, c));
    sty(off + base + P.ns, cadd(bb, dj));
    sty(off + base + 2u * P.ns, csub(a, c));
    sty(off + base + 3u * P.ns, csub(bb, dj));
}

// x (n complex values) → delay-line slot `slot`.
@compute @workgroup_size(256)
fn copy_complex(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
                @builtin(local_invocation_id) li: vec3<u32>) {
    let t = thread(w, nw, li);
    if (t >= P.items) { return; }
    let d = 2u * (P.slot * P.n + t);
    fdl[d] = x[2u * t];
    fdl[d + 1u] = x[2u * t + 1u];
}

// y[q][k] = Σ_p X[t − p][k] · G_{q,p}[k] for every sub-filter pair q (pairs · n threads).
@compute @workgroup_size(256)
fn poly_fdl_mac(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
                @builtin(local_invocation_id) li: vec3<u32>) {
    let t = thread(w, nw, li);
    if (t >= P.items) { return; }
    let q = t / P.n;
    let k = t - q * P.n;
    var re = r_zero();
    var im = r_zero();
    var slot = P.head;
    for (var p = 0u; p < P.depth; p++) {
        let xi_ = 2u * (slot * P.n + k);
        let gi_ = 2u * ((q * P.parts + p) * P.n + k);
        let xr = fdl[xi_];
        let xi = fdl[xi_ + 1u];
        let hr = g[gi_];
        let hi = g[gi_ + 1u];
        re = r_fma(xr, hr, re);
        re = r_fma(r_neg(xi), hi, re);
        im = r_fma(xr, hi, im);
        im = r_fma(xi, hr, im);
        slot = slot + 1u;
        if (slot == P.slots) {
            slot = 0u;
        }
    }
    y[2u * t] = re;
    y[2u * t + 1u] = im;
}

// The valid half of every pair's inverse transform, scaled, in output order (x → io).
@compute @workgroup_size(256)
fn poly_interleave(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
                   @builtin(local_invocation_id) li: vec3<u32>) {
    let t = thread(w, nw, li);
    if (t >= P.items) { return; }
    let j = t / P.l;
    let p = t - j * P.l;
    let src = 2u * ((p >> 1u) * P.n + j + P.block) + (p & 1u);
    io[t] = r_mul(x[src], r_from(P.scale));
}
"#;

/// The buffers of one core, as on the device.
#[derive(Debug, Clone)]
struct Buffers<R> {
    a: Vec<R>,
    b: Vec<R>,
    fdl: Vec<R>,
    io: Vec<R>,
}

/// Run one dispatch of the program on the CPU, thread by thread (mirror of the WGSL).
fn run_dispatch<R: Real>(d: &Dispatch, bufs: &mut Buffers<R>, g: &[R], tw: &[R]) {
    let p = d.params;
    let Buffers { a, b, fdl, io } = bufs;
    let (x, y): (&[R], &mut [R]) = if d.x_is_a {
        (&a[..], &mut b[..])
    } else {
        (&b[..], &mut a[..])
    };
    let twid = |m: usize, dir: i32| -> (R, R) {
        let (re, im) = (tw[2 * m], tw[2 * m + 1]);
        if dir > 0 {
            (re, im.neg())
        } else {
            (re, im)
        }
    };
    let ldx = |i: usize| (x[2 * i], x[2 * i + 1]);
    let cadd = |a: (R, R), b: (R, R)| (a.0.add(b.0), a.1.add(b.1));
    let csub = |a: (R, R), b: (R, R)| (a.0.sub(b.0), a.1.sub(b.1));
    let cmul = |a: (R, R), b: (R, R)| {
        (
            a.0.mul(b.0).sub(a.1.mul(b.1)),
            a.0.mul(b.1).add(a.1.mul(b.0)),
        )
    };
    let n = p.n as usize;
    let ns = p.ns as usize;
    for t in 0..p.items as usize {
        match d.entry {
            0 => {
                y[2 * t] = io[t];
                y[2 * t + 1] = R::default();
            }
            1 => {
                let half = n >> 1;
                let bi = t / half;
                let j = t - bi * half;
                let off = bi * n;
                let k = j & (ns - 1);
                let wv = twid(k * p.tws as usize, p.dir);
                let av = ldx(off + j);
                let tt = cmul(ldx(off + j + half), wv);
                let base = ((j - k) << 1) + k;
                let (s, u) = (cadd(av, tt), csub(av, tt));
                y[2 * (off + base)] = s.0;
                y[2 * (off + base) + 1] = s.1;
                y[2 * (off + base + ns)] = u.0;
                y[2 * (off + base + ns) + 1] = u.1;
            }
            2 => {
                let quarter = n >> 2;
                let bi = t / quarter;
                let j = t - bi * quarter;
                let off = bi * n;
                let k = j & (ns - 1);
                let m = k * p.tws as usize;
                let v0 = ldx(off + j);
                let v1 = cmul(ldx(off + j + quarter), twid(m, p.dir));
                let v2 = cmul(ldx(off + j + 2 * quarter), twid(2 * m, p.dir));
                let v3 = cmul(ldx(off + j + 3 * quarter), twid(3 * m, p.dir));
                let av = cadd(v0, v2);
                let bb = csub(v0, v2);
                let c = cadd(v1, v3);
                let dd = csub(v1, v3);
                let dj = if p.dir > 0 {
                    (dd.1.neg(), dd.0)
                } else {
                    (dd.1, dd.0.neg())
                };
                let base = ((j - k) << 2) + k;
                for (o, v) in [
                    (base, cadd(av, c)),
                    (base + ns, cadd(bb, dj)),
                    (base + 2 * ns, csub(av, c)),
                    (base + 3 * ns, csub(bb, dj)),
                ] {
                    y[2 * (off + o)] = v.0;
                    y[2 * (off + o) + 1] = v.1;
                }
            }
            3 => {
                let dst = 2 * (p.slot as usize * n + t);
                fdl[dst] = x[2 * t];
                fdl[dst + 1] = x[2 * t + 1];
            }
            4 => {
                let q = t / n;
                let k = t - q * n;
                let (mut re, mut im) = (R::default(), R::default());
                let mut slot = p.head as usize;
                for pp in 0..p.depth as usize {
                    let xi_ = 2 * (slot * n + k);
                    let gi_ = 2 * ((q * p.parts as usize + pp) * n + k);
                    let (xr, xi) = (fdl[xi_], fdl[xi_ + 1]);
                    let (hr, hi) = (g[gi_], g[gi_ + 1]);
                    re = xr.fma(hr, re);
                    re = xi.neg().fma(hi, re);
                    im = xr.fma(hi, im);
                    im = xi.fma(hr, im);
                    slot += 1;
                    if slot == p.slots as usize {
                        slot = 0;
                    }
                }
                y[2 * t] = re;
                y[2 * t + 1] = im;
            }
            5 => {
                let l = p.l as usize;
                let j = t / l;
                let ph = t - j * l;
                let src = 2 * ((ph >> 1) * n + j + p.block as usize) + (ph & 1);
                io[t] = x[src].mul(R::from_f64(p.scale as f64));
            }
            other => unreachable!("entry point {other}"),
        }
    }
}

/// The WGSL polyphase core run on the CPU in arithmetic `R`: the same program, the same
/// buffers, the same order of operations as on the device.
#[derive(Debug)]
pub struct WgslPolyphase<R: Real> {
    block: usize,
    l: usize,
    parts: usize,
    head: usize,
    count: usize,
    frame: Vec<f64>,
    bufs: Buffers<R>,
    g: Arc<Vec<R>>,
    tw: Arc<Vec<R>>,
}

impl<R: Real> WgslPolyphase<R> {
    /// Model the WGSL core for the `L = filters.len()` sub-filters.
    pub fn new(filters: &[SpectralFilter]) -> Result<Self, GpuError> {
        let (block, parts) = check_polyphase(filters)?;
        let g = Arc::new(pair_spectra(filters).into_iter().map(R::from_f64).collect());
        let tw = Arc::new(twiddles(2 * block).into_iter().map(R::from_f64).collect());
        Ok(Self::with_tables(g, tw, block, parts, filters.len()))
    }

    fn with_tables(g: Arc<Vec<R>>, tw: Arc<Vec<R>>, block: usize, parts: usize, l: usize) -> Self {
        let n = 2 * block;
        let work = l.div_ceil(2) * 2 * n;
        Self {
            block,
            l,
            parts,
            head: 0,
            count: 0,
            frame: vec![0.0; n],
            bufs: Buffers {
                a: vec![R::default(); work],
                b: vec![R::default(); work],
                fdl: vec![R::default(); parts * 2 * n],
                io: vec![R::default(); n.max(block * l)],
            },
            g,
            tw,
        }
    }
}

impl<R: Real> PolyphaseBlockConvolver for WgslPolyphase<R> {
    fn block(&self) -> usize {
        self.block
    }

    fn phases(&self) -> usize {
        self.l
    }

    fn process_block(&mut self, input: &[f64], out: &mut [f64]) -> Result<(), GpuError> {
        let (block, l) = (self.block, self.l);
        assert_eq!(input.len(), block);
        assert_eq!(out.len(), block * l);
        self.frame[block..].copy_from_slice(input);
        for (d, &v) in self.bufs.io.iter_mut().zip(&self.frame) {
            *d = R::from_f64(v);
        }
        self.head = (self.head + self.parts - 1) % self.parts;
        self.count = (self.count + 1).min(self.parts);
        for d in block_program(block, l, self.parts, self.head, self.count) {
            run_dispatch(&d, &mut self.bufs, &self.g, &self.tw);
        }
        for (o, v) in out.iter_mut().zip(&self.bufs.io) {
            *o = v.to_f64();
        }
        self.frame.copy_within(block.., 0);
        Ok(())
    }

    fn reset(&mut self) {
        self.frame.fill(0.0);
        self.bufs.fdl.fill(R::default());
        self.head = 0;
        self.count = 0;
    }

    fn device_label(&self) -> String {
        format!("WGSL model ({}, CPU)", R::ARITH.id())
    }
}

/// Builds [`WgslPolyphase`] cores in `arith`: the wgpu upsampler path of the resampler,
/// run on the CPU (what the tests compare against the CPU engine).
#[derive(Clone, Copy, Debug)]
pub struct WgslModelFactory {
    /// The arithmetic to model.
    pub arith: Arith,
}

impl PolyphaseFactory for WgslModelFactory {
    fn polyphase(
        &self,
        filters: &[SpectralFilter],
        channels: usize,
    ) -> Result<Vec<Box<dyn PolyphaseBlockConvolver + Send>>, GpuError> {
        fn build<R: Real>(
            filters: &[SpectralFilter],
            channels: usize,
        ) -> Result<Vec<Box<dyn PolyphaseBlockConvolver + Send>>, GpuError> {
            let first = WgslPolyphase::<R>::new(filters)?;
            let (g, tw) = (Arc::clone(&first.g), Arc::clone(&first.tw));
            let (block, parts, l) = (first.block, first.parts, first.l);
            let mut out: Vec<Box<dyn PolyphaseBlockConvolver + Send>> = vec![Box::new(first)];
            for _ in 1..channels {
                out.push(Box::new(WgslPolyphase::<R>::with_tables(
                    Arc::clone(&g),
                    Arc::clone(&tw),
                    block,
                    parts,
                    l,
                )));
            }
            Ok(out)
        }
        match self.arith {
            Arith::F64 => build::<f64>(filters, channels),
            Arith::DoubleF32 => build::<Df64>(filters, channels),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::reference::ReferencePolyphase;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn double_f32_carries_about_48_bits() {
        let mut s = 7u64;
        let mut worst: f64 = 0.0;
        for _ in 0..20_000 {
            let a = lcg(&mut s) * 10f64.powi((lcg(&mut s) * 6.0) as i32);
            let b = lcg(&mut s) * 10f64.powi((lcg(&mut s) * 6.0) as i32);
            let (da, db) = (Df64::from_f64(a), Df64::from_f64(b));
            // Rounding the inputs costs 2^-49 relative; each operation about as much again.
            for (got, want, scale) in [
                (da.add(db).to_f64(), a + b, a.abs() + b.abs()),
                (da.sub(db).to_f64(), a - b, a.abs() + b.abs()),
                (da.mul(db).to_f64(), a * b, (a * b).abs()),
                (da.fma(db, da).to_f64(), a * b + a, (a * b).abs() + a.abs()),
            ] {
                worst = worst.max((got - want).abs() / scale);
            }
        }
        assert!(worst < 2f64.powi(-45), "worst relative error {worst:e}");
        assert!(
            worst > 0.0,
            "not bit-identical to f64: it is a different arithmetic"
        );
        // Round trip through the device layout.
        let v = [0.1, -1.0 / 3.0, 12345.678, 0.0];
        for arith in [Arith::F64, Arith::DoubleF32] {
            let bytes = to_device_bytes(&v, arith);
            assert_eq!(bytes.len(), 8 * v.len());
            let mut back = [0.0; 4];
            from_device_bytes(&bytes, arith, &mut back);
            for (p, q) in v.iter().zip(&back) {
                assert!(
                    (p - q).abs() <= p.abs() * 2f64.powi(-47),
                    "{arith:?} {p} {q}"
                );
            }
        }
        assert_eq!(to_device_bytes(&[1.5], Arith::F64), 1.5f64.to_le_bytes());
    }

    #[test]
    fn params_layout_matches_the_wgsl_struct() {
        let p = Params {
            n: 1,
            dir: -1,
            scale: 0.5,
            one: 1.0,
            parts: 9,
            ..Params::default()
        };
        let b = p.to_bytes();
        assert_eq!(&b[0..4], &1u32.to_le_bytes());
        assert_eq!(&b[8..12], &(-1i32).to_le_bytes());
        assert_eq!(&b[48..52], &0.5f32.to_le_bytes());
        assert_eq!(&b[52..56], &1.0f32.to_le_bytes());
        assert_eq!(&b[60..64], &9u32.to_le_bytes());
        // The struct in the source has exactly these 16 fields of 4 bytes.
        let src = source(Arith::F64);
        let body = src
            .split("struct Params {")
            .nth(1)
            .unwrap()
            .split('}')
            .next()
            .unwrap();
        assert_eq!(body.matches(':').count(), 16);
        assert_eq!(dispatch_size(1), (1, 1));
        assert_eq!(
            dispatch_size(256 * MAX_GROUPS_X as usize),
            (MAX_GROUPS_X, 1)
        );
        assert_eq!(
            dispatch_size(256 * MAX_GROUPS_X as usize + 1),
            (MAX_GROUPS_X, 2)
        );
    }

    #[test]
    fn program_ping_pongs_consistently() {
        for (block, l) in [(4, 1), (8, 3), (64, 16), (512, 5)] {
            let prog = block_program(block, l, 3, 1, 2);
            // pack writes B; each pass reads what the previous one wrote.
            assert!(prog[0].x_is_a);
            let mut written_to_a = false; // pack wrote B
            for d in &prog[1..] {
                match d.entry {
                    1 | 2 | 3 | 5 => assert_eq!(d.x_is_a, written_to_a, "{block}/{l}: {d:?}"),
                    _ => {}
                }
                match d.entry {
                    1 | 2 => written_to_a = !d.x_is_a,
                    4 => written_to_a = !d.x_is_a,
                    _ => {}
                }
            }
            assert_eq!(prog.last().unwrap().entry, 5);
        }
    }

    fn subfilters(l: usize, block: usize, taps: usize) -> Vec<SpectralFilter> {
        let h: Vec<f64> = (0..taps)
            .map(|i| ((i as f64) * 0.011).sin() / (1.0 + i as f64 * 0.003))
            .collect();
        (0..l)
            .map(|p| {
                let s: Vec<f64> = h.iter().skip(p).step_by(l).copied().collect();
                SpectralFilter::new(&s, block)
            })
            .collect()
    }

    fn compare<R: Real>(l: usize, block: usize, taps: usize, tol: f64) -> f64 {
        let subs = subfilters(l, block, taps);
        let mut model = WgslPolyphase::<R>::new(&subs).unwrap();
        let mut reference = ReferencePolyphase::new(&subs).unwrap();
        let mut want = vec![0.0; block * l];
        let mut got = vec![0.0; block * l];
        let mut worst: f64 = 0.0;
        for b in 0..10 {
            let x: Vec<f64> = (0..block)
                .map(|i| ((b * block + i) as f64 * 0.37).cos() * 0.9)
                .collect();
            reference.process_block(&x, &mut want).unwrap();
            model.process_block(&x, &mut got).unwrap();
            let peak = want.iter().fold(1e-30f64, |m, v| m.max(v.abs()));
            for (p, q) in want.iter().zip(&got) {
                worst = worst.max((p - q).abs() / peak);
            }
        }
        assert!(worst < tol, "{:?} L={l} block={block}: {worst:e}", R::ARITH);
        model.reset();
        reference.reset();
        worst
    }

    #[test]
    fn wgsl_model_matches_the_cuda_model_in_f64() {
        for (l, block, taps) in [
            (1, 64, 300),
            (2, 128, 2000),
            (5, 256, 8 * 1024 + 3),
            (16, 32, 4096),
        ] {
            compare::<f64>(l, block, taps, 1e-12);
        }
    }

    #[test]
    fn wgsl_model_in_double_f32_is_within_48_bit_rounding() {
        let mut worst: f64 = 0.0;
        for (l, block, taps) in [
            (1, 64, 300),
            (2, 128, 2000),
            (5, 256, 8 * 1024 + 3),
            (16, 32, 4096),
        ] {
            worst = worst.max(compare::<Df64>(l, block, taps, 1e-11));
        }
        // Clearly better than f32 (≈1e-7) and clearly not f64 either.
        assert!(worst > 1e-17 && worst < 1e-11, "{worst:e}");
    }

    #[test]
    fn resampler_through_the_wgsl_model_matches_the_cpu_engine() {
        use crate::cache::CoefficientCache;
        use crate::fir::FilterKind;
        use crate::resampler::Resampler;
        let cache = CoefficientCache::disabled();
        let kind = FilterKind::parse("sinc-m-16k").unwrap_or_else(|| FilterKind::catalog()[0]);
        for (arith, tol) in [(Arith::F64, 1e-12), (Arith::DoubleF32, 1e-10)] {
            let factory = WgslModelFactory { arith };
            for (inr, outr) in [(44_100, 88_200), (44_100, 352_800), (16_000, 48_000)] {
                let mut cpu = Resampler::new(inr, outr, 2, kind, 150.0, &cache).unwrap();
                let mut gpu =
                    Resampler::new_on(inr, outr, 2, kind, 150.0, &cache, Some(&factory)).unwrap();
                let x: Vec<f64> = (0..2 * 6000)
                    .map(|i| ((i as f64) * 0.0137).sin() * 0.5)
                    .collect();
                let mut a = cpu.push(&x);
                a.extend(cpu.flush());
                let mut b = gpu.push(&x);
                b.extend(gpu.flush());
                assert_eq!(a.len(), b.len());
                let peak = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let err = a
                    .iter()
                    .zip(&b)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0.0, f64::max);
                assert!(err / peak < tol, "{arith:?} {inr}→{outr}: {err:e}");
            }
        }
    }

    #[test]
    fn sources_define_every_entry_point() {
        for arith in [Arith::F64, Arith::DoubleF32] {
            let src = source(arith);
            for e in ENTRY_POINTS {
                assert!(src.contains(&format!("fn {e}(")), "{arith:?}: {e}");
            }
        }
        assert!(!source(Arith::DoubleF32).contains("f64"));
    }
}
