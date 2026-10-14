// Cadenza CUDA kernels: 64-bit partitioned convolution and the polyphase FFT upsampler.
//
// Compiled at runtime with NVRTC (the driver ships it; no CUDA toolkit is needed), so this
// file is plain CUDA C with no includes. Everything is `double`; complex numbers are
// interleaved pairs (re, im) in `double` arrays, which keeps the kernels free of any
// vector-type header and makes the host-side model in `reference.rs` a line-by-line mirror.
//
// FFT: Stockham autosort, radix-4 passes with a final radix-2 pass when log2(n) is odd.
// Every pass reads one buffer and writes the other (ping-pong); `ns` is the size of the
// sub-transforms already completed (1, 4, 16, …), `dir` is -1 for forward and +1 for inverse.
// The inverse is unscaled; `unpack_real_scaled` applies 1/n.

#define CADENZA_PI 3.14159265358979323846

extern "C" {

// Real block → complex (imaginary parts zero).
__global__ void pack_real(const double* __restrict__ in, double* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[2 * i] = in[i];
        out[2 * i + 1] = 0.0;
    }
}

// One Stockham radix-2 butterfly (work item j of n/2) of one n-point transform.
__device__ __forceinline__ void radix2_item(const double* __restrict__ in, double* __restrict__ out,
                                            int n, int ns, int dir, int j)
{
    int half = n >> 1;
    int k = j & (ns - 1);                       // j mod ns (ns is a power of two)
    double ang = (double)dir * CADENZA_PI * (double)k / (double)ns;   // 2π k / (2 ns)
    double wr = cos(ang);
    double wi = sin(ang);
    double ar = in[2 * j];
    double ai = in[2 * j + 1];
    double br = in[2 * (j + half)];
    double bi = in[2 * (j + half) + 1];
    double tr = br * wr - bi * wi;
    double ti = br * wi + bi * wr;
    int base = ((j - k) << 1) + k;              // (j / ns) * 2ns + k
    out[2 * base] = ar + tr;
    out[2 * base + 1] = ai + ti;
    out[2 * (base + ns)] = ar - tr;
    out[2 * (base + ns) + 1] = ai - ti;
}

// One Stockham radix-4 butterfly (work item j of n/4) of one n-point transform.
__device__ __forceinline__ void radix4_item(const double* __restrict__ in, double* __restrict__ out,
                                            int n, int ns, int dir, int j)
{
    int quarter = n >> 2;
    int k = j & (ns - 1);
    double ang = (double)dir * CADENZA_PI * (double)k / (2.0 * (double)ns);   // 2π k / (4 ns)
    double w1r = cos(ang),       w1i = sin(ang);
    double w2r = cos(2.0 * ang), w2i = sin(2.0 * ang);
    double w3r = cos(3.0 * ang), w3i = sin(3.0 * ang);

    double v0r = in[2 * j],                 v0i = in[2 * j + 1];
    double x1r = in[2 * (j + quarter)],     x1i = in[2 * (j + quarter) + 1];
    double x2r = in[2 * (j + 2 * quarter)], x2i = in[2 * (j + 2 * quarter) + 1];
    double x3r = in[2 * (j + 3 * quarter)], x3i = in[2 * (j + 3 * quarter) + 1];

    double v1r = x1r * w1r - x1i * w1i, v1i = x1r * w1i + x1i * w1r;
    double v2r = x2r * w2r - x2i * w2i, v2i = x2r * w2i + x2i * w2r;
    double v3r = x3r * w3r - x3i * w3i, v3i = x3r * w3i + x3i * w3r;

    // 4-point DFT with sign `dir`: multiplying by (dir·i) maps (x, y) → (-dir·y, dir·x).
    double s = (double)dir;
    double a_r = v0r + v2r, a_i = v0i + v2i;    // v0 + v2
    double b_r = v0r - v2r, b_i = v0i - v2i;    // v0 - v2
    double c_r = v1r + v3r, c_i = v1i + v3i;    // v1 + v3
    double d_r = v1r - v3r, d_i = v1i - v3i;    // v1 - v3
    double dj_r = -s * d_i, dj_i = s * d_r;     // (dir·i)(v1 - v3)

    int base = ((j - k) << 2) + k;              // (j / ns) * 4ns + k
    out[2 * base] = a_r + c_r;                  out[2 * base + 1] = a_i + c_i;
    out[2 * (base + ns)] = b_r + dj_r;          out[2 * (base + ns) + 1] = b_i + dj_i;
    out[2 * (base + 2 * ns)] = a_r - c_r;       out[2 * (base + 2 * ns) + 1] = a_i - c_i;
    out[2 * (base + 3 * ns)] = b_r - dj_r;      out[2 * (base + 3 * ns) + 1] = b_i - dj_i;
}

// One Stockham radix-2 pass over n complex values (n/2 threads).
__global__ void stockham_radix2(const double* __restrict__ in, double* __restrict__ out,
                                int n, int ns, int dir)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < (n >> 1)) {
        radix2_item(in, out, n, ns, dir, j);
    }
}

// One Stockham radix-4 pass over n complex values (n/4 threads).
__global__ void stockham_radix4(const double* __restrict__ in, double* __restrict__ out,
                                int n, int ns, int dir)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < (n >> 2)) {
        radix4_item(in, out, n, ns, dir, j);
    }
}

// The radix-2 pass over `batch` independent n-point transforms stored back to back
// (batch · n/2 threads): the polyphase upsampler's inverse transforms, all in one launch.
__global__ void stockham_radix2_batch(const double* __restrict__ in, double* __restrict__ out,
                                      int n, int ns, int dir, int batch)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int half = n >> 1;
    if (t >= (long long)half * batch) {
        return;
    }
    int b = (int)(t / half);
    int j = (int)(t - (long long)b * half);
    long long off = 2LL * b * n;
    radix2_item(in + off, out + off, n, ns, dir, j);
}

// The radix-4 pass over `batch` transforms (batch · n/4 threads).
__global__ void stockham_radix4_batch(const double* __restrict__ in, double* __restrict__ out,
                                      int n, int ns, int dir, int batch)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int quarter = n >> 2;
    if (t >= (long long)quarter * batch) {
        return;
    }
    int b = (int)(t / quarter);
    int j = (int)(t - (long long)b * quarter);
    long long off = 2LL * b * n;
    radix4_item(in + off, out + off, n, ns, dir, j);
}

// Copy n complex values into slot `slot` of the frequency-delay line.
__global__ void copy_complex(const double* __restrict__ src, double* __restrict__ fdl, int n, int slot)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        fdl[2 * (slot * n + i)] = src[2 * i];
        fdl[2 * (slot * n + i) + 1] = src[2 * i + 1];
    }
}

// acc[k] = Σ_p X[t − p][k] · H_p[k] over the `depth` newest spectra; slot `head` is the
// newest, older ones follow it circularly in a ring of `slots` spectra.
__global__ void fdl_mac(const double* __restrict__ fdl, const double* __restrict__ h,
                        double* __restrict__ acc, int n, int depth, int head, int slots)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) {
        return;
    }
    double re = 0.0;
    double im = 0.0;
    int slot = head;
    for (int p = 0; p < depth; p++) {
        double xr = fdl[2 * (slot * n + k)];
        double xi = fdl[2 * (slot * n + k) + 1];
        double hr = h[2 * (p * n + k)];
        double hi = h[2 * (p * n + k) + 1];
        re = fma(xr, hr, re);
        re = fma(-xi, hi, re);
        im = fma(xr, hi, im);
        im = fma(xi, hr, im);
        slot = slot + 1;
        if (slot == slots) {
            slot = 0;
        }
    }
    acc[2 * k] = re;
    acc[2 * k + 1] = im;
}

// Overlap-save output: the real part of the second half of the 2·block-point inverse
// transform, scaled by `scale` (= 1 / (2·block)).
__global__ void unpack_real_scaled(const double* __restrict__ in, double* __restrict__ out,
                                   int block, double scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < block) {
        out[i] = in[2 * (i + block)] * scale;
    }
}

// Polyphase upsampler, multiply-accumulate for every sub-filter pair at once
// (pairs · n threads): acc[q][k] = Σ_p X[t − p][k] · G_{q,p}[k], where
// G_{q,p} = H_{2q,p} + i·H_{2q+1,p} packs two real sub-filters into one complex spectrum
// (the inverse transform of the sum then carries phase 2q in its real part and phase
// 2q + 1 in its imaginary part). `parts` is the partitions per sub-filter (the stride of
// `g`), `depth` the spectra in the delay line so far, `head` the newest of `slots`.
__global__ void poly_fdl_mac(const double* __restrict__ fdl, const double* __restrict__ g,
                             double* __restrict__ acc, int n, int depth, int head, int slots,
                             int parts, int pairs)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)pairs * n) {
        return;
    }
    int q = (int)(t / n);
    int k = (int)(t - (long long)q * n);
    double re = 0.0;
    double im = 0.0;
    int slot = head;
    for (int p = 0; p < depth; p++) {
        long long xi_ = 2LL * ((long long)slot * n + k);
        long long gi_ = 2LL * (((long long)q * parts + p) * n + k);
        double xr = fdl[xi_];
        double xi = fdl[xi_ + 1];
        double hr = g[gi_];
        double hi = g[gi_ + 1];
        re = fma(xr, hr, re);
        re = fma(-xi, hi, re);
        im = fma(xr, hi, im);
        im = fma(xi, hr, im);
        slot = slot + 1;
        if (slot == slots) {
            slot = 0;
        }
    }
    acc[2 * t] = re;
    acc[2 * t + 1] = im;
}

// Polyphase upsampler output (block · l threads): the valid second half of every pair's
// inverse transform, scaled, written in output order out[j·l + p] (phase p of input
// sample j), so the host copies back one ready, interleaved block.
__global__ void poly_interleave(const double* __restrict__ acc, double* __restrict__ out,
                                int block, int l, double scale)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)block * l) {
        return;
    }
    int j = (int)(t / l);
    int p = (int)(t - (long long)j * l);
    int n = 2 * block;
    long long src = 2LL * ((long long)(p >> 1) * n + j + block) + (p & 1);
    out[t] = acc[src] * scale;
}

// Rational resampler (44.1 <-> 48 kHz families), the time-domain polyphase bank: one block
// of 256 threads per output sample j (blockIdx.x). Output j sits at upsampled offset
// u = phase0 + j·m, uses reversed sub-filter p = u mod l (s taps, `sub` holds l·s) and the
// s inputs of `win` ending at first + u / l. Each thread sums the taps k ≡ thread (mod 256)
// with fma, then a tree reduction adds the partial sums (the order `rational_dot` in
// wgsl.rs models).
__global__ void rational_dot(const double* __restrict__ sub, const double* __restrict__ win,
                             double* __restrict__ out, int s, int l, int m, int phase0,
                             int first, int count)
{
    __shared__ double part[256];
    int j = blockIdx.x;
    if (j >= count) {
        return;                                 // the whole block leaves together
    }
    int th = threadIdx.x;
    long long u = (long long)phase0 + (long long)j * m;
    long long p = u % l;
    long long start = (long long)first + u / l + 1 - s;
    double acc = 0.0;
    for (int k = th; k < s; k += 256) {
        acc = fma(sub[p * s + k], win[start + k], acc);
    }
    part[th] = acc;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (th < stride) {
            part[th] = part[th] + part[th + stride];
        }
        __syncthreads();
    }
    if (th == 0) {
        out[j] = part[0];
    }
}

} // extern "C"
