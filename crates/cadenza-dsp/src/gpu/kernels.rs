//! The NVRTC kernel source and a host-side sanity check of it.
//!
//! The kernels are compiled on the user's machine by the driver's NVRTC, so the build
//! never needs a CUDA toolkit – which also means the source cannot be compiled on a
//! machine without an NVIDIA card. [`check_source`] is the next best thing: it verifies
//! that the brackets balance (outside comments and string literals), that the file is
//! wrapped in `extern "C"` so the kernel names are not mangled, and that every kernel the
//! host code launches is defined as `__global__ void <name>(`.

/// The CUDA C source of every kernel (`kernels.cu`).
pub const KERNEL_SOURCE: &str = include_str!("kernels.cu");

/// Name of the NVRTC program (shows up in compile logs).
pub const PROGRAM_NAME: &str = "cadenza_kernels.cu";

/// Every kernel the host launches, in pipeline order: the convolver's six, then the
/// polyphase upsampler's own (it shares `pack_real`, the single transforms and
/// `copy_complex` with the convolver), then the rational resampler's.
pub const KERNEL_NAMES: [&str; 11] = [
    "pack_real",
    "stockham_radix2",
    "stockham_radix4",
    "copy_complex",
    "fdl_mac",
    "unpack_real_scaled",
    "stockham_radix2_batch",
    "stockham_radix4_batch",
    "poly_fdl_mac",
    "poly_interleave",
    "rational_dot",
];

/// Threads per block for every launch (all kernels are one-dimensional and embarrassingly
/// parallel, so the only requirement is a multiple of the warp size).
pub const THREADS_PER_BLOCK: u32 = 256;

/// Syntax-level check of a kernel source: balanced `()`, `[]`, `{}` outside comments and
/// strings, an `extern "C"` block, and a `__global__ void <name>(` definition for each of
/// `names`. Returns a human-readable description of the first problem.
pub fn check_source(src: &str, names: &[&str]) -> Result<(), String> {
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut chars = src.chars().enumerate().peekable();
    let mut line = 1usize;
    while let Some((_, c)) = chars.next() {
        match c {
            '\n' => line += 1,
            '/' if chars.peek().is_some_and(|&(_, n)| n == '/') => {
                for (_, n) in chars.by_ref() {
                    if n == '\n' {
                        line += 1;
                        break;
                    }
                }
            }
            '/' if chars.peek().is_some_and(|&(_, n)| n == '*') => {
                chars.next();
                let mut prev = ' ';
                let mut closed = false;
                for (_, n) in chars.by_ref() {
                    if n == '\n' {
                        line += 1;
                    }
                    if prev == '*' && n == '/' {
                        closed = true;
                        break;
                    }
                    prev = n;
                }
                if !closed {
                    return Err(format!("line {line}: unterminated block comment"));
                }
            }
            '"' | '\'' => {
                let quote = c;
                let mut escaped = false;
                let mut closed = false;
                for (_, n) in chars.by_ref() {
                    if escaped {
                        escaped = false;
                    } else if n == '\\' {
                        escaped = true;
                    } else if n == quote {
                        closed = true;
                        break;
                    } else if n == '\n' {
                        break;
                    }
                }
                if !closed {
                    return Err(format!("line {line}: unterminated string literal"));
                }
            }
            '(' | '[' | '{' => stack.push((c, line)),
            ')' | ']' | '}' => {
                let want = match c {
                    ')' => '(',
                    ']' => '[',
                    _ => '{',
                };
                match stack.pop() {
                    Some((open, _)) if open == want => {}
                    Some((open, at)) => {
                        return Err(format!(
                            "line {line}: `{c}` closes `{open}` opened on line {at}"
                        ))
                    }
                    None => return Err(format!("line {line}: unmatched `{c}`")),
                }
            }
            _ => {}
        }
    }
    if let Some((open, at)) = stack.pop() {
        return Err(format!("`{open}` opened on line {at} is never closed"));
    }
    if !src.contains("extern \"C\"") {
        return Err("no `extern \"C\"` block: kernel names would be mangled".into());
    }
    for name in names {
        let def = format!("__global__ void {name}(");
        if !src.contains(&def) {
            return Err(format!("kernel `{name}` is not defined (expected `{def}`)"));
        }
    }
    Ok(())
}

/// Grid size for `n` one-dimensional work items.
pub fn grid_for(n: usize) -> u32 {
    (n as u32).div_ceil(THREADS_PER_BLOCK).max(1)
}

/// The Stockham pass schedule for an `n`-point transform (`n` a power of two ≥ 2): each
/// entry is `(radix, ns)` with `ns` the size of the sub-transforms completed before the pass.
/// Radix 4 wherever possible, one radix-2 pass at the end when `log2(n)` is odd.
pub fn pass_schedule(n: usize) -> Vec<(u32, usize)> {
    assert!(
        n.is_power_of_two() && n >= 2,
        "FFT size must be a power of two, got {n}"
    );
    let mut passes = Vec::new();
    let mut ns = 1;
    while ns < n {
        if ns * 4 <= n {
            passes.push((4, ns));
            ns *= 4;
        } else {
            passes.push((2, ns));
            ns *= 2;
        }
    }
    passes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_is_balanced_and_defines_every_kernel() {
        check_source(KERNEL_SOURCE, &KERNEL_NAMES).unwrap();
        assert!(KERNEL_SOURCE.contains("extern \"C\""));
        // 64-bit only: no float literals or float math anywhere in the kernels.
        for line in KERNEL_SOURCE.lines() {
            let code = line.split("//").next().unwrap();
            assert!(!code.contains("float"), "f32 in kernel source: {line}");
            assert!(
                !code.contains("cosf") && !code.contains("sinf"),
                "f32 math: {line}"
            );
        }
    }

    #[test]
    fn checker_catches_broken_sources() {
        assert!(check_source("extern \"C\" { __global__ void a( { }", &["a"]).is_err());
        assert!(check_source("extern \"C\" { __global__ void a() { } ", &["a"]).is_err());
        assert!(check_source("extern \"C\" { __global__ void a() { } }", &["b"]).is_err());
        assert!(check_source("__global__ void a() { }", &["a"]).is_err());
        // Brackets inside comments and strings do not count.
        assert!(check_source(
            "extern \"C\" { // (((\n /* { */ __global__ void a() { const char* s = \"}\"; } }",
            &["a"]
        )
        .is_ok());
        assert!(check_source("extern \"C\" { /* never closed", &[]).is_err());
    }

    #[test]
    fn pass_schedules_cover_the_transform() {
        for log2 in 1..=16 {
            let n = 1usize << log2;
            let passes = pass_schedule(n);
            let product: usize = passes.iter().map(|&(r, _)| r as usize).product();
            assert_eq!(product, n);
            let radix2 = passes.iter().filter(|&&(r, _)| r == 2).count();
            assert_eq!(radix2, log2 % 2, "{n}: {passes:?}");
        }
        assert_eq!(grid_for(1), 1);
        assert_eq!(grid_for(256), 1);
        assert_eq!(grid_for(257), 2);
    }
}
