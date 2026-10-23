//! On-disk cache of designed filter coefficients.
//!
//! A 4M-tap minimum-phase design takes seconds and a lot of memory bandwidth; the result
//! only depends on the [`FirSpec`], so it is written once to
//! `~/.cache/cadenza/fir/<key>.f64` (or the platform's cache directory) and read back
//! on every later start.

use crate::fir::FirSpec;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// File signature at the start of every cache file (format version 1).
const MAGIC: &[u8; 8] = b"CADENZA1";

/// Where designed coefficients are stored. `None` directory means "never touch the disk".
#[derive(Clone, Debug, Default)]
pub struct CoefficientCache {
    /// Cache directory; `None` keeps everything in memory.
    dir: Option<PathBuf>,
}

impl CoefficientCache {
    /// The platform cache directory (`~/.cache/cadenza/fir` on Linux), or in-memory only if
    /// the platform has none.
    pub fn default_location() -> Self {
        Self {
            dir: dirs::cache_dir().map(|d| d.join("cadenza").join("fir")),
        }
    }

    /// A cache in a specific directory.
    pub fn in_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: Some(dir.into()),
        }
    }

    /// A cache that never reads or writes files (every design is recomputed).
    pub fn disabled() -> Self {
        Self { dir: None }
    }

    /// The directory used, if any.
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Coefficients for `spec`: from disk when present and valid, otherwise designed now
    /// and stored for next time. Disk errors are logged and never fatal.
    pub fn get_or_design(&self, spec: &FirSpec) -> Vec<f64> {
        let path = self
            .dir
            .as_ref()
            .map(|d| d.join(format!("{}.f64", spec.cache_key())));
        if let Some(p) = &path {
            match read_coefficients(p, spec.taps) {
                Ok(Some(h)) => {
                    log::debug!("loaded {} taps from {}", h.len(), p.display());
                    return h;
                }
                Ok(None) => {}
                Err(e) => log::warn!("ignoring cache file {}: {e}", p.display()),
            }
        }
        log::info!("designing {} ({} taps)…", spec.cache_key(), spec.taps);
        let h = spec.design();
        if let Some(p) = &path {
            if let Err(e) = write_coefficients(p, &h) {
                log::warn!("could not write {}: {e}", p.display());
            }
        }
        h
    }
}

/// Read a cache file: `Ok(None)` when it does not exist, an error when the header is
/// wrong or the tap count differs from `expected_taps`. Layout: magic, `u64` LE tap
/// count, then the taps as `f64` LE.
fn read_coefficients(path: &Path, expected_taps: usize) -> io::Result<Option<Vec<f64>>> {
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut header = [0u8; 16];
    f.read_exact(&mut header)?;
    if &header[..8] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let len = u64::from_le_bytes(header[8..].try_into().unwrap()) as usize;
    if len != expected_taps {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tap count mismatch",
        ));
    }
    let mut bytes = vec![0u8; len * 8];
    f.read_exact(&mut bytes)?;
    Ok(Some(
        bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    ))
}

/// Write `h` in the format [`read_coefficients`] reads, atomically (temp file + rename).
fn write_coefficients(path: &Path, h: &[f64]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // Write to a sibling and rename so a crash never leaves a half-written cache file.
    let tmp = path.with_extension("tmp");
    {
        let mut f = io::BufWriter::new(fs::File::create(&tmp)?);
        f.write_all(MAGIC)?;
        f.write_all(&(h.len() as u64).to_le_bytes())?;
        for v in h {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()?;
    }
    fs::rename(tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fir::Phase;

    #[test]
    fn stores_and_reloads_identical_coefficients() {
        let dir = std::env::temp_dir().join(format!("cadenza-cache-test-{}", std::process::id()));
        let cache = CoefficientCache::in_dir(&dir);
        let spec = FirSpec::from_taps(1 << 10, 0.2, 100.0, Phase::Linear);
        let a = cache.get_or_design(&spec);
        assert!(dir.join(format!("{}.f64", spec.cache_key())).exists());
        let b = cache.get_or_design(&spec);
        assert_eq!(a, b);
        let _ = fs::remove_dir_all(dir);
    }
}
