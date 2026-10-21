//! Updates: look at the latest GitHub release, and on Windows download its installer and
//! run it hands-off (it closes Cadenza, installs, and starts the new version). Elsewhere
//! the release page opens in the browser.
//!
//! With hotfixes on (Settings → Appearance → Updates), the newest of the full releases and the
//! hotfix prereleases CI publishes on every push to main (`hotfix-3.0.6.17`) counts; a full
//! release is newer than every hotfix before it (3.0.7 > 3.0.6.17). Older downloaded
//! installers are deleted before a new one is fetched.

use serde::Deserialize;
use std::path::PathBuf;

const REPO: &str = "Krifmaz/Cadenza";
/// The tag prefix of the builds CI publishes on every push (.github/workflows/hotfix.yml).
const HOTFIX: &str = "hotfix-";

/// A newer release than this build.
#[derive(Clone, Debug)]
pub struct Update {
    pub version: String,
    /// The installer for this system, when the release has one.
    pub asset: Option<(String, String)>,
    pub page: String,
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    assets: Vec<Asset>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}

/// `3.0.12` → `(3, 0, 12, 0)`, a hotfix `hotfix-3.0.12.4` → `(3, 0, 12, 4)`; anything
/// unparsable counts as 0.
pub fn parse_version(v: &str) -> (u32, u32, u32, u32) {
    let mut p = v
        .trim()
        .trim_start_matches(HOTFIX)
        .trim_start_matches('v')
        .split(['.', '-', '+'])
        .map(|x| x.parse::<u32>().unwrap_or(0));
    (
        p.next().unwrap_or(0),
        p.next().unwrap_or(0),
        p.next().unwrap_or(0),
        p.next().unwrap_or(0),
    )
}

/// This build's version: the hotfix number when CI built it as one (`CADENZA_VERSION`).
pub fn current() -> &'static str {
    option_env!("CADENZA_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"))
}

/// The release to offer from GitHub's list: full releases, and with `hotfix` also the hotfix
/// prereleases; the highest version wins. Drafts and other prereleases never count.
fn pick(releases: Vec<Release>, hotfix: bool) -> Option<Release> {
    releases
        .into_iter()
        .filter(|r| !r.draft)
        .filter(|r| {
            let is_hotfix = r.tag_name.starts_with(HOTFIX);
            if r.prerelease {
                hotfix && is_hotfix
            } else {
                !is_hotfix
            }
        })
        .max_by_key(|r| parse_version(&r.tag_name))
}

fn get(url: &str) -> Result<ureq::Response, String> {
    ureq::get(url)
        .set(
            "User-Agent",
            concat!("cadenza-desktop/", env!("CARGO_PKG_VERSION")),
        )
        .set("Accept", "application/vnd.github+json")
        .timeout(std::time::Duration::from_secs(15))
        .call()
        .map_err(|e| e.to_string())
}

/// The installer that fits this system, from a release's file names.
fn pick_asset(names: &[String]) -> Option<usize> {
    let ends: &[&str] = if cfg!(windows) {
        &["-setup.exe"]
    } else if cfg!(target_os = "macos") {
        &[".dmg"]
    } else if std::env::var_os("APPIMAGE").is_some() {
        &[".AppImage"]
    } else {
        &[".deb"]
    };
    names
        .iter()
        .position(|n| ends.iter().any(|e| n.ends_with(e)))
}

/// Ask GitHub for the latest release (with `hotfix`, the newest release or hotfix);
/// `Ok(None)` when this build is up to date.
pub fn check(hotfix: bool) -> Result<Option<Update>, String> {
    let releases: Vec<Release> = if hotfix {
        get(&format!(
            "https://api.github.com/repos/{REPO}/releases?per_page=30"
        ))?
        .into_json()
        .map_err(|e| e.to_string())?
    } else {
        vec![get(&format!(
            "https://api.github.com/repos/{REPO}/releases/latest"
        ))?
        .into_json()
        .map_err(|e| e.to_string())?]
    };
    let Some(r) = pick(releases, hotfix) else {
        return Ok(None);
    };
    if parse_version(&r.tag_name) <= parse_version(current()) {
        return Ok(None);
    }
    let names: Vec<String> = r.assets.iter().map(|a| a.name.clone()).collect();
    let asset = pick_asset(&names).map(|i| {
        (
            r.assets[i].name.clone(),
            r.assets[i].browser_download_url.clone(),
        )
    });
    Ok(Some(Update {
        version: r
            .tag_name
            .trim_start_matches(HOTFIX)
            .trim_start_matches('v')
            .to_string(),
        asset,
        page: r.html_url,
    }))
}

/// Download the installer into the temp folder.
fn download(name: &str, url: &str) -> Result<PathBuf, String> {
    let resp = ureq::get(url)
        .set(
            "User-Agent",
            concat!("cadenza-desktop/", env!("CARGO_PKG_VERSION")),
        )
        .timeout(std::time::Duration::from_secs(600))
        .call()
        .map_err(|e| e.to_string())?;
    let dir = std::env::temp_dir().join("cadenza-update");
    // installers of earlier updates are no use any more
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(name);
    let part = path.with_extension("part");
    let mut f = std::fs::File::create(&part).map_err(|e| e.to_string())?;
    std::io::copy(&mut resp.into_reader(), &mut f).map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&part, &path).map_err(|e| e.to_string())?;
    Ok(path)
}

/// What [`install`] did.
pub enum Installed {
    /// The installer runs and will close this app: quit now.
    Quit,
    /// The release page opened in the browser.
    Opened,
}

/// Install `u`: on Windows the installer in passive mode (`/P`: progress only, closes the
/// running app; `/R`: starts the new version), elsewhere the release page.
pub fn install(u: &Update) -> Result<Installed, String> {
    if cfg!(windows) {
        if let Some((name, url)) = &u.asset {
            let path = download(name, url)?;
            std::process::Command::new(&path)
                .args(["/P", "/R"])
                .spawn()
                .map_err(|e| format!("cannot start the installer: {e}"))?;
            return Ok(Installed::Quit);
        }
    }
    open_url(&u.page);
    Ok(Installed::Opened)
}

pub fn open_url(url: &str) {
    #[cfg(windows)]
    let r = std::process::Command::new("explorer").arg(url).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).spawn();
    #[cfg(not(any(windows, target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = r {
        log::warn!("cannot open {url}: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_by_number() {
        assert!(parse_version("v3.0.10") > parse_version("3.0.9"));
        assert!(parse_version("3.1.0") > parse_version("3.0.99"));
        assert_eq!(parse_version("v3.0.5"), (3, 0, 5, 0));
        assert_eq!(parse_version("junk"), (0, 0, 0, 0));
        assert_eq!(parse_version("hotfix-3.0.6.17"), (3, 0, 6, 17));
        assert!(parse_version("hotfix-3.0.6.17") > parse_version("3.0.6"));
        assert!(parse_version("v3.0.7") > parse_version("hotfix-3.0.6.17"));
    }

    fn rel(tag: &str, prerelease: bool, draft: bool) -> Release {
        Release {
            tag_name: tag.into(),
            html_url: String::new(),
            assets: vec![],
            draft,
            prerelease,
        }
    }

    #[test]
    fn hotfixes_count_only_when_switched_on() {
        let list = || {
            vec![
                rel("hotfix-3.0.6.18", true, true),
                rel("hotfix-3.0.6.17", true, false),
                rel("v3.1.0-beta", true, false),
                rel("v3.0.6", false, false),
            ]
        };
        assert_eq!(pick(list(), true).unwrap().tag_name, "hotfix-3.0.6.17");
        assert_eq!(pick(list(), false).unwrap().tag_name, "v3.0.6");
        let mut l = list();
        l.push(rel("v3.0.7", false, false));
        assert_eq!(
            pick(l, true).unwrap().tag_name,
            "v3.0.7",
            "a full release beats older hotfixes"
        );
        assert!(pick(vec![], true).is_none());
    }

    #[test]
    fn picks_this_systems_installer() {
        let names: Vec<String> = [
            "cadenza-3.0.5-linux-x86_64.tar.gz",
            "cadenza-desktop_3.0.5_amd64.deb",
            "cadenza-desktop_3.0.5_x64-setup.exe",
            "cadenza-desktop_3.0.5_x86_64.AppImage",
            "Cadenza_3.0.5_aarch64.dmg",
            "SHA256SUMS",
        ]
        .map(String::from)
        .to_vec();
        let i = pick_asset(&names).expect("an installer");
        if cfg!(windows) {
            assert!(names[i].ends_with("-setup.exe"));
        } else if cfg!(target_os = "macos") {
            assert!(names[i].ends_with(".dmg"));
        }
    }
}
