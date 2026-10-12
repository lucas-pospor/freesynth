//! Paths stored in project files (relative to the project's directory, with `/` separators), and
//! the user's cache directory.

use std::path::{Component, Path, PathBuf};

/// `path` made absolute (against the current directory) with `.` and `..` resolved lexically.
pub fn absolute(path: &Path) -> PathBuf {
    let path = if path.as_os_str().is_empty() { Path::new(".") } else { path };
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_owned());
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// `path` relative to the directory `base` (both may be relative to the current directory).
/// Falls back to the absolute path when they share no root (different drives on Windows).
pub fn relative_to(path: &Path, base: &Path) -> PathBuf {
    let (path, base) = (absolute(path), absolute(base));
    let (p, b): (Vec<Component>, Vec<Component>) = (path.components().collect(), base.components().collect());
    if p.first() != b.first() {
        return path;
    }
    let common = p.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut out = PathBuf::new();
    for _ in common..b.len() {
        out.push("..");
    }
    for c in &p[common..] {
        out.push(c.as_os_str());
    }
    if out.as_os_str().is_empty() { PathBuf::from(".") } else { out }
}

/// A path as written in project files: `/` separators on every system.
pub fn to_project_string(path: &Path) -> String {
    let text = path.to_string_lossy();
    if cfg!(windows) { text.replace('\\', "/") } else { text.into_owned() }
}

/// `path` (relative to the current directory, or absolute) as a project-file path for a project
/// in directory `project_dir`.
pub fn for_project(path: &Path, project_dir: &Path) -> String {
    to_project_string(&relative_to(path, project_dir))
}

/// The user's cache directory for FreeSynth: `$XDG_CACHE_HOME/freesynth` or
/// `~/.cache/freesynth` on Unix, `%LOCALAPPDATA%\freesynth` on Windows. `None` when the
/// environment names no home or cache directory.
pub fn user_cache_dir() -> Option<PathBuf> {
    let non_empty = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        non_empty("LOCALAPPDATA")?
    } else {
        non_empty("XDG_CACHE_HOME").or_else(|| non_empty("HOME").map(|h| h.join(".cache")))?
    };
    Some(base.join("freesynth"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn relative_paths() {
        assert_eq!(relative_to(Path::new("/a/b/c.png"), Path::new("/a")), PathBuf::from("b/c.png"));
        assert_eq!(relative_to(Path::new("/a/b/c.png"), Path::new("/a/x/y")), PathBuf::from("../../b/c.png"));
        assert_eq!(relative_to(Path::new("/a/./b/../c"), Path::new("/a")), PathBuf::from("c"));
        assert_eq!(relative_to(Path::new("/a"), Path::new("/a")), PathBuf::from("."));
        assert_eq!(for_project(Path::new("/s/video/[###].jpg"), Path::new("/s")), "video/[###].jpg");
        // An empty directory is the current one.
        assert_eq!(relative_to(Path::new("a/b.png"), Path::new("")), PathBuf::from("a/b.png"));
    }
}
