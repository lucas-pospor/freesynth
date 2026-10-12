//! Finding ffmpeg and running it as a subprocess.
//!
//! FreeSynth never links libav: it runs the system's `ffmpeg` and `ffprobe` executables and
//! talks to them over pipes. Every input is opened with `-protocol_whitelist file,pipe`, so
//! ffmpeg reads local files and pipes only, never network URLs (a crafted playlist or concat
//! file cannot make it fetch anything).

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::sync::OnceLock;
use std::thread::JoinHandle;

use crate::error::MediaError;

/// Environment variable naming the ffmpeg executable to use.
pub const FFMPEG_ENV: &str = "FREESYNTH_FFMPEG";
/// Environment variable naming the ffprobe executable to use.
pub const FFPROBE_ENV: &str = "FREESYNTH_FFPROBE";

/// The protocols ffmpeg may open: local files and pipes.
pub(crate) const PROTOCOL_WHITELIST: &str = "file,pipe";

/// How much of the start of ffmpeg's stderr is kept for error messages.
const STDERR_HEAD: usize = 2 << 10;
/// How much of the end of ffmpeg's stderr is kept (ffmpeg prints its fatal error last).
const STDERR_TAIL: usize = 14 << 10;

/// A located ffmpeg installation (the `ffmpeg` and `ffprobe` executables).
#[derive(Debug)]
pub struct Ffmpeg {
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
    version: String,
    /// Major and minor version, when the version string has them (`None` for git builds).
    release: Option<(u32, u32)>,
    encoders: OnceLock<Result<String, String>>,
}

impl Clone for Ffmpeg {
    fn clone(&self) -> Self {
        Self {
            ffmpeg: self.ffmpeg.clone(),
            ffprobe: self.ffprobe.clone(),
            version: self.version.clone(),
            release: self.release,
            encoders: OnceLock::new(),
        }
    }
}

/// The major and minor version in an ffmpeg version string: `n9.0.2` and `9.0` give (9, 0),
/// `4.4.2-0ubuntu0.22.04.1` gives (4, 4). Git builds (`N-112345-g...`) and anything else
/// unparsable give `None` (taken as new).
fn parse_release(version: &str) -> Option<(u32, u32)> {
    let v = version.strip_prefix('n').unwrap_or(version);
    let mut parts = v.split(|c: char| !c.is_ascii_digit());
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().and_then(|m| m.parse().ok()).unwrap_or(0);
    Some((major, minor))
}

/// The name of an executable on this platform.
fn exe_name(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_owned() }
}

/// Finds `name` in the directories of `PATH`.
fn search_path(name: &str) -> Option<PathBuf> {
    let file = exe_name(name);
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join(&file)).find(|p| p.is_file())
}

/// The executable named by `env`, or `name` from `PATH`.
fn locate_tool(env: &str, name: &str) -> Result<PathBuf, MediaError> {
    if let Some(p) = std::env::var_os(env).filter(|v| !v.is_empty()) {
        let p = PathBuf::from(p);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(MediaError::FfmpegMissing(format!("{env} is set to '{}', which is not a file", p.display())))
        };
    }
    search_path(name).ok_or_else(|| {
        MediaError::FfmpegMissing(format!(
            "{name} was not found on PATH. Video files need ffmpeg (https://ffmpeg.org): install it with \
             your package manager (for example 'apt install ffmpeg'), or set {env} to the executable. \
             Image sequences work without it."
        ))
    })
}

impl Ffmpeg {
    /// Finds `ffmpeg` and `ffprobe`: the files named by `FREESYNTH_FFMPEG` and
    /// `FREESYNTH_FFPROBE` when set, otherwise the first ones on `PATH`. Runs `ffmpeg -version`
    /// to check that it starts.
    pub fn locate() -> Result<Self, MediaError> {
        let ffmpeg = locate_tool(FFMPEG_ENV, "ffmpeg")?;
        let ffprobe = match std::env::var_os(FFPROBE_ENV).filter(|v| !v.is_empty()) {
            Some(_) => locate_tool(FFPROBE_ENV, "ffprobe")?,
            // Prefer the ffprobe next to the chosen ffmpeg.
            None => ffmpeg
                .parent()
                .map(|d| d.join(exe_name("ffprobe")))
                .filter(|p| p.is_file())
                .map_or_else(|| locate_tool(FFPROBE_ENV, "ffprobe"), Ok)?,
        };
        Self::with_paths(ffmpeg, ffprobe)
    }

    /// Uses the given executables. Runs `ffmpeg -version` to check that it starts.
    pub fn with_paths(ffmpeg: impl Into<PathBuf>, ffprobe: impl Into<PathBuf>) -> Result<Self, MediaError> {
        let ffmpeg = ffmpeg.into();
        let out = Command::new(&ffmpeg)
            .arg("-version")
            .stdin(Stdio::null())
            .output()
            .map_err(|e| MediaError::FfmpegMissing(format!("cannot run ffmpeg at '{}': {e}", ffmpeg.display())))?;
        if !out.status.success() {
            return Err(MediaError::FfmpegMissing(format!("'{} -version' failed", ffmpeg.display())));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let version = text
            .lines()
            .next()
            .and_then(|l| l.strip_prefix("ffmpeg version "))
            .and_then(|l| l.split_whitespace().next())
            .unwrap_or("unknown")
            .to_owned();
        let release = parse_release(&version);
        Ok(Self { ffmpeg, ffprobe: ffprobe.into(), version, release, encoders: OnceLock::new() })
    }

    /// The ffmpeg executable.
    pub fn ffmpeg_path(&self) -> &Path {
        &self.ffmpeg
    }

    /// The ffprobe executable.
    pub fn ffprobe_path(&self) -> &Path {
        &self.ffprobe
    }

    /// ffmpeg's version string (for example `n9.0.2` or `7.1`).
    pub fn version(&self) -> &str {
        &self.version
    }

    /// True when this ffmpeg is at least version `major.minor` (git builds and unparsable
    /// versions count as new).
    pub fn is_at_least(&self, major: u32, minor: u32) -> bool {
        self.release.is_none_or(|r| r >= (major, minor))
    }

    /// A warning about this ffmpeg's version, when it is older than the oldest one FreeSynth
    /// is meant to work with (4.4).
    pub fn version_warning(&self) -> Option<String> {
        (!self.is_at_least(4, 4))
            .then(|| format!("ffmpeg {} is older than 4.4; colour handling and frame timing may fail", self.version))
    }

    /// The option that sets how frame timestamps are handled on output: `-fps_mode` since
    /// ffmpeg 5.1, `-vsync` before (it takes the same `passthrough` and `vfr` values).
    pub(crate) fn fps_mode_option(&self) -> &'static str {
        if self.is_at_least(5, 1) { "-fps_mode" } else { "-vsync" }
    }

    /// True when this ffmpeg has the encoder `name` (as listed by `ffmpeg -encoders`).
    pub fn has_encoder(&self, name: &str) -> Result<bool, MediaError> {
        let list = self.encoders.get_or_init(|| {
            let out = Command::new(&self.ffmpeg)
                .args(["-hide_banner", "-encoders"])
                .stdin(Stdio::null())
                .stderr(Stdio::null())
                .output()
                .map_err(|e| e.to_string())?;
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        });
        let list =
            list.as_ref().map_err(|e| MediaError::Ffmpeg { what: "ffmpeg -encoders".into(), stderr: e.clone() })?;
        // Lines look like " V....D libx264    libx264 H.264 / AVC ...".
        Ok(list.lines().any(|l| l.split_whitespace().nth(1) == Some(name)))
    }

    /// An `ffmpeg` command with the common quiet options.
    pub(crate) fn ffmpeg(&self) -> Command {
        let mut c = Command::new(&self.ffmpeg);
        c.args(["-nostdin", "-hide_banner", "-loglevel", "error"]);
        c
    }

    /// An `ffprobe` command with the common quiet options and the protocol whitelist.
    pub(crate) fn ffprobe(&self) -> Command {
        let mut c = Command::new(&self.ffprobe);
        c.args(["-v", "error", "-protocol_whitelist", PROTOCOL_WHITELIST]);
        c
    }
}

/// A file path as an ffmpeg or ffprobe argument. Relative paths get a `./` prefix: ffmpeg
/// reads `a:b.mov` as protocol `a` and `-clip.mov` as an option, while `./a:b.mov` and
/// `./-clip.mov` are plain files.
pub(crate) fn file_arg(path: &Path) -> OsString {
    let mut components = path.components();
    match components.next() {
        // `/x`, `./x` and `../x` are read as files already.
        Some(Component::RootDir | Component::CurDir | Component::ParentDir) | None => path.as_os_str().to_owned(),
        // A drive prefix (Windows): `C:\x` is absolute, `C:x` is made absolute.
        Some(Component::Prefix(_)) if path.is_absolute() => path.as_os_str().to_owned(),
        Some(Component::Prefix(_)) => std::path::absolute(path).map_or_else(|_| path.into(), PathBuf::into_os_string),
        Some(_) => Path::new(".").join(path).into_os_string(),
    }
}

/// `-protocol_whitelist file,pipe -i <path>`: an input restricted to local files and pipes.
pub(crate) fn input_args(path: &Path) -> Vec<OsString> {
    vec!["-protocol_whitelist".into(), PROTOCOL_WHITELIST.into(), "-i".into(), file_arg(path)]
}

/// The input arguments for raw frames on stdin.
pub(crate) fn pipe_input_args() -> Vec<OsString> {
    ["-protocol_whitelist", PROTOCOL_WHITELIST, "-i", "pipe:0"].map(OsString::from).to_vec()
}

/// Reads `reader` to its end and keeps its first [`STDERR_HEAD`] and last [`STDERR_TAIL`]
/// bytes, joined by `...` when something was left out in between.
fn collect_head_and_tail(mut reader: impl Read) -> String {
    let mut head = Vec::new();
    let mut tail = VecDeque::new();
    let mut skipped = false;
    let mut buf = [0u8; 4096];
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let mut chunk = &buf[..n];
        if head.len() < STDERR_HEAD {
            let take = chunk.len().min(STDERR_HEAD - head.len());
            head.extend_from_slice(&chunk[..take]);
            chunk = &chunk[take..];
        }
        tail.extend(chunk);
        if tail.len() > STDERR_TAIL {
            tail.drain(..tail.len() - STDERR_TAIL);
            skipped = true;
        }
    }
    let mut text = String::from_utf8_lossy(&head).into_owned();
    let tail: Vec<u8> = tail.into();
    if skipped {
        // Start the tail at a line boundary.
        let start = tail.iter().position(|&b| b == b'\n').map_or(0, |i| i + 1);
        text.push_str("\n...\n");
        text.push_str(&String::from_utf8_lossy(&tail[start..]));
    } else {
        text.push_str(&String::from_utf8_lossy(&tail));
    }
    text
}

/// Collects a child's stderr on a thread, so that the child never blocks on a full pipe.
pub(crate) struct StderrDrain(Option<JoinHandle<String>>);

impl StderrDrain {
    pub fn new(stderr: Option<ChildStderr>) -> Self {
        Self(stderr.map(|s| std::thread::spawn(move || collect_head_and_tail(s))))
    }

    /// What the child printed (after it closed stderr).
    pub fn finish(&mut self) -> String {
        self.0.take().and_then(|h| h.join().ok()).unwrap_or_default()
    }
}

/// Waits for `child` and turns a failure into an error with its stderr.
pub(crate) fn wait_child(child: &mut Child, stderr: &mut StderrDrain, what: &str) -> Result<(), MediaError> {
    let status = child.wait().map_err(|e| MediaError::Ffmpeg { what: what.to_owned(), stderr: e.to_string() })?;
    let text = stderr.finish();
    if status.success() {
        Ok(())
    } else {
        let text = if text.trim().is_empty() { format!("exit status {status}") } else { text };
        Err(MediaError::Ffmpeg { what: what.to_owned(), stderr: text })
    }
}

/// Runs a command to completion and returns its stdout.
pub(crate) fn run_output(mut cmd: Command, what: &str) -> Result<Vec<u8>, MediaError> {
    let out = cmd
        .stdin(Stdio::null())
        .output()
        .map_err(|e| MediaError::Ffmpeg { what: what.to_owned(), stderr: e.to_string() })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        return Err(MediaError::Ffmpeg { what: what.to_owned(), stderr });
    }
    Ok(out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_args_whitelist_protocols() {
        let args = input_args(Path::new("a.mov"));
        assert_eq!(args, ["-protocol_whitelist", "file,pipe", "-i", "./a.mov"].map(OsString::from));
    }

    #[cfg(unix)]
    #[test]
    fn relative_paths_are_never_protocols_or_options() {
        let arg = |p: &str| file_arg(Path::new(p));
        assert_eq!(arg("a:b.mov"), "./a:b.mov");
        assert_eq!(arg("-clip.mov"), "./-clip.mov");
        assert_eq!(arg("dir/.out:x.mov.partial.mov"), "./dir/.out:x.mov.partial.mov");
        assert_eq!(arg("/abs/10:30.mov"), "/abs/10:30.mov");
        assert_eq!(arg("./x.mov"), "./x.mov");
        assert_eq!(arg("../a:b.mov"), "../a:b.mov");
        assert_eq!(pipe_input_args().last().unwrap(), "pipe:0");
    }

    #[test]
    fn versions() {
        assert_eq!(parse_release("n9.0.2"), Some((9, 0)));
        assert_eq!(parse_release("4.4.2-0ubuntu0.22.04.1"), Some((4, 4)));
        assert_eq!(parse_release("5.1.6-0+deb12u1"), Some((5, 1)));
        assert_eq!(parse_release("7"), Some((7, 0)));
        assert_eq!(parse_release("N-112345-gabcdef"), None);
        assert_eq!(parse_release("unknown"), None);
        let ff = |release| Ffmpeg {
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
            version: "x".into(),
            release,
            encoders: OnceLock::new(),
        };
        assert_eq!(ff(Some((4, 4))).fps_mode_option(), "-vsync");
        assert_eq!(ff(Some((5, 1))).fps_mode_option(), "-fps_mode");
        assert_eq!(ff(None).fps_mode_option(), "-fps_mode", "git builds are new");
        assert!(ff(Some((4, 2))).version_warning().is_some() && ff(Some((4, 4))).version_warning().is_none());
    }

    #[test]
    fn stderr_keeps_the_end() {
        let short = collect_head_and_tail(&b"one error"[..]);
        assert_eq!(short, "one error");
        let mut spam: Vec<u8> = Vec::new();
        for i in 0..2000 {
            spam.extend(format!("line {i}: non monotonically increasing dts\n").as_bytes());
        }
        spam.extend(b"Conversion failed!\n");
        let kept = collect_head_and_tail(&spam[..]);
        assert!(kept.starts_with("line 0:") && kept.contains("\n...\n"), "{}", &kept[..80]);
        assert!(kept.ends_with("Conversion failed!\n"));
        assert!(kept.len() <= STDERR_HEAD + STDERR_TAIL + 5);
        let after = kept.split("\n...\n").nth(1).unwrap();
        assert!(after.starts_with("line "), "the tail starts at a line");
    }

    #[test]
    fn missing_tool_names_the_variable() {
        let err = locate_tool("FREESYNTH_TEST_UNSET_TOOL", "freesynth-no-such-tool").unwrap_err().to_string();
        assert!(err.contains("freesynth-no-such-tool was not found") && err.contains("FREESYNTH_TEST_UNSET_TOOL"));
    }
}
