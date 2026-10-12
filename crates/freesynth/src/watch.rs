//! Watch mode: render a project, then render again whenever a keyframe painting, a video input
//! or the project file changes on disk.
//!
//! An artist keeps painting keyframes in their usual editor and sees the shot update. Thanks to
//! the render cache, each render after the first redoes only the frames the change affects. A
//! change during a render cancels it and starts the next one at once.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use freesynth_video::CancelToken;

use crate::error::Error;
use crate::event::{Event, ProjectStats};
use crate::hashing::FileStamp;
use crate::project::Project;
use crate::render::RenderRequest;
use crate::session::Session;

/// Timing of watch mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WatchOptions {
    /// How often files are checked.
    pub poll: Duration,
    /// How long a changed file must stay unchanged before rendering (editors often write a file
    /// in several steps).
    pub settle: Duration,
}

impl Default for WatchOptions {
    fn default() -> Self {
        Self { poll: Duration::from_millis(250), settle: Duration::from_millis(400) }
    }
}

/// What watch mode reports. New kinds of events may be added.
#[derive(Debug)]
#[non_exhaustive]
pub enum WatchEvent<'a> {
    /// A render starts; `reason` says why (the first render, or which files changed).
    Rendering { reason: &'a str },
    /// Progress of the current render.
    Render(&'a Event),
    /// A render finished.
    Rendered(&'a ProjectStats),
    /// A render, or loading the project, failed; watching goes on.
    Failed(&'a Error),
    /// The render was cancelled because a file changed.
    Restarting { changed: &'a [PathBuf] },
    /// Waiting for changes to these files.
    Waiting { files: &'a [PathBuf] },
}

type Stamps = BTreeMap<PathBuf, Option<FileStamp>>;

/// The files whose changes trigger a render.
fn watched_files(project_path: &Path, project: Option<&Project>) -> Vec<PathBuf> {
    let mut files = vec![project_path.to_owned()];
    if let Some(p) = project {
        files.extend(p.keyframes.iter().map(|k| p.resolve(&k.image)));
        for spec in std::iter::once(&p.media.frames).chain(p.media.mask.as_ref()) {
            if freesynth_media::is_video_path(spec) {
                files.push(p.resolve(spec));
            }
        }
    }
    files
}

fn stamps(files: &[PathBuf]) -> Stamps {
    files.iter().map(|f| (f.clone(), FileStamp::of(f).ok())).collect()
}

fn changed(before: &Stamps, now: &Stamps) -> Vec<PathBuf> {
    now.iter().filter(|(f, s)| before.get(*f) != Some(*s)).map(|(f, _)| f.clone()).collect()
}

/// Sleeps for `d` in small steps; false when `stop` was cancelled.
fn sleep(d: Duration, stop: &CancelToken) -> bool {
    let step = Duration::from_millis(50);
    let mut left = d;
    while !left.is_zero() {
        if stop.is_cancelled() {
            return false;
        }
        let s = left.min(step);
        std::thread::sleep(s);
        left -= s;
    }
    !stop.is_cancelled()
}

/// Waits until the files differ from `before`, then until they stay unchanged for
/// `options.settle`. Returns the changed files, or `None` when stopped.
fn wait_for_change(
    files: &[PathBuf],
    before: &Stamps,
    options: WatchOptions,
    stop: &CancelToken,
) -> Option<Vec<PathBuf>> {
    loop {
        if !sleep(options.poll, stop) {
            return None;
        }
        let now = stamps(files);
        let diff = changed(before, &now);
        if diff.is_empty() {
            continue;
        }
        let mut last = now;
        loop {
            if !sleep(options.settle, stop) {
                return None;
            }
            let again = stamps(files);
            if again == last {
                return Some(diff);
            }
            last = again;
        }
    }
}

impl Session {
    /// Renders the project at `project_path`, then keeps it up to date: whenever a keyframe
    /// painting, a video input or the project file changes, the project is reloaded and
    /// rendered again (incrementally). Returns when `stop` is cancelled. Render failures are
    /// reported and do not end the watch; an unreadable project file is waited out the same way.
    pub fn watch(
        &self,
        project_path: &Path,
        request: &RenderRequest,
        options: WatchOptions,
        on_event: &(dyn Fn(WatchEvent<'_>) + Sync),
        stop: &CancelToken,
    ) -> Result<(), Error> {
        let mut reason = String::from("first render");
        loop {
            if stop.is_cancelled() {
                return Ok(());
            }
            let project = Project::load(project_path);
            let files = watched_files(project_path, project.as_ref().ok());
            let before = stamps(&files);
            let project = match project {
                Ok(p) => p,
                Err(e) => {
                    on_event(WatchEvent::Failed(&e));
                    on_event(WatchEvent::Waiting { files: &files });
                    match wait_for_change(&files, &before, options, stop) {
                        Some(diff) => {
                            reason = describe_change(&diff);
                            continue;
                        }
                        None => return Ok(()),
                    }
                }
            };
            on_event(WatchEvent::Rendering { reason: &reason });
            let run = CancelToken::new();
            let (result, restart) = std::thread::scope(|s| {
                let poller = s.spawn(|| {
                    // Cancel the render when a watched file changes (or the watch stops).
                    while !run.is_cancelled() {
                        if !sleep(options.poll, stop) {
                            run.cancel();
                            return None;
                        }
                        let diff = changed(&before, &stamps(&files));
                        if !diff.is_empty() {
                            run.cancel();
                            return Some(diff);
                        }
                    }
                    None
                });
                let report = |e: &Event| on_event(WatchEvent::Render(e));
                let result = self.render_with(&project, request, Some(&report), &run);
                run.cancel(); // ends the poller
                (result, poller.join().ok().flatten())
            });
            if stop.is_cancelled() {
                return Ok(());
            }
            match (result, restart) {
                (Err(e), Some(diff)) if e.is_cancelled() => {
                    on_event(WatchEvent::Restarting { changed: &diff });
                    let _ = sleep(options.settle, stop);
                    reason = describe_change(&diff);
                    continue;
                }
                (Ok(stats), _) => on_event(WatchEvent::Rendered(&stats)),
                (Err(e), _) => on_event(WatchEvent::Failed(&e)),
            }
            // Files changed during the render's last moments count too.
            on_event(WatchEvent::Waiting { files: &files });
            match wait_for_change(&files, &before, options, stop) {
                Some(diff) => reason = describe_change(&diff),
                None => return Ok(()),
            }
        }
    }
}

/// "changed: keys/012.png, shot.fsyn".
fn describe_change(files: &[PathBuf]) -> String {
    let names: Vec<String> = files.iter().map(|f| f.display().to_string()).collect();
    format!("changed: {}", names.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notices_changes_and_settles() {
        let dir = crate::test_util::scratch("watch-stamps");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("a.png");
        std::fs::write(&f, b"1").unwrap();
        let files = vec![f.clone(), dir.join("missing.png")];
        let before = stamps(&files);
        assert!(changed(&before, &stamps(&files)).is_empty());
        let stop = CancelToken::new();
        let options = WatchOptions { poll: Duration::from_millis(10), settle: Duration::from_millis(30) };
        let diff = std::thread::scope(|s| {
            let w = s.spawn(|| wait_for_change(&files, &before, options, &stop));
            std::thread::sleep(Duration::from_millis(50));
            std::fs::write(&f, b"22").unwrap();
            w.join().unwrap()
        });
        assert_eq!(diff, Some(vec![f.clone()]));
        stop.cancel();
        assert_eq!(wait_for_change(&files, &stamps(&files), options, &stop), None);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
