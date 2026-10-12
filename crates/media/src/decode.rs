//! Decoding a video file into RGBA frames through an ffmpeg pipe.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Stdio};

use freesynth_core::Image;

use crate::colour::ColourTags;
use crate::error::MediaError;
use crate::ffmpeg::{Ffmpeg, StderrDrain, input_args, wait_child};
use crate::probe::VideoInfo;

/// Decodes the first video stream of a file into 8-bit RGBA frames, in presentation order,
/// one frame per stored frame (no frames are dropped or duplicated, whatever the timestamps
/// say). YUV is converted with the colour tags given (normally
/// [`VideoInfo::decode_colour`]).
///
/// Iterate it to get the frames. Dropping it early stops ffmpeg.
pub struct VideoDecoder {
    child: Child,
    stdout: Option<BufReader<ChildStdout>>,
    stderr: StderrDrain,
    path: PathBuf,
    width: usize,
    height: usize,
    decoded: usize,
    done: bool,
}

impl VideoDecoder {
    /// Starts decoding `path`, described by `info` (from [`probe`](crate::probe)), converting
    /// with `colour`.
    pub fn open(ff: &Ffmpeg, path: &Path, info: &VideoInfo, colour: &ColourTags) -> Result<Self, MediaError> {
        Self::open_range(ff, path, info, colour, 0, None)
    }

    /// Like [`open`](Self::open), but returns only the stored frames from index `skip` on (the
    /// earlier ones are decoded and dropped inside ffmpeg, so frame numbers stay exact on any
    /// file), and at most `count` of them.
    pub fn open_range(
        ff: &Ffmpeg,
        path: &Path,
        info: &VideoInfo,
        colour: &ColourTags,
        skip: usize,
        count: Option<usize>,
    ) -> Result<Self, MediaError> {
        // The output size is pinned: frames are read as fixed-size chunks, and a resolution
        // change in the middle of the stream must not misalign them.
        let mut filter = String::new();
        if skip > 0 {
            filter.push_str(&format!("select=gte(n\\,{skip}),"));
        }
        filter.push_str(&format!(
            "scale=w={}:h={}:in_color_matrix={}:in_range={}:flags=bicubic+accurate_rnd+full_chroma_int,format=rgba",
            info.width,
            info.height,
            colour.swscale_matrix(),
            colour.swscale_range()
        ));
        let mut cmd = ff.ffmpeg();
        cmd.args(input_args(path))
            .args(["-map", "0:v:0", "-an", "-sn", "-dn"])
            .args([ff.fps_mode_option(), "passthrough", "-vf"])
            .arg(filter);
        if let Some(n) = count {
            cmd.args(["-frames:v", &n.to_string()]);
        }
        cmd.args(["-f", "rawvideo", "pipe:1"]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| MediaError::Ffmpeg {
            what: format!("starting ffmpeg for '{}'", path.display()),
            stderr: e.to_string(),
        })?;
        let stdout = child.stdout.take().map(|s| BufReader::with_capacity(1 << 20, s));
        let stderr = StderrDrain::new(child.stderr.take());
        Ok(Self {
            child,
            stdout,
            stderr,
            path: path.to_owned(),
            width: info.width,
            height: info.height,
            decoded: 0,
            done: false,
        })
    }

    /// Frames decoded so far.
    pub fn decoded(&self) -> usize {
        self.decoded
    }

    /// Reads one frame; `None` at the end of the stream.
    fn read_frame(&mut self) -> Result<Option<Image>, MediaError> {
        let size = self.width * self.height * 4;
        let mut data = vec![0u8; size];
        let stdout = self.stdout.as_mut().expect("stdout is taken only at the end");
        let mut filled = 0;
        while filled < size {
            match stdout.read(&mut data[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(source) => return Err(MediaError::Io { path: self.path.clone(), source }),
            }
        }
        if filled == 0 {
            return Ok(None);
        }
        if filled < size {
            return Err(MediaError::Ffmpeg {
                what: format!("decoding '{}'", self.path.display()),
                stderr: format!("frame {} ended after {filled} of {size} bytes", self.decoded),
            });
        }
        self.decoded += 1;
        Ok(Some(Image::from_raw(self.width, self.height, 4, data)))
    }

    /// Ends the decode and reports ffmpeg's failure, if any.
    fn finish(&mut self) -> Result<(), MediaError> {
        self.done = true;
        drop(self.stdout.take());
        wait_child(&mut self.child, &mut self.stderr, &format!("decoding '{}'", self.path.display()))
    }
}

impl Iterator for VideoDecoder {
    type Item = Result<Image, MediaError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.read_frame() {
            Ok(Some(frame)) => Some(Ok(frame)),
            Ok(None) => self.finish().err().map(Err),
            Err(e) => {
                self.done = true;
                let _ = self.child.kill();
                let _ = self.child.wait();
                Some(Err(e))
            }
        }
    }
}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
