//! Video and image I/O for FreeSynth.
//!
//! - [`Ffmpeg`]: finds the system's `ffmpeg` and `ffprobe` and runs them as subprocesses over
//!   pipes (no libav linking). Every input is opened with `-protocol_whitelist file,pipe`, so
//!   ffmpeg only ever reads local files and pipes. Without ffmpeg, video files fail with a
//!   clear message ([`MediaError::FfmpegMissing`]) and image sequences keep working. ffmpeg 4.4
//!   or newer is expected ([`Ffmpeg::version_warning`]).
//! - [`probe`]: a video's size, frame rate, colour tags and a frame-to-timestamp index
//!   ([`VideoInfo::pts`]), so that frame numbers stay exact on variable-frame-rate footage.
//! - [`VideoDecoder`]: decodes every stored frame, in order, to RGBA with the stream's colour
//!   matrix and range ([`ColourTags`]); untagged video gets the usual defaults.
//! - [`VideoEncoder`] and [`encode_files`]: write ProRes, H.264, H.265, FFV1, PNG or VP9 with
//!   explicit colour tags (ffmpeg writes none by default).
//! - [`save_png`] and [`save_frame`]: 8- and 16-bit PNG frames, written atomically.
//!
//! ```no_run
//! use freesynth_media::{EncodeSettings, Ffmpeg, VideoDecoder, VideoEncoder, probe};
//! use std::path::Path;
//!
//! let ff = Ffmpeg::locate()?;
//! let info = probe(&ff, Path::new("in.mov"))?;
//! let settings = EncodeSettings {
//!     frame_rate: info.frame_rate,
//!     colour: info.decode_colour(),
//!     ..EncodeSettings::default()
//! };
//! let mut out = VideoEncoder::create(&ff, Path::new("out.mov"), info.width, info.height, &settings)?;
//! for frame in VideoDecoder::open(&ff, Path::new("in.mov"), &info, &info.decode_colour())? {
//!     out.write(&frame?)?;
//! }
//! out.finish()?;
//! # Ok::<(), freesynth_media::MediaError>(())
//! ```

mod colour;
mod decode;
mod encode;
mod error;
mod ffmpeg;
mod image_io;
mod probe;

pub use crate::colour::{
    ColourTags, MATRIX_NAMES, PRIMARIES_NAMES, RANGE_NAMES, TRANSFER_NAMES, has_alpha_format, is_rgb_format,
};
pub use crate::decode::VideoDecoder;
pub use crate::encode::{Codec, EncodeSettings, VideoEncoder, check_encoder, encode_files};
pub use crate::error::MediaError;
pub use crate::ffmpeg::{FFMPEG_ENV, FFPROBE_ENV, Ffmpeg};
pub use crate::image_io::{BitDepth, load_rgba16, save_frame, save_png, save_png16, to_rgba8, write_atomic};
pub use crate::probe::{Rational, VideoInfo, probe};

/// File extensions treated as video files (compared without case). Anything else is an image
/// pattern or a directory of images.
pub const VIDEO_EXTENSIONS: [&str; 11] =
    ["mov", "mp4", "m4v", "mkv", "avi", "webm", "mxf", "mts", "m2ts", "mpg", "wmv"];

/// True when `path` names a video file by its extension.
pub fn is_video_path(path: impl AsRef<std::path::Path>) -> bool {
    path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| VIDEO_EXTENSIONS.iter().any(|v| v.eq_ignore_ascii_case(e)))
}

#[cfg(test)]
mod tests {
    #[test]
    fn video_paths() {
        assert!(super::is_video_path("shot.MOV"));
        assert!(super::is_video_path("a/b.mkv"));
        assert!(!super::is_video_path("frames/[####].png"));
        assert!(!super::is_video_path("frames"));
    }
}
