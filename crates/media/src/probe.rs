//! Reading a video's stream description and its frame-to-timestamp index with ffprobe.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::colour::{ColourTags, has_alpha_format};
use crate::error::MediaError;
use crate::ffmpeg::{Ffmpeg, file_arg, run_output};

/// A rational number (`num / den`), as ffmpeg writes frame rates and time bases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

impl Rational {
    pub const fn new(num: i64, den: i64) -> Self {
        Self { num, den }
    }

    /// Parses `30000/1001`, `25` or `0.5`-free integer forms. `0/0` and malformed text give
    /// `None`.
    pub fn parse(text: &str) -> Option<Self> {
        let (num, den) = match text.split_once('/') {
            Some((n, d)) => (n.trim().parse().ok()?, d.trim().parse().ok()?),
            None => (text.trim().parse().ok()?, 1),
        };
        (num > 0 && den > 0).then_some(Self { num, den })
    }

    /// A rational close to `value` (for frame rates given as decimals such as 29.97).
    pub fn approximate(value: f64) -> Option<Self> {
        if !value.is_finite() || value <= 0.0 {
            return None;
        }
        // Common NTSC rates first, then thousandths.
        for (n, d) in [(24000, 1001), (30000, 1001), (60000, 1001), (120000, 1001)] {
            if (value - n as f64 / d as f64).abs() < 5e-4 {
                return Some(Self::new(n, d));
            }
        }
        let den = if value.fract() == 0.0 { 1 } else { 1000 };
        Some(Self::new((value * den as f64).round() as i64, den))
    }

    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 { write!(f, "{}", self.num) } else { write!(f, "{}/{}", self.num, self.den) }
    }
}

/// What ffprobe says about a video file's first video stream, plus the presentation timestamp
/// of every frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoInfo {
    /// Frame size as decoded (after applying a rotation tag, as ffmpeg does by default).
    pub width: usize,
    pub height: usize,
    /// Codec name, such as `h264` or `prores`.
    pub codec: String,
    /// Stored pixel format, such as `yuv420p` or `yuva444p10le`.
    pub pix_fmt: String,
    /// True when the stored pixel format has an alpha channel.
    pub has_alpha: bool,
    /// Average frame rate.
    pub frame_rate: Rational,
    /// Time base of the timestamps in [`pts`](Self::pts).
    pub time_base: Rational,
    /// Presentation timestamp of each frame, in presentation order: frame `i` of a sequential
    /// decode is shown at `pts[i] * time_base` seconds. Empty when ffprobe reported no
    /// timestamps.
    pub pts: Vec<i64>,
    /// Number of frames (the length of `pts`, or the container's count without timestamps).
    pub frame_count: usize,
    /// Colour tags as stored (`None` where untagged).
    pub colour: ColourTags,
    /// Display rotation in degrees (counter-clockwise), from the stream's display matrix.
    pub rotation: i32,
    /// True when the file also holds an audio stream (FreeSynth's outputs carry no audio).
    #[serde(default)]
    pub has_audio: bool,
}

impl VideoInfo {
    /// True when the frames are not evenly spaced in time (variable frame rate): any gap
    /// between consecutive timestamps differs from the first by more than one tick.
    pub fn is_vfr(&self) -> bool {
        let mut gaps = self.pts.windows(2).map(|w| w[1] - w[0]);
        let Some(first) = gaps.next() else { return false };
        gaps.any(|g| (g - first).abs() > 1)
    }

    /// How long frame `i` is shown, in seconds: the gap to the next timestamp (the last frame
    /// gets the previous gap, or one frame period).
    pub fn frame_duration(&self, i: usize) -> f64 {
        let tb = self.time_base.as_f64();
        let period = 1.0 / self.frame_rate.as_f64();
        match (self.pts.get(i), self.pts.get(i + 1)) {
            (Some(a), Some(b)) if b > a => (b - a) as f64 * tb,
            _ if i > 0 && i < self.pts.len() => self.frame_duration(i - 1),
            _ => period,
        }
    }

    /// The tags used to decode: the stored ones, with the gaps filled by the usual defaults
    /// ([`ColourTags::completed`]).
    pub fn decode_colour(&self) -> ColourTags {
        self.colour.completed(self.width, self.height, &self.pix_fmt)
    }
}

#[derive(Deserialize)]
struct ProbeOutput {
    #[serde(default)]
    streams: Vec<ProbeStream>,
}

#[derive(Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    codec_name: Option<String>,
    width: Option<usize>,
    height: Option<usize>,
    pix_fmt: Option<String>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    time_base: Option<String>,
    nb_frames: Option<String>,
    color_space: Option<String>,
    color_range: Option<String>,
    color_primaries: Option<String>,
    color_transfer: Option<String>,
    #[serde(default)]
    side_data_list: Vec<SideData>,
}

#[derive(Deserialize)]
struct SideData {
    rotation: Option<f64>,
}

/// Reads the description and frame index of the first video stream of `path`.
pub fn probe(ff: &Ffmpeg, path: &Path) -> Result<VideoInfo, MediaError> {
    if !path.is_file() {
        return Err(MediaError::Io {
            path: path.to_owned(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        });
    }
    // Every stream is listed (to see whether there is audio); the first video stream is used.
    let mut cmd = ff.ffprobe();
    cmd.args(["-show_entries"])
        .arg(
            "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,time_base,nb_frames,\
             color_space,color_range,color_primaries,color_transfer:stream_side_data=rotation",
        )
        .args(["-of", "json", "-i"])
        .arg(file_arg(path));
    let what = format!("ffprobe '{}'", path.display());
    // The stream description and the packet index come from two ffprobe runs, side by side.
    let (json, pts) = std::thread::scope(|s| {
        let packets = s.spawn(|| packet_pts(ff, path));
        let json = run_output(cmd, &what);
        (json, packets.join().unwrap_or_else(|_| Ok(Vec::new())))
    });
    let json = json?;
    let out: ProbeOutput = serde_json::from_slice(&json)
        .map_err(|e| MediaError::Ffmpeg { what: what.clone(), stderr: format!("unreadable output: {e}") })?;
    let has_audio = out.streams.iter().any(|s| s.codec_type.as_deref() == Some("audio"));
    let s = out
        .streams
        .into_iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .ok_or_else(|| MediaError::Unsupported(format!("'{}' has no video stream", path.display())))?;
    let (Some(mut width), Some(mut height)) = (s.width, s.height) else {
        return Err(MediaError::Unsupported(format!("'{}': the video stream has no frame size", path.display())));
    };
    let rotation = s.side_data_list.iter().find_map(|d| d.rotation).unwrap_or(0.0).round() as i32;
    if rotation.rem_euclid(180) == 90 {
        std::mem::swap(&mut width, &mut height);
    }
    let pix_fmt = s.pix_fmt.unwrap_or_default();
    let frame_rate = s
        .avg_frame_rate
        .as_deref()
        .and_then(Rational::parse)
        .or_else(|| s.r_frame_rate.as_deref().and_then(Rational::parse))
        .unwrap_or(Rational::new(25, 1));
    let time_base = s.time_base.as_deref().and_then(Rational::parse).unwrap_or(Rational::new(1, 90000));
    let pts = pts?;
    let frame_count = if pts.is_empty() { s.nb_frames.and_then(|n| n.parse().ok()).unwrap_or(0) } else { pts.len() };
    Ok(VideoInfo {
        width,
        height,
        codec: s.codec_name.unwrap_or_default(),
        has_alpha: has_alpha_format(&pix_fmt),
        pix_fmt,
        frame_rate,
        time_base,
        pts,
        frame_count,
        colour: ColourTags::from_probe(
            s.color_space.as_deref(),
            s.color_range.as_deref(),
            s.color_primaries.as_deref(),
            s.color_transfer.as_deref(),
        ),
        rotation,
        has_audio,
    })
}

/// The presentation timestamps of the stream's packets, sorted: one per frame, in the order a
/// sequential decode returns the frames. Reading packets needs no decoding, so this is quick
/// even for long files.
fn packet_pts(ff: &Ffmpeg, path: &Path) -> Result<Vec<i64>, MediaError> {
    let mut cmd = ff.ffprobe();
    cmd.args(["-select_streams", "v:0", "-show_entries", "packet=pts,flags", "-of", "csv=p=0", "-i"])
        .arg(file_arg(path));
    let out = run_output(cmd, &format!("ffprobe packets of '{}'", path.display()))?;
    let text = String::from_utf8_lossy(&out);
    let mut pts = Vec::new();
    for line in text.lines() {
        let mut fields = line.split(',');
        let (Some(value), flags) = (fields.next(), fields.next().unwrap_or("")) else { continue };
        // Packets flagged as discarded (edit lists) are not presented.
        if flags.contains('D') {
            continue;
        }
        match value.trim().parse::<i64>() {
            Ok(p) => pts.push(p),
            // A packet without a timestamp: give up on the index (timing falls back to the
            // frame rate).
            Err(_) => return Ok(Vec::new()),
        }
    }
    pts.sort_unstable();
    Ok(pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(pts: Vec<i64>) -> VideoInfo {
        VideoInfo {
            width: 4,
            height: 4,
            codec: "h264".into(),
            pix_fmt: "yuv420p".into(),
            has_alpha: false,
            frame_rate: Rational::new(25, 1),
            time_base: Rational::new(1, 1000),
            frame_count: pts.len(),
            pts,
            colour: ColourTags::default(),
            rotation: 0,
            has_audio: false,
        }
    }

    #[test]
    fn rationals() {
        assert_eq!(Rational::parse("30000/1001"), Some(Rational::new(30000, 1001)));
        assert_eq!(Rational::parse("25"), Some(Rational::new(25, 1)));
        assert_eq!(Rational::parse("0/0"), None);
        assert_eq!(Rational::approximate(29.97), Some(Rational::new(30000, 1001)));
        assert_eq!(Rational::approximate(24.0), Some(Rational::new(24, 1)));
        assert_eq!(Rational::approximate(12.5), Some(Rational::new(12500, 1000)));
        assert_eq!(Rational::new(30000, 1001).to_string(), "30000/1001");
    }

    #[test]
    fn vfr_detection_and_durations() {
        let cfr = info(vec![0, 40, 80, 120]);
        assert!(!cfr.is_vfr());
        assert!((cfr.frame_duration(1) - 0.04).abs() < 1e-9);
        assert!((cfr.frame_duration(3) - 0.04).abs() < 1e-9, "last frame repeats the previous gap");
        let vfr = info(vec![0, 40, 93, 120]);
        assert!(vfr.is_vfr());
        assert!((vfr.frame_duration(1) - 0.053).abs() < 1e-9);
        // One-tick jitter (rounded timestamps) is not VFR.
        assert!(!info(vec![0, 33, 67, 100]).is_vfr());
        assert!((info(vec![]).frame_duration(0) - 0.04).abs() < 1e-9);
    }
}
