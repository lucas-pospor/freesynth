//! Encoding frames into a video file through an ffmpeg pipe.

use std::ffi::OsString;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};

use freesynth_core::Image;

use crate::colour::ColourTags;
use crate::error::{MediaError, io_error};
use crate::ffmpeg::{Ffmpeg, StderrDrain, file_arg, input_args, pipe_input_args, wait_child};
use crate::image_io::BitDepth;
use crate::probe::Rational;

/// Video codecs FreeSynth can write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Codec {
    /// Chosen from the file extension: ProRes for `.mov` (4444 with alpha), H.264 for `.mp4`
    /// and `.m4v`, FFV1 (lossless) for `.mkv` and `.avi`, VP9 for `.webm`.
    #[default]
    Auto,
    /// Apple ProRes 422 HQ (10-bit 4:2:2), the usual intermediate for compositing.
    ProRes,
    /// Apple ProRes 4444 (10-bit 4:4:4, with alpha when the frames have it).
    ProRes4444,
    /// H.264 (libx264), 8-bit 4:2:0 (4:4:4 for odd frame sizes). No alpha.
    H264,
    /// H.265/HEVC (libx265), 4:2:0, 10-bit for 16-bit output. No alpha.
    H265,
    /// FFV1, lossless RGB (16 bits per channel for 16-bit output).
    Ffv1,
    /// PNG frames inside the container, lossless RGB(A), 8 or 16 bits.
    Png,
    /// VP9 (libvpx), 4:2:0, with alpha when the frames have it.
    Vp9,
}

impl Codec {
    /// All codecs, for help texts.
    pub const ALL: [Codec; 8] =
        [Codec::Auto, Codec::ProRes, Codec::ProRes4444, Codec::H264, Codec::H265, Codec::Ffv1, Codec::Png, Codec::Vp9];

    /// The setting's lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Codec::Auto => "auto",
            Codec::ProRes => "prores",
            Codec::ProRes4444 => "prores4444",
            Codec::H264 => "h264",
            Codec::H265 => "h265",
            Codec::Ffv1 => "ffv1",
            Codec::Png => "png",
            Codec::Vp9 => "vp9",
        }
    }

    /// The concrete codec for a file at `path`; frames with alpha turn ProRes 422 into 4444.
    pub fn resolve(self, path: &Path, alpha: bool) -> Result<Codec, MediaError> {
        let codec = match self {
            Codec::Auto => {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                match ext.as_str() {
                    "mov" => Codec::ProRes,
                    "mp4" | "m4v" => Codec::H264,
                    "mkv" | "avi" => Codec::Ffv1,
                    "webm" => Codec::Vp9,
                    _ => {
                        return Err(MediaError::Invalid(format!(
                            "no default codec for '{}': use .mov, .mp4, .mkv or .webm, or set the codec",
                            path.display()
                        )));
                    }
                }
            }
            other => other,
        };
        Ok(if codec == Codec::ProRes && alpha { Codec::ProRes4444 } else { codec })
    }

    /// The ffmpeg encoder it uses.
    pub fn encoder(self) -> &'static str {
        match self {
            // prores_aw rather than prores_ks: at the HQ profile it is about twice as fast and
            // needs less than half the CPU time (which the optical flow shares during a GPU
            // render), at an equal or higher PSNR for a slightly larger file.
            Codec::Auto | Codec::ProRes | Codec::ProRes4444 => "prores_aw",
            Codec::H264 => "libx264",
            Codec::H265 => "libx265",
            Codec::Ffv1 => "ffv1",
            Codec::Png => "png",
            Codec::Vp9 => "libvpx-vp9",
        }
    }

    /// The valid constant-quality factors (CRF) of a concrete codec, for the codecs that take
    /// one ([`EncodeSettings::quality`]); `None` for the others.
    pub fn quality_range(self) -> Option<std::ops::RangeInclusive<u32>> {
        match self {
            Codec::H264 | Codec::H265 => Some(0..=51),
            Codec::Vp9 => Some(0..=63),
            _ => None,
        }
    }

    /// True when the codec can store alpha.
    pub fn supports_alpha(self) -> bool {
        matches!(self, Codec::ProRes4444 | Codec::Ffv1 | Codec::Png | Codec::Vp9)
    }

    /// True for codecs that store RGB directly (no YUV matrix).
    fn is_rgb(self) -> bool {
        matches!(self, Codec::Ffv1 | Codec::Png)
    }

    /// True for codecs that are limited-range by convention (players assume `tv` range).
    fn is_limited_range(self) -> bool {
        matches!(self, Codec::Auto | Codec::ProRes | Codec::ProRes4444)
    }
}

impl std::str::FromStr for Codec {
    type Err = MediaError;

    fn from_str(s: &str) -> Result<Self, MediaError> {
        Codec::ALL
            .into_iter()
            .find(|c| c.name() == s)
            .or(if s == "hevc" { Some(Codec::H265) } else { None })
            .ok_or_else(|| {
                let names: Vec<_> = Codec::ALL.iter().map(|c| c.name()).collect();
                MediaError::Invalid(format!("unknown codec '{s}' (expected {})", names.join(", ")))
            })
    }
}

/// How a video is encoded.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodeSettings {
    /// The codec ([`Codec::Auto`] picks one from the extension).
    pub codec: Codec,
    /// Constant-quality factor (CRF) for H.264, H.265 and VP9; `None` uses FreeSynth's
    /// defaults (16, 18 and 24: visually lossless for H.264). Ignored by the other codecs.
    pub quality: Option<u32>,
    /// Frame rate of the frames written through the pipe.
    pub frame_rate: Rational,
    /// Colour tags to write (gaps are filled with BT.709 defaults).
    pub colour: ColourTags,
    /// Keep the frames' alpha channel (when the codec can).
    pub alpha: bool,
    /// Bits per channel of the frames handed over (8, or 16 for high-precision output).
    pub depth: BitDepth,
}

impl Default for EncodeSettings {
    fn default() -> Self {
        Self {
            codec: Codec::Auto,
            quality: None,
            frame_rate: Rational::new(25, 1),
            colour: ColourTags::bt709(),
            alpha: false,
            depth: BitDepth::Eight,
        }
    }
}

/// The output-side arguments: codec, pixel format, colour conversion and tags.
struct OutputPlan {
    codec: Codec,
    args: Vec<String>,
    filter: String,
    /// Notes for the user (alpha dropped, odd size, ...).
    notes: Vec<String>,
}

fn output_plan(path: &Path, width: usize, height: usize, s: &EncodeSettings) -> Result<OutputPlan, MediaError> {
    let codec = s.codec.resolve(path, s.alpha)?;
    let mut notes = Vec::new();
    let alpha = s.alpha && codec.supports_alpha();
    if s.alpha && !alpha {
        notes.push(format!("{} cannot store alpha; the alpha channel is dropped", codec.name()));
    }
    let deep = s.depth == BitDepth::Sixteen;
    let even = width.is_multiple_of(2) && height.is_multiple_of(2);
    let (pix_fmt, mut args): (&str, Vec<String>) = match codec {
        Codec::Auto | Codec::ProRes => ("yuv422p10le", vec!["-profile:v".into(), "3".into()]),
        Codec::ProRes4444 => {
            let fmt = if alpha { "yuva444p10le" } else { "yuv444p10le" };
            (fmt, vec!["-profile:v".into(), "4".into()])
        }
        Codec::H264 => {
            if !even {
                notes.push(format!("{width}x{height} is odd; H.264 is written as 4:4:4, which some players lack"));
            }
            let crf = s.quality.unwrap_or(16).to_string();
            (if even { "yuv420p" } else { "yuv444p" }, vec!["-crf".into(), crf, "-preset".into(), "medium".into()])
        }
        Codec::H265 => {
            let crf = s.quality.unwrap_or(18).to_string();
            let fmt = match (even, deep) {
                (true, false) => "yuv420p",
                (true, true) => "yuv420p10le",
                (false, false) => "yuv444p",
                (false, true) => "yuv444p10le",
            };
            (fmt, vec!["-crf".into(), crf, "-x265-params".into(), "log-level=error".into()])
        }
        Codec::Ffv1 => {
            let fmt = match (alpha, deep) {
                (false, false) => "bgr0",
                (true, false) => "bgra",
                (false, true) => "gbrp16le",
                (true, true) => "gbrap16le",
            };
            (fmt, vec!["-level".into(), "3".into(), "-slices".into(), "16".into(), "-slicecrc".into(), "1".into()])
        }
        Codec::Png => {
            let fmt = match (alpha, deep) {
                (false, false) => "rgb24",
                (true, false) => "rgba",
                (false, true) => "rgb48be",
                (true, true) => "rgba64be",
            };
            (fmt, Vec::new())
        }
        Codec::Vp9 => {
            let crf = s.quality.unwrap_or(24).to_string();
            let fmt = if alpha { "yuva420p" } else { "yuv420p" };
            if !even {
                return Err(MediaError::Invalid(format!("VP9 needs an even frame size, not {width}x{height}")));
            }
            (fmt, vec!["-crf".into(), crf, "-b:v".into(), "0".into(), "-row-mt".into(), "1".into()])
        }
    };
    let mut full = vec!["-c:v".to_owned(), codec.encoder().to_owned()];
    if matches!(codec, Codec::ProRes | Codec::ProRes4444) {
        full.extend(["-vendor".into(), "apl0".into()]);
    }
    full.append(&mut args);
    // The tags go into the filter graph: anything but a plain name is dropped first (project
    // files are checked when parsed, but these settings can come from anywhere).
    let (clean, mut dropped) = s.colour.sanitized();
    notes.append(&mut dropped);
    let tags = clean.completed(width, height, "yuv420p");
    let primaries = tags.primaries.clone().unwrap_or_else(|| "bt709".into());
    let transfer = tags.transfer.clone().unwrap_or_else(|| "bt709".into());
    let filter = if codec.is_rgb() {
        format!("format={pix_fmt},setparams=color_primaries={primaries}:color_trc={transfer}:colorspace=gbr:range=pc")
    } else {
        // The colorspace tag names the matrix actually used for the conversion.
        let (tags, swscale) = tags.for_yuv_output(width, height, codec.is_limited_range());
        let matrix = tags.matrix.clone().unwrap_or_else(|| "bt709".into());
        let range = tags.swscale_range();
        format!(
            "scale=out_color_matrix={swscale}:out_range={range}:flags=bicubic+accurate_rnd+full_chroma_int,\
             format={pix_fmt},setparams=color_primaries={primaries}:color_trc={transfer}:colorspace={matrix}:\
             range={range}"
        )
    };
    Ok(OutputPlan { codec, args: full, filter, notes })
}

/// A file name next to `path` for the encode in progress (same extension, so that ffmpeg
/// picks the same container). It replaces `path` only when the encode succeeds.
fn partial_path(path: &Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
    path.with_file_name(format!(".{name}.partial.{ext}"))
}

/// Checks the encoder exists before starting a long render.
pub fn check_encoder(ff: &Ffmpeg, path: &Path, settings: &EncodeSettings) -> Result<Codec, MediaError> {
    let codec = settings.codec.resolve(path, settings.alpha)?;
    if !ff.has_encoder(codec.encoder())? {
        let hint = match codec {
            Codec::H264 | Codec::H265 | Codec::Vp9 => {
                "install an ffmpeg build with it, or write .mov (ProRes) or .mkv (FFV1) instead".to_owned()
            }
            _ => "install a complete ffmpeg build".to_owned(),
        };
        return Err(MediaError::EncoderMissing { encoder: codec.encoder().into(), hint });
    }
    Ok(codec)
}

/// Writes frames, in order, into a video file through ffmpeg. The file appears only when
/// [`finish`](Self::finish) succeeds (until then ffmpeg writes a hidden partial file next to it).
pub struct VideoEncoder {
    child: Child,
    stdin: Option<BufWriter<ChildStdin>>,
    stderr: StderrDrain,
    path: PathBuf,
    partial: PathBuf,
    width: usize,
    height: usize,
    depth: BitDepth,
    frames: usize,
    notes: Vec<String>,
    codec: Codec,
    /// Set once the file is in place (nothing to clean up on drop).
    finished: bool,
}

impl VideoEncoder {
    /// Starts an encode of `width` x `height` frames into `path`.
    pub fn create(
        ff: &Ffmpeg,
        path: &Path,
        width: usize,
        height: usize,
        settings: &EncodeSettings,
    ) -> Result<Self, MediaError> {
        let plan = output_plan(path, width, height, settings)?;
        check_encoder(ff, path, settings)?;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir).map_err(io_error(dir))?;
        }
        let partial = partial_path(path);
        let pix_in = match settings.depth {
            BitDepth::Eight => "rgba",
            BitDepth::Sixteen => "rgba64le",
        };
        let mut cmd = ff.ffmpeg();
        cmd.args(["-f", "rawvideo", "-pix_fmt", pix_in, "-s", &format!("{width}x{height}")])
            .args(["-framerate", &settings.frame_rate.to_string()])
            .args(pipe_input_args())
            .arg("-vf")
            .arg(&plan.filter)
            .args(&plan.args)
            .args(["-an", "-y"])
            .arg(file_arg(&partial))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| MediaError::Ffmpeg {
            what: format!("starting ffmpeg for '{}'", path.display()),
            stderr: e.to_string(),
        })?;
        let stdin = child.stdin.take().map(|s| BufWriter::with_capacity(1 << 20, s));
        let stderr = StderrDrain::new(child.stderr.take());
        Ok(Self {
            child,
            stdin,
            stderr,
            path: path.to_owned(),
            partial,
            width,
            height,
            depth: settings.depth,
            frames: 0,
            notes: plan.notes,
            codec: plan.codec,
            finished: false,
        })
    }

    /// Notes about the encode for the user (alpha dropped, odd frame size, ...).
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// The codec in use.
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Frames written so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    fn pipe_error(&mut self, source: std::io::Error) -> MediaError {
        // A broken pipe means ffmpeg stopped: its stderr says why.
        drop(self.stdin.take());
        match wait_child(&mut self.child, &mut self.stderr, &format!("encoding '{}'", self.path.display())) {
            Err(e) => e,
            Ok(()) => MediaError::Io { path: self.path.clone(), source },
        }
    }

    /// Appends one frame (8-bit, 1 to 4 channels, the encoder's size).
    pub fn write(&mut self, image: &Image) -> Result<(), MediaError> {
        if image.size() != (self.width, self.height) {
            return Err(MediaError::Invalid(format!(
                "frame {} is {}x{}, the video is {}x{}",
                self.frames,
                image.width(),
                image.height(),
                self.width,
                self.height
            )));
        }
        let rgba = crate::image_io::to_rgba8(image);
        let bytes = match self.depth {
            BitDepth::Eight => rgba,
            BitDepth::Sixteen => rgba.iter().flat_map(|&v| (u16::from(v) * 257).to_le_bytes()).collect(),
        };
        self.write_bytes(&bytes)
    }

    /// Appends one 16-bit RGBA frame (`width * height * 4` values). Only for 16-bit encoders.
    pub fn write_rgba16(&mut self, data: &[u16]) -> Result<(), MediaError> {
        if self.depth != BitDepth::Sixteen || data.len() != self.width * self.height * 4 {
            return Err(MediaError::Invalid("16-bit frame does not match the encoder".into()));
        }
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.write_bytes(&bytes)
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), MediaError> {
        let stdin = self.stdin.as_mut().expect("stdin is open until finish");
        if let Err(e) = stdin.write_all(bytes) {
            return Err(self.pipe_error(e));
        }
        self.frames += 1;
        Ok(())
    }

    /// Closes the pipe, waits for ffmpeg and moves the file into place. Returns the number of
    /// frames written.
    pub fn finish(mut self) -> Result<usize, MediaError> {
        if let Some(mut stdin) = self.stdin.take()
            && let Err(e) = stdin.flush()
        {
            self.stdin = Some(stdin);
            return Err(self.pipe_error(e));
        }
        wait_child(&mut self.child, &mut self.stderr, &format!("encoding '{}'", self.path.display()))?;
        fs::rename(&self.partial, &self.path).map_err(io_error(&self.path))?;
        self.finished = true;
        Ok(self.frames)
    }
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.child.kill();
            let _ = self.child.wait();
            let _ = fs::remove_file(&self.partial);
        }
    }
}

/// Encodes image files (PNG, 8 or 16 bits) into a video, giving frame `i` the display time
/// `durations[i]` (seconds). Used for variable-frame-rate output, which a raw pipe cannot carry.
/// `list_dir` receives a temporary list file.
pub fn encode_files(
    ff: &Ffmpeg,
    frames: &[(PathBuf, f64)],
    path: &Path,
    width: usize,
    height: usize,
    settings: &EncodeSettings,
    list_dir: &Path,
) -> Result<Vec<String>, MediaError> {
    if frames.is_empty() {
        return Err(MediaError::Invalid("no frames to encode".into()));
    }
    let plan = output_plan(path, width, height, settings)?;
    check_encoder(ff, path, settings)?;
    fs::create_dir_all(list_dir).map_err(io_error(list_dir))?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).map_err(io_error(dir))?;
    }
    let mut list = String::from("ffconcat version 1.0\n");
    let quote = |p: &Path| {
        let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_owned());
        format!("'{}'", abs.to_string_lossy().replace('\'', "'\\''"))
    };
    for (file, duration) in frames {
        // A fine time base for each image, so that the durations are not rounded to 1/25 s.
        list.push_str(&format!("file {}\noption framerate 90000\nduration {duration:.6}\n", quote(file)));
    }
    // The concat demuxer ignores the last entry's duration unless the file is listed again;
    // the extra frame is cut by -frames:v.
    list.push_str(&format!("file {}\noption framerate 90000\n", quote(&frames[frames.len() - 1].0)));
    let list_path = list_dir.join(format!("encode-{}.ffconcat", std::process::id()));
    fs::write(&list_path, list).map_err(io_error(&list_path))?;
    let partial = partial_path(path);
    let mut cmd = ff.ffmpeg();
    let mut input: Vec<OsString> = vec!["-f".into(), "concat".into(), "-safe".into(), "0".into()];
    input.extend(input_args(&list_path));
    cmd.args(input)
        .arg("-vf")
        .arg(&plan.filter)
        .args(&plan.args)
        .args([ff.fps_mode_option(), "vfr", "-frames:v", &frames.len().to_string(), "-an", "-y"])
        .arg(file_arg(&partial))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| MediaError::Ffmpeg {
        what: format!("starting ffmpeg for '{}'", path.display()),
        stderr: e.to_string(),
    })?;
    let mut stderr = StderrDrain::new(child.stderr.take());
    let result = wait_child(&mut child, &mut stderr, &format!("encoding '{}'", path.display()));
    let _ = fs::remove_file(&list_path);
    if let Err(e) = result {
        let _ = fs::remove_file(&partial);
        return Err(e);
    }
    fs::rename(&partial, path).map_err(io_error(path))?;
    Ok(plan.notes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codecs_from_extensions() {
        let r = |p: &str, alpha| Codec::Auto.resolve(Path::new(p), alpha).unwrap();
        assert_eq!(r("a.mov", false), Codec::ProRes);
        assert_eq!(r("a.MOV", true), Codec::ProRes4444);
        assert_eq!(r("a.mp4", false), Codec::H264);
        assert_eq!(r("a.mkv", true), Codec::Ffv1);
        assert_eq!(r("a.webm", false), Codec::Vp9);
        assert!(Codec::Auto.resolve(Path::new("a.gif"), false).is_err());
        assert_eq!(Codec::H264.resolve(Path::new("a.mov"), true).unwrap(), Codec::H264);
        for c in Codec::ALL {
            assert_eq!(c.name().parse::<Codec>().unwrap(), c);
        }
        assert_eq!("hevc".parse::<Codec>().unwrap(), Codec::H265);
        assert!("divx".parse::<Codec>().is_err());
        assert_eq!(Codec::Vp9.quality_range(), Some(0..=63));
        assert_eq!(Codec::H265.quality_range(), Some(0..=51));
        assert_eq!(Codec::ProRes.quality_range(), None);
    }

    #[test]
    fn plans_carry_colour_tags() {
        let s = EncodeSettings { colour: ColourTags::bt709(), ..EncodeSettings::default() };
        let p = output_plan(Path::new("o.mov"), 64, 48, &s).unwrap();
        assert_eq!(p.codec, Codec::ProRes);
        assert!(p.filter.contains("out_color_matrix=bt709:out_range=tv"), "{}", p.filter);
        assert!(p.filter.contains("setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"));
        assert!(p.args.windows(2).any(|w| w == ["-profile:v", "3"]));
        // SD tags convert with the BT.601 matrix.
        let sd = EncodeSettings {
            colour: ColourTags::from_probe(Some("smpte170m"), Some("tv"), Some("smpte170m"), Some("bt709")),
            alpha: true,
            codec: Codec::H264,
            ..EncodeSettings::default()
        };
        let p = output_plan(Path::new("o.mp4"), 63, 48, &sd).unwrap();
        assert!(p.filter.contains("out_color_matrix=bt601") && p.filter.contains("colorspace=smpte170m"));
        assert!(p.filter.contains("format=yuv444p"), "odd width");
        assert_eq!(p.notes.len(), 2, "{:?}", p.notes);
        // RGB codecs keep RGB and tag it as such.
        let lossless = EncodeSettings { depth: BitDepth::Sixteen, alpha: true, ..EncodeSettings::default() };
        let p = output_plan(Path::new("o.mkv"), 64, 48, &lossless).unwrap();
        assert!(p.filter.starts_with("format=gbrap16le,") && p.filter.contains("colorspace=gbr"), "{}", p.filter);
    }

    #[test]
    fn plans_never_carry_filter_syntax_from_tags() {
        let s = EncodeSettings {
            colour: ColourTags {
                matrix: Some("bt709,null[x];movie=http\\://127.0.0.1\\:8766/a.mov,nullsink;[x]setparams".into()),
                range: Some("tv".into()),
                primaries: Some("bt709".into()),
                transfer: Some("bt709:x=1".into()),
            },
            ..EncodeSettings::default()
        };
        for path in ["o.mov", "o.mkv"] {
            let p = output_plan(Path::new(path), 1920, 1080, &s).unwrap();
            assert!(
                !p.filter.contains("movie") && !p.filter.contains(';') && !p.filter.contains("x=1"),
                "{}",
                p.filter
            );
            assert_eq!(p.notes.len(), 2, "{:?}", p.notes);
        }
    }

    #[test]
    fn rgb_sources_get_a_real_matrix_tag() {
        // An RGB-stored input (ffprobe: gbr/pc) written as ProRes: converted and tagged BT.709,
        // limited range.
        let s = EncodeSettings {
            colour: ColourTags::from_probe(Some("gbr"), Some("pc"), Some("bt709"), Some("bt709")),
            ..EncodeSettings::default()
        };
        let p = output_plan(Path::new("o.mov"), 1920, 1080, &s).unwrap();
        assert!(p.filter.contains("out_color_matrix=bt709:out_range=tv"), "{}", p.filter);
        assert!(p.filter.contains("colorspace=bt709:range=tv"), "{}", p.filter);
        let p = output_plan(Path::new("o.mp4"), 640, 480, &s).unwrap();
        assert!(p.filter.contains("out_color_matrix=bt601") && p.filter.contains("colorspace=smpte170m"));
        // Full-range YUV stays full range in H.264, not in ProRes.
        let full = EncodeSettings {
            colour: ColourTags::from_probe(Some("bt709"), Some("pc"), None, None),
            ..EncodeSettings::default()
        };
        assert!(output_plan(Path::new("o.mp4"), 64, 48, &full).unwrap().filter.contains("range=pc"));
        assert!(output_plan(Path::new("o.mov"), 64, 48, &full).unwrap().filter.contains("colorspace=bt709:range=tv"));
    }

    #[test]
    fn partial_files_keep_the_extension() {
        assert_eq!(partial_path(Path::new("out/shot.mov")), PathBuf::from("out/.shot.mov.partial.mov"));
    }
}
