//! Project files (`.fsyn`): a shot, its keyframes, the look and the output, in TOML.
//!
//! ```toml
//! format_version = 1
//!
//! [media]
//! frames = "video/[###].jpg"   # a frame pattern ('#' run = zero-padded number), a directory,
//!                              # or a video file ("shot.mov", read through ffmpeg)
//! # mask = "mask/[###].png"    # optional mask sequence (pattern, directory or video file)
//! # first = 0                  # optional frame range (default: every frame found)
//! # last = 99
//! # start_number = 0           # frame number of a video file's first frame
//! # fps = 25.0                 # frame rate of an image sequence, for video output
//! # colour = { matrix = "bt709", range = "tv" }  # override a video's colour tags (ffmpeg names)
//!
//! [[keyframe]]
//! frame = 0
//! image = "keys/000.png"
//!
//! [[keyframe]]
//! frame = 99
//! image = "keys/099.png"
//!
//! [guides]                     # total weight per guide layer
//! style = 1.0
//! video = 4.0
//! temporal = 1.0
//! mapping = 10.0
//! mask = 1.0
//! gate_mapping = true          # weight the mapping guide by tracking confidence
//! gate_temporal = false        # weight the temporal guide by flow consistency
//! temporal_gate_erosion = 0    # pixels the inconsistent regions grow by (with gate_temporal)
//!
//! [engine]
//! backend = "auto"             # auto, gpu or cpu
//! preset = "default"           # compat, default, fast or high
//! uniformity = 3500.0
//! patch_size = 5
//! seed = 1
//! warm_start = false           # faster: start each frame from the previous one (preview)
//! warm_start_levels = 2        # finest pyramid levels run with a warm start
//! warm_start_iterations = 3    # search/vote iterations per level with a warm start
//!
//! [output]
//! frames = "out/[#####].png"   # a frame pattern or a directory (NNNNN.png inside)
//! # video = "out.mov"          # and/or a video file, encoded with ffmpeg
//! bit_depth = 8                # 8 or 16 (16 needs PNG frames)
//! codec = "auto"               # auto (by extension), prores, prores4444, h264, h265, ffv1, png, vp9
//! # quality = 16               # CRF of h264, h265 and vp9
//! blend = "layered"            # the default; or "adaptive" (sharper) or "linear" (crossfade)
//! keyframes = "synthesized"    # or "painting"
//!
//! [blending]                   # tuning of the adaptive and layered blends
//! softness = 0.33              # larger: steadier, smaller: sharper selection
//! error_smoothing = 0.125
//! temporal_prior = 1.0
//! data_weight = 0.05
//! detail = 0.0
//! shading_scale = 0.006        # "layered" only: shading/detail split (fraction of the frame)
//! ```
//!
//! Every field has a default, so a project needs only `media.frames`, the keyframes and
//! `output.frames` or `output.video`. Relative paths are relative to the project file and are
//! written with `/` on every system. Frame patterns are parsed, never used as format strings,
//! so a `%` in a path is just a character. Unknown fields and unknown setting names (a backend,
//! preset, blend mode, codec or keyframe output) are errors that point at their line, so that a
//! misspelt setting is not silently ignored.
//!
//! The same fields can be stored as JSON (a file ending in `.json`), for programs that generate
//! projects: [`Project::to_json`] and [`Project::from_json`].

use std::fs;
use std::path::{Path, PathBuf};

use freesynth_core::{Params, Preset};
use freesynth_media::{BitDepth, Codec, ColourTags, MediaError, is_video_path};
use freesynth_video::{
    Backend, BlendMode, BlendSettings, GuideWeights, KeyframeOutput, VideoError, VideoParams, WarmStartSettings,
};
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Version of the project format written by this library.
pub const FORMAT_VERSION: u32 = 1;

/// The project file extension (without the dot).
pub const EXTENSION: &str = "fsyn";

/// A setting stored in project files by its lower-case name.
trait Named: Sized {
    fn name(&self) -> &'static str;
    fn from_name(name: &str) -> Result<Self, String>;
}

impl Named for Backend {
    fn name(&self) -> &'static str {
        Backend::name(*self)
    }

    fn from_name(name: &str) -> Result<Self, String> {
        name.parse().map_err(|e: VideoError| e.to_string())
    }
}

impl Named for Preset {
    fn name(&self) -> &'static str {
        Preset::name(*self)
    }

    fn from_name(name: &str) -> Result<Self, String> {
        name.parse().map_err(|e: freesynth_core::Error| e.to_string())
    }
}

impl Named for BlendMode {
    fn name(&self) -> &'static str {
        BlendMode::name(*self)
    }

    fn from_name(name: &str) -> Result<Self, String> {
        name.parse().map_err(|e: VideoError| e.to_string())
    }
}

impl Named for KeyframeOutput {
    fn name(&self) -> &'static str {
        KeyframeOutput::name(*self)
    }

    fn from_name(name: &str) -> Result<Self, String> {
        name.parse().map_err(|e: VideoError| e.to_string())
    }
}

impl Named for Codec {
    fn name(&self) -> &'static str {
        Codec::name(*self)
    }

    fn from_name(name: &str) -> Result<Self, String> {
        name.parse().map_err(|e: MediaError| e.to_string())
    }
}

/// Serde adapter for [`Named`] settings: written as the name, and an unknown name fails the
/// parse with the position in the file.
mod by_name {
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    use super::Named;

    pub fn serialize<T: Named, S: Serializer>(value: &T, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(value.name())
    }

    pub fn deserialize<'de, T: Named, D: Deserializer<'de>>(d: D) -> Result<T, D::Error> {
        let name = String::deserialize(d)?;
        T::from_name(&name).map_err(D::Error::custom)
    }
}

/// Serde adapter for [`Media::colour`]: each tag must be one of ffmpeg's names
/// ([`freesynth_media::MATRIX_NAMES`] and the like), checked when the file is parsed so that an
/// unknown name fails with its position in the file. (The tags end up in ffmpeg command lines.)
mod colour_tags {
    use freesynth_media::{ColourTags, MATRIX_NAMES, PRIMARIES_NAMES, RANGE_NAMES, TRANSFER_NAMES};
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer};

    fn named<'de, D: Deserializer<'de>>(d: D, what: &str, names: &[&str]) -> Result<Option<String>, D::Error> {
        // A JSON `null` is an unset tag.
        let Some(value) = Option::<String>::deserialize(d)? else { return Ok(None) };
        if names.contains(&value.as_str()) {
            Ok(Some(value))
        } else {
            Err(D::Error::custom(format!("unknown colour {what} '{value}' (expected one of {})", names.join(", "))))
        }
    }

    fn matrix<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
        named(d, "matrix", &MATRIX_NAMES)
    }

    fn range<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
        named(d, "range", &RANGE_NAMES)
    }

    fn primaries<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
        named(d, "primaries", &PRIMARIES_NAMES)
    }

    fn transfer<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
        named(d, "transfer", &TRANSFER_NAMES)
    }

    /// The `[media.colour]` table as it may appear in a project file.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Checked {
        #[serde(default, deserialize_with = "matrix")]
        matrix: Option<String>,
        #[serde(default, deserialize_with = "range")]
        range: Option<String>,
        #[serde(default, deserialize_with = "primaries")]
        primaries: Option<String>,
        #[serde(default, deserialize_with = "transfer")]
        transfer: Option<String>,
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<ColourTags>, D::Error> {
        let c = Option::<Checked>::deserialize(d)?;
        Ok(c.map(|c| ColourTags { matrix: c.matrix, range: c.range, primaries: c.primaries, transfer: c.transfer }))
    }
}

/// Serde checks of engine settings, so that a bad value fails the parse with its position in
/// the file.
mod engine_values {
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer};

    pub fn patch_size<'de, D: Deserializer<'de>>(d: D) -> Result<usize, D::Error> {
        let v = usize::deserialize(d)?;
        super::check_patch_size(v).map_err(D::Error::custom)?;
        Ok(v)
    }

    pub fn uniformity<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
        let v = f32::deserialize(d)?;
        super::check_uniformity(v).map_err(D::Error::custom)?;
        Ok(v)
    }
}

/// Checks `engine.patch_size`.
fn check_patch_size(v: usize) -> Result<(), String> {
    if v < 3 || v.is_multiple_of(2) {
        return Err(format!("engine.patch_size must be odd and at least 3, not {v}"));
    }
    Ok(())
}

/// Checks `engine.uniformity`.
fn check_uniformity(v: f32) -> Result<(), String> {
    if !(v.is_finite() && v >= 0.0) {
        return Err(format!("engine.uniformity must be a number of at least 0, not {v}"));
    }
    Ok(())
}

/// The input sequences.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Media {
    /// Input video frames: a pattern such as `video/[####].png`, a directory of numbered
    /// images, or a video file such as `shot.mov`.
    pub frames: String,
    /// Optional mask sequence, as a pattern, a directory or a video file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
    /// First frame to render (default: the first frame found).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first: Option<usize>,
    /// Last frame to render (default: the last frame found).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last: Option<usize>,
    /// Frame number of the first frame of a video file (frames of image sequences are
    /// numbered by their file names).
    #[serde(skip_serializing_if = "is_zero")]
    pub start_number: usize,
    /// Frame rate of an image-sequence input, used when the output is a video (video files
    /// carry their own timing). Default 25.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    /// Colour tags of a video input, overriding (or filling in) the file's own. Missing tags
    /// default to BT.709 for HD and BT.601 for smaller video. Each tag must be one of ffmpeg's
    /// names (matrix `bt709`, `smpte170m`, `bt470bg`, `bt2020nc`, ...; range `tv` or `pc`).
    #[serde(skip_serializing_if = "Option::is_none", deserialize_with = "colour_tags::deserialize")]
    pub colour: Option<ColourTags>,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

/// One painted keyframe.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyframeEntry {
    /// The frame it stylizes.
    pub frame: usize,
    /// The painting's image file.
    pub image: String,
}

/// Guide weights and how the guides follow the tracking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Guides {
    /// Weight of the keyframe style itself.
    pub style: f32,
    /// Weight of the appearance guide (keyframe video frame against the current frame).
    pub video: f32,
    /// Weight of the temporal (de-flicker) guide.
    pub temporal: f32,
    /// Weight of the mapping (positional) guide.
    pub mapping: f32,
    /// Weight of the mask guide.
    pub mask: f32,
    /// Weight the mapping guide per pixel by the position tracker's confidence, so that it stops
    /// pulling where the track was lost. On by default.
    pub gate_mapping: bool,
    /// Weight the temporal guide per pixel by the consistency of the flow that warps the
    /// previous frame, so that it stops pulling where that warp is wrong (occlusions). Costs one
    /// extra flow per frame. Off by default.
    pub gate_temporal: bool,
    /// With `gate_temporal`: pixels by which inconsistent regions grow before the temporal guide
    /// is switched off there.
    pub temporal_gate_erosion: usize,
}

impl Default for Guides {
    fn default() -> Self {
        let v = VideoParams::default();
        let w = v.weights;
        Self {
            style: w.style,
            video: w.video,
            temporal: w.temporal,
            mapping: w.mapping,
            mask: w.mask,
            gate_mapping: v.gate_mapping,
            gate_temporal: v.gate_temporal,
            temporal_gate_erosion: v.temporal_gate_erosion,
        }
    }
}

/// Engine settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineSettings {
    /// `auto`, `gpu` or `cpu`.
    #[serde(with = "by_name")]
    pub backend: Backend,
    /// Synthesis preset: `compat`, `default`, `fast` or `high`.
    #[serde(with = "by_name")]
    pub preset: Preset,
    /// Weight of the uniformity term (how evenly the keyframe is used).
    #[serde(deserialize_with = "engine_values::uniformity")]
    pub uniformity: f32,
    /// Patch size, odd and at least 3.
    #[serde(deserialize_with = "engine_values::patch_size")]
    pub patch_size: usize,
    /// Random seed of the synthesis.
    pub seed: u32,
    /// Start every frame from the previous frame's result moved along the optical flow, running
    /// only the finest pyramid levels: about twice as fast per frame, at some cost in fidelity
    /// to the keyframe (a preview setting). Off by default.
    pub warm_start: bool,
    /// With `warm_start`: the finest pyramid levels run (at least 1).
    pub warm_start_levels: usize,
    /// With `warm_start`: search/vote iterations at each of those levels (at least 1).
    pub warm_start_iterations: usize,
}

impl Default for EngineSettings {
    fn default() -> Self {
        let p = Params::default();
        let w = WarmStartSettings::default();
        Self {
            backend: Backend::Auto,
            preset: p.preset,
            uniformity: p.uniformity_weight,
            patch_size: p.patch_size,
            seed: p.seed,
            warm_start: false,
            warm_start_levels: w.levels,
            warm_start_iterations: w.search_vote_iters,
        }
    }
}

/// Tuning of the adaptive and layered blends (`output.blend`); see [`BlendSettings`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Blending {
    /// How soft the per-pixel selection between the two passes is, in grey levels of patch
    /// error. Larger is steadier in time, smaller keeps more detail.
    pub softness: f32,
    /// Spatial smoothing of the errors before the selection, as a fraction of the frame size.
    pub error_smoothing: f32,
    /// How strongly the selection follows the position in the keyframe interval.
    pub temporal_prior: f32,
    /// How strongly the low frequencies stay at the linear crossfade.
    pub data_weight: f32,
    /// How much of the detail lost by mixing two stroke textures is restored (0 to 1).
    pub detail: f32,
    /// `blend = "layered"` only: the scale that separates shading from detail, as a fraction of
    /// the larger frame side. The base (shading and shadows) is crossfaded and only the finer
    /// detail is selected, so soft shadows keep their soft edges. 0 is the crossfade.
    pub shading_scale: f32,
}

impl Default for Blending {
    fn default() -> Self {
        let b = BlendSettings::default();
        Self {
            softness: b.softness,
            error_smoothing: b.error_smoothing,
            temporal_prior: b.temporal_prior,
            data_weight: b.data_weight,
            detail: b.detail,
            shading_scale: b.shading_scale,
        }
    }
}

impl Blending {
    fn settings(&self) -> BlendSettings {
        BlendSettings {
            softness: self.softness,
            error_smoothing: self.error_smoothing,
            temporal_prior: self.temporal_prior,
            data_weight: self.data_weight,
            detail: self.detail,
            shading_scale: self.shading_scale,
        }
    }
}

/// Output settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputSettings {
    /// Output frames: a pattern such as `out/[#####].png` or a directory. May be empty when
    /// [`video`](Self::video) is set (the frames then live in the render cache).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub frames: String,
    /// Output video file (for example `out.mov`), encoded with ffmpeg after the frames.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<String>,
    /// Bits per channel of the output: 8 or 16. 16-bit frames must be PNG.
    pub bit_depth: u32,
    /// Video codec: `auto` (by extension: ProRes for .mov, H.264 for .mp4, FFV1 for .mkv, VP9
    /// for .webm), `prores`, `prores4444`, `h264`, `h265`, `ffv1`, `png` or `vp9`.
    #[serde(with = "by_name")]
    pub codec: Codec,
    /// Constant-quality factor (CRF) for H.264, H.265 and VP9 (lower is better).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<u32>,
    /// How the two passes of a keyframe interval are combined: `layered` (the default: the
    /// shading crossfaded and the fine detail selected per pixel by synthesis error),
    /// `adaptive` (selection across all frequencies, sharper) or `linear` (crossfade). The
    /// `adaptive` and `layered` blends are tuned in `[blending]`.
    #[serde(with = "by_name")]
    pub blend: BlendMode,
    /// What to write at keyframe frames: `synthesized` or `painting`.
    #[serde(with = "by_name")]
    pub keyframes: KeyframeOutput,
}

impl Default for OutputSettings {
    fn default() -> Self {
        Self {
            frames: String::new(),
            video: None,
            bit_depth: 8,
            codec: Codec::Auto,
            quality: None,
            blend: BlendMode::default(),
            keyframes: KeyframeOutput::default(),
        }
    }
}

impl OutputSettings {
    /// The output bit depth.
    pub fn depth(&self) -> Result<BitDepth, Error> {
        BitDepth::from_bits(self.bit_depth).map_err(|e| Error::Project(format!("output.bit_depth: {e}")))
    }
}

/// A keyframe video project.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Project {
    /// Format version of the file ([`FORMAT_VERSION`]).
    pub format_version: u32,
    #[serde(default)]
    pub media: Media,
    #[serde(default, rename = "keyframe")]
    pub keyframes: Vec<KeyframeEntry>,
    #[serde(default)]
    pub guides: Guides,
    #[serde(default)]
    pub engine: EngineSettings,
    #[serde(default)]
    pub output: OutputSettings,
    #[serde(default)]
    pub blending: Blending,
    /// Directory relative paths are resolved against (the project file's directory).
    #[serde(skip)]
    base_dir: PathBuf,
    /// Name of the project (the file stem when loaded from a file), naming its cache.
    #[serde(skip)]
    name: String,
}

impl Project {
    /// A project with default settings for the frames at `frames` (pattern, directory or video
    /// file), writing to `output` (a pattern or directory, or a video file). Relative paths are
    /// relative to the current directory.
    pub fn new(frames: impl Into<String>, output: impl Into<String>) -> Self {
        let output = output.into();
        let mut out = OutputSettings::default();
        if is_video_path(&output) {
            out.video = Some(output);
        } else {
            out.frames = output;
        }
        Self {
            format_version: FORMAT_VERSION,
            media: Media { frames: frames.into(), ..Media::default() },
            keyframes: Vec::new(),
            guides: Guides::default(),
            engine: EngineSettings::default(),
            output: out,
            blending: Blending::default(),
            base_dir: PathBuf::new(),
            name: String::from("project"),
        }
    }

    /// Adds a keyframe painting for frame `frame`.
    pub fn add_keyframe(&mut self, frame: usize, image: impl Into<String>) -> &mut Self {
        self.keyframes.push(KeyframeEntry { frame, image: image.into() });
        self
    }

    /// Parses a project from TOML; relative paths will resolve against `base_dir`.
    pub fn from_toml(text: &str, base_dir: impl Into<PathBuf>) -> Result<Self, Error> {
        let head: Head = toml::from_str(text).map_err(|e| Error::Project(e.to_string()))?;
        head.check()?;
        let project: Project = if head.has_legacy_mapping_cost()? {
            let mut doc: toml::Table = toml::from_str(text).map_err(|e| Error::Project(e.to_string()))?;
            if let Some(toml::Value::Table(guides)) = doc.get_mut("guides") {
                guides.remove("mapping_cost");
            }
            toml::Value::Table(doc).try_into()
        } else {
            toml::from_str(text)
        }
        .map_err(|e| Error::Project(e.to_string()))?;
        Ok(project.located(base_dir.into()))
    }

    /// Parses a project from JSON, the interchange form of the same fields (for programs that
    /// would rather not write TOML); relative paths will resolve against `base_dir`.
    pub fn from_json(text: &str, base_dir: impl Into<PathBuf>) -> Result<Self, Error> {
        let head: Head = serde_json::from_str(text).map_err(|e| Error::Project(e.to_string()))?;
        head.check()?;
        let project: Project = if head.has_legacy_mapping_cost()? {
            let mut doc: serde_json::Value = serde_json::from_str(text).map_err(|e| Error::Project(e.to_string()))?;
            if let Some(guides) = doc.get_mut("guides").and_then(|g| g.as_object_mut()) {
                guides.remove("mapping_cost");
            }
            serde_json::from_value(doc)
        } else {
            serde_json::from_str(text)
        }
        .map_err(|e| Error::Project(e.to_string()))?;
        Ok(project.located(base_dir.into()))
    }

    /// Sets the directory of a parsed project.
    fn located(mut self, base_dir: PathBuf) -> Self {
        self.base_dir = base_dir;
        self.name = String::from("project");
        self
    }

    /// Loads a project file: TOML, or JSON when the file name ends in `.json`. Its directory
    /// ([`base_dir`](Self::base_dir)) is made absolute.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| Error::Io { path: path.to_owned(), source })?;
        // Absolute, so that the render cache does not depend on the working directory.
        let base = crate::paths::absolute(path.parent().unwrap_or(Path::new("")));
        let parsed = if is_json_path(path) { Self::from_json(&text, base) } else { Self::from_toml(&text, base) };
        let mut project = parsed.map_err(|e| match e {
            Error::Project(msg) => Error::Project(format!("{}: {msg}", path.display())),
            other => other,
        })?;
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            project.name = stem.to_owned();
        }
        Ok(project)
    }

    /// The project as TOML.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("projects always serialize")
    }

    /// The project as JSON (the same fields as the TOML form).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("projects always serialize")
    }

    /// Writes the project file (creating its directory): TOML, or JSON when the file name ends
    /// in `.json`. Relative paths keep their meaning only if the file goes to the directory they
    /// are relative to.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir).map_err(|source| Error::Io { path: dir.to_owned(), source })?;
        }
        let text = if is_json_path(path) { self.to_json() + "\n" } else { self.to_toml() };
        fs::write(path, text).map_err(|source| Error::Io { path: path.to_owned(), source })
    }

    /// The directory relative paths resolve against.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Sets the directory relative paths resolve against.
    pub fn set_base_dir(&mut self, dir: impl Into<PathBuf>) {
        self.base_dir = dir.into();
    }

    /// The project's name (its file stem, or `project`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets the project's name (it names the render cache).
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// `path` resolved against the project's directory.
    pub fn resolve(&self, path: &str) -> PathBuf {
        self.base_dir.join(path)
    }

    /// The default render cache directory: `.freesynth-cache/<name>` next to the project file.
    /// It holds rendered frames (when the output is only a video), decoded video input and the
    /// record of what was rendered from which inputs.
    pub fn default_cache_dir(&self) -> PathBuf {
        self.base_dir.join(".freesynth-cache").join(self.file_name())
    }

    /// The project's cache directory inside a cache folder shared by many projects
    /// ([`SessionOptions::cache_dir`](crate::SessionOptions::cache_dir)): its name followed by
    /// a hash of the project's directory, so that projects of the same name in different
    /// folders keep separate caches.
    pub fn cache_dir_in(&self, root: &Path) -> PathBuf {
        let place = crate::hashing::Hasher::new("project-dir")
            .str(&crate::paths::absolute(&self.base_dir).to_string_lossy())
            .finish();
        root.join(format!("{}-{}", self.file_name(), &place.short()[..8]))
    }

    /// The project's name made safe for a file name.
    fn file_name(&self) -> String {
        let name: String =
            self.name.chars().map(|c| if c.is_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect();
        if name.is_empty() || name.chars().all(|c| c == '.') { "project".into() } else { name }
    }

    /// The project with its directory made absolute (paths resolve the same way, whatever the
    /// working directory becomes).
    pub(crate) fn absolute(&self) -> Project {
        let mut p = self.clone();
        p.base_dir = crate::paths::absolute(&self.base_dir);
        p
    }

    /// The backend the project asks for.
    pub fn backend(&self) -> Backend {
        self.engine.backend
    }

    /// Checks the project against its files without rendering, decoding or writing anything:
    /// [`validate`](Self::validate), then the inputs exist, the frame range and the keyframes
    /// lie within the input, the paintings have the frame size, and no output would overwrite
    /// an input. Video inputs are probed with ffmpeg. A host can call this before opening a
    /// device, so that mistakes show up at once; renders make the same checks.
    pub fn check_inputs(&self) -> Result<(), Error> {
        crate::render::preflight(self)
    }

    /// The render's look parameters.
    pub fn video_params(&self) -> Result<VideoParams, Error> {
        let g = &self.guides;
        let e = &self.engine;
        check_patch_size(e.patch_size).map_err(Error::Project)?;
        check_uniformity(e.uniformity).map_err(Error::Project)?;
        let synthesis = Params {
            uniformity_weight: e.uniformity,
            patch_size: e.patch_size,
            seed: e.seed,
            ..Params::with_preset(e.preset)
        };
        let params = VideoParams {
            weights: GuideWeights {
                style: g.style,
                video: g.video,
                temporal: g.temporal,
                mapping: g.mapping,
                mask: g.mask,
            },
            synthesis,
            gate_mapping: g.gate_mapping,
            gate_temporal: g.gate_temporal,
            temporal_gate_erosion: g.temporal_gate_erosion,
            blend: self.output.blend,
            blending: self.blending.settings(),
            keyframe_output: self.output.keyframes,
            warm_start: e.warm_start.then_some(WarmStartSettings {
                levels: e.warm_start_levels,
                search_vote_iters: e.warm_start_iterations,
            }),
            ..VideoParams::default()
        };
        params.validate().map_err(|e| Error::Project(e.to_string()))?;
        Ok(params)
    }

    /// Checks that the project names its inputs, keyframes and output.
    pub fn validate(&self) -> Result<(), Error> {
        if self.media.frames.is_empty() {
            return Err(Error::Project("media.frames is not set".into()));
        }
        let o = &self.output;
        if o.frames.is_empty() && o.video.is_none() {
            return Err(Error::Project("set output.frames or output.video".into()));
        }
        if self.keyframes.is_empty() {
            return Err(Error::Project("the project has no keyframe".into()));
        }
        let mut frames: Vec<usize> = self.keyframes.iter().map(|k| k.frame).collect();
        frames.sort_unstable();
        if let Some(w) = frames.windows(2).find(|w| w[0] == w[1]) {
            return Err(Error::Project(format!("frame {} has two keyframes", w[0])));
        }
        if let (Some(first), Some(last)) = (self.media.first, self.media.last)
            && first > last
        {
            return Err(Error::Project(format!("media.first ({first}) is after media.last ({last})")));
        }
        if let Some(fps) = self.media.fps
            && !(fps.is_finite() && fps > 0.0)
        {
            return Err(Error::Project(format!("media.fps must be positive, not {fps}")));
        }
        let depth = o.depth()?;
        if !o.frames.is_empty() && is_video_path(&o.frames) {
            return Err(Error::Project(format!("output.frames is a video file ('{}'); use output.video", o.frames)));
        }
        if depth == BitDepth::Sixteen && !o.frames.is_empty() {
            let ext = Path::new(&o.frames).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
            if ext.is_some_and(|e| e != "png") {
                return Err(Error::Project("16-bit output needs PNG frames (output.frames must end in .png)".into()));
            }
        }
        if let Some(video) = &o.video {
            if !is_video_path(video) {
                return Err(Error::Project(format!("output.video '{video}' is not a video file name")));
            }
            let codec =
                o.codec.resolve(Path::new(video), false).map_err(|e| Error::Project(format!("output.codec: {e}")))?;
            if let (Some(q), Some(range)) = (o.quality, codec.quality_range())
                && !range.contains(&q)
            {
                return Err(Error::Project(format!(
                    "output.quality {q} is out of range for {} ({}-{})",
                    codec.name(),
                    range.start(),
                    range.end()
                )));
            }
        }
        self.video_params()?;
        Ok(())
    }
}

/// The format version alone, read before the rest of the file: a file from a newer version
/// may hold fields this library does not know, and should fail with the version message rather
/// than with "unknown field". Also reads the retired `guides.mapping_cost` key.
#[derive(Deserialize)]
struct Head {
    format_version: Option<u32>,
    guides: Option<HeadGuides>,
}

/// The retired `guides.mapping_cost` key. Older files may hold `mapping_cost = "guide"`, which
/// is what every project now does, so the key is dropped before the file is read.
#[derive(Deserialize)]
struct HeadGuides {
    mapping_cost: Option<String>,
}

impl Head {
    fn check(&self) -> Result<(), Error> {
        match self.format_version {
            None => Err(Error::Project(format!("format_version is missing (add format_version = {FORMAT_VERSION})"))),
            Some(v) if v == 0 || v > FORMAT_VERSION => Err(Error::Project(format!(
                "format version {v} is not supported (this library reads up to {FORMAT_VERSION})"
            ))),
            Some(_) => Ok(()),
        }
    }

    /// Whether the file holds the retired `guides.mapping_cost = "guide"` (to be ignored). Any
    /// other value is an error.
    fn has_legacy_mapping_cost(&self) -> Result<bool, Error> {
        match self.guides.as_ref().and_then(|g| g.mapping_cost.as_deref()) {
            None => Ok(false),
            Some("guide") => Ok(true),
            Some(other) => Err(Error::Project(format!(
                "guides.mapping_cost = \"{other}\" is not supported: the mapping guide is always an 8-bit \
                 guide image pair, so remove the key"
            ))),
        }
    }
}

/// True when `path` ends in `.json` (any case): a project in its JSON form.
fn is_json_path(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("json"))
}

/// Image file extensions picked up as keyframe paintings (compared without case).
const KEY_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "bmp", "tga"];

/// Splits a file stem into its text before the trailing digits and the digits.
fn trailing_digits(stem: &str) -> Option<(&str, &str)> {
    let start = stem.len() - stem.bytes().rev().take_while(u8::is_ascii_digit).count();
    (start < stem.len()).then(|| stem.split_at(start))
}

/// Finds numbered keyframe paintings: `spec` is a pattern such as `keys/[###].png` (its `#`
/// run stands for the frame number) or a directory whose image files end in a frame number
/// (`keys/000.png`, `keys/key_0042.jpg`). Unlike input frames, keyframes need not be
/// contiguous. Returns `(frame, path)` pairs sorted by frame.
pub fn find_keyframes(spec: &Path) -> Result<Vec<(usize, PathBuf)>, Error> {
    let name = spec.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let (dir, prefix, suffix) = if name.contains('#') {
        let end = name.rfind('#').expect("contains #");
        let start = name[..end].trim_end_matches('#').len();
        let (mut prefix, mut suffix) = (&name[..start], &name[end + 1..]);
        if prefix.ends_with('[') && suffix.starts_with(']') {
            prefix = &prefix[..prefix.len() - 1];
            suffix = &suffix[1..];
        }
        (spec.parent().map(Path::to_path_buf).unwrap_or_default(), Some(prefix.to_owned()), Some(suffix.to_owned()))
    } else {
        (spec.to_path_buf(), None, None)
    };
    let list_dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir.as_path() };
    let entries = fs::read_dir(list_dir).map_err(|source| Error::Io { path: list_dir.to_owned(), source })?;
    let mut found: Vec<(usize, PathBuf, String)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let digits_and_naming = match (&prefix, &suffix) {
            (Some(p), Some(s)) => file
                .strip_prefix(p.as_str())
                .and_then(|r| r.strip_suffix(s.as_str()))
                .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
                .map(|d| (d.to_owned(), String::new())),
            _ => {
                let (Some(stem), Some(ext)) =
                    (path.file_stem().and_then(|s| s.to_str()), path.extension().and_then(|e| e.to_str()))
                else {
                    continue;
                };
                if !KEY_EXTENSIONS.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                    continue;
                }
                trailing_digits(stem).map(|(p, d)| (d.to_owned(), format!("{p}*.{}", ext.to_ascii_lowercase())))
            }
        };
        let Some((digits, naming)) = digits_and_naming else { continue };
        let Ok(frame) = digits.parse::<usize>() else { continue };
        found.push((frame, path, naming));
    }
    // In a directory, use the most common naming (keyframes next to other numbered files).
    if prefix.is_none() {
        let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
        for (_, _, naming) in &found {
            *counts.entry(naming.as_str()).or_default() += 1;
        }
        if let Some(best) = counts.iter().max_by_key(|(_, n)| **n).map(|(k, _)| k.to_string()) {
            found.retain(|(_, _, naming)| *naming == best);
        }
    }
    let mut out: Vec<(usize, PathBuf)> = found.into_iter().map(|(f, p, _)| (f, p)).collect();
    out.sort();
    if let Some(w) = out.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(Error::Project(format!(
            "'{}' and '{}' are both keyframe {}",
            w[0].1.display(),
            w[1].1.display(),
            w[0].0
        )));
    }
    if out.is_empty() {
        return Err(Error::Project(format!("no numbered keyframe images found for '{}'", spec.display())));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_project_gets_defaults() {
        let text = r#"
            format_version = 1
            media.frames = "video/[###].jpg"
            output.frames = "out"
            [[keyframe]]
            frame = 0
            image = "keys/000.jpg"
        "#;
        let p = Project::from_toml(text, "/shots/fox").unwrap();
        p.validate().unwrap();
        assert_eq!(p.resolve(&p.media.frames), PathBuf::from("/shots/fox/video/[###].jpg"));
        assert_eq!(p.keyframes, [KeyframeEntry { frame: 0, image: "keys/000.jpg".into() }]);
        let v = p.video_params().unwrap();
        assert_eq!(v, VideoParams::default(), "a project gets the recommended look");
        assert_eq!(p.backend(), Backend::Auto);
    }

    #[test]
    fn round_trips_through_toml() {
        let mut p = Project::new("frames", "out/[####].png");
        p.add_keyframe(3, "k3.png").add_keyframe(40, "k40.png");
        p.media.mask = Some("mask".into());
        p.guides.mapping = 50.0;
        p.engine.preset = Preset::Fast;
        p.output.keyframes = KeyframeOutput::Painting;
        let text = p.to_toml();
        assert!(text.contains("preset = \"fast\"") && text.contains("keyframes = \"painting\""), "{text}");
        let back = Project::from_toml(&text, "").unwrap();
        assert_eq!(back, p);
        let json = p.to_json();
        assert!(json.contains("\"preset\": \"fast\""), "{json}");
        assert_eq!(Project::from_json(&json, "").unwrap(), p, "JSON is an equivalent form");
        assert!(Project::from_json("{\"format_version\": 1, \"engine\": {\"preset\": \"ultra\"}}", "").is_err());
        // A partial colour override survives both forms.
        p.media.colour = Some(ColourTags { matrix: Some("bt709".into()), ..ColourTags::default() });
        assert_eq!(Project::from_toml(&p.to_toml(), "").unwrap(), p);
        let json = p.to_json();
        assert!(!json.contains("null"), "{json}");
        assert_eq!(Project::from_json(&json, "").unwrap(), p);
        let explicit = r#"{"format_version": 1, "media": {"colour": {"matrix": "bt709", "range": null}}}"#;
        assert_eq!(Project::from_json(explicit, "").unwrap().media.colour, p.media.colour, "null is unset");
        let v = back.video_params().unwrap();
        assert_eq!(v.weights.mapping, 50.0);
        assert_eq!(v.synthesis.preset, Preset::Fast);
        assert_eq!(v.keyframe_output, KeyframeOutput::Painting);
    }

    #[test]
    fn video_quality_settings() {
        let text = r#"
            format_version = 1
            media.frames = "f"
            [[keyframe]]
            frame = 0
            image = "k.png"
            [guides]
            gate_mapping = false
            gate_temporal = true
            temporal_gate_erosion = 2
            [engine]
            warm_start = true
            warm_start_levels = 1
            [output]
            frames = "out"
            blend = "crossfade"
            [blending]
            softness = 0.5
        "#;
        let p = Project::from_toml(text, "").unwrap();
        let v = p.video_params().unwrap();
        assert!(!v.gate_mapping && v.gate_temporal && v.temporal_gate_erosion == 2);
        assert_eq!(v.warm_start, Some(WarmStartSettings { levels: 1, search_vote_iters: 3 }));
        assert_eq!(v.blend, BlendMode::Linear);
        assert_eq!(v.blending, BlendSettings { softness: 0.5, ..BlendSettings::default() });
        assert_eq!(Project::from_toml(&p.to_toml(), "").unwrap(), p);
        assert_eq!(Project::from_json(&p.to_json(), "").unwrap(), p);
        // The default project writes the recommended look out in full, so it can be edited.
        let text = Project::new("f", "o").to_toml();
        for field in ["gate_mapping = true", "blend = \"layered\"", "softness = 0.33", "warm_start = false"] {
            assert!(text.contains(field), "{field} in {text}");
        }
        // Bad values fail validation or the parse.
        let mut bad = p.clone();
        bad.engine.warm_start_levels = 0;
        assert!(bad.validate().is_err());
        let mut bad = p.clone();
        bad.blending.detail = 2.0;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn retired_mapping_cost_key() {
        // Older files may name the 8-bit mapping guide explicitly; the key is ignored.
        let text = "format_version = 1\nmedia.frames = 'f'\n[guides]\nmapping = 20.0\nmapping_cost = 'guide'\n";
        let p = Project::from_toml(text, "").unwrap();
        assert_eq!(p.guides.mapping, 20.0);
        assert_eq!(
            p,
            Project::from_toml("format_version = 1\nmedia.frames = 'f'\nguides.mapping = 20.0\n", "").unwrap()
        );
        assert!(!p.to_toml().contains("mapping_cost"));
        let json =
            r#"{"format_version": 1, "media": {"frames": "f"}, "guides": {"mapping": 20.0, "mapping_cost": "guide"}}"#;
        assert_eq!(Project::from_json(json, "").unwrap(), p);
        // Other values, and keys that only went with them, fail.
        for text in [
            "format_version = 1\nguides.mapping_cost = 'analytic'\n",
            "format_version = 1\nguides.mapping_cost = 'exact'\n",
        ] {
            let err = Project::from_toml(text, "").unwrap_err();
            assert!(err.to_string().contains("is not supported"), "{err}");
        }
        let err =
            Project::from_json(r#"{"format_version": 1, "guides": {"mapping_cost": "analytic"}}"#, "").unwrap_err();
        assert!(err.to_string().contains("guides.mapping_cost = \"analytic\" is not supported"), "{err}");
        let text = "format_version = 1\nguides.mapping_cost = 'guide'\nguides.mapping_robust = 8.0\n";
        assert!(Project::from_toml(text, "").unwrap_err().to_string().contains("mapping_robust"));
    }

    #[test]
    fn video_outputs_and_new_fields() {
        let mut p = Project::new("shot.mov", "out.mov");
        assert_eq!((p.output.frames.as_str(), p.output.video.as_deref()), ("", Some("out.mov")));
        p.add_keyframe(0, "k.png");
        p.media.start_number = 1;
        p.media.colour = Some(ColourTags::bt709());
        p.output.bit_depth = 16;
        p.output.codec = Codec::Ffv1;
        p.validate().unwrap();
        let text = p.to_toml();
        assert!(text.contains("codec = \"ffv1\"") && text.contains("start_number = 1"), "{text}");
        assert_eq!(Project::from_toml(&text, "").unwrap(), p);
        p.output.video = Some("out.gif".into());
        assert!(p.validate().is_err(), "not a video file name");
        p.output.video = None;
        assert!(p.validate().unwrap_err().to_string().contains("output.frames or output.video"));
        p.output.frames = "out/[###].jpg".into();
        assert!(p.validate().unwrap_err().to_string().contains("16-bit"));
        p.output.bit_depth = 12;
        assert!(p.validate().is_err());
        p.output.bit_depth = 8;
        p.validate().unwrap();
        p.add_keyframe(0, "again.png");
        assert!(p.validate().unwrap_err().to_string().contains("two keyframes"));
        assert!(Project::from_toml("format_version = 1\noutput.codec = 'divx'", "").is_err());
        p.keyframes.pop();
        p.output.codec = Codec::Auto;
        p.output.video = Some("out.webm".into());
        p.output.quality = Some(64);
        assert!(p.validate().unwrap_err().to_string().contains("out of range for vp9 (0-63)"));
        p.output.quality = Some(63);
        p.validate().unwrap();
        p.output.video = Some("out.mp4".into());
        assert!(p.validate().unwrap_err().to_string().contains("h264 (0-51)"));
        p.output.video = None;
        p.output.quality = None;
        let mut named = Project::new("f", "o");
        named.set_name("my shot");
        named.set_base_dir("/p");
        assert_eq!(named.default_cache_dir(), PathBuf::from("/p/.freesynth-cache/my_shot"));
        // In a shared cache folder, projects of one name in two folders stay apart.
        let shared = named.cache_dir_in(Path::new("/c"));
        assert!(shared.starts_with("/c") && shared.file_name().unwrap().to_str().unwrap().starts_with("my_shot-"));
        let mut elsewhere = named.clone();
        elsewhere.set_base_dir("/q");
        assert_ne!(elsewhere.cache_dir_in(Path::new("/c")), shared);
        assert_eq!(named.cache_dir_in(Path::new("/c")), shared, "stable");
    }

    #[test]
    fn json_project_files() {
        let dir = crate::test_util::scratch("json-project");
        let mut p = Project::new("frames/[###].png", "out.mov");
        p.add_keyframe(5, "keys/005.png");
        let path = dir.join("shot.json");
        p.save(&path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.trim_start().starts_with('{'), "{text}");
        let back = Project::load(&path).unwrap();
        assert_eq!(
            (back.keyframes.clone(), back.output.video.clone(), back.name()),
            (p.keyframes, p.output.video, "shot")
        );
        assert_eq!(back.base_dir(), crate::paths::absolute(&dir), "loaded projects have an absolute directory");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn finds_sparse_keyframes() {
        let dir = crate::test_util::scratch("find-keys");
        fs::create_dir_all(&dir).unwrap();
        for name in ["000.png", "099.png", "042.png", "notes.txt", "thumb.png"] {
            fs::write(dir.join(name), b"x").unwrap();
        }
        let keys = find_keyframes(&dir).unwrap();
        assert_eq!(keys.iter().map(|k| k.0).collect::<Vec<_>>(), [0, 42, 99]);
        let keys = find_keyframes(&dir.join("[###].png")).unwrap();
        assert_eq!(keys.len(), 3);
        assert!(find_keyframes(&dir.join("k_[##].png")).is_err());
        fs::write(dir.join("0099.png"), b"x").unwrap();
        assert!(find_keyframes(&dir).unwrap_err().to_string().contains("both keyframe 99"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn newer_versions_fail_with_the_version_message() {
        let newer = "format_version = 2\nmedia.frames = 'f'\n[look]\nstyle = 'oil'\n";
        let err = Project::from_toml(newer, "").unwrap_err().to_string();
        assert!(err.contains("format version 2 is not supported"), "{err}");
        let err = Project::from_json("{\"format_version\": 3, \"look\": {}}", "").unwrap_err().to_string();
        assert!(err.contains("format version 3 is not supported"), "{err}");
        let err = Project::from_toml("media.frames = 'f'\n", "").unwrap_err().to_string();
        assert!(err.contains("format_version is missing"), "{err}");
        // The current version still rejects unknown fields.
        let err = Project::from_toml("format_version = 1\n[look]\n", "").unwrap_err().to_string();
        assert!(err.contains("unknown field `look`"), "{err}");
    }

    #[test]
    fn colour_tags_are_checked_names() {
        let ok = "format_version = 1\n[media.colour]\nmatrix = 'smpte170m'\nrange = 'tv'\ntransfer = 'iec61966-2-1'\n";
        let p = Project::from_toml(ok, "").unwrap();
        let c = p.media.colour.clone().unwrap();
        assert_eq!((c.matrix.as_deref(), c.primaries), (Some("smpte170m"), None));
        assert_eq!(Project::from_toml(&p.to_toml(), "").unwrap(), p);
        // Filter-graph syntax (or any unknown name) fails the parse and points at its line.
        let evil = "format_version = 1\n[media.colour]\nmatrix = 'bt709,null[x];movie=http\\\\://h/a.mov,nullsink'\n";
        let err = Project::from_toml(evil, "").unwrap_err().to_string();
        assert!(err.contains("unknown colour matrix") && err.contains("line 3"), "{err}");
        for bad in ["range = 'full'", "primaries = 'p3'", "transfer = 'srgb'", "gamma = 'x'"] {
            let text = format!("format_version = 1\n[media.colour]\n{bad}\n");
            assert!(Project::from_toml(&text, "").is_err(), "{bad}");
        }
        let json = r#"{"format_version": 1, "media": {"colour": {"matrix": "bt709;x"}}}"#;
        assert!(Project::from_json(json, "").unwrap_err().to_string().contains("unknown colour matrix"));
    }

    #[test]
    fn rejects_bad_projects() {
        assert!(Project::from_toml("format_version = 1\nmedia.framez = 'x'", "").is_err());
        assert!(Project::from_toml("format_version = 99", "").is_err());
        let mut p = Project::new("f", "o");
        assert!(p.validate().is_err(), "no keyframe");
        p.add_keyframe(0, "k.png");
        p.validate().unwrap();
        p.guides.video = 0.0;
        assert!(p.validate().is_err());
        p.guides.video = 4.0;
        for bad in [0, 1, 4] {
            p.engine.patch_size = bad;
            assert!(p.validate().unwrap_err().to_string().contains("patch_size must be odd"), "{bad}");
        }
        p.engine.patch_size = 7;
        p.engine.uniformity = f32::NAN;
        assert!(p.validate().unwrap_err().to_string().contains("engine.uniformity"));
        // In a file, bad engine values point at their line.
        for (setting, expected) in [
            ("engine.patch_size = 4", "patch_size must be odd"),
            ("engine.uniformity = nan", "engine.uniformity"),
            ("engine.uniformity = -inf", "engine.uniformity"),
        ] {
            let text = format!("format_version = 1\nmedia.frames = 'f'\n{setting}\n");
            let err = Project::from_toml(&text, "").unwrap_err().to_string();
            assert!(err.contains(expected) && err.contains("line 3"), "{err}");
        }
        // Unknown names fail the parse and point at their line.
        for (setting, expected) in [
            ("engine.backend = 'cuda'", "unknown backend 'cuda'"),
            ("engine.preset = 'ultra'", "unknown preset 'ultra'"),
            ("output.blend = 'poisson'", "unknown blend mode 'poisson'"),
            ("output.keyframes = 'both'", "unknown keyframe output 'both'"),
        ] {
            let text = format!("format_version = 1\nmedia.frames = 'f'\n{setting}\n");
            let err = Project::from_toml(&text, "").unwrap_err().to_string();
            assert!(err.contains(expected) && err.contains("line 3"), "{err}");
        }
    }
}
