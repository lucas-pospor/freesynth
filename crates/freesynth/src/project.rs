//! Project files: a shot, its keyframes, the look and the output, in TOML.
//!
//! ```toml
//! format_version = 1
//!
//! [media]
//! frames = "video/[###].jpg"   # a frame pattern ('#' run = zero-padded number) or a directory
//! # mask = "mask/[###].png"    # optional mask sequence
//! # first = 0                  # optional frame range (default: every frame found)
//! # last = 99
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
//! gate_mapping = false         # weight the mapping guide by tracking confidence
//!
//! [engine]
//! backend = "auto"             # auto, gpu or cpu
//! preset = "default"           # compat, default, fast or high
//! uniformity = 3500.0
//! patch_size = 5
//! seed = 1
//!
//! [output]
//! frames = "out/[#####].png"   # a frame pattern or a directory (NNNNN.png inside)
//! blend = "linear"
//! keyframes = "synthesized"    # or "painting"
//! ```
//!
//! Every field has a default, so a project needs only `media.frames`, the keyframes and
//! `output.frames`. Relative paths are relative to the project file. Unknown fields and
//! unknown setting names (a backend, preset, blend mode or keyframe output) are errors that
//! point at their line, so that a misspelt setting is not silently ignored.

use std::fs;
use std::path::{Path, PathBuf};

use freesynth_core::{Params, Preset};
use freesynth_video::{Backend, BlendMode, GuideWeights, KeyframeOutput, VideoError, VideoParams};
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Version of the project format written by this library.
pub const FORMAT_VERSION: u32 = 1;

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

/// The input sequences.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Media {
    /// Input video frames: a pattern such as `video/[####].png` or a directory of numbered
    /// images.
    pub frames: String,
    /// Optional mask sequence, as a pattern or a directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
    /// First frame to render (default: the first frame found).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first: Option<usize>,
    /// Last frame to render (default: the last frame found).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last: Option<usize>,
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

/// Guide weights.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Guides {
    pub style: f32,
    pub video: f32,
    pub temporal: f32,
    pub mapping: f32,
    pub mask: f32,
    /// Weight the mapping guide per pixel by the position tracker's confidence.
    pub gate_mapping: bool,
}

impl Default for Guides {
    fn default() -> Self {
        let w = GuideWeights::CLASSIC;
        Self { style: w.style, video: w.video, temporal: w.temporal, mapping: w.mapping, mask: w.mask, gate_mapping: false }
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
    pub uniformity: f32,
    /// Patch size, odd and at least 3.
    pub patch_size: usize,
    /// Random seed of the synthesis.
    pub seed: u32,
}

impl Default for EngineSettings {
    fn default() -> Self {
        let p = Params::default();
        Self {
            backend: Backend::Auto,
            preset: p.preset,
            uniformity: p.uniformity_weight,
            patch_size: p.patch_size,
            seed: p.seed,
        }
    }
}

/// Output settings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputSettings {
    /// Output frames: a pattern such as `out/[#####].png` or a directory.
    pub frames: String,
    /// How the two passes of a keyframe interval are combined: `linear`.
    #[serde(with = "by_name")]
    pub blend: BlendMode,
    /// What to write at keyframe frames: `synthesized` or `painting`.
    #[serde(with = "by_name")]
    pub keyframes: KeyframeOutput,
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
    /// Directory relative paths are resolved against (the project file's directory).
    #[serde(skip)]
    base_dir: PathBuf,
}

impl Project {
    /// A project with default settings for the frames at `frames` (pattern or directory),
    /// writing to `output` (pattern or directory). Relative paths are relative to the current
    /// directory.
    pub fn new(frames: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            media: Media { frames: frames.into(), ..Media::default() },
            keyframes: Vec::new(),
            guides: Guides::default(),
            engine: EngineSettings::default(),
            output: OutputSettings { frames: output.into(), ..OutputSettings::default() },
            base_dir: PathBuf::new(),
        }
    }

    /// Adds a keyframe painting for frame `frame`.
    pub fn add_keyframe(&mut self, frame: usize, image: impl Into<String>) -> &mut Self {
        self.keyframes.push(KeyframeEntry { frame, image: image.into() });
        self
    }

    /// Parses a project from TOML; relative paths will resolve against `base_dir`.
    pub fn from_toml(text: &str, base_dir: impl Into<PathBuf>) -> Result<Self, Error> {
        let mut project: Project = toml::from_str(text).map_err(|e| Error::Project(e.to_string()))?;
        if project.format_version == 0 || project.format_version > FORMAT_VERSION {
            return Err(Error::Project(format!(
                "format version {} is not supported (this library reads up to {FORMAT_VERSION})",
                project.format_version
            )));
        }
        project.base_dir = base_dir.into();
        Ok(project)
    }

    /// Loads a project file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| Error::Io { path: path.to_owned(), source })?;
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        Self::from_toml(&text, base).map_err(|e| match e {
            Error::Project(msg) => Error::Project(format!("{}: {msg}", path.display())),
            other => other,
        })
    }

    /// The project as TOML.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("projects always serialize")
    }

    /// Writes the project file. Relative paths keep their meaning only if the file goes to the
    /// directory they are relative to.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        fs::write(path, self.to_toml()).map_err(|source| Error::Io { path: path.to_owned(), source })
    }

    /// The directory relative paths resolve against.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Sets the directory relative paths resolve against.
    pub fn set_base_dir(&mut self, dir: impl Into<PathBuf>) {
        self.base_dir = dir.into();
    }

    /// `path` resolved against the project's directory.
    pub fn resolve(&self, path: &str) -> PathBuf {
        self.base_dir.join(path)
    }

    /// The backend the project asks for.
    pub fn backend(&self) -> Backend {
        self.engine.backend
    }

    /// The render's look parameters.
    pub fn video_params(&self) -> Result<VideoParams, Error> {
        let g = &self.guides;
        let e = &self.engine;
        let synthesis = Params { uniformity_weight: e.uniformity, patch_size: e.patch_size, seed: e.seed, ..Params::with_preset(e.preset) };
        let params = VideoParams {
            weights: GuideWeights { style: g.style, video: g.video, temporal: g.temporal, mapping: g.mapping, mask: g.mask },
            synthesis,
            gate_mapping: g.gate_mapping,
            blend: self.output.blend,
            keyframe_output: self.output.keyframes,
            ..VideoParams::classic()
        };
        params.validate().map_err(|e| Error::Project(e.to_string()))?;
        Ok(params)
    }

    /// Checks that the project names its inputs, keyframes and output.
    pub fn validate(&self) -> Result<(), Error> {
        if self.media.frames.is_empty() {
            return Err(Error::Project("media.frames is not set".into()));
        }
        if self.output.frames.is_empty() {
            return Err(Error::Project("output.frames is not set".into()));
        }
        if self.keyframes.is_empty() {
            return Err(Error::Project("the project has no keyframe".into()));
        }
        if let (Some(first), Some(last)) = (self.media.first, self.media.last)
            && first > last
        {
            return Err(Error::Project(format!("media.first ({first}) is after media.last ({last})")));
        }
        self.video_params()?;
        Ok(())
    }
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
        assert_eq!(v, VideoParams::classic());
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
        let v = back.video_params().unwrap();
        assert_eq!(v.weights.mapping, 50.0);
        assert_eq!(v.synthesis.preset, Preset::Fast);
        assert_eq!(v.keyframe_output, KeyframeOutput::Painting);
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
