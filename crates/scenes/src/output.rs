//! The on-disk scene layout and its `scene.toml` description.
//!
//! A scene directory contains:
//!
//! | Path | Content |
//! |---|---|
//! | `scene.toml` | [`SceneInfo`], written last (its presence means the scene is complete) |
//! | `frames/NNNNN.png` | input video frames, numbered from `00000` |
//! | `keys/NNNNN.png` | keyframe styles, named after the frame they stylize |
//! | `truth/NNNNN.png` | ground-truth stylization of every frame (generated scenes only) |
//! | `flow_fw/NNNNN.flo` | flow from frame `NNNNN` to frame `NNNNN + 1` |
//! | `flow_bw/NNNNN.flo` | flow from frame `NNNNN + 1` back to frame `NNNNN` |
//! | `occ_fw/NNNNN.png`, `occ_bw/NNNNN.png` | occlusion masks of those flows, 255 = occluded |
//! | `flow_fw_G/`, `flow_bw_G/`, `occ_fw_G/`, `occ_bw_G/` | the same for a gap of `G` frames |
//!
//! Both flow directions are indexed by the first frame of the pair, so `flow_fw/00007.flo` and
//! `flow_bw/00007.flo` are the two directions between frames 7 and 8. Flow files use the
//! Middlebury format ([`freesynth_flow::flo`]). A forward occlusion mask is defined on the
//! pixels of the earlier frame, a backward one on the pixels of the later frame.

use std::collections::VecDeque;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ExtendedColorType, ImageEncoder};
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::scene::{LayerMap, Scene, SceneParams};

/// Name of the scene description file.
pub const SCENE_FILE: &str = "scene.toml";
/// Version of the scene layout written by this crate.
pub const FORMAT_VERSION: u32 = 1;
/// Licence of every generated scene.
pub const GENERATED_LICENCE: &str = "CC0-1.0";

/// Direction of a flow field within a frame pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// From the earlier frame to the later one.
    Forward,
    /// From the later frame to the earlier one.
    Backward,
}

impl Direction {
    fn suffix(self) -> &'static str {
        match self {
            Direction::Forward => "fw",
            Direction::Backward => "bw",
        }
    }
}

fn numbered(dir: &Path, sub: &str, index: usize, ext: &str) -> PathBuf {
    dir.join(sub).join(format!("{index:05}.{ext}"))
}

fn gap_dir(kind: &str, direction: Direction, gap: usize) -> String {
    match gap {
        1 => format!("{kind}_{}", direction.suffix()),
        g => format!("{kind}_{}_{g}", direction.suffix()),
    }
}

/// `frames/NNNNN.png`.
pub fn frame_path(dir: &Path, t: usize) -> PathBuf {
    numbered(dir, "frames", t, "png")
}

/// `keys/NNNNN.png`.
pub fn key_path(dir: &Path, t: usize) -> PathBuf {
    numbered(dir, "keys", t, "png")
}

/// `truth/NNNNN.png`.
pub fn truth_path(dir: &Path, t: usize) -> PathBuf {
    numbered(dir, "truth", t, "png")
}

/// Flow between frames `first` and `first + gap`, in `direction`.
pub fn flow_path(dir: &Path, direction: Direction, gap: usize, first: usize) -> PathBuf {
    numbered(dir, &gap_dir("flow", direction, gap), first, "flo")
}

/// Occlusion mask of the flow at [`flow_path`].
pub fn occlusion_path(dir: &Path, direction: Direction, gap: usize, first: usize) -> PathBuf {
    numbered(dir, &gap_dir("occ", direction, gap), first, "png")
}

/// Contents of `scene.toml`. Generated scenes and fetched clips use the same description.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneInfo {
    pub format_version: u32,
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    /// Frames that have a style in `keys/` (may be empty for footage without painted keys).
    pub keyframes: Vec<usize>,
    /// Directory of the guide images (the input video), relative to the scene directory.
    pub guide: String,
    /// Style family: `rigid`, `painterly` or `hatching` for generated scenes, `footage` for
    /// real clips.
    pub family: String,
    /// SPDX licence identifier of the scene's images.
    pub licence: String,
    /// Credit line required by the licence, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<String>,
    /// Where the images come from: the generator name and version, or a URL.
    pub source: String,
    /// How the flow files were made (`exact` for generated scenes), if there are any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flow: Option<String>,
    /// Frame gaps that have flow files.
    #[serde(default)]
    pub flow_gaps: Vec<usize>,
    /// Whether `occ_*` occlusion masks are present.
    #[serde(default)]
    pub occlusion_masks: bool,
    /// Whether `truth/` holds a ground-truth stylization of every frame.
    #[serde(default)]
    pub ground_truth: bool,
    /// Parameters that regenerate the scene exactly (generated scenes only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<SceneParams>,
}

impl SceneInfo {
    /// Reads `scene.toml` from a scene directory.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let path = dir.join(SCENE_FILE);
        let text = fs::read_to_string(&path)?;
        toml::from_str(&text).map_err(|e| Error::Parse(format!("{}: {e}", path.display())))
    }

    /// Writes `scene.toml` into a scene directory.
    pub fn save(&self, dir: &Path) -> Result<(), Error> {
        let body = toml::to_string(self).map_err(|e| Error::Parse(e.to_string()))?;
        let header = "# FreeSynth scene description. Layout: frames/NNNNN.png, keys/NNNNN.png, truth/NNNNN.png,\n\
                      # flow_fw/NNNNN.flo (frame N to N+1), flow_bw/NNNNN.flo (frame N+1 to N), occ_fw/ and\n\
                      # occ_bw/ masks (255 = occluded); a suffix _G marks a gap of G frames.\n";
        fs::write(dir.join(SCENE_FILE), format!("{header}{body}"))?;
        Ok(())
    }
}

/// Generates the scene described by `params` into `dir` (created if needed) and returns its
/// description. Frames are rendered and written one at a time, so memory use stays at a few
/// frames whatever the length of the sequence.
pub fn write_scene(params: &SceneParams, name: &str, dir: &Path) -> Result<SceneInfo, Error> {
    let scene = Scene::new(params)?;
    let keys = params.resolved_keyframes();
    let gaps = params.resolved_flow_gaps();
    let mut subdirs = vec!["frames".to_owned(), "keys".to_owned(), "truth".to_owned()];
    for &g in &gaps {
        for d in [Direction::Forward, Direction::Backward] {
            subdirs.push(gap_dir("flow", d, g));
            subdirs.push(gap_dir("occ", d, g));
        }
    }
    for sub in &subdirs {
        fs::create_dir_all(dir.join(sub))?;
    }
    let _ = fs::remove_file(dir.join(SCENE_FILE));

    let max_gap = gaps.last().copied().unwrap_or(0);
    let mut recent: VecDeque<LayerMap> = VecDeque::with_capacity(max_gap + 1);
    let (w, h) = (params.width as u32, params.height as u32);
    for t in 0..params.frames {
        let frame = scene.render(t);
        save_png(&frame_path(dir, t), frame.image.as_raw(), w, h, ExtendedColorType::Rgb8)?;
        save_png(&truth_path(dir, t), frame.truth.as_raw(), w, h, ExtendedColorType::Rgb8)?;
        if keys.binary_search(&t).is_ok() {
            save_png(&key_path(dir, t), frame.truth.as_raw(), w, h, ExtendedColorType::Rgb8)?;
        }
        if recent.len() > max_gap {
            recent.pop_front();
        }
        recent.push_back(frame.layers);
        let current = recent.back().expect("just pushed");
        for &g in gaps.iter().filter(|&&g| g <= t) {
            let first = t - g;
            let earlier = &recent[recent.len() - 1 - g];
            let fw = scene.flow(first, t, earlier, current);
            let bw = scene.flow(t, first, current, earlier);
            for (d, f) in [(Direction::Forward, fw), (Direction::Backward, bw)] {
                f.flow.save(flow_path(dir, d, g, first))?;
                save_png(&occlusion_path(dir, d, g, first), f.occlusion.as_raw(), w, h, ExtendedColorType::L8)?;
            }
        }
    }

    let info = SceneInfo {
        format_version: FORMAT_VERSION,
        name: name.to_owned(),
        width: params.width,
        height: params.height,
        frames: params.frames,
        keyframes: keys,
        guide: "frames".to_owned(),
        family: params.family.to_string(),
        licence: GENERATED_LICENCE.to_owned(),
        attribution: None,
        source: concat!("freesynth-scenes ", env!("CARGO_PKG_VERSION")).to_owned(),
        flow: (!gaps.is_empty()).then(|| "exact".to_owned()),
        flow_gaps: gaps,
        occlusion_masks: true,
        ground_truth: true,
        generator: Some(params.clone()),
    };
    info.save(dir)?;
    Ok(info)
}

/// Writes an 8-bit PNG with fast compression (scenes are regenerated often).
fn save_png(path: &Path, data: &[u8], w: u32, h: u32, color: ExtendedColorType) -> Result<(), Error> {
    let mut file = BufWriter::new(fs::File::create(path)?);
    PngEncoder::new_with_quality(&mut file, CompressionType::Fast, FilterType::Adaptive)
        .write_image(data, w, h, color)?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_follow_the_layout() {
        let d = Path::new("s");
        assert_eq!(frame_path(d, 7), Path::new("s/frames/00007.png"));
        assert_eq!(key_path(d, 29), Path::new("s/keys/00029.png"));
        assert_eq!(flow_path(d, Direction::Forward, 1, 3), Path::new("s/flow_fw/00003.flo"));
        assert_eq!(flow_path(d, Direction::Backward, 5, 0), Path::new("s/flow_bw_5/00000.flo"));
        assert_eq!(occlusion_path(d, Direction::Backward, 1, 12), Path::new("s/occ_bw/00012.png"));
    }

    #[test]
    fn scene_info_round_trips_through_toml() {
        let info = SceneInfo {
            format_version: FORMAT_VERSION,
            name: "x".into(),
            width: 64,
            height: 48,
            frames: 3,
            keyframes: vec![0, 2],
            guide: "frames".into(),
            family: "painterly".into(),
            licence: GENERATED_LICENCE.into(),
            attribution: Some("someone".into()),
            source: "test".into(),
            flow: Some("exact".into()),
            flow_gaps: vec![1],
            occlusion_masks: true,
            ground_truth: true,
            generator: Some(SceneParams::default()),
        };
        let text = toml::to_string(&info).unwrap();
        assert_eq!(toml::from_str::<SceneInfo>(&text).unwrap(), info);
    }
}
