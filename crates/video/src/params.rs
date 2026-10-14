//! Look parameters (guides, engine, flow, blending) and resource options of a render.

use std::path::PathBuf;

use freesynth_core::Params;
use freesynth_flow::track::TrackParams;
use freesynth_flow::{DisParams, Preset as FlowPreset};

use crate::blend::{BlendMode, BlendSettings};
use crate::error::{VideoError, invalid};

/// Total weight of each guide layer. Within a layer the weight is split evenly over the layer's
/// informative channels, as in the engine's command-line tool.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuideWeights {
    /// The keyframe style itself.
    pub style: f32,
    /// Appearance guide: the keyframe's video frame against the current video frame.
    pub video: f32,
    /// Temporal (de-flicker) guide: the keyframe style against the chain's previous output
    /// warped to the current frame.
    pub temporal: f32,
    /// Mapping (positional) guide: keyframe coordinates against the coordinates tracked to the
    /// current frame.
    pub mapping: f32,
    /// Mask guide, when the shot has a mask: the keyframe's alpha against the frame's mask.
    pub mask: f32,
}

impl GuideWeights {
    /// The classic weights: style 1, video 4, temporal 1, mapping 10, mask 1. The mapping guide
    /// dominates, so the style stays attached to the moving content.
    pub const CLASSIC: GuideWeights = GuideWeights { style: 1.0, video: 4.0, temporal: 1.0, mapping: 10.0, mask: 1.0 };

    fn validate(&self) -> Result<(), VideoError> {
        let all = [self.style, self.video, self.temporal, self.mapping, self.mask];
        if all.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return invalid("guide weights must be finite and non-negative");
        }
        if self.video == 0.0 {
            return invalid("the video guide weight must be positive");
        }
        Ok(())
    }
}

impl Default for GuideWeights {
    fn default() -> Self {
        Self::CLASSIC
    }
}

/// What is written for a keyframe's own frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KeyframeOutput {
    /// The synthesized frame, which is temporally consistent with its neighbours.
    #[default]
    Synthesized,
    /// The keyframe painting itself.
    Painting,
}

impl KeyframeOutput {
    /// The setting's lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            KeyframeOutput::Synthesized => "synthesized",
            KeyframeOutput::Painting => "painting",
        }
    }
}

impl std::str::FromStr for KeyframeOutput {
    type Err = VideoError;

    fn from_str(s: &str) -> Result<Self, VideoError> {
        match s {
            "synthesized" => Ok(KeyframeOutput::Synthesized),
            "painting" => Ok(KeyframeOutput::Painting),
            _ => invalid(format!("unknown keyframe output '{s}' (expected synthesized or painting)")),
        }
    }
}

/// The video warm start ([`VideoParams::warm_start`]): every frame of a chain starts from the
/// previous frame's final nearest-neighbour field moved along the optical flow, and runs only
/// the finest pyramid levels with fewer search/vote iterations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarmStartSettings {
    /// Finest pyramid levels run (at least 1).
    pub levels: usize,
    /// Search/vote iterations at each of those levels (at least 1).
    pub search_vote_iters: usize,
}

impl Default for WarmStartSettings {
    /// The 2 finest levels with 3 search/vote iterations each.
    fn default() -> Self {
        Self { levels: 2, search_vote_iters: 3 }
    }
}

/// Everything that determines how a shot looks.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoParams {
    /// Guide weights.
    pub weights: GuideWeights,
    /// Synthesis engine parameters (preset, patch size, uniformity, seed, ...). The same for
    /// every frame.
    pub synthesis: Params,
    /// Optical flow parameters.
    pub flow: DisParams,
    /// Position tracking (anchoring and confidence) for the mapping guide.
    pub tracking: TrackParams,
    /// Scale the mapping guide per pixel by the tracker's confidence, so that it stops pulling
    /// where the tracked position is unreliable (disocclusions, broken tracks).
    pub gate_mapping: bool,
    /// Scale the temporal guide per pixel by the forward-backward consistency of the one-step
    /// flow that warps the previous output, so that it stops pulling where that warp is wrong
    /// (content that was hidden in the previous frame, flow errors). Costs one extra flow per
    /// frame (the reverse direction).
    pub gate_temporal: bool,
    /// With [`gate_temporal`](Self::gate_temporal): pixels by which regions that fail the
    /// consistency check grow before the temporal guide is switched off there.
    pub temporal_gate_erosion: usize,
    /// How the two passes of an interval are combined.
    pub blend: BlendMode,
    /// Settings of the adaptive and layered blends ([`BlendMode`]).
    pub blending: BlendSettings,
    /// What is written at keyframe frames.
    pub keyframe_output: KeyframeOutput,
    /// Start each frame of a chain from the previous frame's result instead of from scratch: a
    /// speed option (`None`: every frame runs the full pyramid).
    pub warm_start: Option<WarmStartSettings>,
}

impl VideoParams {
    /// The classic preset: style 1, video 4, temporal 1, mapping 10, mask 1
    /// ([`GuideWeights::CLASSIC`]), an ungated 8-bit mapping guide and a linear crossfade
    /// between the passes of an interval. Positions come from anchored tracking with
    /// forward-backward checks, and the synthesis engine runs its default preset.
    pub fn classic() -> Self {
        Self {
            weights: GuideWeights::CLASSIC,
            synthesis: Params::default(),
            flow: DisParams::preset(FlowPreset::Medium),
            tracking: TrackParams::default(),
            gate_mapping: false,
            gate_temporal: false,
            temporal_gate_erosion: 0,
            blend: BlendMode::Linear,
            blending: BlendSettings::default(),
            keyframe_output: KeyframeOutput::Synthesized,
            warm_start: None,
        }
    }

    /// Checks the parameters for consistency.
    pub fn validate(&self) -> Result<(), VideoError> {
        self.weights.validate()?;
        self.flow.validate()?;
        self.tracking.validate()?;
        self.blending.validate()?;
        if let Some(w) = self.warm_start
            && (w.levels == 0 || w.search_vote_iters == 0)
        {
            return invalid("a warm start needs at least one level and one search/vote iteration");
        }
        if self.synthesis.return_errors {
            return invalid(
                "synthesis.return_errors must be off: the renderer asks for errors when the blend needs them",
            );
        }
        Ok(())
    }
}

impl Default for VideoParams {
    /// The recommended look: the classic guide weights with the mapping guide gated by the
    /// tracker's confidence, and the [layered](BlendMode::Layered) blend between passes.
    /// Compared with [`classic`](VideoParams::classic) it keeps the style attached to tracked
    /// content without dragging it into regions whose track was lost, and it keeps the passes'
    /// detail in the middle of keyframe intervals instead of crossfading it away, while keeping
    /// soft shadows soft.
    fn default() -> Self {
        Self { gate_mapping: true, blend: BlendMode::Layered, ..Self::classic() }
    }
}

/// Where optical flow is computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlowBackend {
    /// On the render's device, next to the synthesis (the shared GPU device, or the CPU).
    Device,
    /// Always on the CPU (the default). With GPU synthesis this keeps flow off the GPU queue,
    /// where every flow readback would wait for the syntheses queued before it; the CPU is
    /// mostly idle during GPU synthesis anyway.
    #[default]
    Cpu,
}

/// How much memory and parallelism a render may use. [`RenderOptions::resources`] shows how
/// they are resolved for a shot.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderOptions {
    /// Total memory the render should stay within, in bytes: running chains, the frame cache,
    /// the pass buffers and the flow cache. Decides the default parallelism and shrinks the
    /// caches when needed (only with a [`spill_dir`](Self::spill_dir)). Default:
    /// [`default_memory_budget`](crate::default_memory_budget), half the memory, at most 4 GiB.
    pub memory_budget: usize,
    /// Keyframe chains synthesized at the same time. `None` runs as many as the shot has, up to
    /// [`MAX_DEFAULT_PARALLEL`](crate::resources::MAX_DEFAULT_PARALLEL), the CPU threads and
    /// what fits in the memory budget.
    pub parallel_chains: Option<usize>,
    /// The most bytes of decoded input frames kept in memory, per input sequence (less when
    /// the memory budget requires it).
    pub frame_cache_bytes: usize,
    /// The most bytes of synthesized frames waiting for their blend partner kept in memory
    /// (less when the memory budget requires it).
    pub buffer_bytes: usize,
    /// Bytes of flow fields kept for reuse (at least three per parallel chain, for up to four
    /// chains, are always kept).
    pub flow_cache_bytes: usize,
    /// Where optical flow is computed.
    pub flow_backend: FlowBackend,
    /// Directory for frames beyond the memory budgets. Each render spills into its own
    /// subdirectory, so renders sharing the directory do not collide; the subdirectory is
    /// removed afterwards, and so is this directory if the render created it and it is empty.
    /// `None` keeps everything in memory regardless of the budgets. Avoid RAM-backed temporary
    /// directories here.
    pub spill_dir: Option<PathBuf>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            memory_budget: crate::resources::default_memory_budget(),
            parallel_chains: None,
            frame_cache_bytes: 1 << 30,
            buffer_bytes: 1 << 30,
            flow_cache_bytes: 256 << 20,
            flow_backend: FlowBackend::Cpu,
            spill_dir: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_valid() {
        let d = VideoParams::default();
        d.validate().unwrap();
        assert!(d.gate_mapping && !d.gate_temporal);
        assert_eq!(d.blend, BlendMode::Layered);
        let p = VideoParams::classic();
        p.validate().unwrap();
        assert!(!p.gate_mapping && p.blend == BlendMode::Linear);
        assert_eq!(p.weights, GuideWeights { style: 1.0, video: 4.0, temporal: 1.0, mapping: 10.0, mask: 1.0 });
        let mut bad = p.clone();
        bad.weights.mapping = -1.0;
        assert!(bad.validate().is_err());
        let mut bad = p;
        bad.synthesis.return_errors = true;
        assert!(bad.validate().is_err());
        for k in [KeyframeOutput::Synthesized, KeyframeOutput::Painting] {
            assert_eq!(k.name().parse::<KeyframeOutput>().unwrap(), k);
        }
        assert!("both".parse::<KeyframeOutput>().is_err());
        let cold = WarmStartSettings { levels: 0, ..WarmStartSettings::default() };
        let bad = VideoParams { warm_start: Some(cold), ..VideoParams::default() };
        assert!(bad.validate().is_err());
        let ok = VideoParams { warm_start: Some(WarmStartSettings::default()), ..VideoParams::default() };
        ok.validate().unwrap();
    }
}
