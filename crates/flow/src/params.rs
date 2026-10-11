//! DIS parameters, presets and the pyramid plan shared by every backend.

use crate::error::{FlowError, invalid};

/// Edge length of the square patches aligned by the inverse search. The GPU kernels are built
/// around it (one thread per patch row), so it is fixed.
pub const PATCH_SIZE: usize = 8;

/// Parameter presets, in increasing order of cost and accuracy. The first three follow the
/// settings that are common for DIS (finest level, patch stride, iteration counts); the
/// smoothness weights and the parallel propagation are this implementation's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preset {
    /// Quarter resolution, sparse patches, one propagation pass, two refinement iterations on
    /// the finest level.
    UltraFast,
    /// Quarter resolution, sparse patches, two propagation passes, refinement on half and
    /// quarter resolution.
    Fast,
    /// Half resolution, dense patches, four propagation passes, refinement from full to
    /// quarter resolution.
    #[default]
    Medium,
    /// [`Medium`](Preset::Medium) with twice the refinement iterations: 4 to 14% lower error at
    /// one-frame gaps on the test sequences (little change at larger gaps) for about 1.7 times
    /// the GPU time. Meant for the GPU; on the CPU the refinement is most of the cost.
    High,
}

impl Preset {
    /// All presets, fastest first.
    pub const ALL: [Preset; 4] = [Preset::UltraFast, Preset::Fast, Preset::Medium, Preset::High];

    /// The preset's name in lower case (`ultrafast`, `fast`, `medium`, `high`).
    pub fn name(self) -> &'static str {
        match self {
            Preset::UltraFast => "ultrafast",
            Preset::Fast => "fast",
            Preset::Medium => "medium",
            Preset::High => "high",
        }
    }
}

impl std::str::FromStr for Preset {
    type Err = FlowError;

    fn from_str(s: &str) -> Result<Self, FlowError> {
        Preset::ALL.into_iter().find(|p| p.name().eq_ignore_ascii_case(s)).ok_or_else(|| {
            FlowError::InvalidInput(format!("unknown flow preset '{s}' (ultrafast, fast, medium, high)"))
        })
    }
}

impl std::fmt::Display for Preset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Parameters of Dense Inverse Search (Kroeger, Timofte, Dai, Van Gool, "Fast Optical Flow
/// using Dense Inverse Search", ECCV 2016).
///
/// Pyramid level `l` has the input size divided by `2^l` (rounded down). Patches are aligned
/// from the coarsest level down to [`finest_level`](Self::finest_level); the variational
/// refinement can continue on finer levels ([`vr_finest_level`](Self::vr_finest_level)); the
/// result is upsampled to full resolution.
#[derive(Clone, Debug, PartialEq)]
pub struct DisParams {
    /// Finest pyramid level on which patches are aligned (0 = full resolution).
    pub finest_level: usize,
    /// Coarsest pyramid level; `None` picks it from the image size: the level where the larger
    /// side is closest to four patches (see [`DisParams::plan`]).
    pub coarsest_level: Option<usize>,
    /// Distance between neighbouring patches in pixels (patches are [`PATCH_SIZE`] wide, so a
    /// stride below that makes them overlap).
    pub patch_stride: usize,
    /// Gauss-Newton iterations of the first inverse search pass, per patch and level.
    pub gd_iterations: usize,
    /// A patch's search stops early once a Gauss-Newton step moves it by less than this many
    /// pixels (of its level); 0 always runs every iteration. Most patches converge in a few
    /// iterations, so this saves most of the search time at no measurable cost in accuracy.
    pub gd_tolerance: f32,
    /// Subtract the patch means before matching, which makes the search robust to local
    /// brightness changes.
    pub mean_normalization: bool,
    /// Propagation passes per level, after the first search pass (0 disables propagation).
    /// In each pass every patch compares the matching cost of its own displacement with the
    /// displacements of its four grid neighbours. If a neighbour's displacement is clearly
    /// better (see [`propagation_margin`](Self::propagation_margin)), the patch takes the best
    /// one and refines it with [`propagation_iterations`](Self::propagation_iterations)
    /// Gauss-Newton iterations; otherwise it keeps its own result. Information therefore
    /// spreads by one patch per pass, which repairs patches whose search went astray (large
    /// motion, little texture).
    pub propagation_passes: usize,
    /// Gauss-Newton iterations after a patch takes a neighbour's displacement.
    pub propagation_iterations: usize,
    /// A neighbour's displacement is taken only when its matching cost is below this fraction
    /// of the patch's own cost. Values below 1 keep noise-level differences from replacing a
    /// patch's converged estimate.
    pub propagation_margin: f32,
    /// Finest level of the variational refinement, at most `finest_level`. Levels between it
    /// and `finest_level` are not searched: the coarser flow is upsampled and refined.
    pub vr_finest_level: usize,
    /// Number of levels refined, counting up from `vr_finest_level` (0 disables the
    /// refinement). Coarser levels are only searched and densified; the refinement matters
    /// little there and its long dependency chains cost more than their size suggests. It
    /// must cover the levels below `finest_level`.
    pub vr_levels: usize,
    /// Fixed-point iterations of the variational refinement per refined level (0 disables it).
    /// Each one re-linearises the robust penalties and runs
    /// [`vr_inner_iterations`](Self::vr_inner_iterations) relaxation sweeps.
    pub vr_iterations: usize,
    /// Successive over-relaxation sweeps per fixed-point iteration.
    pub vr_inner_iterations: usize,
    /// Weight of the smoothness term.
    pub vr_alpha: f32,
    /// Weight of the brightness constancy term.
    pub vr_delta: f32,
    /// Weight of the gradient constancy term.
    pub vr_gamma: f32,
    /// Over-relaxation factor of the sweeps, in (0, 2).
    pub vr_omega: f32,
    /// Regulariser of the data term normalisation, in intensity units per pixel: each data
    /// term is divided by `|gradient|^2 + zeta^2`, which keeps flat, noisy regions from being
    /// amplified.
    pub vr_zeta: f32,
    /// Smoothing constant of the robust penalty `sqrt(s^2 + epsilon^2)`, greater than 0.
    pub vr_epsilon: f32,
}

impl DisParams {
    /// The parameters of a preset.
    pub fn preset(preset: Preset) -> Self {
        let medium = Self {
            finest_level: 1,
            coarsest_level: None,
            patch_stride: 3,
            gd_iterations: 25,
            gd_tolerance: 0.01,
            mean_normalization: true,
            propagation_passes: 4,
            propagation_iterations: 8,
            propagation_margin: 0.75,
            vr_finest_level: 0,
            vr_levels: 3,
            vr_iterations: 10,
            vr_inner_iterations: 5,
            vr_alpha: 20.0,
            vr_delta: 5.0,
            vr_gamma: 10.0,
            vr_omega: 1.6,
            vr_zeta: 1.0,
            vr_epsilon: 0.001,
        };
        match preset {
            Preset::Medium => medium,
            Preset::High => Self { vr_iterations: 20, ..medium },
            Preset::Fast => Self {
                finest_level: 2,
                patch_stride: 4,
                gd_iterations: 16,
                propagation_passes: 2,
                propagation_iterations: 6,
                vr_finest_level: 1,
                vr_levels: 2,
                vr_iterations: 5,
                ..medium
            },
            Preset::UltraFast => Self {
                finest_level: 2,
                patch_stride: 4,
                gd_iterations: 12,
                propagation_passes: 1,
                propagation_iterations: 4,
                vr_finest_level: 2,
                vr_levels: 1,
                vr_iterations: 2,
                ..medium
            },
        }
    }

    /// Checks the parameters for consistency.
    pub fn validate(&self) -> Result<(), FlowError> {
        if !(1..=PATCH_SIZE).contains(&self.patch_stride) {
            return invalid(format!("patch stride must be 1..={PATCH_SIZE}, got {}", self.patch_stride));
        }
        let iterations =
            [self.gd_iterations, self.propagation_iterations, self.vr_iterations, self.vr_inner_iterations];
        if iterations.iter().any(|&n| n > 1000) {
            return invalid("iteration counts must be at most 1000");
        }
        if !(self.gd_tolerance.is_finite() && self.gd_tolerance >= 0.0) {
            return invalid("the search tolerance must be finite and non-negative");
        }
        if self.propagation_passes > 16 {
            return invalid("at most 16 propagation passes are supported");
        }
        if !(self.propagation_margin > 0.0 && self.propagation_margin <= 1.0) {
            return invalid("the propagation margin must be in (0, 1]");
        }
        if self.finest_level > 15 {
            return invalid("finest level must be at most 15");
        }
        if let Some(c) = self.coarsest_level
            && c < self.finest_level
        {
            return invalid("the coarsest level must not be finer than the finest level");
        }
        if self.vr_finest_level > self.finest_level {
            return invalid("the finest refined level must not be coarser than the finest searched level");
        }
        if self.refines() && self.vr_finest_level + self.vr_levels < self.finest_level {
            return invalid("the refined levels must include every level below the finest searched level");
        }
        let weights = [self.vr_alpha, self.vr_delta, self.vr_gamma, self.vr_zeta];
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return invalid("variational refinement weights must be finite and non-negative");
        }
        if !(self.vr_epsilon.is_finite() && self.vr_epsilon > 0.0) {
            return invalid("the robust penalty constant must be finite and positive");
        }
        if !(self.vr_omega > 0.0 && self.vr_omega < 2.0) {
            return invalid("the over-relaxation factor must be in (0, 2)");
        }
        Ok(())
    }

    fn refines(&self) -> bool {
        self.vr_iterations > 0 && self.vr_levels > 0
    }

    /// The pyramid levels used for a `width` x `height` input.
    ///
    /// Levels finer than the finest searched one and coarser than the coarsest one are dropped
    /// while they cannot hold a whole patch. Without an explicit
    /// [`coarsest_level`](Self::coarsest_level), the search starts where the larger side is
    /// closest (on a log scale) to four patches. The aim is that motion of up to about a quarter
    /// of the larger side shrinks to about one patch width on the coarsest level, where the
    /// inverse search starts from zero flow and can still reach it; a coarser start would add
    /// little range, and its levels would hold too few patches to be reliable.
    pub fn plan(&self, width: usize, height: usize) -> Result<Plan, FlowError> {
        self.validate()?;
        if width < PATCH_SIZE || height < PATCH_SIZE {
            return invalid(format!("images must be at least {PATCH_SIZE}x{PATCH_SIZE} pixels, got {width}x{height}"));
        }
        // Finest level that still fits a patch.
        let fits = |l: usize| (width >> l) >= PATCH_SIZE && (height >> l) >= PATCH_SIZE;
        let mut finest = self.finest_level;
        while !fits(finest) {
            finest -= 1;
        }
        let auto = (width.max(height) as f64 / (4 * PATCH_SIZE) as f64).log2().round().max(0.0) as usize;
        let mut coarsest = self.coarsest_level.unwrap_or(auto).max(finest);
        while !fits(coarsest) {
            coarsest -= 1;
        }
        let refined = self.refines().then(|| {
            let lo = self.vr_finest_level.min(finest);
            // Levels skipped by clamping `finest` stay refined, so the range only grows.
            let hi = (self.vr_finest_level + self.vr_levels - 1).max(finest.saturating_sub(1)).min(coarsest);
            (lo, hi)
        });
        Ok(Plan { width, height, finest, coarsest, refined })
    }
}

impl Default for DisParams {
    fn default() -> Self {
        Self::preset(Preset::default())
    }
}

/// The pyramid levels of one flow computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Input width.
    pub width: usize,
    /// Input height.
    pub height: usize,
    /// Finest level searched.
    pub finest: usize,
    /// Coarsest level searched (inclusive).
    pub coarsest: usize,
    /// Finest and coarsest refined level (inclusive), if the refinement is on. The finest
    /// refined level may lie below `finest`.
    pub refined: Option<(usize, usize)>,
}

impl Plan {
    /// Size of pyramid level `l`.
    pub fn level_size(&self, l: usize) -> (usize, usize) {
        (self.width >> l, self.height >> l)
    }

    /// Number of patches along x and y on level `l` for the given stride. The last patch in each
    /// direction is moved inwards so that it ends at the image border.
    pub fn patch_grid(&self, l: usize, stride: usize) -> (usize, usize) {
        let (w, h) = self.level_size(l);
        ((w - PATCH_SIZE).div_ceil(stride) + 1, (h - PATCH_SIZE).div_ceil(stride) + 1)
    }

    /// Finest level computed; its flow is upsampled to full resolution.
    pub fn output_level(&self) -> usize {
        self.refined.map_or(self.finest, |(lo, _)| lo.min(self.finest))
    }

    /// True if level `l` is searched.
    pub fn is_searched(&self, l: usize) -> bool {
        (self.finest..=self.coarsest).contains(&l)
    }

    /// True if level `l` is refined.
    pub fn is_refined(&self, l: usize) -> bool {
        self.refined.is_some_and(|(lo, hi)| (lo..=hi).contains(&l))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_validate_and_parse() {
        for p in Preset::ALL {
            DisParams::preset(p).validate().unwrap();
            assert_eq!(p.name().parse::<Preset>().unwrap(), p);
        }
        assert!("slow".parse::<Preset>().is_err());
        assert!(DisParams { patch_stride: 0, ..DisParams::default() }.validate().is_err());
        assert!(DisParams { vr_omega: 2.0, ..DisParams::default() }.validate().is_err());
        assert!(DisParams { coarsest_level: Some(0), ..DisParams::default() }.validate().is_err());
        assert!(DisParams { vr_finest_level: 2, ..DisParams::default() }.validate().is_err());
        assert!(DisParams { vr_levels: 1, finest_level: 2, ..DisParams::default() }.validate().is_err());
        assert!(DisParams { vr_epsilon: 0.0, ..DisParams::default() }.validate().is_err());
        assert!(DisParams { propagation_margin: 1.5, ..DisParams::default() }.validate().is_err());
        assert!(DisParams { gd_tolerance: -1.0, ..DisParams::default() }.validate().is_err());
        assert!(DisParams { gd_tolerance: f32::NAN, ..DisParams::default() }.validate().is_err());
    }

    #[test]
    fn pyramid_plan() {
        let medium = DisParams::default();
        let p = medium.plan(960, 540).unwrap();
        assert_eq!((p.finest, p.coarsest, p.refined, p.output_level()), (1, 5, Some((0, 2)), 0));
        assert!(p.is_searched(1) && !p.is_searched(0) && p.is_refined(0) && !p.is_refined(3));
        assert_eq!(p.level_size(5), (30, 16));
        assert_eq!(p.patch_grid(1, 3), (159, 89));
        let p = medium.plan(480, 270).unwrap();
        assert_eq!((p.finest, p.coarsest), (1, 4));
        // Small images clamp both ends to levels that still fit a patch.
        let p = DisParams::preset(Preset::Fast).plan(20, 12).unwrap();
        assert_eq!((p.finest, p.coarsest, p.refined, p.output_level()), (0, 0, Some((0, 0)), 0));
        let p = DisParams::preset(Preset::UltraFast).plan(960, 540).unwrap();
        assert_eq!((p.finest, p.refined, p.output_level()), (2, Some((2, 2)), 2));
        let p = DisParams { vr_iterations: 0, ..DisParams::default() }.plan(960, 540).unwrap();
        assert_eq!((p.finest, p.refined, p.output_level()), (1, None, 1));
        assert!(medium.plan(7, 100).is_err());
    }
}
