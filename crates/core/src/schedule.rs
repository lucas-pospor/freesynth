//! Quality presets and the per-level schedule they resolve to.
//!
//! A [`Params`] holds a [`Preset`] plus optional explicit settings. [`Params::schedule`] combines
//! them, for a given input and engine, into a [`Schedule`]: the concrete number of pyramid levels
//! and the per-level iteration counts, stop-mask use and voting rules that the engines run. Any
//! setting given explicitly in [`Params`] wins over the preset.

use std::fmt;
use std::str::FromStr;

use crate::error::{Error, invalid};
use crate::pyramid::max_pyramid_levels;
use crate::synth::{Params, PerLevel, SynthesisInput, VoteMode};
use crate::vote::VoteRounding;

/// A named trade-off between speed and quality. See [`Params::preset`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Preset {
    /// The 2019 ebsynth schedule and voting: 4 PatchMatch iterations per search/vote iteration,
    /// plain voting with truncation, no stop mask.
    Compat,
    /// Roughly twice as fast as [`Compat`](Preset::Compat) at about the same quality: 2
    /// PatchMatch iterations per search/vote iteration (4 with 3x3 patches, which need them),
    /// weighted voting, rounding.
    #[default]
    Default,
    /// [`Default`](Preset::Default) plus the stop-threshold mask, and on the GPU 4 and 3
    /// search/vote iterations at the two finest levels. Faster, with a few percent higher patch
    /// error.
    Fast,
    /// The full [`Compat`](Preset::Compat) schedule (4 PatchMatch iterations) with weighted
    /// voting and rounding.
    High,
}

impl Preset {
    /// All presets, fastest last.
    pub const ALL: [Preset; 4] = [Preset::Compat, Preset::High, Preset::Default, Preset::Fast];

    /// The preset's lower-case name, as accepted by [`FromStr`].
    pub fn name(self) -> &'static str {
        match self {
            Preset::Compat => "compat",
            Preset::Default => "default",
            Preset::Fast => "fast",
            Preset::High => "high",
        }
    }
}

impl fmt::Display for Preset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Preset {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Error> {
        Preset::ALL
            .into_iter()
            .find(|p| p.name() == s)
            .ok_or_else(|| Error::InvalidInput(format!("unknown preset '{s}' (expected compat, default, fast or high)")))
    }
}

/// The engine a [`Schedule`] is resolved for. Some presets differ between engines because the
/// engines' costs differ (see [`Preset::Fast`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Engine {
    /// The CPU engine of this crate.
    Cpu,
    /// A GPU engine.
    Gpu,
}

/// The concrete settings of one synthesis: [`Params`] with the preset applied, for a known
/// number of pyramid levels. Per-level vectors are indexed coarsest level first.
///
/// A schedule is resolved for one [`Engine`]. Only [`Preset::Fast`] differs between engines
/// (fewer search/vote iterations at the finest levels on the GPU), so the same `Params` run with
/// `Fast` give different results if a pipeline falls back from the GPU to the CPU. Set
/// [`Params::search_vote_iters`] explicitly to get the same schedule on both.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    /// Number of pyramid levels that will be run.
    pub num_levels: usize,
    /// Search/vote iterations per level.
    pub search_vote_iters: Vec<usize>,
    /// PatchMatch iterations per search/vote iteration, per level.
    pub patch_match_iters: Vec<usize>,
    /// Stop threshold per level (only used with [`use_stop_threshold`](Self::use_stop_threshold)).
    pub stop_threshold: Vec<u32>,
    /// Whether converged pixels are skipped by PatchMatch.
    pub use_stop_threshold: bool,
    /// How patches are blended after each search.
    pub vote_mode: VoteMode,
    /// How votes are converted to 8 bits.
    pub vote_rounding: VoteRounding,
}

impl Params {
    /// Parameters of the given preset, with no explicit overrides.
    pub fn with_preset(preset: Preset) -> Self {
        Self { preset, ..Self::default() }
    }

    /// Parameters that reproduce the 2019 ebsynth command-line tool (the [`Preset::Compat`]
    /// preset). Add [`bit_exact`](Self::bit_exact) and a single thread for bit-identical output.
    pub fn compat() -> Self {
        Self::with_preset(Preset::Compat)
    }

    /// Resolves the preset and the explicit settings into the concrete schedule that `engine`
    /// runs for `input`, checking the input and the parameters on the way.
    ///
    /// Explicit settings always win over the preset. [`bit_exact`](Self::bit_exact) does not
    /// enter the schedule: the 2019 regression mode is [`Preset::Compat`] plus `bit_exact`.
    pub fn schedule(&self, input: &SynthesisInput, engine: Engine) -> Result<Schedule, Error> {
        input.validate()?;
        if self.patch_size < 3 || self.patch_size.is_multiple_of(2) {
            return invalid("patch size must be an odd number >= 3");
        }
        if !self.uniformity_weight.is_finite() {
            return invalid("uniformity weight must be finite");
        }
        let max_levels = max_pyramid_levels(input.source_style.size(), input.target_guide.size(), self.patch_size);
        if max_levels == 0 {
            return invalid(format!(
                "source and target must be at least {0}x{0} pixels for patch size {1}",
                2 * self.patch_size + 1,
                self.patch_size
            ));
        }
        let num_levels = match self.num_pyramid_levels {
            Some(0) => return invalid("number of pyramid levels must be at least 1"),
            Some(n) => n.min(max_levels),
            None => max_levels,
        };
        self.schedule_for_levels(num_levels, engine)
    }

    /// Like [`schedule`](Self::schedule) for a known number of pyramid levels, without looking
    /// at an input.
    pub fn schedule_for_levels(&self, num_levels: usize, engine: Engine) -> Result<Schedule, Error> {
        let expand = |name: &str, v: &PerLevel<usize>| -> Result<Vec<usize>, Error> {
            if !v.covers(num_levels) {
                return invalid(format!("{name} must cover all {num_levels} pyramid levels"));
            }
            Ok((0..num_levels).map(|l| v.get(l)).collect())
        };
        let preset = self.preset;
        let tuned = preset != Preset::Compat;

        let search_vote_iters = match &self.search_vote_iters {
            Some(v) => expand("search/vote iterations", v)?,
            None => {
                let mut v = vec![6; num_levels];
                if preset == Preset::Fast && engine == Engine::Gpu {
                    // The two finest levels converge in fewer iterations; they also cost the most.
                    for (level, iters) in v.iter_mut().rev().zip([3, 4]) {
                        *level = iters;
                    }
                }
                v
            }
        };
        let patch_match_iters = match &self.patch_match_iters {
            Some(v) => expand("PatchMatch iterations", v)?,
            None => {
                // 3x3 patches gather too little context per propagation step to converge in 2.
                let pm = match preset {
                    Preset::Compat | Preset::High => 4,
                    Preset::Default | Preset::Fast if self.patch_size <= 3 => 4,
                    Preset::Default | Preset::Fast => 2,
                };
                vec![pm; num_levels]
            }
        };
        if !self.stop_threshold.covers(num_levels) {
            return invalid(format!("stop thresholds must cover all {num_levels} pyramid levels"));
        }
        let stop_threshold = (0..num_levels).map(|l| self.stop_threshold.get(l)).collect();
        Ok(Schedule {
            num_levels,
            search_vote_iters,
            patch_match_iters,
            stop_threshold,
            use_stop_threshold: self.use_stop_threshold.unwrap_or(preset == Preset::Fast),
            vote_mode: self.vote_mode.unwrap_or(if tuned { VoteMode::Weighted } else { VoteMode::Plain }),
            vote_rounding: self.vote_rounding.unwrap_or(if tuned { VoteRounding::Nearest } else { VoteRounding::Truncate }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    fn input(w: usize, h: usize) -> SynthesisInput {
        SynthesisInput {
            source_style: Image::new(w, h, 3),
            source_guide: Image::new(w, h, 1),
            target_guide: Image::new(w, h, 1),
            target_modulation: None,
            style_weights: vec![1.0; 3],
            guide_weights: vec![1.0],
        }
    }

    #[test]
    fn presets_resolve_as_documented() {
        let s = |p: &Params, e| p.schedule_for_levels(5, e).unwrap();
        let compat = s(&Params::compat(), Engine::Cpu);
        assert_eq!(compat.search_vote_iters, [6; 5]);
        assert_eq!(compat.patch_match_iters, [4; 5]);
        assert!(!compat.use_stop_threshold);
        assert_eq!((compat.vote_mode, compat.vote_rounding), (VoteMode::Plain, VoteRounding::Truncate));
        assert_eq!(s(&Params::compat(), Engine::Gpu), compat);

        let high = s(&Params::with_preset(Preset::High), Engine::Gpu);
        assert_eq!((high.search_vote_iters.clone(), high.patch_match_iters.clone()), (vec![6; 5], vec![4; 5]));
        assert_eq!((high.vote_mode, high.vote_rounding), (VoteMode::Weighted, VoteRounding::Nearest));

        let default = s(&Params::default(), Engine::Cpu);
        assert_eq!(Params::default().preset, Preset::Default);
        assert_eq!(default.patch_match_iters, [2; 5]);
        assert_eq!(default.search_vote_iters, [6; 5]);
        assert!(!default.use_stop_threshold);
        assert_eq!(default.vote_mode, VoteMode::Weighted);
        assert_eq!(s(&Params::default(), Engine::Gpu), default);

        let fast = Params::with_preset(Preset::Fast);
        let (cpu, gpu) = (s(&fast, Engine::Cpu), s(&fast, Engine::Gpu));
        assert!(cpu.use_stop_threshold && gpu.use_stop_threshold);
        assert_eq!(cpu.search_vote_iters, [6; 5]);
        assert_eq!(gpu.search_vote_iters, [6, 6, 6, 4, 3]);
        assert_eq!(gpu.patch_match_iters, [2; 5]);
        assert_eq!(fast.schedule_for_levels(1, Engine::Gpu).unwrap().search_vote_iters, [3]);
    }

    #[test]
    fn small_patches_keep_four_patchmatch_iterations() {
        for preset in [Preset::Default, Preset::Fast] {
            let p = Params { patch_size: 3, ..Params::with_preset(preset) };
            assert_eq!(p.schedule_for_levels(3, Engine::Gpu).unwrap().patch_match_iters, [4; 3]);
        }
    }

    #[test]
    fn explicit_settings_override_the_preset() {
        let p = Params {
            search_vote_iters: Some(PerLevel::Uniform(2)),
            patch_match_iters: Some(PerLevel::Levels(vec![5, 4, 3])),
            use_stop_threshold: Some(false),
            vote_mode: Some(VoteMode::Plain),
            vote_rounding: Some(VoteRounding::Truncate),
            ..Params::with_preset(Preset::Fast)
        };
        let s = p.schedule_for_levels(3, Engine::Gpu).unwrap();
        assert_eq!(s.search_vote_iters, [2; 3]);
        assert_eq!(s.patch_match_iters, [5, 4, 3]);
        assert!(!s.use_stop_threshold);
        assert_eq!((s.vote_mode, s.vote_rounding), (VoteMode::Plain, VoteRounding::Truncate));

        let compat_weighted = Params { vote_mode: Some(VoteMode::Weighted), ..Params::compat() };
        assert_eq!(compat_weighted.schedule_for_levels(2, Engine::Cpu).unwrap().vote_mode, VoteMode::Weighted);
    }

    #[test]
    fn bit_exact_does_not_change_the_schedule() {
        for preset in Preset::ALL {
            let p = Params::with_preset(preset);
            let exact = Params { bit_exact: true, ..p.clone() };
            assert_eq!(exact.schedule_for_levels(2, Engine::Cpu).unwrap(), p.schedule_for_levels(2, Engine::Cpu).unwrap());
        }
    }

    #[test]
    fn schedule_checks_input_and_levels() {
        let inp = input(40, 40);
        let s = Params::default().schedule(&inp, Engine::Cpu).unwrap();
        assert_eq!(s.num_levels, max_pyramid_levels((40, 40), (40, 40), 5));
        let s = Params { num_pyramid_levels: Some(99), ..Params::default() }.schedule(&inp, Engine::Cpu).unwrap();
        assert_eq!(s.num_levels, max_pyramid_levels((40, 40), (40, 40), 5));
        assert!(Params { num_pyramid_levels: Some(0), ..Params::default() }.schedule(&inp, Engine::Cpu).is_err());
        assert!(Params { patch_size: 4, ..Params::default() }.schedule(&inp, Engine::Cpu).is_err());
        assert!(Params::default().schedule(&input(8, 8), Engine::Cpu).is_err());
        let short = Params { search_vote_iters: Some(PerLevel::Levels(vec![1])), num_pyramid_levels: Some(2), ..Params::default() };
        assert!(short.schedule(&inp, Engine::Cpu).is_err());
    }

    #[test]
    fn preset_names_round_trip() {
        for p in Preset::ALL {
            assert_eq!(p.name().parse::<Preset>().unwrap(), p);
            assert_eq!(p.to_string(), p.name());
        }
        assert!("turbo".parse::<Preset>().is_err());
    }
}
