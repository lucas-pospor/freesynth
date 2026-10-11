//! Paired comparisons with bootstrap confidence intervals.
//!
//! Two variants A and B are measured on the same units (the same pixels, frame pairs, or
//! image-and-seed runs). The quantity of interest is the mean paired difference `B - A`; its
//! uncertainty is estimated by resampling the units with replacement (Efron and Tibshirani,
//! "An Introduction to the Bootstrap", 1993) and taking percentiles of the resampled means.
//! Pairing removes the variation that both variants share (hard pixels are hard for both), so
//! the interval is much narrower than one built from two independent means.
//!
//! The relative difference `mean(B) / mean(A) - 1` comes from the same resamples, and so does the
//! mean of the per-unit ratios `B / A - 1`, which weights every unit equally when units differ in
//! scale.
//!
//! Two refinements:
//!
//! - Units that follow each other in time (consecutive frame pairs, overlapping frame windows)
//!   are serially correlated, and resampling them one by one makes the interval too narrow.
//!   [`paired_block_bootstrap`] resamples runs of consecutive units instead (the moving-block
//!   bootstrap of Kunsch, "The jackknife and the bootstrap for general stationary
//!   observations", Annals of Statistics, 1989).
//! - With few units (for example 15 image-and-seed runs) the percentile interval undercovers.
//!   [`PairedComparison::add_t_intervals`] adds the Student-t interval of the same means for
//!   comparison; a difference counts as significant only when both intervals exclude zero.

use serde::{Deserialize, Serialize};

use crate::error::{Error, invalid};
use crate::rng::SplitMix64;
use crate::stats::{mean, quantile_sorted, sorted, t_interval};

/// Resampling settings.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Number of bootstrap resamples.
    pub resamples: usize,
    /// Seed of the resampling generator.
    pub seed: u64,
    /// Two-sided confidence level, for example 0.95.
    pub confidence: f64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self { resamples: 10_000, seed: 1, confidence: 0.95 }
    }
}

/// Result of [`paired_bootstrap`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairedComparison {
    /// Number of paired units.
    pub n: usize,
    pub mean_a: f64,
    pub mean_b: f64,
    /// Mean of `b - a`.
    pub diff: f64,
    /// Confidence interval of `diff`.
    pub diff_lo: f64,
    pub diff_hi: f64,
    /// `mean_b / mean_a - 1`.
    pub rel: f64,
    /// Confidence interval of `rel`.
    pub rel_lo: f64,
    pub rel_hi: f64,
    /// Mean over units of `b / a - 1`, with its confidence interval: the relative difference with
    /// every unit weighted equally, for pooling units of different scale (for example several
    /// images). `None` unless every `a` is positive; [`compare_reports`](crate::compare::compare_reports)
    /// also leaves it out of sample-level comparisons.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pair_rel: Option<[f64; 3]>,
    /// Two-sided bootstrap p-value for "no difference": twice the smaller fraction of resampled
    /// differences on either side of zero (at least `1 / resamples`).
    pub p: f64,
    /// Block length when the units were resampled in moving blocks ([`paired_block_bootstrap`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block: Option<usize>,
    /// Student-t confidence interval of `diff`, when added by
    /// [`add_t_intervals`](Self::add_t_intervals).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_diff: Option<[f64; 2]>,
    /// Student-t confidence interval of the mean per-unit ratio (`pair_rel`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_pair_rel: Option<[f64; 2]>,
}

impl PairedComparison {
    /// Whether the confidence interval of the difference excludes zero; when a Student-t
    /// interval was added, it must exclude zero too.
    pub fn significant(&self) -> bool {
        let excludes = |lo: f64, hi: f64| lo > 0.0 || hi < 0.0;
        excludes(self.diff_lo, self.diff_hi) && self.t_diff.is_none_or(|[lo, hi]| excludes(lo, hi))
    }

    /// Adds Student-t intervals of the mean difference and of the mean per-unit ratio, from the
    /// same samples the comparison was made from. The percentile bootstrap undercovers with few
    /// independent units; the t interval is the usual small-sample check.
    pub fn add_t_intervals(&mut self, a: &[f64], b: &[f64], confidence: f64) {
        let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
        self.t_diff = t_interval(&diffs, confidence);
        if self.pair_rel.is_some() {
            let ratios: Vec<f64> = a.iter().zip(b).map(|(x, y)| y / x - 1.0).collect();
            self.t_pair_rel = t_interval(&ratios, confidence);
        }
    }
}

/// Paired percentile bootstrap of `b - a`. `a[i]` and `b[i]` must be measurements of the same
/// unit, and the units must be independent.
pub fn paired_bootstrap(a: &[f64], b: &[f64], cfg: &BootstrapConfig) -> Result<PairedComparison, Error> {
    resample(a, b, cfg, 1)
}

/// [`paired_bootstrap`] for units in a serially correlated order (for example consecutive frame
/// pairs): each resample joins randomly placed runs of `block` consecutive units (the last run
/// cut short) until it has as many units as the data. `block` is capped at the number of units.
pub fn paired_block_bootstrap(a: &[f64], b: &[f64], cfg: &BootstrapConfig, block: usize) -> Result<PairedComparison, Error> {
    if block == 0 {
        return invalid("block length must be at least 1");
    }
    let mut c = resample(a, b, cfg, block.min(a.len().max(1)))?;
    c.block = Some(block.min(a.len()));
    Ok(c)
}

fn resample(a: &[f64], b: &[f64], cfg: &BootstrapConfig, block: usize) -> Result<PairedComparison, Error> {
    if a.len() != b.len() {
        return invalid(format!("paired samples differ in length ({} vs {})", a.len(), b.len()));
    }
    if a.is_empty() {
        return invalid("no paired samples");
    }
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return invalid("paired samples must be finite");
    }
    if cfg.resamples == 0 || !(0.0..1.0).contains(&cfg.confidence) {
        return invalid("need at least one resample and a confidence level in [0, 1)");
    }
    let n = a.len();
    let mut rng = SplitMix64::new(cfg.seed);
    let mut diffs = Vec::with_capacity(cfg.resamples);
    let mut rels = Vec::with_capacity(cfg.resamples);
    let ratios: Option<Vec<f64>> = a.iter().all(|&x| x > 0.0).then(|| a.iter().zip(b).map(|(x, y)| y / x - 1.0).collect());
    let mut ratio_means = Vec::with_capacity(cfg.resamples);
    for _ in 0..cfg.resamples {
        let (mut sa, mut sb, mut sr) = (0.0, 0.0, 0.0);
        let mut taken = 0;
        while taken < n {
            let start = rng.below(n - block + 1);
            for i in start..start + block.min(n - taken) {
                sa += a[i];
                sb += b[i];
                if let Some(r) = &ratios {
                    sr += r[i];
                }
            }
            taken += block;
        }
        diffs.push((sb - sa) / n as f64);
        rels.push(sb / sa - 1.0);
        ratio_means.push(sr / n as f64);
    }
    let alpha = (1.0 - cfg.confidence) / 2.0;
    let d = sorted(&diffs);
    let r = sorted(&rels);
    let below = diffs.iter().filter(|&&x| x <= 0.0).count() as f64;
    let above = diffs.iter().filter(|&&x| x >= 0.0).count() as f64;
    let p = (2.0 * below.min(above) / cfg.resamples as f64).clamp(1.0 / cfg.resamples as f64, 1.0);
    let (mean_a, mean_b) = (mean(a), mean(b));
    let pair_rel = ratios.map(|r| {
        let q = sorted(&ratio_means);
        [mean(&r), quantile_sorted(&q, alpha), quantile_sorted(&q, 1.0 - alpha)]
    });
    Ok(PairedComparison {
        n,
        mean_a,
        mean_b,
        diff: mean_b - mean_a,
        diff_lo: quantile_sorted(&d, alpha),
        diff_hi: quantile_sorted(&d, 1.0 - alpha),
        rel: mean_b / mean_a - 1.0,
        rel_lo: quantile_sorted(&r, alpha),
        rel_hi: quantile_sorted(&r, 1.0 - alpha),
        pair_rel,
        p,
        block: None,
        t_diff: None,
        t_pair_rel: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal(n: usize, mean: f64, sd: f64, rng: &mut SplitMix64) -> Vec<f64> {
        (0..n).map(|_| mean + sd * rng.normal()).collect()
    }

    #[test]
    fn detects_a_known_shift() {
        let mut rng = SplitMix64::new(11);
        // Large shared variation (sd 10), small true difference (0.5) with small paired noise.
        let a = normal(2000, 100.0, 10.0, &mut rng);
        let b: Vec<f64> = a.iter().map(|&x| x + 0.5 + 0.2 * rng.normal()).collect();
        let c = paired_bootstrap(&a, &b, &BootstrapConfig { resamples: 2000, ..Default::default() }).unwrap();
        assert!(c.diff_lo < 0.5 && 0.5 < c.diff_hi, "{c:?}");
        assert!(c.significant());
        assert!(c.p <= 1.0 / 2000.0 + 1e-12);
        assert!((c.rel - 0.005).abs() < 0.001);
        assert!(c.rel_lo < c.rel && c.rel < c.rel_hi);
        let [pr, lo, hi] = c.pair_rel.unwrap();
        assert!((pr - 0.005).abs() < 0.001 && lo < pr && pr < hi);
        // Unpaired, a difference of 0.5 would be hidden by the sd of 10 (SE ~ 0.3 per mean).
        assert!(c.diff_hi - c.diff_lo < 0.05);
    }

    #[test]
    fn no_difference_is_not_significant() {
        let mut rng = SplitMix64::new(5);
        let a = normal(500, 10.0, 2.0, &mut rng);
        let b: Vec<f64> = a.iter().map(|&x| x + rng.normal()).collect();
        let c = paired_bootstrap(&a, &b, &BootstrapConfig::default()).unwrap();
        assert!(!c.significant(), "{c:?}");
        assert!(c.p > 0.05);
    }

    #[test]
    fn interval_covers_the_true_mean_at_the_nominal_rate() {
        // 300 experiments with n = 40 paired normal differences of mean 1: about 95% of the
        // intervals should contain 1 (the percentile bootstrap undercovers slightly at small n).
        let mut rng = SplitMix64::new(99);
        let cfg = BootstrapConfig { resamples: 1000, seed: 3, confidence: 0.95 };
        let trials = 300;
        let mut covered = 0;
        for _ in 0..trials {
            let a = normal(40, 0.0, 1.0, &mut rng);
            let b: Vec<f64> = a.iter().map(|&x| x + 1.0 + rng.normal()).collect();
            let c = paired_bootstrap(&a, &b, &cfg).unwrap();
            covered += usize::from(c.diff_lo <= 1.0 && 1.0 <= c.diff_hi);
        }
        let rate = covered as f64 / trials as f64;
        assert!((0.88..=0.99).contains(&rate), "coverage {rate}");
    }

    #[test]
    fn block_bootstrap_widens_intervals_of_correlated_units() {
        // An AR(1) series of paired differences with strong serial correlation (phi = 0.8):
        // resampling single units underestimates the spread of the mean.
        let mut rng = SplitMix64::new(21);
        let n = 400;
        let mut d = vec![0.0f64; n];
        for i in 1..n {
            d[i] = 0.8 * d[i - 1] + rng.normal();
        }
        let a = vec![10.0; n];
        let b: Vec<f64> = d.iter().map(|x| 10.0 + x).collect();
        let cfg = BootstrapConfig { resamples: 2000, ..Default::default() };
        let iid = paired_bootstrap(&a, &b, &cfg).unwrap();
        let blocked = paired_block_bootstrap(&a, &b, &cfg, 20).unwrap();
        assert_eq!(blocked.block, Some(20));
        assert_eq!((iid.diff, iid.block), (blocked.diff, None));
        let width = |c: &PairedComparison| c.diff_hi - c.diff_lo;
        // The long-run sd is 1 / (1 - phi) = 5 times the innovation sd, against about 1.67 for
        // the marginal sd, so the honest interval is about 3 times wider.
        assert!(width(&blocked) > 2.0 * width(&iid), "{} vs {}", width(&blocked), width(&iid));
        // Block length 1 is the plain bootstrap; overlong blocks are capped.
        assert_eq!(paired_block_bootstrap(&a, &b, &cfg, 1).unwrap().diff_lo, iid.diff_lo);
        assert_eq!(paired_block_bootstrap(&a[..5], &b[..5], &cfg, 50).unwrap().block, Some(5));
        assert!(paired_block_bootstrap(&a, &b, &cfg, 0).is_err());
    }

    #[test]
    fn t_intervals_and_significance() {
        let a = [100.0, 101.0, 99.0, 100.5, 100.2];
        let b = [100.4, 101.1, 99.6, 100.6, 100.5];
        let mut c = paired_bootstrap(&a, &b, &BootstrapConfig::default()).unwrap();
        assert!(c.significant());
        c.add_t_intervals(&a, &b, 0.95);
        let [lo, hi] = c.t_diff.unwrap();
        // The t interval is wider than the percentile interval at n = 5.
        assert!(lo < c.diff_lo && hi > c.diff_hi, "{c:?}");
        assert!(c.t_pair_rel.is_some());
        // A t interval that includes zero makes the result not significant.
        c.t_diff = Some([-0.1, 0.5]);
        assert!(!c.significant());
    }

    #[test]
    fn reproducible_and_validated() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.5, 2.0, 3.5, 4.5];
        let cfg = BootstrapConfig { resamples: 500, seed: 7, confidence: 0.9 };
        assert_eq!(paired_bootstrap(&a, &b, &cfg).unwrap(), paired_bootstrap(&a, &b, &cfg).unwrap());
        assert!(paired_bootstrap(&a, &b[..3], &cfg).is_err());
        assert!(paired_bootstrap(&[], &[], &cfg).is_err());
        assert!(paired_bootstrap(&[f64::NAN], &[1.0], &cfg).is_err());
        assert!(paired_bootstrap(&a, &b, &BootstrapConfig { resamples: 0, ..cfg }).is_err());
        assert!(paired_bootstrap(&[0.0, 1.0], &[1.0, 1.0], &cfg).unwrap().pair_rel.is_none());
    }
}
