//! Paired comparison of two sets of reports.
//!
//! Two levels of pairing are supported:
//!
//! - **Samples**: one report per side, measured on the same units (the sample sets have equal
//!   keys, for example two images measured on the same pixel set). The bootstrap resamples the
//!   units. This shows whether two given images differ; it does not include run-to-run noise.
//!   Per-frame and per-frame-pair samples of a sequence (keys starting with `frames/` or
//!   `pairs/`) are serially correlated, and gap-`g` windows overlap; they are resampled in moving
//!   blocks of `max(g, 5)` consecutive units (see [`block_length`]).
//! - **Runs**: several reports per side, paired by position (for example the same images and
//!   seeds for both variants). The bootstrap resamples the runs, using each report's scalar
//!   metric. This is the level to use for deciding about an algorithm change, because seed noise
//!   is part of the spread. Below [`T_INTERVAL_BELOW`] pairs, a Student-t interval is reported as
//!   well, and a difference counts as significant only if both intervals exclude zero.
//!
//! Both levels check that each pair measured the same problem: reports whose inputs that define
//! the measurement (style, guides, patch size, pixel set, reference, flow, frame range, ...)
//! differ get a warning, because a mis-ordered list or another pixel set silently pairs the
//! wrong things or adds noise.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::bootstrap::{BootstrapConfig, PairedComparison, paired_block_bootstrap, paired_bootstrap};
use crate::error::{Error, invalid};
use crate::report::{Report, TOOL, format_value};

/// Fewer run pairs than this get a warning: with few units, the percentile bootstrap has few
/// distinct resamples and its intervals undercover.
pub const MIN_RUNS: usize = 10;

/// Run-level comparisons with fewer pairs than this also get a Student-t interval.
pub const T_INTERVAL_BELOW: usize = 30;

/// Shortest block for the moving-block bootstrap of sequence samples.
pub const MIN_BLOCK: usize = 5;

/// Report inputs that define what was measured. Paired reports should agree on every one of
/// these that both record.
pub const PAIRED_INPUTS: &[&str] = &[
    "style",
    "style_weight",
    "guides",
    "patch_size",
    "pixel_set",
    "samples",
    "source_size",
    "reference",
    "reference_frames",
    "sigma",
    "flow_fw",
    "flow_bw",
    "direct_flow_gaps",
    "first",
    "last",
    "gaps",
];

/// The moving-block length for the samples with key `key`: `max(g, 5)` for gap-`g` frame pairs
/// (`pairs/gapG/...`), 5 for per-frame values (`frames/...`), and `None` (independent units,
/// such as pixels) otherwise.
pub fn block_length(key: &str) -> Option<usize> {
    if let Some(rest) = key.strip_prefix("pairs/gap") {
        let gap: usize = rest.split('/').next()?.parse().ok()?;
        Some(gap.max(MIN_BLOCK))
    } else {
        key.starts_with("frames/").then_some(MIN_BLOCK)
    }
}

/// Warnings for paired reports whose measurement-defining inputs differ.
fn input_mismatches(a: &[(String, Report)], b: &[(String, Report)]) -> Vec<String> {
    let mut out = Vec::new();
    for &key in PAIRED_INPUTS {
        let bad: Vec<String> = a
            .iter()
            .zip(b)
            .filter(
                |((_, ra), (_, rb))| matches!((ra.inputs.get(key), rb.inputs.get(key)), (Some(x), Some(y)) if x != y),
            )
            .map(|((na, _), (nb, _))| format!("{na} vs {nb}"))
            .collect();
        if !bad.is_empty() {
            let shown = bad.iter().take(3).cloned().collect::<Vec<_>>().join("; ");
            let more = if bad.len() > 3 { format!(" and {} more", bad.len() - 3) } else { String::new() };
            out.push(format!("input '{key}' differs within {} of {} pairs ({shown}{more}): check that the reports measure the same problem and are in the same order", bad.len(), a.len()));
        }
    }
    out
}

/// Pairing level of a comparison row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Samples,
    Runs,
}

/// The comparison of one metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub metric: String,
    pub level: Level,
    #[serde(flatten)]
    pub result: PairedComparison,
    pub significant: bool,
}

/// A full comparison, as written by `freesynth-eval compare`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub tool: String,
    pub version: String,
    pub kind: String,
    /// Names (usually file names) of the A and B reports, in pairing order.
    pub a: Vec<String>,
    pub b: Vec<String>,
    pub bootstrap: BootstrapConfig,
    pub rows: Vec<Row>,
    /// Caveats about the comparison as a whole.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    /// Metrics that were present but could not be compared, with the reason.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<(String, String)>,
}

/// Compares reports `a[i]` with `b[i]`.
///
/// `metrics` selects metrics (all metrics common to every report when empty). `level` forces a
/// pairing level; by default one pair is compared at the sample level and several at the run
/// level.
pub fn compare_reports(
    a: &[(String, Report)],
    b: &[(String, Report)],
    metrics: &[String],
    level: Option<Level>,
    cfg: &BootstrapConfig,
) -> Result<Comparison, Error> {
    if a.len() != b.len() || a.is_empty() {
        return invalid(format!("need the same, non-zero number of A and B reports (got {} and {})", a.len(), b.len()));
    }
    let kind = &a[0].1.kind;
    if let Some((name, r)) = a.iter().chain(b).find(|(_, r)| &r.kind != kind) {
        return invalid(format!("'{name}' is a {} report, expected {kind}", r.kind));
    }
    let level = level.unwrap_or(if a.len() == 1 { Level::Samples } else { Level::Runs });
    // By default: every metric all reports have (with per-unit samples at the sample level).
    let names: Vec<String> = if metrics.is_empty() {
        let everywhere = |m: &String| match level {
            Level::Runs => a.iter().chain(b).all(|(_, r)| r.metrics.contains_key(m)),
            Level::Samples => a.iter().chain(b).all(|(_, r)| r.samples.contains_key(m)),
        };
        let mut names: Vec<String> =
            a[0].1.metrics.keys().chain(a[0].1.samples.keys()).filter(|m| everywhere(m)).cloned().collect();
        names.sort();
        names.dedup();
        names
    } else {
        metrics.to_vec()
    };
    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    let mut blocked = false;
    for m in names {
        let mut block = None;
        let values = match level {
            Level::Runs => {
                if a.len() < 2 {
                    skipped.push((m, "run-level comparison needs at least two pairs".into()));
                    continue;
                }
                let get = |set: &[(String, Report)]| -> Result<Vec<f64>, Error> {
                    set.iter()
                        .map(|(name, r)| {
                            r.metrics
                                .get(&m)
                                .copied()
                                .ok_or_else(|| Error::Invalid(format!("'{name}' has no metric '{m}'")))
                        })
                        .collect()
                };
                (get(a)?, get(b)?)
            }
            Level::Samples => {
                if a.len() != 1 {
                    skipped.push((m, "sample-level comparison takes exactly one pair".into()));
                    continue;
                }
                match (a[0].1.samples.get(&m), b[0].1.samples.get(&m)) {
                    (Some(sa), Some(sb)) if sa.key == sb.key && sa.values.len() == sb.values.len() => {
                        block = block_length(&sa.key);
                        (sa.values.clone(), sb.values.clone())
                    }
                    (Some(sa), Some(sb)) => {
                        skipped.push((m, format!("measured on different units ('{}' vs '{}')", sa.key, sb.key)));
                        continue;
                    }
                    _ => {
                        skipped.push((m, "no per-unit samples".into()));
                        continue;
                    }
                }
            }
        };
        let mut result = match block {
            Some(len) => {
                blocked = true;
                paired_block_bootstrap(&values.0, &values.1, cfg, len)?
            }
            None => paired_bootstrap(&values.0, &values.1, cfg)?,
        };
        match level {
            // Per-pixel or per-frame values can be close to zero, and a mean of their ratios is
            // then dominated by a few tiny denominators (on a real image it can even have the
            // opposite sign to the difference of means). Keep it for runs only.
            Level::Samples => result.pair_rel = None,
            Level::Runs if values.0.len() < T_INTERVAL_BELOW => {
                result.add_t_intervals(&values.0, &values.1, cfg.confidence)
            }
            Level::Runs => {}
        }
        rows.push(Row { metric: m, level, significant: result.significant(), result });
    }
    let mut warnings = input_mismatches(a, b);
    if level == Level::Runs && a.len() < MIN_RUNS {
        warnings.push(format!(
            "only {} paired runs: percentile bootstrap intervals are too narrow below about {MIN_RUNS} pairs; use more images or seeds",
            a.len()
        ));
    }
    if level == Level::Runs && a.len() < T_INTERVAL_BELOW && !rows.is_empty() {
        warnings.push(format!(
            "fewer than {T_INTERVAL_BELOW} paired runs: Student-t intervals are shown as well, and a row is marked significant only if both intervals exclude zero"
        ));
    }
    if level == Level::Samples && !rows.is_empty() {
        warnings.push("sample-level comparison of two single results: run-to-run noise is not included".into());
    }
    if blocked {
        warnings.push(format!(
            "consecutive frames and frame pairs are correlated: they were resampled in moving blocks of max(gap, {MIN_BLOCK}) units (the block column)"
        ));
    }
    Ok(Comparison {
        tool: TOOL.into(),
        version: env!("CARGO_PKG_VERSION").into(),
        kind: "compare".into(),
        a: a.iter().map(|(n, _)| n.clone()).collect(),
        b: b.iter().map(|(n, _)| n.clone()).collect(),
        bootstrap: *cfg,
        rows,
        warnings,
        skipped,
    })
}

impl Comparison {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("comparisons always serialise")
    }

    /// A Markdown table of the rows.
    pub fn to_markdown(&self) -> String {
        let pct = (self.bootstrap.confidence * 100.0).round();
        let mut s = format!(
            "A: {}\n\nB: {}\n\nPaired bootstrap, {} resamples, seed {}, {pct}% intervals.\n\n",
            self.a.join(", "),
            self.b.join(", "),
            self.bootstrap.resamples,
            self.bootstrap.seed
        );
        let _ = writeln!(
            s,
            "| metric | level | n | block | A | B | B - A [{pct}% CI] | B / A - 1 [{pct}% CI] | mean of per-unit B / A - 1 | p |"
        );
        s.push_str("|---|---|---:|---:|---:|---:|---|---|---|---:|\n");
        let t = |iv: Option<[f64; 2]>, fmt: &dyn Fn(f64) -> String| {
            iv.map_or(String::new(), |[lo, hi]| format!(" (t: [{}, {}])", fmt(lo), fmt(hi)))
        };
        let pct_of = |v: f64| format!("{:+.2}%", 100.0 * v);
        for r in &self.rows {
            let c = &r.result;
            let mark = if r.significant { " *" } else { "" };
            let _ = writeln!(
                s,
                "| {}{mark} | {} | {} | {} | {} | {} | {} [{}, {}]{} | {:+.2}% [{:+.2}%, {:+.2}%] | {} | {} |",
                r.metric,
                match r.level {
                    Level::Samples => "samples",
                    Level::Runs => "runs",
                },
                c.n,
                c.block.map_or("-".to_string(), |b| b.to_string()),
                format_value(c.mean_a),
                format_value(c.mean_b),
                format_value(c.diff),
                format_value(c.diff_lo),
                format_value(c.diff_hi),
                t(c.t_diff, &format_value),
                100.0 * c.rel,
                100.0 * c.rel_lo,
                100.0 * c.rel_hi,
                c.pair_rel.map_or("-".to_string(), |[m, lo, hi]| format!(
                    "{} [{}, {}]{}",
                    pct_of(m),
                    pct_of(lo),
                    pct_of(hi),
                    t(c.t_pair_rel, &pct_of)
                )),
                if c.p < 0.001 { "< 0.001".to_string() } else { format!("{:.3}", c.p) }
            );
        }
        if !self.rows.is_empty() {
            s.push_str("\n`*`: the interval of B - A excludes zero (both intervals, where a t interval is shown).\n");
        }
        for w in &self.warnings {
            let _ = writeln!(s, "\nNote: {w}.");
        }
        for (m, why) in &self.skipped {
            let _ = writeln!(s, "\nNot compared: {m} ({why}).");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(total: f64, samples: Option<(&str, Vec<f64>)>) -> Report {
        let mut r = Report::new("sampled");
        r.metric("total", total);
        if let Some((key, v)) = samples {
            r.samples("total", key, v);
        }
        r
    }

    #[test]
    fn sample_level_needs_equal_keys() {
        let a = vec![("a".to_string(), report(2.0, Some(("k", vec![1.0, 2.0, 3.0]))))];
        let b = vec![("b".to_string(), report(3.0, Some(("k", vec![2.0, 3.0, 4.0]))))];
        let c = compare_reports(&a, &b, &[], None, &BootstrapConfig { resamples: 200, ..Default::default() }).unwrap();
        assert_eq!(c.rows.len(), 1);
        assert_eq!(c.rows[0].level, Level::Samples);
        assert!(c.rows.iter().all(|r| r.result.pair_rel.is_none()));
        assert_eq!(c.rows[0].result.diff, 1.0);
        assert_eq!((c.rows[0].result.diff_lo, c.rows[0].result.diff_hi), (1.0, 1.0));
        assert!(c.to_markdown().contains("| total * | samples | 3 | - |"));
        let back: Comparison = serde_json::from_str(&c.to_json()).unwrap();
        assert_eq!(back, c);

        let b2 = vec![("b".to_string(), report(3.0, Some(("other", vec![2.0, 3.0, 4.0]))))];
        let c = compare_reports(&a, &b2, &[], None, &BootstrapConfig::default()).unwrap();
        assert!(c.rows.is_empty());
        assert_eq!(c.skipped.len(), 1);
    }

    #[test]
    fn run_level_pairs_by_position() {
        let a: Vec<(String, Report)> = (0..5).map(|i| (format!("a{i}"), report(100.0 + i as f64, None))).collect();
        let b: Vec<(String, Report)> = (0..5).map(|i| (format!("b{i}"), report(102.0 + i as f64, None))).collect();
        let c =
            compare_reports(&a, &b, &["total".into()], None, &BootstrapConfig { resamples: 300, ..Default::default() })
                .unwrap();
        assert_eq!(c.rows[0].level, Level::Runs);
        assert_eq!(c.warnings.len(), 2, "5 pairs are few, and get t intervals: {:?}", c.warnings);
        assert_eq!(c.rows[0].result.n, 5);
        assert!((c.rows[0].result.diff - 2.0).abs() < 1e-12);
        // Constant differences: a zero-width t interval at 2.
        assert_eq!(c.rows[0].result.t_diff, Some([2.0, 2.0]));
        assert!(c.to_markdown().contains("(t: [2.000, 2.000])"));
        assert!(compare_reports(&a, &b[..4], &[], None, &BootstrapConfig::default()).is_err());
        assert!(compare_reports(&a, &b, &["missing".into()], None, &BootstrapConfig::default()).is_err());
        let mut other = b.clone();
        other[0].1.kind = "energy".into();
        assert!(compare_reports(&a, &other, &[], None, &BootstrapConfig::default()).is_err());
    }

    #[test]
    fn warns_about_pairs_that_measure_different_things() {
        let run = |name: &str, px: &str, total: f64| {
            let mut r = report(total, None);
            r.input("pixel_set", px).input("patch_size", 5).input("image", name);
            (name.to_string(), r)
        };
        let a: Vec<_> = (0..12).map(|i| run(&format!("a{i}"), "pixels/s1", 100.0 + i as f64)).collect();
        let mut b: Vec<_> = (0..12).map(|i| run(&format!("b{i}"), "pixels/s1", 101.0 + i as f64)).collect();
        let cfg = BootstrapConfig { resamples: 200, ..Default::default() };
        let clean = compare_reports(&a, &b, &[], None, &cfg).unwrap();
        assert!(!clean.warnings.iter().any(|w| w.contains("differs")), "{:?}", clean.warnings);
        b[3].1.input("pixel_set", "pixels/s2");
        let c = compare_reports(&a, &b, &[], None, &cfg).unwrap();
        let w: Vec<_> = c.warnings.iter().filter(|w| w.contains("differs")).collect();
        assert_eq!(w.len(), 1, "{:?}", c.warnings);
        assert!(w[0].contains("'pixel_set'") && w[0].contains("a3 vs b3") && w[0].contains("1 of 12"));
    }

    #[test]
    fn sequence_samples_use_blocks() {
        assert_eq!(block_length("pairs/gap1/0-20/flow=x"), Some(5));
        assert_eq!(block_length("pairs/gap8/0-20/flow=x/without:3"), Some(8));
        assert_eq!(block_length("frames/1-19/flow=x"), Some(5));
        assert_eq!(block_length("pixels/uniform-v1/10x10/n5/seed1"), None);
        assert_eq!(block_length("pairs/gapx/0-3"), None);

        let values = |off: f64| (0..30).map(|i| off + (i as f64 * 0.3).sin()).collect::<Vec<_>>();
        let mut ra = report(1.0, None);
        ra.samples("warp5", "pairs/gap5/0-34/flow=f", values(1.0));
        let mut rb = report(1.0, None);
        rb.samples("warp5", "pairs/gap5/0-34/flow=f", values(1.1));
        let c = compare_reports(
            &[("a".into(), ra)],
            &[("b".into(), rb)],
            &["warp5".into()],
            None,
            &BootstrapConfig { resamples: 200, ..Default::default() },
        )
        .unwrap();
        assert_eq!(c.rows[0].result.block, Some(5));
        assert!(c.warnings.iter().any(|w| w.contains("moving blocks")));
        assert!(c.to_markdown().contains("| warp5 * | samples | 30 | 5 |"));
    }
}
