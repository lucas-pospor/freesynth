//! The JSON result format shared by all metrics, and Markdown rendering.
//!
//! Every measurement produces one [`Report`]:
//!
//! ```json
//! {
//!   "tool": "freesynth-eval",
//!   "version": "0.1.0",
//!   "kind": "sampled",
//!   "label": "seed 1",
//!   "inputs": { "image": "out.png", "patch_size": 5 },
//!   "metrics": { "total": 105.4, "style": 27.1, "guide": 78.3 },
//!   "samples": { "total": { "key": "pixels/uniform-v1/1200x912/n20000/seed1", "values": [ ... ] } }
//! }
//! ```
//!
//! `metrics` holds scalar results. `samples` holds per-unit values (pixels, frame pairs, frames)
//! behind some of them; two reports can be compared unit by unit only when the `key`s of a sample
//! set are equal, which guarantees that both were measured on the same units.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Name written to [`Report::tool`].
pub const TOOL: &str = "freesynth-eval";

/// One measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub tool: String,
    pub version: String,
    /// Which measurement produced the report (`energy`, `sampled`, `occupancy`, `sharpness`,
    /// `temporal`).
    pub kind: String,
    /// Free-form name of the measured item, for tables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// What was measured and how (file names, parameters).
    #[serde(default)]
    pub inputs: BTreeMap<String, serde_json::Value>,
    /// Scalar results.
    pub metrics: BTreeMap<String, f64>,
    /// Per-unit values behind some metrics, for paired comparisons.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub samples: BTreeMap<String, Samples>,
    /// The target pixels a sampled measurement used, so that another image can be measured on
    /// the same pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixels: Option<PixelSet>,
}

/// Per-unit values of one metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Samples {
    /// Identifies the units (which pixels, which frame pairs).
    pub key: String,
    pub values: Vec<f64>,
}

/// A set of target pixels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelSet {
    /// Identifies the set; equal keys mean equal sets.
    pub key: String,
    pub width: usize,
    pub height: usize,
    /// `[x, y]` of every pixel.
    pub coords: Vec<[u32; 2]>,
}

impl Report {
    /// An empty report of the given kind.
    pub fn new(kind: &str) -> Self {
        Self {
            tool: TOOL.into(),
            version: env!("CARGO_PKG_VERSION").into(),
            kind: kind.into(),
            label: None,
            inputs: BTreeMap::new(),
            metrics: BTreeMap::new(),
            samples: BTreeMap::new(),
            pixels: None,
        }
    }

    /// Records an input.
    pub fn input(&mut self, name: &str, value: impl Into<serde_json::Value>) -> &mut Self {
        self.inputs.insert(name.into(), value.into());
        self
    }

    /// Records a scalar result. Non-finite values are skipped (JSON cannot represent them).
    pub fn metric(&mut self, name: &str, value: f64) -> &mut Self {
        if value.is_finite() {
            self.metrics.insert(name.into(), value);
        }
        self
    }

    /// Records per-unit values.
    pub fn samples(&mut self, name: &str, key: impl Into<String>, values: Vec<f64>) -> &mut Self {
        self.samples.insert(name.into(), Samples { key: key.into(), values });
        self
    }

    /// Pretty-printed JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("reports always serialise")
    }

    pub fn from_json(s: &str) -> Result<Self, Error> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Invalid(format!("cannot read '{}': {e}", path.display())))?;
        Self::from_json(&text).map_err(|e| Error::Invalid(format!("'{}': {e}", path.display())))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        std::fs::write(path, self.to_json() + "\n")?;
        Ok(())
    }

    /// A short Markdown summary: title and a two-column table of the metrics.
    pub fn to_markdown(&self) -> String {
        let mut s = format!("### {}", self.kind);
        if let Some(l) = &self.label {
            let _ = write!(s, ": {l}");
        }
        s.push_str("\n\n| metric | value |\n|---|---:|\n");
        for (k, v) in &self.metrics {
            let _ = writeln!(s, "| {k} | {} |", format_value(*v));
        }
        s
    }
}

/// Formats a metric value with about four significant digits.
pub fn format_value(v: f64) -> String {
    let a = v.abs();
    if v == 0.0 {
        "0".into()
    } else if a >= 1000.0 {
        format!("{v:.1}")
    } else if a >= 1.0 {
        format!("{v:.3}")
    } else if a >= 1e-3 {
        format!("{v:.5}")
    } else {
        format!("{v:.3e}")
    }
}

/// A Markdown table with one row per report and one column per metric.
///
/// `metrics` selects and orders the columns; when empty, every metric that appears in any report
/// is shown. Rows are named by label, falling back to `names`.
pub fn markdown_table(reports: &[(String, Report)], metrics: &[String]) -> String {
    let columns: Vec<String> = if metrics.is_empty() {
        let mut all: Vec<String> = reports.iter().flat_map(|(_, r)| r.metrics.keys().cloned()).collect();
        all.sort();
        all.dedup();
        all
    } else {
        metrics.to_vec()
    };
    let mut s = String::from("| run |");
    for c in &columns {
        let _ = write!(s, " {c} |");
    }
    s.push_str("\n|---|");
    s.push_str(&"---:|".repeat(columns.len()));
    s.push('\n');
    for (name, r) in reports {
        let _ = write!(s, "| {} |", r.label.as_deref().unwrap_or(name));
        for c in &columns {
            match r.metrics.get(c) {
                Some(v) => {
                    let _ = write!(s, " {} |", format_value(*v));
                }
                None => s.push_str(" - |"),
            }
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip() {
        let mut r = Report::new("sampled");
        r.input("image", "a.png").input("patch_size", 5).metric("total", 1.5).metric("bad", f64::NAN);
        r.samples("total", "k", vec![1.0, 2.0]);
        r.pixels = Some(PixelSet { key: "k".into(), width: 4, height: 3, coords: vec![[0, 0], [3, 2]] });
        r.label = Some("x".into());
        let back = Report::from_json(&r.to_json()).unwrap();
        assert_eq!(back, r);
        assert!(!back.metrics.contains_key("bad"));
        // Optional fields may be absent.
        let minimal = r#"{"tool":"freesynth-eval","version":"0","kind":"energy","metrics":{"total":2}}"#;
        let m = Report::from_json(minimal).unwrap();
        assert_eq!(m.metrics["total"], 2.0);
        assert!(m.samples.is_empty() && m.pixels.is_none());
    }

    #[test]
    fn markdown() {
        let mut a = Report::new("energy");
        a.metric("total", 110.123).metric("style", 27.5);
        let mut b = Report::new("energy");
        b.metric("total", 0.000_12);
        b.label = Some("b".into());
        let t = markdown_table(&[("a.json".into(), a.clone()), ("b.json".into(), b)], &[]);
        assert!(t.starts_with("| run | style | total |\n|---|---:|---:|\n"));
        assert!(t.contains("| a.json | 27.500 | 110.123 |"));
        assert!(t.contains("| b | - | 1.200e-4 |"));
        assert!(a.to_markdown().contains("| total | 110.123 |"));
    }
}
