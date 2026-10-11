//! The corpus manifest: a list of named scenes with their generator parameters.
//!
//! ```toml
//! format_version = 1
//!
//! [[scene]]
//! name = "tiny-rigid"
//! tier = "ci"
//! description = "Smallest rigid case"
//! params = { width = 64, height = 48, frames = 6, family = "rigid" }
//! ```
//!
//! `params` takes the fields of [`SceneParams`]; missing fields take their defaults.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::scene::SceneParams;

/// Which test tier a scene belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Tiny cases that generate in milliseconds, for continuous integration.
    Ci,
    /// Full-size cases for manual and nightly evaluation.
    Full,
}

/// One manifest entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestScene {
    /// Directory name of the generated scene.
    pub name: String,
    pub tier: Tier,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub params: SceneParams,
}

/// The whole manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format_version: u32,
    #[serde(default, rename = "scene")]
    pub scenes: Vec<ManifestScene>,
}

impl Manifest {
    /// Parses a manifest and validates every entry.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let manifest: Manifest = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        if manifest.format_version != 1 {
            return Err(Error::Parse(format!("unsupported manifest format_version {}", manifest.format_version)));
        }
        let mut names = std::collections::HashSet::new();
        for s in &manifest.scenes {
            if s.name.is_empty() || !s.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                return Err(Error::Parse(format!("scene name '{}' must be non-empty [A-Za-z0-9_-]", s.name)));
            }
            if !names.insert(s.name.as_str()) {
                return Err(Error::Parse(format!("scene name '{}' appears twice", s.name)));
            }
            s.params.validate().map_err(|e| Error::Parse(format!("scene '{}': {e}", s.name)))?;
        }
        Ok(manifest)
    }

    /// Reads and parses a manifest file.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text).map_err(|e| Error::Parse(format!("{}: {e}", path.display())))
    }

    /// The entry called `name`.
    pub fn scene(&self, name: &str) -> Option<&ManifestScene> {
        self.scenes.iter().find(|s| s.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::StyleFamily;

    #[test]
    fn parses_entries_with_defaults() {
        let m = Manifest::parse(
            r#"
            format_version = 1
            [[scene]]
            name = "a"
            tier = "ci"
            params = { width = 64, height = 48, family = "hatching" }
            [[scene]]
            name = "b"
            tier = "full"
            "#,
        )
        .unwrap();
        assert_eq!(m.scenes.len(), 2);
        let a = &m.scene("a").unwrap().params;
        assert_eq!((a.width, a.height, a.family, a.frames), (64, 48, StyleFamily::Hatching, 30));
        assert_eq!(m.scene("b").unwrap().params, SceneParams::default());
    }

    #[test]
    fn rejects_bad_manifests() {
        let dup = "format_version = 1\n[[scene]]\nname='a'\ntier='ci'\n[[scene]]\nname='a'\ntier='ci'\n";
        assert!(Manifest::parse(dup).is_err());
        let typo = "format_version = 1\n[[scene]]\nname='a'\ntier='ci'\nparams = { widht = 3 }\n";
        assert!(Manifest::parse(typo).is_err());
        let invalid = "format_version = 1\n[[scene]]\nname='a'\ntier='ci'\nparams = { frames = 0 }\n";
        assert!(Manifest::parse(invalid).is_err());
        assert!(Manifest::parse("format_version = 2\n").is_err());
    }

    #[test]
    fn the_corpus_manifest_is_valid() {
        // Only inside the FreeSynth repository (not in a packaged crate).
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/manifest.toml");
        if !path.exists() {
            return;
        }
        let m = Manifest::load(&path).unwrap();
        assert!(m.scenes.iter().any(|s| s.tier == Tier::Ci));
        for family in StyleFamily::ALL {
            assert!(m.scenes.iter().any(|s| s.tier == Tier::Ci && s.params.family == family));
        }
    }
}
