//! Video colour tags: the YUV matrix, range, primaries and transfer function.
//!
//! Synthesis runs on display-referred RGB (as 8-bit images are). Decoding YUV video with the
//! wrong matrix shifts colours visibly (BT.601 against BT.709 differs by up to dozens of levels),
//! and an encode without tags leaves players guessing. So FreeSynth reads the tags of the input,
//! decodes with them (or with documented defaults when the input has none) and writes the same
//! tags on the output.

use serde::{Deserialize, Serialize};

/// The colour description of a video stream, with ffmpeg's names (`bt709`, `smpte170m`, `tv`,
/// ...). `None` means untagged.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColourTags {
    /// YUV-to-RGB matrix (`colorspace`): `bt709`, `smpte170m`, `bt470bg`, `bt2020nc`, ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix: Option<String>,
    /// Value range: `tv` (limited) or `pc` (full).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<String>,
    /// Colour primaries: `bt709`, `smpte170m`, `bt470bg`, `bt2020`, ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primaries: Option<String>,
    /// Transfer function: `bt709`, `iec61966-2-1` (sRGB), `smpte2084` (PQ), ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer: Option<String>,
}

/// The matrix names a project may set (ffmpeg's `colorspace` names).
pub const MATRIX_NAMES: [&str; 9] =
    ["bt709", "smpte170m", "bt470bg", "bt2020nc", "bt2020c", "smpte240m", "fcc", "ycgco", "gbr"];

/// The range names a project may set: `tv` (limited) and `pc` (full).
pub const RANGE_NAMES: [&str; 2] = ["tv", "pc"];

/// The colour primaries a project may set (ffmpeg's `color_primaries` names).
pub const PRIMARIES_NAMES: [&str; 12] = [
    "bt709",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "film",
    "bt2020",
    "smpte428",
    "smpte431",
    "smpte432",
    "jedec-p22",
    "ebu3213",
];

/// The transfer functions a project may set (ffmpeg's `color_trc` names).
pub const TRANSFER_NAMES: [&str; 16] = [
    "bt709",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "linear",
    "log100",
    "log316",
    "iec61966-2-4",
    "bt1361e",
    "iec61966-2-1",
    "bt2020-10",
    "bt2020-12",
    "smpte2084",
    "smpte428",
    "arib-std-b67",
];

/// True when `value` can go into an ffmpeg filter graph as an option value as it is: only
/// lower-case ASCII letters, digits and `-` (no `:`, `,`, `;`, `[`, `=` or quotes, which would
/// end the option or the filter).
pub(crate) fn is_plain_name(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Turns ffprobe's `unknown`/`unspecified`/empty into `None`.
fn known(value: Option<&str>) -> Option<String> {
    value.filter(|v| !v.is_empty() && *v != "unknown" && *v != "unspecified" && *v != "reserved").map(str::to_owned)
}

impl ColourTags {
    /// Tags as ffprobe reports them (`unknown` values become `None`).
    pub fn from_probe(
        matrix: Option<&str>,
        range: Option<&str>,
        primaries: Option<&str>,
        transfer: Option<&str>,
    ) -> Self {
        Self { matrix: known(matrix), range: known(range), primaries: known(primaries), transfer: known(transfer) }
    }

    /// BT.709 throughout, limited range: the usual tags of HD video, and what FreeSynth writes
    /// for output made from image sequences (sRGB images share BT.709's primaries).
    pub fn bt709() -> Self {
        let s = |v: &str| Some(v.to_owned());
        Self { matrix: s("bt709"), range: s("tv"), primaries: s("bt709"), transfer: s("bt709") }
    }

    /// True when no tag is set.
    pub fn is_untagged(&self) -> bool {
        self.matrix.is_none() && self.range.is_none() && self.primaries.is_none() && self.transfer.is_none()
    }

    /// These tags with the gaps filled the way players usually guess for a video of
    /// `width` x `height` stored as `pix_fmt`: BT.709 for HD (at least 1280 wide or 720 high),
    /// BT.601 (`smpte170m`) for smaller video, limited range except for the full-range `yuvj`
    /// formats and RGB formats.
    pub fn completed(&self, width: usize, height: usize, pix_fmt: &str) -> Self {
        let hd = width >= 1280 || height >= 720;
        let sd_or_hd = if hd { "bt709" } else { "smpte170m" };
        let full = pix_fmt.starts_with("yuvj") || is_rgb_format(pix_fmt);
        let matrix = self.matrix.clone().unwrap_or_else(|| sd_or_hd.to_owned());
        let primaries = self.primaries.clone().unwrap_or_else(|| match matrix.as_str() {
            "bt709" => "bt709".to_owned(),
            "bt2020nc" | "bt2020c" => "bt2020".to_owned(),
            "bt470bg" => "bt470bg".to_owned(),
            _ => sd_or_hd.to_owned(),
        });
        let transfer = self.transfer.clone().unwrap_or_else(|| "bt709".to_owned());
        let range = self.range.clone().unwrap_or_else(|| if full { "pc" } else { "tv" }.to_owned());
        Self { matrix: Some(matrix), range: Some(range), primaries: Some(primaries), transfer: Some(transfer) }
    }

    /// These tags without the values that are not plain names ([`is_plain_name`]), with a
    /// note for each value dropped. Tags go into ffmpeg filter graphs, so a value such as
    /// `bt709,movie=...` must never reach one.
    pub(crate) fn sanitized(&self) -> (Self, Vec<String>) {
        let mut notes = Vec::new();
        let mut keep = |what: &str, value: &Option<String>| match value {
            Some(v) if !is_plain_name(v) => {
                notes.push(format!("the colour {what} tag {v:?} is not a valid name and is ignored"));
                None
            }
            other => other.clone(),
        };
        let clean = Self {
            matrix: keep("matrix", &self.matrix),
            range: keep("range", &self.range),
            primaries: keep("primaries", &self.primaries),
            transfer: keep("transfer", &self.transfer),
        };
        (clean, notes)
    }

    /// The colour tags to write on a YUV encode of `width` x `height` frames, and the swscale
    /// matrix that converts the RGB frames to match them. The written `colorspace` tag always
    /// names the matrix the conversion uses: a source matrix swscale cannot produce (RGB
    /// sources tagged `gbr`, `ycgco`, untagged ...) is replaced by the usual default for the
    /// frame size (BT.709 for HD, BT.601 `smpte170m` for SD), with limited range. `limited`
    /// forces limited range (for codecs that are limited-range by convention, such as ProRes).
    /// Call on [`completed`](Self::completed) tags.
    pub(crate) fn for_yuv_output(&self, width: usize, height: usize, limited: bool) -> (Self, &'static str) {
        let hd = width >= 1280 || height >= 720;
        let source = match self.matrix.as_deref() {
            Some("bt709") => Some(("bt709", "bt709")),
            Some("smpte170m") => Some(("smpte170m", "bt601")),
            Some("bt470bg") => Some(("bt470bg", "bt601")),
            Some("bt2020nc" | "bt2020c") => Some(("bt2020nc", "bt2020")),
            Some("smpte240m") => Some(("smpte240m", "smpte240m")),
            Some("fcc") => Some(("fcc", "fcc")),
            _ => None,
        };
        let ((matrix, swscale), range) = match source {
            Some(m) => (m, if limited { "tv" } else { self.swscale_range() }),
            None => (if hd { ("bt709", "bt709") } else { ("smpte170m", "bt601") }, "tv"),
        };
        let tags = Self { matrix: Some(matrix.to_owned()), range: Some(range.to_owned()), ..self.clone() };
        (tags, swscale)
    }

    /// The first value that is not in the vocabulary a project may use ([`MATRIX_NAMES`],
    /// [`RANGE_NAMES`], [`PRIMARIES_NAMES`], [`TRANSFER_NAMES`]), as an error message.
    pub fn check_names(&self) -> Result<(), String> {
        let check = |what: &str, value: &Option<String>, names: &[&str]| match value {
            Some(v) if !names.contains(&v.as_str()) => {
                Err(format!("unknown colour {what} '{v}' (expected one of {})", names.join(", ")))
            }
            _ => Ok(()),
        };
        check("matrix", &self.matrix, &MATRIX_NAMES)?;
        check("range", &self.range, &RANGE_NAMES)?;
        check("primaries", &self.primaries, &PRIMARIES_NAMES)?;
        check("transfer", &self.transfer, &TRANSFER_NAMES)
    }

    /// The swscale name of the matrix (for `in_color_matrix`/`out_color_matrix`).
    pub(crate) fn swscale_matrix(&self) -> &'static str {
        match self.matrix.as_deref() {
            Some("bt709") => "bt709",
            Some("smpte170m" | "bt470bg") => "bt601",
            Some("bt2020nc" | "bt2020c") => "bt2020",
            Some("smpte240m") => "smpte240m",
            Some("fcc") => "fcc",
            _ => "bt709",
        }
    }

    /// The swscale name of the range.
    pub(crate) fn swscale_range(&self) -> &'static str {
        match self.range.as_deref() {
            Some("pc" | "full" | "jpeg") => "pc",
            _ => "tv",
        }
    }

    /// True for transfer functions FreeSynth does not handle as display-referred video (HDR).
    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer.as_deref(), Some("smpte2084" | "arib-std-b67"))
    }

    /// A short description such as `bt709/tv, primaries bt709, transfer bt709`.
    pub fn describe(&self) -> String {
        let s = |v: &Option<String>| v.clone().unwrap_or_else(|| "?".into());
        format!(
            "matrix {}, range {}, primaries {}, transfer {}",
            s(&self.matrix),
            s(&self.range),
            s(&self.primaries),
            s(&self.transfer)
        )
    }
}

/// True for ffmpeg pixel formats that store RGB (no YUV matrix involved).
pub fn is_rgb_format(pix_fmt: &str) -> bool {
    ["rgb", "bgr", "argb", "abgr", "gbr", "0rgb", "0bgr", "x2rgb", "x2bgr"].iter().any(|p| pix_fmt.starts_with(p))
}

/// True for ffmpeg pixel formats with an alpha channel.
pub fn has_alpha_format(pix_fmt: &str) -> bool {
    ["yuva", "gbrap", "rgba", "bgra", "argb", "abgr", "ya8", "ya16", "ayuv", "vuya", "rgbaf"]
        .iter()
        .any(|p| pix_fmt.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_tags_are_none() {
        let t = ColourTags::from_probe(Some("bt709"), Some("tv"), Some("unknown"), None);
        assert_eq!(t.matrix.as_deref(), Some("bt709"));
        assert_eq!(t.primaries, None);
        assert!(!t.is_untagged());
        assert!(ColourTags::from_probe(Some("unknown"), None, None, Some("")).is_untagged());
    }

    #[test]
    fn defaults_follow_the_resolution() {
        let hd = ColourTags::default().completed(1920, 1080, "yuv420p");
        assert_eq!(hd, ColourTags::bt709());
        let sd = ColourTags::default().completed(720, 404, "yuv420p");
        assert_eq!(sd.matrix.as_deref(), Some("smpte170m"));
        assert_eq!(sd.swscale_matrix(), "bt601");
        assert_eq!(sd.range.as_deref(), Some("tv"));
        let jpeg = ColourTags::default().completed(640, 480, "yuvj420p");
        assert_eq!(jpeg.swscale_range(), "pc");
        // Present tags are kept.
        let tagged = ColourTags::from_probe(Some("bt2020nc"), Some("tv"), None, Some("smpte2084"));
        let c = tagged.completed(3840, 2160, "yuv420p10le");
        assert_eq!((c.primaries.as_deref(), c.swscale_matrix()), (Some("bt2020"), "bt2020"));
        assert!(c.is_hdr());
    }

    #[test]
    fn filter_graph_values_are_plain_names() {
        let evil = ColourTags {
            matrix: Some("bt709,null[x];movie=http\\://h/a.mov,nullsink;[x]setparams=colorspace=bt709".into()),
            range: Some("tv".into()),
            primaries: Some("bt709:x".into()),
            transfer: Some("iec61966-2-1".into()),
        };
        let (clean, notes) = evil.sanitized();
        assert_eq!((clean.matrix, clean.primaries), (None, None));
        assert_eq!((clean.range.as_deref(), clean.transfer.as_deref()), (Some("tv"), Some("iec61966-2-1")));
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(evil.check_names().unwrap_err().contains("unknown colour matrix"));
        ColourTags::bt709().check_names().unwrap();
        let odd = ColourTags { range: Some("full".into()), ..ColourTags::default() };
        assert!(odd.check_names().unwrap_err().contains("tv, pc"));
    }

    #[test]
    fn yuv_output_tags_name_the_conversion_matrix() {
        // RGB sources (FFV1 bgr0, PNG in MOV) are tagged gbr/pc: the YUV output gets a real
        // matrix and limited range.
        let rgb = ColourTags::from_probe(Some("gbr"), Some("pc"), Some("bt709"), Some("iec61966-2-1"));
        let (t, sws) = rgb.completed(1920, 1080, "bgr0").for_yuv_output(1920, 1080, false);
        assert_eq!((t.matrix.as_deref(), t.range.as_deref(), sws), (Some("bt709"), Some("tv"), "bt709"));
        assert_eq!(t.transfer.as_deref(), Some("iec61966-2-1"), "primaries and transfer are kept");
        let (t, sws) = rgb.completed(640, 480, "bgr0").for_yuv_output(640, 480, false);
        assert_eq!((t.matrix.as_deref(), sws), (Some("smpte170m"), "bt601"));
        let (t, _) = ColourTags { matrix: Some("ycgco".into()), ..ColourTags::default() }.for_yuv_output(64, 48, false);
        assert_eq!(t.matrix.as_deref(), Some("smpte170m"));
        // YUV matrices swscale knows are kept (constant-luminance BT.2020 is converted and
        // tagged as the non-constant one).
        let pal = ColourTags::from_probe(Some("bt470bg"), Some("pc"), None, None).completed(720, 576, "yuvj420p");
        let (t, sws) = pal.for_yuv_output(720, 576, false);
        assert_eq!((t.matrix.as_deref(), t.range.as_deref(), sws), (Some("bt470bg"), Some("pc"), "bt601"));
        let (t, _) = pal.for_yuv_output(720, 576, true);
        assert_eq!(t.range.as_deref(), Some("tv"), "limited-range codecs");
        let c = ColourTags { matrix: Some("bt2020c".into()), ..ColourTags::default() };
        assert_eq!(c.for_yuv_output(3840, 2160, false).0.matrix.as_deref(), Some("bt2020nc"));
    }

    #[test]
    fn pixel_format_classes() {
        assert!(has_alpha_format("yuva444p10le") && has_alpha_format("rgba64le") && has_alpha_format("gbrap"));
        assert!(!has_alpha_format("gray") && !has_alpha_format("pal8") && !has_alpha_format("yuv420p"));
        assert!(is_rgb_format("rgb24") && is_rgb_format("gbrp16le") && !is_rgb_format("yuv444p"));
    }
}
