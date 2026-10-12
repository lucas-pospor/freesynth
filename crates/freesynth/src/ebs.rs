//! Importing EbSynth `.ebs` project files.
//!
//! An `.ebs` file describes a shot as three frame patterns (video, mask and keyframe images),
//! the guide weights, and a list of keyframes, each with the frames it should cover in each
//! direction and its own output folder. FreeSynth reads it into a [`Project`]: the patterns and
//! weights carry over directly, and the per-keyframe ranges become one shot whose overlapping
//! ranges are blended into a single output sequence.
//! Anything that does not carry over exactly is reported as a warning. Writing `.ebs` files is
//! not supported.
//!
//! The format is little-endian: the magic `EBS05\0`; the video, mask and keys patterns (each an
//! `i32` byte length and UTF-8 bytes, paths relative to the file with `\` separators); a mask
//! on/off byte; six `f32` weights (keyframe, video, mask, mapping, de-flicker, diversity); the
//! keyframe count (`i32`); per keyframe its frame (`i32`), backward and forward on/off bytes,
//! backward and forward stop frames (`i32`) and output pattern; then an optional trailer tagged
//! `V02\0` with the synthesis detail (`i32`, 1 = high to 4 = lowest), a use-GPU byte, the
//! After Effects export frame rate (`f32`) and the window width (`i32`).

use std::path::Path;

use freesynth_core::Preset;
use freesynth_video::{Backend, pattern_path};

use crate::error::Error;
use crate::paths::{for_project, to_project_string};
use crate::project::{KeyframeEntry, Project};

/// The magic bytes at the start of an `.ebs` file.
const MAGIC: &[u8; 6] = b"EBS05\0";
/// The tag of the optional trailer.
const TRAILER: &[u8; 4] = b"V02\0";
/// More keyframes than this is taken as a corrupt file.
const MAX_KEYFRAMES: i32 = 100_000;
/// Longer strings than this are taken as a corrupt file.
const MAX_STRING: i32 = 1 << 16;

/// One keyframe record of an `.ebs` file.
#[derive(Clone, Debug, PartialEq)]
pub struct EbsKeyframe {
    /// The keyframe's frame number.
    pub frame: i32,
    /// Lowest frame to synthesize backward to, when backward synthesis is on.
    pub backward_stop: Option<i32>,
    /// Highest frame to synthesize forward to, when forward synthesis is on.
    pub forward_stop: Option<i32>,
    /// The keyframe's output pattern (as stored).
    pub output: String,
}

impl EbsKeyframe {
    /// First frame the keyframe covers.
    pub fn start(&self) -> i32 {
        self.backward_stop.unwrap_or(self.frame)
    }

    /// Last frame the keyframe covers.
    pub fn end(&self) -> i32 {
        self.forward_stop.unwrap_or(self.frame)
    }
}

/// The optional trailer of an `.ebs` file.
#[derive(Clone, Debug, PartialEq)]
pub struct EbsTrailer {
    /// Synthesis detail: 1 high, 2 medium, 3 low, 4 lowest.
    pub synthesis_detail: i32,
    pub use_gpu: bool,
    /// Frame rate of the After Effects export.
    pub ae_fps: f32,
    pub window_width: i32,
}

/// The contents of an `.ebs` file, as stored.
#[derive(Clone, Debug, PartialEq)]
pub struct EbsFile {
    pub video: String,
    pub mask: String,
    pub keys: String,
    pub mask_enabled: bool,
    pub weight_keyframe: f32,
    pub weight_video: f32,
    pub weight_mask: f32,
    pub weight_mapping: f32,
    pub weight_deflicker: f32,
    pub diversity: f32,
    pub keyframes: Vec<EbsKeyframe>,
    pub trailer: Option<EbsTrailer>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len()).ok_or_else(|| {
            Error::Import(format!("the file ends inside the {what} (at byte {} of {})", self.pos, self.bytes.len()))
        })?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn i32(&mut self, what: &str) -> Result<i32, Error> {
        Ok(i32::from_le_bytes(self.take(4, what)?.try_into().expect("4 bytes")))
    }

    fn f32(&mut self, what: &str) -> Result<f32, Error> {
        let v = f32::from_le_bytes(self.take(4, what)?.try_into().expect("4 bytes"));
        if !v.is_finite() {
            return Err(Error::Import(format!("the {what} is not a finite number")));
        }
        Ok(v)
    }

    fn flag(&mut self, what: &str) -> Result<bool, Error> {
        Ok(self.take(1, what)?[0] != 0)
    }

    fn string(&mut self, what: &str) -> Result<String, Error> {
        let len = self.i32(what)?;
        if !(0..=MAX_STRING).contains(&len) {
            return Err(Error::Import(format!("the {what} has an invalid length ({len})")));
        }
        let bytes = self.take(len as usize, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::Import(format!("the {what} is not valid UTF-8")))
    }
}

/// Parses the bytes of an `.ebs` file.
pub fn parse_ebs(bytes: &[u8]) -> Result<EbsFile, Error> {
    let mut r = Reader { bytes, pos: 0 };
    if r.take(MAGIC.len(), "header")? != MAGIC {
        return Err(Error::Import("not an EbSynth project file (expected the EBS05 header)".into()));
    }
    let video = r.string("video pattern")?;
    let mask = r.string("mask pattern")?;
    let keys = r.string("keys pattern")?;
    let mask_enabled = r.flag("mask switch")?;
    let weight_keyframe = r.f32("keyframe weight")?;
    let weight_video = r.f32("video weight")?;
    let weight_mask = r.f32("mask weight")?;
    let weight_mapping = r.f32("mapping weight")?;
    let weight_deflicker = r.f32("de-flicker weight")?;
    let diversity = r.f32("diversity")?;
    let count = r.i32("keyframe count")?;
    if !(0..=MAX_KEYFRAMES).contains(&count) {
        return Err(Error::Import(format!("invalid keyframe count {count}")));
    }
    let mut keyframes = Vec::with_capacity(count as usize);
    for i in 0..count {
        let what = |field: &str| format!("{field} of keyframe {}", i + 1);
        let frame = r.i32(&what("frame"))?;
        let backward = r.flag(&what("backward switch"))?;
        let forward = r.flag(&what("forward switch"))?;
        let backward_stop = r.i32(&what("backward stop"))?;
        let forward_stop = r.i32(&what("forward stop"))?;
        let output = r.string(&what("output pattern"))?;
        keyframes.push(EbsKeyframe {
            frame,
            backward_stop: backward.then_some(backward_stop),
            forward_stop: forward.then_some(forward_stop),
            output,
        });
    }
    // The trailer is optional; anything else after the keyframes is ignored.
    let trailer = match r.take(TRAILER.len(), "trailer") {
        Ok(tag) if tag == TRAILER => Some(EbsTrailer {
            synthesis_detail: r.i32("synthesis detail")?,
            use_gpu: r.flag("GPU switch")?,
            ae_fps: r.f32("export frame rate")?,
            window_width: r.i32("window width")?,
        }),
        _ => None,
    };
    Ok(EbsFile {
        video,
        mask,
        keys,
        mask_enabled,
        weight_keyframe,
        weight_video,
        weight_mask,
        weight_mapping,
        weight_deflicker,
        diversity,
        keyframes,
        trailer,
    })
}

/// Converts a stored frame pattern to FreeSynth's syntax: `/` separators, and a printf-style
/// `%0Nd` turned into `[N×#]`. Returns the pattern and a warning
/// when something could not be converted.
pub fn convert_pattern(stored: &str) -> (String, Option<String>) {
    let text = stored.replace('\\', "/");
    let has_group = text.rfind(']').is_some_and(|end| {
        let before = &text[..end];
        let hashes = before.len() - before.trim_end_matches('#').len();
        hashes > 0 && before[..before.len() - hashes].ends_with('[')
    });
    if !text.contains('%') {
        return (text, None);
    }
    // Patterns may also be printf-style: `%%` is a literal `%`, and `%0Nd`, `%Nd` or `%d` is the
    // frame number.
    let mut out = String::new();
    let mut warning = None;
    let mut chars = text.chars().peekable();
    let mut converted = has_group;
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let mut spec = String::new();
        while let Some(&d) = chars.peek() {
            if d.is_ascii_digit() {
                spec.push(d);
                chars.next();
            } else {
                break;
            }
        }
        match chars.next() {
            Some('%') if spec.is_empty() => out.push('%'),
            Some('d') if !converted => {
                let width: usize = spec.trim_start_matches('0').parse().unwrap_or(1).max(1);
                out.push('[');
                out.extend(std::iter::repeat_n('#', width));
                out.push(']');
                converted = true;
            }
            other => {
                warning = Some(format!("the pattern '{stored}' holds a '%' sequence that was kept as written"));
                out.push('%');
                out.push_str(&spec);
                out.extend(other);
            }
        }
    }
    (out, warning)
}

/// The number of `#` in a pattern's frame number group (5 when there is none).
fn number_width(pattern: &str) -> usize {
    let name = pattern.rsplit('/').next().unwrap_or(pattern);
    let mut best = 0;
    let mut run = 0;
    for c in name.chars() {
        run = if c == '#' { run + 1 } else { 0 };
        best = best.max(run);
    }
    if best == 0 { 5 } else { best }
}

/// Why a converted pattern does not stay inside the `.ebs` file's folder, if it does not: an
/// absolute path (Unix, or Windows with a drive letter or a server name), or a relative path
/// whose `..` components climb above the folder.
fn outside_reason(pattern: &str) -> Option<&'static str> {
    let b = pattern.as_bytes();
    if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        return Some("is a Windows path with a drive letter");
    }
    if pattern.starts_with("//") {
        return Some("is a network path");
    }
    if pattern.starts_with('/') {
        return Some("is an absolute path");
    }
    let mut depth = 0i32;
    for part in pattern.split('/') {
        match part {
            "" | "." => {}
            ".." => depth -= 1,
            _ => depth += 1,
        }
        if depth < 0 {
            return Some("leaves the .ebs file's folder");
        }
    }
    None
}

/// The result of an import: the project and what did not carry over exactly. Fields may be
/// added.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Import {
    pub project: Project,
    /// What did not carry over exactly.
    pub warnings: Vec<String>,
    /// Notes about the output pattern (it was replaced, or the per-keyframe folders were
    /// merged): moot when the caller sets the output itself.
    pub output_warnings: Vec<String>,
}

/// Reads the `.ebs` file at `ebs_path` into a project to be saved at `project_path`: stored
/// paths are rebased from the `.ebs` file's directory onto the project's directory.
pub fn import_ebs(ebs_path: &Path, project_path: &Path) -> Result<Import, Error> {
    let bytes = std::fs::read(ebs_path).map_err(|source| Error::Io { path: ebs_path.to_owned(), source })?;
    let ebs = parse_ebs(&bytes).map_err(|e| match e {
        Error::Import(msg) => Error::Import(format!("{}: {msg}", ebs_path.display())),
        other => other,
    })?;
    let ebs_dir = ebs_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let project_dir = project_path.parent().map(Path::to_path_buf).unwrap_or_default();
    import_parsed(&ebs, &ebs_dir, &project_dir, project_path)
}

/// Builds the project for a parsed `.ebs` file whose paths are relative to `ebs_dir`.
fn import_parsed(ebs: &EbsFile, ebs_dir: &Path, project_dir: &Path, project_path: &Path) -> Result<Import, Error> {
    let mut warnings = Vec::new();
    let mut rebase = |stored: &str, what: &str| -> Result<String, Error> {
        let (pattern, warning) = convert_pattern(stored);
        warnings.extend(warning);
        if pattern.is_empty() {
            return Err(Error::Import(format!("the {what} pattern is empty")));
        }
        if let Some(reason) = outside_reason(&pattern) {
            warnings.push(format!("the {what} pattern '{pattern}' {reason}; check that it points at the right files"));
        }
        Ok(for_project(&ebs_dir.join(&pattern), project_dir))
    };
    let video = rebase(&ebs.video, "video")?;
    let keys = rebase(&ebs.keys, "keys")?;
    let mask = if ebs.mask_enabled { Some(rebase(&ebs.mask, "mask")?) } else { None };
    if ebs.keyframes.is_empty() {
        return Err(Error::Import("the project has no keyframe".into()));
    }
    if let Some(k) = ebs.keyframes.iter().find(|k| k.frame < 0 || k.start() < 0 || k.end() < 0) {
        return Err(Error::Import(format!("keyframe {} has a negative frame number or stop", k.frame)));
    }
    for k in &ebs.keyframes {
        if k.start() > k.frame || k.end() < k.frame {
            return Err(Error::Import(format!(
                "keyframe {}: its stop frames ({}-{}) do not include the keyframe",
                k.frame,
                k.start(),
                k.end()
            )));
        }
    }

    let mut project = Project::new(video.clone(), "");
    project.media.mask = mask;
    let g = &mut project.guides;
    g.style = ebs.weight_keyframe;
    g.video = ebs.weight_video;
    g.mask = ebs.weight_mask;
    g.mapping = ebs.weight_mapping;
    g.temporal = ebs.weight_deflicker;
    project.engine.uniformity = ebs.diversity;
    if let Some(t) = &ebs.trailer {
        project.engine.preset = match t.synthesis_detail {
            1 => Preset::High,
            2 => Preset::Default,
            3 => Preset::Fast,
            other => {
                warnings.push(format!("synthesis detail {other} has no equivalent; using the fast preset"));
                Preset::Fast
            }
        };
        if !t.use_gpu {
            project.engine.backend = Backend::Cpu;
        }
        if t.ae_fps.is_finite() && t.ae_fps > 0.0 {
            project.media.fps = Some(f64::from(t.ae_fps));
        }
    }

    // Keyframes, sorted, one per frame.
    let mut sorted = ebs.keyframes.clone();
    sorted.sort_by_key(|k| k.frame);
    sorted.dedup_by(|b, a| {
        let dup = a.frame == b.frame;
        if dup {
            warnings.push(format!("frame {} has two keyframes; the first one is used", a.frame));
        }
        dup
    });
    for k in &sorted {
        let frame = k.frame as usize;
        let image = pattern_path(&keys, frame)
            .map_err(|e| Error::Import(format!("keys pattern: {e}")))?
            .to_string_lossy()
            .into_owned();
        project.keyframes.push(KeyframeEntry { frame, image: to_project_string(Path::new(&image)) });
    }

    // The shot covers every keyframe's range; between neighbouring keyframes FreeSynth always
    // runs both passes and blends them.
    let first = sorted.iter().map(EbsKeyframe::start).min().unwrap_or(0) as usize;
    let last = sorted.iter().map(EbsKeyframe::end).max().unwrap_or(0) as usize;
    project.media.first = Some(first);
    project.media.last = Some(last);
    for pair in sorted.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if a.end() < b.frame || b.start() > a.frame {
            warnings.push(format!(
                "keyframes {} and {} did not both cover the frames between them; FreeSynth renders and blends \
                 both passes over frames {}-{}",
                a.frame, b.frame, a.frame, b.frame
            ));
        }
    }

    // One blended output instead of one folder per keyframe. A render overwrites numbered
    // images in the output folder, so a stored pattern outside the .ebs file's folder is not
    // trusted.
    let mut output_warnings = Vec::new();
    let outputs: Vec<String> = sorted.iter().map(|k| convert_pattern(&k.output).0).filter(|o| !o.is_empty()).collect();
    let mut shared = outputs.first().filter(|o| outputs.iter().all(|x| x == *o) && o.contains('#')).cloned();
    let stem = project_path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let default_output = format!("{stem}-out/[{}].png", "#".repeat(number_width(&video)));
    if let Some(reason) = shared.as_deref().and_then(outside_reason) {
        output_warnings.push(format!(
            "the output pattern '{}' {reason}; writing to '{default_output}' instead (use --out to choose)",
            shared.take().unwrap_or_default()
        ));
    } else if shared.is_none() && sorted.len() > 1 {
        output_warnings
            .push(format!("the {} keyframe output folders are replaced by one blended sequence", sorted.len()));
    }
    project.output.frames = match shared {
        Some(o) => for_project(&ebs_dir.join(o), project_dir),
        None => default_output,
    };
    project.set_base_dir(project_dir);
    if let Some(stem) = project_path.file_stem().and_then(|s| s.to_str()) {
        project.set_name(stem);
    }
    Ok(Import { project, warnings, output_warnings })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Encodes an `.ebs` file (for tests).
    pub fn encode(e: &EbsFile) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        let s = |out: &mut Vec<u8>, t: &str| {
            out.extend((t.len() as i32).to_le_bytes());
            out.extend(t.as_bytes());
        };
        s(&mut out, &e.video);
        s(&mut out, &e.mask);
        s(&mut out, &e.keys);
        out.push(u8::from(e.mask_enabled));
        for w in [e.weight_keyframe, e.weight_video, e.weight_mask, e.weight_mapping, e.weight_deflicker, e.diversity] {
            out.extend(w.to_le_bytes());
        }
        out.extend((e.keyframes.len() as i32).to_le_bytes());
        for k in &e.keyframes {
            out.extend(k.frame.to_le_bytes());
            out.push(u8::from(k.backward_stop.is_some()));
            out.push(u8::from(k.forward_stop.is_some()));
            out.extend(k.backward_stop.unwrap_or(1).to_le_bytes());
            out.extend(k.forward_stop.unwrap_or(1).to_le_bytes());
            s(&mut out, &k.output);
        }
        if let Some(t) = &e.trailer {
            out.extend(TRAILER);
            out.extend(t.synthesis_detail.to_le_bytes());
            out.push(u8::from(t.use_gpu));
            out.extend(t.ae_fps.to_le_bytes());
            out.extend(t.window_width.to_le_bytes());
        }
        out
    }

    /// A two-keyframe project like the ones EbSynth lays out automatically.
    pub fn sample() -> EbsFile {
        EbsFile {
            video: "video\\[###].jpg".into(),
            mask: "mask\\[###].png".into(),
            keys: "keys\\[###].jpg".into(),
            mask_enabled: false,
            weight_keyframe: 1.0,
            weight_video: 4.0,
            weight_mask: 1.0,
            weight_mapping: 10.0,
            weight_deflicker: 1.0,
            diversity: 3500.0,
            keyframes: vec![
                EbsKeyframe {
                    frame: 0,
                    backward_stop: None,
                    forward_stop: Some(99),
                    output: "out-0\\[###].png".into(),
                },
                EbsKeyframe {
                    frame: 99,
                    backward_stop: Some(0),
                    forward_stop: None,
                    output: "out-99\\[###].png".into(),
                },
            ],
            trailer: None,
        }
    }

    #[test]
    fn parses_what_it_encodes() {
        let e = sample();
        let bytes = encode(&e);
        assert_eq!(bytes.len(), 157, "the classic two-keyframe file without trailer");
        assert_eq!(parse_ebs(&bytes).unwrap(), e);
        let mut with_trailer = e.clone();
        with_trailer.trailer =
            Some(EbsTrailer { synthesis_detail: 1, use_gpu: false, ae_fps: 24.0, window_width: 704 });
        let bytes = encode(&with_trailer);
        assert_eq!(bytes.len(), 174);
        assert_eq!(parse_ebs(&bytes).unwrap(), with_trailer);
        // Trailing bytes that are not a trailer are ignored.
        let mut junk = encode(&e);
        junk.extend(b"XYZ!1234");
        assert_eq!(parse_ebs(&junk).unwrap().trailer, None);
    }

    #[test]
    fn rejects_corrupt_files() {
        let good = encode(&sample());
        assert!(parse_ebs(b"EBS04\0").unwrap_err().to_string().contains("EBS05"));
        let err = parse_ebs(&good[..100]).unwrap_err().to_string();
        assert!(err.contains("ends inside"), "{err}");
        let mut negative = good.clone();
        negative[6..10].copy_from_slice(&(-5i32).to_le_bytes());
        assert!(parse_ebs(&negative).unwrap_err().to_string().contains("invalid length (-5)"));
        let mut nan = good.clone();
        let w = 6 + 4 + 15 + 4 + 14 + 4 + 14 + 1;
        nan[w..w + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(parse_ebs(&nan).is_err());
    }

    #[test]
    fn patterns_convert() {
        assert_eq!(convert_pattern("video\\[###].jpg"), ("video/[###].jpg".into(), None));
        assert_eq!(convert_pattern("v\\%04d.png").0, "v/[####].png");
        assert_eq!(convert_pattern("v/%d.png").0, "v/[#].png");
        assert_eq!(convert_pattern("100%%\\[##].png").0, "100%/[##].png");
        let (kept, warning) = convert_pattern("v/%s_[##].png");
        assert_eq!(kept, "v/%s_[##].png");
        assert!(warning.is_some());
        assert_eq!(number_width("v/[###].jpg"), 3);
        assert_eq!(number_width("v/frame.jpg"), 5);
    }

    #[test]
    fn imports_into_a_project() {
        let mut e = sample();
        e.trailer = Some(EbsTrailer { synthesis_detail: 3, use_gpu: false, ae_fps: 24.0, window_width: 704 });
        e.weight_mapping = 20.0;
        e.mask_enabled = true;
        let import =
            import_parsed(&e, Path::new("/shots/fox"), Path::new("/shots"), Path::new("/shots/fox.fsyn")).unwrap();
        let p = &import.project;
        assert_eq!(p.media.frames, "fox/video/[###].jpg");
        assert_eq!(p.media.mask.as_deref(), Some("fox/mask/[###].png"));
        assert_eq!((p.media.first, p.media.last, p.media.fps), (Some(0), Some(99), Some(24.0)));
        let keys: Vec<_> = p.keyframes.iter().map(|k| (k.frame, k.image.as_str())).collect();
        assert_eq!(keys, [(0, "fox/keys/000.jpg"), (99, "fox/keys/099.jpg")]);
        assert_eq!((p.guides.mapping, p.guides.temporal), (20.0, 1.0));
        assert_eq!((p.engine.preset, p.engine.backend), (Preset::Fast, Backend::Cpu));
        assert_eq!(p.output.frames, "fox-out/[###].png");
        assert!(import.warnings.is_empty(), "{:?}", import.warnings);
        assert_eq!(import.output_warnings.len(), 1, "{:?}", import.output_warnings);
        p.validate().unwrap();
        // The project survives a save and load.
        let back = Project::from_toml(&p.to_toml(), "/shots").unwrap();
        assert_eq!(back.keyframes, p.keyframes);

        // Ranges that leave a gap between neighbours are reported.
        let mut gap = sample();
        gap.keyframes[0].forward_stop = Some(40);
        gap.keyframes[1].backward_stop = Some(60);
        gap.keyframes[0].output = "out\\[###].png".into();
        gap.keyframes[1].output = "out\\[###].png".into();
        let import = import_parsed(&gap, Path::new("p"), Path::new("p"), Path::new("p/x.fsyn")).unwrap();
        assert!(import.warnings.iter().any(|w| w.contains("did not both cover")), "{:?}", import.warnings);
        assert_eq!(import.project.output.frames, "out/[###].png", "a shared output pattern is kept");
        // Patterns outside the .ebs folder are reported; such an output is replaced.
        let mut far = sample();
        far.video = "..\\..\\elsewhere\\[###].jpg".into();
        for k in &mut far.keyframes {
            k.output = "C:\\renders\\[###].png".into();
        }
        let import = import_parsed(&far, Path::new("/p/q"), Path::new("/p/q"), Path::new("/p/q/x.fsyn")).unwrap();
        assert!(import.warnings.iter().any(|w| w.contains("leaves the .ebs file's folder")), "{:?}", import.warnings);
        assert_eq!(import.project.media.frames, "../../elsewhere/[###].jpg");
        assert_eq!(import.project.output.frames, "x-out/[###].png");
        assert!(import.output_warnings[0].contains("drive letter"), "{:?}", import.output_warnings);
        for k in &mut far.keyframes {
            k.output = "/tmp/[###].png".into();
        }
        let import = import_parsed(&far, Path::new("/p/q"), Path::new("/p/q"), Path::new("/p/q/x.fsyn")).unwrap();
        assert_eq!(import.project.output.frames, "x-out/[###].png");
        assert!(import.output_warnings[0].contains("absolute path"));
        assert_eq!(outside_reason("a/../b/[#].png"), None);
        assert_eq!(outside_reason("a/../../b.png"), Some("leaves the .ebs file's folder"));
        // A keyframe whose stops exclude it is an error.
        let mut bad = sample();
        bad.keyframes[1].backward_stop = Some(120);
        assert!(import_parsed(&bad, Path::new("p"), Path::new("p"), Path::new("p/x.fsyn")).is_err());
    }
}
