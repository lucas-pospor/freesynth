//! `freesynth`: command-line front end compatible with the 2019 `ebsynth` tool.
//!
//! Accepts the same options, defaults and error messages as the original (`ebsynth.cpp`), plus a
//! few FreeSynth extensions listed in the usage text.

mod backend;

use std::process::ExitCode;

use freesynth_core::{
    GuideLayer, Image, MAX_GUIDE_CHANNELS, MAX_STYLE_CHANNELS, OmegaSync, Params, PerLevel, SynthesisInput, io,
    max_pyramid_levels,
};

use crate::backend::Backend;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    ExitCode::from(run(&args))
}

fn print_usage(program: &str) {
    println!("usage: {program} [options]");
    println!();
    println!("options:");
    println!("  -style <style.png>");
    println!("  -guide <source.png> <target.png>");
    println!("  -output <output.png>");
    println!("  -weight <value>");
    println!("  -uniformity <value>");
    println!("  -patchsize <size>");
    println!("  -pyramidlevels <number>");
    println!("  -searchvoteiters <number>");
    println!("  -patchmatchiters <number>");
    println!("  -stopthreshold <value>");
    println!("  -extrapass3x3");
    println!("  -backend [cpu|gpu]");
    println!();
    println!("freesynth extensions:");
    println!("  -seed <number>        random seed (default 1)");
    println!("  -threads <number>     worker threads (default: all cores)");
    println!("  -bands <number>       PatchMatch bands (default: one per thread)");
    println!("  -deterministic        reproducible results for a given seed and band count");
    println!("  -bitexact             match the 2019 CPU build exactly (use with -threads 1)");
    println!("  -dumpnnf <file.nnf>   also write the final nearest-neighbour field (FSNNF1 format)");
    println!();
}

/// Command-line state, mirroring the locals of the original `main`.
struct Options {
    style_file: String,
    style_weight: f32,
    output_file: String,
    guides: Vec<GuideArg>,
    uniformity: f32,
    patch_size: i32,
    pyramid_levels: i32,
    search_vote_iters: i32,
    patch_match_iters: i32,
    stop_threshold: i32,
    extra_pass_3x3: bool,
    backend: Backend,
    seed: i32,
    threads: i32,
    bands: i32,
    deterministic: bool,
    bit_exact: bool,
    /// Where to write the final nearest-neighbour field, if anywhere.
    dump_nnf: Option<String>,
}

struct GuideArg {
    source_file: String,
    target_file: String,
    /// Negative means "not given".
    weight: f32,
}

/// Which weight a following `-weight` applies to.
#[derive(Clone, Copy)]
enum WeightTarget {
    None,
    Style,
    Guide(usize),
}

/// Outcome of trying to parse one option.
enum Parsed<T> {
    /// The current argument is a different option.
    NoMatch,
    /// Matched; the value is valid.
    Value(T),
    /// Matched, but the value is missing or malformed (an error was printed).
    Fail,
}

/// Argument cursor with the original's `tryToParse*Arg` helpers.
struct Parser<'a> {
    args: &'a [String],
    i: usize,
}

impl<'a> Parser<'a> {
    fn value(&mut self, name: &str) -> Option<&'a str> {
        let v = self.args.get(self.i).map(String::as_str);
        if v.is_none() {
            println!("error: missing argument for the {name} option");
        }
        v
    }

    fn string(&mut self, name: &str) -> Parsed<String> {
        if self.args[self.i] != name {
            return Parsed::NoMatch;
        }
        self.i += 1;
        match self.value(name) {
            Some(v) => {
                self.i += 1;
                Parsed::Value(v.to_owned())
            }
            None => Parsed::Fail,
        }
    }

    fn string_pair(&mut self, name: &str) -> Parsed<(String, String)> {
        if self.args[self.i] != name {
            return Parsed::NoMatch;
        }
        self.i += 1;
        if self.i + 1 < self.args.len() {
            let pair = (self.args[self.i].clone(), self.args[self.i + 1].clone());
            self.i += 2;
            Parsed::Value(pair)
        } else {
            println!("error: missing argument for the {name} option");
            Parsed::Fail
        }
    }

    /// Like `std::stoi`/`std::stof` with a full-match check: leading whitespace is skipped and
    /// the whole remaining string must be a number.
    fn number<T: std::str::FromStr>(&mut self, name: &str) -> Parsed<T> {
        if self.args[self.i] != name {
            return Parsed::NoMatch;
        }
        self.i += 1;
        let Some(v) = self.value(name) else { return Parsed::Fail };
        match v.trim_start().parse::<T>() {
            Ok(x) => {
                self.i += 1;
                Parsed::Value(x)
            }
            Err(_) => {
                println!("error: bad {name} argument '{v}'");
                Parsed::Fail
            }
        }
    }

    fn flag(&mut self, name: &str) -> bool {
        if self.args[self.i] == name {
            self.i += 1;
            true
        } else {
            false
        }
    }
}

/// Parses the command line. `Err(code)` means exit with that code (an error was printed).
fn parse_args(args: &[String]) -> Result<Options, u8> {
    let mut o = Options {
        style_file: String::new(),
        style_weight: -1.0,
        output_file: "output.png".into(),
        guides: Vec::new(),
        uniformity: 3500.0,
        patch_size: 5,
        pyramid_levels: -1,
        search_vote_iters: 6,
        patch_match_iters: 4,
        stop_threshold: 5,
        extra_pass_3x3: false,
        backend: Backend::default_backend(),
        seed: 1,
        threads: 0,
        bands: 0,
        deterministic: false,
        bit_exact: false,
        dump_nnf: None,
    };
    let mut weight_target = WeightTarget::None;
    let mut p = Parser { args, i: 1 };

    // Each branch either matches the current option or falls through to the next one.
    macro_rules! try_opt {
        ($parsed:expr, |$v:ident| $body:block) => {
            match $parsed {
                Parsed::Value($v) => {
                    $body;
                    continue;
                }
                Parsed::Fail => return Err(1),
                Parsed::NoMatch => {}
            }
        };
    }

    while p.i < args.len() {
        try_opt!(p.string("-style"), |v| {
            o.style_file = v;
            o.style_weight = -1.0;
            weight_target = WeightTarget::Style;
        });
        try_opt!(p.string_pair("-guide"), |v| {
            o.guides.push(GuideArg { source_file: v.0, target_file: v.1, weight: -1.0 });
            weight_target = WeightTarget::Guide(o.guides.len() - 1);
        });
        try_opt!(p.string("-output"), |v| {
            o.output_file = v;
        });
        try_opt!(p.number::<f32>("-weight"), |w| {
            match weight_target {
                WeightTarget::None => {
                    println!("error: at least one -style or -guide option must precede the -weight option!");
                    return Err(1);
                }
                _ if w.is_nan() || w < 0.0 => {
                    println!("error: weights must be non-negaitve!");
                    return Err(1);
                }
                WeightTarget::Style => o.style_weight = w,
                WeightTarget::Guide(g) => o.guides[g].weight = w,
            }
        });
        try_opt!(p.number::<f32>("-uniformity"), |v| {
            o.uniformity = v;
        });
        try_opt!(p.number::<i32>("-patchsize"), |v| {
            if v < 3 {
                println!("error: patchsize is too small!");
                return Err(1);
            }
            if v % 2 == 0 {
                println!("error: patchsize must be an odd number!");
                return Err(1);
            }
            o.patch_size = v;
        });
        try_opt!(p.number::<i32>("-pyramidlevels"), |v| {
            if v < 1 {
                println!("error: bad argument for -pyramidlevels!");
                return Err(1);
            }
            o.pyramid_levels = v;
        });
        try_opt!(p.number::<i32>("-searchvoteiters"), |v| {
            if v < 0 {
                println!("error: bad argument for -searchvoteiters!");
                return Err(1);
            }
            o.search_vote_iters = v;
        });
        try_opt!(p.number::<i32>("-patchmatchiters"), |v| {
            if v < 0 {
                println!("error: bad argument for -patchmatchiters!");
                return Err(1);
            }
            o.patch_match_iters = v;
        });
        try_opt!(p.number::<i32>("-stopthreshold"), |v| {
            if v < 0 {
                println!("error: bad argument for -stopthreshold!");
                return Err(1);
            }
            o.stop_threshold = v;
        });
        try_opt!(p.string("-backend"), |name| {
            let Some(b) = Backend::parse(&name) else {
                println!("error: unrecognized backend '{name}'");
                return Err(1);
            };
            if !b.is_available() {
                println!("error: the {} backend is not available!", b.name());
                return Err(1);
            }
            o.backend = b;
        });
        if p.flag("-extrapass3x3") {
            o.extra_pass_3x3 = true;
            continue;
        }
        // FreeSynth extensions.
        try_opt!(p.number::<i32>("-seed"), |v| {
            o.seed = v;
        });
        try_opt!(p.number::<i32>("-threads"), |v| {
            if v < 1 {
                println!("error: bad argument for -threads!");
                return Err(1);
            }
            o.threads = v;
        });
        try_opt!(p.number::<i32>("-bands"), |v| {
            if v < 1 {
                println!("error: bad argument for -bands!");
                return Err(1);
            }
            o.bands = v;
        });
        if p.flag("-deterministic") {
            o.deterministic = true;
            continue;
        }
        if p.flag("-bitexact") {
            o.bit_exact = true;
            continue;
        }
        try_opt!(p.string("-dumpnnf"), |v| {
            o.dump_nnf = Some(v);
        });
        println!("error: unrecognized option '{}'", args[p.i]);
        return Err(1);
    }
    Ok(o)
}

fn load(file: &str) -> Result<Image, u8> {
    io::load_rgba(file).map_err(|e| {
        println!("error: failed to load '{file}'");
        println!("{e}");
        1
    })
}

fn run(args: &[String]) -> u8 {
    if args.len() < 2 {
        print_usage(args.first().map_or("freesynth", String::as_str));
        return 1;
    }
    match run_inner(args) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

fn run_inner(args: &[String]) -> Result<(), u8> {
    let o = parse_args(args)?;

    let style = load(&o.style_file)?;
    let num_style_channels = io::eval_num_channels(&style);

    let mut guides = Vec::with_capacity(o.guides.len());
    let mut num_guide_channels = 0;
    for (i, g) in o.guides.iter().enumerate() {
        let source = load(&g.source_file)?;
        let target = load(&g.target_file)?;
        if source.size() != style.size() {
            println!("error: source guide '{}' doesn't match the resolution of '{}'", g.source_file, o.style_file);
            return Err(1);
        }
        if i > 0 && target.size() != guides.first().map(|g: &GuideLayer| g.target.size()).unwrap_or_default() {
            println!("error: target guide '{}' doesn't match the resolution of '{}'", g.target_file, o.guides[0].target_file);
            return Err(1);
        }
        num_guide_channels += io::eval_num_channels(&source).max(io::eval_num_channels(&target));
        let mut layer = GuideLayer::new(source, target);
        if g.weight >= 0.0 {
            layer = layer.with_weight(g.weight);
        }
        guides.push(layer);
    }

    if num_style_channels > MAX_STYLE_CHANNELS {
        println!("error: too many style channels ({num_style_channels}), maximum number is {MAX_STYLE_CHANNELS}");
        return Err(1);
    }
    if num_guide_channels > MAX_GUIDE_CHANNELS {
        println!("error: too many guide channels ({num_guide_channels}), maximum number is {MAX_GUIDE_CHANNELS}");
        return Err(1);
    }
    if guides.is_empty() {
        println!("error: at least one -guide option is required!");
        return Err(1);
    }

    let style_weight = (o.style_weight >= 0.0).then_some(o.style_weight);
    let input = SynthesisInput::from_rgba_layers(&style, style_weight, &guides).map_err(|e| {
        println!("error: {e}");
        1
    })?;

    let patch_size = o.patch_size as usize;
    let max_levels = max_pyramid_levels(input.source_style.size(), input.target_guide.size(), patch_size);
    if max_levels == 0 {
        println!("error: the images are too small for patchsize {patch_size}!");
        return Err(1);
    }
    let num_levels = if o.pyramid_levels == -1 { max_levels } else { (o.pyramid_levels as usize).min(max_levels) };

    let params = Params {
        uniformity_weight: o.uniformity,
        patch_size,
        num_pyramid_levels: Some(num_levels),
        search_vote_iters: PerLevel::Uniform(o.search_vote_iters as usize),
        patch_match_iters: PerLevel::Uniform(o.patch_match_iters as usize),
        stop_threshold: PerLevel::Uniform(o.stop_threshold as u32),
        extra_pass_3x3: o.extra_pass_3x3,
        seed: o.seed as u32,
        num_threads: (o.threads > 0).then_some(o.threads as usize),
        num_bands: (o.bands > 0).then_some(o.bands as usize),
        omega_sync: if o.deterministic { OmegaSync::PerTile } else { OmegaSync::Shared },
        bit_exact: o.bit_exact,
        ..Params::default()
    };

    println!("uniformity: {:.0}", o.uniformity);
    println!("patchsize: {}", o.patch_size);
    println!("pyramidlevels: {num_levels}");
    println!("searchvoteiters: {}", o.search_vote_iters);
    println!("patchmatchiters: {}", o.patch_match_iters);
    println!("stopthreshold: {}", o.stop_threshold);
    println!("extrapass3x3: {}", if o.extra_pass_3x3 { "yes" } else { "no" });
    println!("backend: {}", o.backend.name());

    let output = o.backend.run(&input, &params).map_err(|e| {
        println!("error: {e}");
        1
    })?;

    if let Err(e) = io::save_png(&output.image, &o.output_file) {
        println!("error: failed to write '{}'", o.output_file);
        println!("{e}");
        return Err(1);
    }
    println!("result was written to {}", o.output_file);
    if let Some(path) = &o.dump_nnf {
        if let Err(e) = io::save_nnf(&output.nnf, path) {
            println!("error: failed to write '{path}'");
            println!("{e}");
            return Err(1);
        }
        println!("nnf was written to {path}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        std::iter::once("freesynth").chain(s.split_whitespace()).map(String::from).collect()
    }

    #[test]
    fn defaults_match_ebsynth() {
        let o = parse_args(&args("-style a.png -guide b.png c.png")).ok().unwrap();
        assert_eq!(o.uniformity, 3500.0);
        assert_eq!(o.patch_size, 5);
        assert_eq!(o.pyramid_levels, -1);
        assert_eq!(o.search_vote_iters, 6);
        assert_eq!(o.patch_match_iters, 4);
        assert_eq!(o.stop_threshold, 5);
        assert!(!o.extra_pass_3x3);
        assert_eq!(o.output_file, "output.png");
        assert_eq!(o.style_weight, -1.0);
        assert_eq!(o.guides[0].weight, -1.0);
        assert_eq!(o.backend, Backend::Cpu);
        assert_eq!(o.dump_nnf, None);
    }

    #[test]
    fn dump_nnf_takes_a_path() {
        let o = parse_args(&args("-style s.png -guide a b -dumpnnf out.nnf")).ok().unwrap();
        assert_eq!(o.dump_nnf.as_deref(), Some("out.nnf"));
        assert!(parse_args(&args("-style s.png -dumpnnf")).is_err());
    }

    #[test]
    fn weights_apply_to_preceding_layer() {
        let o = parse_args(&args("-style s.png -weight 2 -guide a b -guide c d -weight 0.5 -output o.png")).ok().unwrap();
        assert_eq!(o.style_weight, 2.0);
        assert_eq!(o.guides[0].weight, -1.0);
        assert_eq!(o.guides[1].weight, 0.5);
        assert_eq!(o.output_file, "o.png");
    }

    #[test]
    fn rejects_bad_arguments() {
        for bad in [
            "-weight 1 -style s",
            "-style s -weight -1",
            "-patchsize 4",
            "-patchsize 1",
            "-patchsize 5x",
            "-pyramidlevels 0",
            "-searchvoteiters -1",
            "-stopthreshold -2",
            "-backend metal",
            "-guide a",
            "-style",
            "-frobnicate",
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad}");
        }
        let o = parse_args(&args("-patchsize 7 -uniformity 1e3 -extrapass3x3 -backend gpu")).ok().unwrap();
        assert_eq!((o.patch_size, o.uniformity, o.extra_pass_3x3, o.backend), (7, 1000.0, true, Backend::Gpu));
    }
}
