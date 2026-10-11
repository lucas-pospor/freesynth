# FreeSynth

FreeSynth is a free and open-source reimplementation of [EbSynth](https://github.com/jamriska/ebsynth).
It does example-based image synthesis and stylizes video from keyframes. The GPU code isn't tied
to one GPU vendor.

The project is in early development.

```bash
cargo build --release
```

License: GPL-3.0-or-later.

## Command line

`freesynth` accepts the same options and defaults as the 2019 `ebsynth` tool:

```bash
target/release/freesynth -style source_style.png \
    -guide source_guide.png target_guide.png -weight 2 \
    -output output.png
```

Options: `-style`, `-guide <src> <tgt>`, `-weight` (applies to the preceding `-style`/`-guide`),
`-output`, `-uniformity` (3500), `-patchsize` (5), `-pyramidlevels` (max), `-searchvoteiters` (6),
`-patchmatchiters` (4), `-stopthreshold` (5; accepted but not applied, as in the 2019 CPU
engine), `-extrapass3x3`, `-backend cpu|gpu`.

Extra options: `-seed <n>`, `-threads <n>`, `-bands <n>` (number of PatchMatch bands, one per
thread by default), `-deterministic` (same output for a given seed and band count), `-bitexact`
(with `-threads 1`, reproduces the 2019 CPU build's arithmetic; slower).

The output is always written as PNG, whatever the file name.

The library entry point is `freesynth_core::synthesize` (see `crates/core/src/lib.rs`).
