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

`freesynth` accepts the options of the 2019 `ebsynth` tool:

```bash
target/release/freesynth -style source_style.png \
    -guide source_guide.png target_guide.png -weight 2 \
    -output output.png
```

Options: `-style`, `-guide <src> <tgt>`, `-weight` (applies to the preceding `-style`/`-guide`),
`-output`, `-uniformity` (3500), `-patchsize` (5), `-pyramidlevels` (max), `-searchvoteiters` (6),
`-patchmatchiters`, `-stopthreshold` (5), `-extrapass3x3`, `-backend cpu|gpu`.

By default `freesynth` uses its own `default` preset: 2 PatchMatch iterations instead of 4 (4 for
patch size 3), weighted voting and rounded votes. On our test images it is about 1.8 times faster
than the 2019 schedule on both backends, with patch energy within about 1%. `-preset fast` trades
a little quality for more speed, `-preset high` runs the full 2019 schedule with weighted voting,
and `-compat` reproduces the 2019 tool. Options given explicitly override the preset.

Other options: `-preset`, `-compat`, `-vote plain|weighted`, `-stopmask on|off`,
`-modulation <map.png>` (per-pixel weight map for the preceding `-guide`), `-seed <n>`,
`-threads <n>`, `-bands <n>`, `-deterministic` (same output for a given seed and band count),
`-bitexact` (with `-threads 1`, reproduces the 2019 CPU build's arithmetic; implies `-compat`),
`-dumpnnf <file>` and `-dumperror <file.pfm>`.

The output is always written as PNG, whatever the file name.

The library entry point is `freesynth_core::synthesize` (see `crates/core/src/lib.rs`).

## Evaluation tools

`freesynth-eval` measures synthesis quality. It computes patch energy from a saved
nearest-neighbour field (`freesynth -dumpnnf`) or exactly on sampled pixels, how much of the
source gets used, and sharpness. For video it measures temporal warping error against an external
optical flow. `freesynth-eval compare` compares runs in pairs with bootstrap and t intervals.

`freesynth-scenes` generates CC0 test sequences with exact optical flow and ground-truth
stylizations. `corpus/manifest.toml` lists the scenes, and `corpus/fetch-clips.sh` downloads a few
Creative Commons clips together with their attribution.

```bash
cargo run --release -p freesynth-scenes -- corpus corpus/manifest.toml corpus/generated --tier ci
cargo test -p freesynth-eval --test ci_quick
```

## Optical flow

`freesynth-flow` computes dense optical flow with DIS (Kroeger et al., 2016) on the GPU through
wgpu or on the CPU, with matching results on both. At 960x540 a flow takes about 0.9 ms of GPU time
on a Radeon RX 9060 XT and about 9 ms on 12 CPU threads. The crate also builds position fields: it
tracks pixels from a keyframe by chaining flow with periodic re-anchoring, and keeps a per-pixel
confidence map from forward-backward consistency.
