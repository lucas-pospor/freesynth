# FreeSynth

FreeSynth is a free and open-source reimplementation of [EbSynth](https://github.com/jamriska/ebsynth).
It does example-based image synthesis and stylizes video from keyframes. The GPU code isn't tied
to one GPU vendor.

The project is in early development.

```bash
cargo build --release
```

License: GPL-3.0-or-later.

## Patents

FreeSynth's patch search uses the PatchMatch algorithm, which in the United States is covered by
Adobe patents, including US 8,861,869. Some of the other techniques it uses may also be patented.

The GPL-3.0 license covers copyright. It does not grant you a license to patents held by third
parties such as Adobe, and building or distributing FreeSynth does not clear you of patent
obligations. Checking that your use does not infringe anyone's patents is your own responsibility.
This note is not legal advice; if you intend to ship FreeSynth in a product or use it in the United
States, talk to a patent attorney.

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
`-search improved|compat` (GPU PatchMatch variant: checkerboard propagation in every preset except
compat, or the 2019 CUDA search), `-modulation <map.png>` (per-pixel weight map for the preceding
`-guide`), `-seed <n>`,
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

## Video

`freesynth video` stylizes a clip from painted keyframes in one go:

```bash
target/release/freesynth video -i clip.mov --key 0=paint_000.png --key 99=paint_099.png -o out.mov
```

Each keyframe is propagated forward and backward up to its neighbouring keyframes. The guides come
from the input frames, from position fields built with `freesynth-flow`, and from the previous
output frame; the positional guide is weighted by how reliably each pixel was tracked. Where two
keyframes overlap, the passes are blended: the broad shading is crossfaded, and the finer detail is
taken per pixel from the pass with the lower patch error, so soft shadows stay soft. `--blend
adaptive` selects the detail at every scale instead (sharper), and `--blend linear` crossfades the
whole frame. Video files are read and written
through ffmpeg, which must be installed, and their colour tags are kept. `--bit-depth 16` writes
16-bit output.

For longer work, keep the settings in a project file:

```bash
target/release/freesynth new -i clip.mov --keys paintings/ project.fsyn
target/release/freesynth render project.fsyn
```

`render` only re-renders frames whose inputs changed, and `render --watch` re-renders as soon as a
keyframe painting is saved. `freesynth import-ebs` converts EbSynth `.ebs` projects, and
`freesynth devices` lists the GPUs and the ffmpeg in use. The `freesynth` crate offers the same
projects and render sessions as a library.
