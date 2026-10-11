#!/usr/bin/env bash
# Downloads a few short, openly licensed clips for the FreeSynth test corpus.
#
# Each clip is a short excerpt of a Blender Foundation open movie (Creative Commons
# Attribution). The script downloads the source file, checks its SHA-256, extracts the
# excerpt as PNG frames with ffmpeg and deletes the source file again. The result goes to
# corpus/external/ (ignored by git; CC BY material is never committed):
#
#   corpus/external/<clip>/frames/00000.png ...   the excerpt
#   corpus/external/<clip>/scene.toml             size, frame count, licence, attribution
#   corpus/external/ATTRIBUTION.md                credits for every fetched clip
#
# The clips have no keyframe styles and no flow; paint keys into <clip>/keys/NNNNN.png and
# compute flow with an independent estimator before using them for temporal metrics.
#
# Usage: corpus/fetch-clips.sh [--only <clip>]... [--keep] [--list] [--out <dir>]
#   --only <clip>   fetch only this clip (repeatable)
#   --keep          keep the downloaded source files in <out>/.downloads
#   --list          print the clip table and exit
#   --out <dir>     output directory (default: corpus/external next to this script)
#
# Needs curl, ffmpeg, ffprobe and sha256sum (or shasum). Total download: about 160 MB.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/external"
keep=0
only=()

# Fields, separated by ';': name, source URL, SHA-256, start (s), frames, ffmpeg filter,
# input colour matrix, licence (SPDX), title, credit line
clips=(
  "sintel-alley;https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4;cb0fe73fc0a7d543459996c0cdab730997e6eac1013d3ede18796f777cb7f273;18.25;42;crop=1280:544:0:88,scale=960:408;bt709;CC-BY-3.0;Sintel (trailer);(c) copyright Blender Foundation | durian.blender.org"
  "ed-walk;https://download.blender.org/ED/elephantsdream-720-h264-st-aac.mov;6fc295e0e92835316ec95929ccfc664b05ed2434f673e6f5e684f98f7558a547;38.0;100;crop=720:404:0:0;bt601;CC-BY-2.5;Elephants Dream;(c) copyright 2006, Blender Foundation / Netherlands Media Art Institute / www.elephantsdream.org"
  "ed-faces;https://download.blender.org/ED/elephantsdream-720-h264-st-aac.mov;6fc295e0e92835316ec95929ccfc664b05ed2434f673e6f5e684f98f7558a547;295.0;100;crop=720:404:0:0;bt601;CC-BY-2.5;Elephants Dream;(c) copyright 2006, Blender Foundation / Netherlands Media Art Institute / www.elephantsdream.org"
  "ed-cables;https://download.blender.org/ED/elephantsdream-720-h264-st-aac.mov;6fc295e0e92835316ec95929ccfc664b05ed2434f673e6f5e684f98f7558a547;546.0;100;crop=720:404:0:0;bt601;CC-BY-2.5;Elephants Dream;(c) copyright 2006, Blender Foundation / Netherlands Media Art Institute / www.elephantsdream.org"
)

licence_url() {
  case "$1" in
    CC-BY-2.5) echo "https://creativecommons.org/licenses/by/2.5/" ;;
    CC-BY-3.0) echo "https://creativecommons.org/licenses/by/3.0/" ;;
    *) echo "unknown licence $1" >&2; exit 1 ;;
  esac
}

die() { echo "fetch-clips: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) [[ $# -ge 2 ]] || die "--only needs a clip name"; only+=("$2"); shift 2 ;;
    --keep) keep=1; shift ;;
    --out) [[ $# -ge 2 ]] || die "--out needs a directory"; out="$2"; shift 2 ;;
    --list)
      for c in "${clips[@]}"; do
        IFS=';' read -r name url _ start frames _ _ licence title _ <<<"$c"
        printf '%-14s %3s frames from %6s s of %s (%s)\n' "$name" "$frames" "$start" "$title" "$licence"
      done
      exit 0 ;;
    -h|--help) sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;;
    *) die "unknown argument $1 (try --help)" ;;
  esac
done

for tool in curl ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || die "$tool is required"
done
if command -v sha256sum >/dev/null; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null; then
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  die "sha256sum or shasum is required"
fi

for name in "${only[@]+"${only[@]}"}"; do
  printf '%s\n' "${clips[@]}" | grep -q "^$name;" || die "no clip named $name (see --list)"
done

selected() {
  [[ ${#only[@]} -eq 0 ]] && return 0
  local n
  for n in "${only[@]}"; do [[ "$n" == "$1" ]] && return 0; done
  return 1
}

downloads="$out/.downloads"
mkdir -p "$downloads"

# Downloads $1 to $2 unless a file with the right checksum $3 is already there.
fetch() {
  local url="$1" dest="$2" sum="$3"
  if [[ -f "$dest" && "$(sha256 "$dest")" == "$sum" ]]; then
    return
  fi
  echo "downloading $url"
  curl -fL --retry 3 --progress-bar --proto '=https' -o "$dest.part" "$url"
  local got
  got="$(sha256 "$dest.part")"
  if [[ "$got" != "$sum" ]]; then
    rm -f "$dest.part"
    die "checksum mismatch for $url (expected $sum, got $got)"
  fi
  mv "$dest.part" "$dest"
}

fetched=()
for c in "${clips[@]}"; do
  IFS=';' read -r name url sum start frames filter matrix licence title credit <<<"$c"
  selected "$name" || continue
  src="$downloads/$(basename "$url")"
  fetch "$url" "$src" "$sum"

  dir="$out/$name"
  rm -rf "$dir"
  mkdir -p "$dir/frames"
  echo "extracting $name: $frames frames from ${start}s"
  ffmpeg -nostdin -hide_banner -loglevel error -protocol_whitelist file \
    -ss "$start" -i "$src" -map 0:v:0 -an -frames:v "$frames" \
    -vf "$filter,scale=in_color_matrix=$matrix:in_range=tv:out_range=pc,format=rgb24" \
    -start_number 0 "$dir/frames/%05d.png"
  count=$(find "$dir/frames" -name '*.png' | wc -l)
  [[ "$count" -eq "$frames" ]] || die "$name: expected $frames frames, got $count"
  size=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$dir/frames/00000.png")
  width="${size%,*}"
  height="${size#*,}"

  cat >"$dir/scene.toml" <<EOF
# FreeSynth scene description (fetched clip). Frames: frames/NNNNN.png.
format_version = 1
name = "$name"
width = $width
height = $height
frames = $frames
keyframes = []
guide = "frames"
family = "footage"
licence = "$licence"
attribution = "$title, $credit. Licensed under $licence ($(licence_url "$licence")). Excerpt of $frames frames starting at ${start} s, ffmpeg filter $filter, converted to PNG."
source = "$url"
flow_gaps = []
occlusion_masks = false
ground_truth = false
EOF
  fetched+=("$name")
done

# Attribution for every clip present in the output directory.
{
  echo "# Attribution for fetched clips"
  echo
  echo "These frames are excerpts of films released under Creative Commons Attribution licences."
  echo "They were fetched by corpus/fetch-clips.sh and are not part of the FreeSynth repository."
  echo
  for c in "${clips[@]}"; do
    IFS=';' read -r name url _ start frames filter _ licence title credit <<<"$c"
    [[ -f "$out/$name/scene.toml" ]] || continue
    echo "## $name"
    echo
    echo "- Work: $title, $credit"
    echo "- Licence: $licence, $(licence_url "$licence")"
    echo "- Source: $url"
    echo "- Changes: excerpt of $frames frames starting at ${start} s; $filter; converted to PNG."
    echo
  done
} >"$out/ATTRIBUTION.md"

if [[ $keep -eq 0 ]]; then
  rm -rf "$downloads"
fi
echo "fetched ${#fetched[@]} clip(s) into $out"
