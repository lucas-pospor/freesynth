//! Round trips through the system ffmpeg. Every test is skipped when ffmpeg is not installed.

use std::path::{Path, PathBuf};

use freesynth_core::Image;
use freesynth_media::{
    BitDepth, Codec, ColourTags, EncodeSettings, Ffmpeg, MediaError, Rational, VideoDecoder, VideoEncoder,
    encode_files, probe, save_png,
};

fn ffmpeg() -> Option<Ffmpeg> {
    match Ffmpeg::locate() {
        Ok(ff) => Some(ff),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("freesynth-media-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A smooth colour gradient that moves with `t` (smooth, so 4:2:2 chroma loses little).
fn frame(w: usize, h: usize, t: usize) -> Image {
    let data = (0..w * h).flat_map(|i| {
        let (x, y) = (i % w, i / w);
        [(40 + (x + 2 * t) * 150 / w) as u8, (60 + y * 120 / h) as u8, (200 - (x + y) * 80 / (w + h)) as u8, 255]
    });
    Image::from_raw(w, h, 4, data.collect())
}

fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    let (a, b) = (freesynth_media::to_rgba8(a), freesynth_media::to_rgba8(b));
    let sum: u64 = a.iter().zip(&b).map(|(&x, &y)| u64::from(x.abs_diff(y))).sum();
    sum as f64 / a.len() as f64
}

#[test]
fn prores_round_trip_keeps_colours_and_tags() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("prores");
    let path = dir.join("clip.mov");
    let (w, h) = (64, 48);
    let tags = ColourTags::bt709();
    let settings =
        EncodeSettings { frame_rate: Rational::new(24, 1), colour: tags.clone(), ..EncodeSettings::default() };
    let mut enc = VideoEncoder::create(&ff, &path, w, h, &settings).unwrap();
    assert_eq!(enc.codec(), Codec::ProRes);
    for t in 0..6 {
        enc.write(&frame(w, h, t)).unwrap();
    }
    assert_eq!(enc.finish().unwrap(), 6);

    let info = probe(&ff, &path).unwrap();
    assert_eq!((info.width, info.height, info.frame_count), (w, h, 6));
    assert_eq!(info.codec, "prores");
    assert_eq!(info.frame_rate, Rational::new(24, 1));
    assert_eq!(info.colour, tags, "the encode writes all four tags");
    assert!(!info.is_vfr());
    let frames: Vec<Image> =
        VideoDecoder::open(&ff, &path, &info, &info.decode_colour()).unwrap().collect::<Result<_, _>>().unwrap();
    assert_eq!(frames.len(), 6);
    for (t, f) in frames.iter().enumerate() {
        let err = mean_abs_diff(f, &frame(w, h, t));
        assert!(err < 1.0, "frame {t}: mean error {err}");
    }
    // Decoding with the wrong matrix is visibly off: the tags matter.
    let wrong = ColourTags::from_probe(Some("smpte170m"), Some("tv"), None, None);
    let off = VideoDecoder::open(&ff, &path, &info, &wrong).unwrap().next().unwrap().unwrap();
    assert!(mean_abs_diff(&off, &frame(w, h, 0)) > 2.0);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn lossless_16_bit_and_alpha() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("ffv1");
    let path = dir.join("clip.mkv");
    let (w, h) = (32, 24);
    let mut img = frame(w, h, 0);
    let mut data = img.data().to_vec();
    for (i, px) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        px[3] = (i % 256) as u8;
    }
    img = Image::from_raw(w, h, 4, data);
    let settings = EncodeSettings { alpha: true, depth: BitDepth::Sixteen, ..EncodeSettings::default() };
    let mut enc = VideoEncoder::create(&ff, &path, w, h, &settings).unwrap();
    assert_eq!(enc.codec(), Codec::Ffv1);
    enc.write(&img).unwrap();
    enc.write(&img).unwrap();
    enc.finish().unwrap();
    let info = probe(&ff, &path).unwrap();
    assert!(info.has_alpha, "{}", info.pix_fmt);
    assert!(info.pix_fmt.contains("16"), "{}", info.pix_fmt);
    let back = VideoDecoder::open(&ff, &path, &info, &info.decode_colour()).unwrap().next().unwrap().unwrap();
    // FFV1 is lossless; only the conversion back to 8 bits may round by a level.
    let worst = back.data().iter().zip(img.data()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    assert!(worst <= 1, "largest difference {worst}");
    let alpha_ok =
        back.data().as_chunks::<4>().0.iter().zip(img.data().as_chunks::<4>().0).all(|(a, b)| a[3].abs_diff(b[3]) <= 1);
    assert!(alpha_ok, "alpha survives");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn decodes_a_frame_range_exactly() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("range");
    let path = dir.join("clip.mkv");
    let (w, h) = (32, 24);
    let settings = EncodeSettings { frame_rate: Rational::new(25, 1), ..EncodeSettings::default() };
    let mut enc = VideoEncoder::create(&ff, &path, w, h, &settings).unwrap();
    let frames: Vec<Image> = (0..10).map(|t| frame(w, h, 3 * t)).collect();
    for f in &frames {
        enc.write(f).unwrap();
    }
    enc.finish().unwrap();
    let info = probe(&ff, &path).unwrap();
    assert!(!info.has_audio);
    let got: Vec<Image> = VideoDecoder::open_range(&ff, &path, &info, &info.decode_colour(), 4, Some(3))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(got.len(), 3);
    for (i, g) in got.iter().enumerate() {
        // FFV1 is lossless: each frame matches its own source frame and no other.
        assert!(mean_abs_diff(g, &frames[4 + i]) < 0.01, "frame {}", 4 + i);
        assert!(mean_abs_diff(g, &frames[3 + i]) > 1.0);
    }
    // A count past the end gives the frames that exist.
    let tail = VideoDecoder::open_range(&ff, &path, &info, &info.decode_colour(), 8, Some(5)).unwrap();
    assert_eq!(tail.count(), 2);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn variable_frame_rate_timing_survives() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("vfr");
    let (w, h) = (32, 24);
    let durations = [0.04, 0.04, 0.1, 0.02, 0.04];
    let files: Vec<(PathBuf, f64)> = durations
        .iter()
        .enumerate()
        .map(|(t, &d)| {
            let p = dir.join(format!("{t:03}.png"));
            save_png(&frame(w, h, t), &p, BitDepth::Eight).unwrap();
            (p, d)
        })
        .collect();
    let path = dir.join("vfr.mov");
    let settings = EncodeSettings { codec: Codec::Png, ..EncodeSettings::default() };
    encode_files(&ff, &files, &path, w, h, &settings, &dir).unwrap();
    let info = probe(&ff, &path).unwrap();
    assert_eq!(info.frame_count, 5);
    assert!(info.is_vfr(), "{:?} / {:?}", info.pts, info.time_base);
    for (i, d) in durations.iter().enumerate().take(4) {
        assert!((info.frame_duration(i) - d).abs() < 2e-3, "frame {i}: {} vs {d}", info.frame_duration(i));
    }
    // The decoder returns every stored frame exactly once.
    let n = VideoDecoder::open(&ff, &path, &info, &info.decode_colour()).unwrap().count();
    assert_eq!(n, 5);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn failures_are_clear() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("errors");
    let err = probe(&ff, &dir.join("missing.mov")).unwrap_err();
    assert!(matches!(err, MediaError::Io { .. }), "{err}");
    let text = dir.join("not-a-video.mov");
    std::fs::write(&text, "hello").unwrap();
    let err = probe(&ff, &text).unwrap_err().to_string();
    assert!(err.contains("ffprobe"), "{err}");
    // A protocol outside the whitelist is refused.
    let list = dir.join("net.ffconcat");
    std::fs::write(&list, "ffconcat version 1.0\nfile 'http://127.0.0.1:9/x.png'\n").unwrap();
    assert!(probe(&ff, &list).is_err());
    let missing = Ffmpeg::with_paths(dir.join("no-ffmpeg"), dir.join("no-ffprobe")).unwrap_err();
    assert!(matches!(missing, MediaError::FfmpegMissing(_)), "{missing}");
    std::fs::remove_dir_all(dir).unwrap();
}
