//! Guide images for one frame of a keyframe chain.
//!
//! The synthesis engine matches 8-bit guide channels, so every per-frame signal is turned into a
//! guide image here:
//!
//! - the **mapping** (positional) guide encodes keyframe positions as colours ([`encode_positions`]);
//! - the **temporal** guide is the chain's previous output warped to the current frame by optical
//!   flow ([`warp_rgba`]);
//! - the **mask** guide pairs the keyframe's alpha ([`alpha_as_gray`]) with the frame's mask.

use freesynth_core::{Image, io};
use freesynth_flow::FlowField;
use freesynth_flow::track::{ConfidenceMap, PositionField};
use rayon::prelude::*;

/// Encodes a position field as an RGBA mapping guide: `R = trunc(x / (S - 1) * 255)`,
/// `G = trunc(y / (S - 1) * 255)`, `B = 0`, `A = 255`, with `S` = `max(width, height)` of the
/// keyframe, so both axes share one scale. This compact 8-bit encoding quantises positions to
/// steps of `(S - 1) / 255` pixels (3.8 px at 960 px), which at the usual mapping weight costs
/// nothing measurable.
pub fn encode_positions(positions: &PositionField, keyframe_size: (usize, usize)) -> Image {
    let (w, h) = positions.size();
    let scale = 255.0 / (keyframe_size.0.max(keyframe_size.1).max(2) - 1) as f32;
    let code = |v: f32| (v * scale).clamp(0.0, 255.0) as u8;
    let mut data = vec![0u8; w * h * 4];
    data.par_chunks_mut(4).zip(positions.data().par_iter()).for_each(|(px, p)| {
        px.copy_from_slice(&[code(p[0]), code(p[1]), 0, 255]);
    });
    Image::from_raw(w, h, 4, data)
}

/// `image` (RGBA) warped by `flow`: `out(x) = image(x + flow(x))`, bilinear, with sample
/// positions clamped to the image and values rounded to the nearest level.
///
/// # Panics
/// If the sizes differ or the image is not RGBA.
pub fn warp_rgba(image: &Image, flow: &FlowField) -> Image {
    assert_eq!(image.size(), flow.size(), "image and flow differ in size");
    assert_eq!(image.channels(), 4, "warp_rgba expects RGBA");
    let (w, h) = image.size();
    let src = image.data();
    let mut data = vec![0u8; w * h * 4];
    data.par_chunks_mut(w * 4).zip(flow.data().par_chunks(w)).enumerate().for_each(|(y, (row, f))| {
        for (x, (px, d)) in row.as_chunks_mut::<4>().0.iter_mut().zip(f).enumerate() {
            let sx = (x as f32 + d[0]).clamp(0.0, (w - 1) as f32);
            let sy = (y as f32 + d[1]).clamp(0.0, (h - 1) as f32);
            let (x0, y0) = (sx as usize, sy as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
            let at = |x: usize, y: usize, c: usize| f32::from(src[(y * w + x) * 4 + c]);
            for (c, v) in px.iter_mut().enumerate() {
                let top = at(x0, y0, c) + (at(x1, y0, c) - at(x0, y0, c)) * fx;
                let bottom = at(x0, y1, c) + (at(x1, y1, c) - at(x0, y1, c)) * fx;
                *v = (top + (bottom - top) * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    Image::from_raw(w, h, 4, data)
}

/// Expands a 1- to 4-channel image (grey, grey + alpha, RGB, RGBA, as kept by the engine's
/// channel selection) to RGBA.
///
/// # Panics
/// If the image has more than 4 channels.
pub fn to_rgba(image: &Image) -> Image {
    let n = image.channels();
    if n == 4 {
        return image.clone();
    }
    let mut data = Vec::with_capacity(image.width() * image.height() * 4);
    for px in image.data().chunks_exact(n) {
        data.extend_from_slice(&match *px {
            [g] => [g, g, g, 255],
            [g, a] => [g, g, g, a],
            [r, g, b] => [r, g, b, 255],
            _ => panic!("to_rgba expects 1 to 4 channels, got {n}"),
        });
    }
    Image::from_raw(image.width(), image.height(), 4, data)
}

/// The alpha channel of an RGBA image as an opaque grey RGBA image (the mask guide's source).
pub fn alpha_as_gray(rgba: &Image) -> Image {
    assert_eq!(rgba.channels(), 4, "alpha_as_gray expects RGBA");
    let data = rgba.data().as_chunks::<4>().0.iter().flat_map(|px| [px[3], px[3], px[3], 255]).collect();
    Image::from_raw(rgba.width(), rgba.height(), 4, data)
}

/// A confidence map as a one-channel weight map (`round(255 c)`), for guide modulation.
pub fn confidence_weights(confidence: &ConfidenceMap) -> Image {
    let (w, h) = confidence.size();
    let data = confidence.data().iter().map(|&c| (c.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    Image::from_raw(w, h, 1, data)
}

/// True when the RGBA image has any pixel that is not fully opaque (it then has a meaningful
/// alpha channel for the engine).
pub fn has_alpha(rgba: &Image) -> bool {
    io::eval_num_channels(rgba).is_multiple_of(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_encoding_matches_the_8_bit_convention() {
        // 10 x 4 keyframe: S - 1 = 9, so x = 9 maps to 255 and y = 3 to trunc(3 / 9 * 255) = 85.
        let p = PositionField::identity(10, 4);
        let img = encode_positions(&p, (10, 4));
        assert_eq!(img.pixel(9, 3), &[255, 85, 0, 255]);
        assert_eq!(img.pixel(1, 1), &[28, 28, 0, 255]);
        let off = PositionField::from_vec(1, 1, vec![[-3.0, 40.0]]);
        assert_eq!(encode_positions(&off, (10, 4)).pixel(0, 0), &[0, 255, 0, 255]);
    }

    #[test]
    fn warping_follows_the_flow_convention() {
        let (w, h) = (6, 3);
        let data: Vec<u8> = (0..w * h).flat_map(|i| [(i * 10) as u8, 0, 0, 255]).collect();
        let img = Image::from_raw(w, h, 4, data);
        // out(x) = img(x + 1): every pixel takes its right neighbour, clamped at the edge.
        let out = warp_rgba(&img, &FlowField::constant(w, h, [1.0, 0.0]));
        assert_eq!(out.pixel(0, 1)[0], img.pixel(1, 1)[0]);
        assert_eq!(out.pixel(5, 1)[0], img.pixel(5, 1)[0]);
        // Half a pixel blends and rounds.
        let out = warp_rgba(&img, &FlowField::constant(w, h, [0.5, 0.0]));
        assert_eq!(out.pixel(0, 0)[0], 5);
        assert_eq!(warp_rgba(&img, &FlowField::new(w, h)), img);
    }

    #[test]
    fn channel_expansion() {
        let grey = Image::from_raw(1, 1, 1, vec![7]);
        assert_eq!(to_rgba(&grey).data(), &[7, 7, 7, 255]);
        let ga = Image::from_raw(1, 1, 2, vec![7, 9]);
        assert_eq!(to_rgba(&ga).data(), &[7, 7, 7, 9]);
        let rgb = Image::from_raw(1, 1, 3, vec![1, 2, 3]);
        assert_eq!(to_rgba(&rgb).data(), &[1, 2, 3, 255]);
        let rgba = Image::from_raw(1, 2, 4, vec![1, 2, 3, 4, 5, 6, 7, 255]);
        assert_eq!(alpha_as_gray(&rgba).data(), &[4, 4, 4, 255, 255, 255, 255, 255]);
        assert!(has_alpha(&rgba));
        assert!(!has_alpha(&to_rgba(&rgb)));
        let c = ConfidenceMap::from_vec(2, 1, vec![0.0, 1.0]);
        assert_eq!(confidence_weights(&c).data(), &[0, 255]);
    }
}
