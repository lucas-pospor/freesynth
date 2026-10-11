//! A baseline and progressive JPEG decoder that reproduces stb_image v2.12 exactly.
//!
//! The 2019 `ebsynth` tool loads every image with `stbi_load` (the stb_image copy shipped with
//! its source). Other JPEG decoders use different IDCTs, chroma upsamplers and colour
//! conversions, so their pixels differ slightly; PatchMatch is chaotic, which turns those small
//! input differences into a visibly different synthesis. This module is a direct translation of
//! stb_image's public-domain JPEG path (Huffman and progressive decoding, the integer IDCT,
//! the "fancy" chroma upsampling and the fixed-point YCbCr to RGB conversion), so a JPEG decodes
//! to the same bytes as in the reference build.
//!
//! The reference was compiled for x86-64, where stb_image uses its SSE2 kernels. The SSE2
//! upsampler and colour converter are bit-identical to the scalar code (and are implemented
//! here in scalar form); the SSE2 IDCT is identical on valid streams but saturates its 16-bit
//! intermediates, so [`idct_block`] emulates the SSE2 arithmetic lane by lane to also match on
//! extreme coefficients.
//!
//! Like stb_image v2.12 this decodes 8-bit greyscale and 3-component (YCbCr, or RGB when the
//! component ids are 'R', 'G', 'B') baseline and progressive JPEGs; it does not interpret Adobe
//! or EXIF markers. Corrupt entropy-coded data is decoded on a best-effort basis as stb_image
//! does (missing data reads as zeros), but malformed headers and table overflows are errors,
//! never panics.

// The IDCT constants are written with stb_image's literals; they are rounded to f32 exactly as
// in the C code.
#![allow(clippy::excessive_precision)]

use image::ImageFormat;
use image::error::{DecodingError, ImageError};

use crate::error::Error;
use crate::image::Image;

/// Number of bits of the Huffman lookahead table.
const FAST_BITS: u32 = 9;
/// "No marker" value of the marker cache (never a valid marker).
const MARKER_NONE: u8 = 0xff;

fn err<T>(msg: &str) -> Result<T, Error> {
    Err(Error::Image(ImageError::Decoding(DecodingError::new(ImageFormat::Jpeg.into(), msg.to_string()))))
}

/// Whether `bytes` start like a JPEG file, using the same test as stb_image (`stbi__jpeg_test`):
/// the first marker is SOI.
pub(crate) fn is_jpeg(bytes: &[u8]) -> bool {
    let mut s = Reader::new(bytes);
    let mut bits = Bits::new();
    get_marker(&mut s, &mut bits) == 0xd8
}

/// Decodes a JPEG file to 8-bit RGBA, exactly like `stbi_load(..., 4)` in stb_image v2.12.
pub fn decode_rgba(bytes: &[u8]) -> Result<Image, Error> {
    let mut j = Jpeg::new(bytes);
    j.decode_image()?;
    Ok(j.convert_rgba())
}

// ---------------------------------------------------------------------------------------------
// Byte source

/// The input stream; like stb_image, reads past the end return zeros.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn get8(&mut self) -> u8 {
        match self.buf.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                b
            }
            None => 0,
        }
    }

    fn get16be(&mut self) -> i32 {
        let hi = self.get8() as i32;
        (hi << 8) | self.get8() as i32
    }

    fn at_eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn skip(&mut self, n: i32) {
        if n < 0 {
            self.pos = self.buf.len();
        } else {
            self.pos = self.pos.saturating_add(n as usize);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Huffman tables

#[derive(Clone)]
struct Huffman {
    fast: [u8; 1 << FAST_BITS],
    code: [u16; 256],
    values: [u8; 256],
    size: [u8; 257],
    maxcode: [u32; 18],
    /// Old 'firstsymbol' - old 'firstcode'.
    delta: [i32; 17],
}

/// An all-zero table. stb_image leaves tables that a (corrupt) file never defines
/// uninitialised; zeros are what a fresh allocation holds, so undefined tables decode alike.
impl Default for Huffman {
    fn default() -> Self {
        Self { fast: [0; 1 << FAST_BITS], code: [0; 256], values: [0; 256], size: [0; 257], maxcode: [0; 18], delta: [0; 17] }
    }
}

/// `stbi__build_huffman`.
fn build_huffman(h: &mut Huffman, count: &[i32; 16]) -> Result<(), Error> {
    if count.iter().sum::<i32>() > 256 {
        // stb_image would overflow its tables here.
        return err("bad code lengths");
    }
    let mut k = 0usize;
    for (i, &c) in count.iter().enumerate() {
        for _ in 0..c {
            h.size[k] = (i + 1) as u8;
            k += 1;
        }
    }
    h.size[k] = 0;

    let mut code: i32 = 0;
    k = 0;
    let mut j = 1usize;
    while j <= 16 {
        h.delta[j] = k as i32 - code;
        if h.size[k] as usize == j {
            while h.size[k] as usize == j {
                h.code[k] = code as u16;
                code += 1;
                k += 1;
            }
            if code > (1 << j) {
                return err("bad code lengths");
            }
        }
        h.maxcode[j] = (code << (16 - j)) as u32;
        code <<= 1;
        j += 1;
    }
    h.maxcode[j] = 0xffff_ffff;

    h.fast = [255; 1 << FAST_BITS];
    for i in 0..k {
        let s = h.size[i] as u32;
        if s <= FAST_BITS {
            let c = (h.code[i] as usize) << (FAST_BITS - s);
            let m = 1usize << (FAST_BITS - s);
            for e in &mut h.fast[c..c + m] {
                *e = i as u8;
            }
        }
    }
    Ok(())
}

/// `stbi__build_fast_ac`: a table that decodes run, magnitude and value of small AC
/// coefficients in one lookup.
fn build_fast_ac(fast_ac: &mut [i16; 1 << FAST_BITS], h: &Huffman) {
    for (i, out) in fast_ac.iter_mut().enumerate() {
        let fast = h.fast[i];
        *out = 0;
        if fast < 255 {
            let rs = h.values[fast as usize] as i32;
            let run = (rs >> 4) & 15;
            let magbits = rs & 15;
            let len = h.size[fast as usize] as i32;
            if magbits != 0 && len + magbits <= FAST_BITS as i32 {
                let mut k = (((i as i32) << len) & ((1 << FAST_BITS) - 1)) >> (FAST_BITS as i32 - magbits);
                let m = 1 << (magbits - 1);
                if k < m {
                    k += (-1i32 << magbits) + 1;
                }
                if (-128..=127).contains(&k) {
                    *out = ((k << 8) + (run << 4) + (len + magbits)) as i16;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Entropy decoder

/// `(1 << n) - 1`.
const BMASK: [u32; 17] = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535];
/// `(-1 << n) + 1`.
const JBIAS: [i32; 16] = [0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047, -4095, -8191, -16383, -32767];

/// Where the coefficient at zigzag position `k` lives in the row-major 8x8 block (with
/// padding so that corrupt runs past the end land on 63).
const DEZIGZAG: [u8; 64 + 15] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
];

/// Bit-level state of the entropy decoder.
struct Bits {
    code_buffer: u32,
    code_bits: i32,
    /// Marker seen while filling the bit buffer.
    marker: u8,
    /// Set once a marker was seen: no more entropy-coded bytes.
    nomore: bool,
    eob_run: i32,
}

impl Bits {
    fn new() -> Self {
        Self { code_buffer: 0, code_bits: 0, marker: MARKER_NONE, nomore: false, eob_run: 0 }
    }

    /// `stbi__grow_buffer_unsafe`.
    fn grow(&mut self, s: &mut Reader<'_>) {
        loop {
            let b = if self.nomore { 0 } else { s.get8() as u32 };
            if b == 0xff {
                let c = s.get8();
                if c != 0 {
                    self.marker = c;
                    self.nomore = true;
                    return;
                }
            }
            // After a marker `b` is 0, so a shift past the word size (possible once code_bits
            // went negative on corrupt data) cannot lose bits.
            self.code_buffer |= b.checked_shl((24 - self.code_bits) as u32).unwrap_or(0);
            self.code_bits += 8;
            if self.code_bits > 24 {
                return;
            }
        }
    }

    /// `stbi__jpeg_huff_decode`; `None` for an invalid code.
    #[inline]
    fn huff_decode(&mut self, s: &mut Reader<'_>, h: &Huffman) -> Option<i32> {
        if self.code_bits < 16 {
            self.grow(s);
        }
        let c = ((self.code_buffer >> (32 - FAST_BITS)) & ((1 << FAST_BITS) - 1)) as usize;
        let k = h.fast[c];
        if k < 255 {
            let sz = h.size[k as usize] as i32;
            if sz > self.code_bits {
                return None;
            }
            self.code_buffer <<= sz;
            self.code_bits -= sz;
            return Some(h.values[k as usize] as i32);
        }
        let temp = self.code_buffer >> 16;
        let mut k = FAST_BITS as usize + 1;
        while k < 17 && temp >= h.maxcode[k] {
            k += 1;
        }
        if k == 17 {
            self.code_bits -= 16;
            return None;
        }
        if k as i32 > self.code_bits {
            return None;
        }
        let c = ((self.code_buffer >> (32 - k)) & BMASK[k]) as i32 + h.delta[k];
        let value = *h.values.get(usize::try_from(c).ok()?)?;
        self.code_bits -= k as i32;
        self.code_buffer <<= k;
        Some(value as i32)
    }

    /// `stbi__extend_receive`: reads `n` bits and sign-extends them.
    #[inline]
    fn extend_receive(&mut self, s: &mut Reader<'_>, n: i32) -> Result<i32, Error> {
        if !(0..=15).contains(&n) {
            // stb_image asserts (and would index its tables out of bounds).
            return err("bad huffman code");
        }
        if self.code_bits < n {
            self.grow(s);
        }
        let sgn = (self.code_buffer as i32) >> 31;
        let mut k = self.code_buffer.rotate_left(n as u32);
        self.code_buffer = k & !BMASK[n as usize];
        k &= BMASK[n as usize];
        self.code_bits -= n;
        Ok((k as i32).wrapping_add(JBIAS[n as usize] & !sgn))
    }

    /// `stbi__jpeg_get_bits`.
    fn get_bits(&mut self, s: &mut Reader<'_>, n: i32) -> i32 {
        if self.code_bits < n {
            self.grow(s);
        }
        let mut k = self.code_buffer.rotate_left(n as u32);
        self.code_buffer = k & !BMASK[n as usize];
        k &= BMASK[n as usize];
        self.code_bits -= n;
        k as i32
    }

    /// `stbi__jpeg_get_bit`.
    fn get_bit(&mut self, s: &mut Reader<'_>) -> bool {
        if self.code_bits < 1 {
            self.grow(s);
        }
        let k = self.code_buffer;
        self.code_buffer <<= 1;
        self.code_bits -= 1;
        k & 0x8000_0000 != 0
    }
}

/// `stbi__get_marker`: the pending marker from the entropy stream, else the next marker in the
/// stream, or [`MARKER_NONE`].
fn get_marker(s: &mut Reader<'_>, bits: &mut Bits) -> u8 {
    if bits.marker != MARKER_NONE {
        let x = bits.marker;
        bits.marker = MARKER_NONE;
        return x;
    }
    let mut x = s.get8();
    if x != 0xff {
        return MARKER_NONE;
    }
    while x == 0xff {
        x = s.get8();
    }
    x
}

/// Parameters of the current scan.
#[derive(Clone, Copy, Default)]
struct Scan {
    spec_start: i32,
    spec_end: i32,
    succ_high: i32,
    succ_low: i32,
}

/// `stbi__jpeg_decode_block`: one baseline block, dequantised, in row-major order.
#[allow(clippy::too_many_arguments)]
fn decode_block(
    bits: &mut Bits,
    s: &mut Reader<'_>,
    data: &mut [i16; 64],
    hdc: &Huffman,
    hac: &Huffman,
    fac: &[i16; 1 << FAST_BITS],
    dc_pred: &mut i32,
    dequant: &[u8; 64],
) -> Result<(), Error> {
    if bits.code_bits < 16 {
        bits.grow(s);
    }
    let Some(t) = bits.huff_decode(s, hdc) else { return err("bad huffman code") };
    *data = [0; 64];
    let diff = if t != 0 { bits.extend_receive(s, t)? } else { 0 };
    let dc = dc_pred.wrapping_add(diff);
    *dc_pred = dc;
    data[0] = dc.wrapping_mul(dequant[0] as i32) as i16;

    let mut k: usize = 1;
    loop {
        if bits.code_bits < 16 {
            bits.grow(s);
        }
        let c = ((bits.code_buffer >> (32 - FAST_BITS)) & ((1 << FAST_BITS) - 1)) as usize;
        let r = fac[c] as i32;
        if r != 0 {
            // Fast path: run, length and value from one lookup.
            k += ((r >> 4) & 15) as usize;
            let sz = r & 15;
            bits.code_buffer <<= sz;
            bits.code_bits -= sz;
            let zig = DEZIGZAG[k] as usize;
            k += 1;
            data[zig] = ((r >> 8) * dequant[zig] as i32) as i16;
        } else {
            let Some(rs) = bits.huff_decode(s, hac) else { return err("bad huffman code") };
            let sz = rs & 15;
            let r = rs >> 4;
            if sz == 0 {
                if rs != 0xf0 {
                    break; // end of block
                }
                k += 16;
            } else {
                k += r as usize;
                let zig = DEZIGZAG[k] as usize;
                k += 1;
                data[zig] = bits.extend_receive(s, sz)?.wrapping_mul(dequant[zig] as i32) as i16;
            }
        }
        if k >= 64 {
            break;
        }
    }
    Ok(())
}

/// `stbi__jpeg_decode_block_prog_dc`.
fn decode_block_prog_dc(
    bits: &mut Bits,
    s: &mut Reader<'_>,
    data: &mut [i16],
    hdc: &Huffman,
    dc_pred: &mut i32,
    scan: Scan,
) -> Result<(), Error> {
    if scan.spec_end != 0 {
        return err("can't merge dc and ac");
    }
    if bits.code_bits < 16 {
        bits.grow(s);
    }
    if scan.succ_high == 0 {
        // First DC scan.
        data[..64].fill(0);
        // stb_image does not check this code (an invalid one trips an assertion there).
        let Some(t) = bits.huff_decode(s, hdc) else { return err("bad huffman code") };
        let diff = if t != 0 { bits.extend_receive(s, t)? } else { 0 };
        let dc = dc_pred.wrapping_add(diff);
        *dc_pred = dc;
        data[0] = dc.wrapping_shl(scan.succ_low as u32) as i16;
    } else if bits.get_bit(s) {
        // DC refinement.
        data[0] = data[0].wrapping_add((1i32 << scan.succ_low) as i16);
    }
    Ok(())
}

/// `stbi__jpeg_decode_block_prog_ac`.
fn decode_block_prog_ac(
    bits: &mut Bits,
    s: &mut Reader<'_>,
    data: &mut [i16],
    hac: &Huffman,
    fac: &[i16; 1 << FAST_BITS],
    scan: Scan,
) -> Result<(), Error> {
    if scan.spec_start == 0 {
        return err("can't merge dc and ac");
    }
    if scan.succ_high == 0 {
        let shift = scan.succ_low as u32;
        if bits.eob_run != 0 {
            bits.eob_run -= 1;
            return Ok(());
        }
        let mut k = scan.spec_start as usize;
        loop {
            if bits.code_bits < 16 {
                bits.grow(s);
            }
            let c = ((bits.code_buffer >> (32 - FAST_BITS)) & ((1 << FAST_BITS) - 1)) as usize;
            let r = fac[c] as i32;
            if r != 0 {
                k += ((r >> 4) & 15) as usize;
                let sz = r & 15;
                bits.code_buffer <<= sz;
                bits.code_bits -= sz;
                let zig = DEZIGZAG[k] as usize;
                k += 1;
                data[zig] = (r >> 8).wrapping_shl(shift) as i16;
            } else {
                let Some(rs) = bits.huff_decode(s, hac) else { return err("bad huffman code") };
                let sz = rs & 15;
                let r = rs >> 4;
                if sz == 0 {
                    if r < 15 {
                        bits.eob_run = 1 << r;
                        if r != 0 {
                            bits.eob_run += bits.get_bits(s, r);
                        }
                        bits.eob_run -= 1;
                        break;
                    }
                    k += 16;
                } else {
                    k += r as usize;
                    let zig = DEZIGZAG[k] as usize;
                    k += 1;
                    data[zig] = bits.extend_receive(s, sz)?.wrapping_shl(shift) as i16;
                }
            }
            if k as i32 > scan.spec_end {
                break;
            }
        }
    } else {
        // Refinement scan for these AC coefficients.
        let bit = (1i32 << scan.succ_low) as i16;
        let refine = |bits: &mut Bits, s: &mut Reader<'_>, p: &mut i16| {
            if bits.get_bit(s) && (*p & bit) == 0 {
                if *p > 0 {
                    *p = p.wrapping_add(bit);
                } else {
                    *p = p.wrapping_sub(bit);
                }
            }
        };
        if bits.eob_run != 0 {
            bits.eob_run -= 1;
            for k in scan.spec_start..=scan.spec_end {
                let p = &mut data[DEZIGZAG[k as usize] as usize];
                if *p != 0 {
                    refine(bits, s, p);
                }
            }
        } else {
            let mut k = scan.spec_start;
            loop {
                let Some(rs) = bits.huff_decode(s, hac) else { return err("bad huffman code") };
                let mut sz = rs & 15;
                let mut r = rs >> 4;
                if sz == 0 {
                    if r < 15 {
                        bits.eob_run = (1 << r) - 1;
                        if r != 0 {
                            bits.eob_run += bits.get_bits(s, r);
                        }
                        r = 64; // force end of block
                    }
                    // r == 15, s == 0: a run of 15 zeros followed by a zero, nothing special.
                } else {
                    if sz != 1 {
                        return err("bad huffman code");
                    }
                    sz = if bits.get_bit(s) { bit as i32 } else { -(bit as i32) };
                }
                // Advance by r.
                while k <= scan.spec_end {
                    let p = &mut data[DEZIGZAG[k as usize] as usize];
                    k += 1;
                    if *p != 0 {
                        refine(bits, s, p);
                    } else {
                        if r == 0 {
                            *p = sz as i16;
                            break;
                        }
                        r -= 1;
                    }
                }
                if k > scan.spec_end {
                    break;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// IDCT

/// `stbi__f2f`: a constant in 12-bit fixed point (`(int)(x * 4096 + 0.5)`, the product in
/// single and the sum in double precision, truncated).
const fn f2f(x: f32) -> i32 {
    ((x * 4096.0) as f64 + 0.5) as i32
}

/// Coefficient pairs of the SSE2 IDCT's `_mm_madd_epi16` rotations.
const ROT0_0: (i32, i32) = (f2f(0.5411961), f2f(0.5411961) + f2f(-1.847759065));
const ROT0_1: (i32, i32) = (f2f(0.5411961) + f2f(0.765366865), f2f(0.5411961));
const ROT1_0: (i32, i32) = (f2f(1.175875602) + f2f(-0.899976223), f2f(1.175875602));
const ROT1_1: (i32, i32) = (f2f(1.175875602), f2f(1.175875602) + f2f(-2.562915447));
const ROT2_0: (i32, i32) = (f2f(-1.961570560) + f2f(0.298631336), f2f(-1.961570560));
const ROT2_1: (i32, i32) = (f2f(-1.961570560), f2f(-1.961570560) + f2f(3.072711026));
const ROT3_0: (i32, i32) = (f2f(-0.390180644) + f2f(2.053119869), f2f(-0.390180644));
const ROT3_1: (i32, i32) = (f2f(-0.390180644), f2f(-0.390180644) + f2f(1.501321110));

/// `_mm_madd_epi16` on one lane pair: `x * c.0 + y * c.1` with wrapping 32-bit addition.
#[inline(always)]
fn madd(x: i16, y: i16, c: (i32, i32)) -> i32 {
    (x as i32 * c.0).wrapping_add(y as i32 * c.1)
}

/// `_mm_packs_epi32` on one value.
#[inline(always)]
fn sat16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// One 1-D pass of the SSE2 IDCT (`dct_pass`) over the 8 lanes of `rows` (`rows[i][lane]` is
/// input `i` of that lane). 16-bit sums wrap and the outputs saturate, as in SSE2.
#[allow(clippy::needless_range_loop)] // lanes index across the rows, as SSE2 registers do
fn idct_pass(rows: &mut [[i16; 8]; 8], bias: i32, shift: u32) {
    for lane in 0..8 {
        let r: [i16; 8] = std::array::from_fn(|i| rows[i][lane]);
        // Even part.
        let t2e = madd(r[2], r[6], ROT0_0);
        let t3e = madd(r[2], r[6], ROT0_1);
        let t0e = (r[0].wrapping_add(r[4]) as i32) << 12;
        let t1e = (r[0].wrapping_sub(r[4]) as i32) << 12;
        let x0 = t0e.wrapping_add(t3e);
        let x3 = t0e.wrapping_sub(t3e);
        let x1 = t1e.wrapping_add(t2e);
        let x2 = t1e.wrapping_sub(t2e);
        // Odd part.
        let y0o = madd(r[7], r[3], ROT2_0);
        let y2o = madd(r[7], r[3], ROT2_1);
        let y1o = madd(r[5], r[1], ROT3_0);
        let y3o = madd(r[5], r[1], ROT3_1);
        let sum17 = r[1].wrapping_add(r[7]);
        let sum35 = r[3].wrapping_add(r[5]);
        let y4o = madd(sum17, sum35, ROT1_0);
        let y5o = madd(sum17, sum35, ROT1_1);
        let x4 = y0o.wrapping_add(y4o);
        let x5 = y1o.wrapping_add(y5o);
        let x6 = y2o.wrapping_add(y5o);
        let x7 = y3o.wrapping_add(y4o);
        // Butterflies.
        let mut out = [0i16; 8];
        for (i, (a, b)) in [(x0, x7), (x1, x6), (x2, x5), (x3, x4)].into_iter().enumerate() {
            let ab = a.wrapping_add(bias);
            out[i] = sat16(ab.wrapping_add(b) >> shift);
            out[7 - i] = sat16(ab.wrapping_sub(b) >> shift);
        }
        for (i, v) in out.into_iter().enumerate() {
            rows[i][lane] = v;
        }
    }
}

fn transpose(rows: &mut [[i16; 8]; 8]) {
    let t = *rows;
    *rows = std::array::from_fn(|i| std::array::from_fn(|j| t[j][i]));
}

/// `stbi__idct_simd` (SSE2): inverse DCT of a dequantised row-major block into an 8x8 pixel
/// block at `out` with row stride `stride`.
fn idct_block(out: &mut [u8], stride: usize, data: &[i16; 64]) {
    let mut rows: [[i16; 8]; 8] = std::array::from_fn(|i| std::array::from_fn(|j| data[i * 8 + j]));
    idct_pass(&mut rows, 512, 10);
    transpose(&mut rows);
    idct_pass(&mut rows, 65536 + (128 << 17), 17);
    // rows[x][y] now holds pixel (x, y); `_mm_packus_epi16` clamps to 0..=255.
    for y in 0..8 {
        let o = &mut out[y * stride..y * stride + 8];
        for (x, v) in o.iter_mut().enumerate() {
            *v = rows[x][y].clamp(0, 255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Decoder state

#[derive(Default)]
struct Component {
    id: i32,
    h: i32,
    v: i32,
    tq: usize,
    hd: usize,
    ha: usize,
    dc_pred: i32,
    /// Effective size in samples.
    x: i32,
    y: i32,
    /// Allocated size (whole MCUs).
    w2: i32,
    h2: i32,
    data: Vec<u8>,
    /// Coefficients (progressive only), 64 per block.
    coeff: Vec<i16>,
    coeff_w: i32,
}

struct Jpeg<'a> {
    s: Reader<'a>,
    bits: Bits,
    img_x: i32,
    img_y: i32,
    img_n: usize,
    huff_dc: [Huffman; 4],
    huff_ac: [Huffman; 4],
    dequant: [[u8; 64]; 4],
    fast_ac: [[i16; 1 << FAST_BITS]; 4],
    img_h_max: i32,
    img_v_max: i32,
    img_mcu_x: i32,
    img_mcu_y: i32,
    comp: [Component; 4],
    progressive: bool,
    scan: Scan,
    rgb: i32,
    scan_n: usize,
    order: [usize; 4],
    restart_interval: i32,
    todo: i32,
}

impl<'a> Jpeg<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            s: Reader::new(bytes),
            bits: Bits::new(),
            img_x: 0,
            img_y: 0,
            img_n: 0,
            huff_dc: Default::default(),
            huff_ac: Default::default(),
            dequant: [[0; 64]; 4],
            fast_ac: [[0; 1 << FAST_BITS]; 4],
            img_h_max: 1,
            img_v_max: 1,
            img_mcu_x: 0,
            img_mcu_y: 0,
            comp: Default::default(),
            progressive: false,
            scan: Scan::default(),
            rgb: 0,
            scan_n: 0,
            order: [0; 4],
            restart_interval: 0,
            todo: 0,
        }
    }

    fn get_marker(&mut self) -> u8 {
        get_marker(&mut self.s, &mut self.bits)
    }

    /// `stbi__jpeg_reset`: restart the entropy decoder and DC prediction.
    fn reset(&mut self) {
        self.bits.code_bits = 0;
        self.bits.code_buffer = 0;
        self.bits.nomore = false;
        for c in &mut self.comp[..3] {
            c.dc_pred = 0;
        }
        self.bits.marker = MARKER_NONE;
        self.todo = if self.restart_interval != 0 { self.restart_interval } else { 0x7fff_ffff };
        self.bits.eob_run = 0;
    }

    /// Restart-interval bookkeeping after an MCU; `false` ends the scan (no restart marker
    /// where one was due: stb_image keeps the partial data).
    fn after_mcu(&mut self) -> bool {
        self.todo -= 1;
        if self.todo <= 0 {
            if self.bits.code_bits < 24 {
                self.bits.grow(&mut self.s);
            }
            if !(0xd0..=0xd7).contains(&self.bits.marker) {
                return false;
            }
            self.reset();
        }
        true
    }

    /// `stbi__parse_entropy_coded_data`.
    fn parse_entropy_coded_data(&mut self) -> Result<(), Error> {
        self.reset();
        let scan = self.scan;
        let mut data = [0i16; 64];
        if self.scan_n == 1 {
            // Non-interleaved: one block per MCU, in scan-line order over the component's
            // own blocks.
            let n = self.order[0];
            let w = (self.comp[n].x + 7) >> 3;
            let h = (self.comp[n].y + 7) >> 3;
            for j in 0..h {
                for i in 0..w {
                    let c = &mut self.comp[n];
                    if !self.progressive {
                        let ha = c.ha;
                        decode_block(
                            &mut self.bits,
                            &mut self.s,
                            &mut data,
                            &self.huff_dc[c.hd],
                            &self.huff_ac[ha],
                            &self.fast_ac[ha],
                            &mut c.dc_pred,
                            &self.dequant[c.tq],
                        )?;
                        let w2 = c.w2 as usize;
                        let off = w2 * j as usize * 8 + i as usize * 8;
                        idct_block(&mut c.data[off..], w2, &data);
                    } else {
                        let off = 64 * (i + j * c.coeff_w) as usize;
                        let block = &mut c.coeff[off..off + 64];
                        if scan.spec_start == 0 {
                            decode_block_prog_dc(&mut self.bits, &mut self.s, block, &self.huff_dc[c.hd], &mut c.dc_pred, scan)?;
                        } else {
                            let ha = c.ha;
                            decode_block_prog_ac(&mut self.bits, &mut self.s, block, &self.huff_ac[ha], &self.fast_ac[ha], scan)?;
                        }
                    }
                    if !self.after_mcu() {
                        return Ok(());
                    }
                }
            }
        } else {
            // Interleaved MCUs.
            for j in 0..self.img_mcu_y {
                for i in 0..self.img_mcu_x {
                    for k in 0..self.scan_n {
                        let n = self.order[k];
                        let c = &mut self.comp[n];
                        for y in 0..c.v {
                            for x in 0..c.h {
                                if !self.progressive {
                                    let x2 = ((i * c.h + x) * 8) as usize;
                                    let y2 = ((j * c.v + y) * 8) as usize;
                                    let ha = c.ha;
                                    decode_block(
                                        &mut self.bits,
                                        &mut self.s,
                                        &mut data,
                                        &self.huff_dc[c.hd],
                                        &self.huff_ac[ha],
                                        &self.fast_ac[ha],
                                        &mut c.dc_pred,
                                        &self.dequant[c.tq],
                                    )?;
                                    let w2 = c.w2 as usize;
                                    idct_block(&mut c.data[w2 * y2 + x2..], w2, &data);
                                } else {
                                    let x2 = i * c.h + x;
                                    let y2 = j * c.v + y;
                                    let off = 64 * (x2 + y2 * c.coeff_w) as usize;
                                    let block = &mut c.coeff[off..off + 64];
                                    decode_block_prog_dc(&mut self.bits, &mut self.s, block, &self.huff_dc[c.hd], &mut c.dc_pred, scan)?;
                                }
                            }
                        }
                    }
                    if !self.after_mcu() {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    /// `stbi__jpeg_finish`: dequantise and transform the coefficients of a progressive image.
    fn finish(&mut self) {
        if !self.progressive {
            return;
        }
        for n in 0..self.img_n {
            let c = &mut self.comp[n];
            let w = (c.x + 7) >> 3;
            let h = (c.y + 7) >> 3;
            let dq = &self.dequant[c.tq];
            let w2 = c.w2 as usize;
            for j in 0..h {
                for i in 0..w {
                    let off = 64 * (i + j * c.coeff_w) as usize;
                    let mut data = [0i16; 64];
                    for (k, d) in data.iter_mut().enumerate() {
                        *d = c.coeff[off + k].wrapping_mul(dq[k] as i16);
                    }
                    idct_block(&mut c.data[w2 * j as usize * 8 + i as usize * 8..], w2, &data);
                }
            }
        }
    }

    /// `stbi__process_marker`.
    fn process_marker(&mut self, m: u8) -> Result<(), Error> {
        match m {
            MARKER_NONE => return err("expected marker"),
            0xdd => {
                // DRI: restart interval.
                if self.s.get16be() != 4 {
                    return err("bad DRI len");
                }
                self.restart_interval = self.s.get16be();
                return Ok(());
            }
            0xdb => {
                // DQT: quantisation tables.
                let mut l = self.s.get16be() - 2;
                while l > 0 {
                    let q = self.s.get8();
                    let p = q >> 4;
                    let t = (q & 15) as usize;
                    if p != 0 {
                        return err("bad DQT type");
                    }
                    if t > 3 {
                        return err("bad DQT table");
                    }
                    for &z in &DEZIGZAG[..64] {
                        self.dequant[t][z as usize] = self.s.get8();
                    }
                    l -= 65;
                }
                return if l == 0 { Ok(()) } else { err("bad DQT len") };
            }
            0xc4 => {
                // DHT: Huffman tables.
                let mut l = self.s.get16be() - 2;
                while l > 0 {
                    let q = self.s.get8();
                    let tc = (q >> 4) as usize;
                    let th = (q & 15) as usize;
                    if tc > 1 || th > 3 {
                        return err("bad DHT header");
                    }
                    let mut sizes = [0i32; 16];
                    for v in &mut sizes {
                        *v = self.s.get8() as i32;
                    }
                    let n: i32 = sizes.iter().sum();
                    l -= 17;
                    let h = if tc == 0 { &mut self.huff_dc[th] } else { &mut self.huff_ac[th] };
                    build_huffman(h, &sizes)?;
                    for v in &mut h.values[..n as usize] {
                        *v = self.s.get8();
                    }
                    if tc != 0 {
                        build_fast_ac(&mut self.fast_ac[th], &self.huff_ac[th]);
                    }
                    l -= n;
                }
                return if l == 0 { Ok(()) } else { err("bad DHT len") };
            }
            _ => {}
        }
        // Comments and APP blocks are skipped.
        if (0xe0..=0xef).contains(&m) || m == 0xfe {
            let len = self.s.get16be();
            self.s.skip(len - 2);
            return Ok(());
        }
        err("unknown marker")
    }

    /// `stbi__process_scan_header` (after SOS).
    fn process_scan_header(&mut self) -> Result<(), Error> {
        let ls = self.s.get16be();
        self.scan_n = self.s.get8() as usize;
        if self.scan_n < 1 || self.scan_n > 4 || self.scan_n > self.img_n {
            return err("bad SOS component count");
        }
        if ls != 6 + 2 * self.scan_n as i32 {
            return err("bad SOS len");
        }
        for i in 0..self.scan_n {
            let id = self.s.get8() as i32;
            let q = self.s.get8() as usize;
            let Some(which) = (0..self.img_n).find(|&w| self.comp[w].id == id) else {
                return err("bad SOS component id");
            };
            let c = &mut self.comp[which];
            c.hd = q >> 4;
            if c.hd > 3 {
                return err("bad DC huff");
            }
            c.ha = q & 15;
            if c.ha > 3 {
                return err("bad AC huff");
            }
            self.order[i] = which;
        }
        let spec_start = self.s.get8() as i32;
        let mut spec_end = self.s.get8() as i32;
        let aa = self.s.get8() as i32;
        let (succ_high, succ_low) = (aa >> 4, aa & 15);
        if self.progressive {
            if spec_start > 63 || spec_end > 63 || spec_start > spec_end || succ_high > 13 || succ_low > 13 {
                return err("bad SOS");
            }
        } else {
            if spec_start != 0 || succ_high != 0 || succ_low != 0 {
                return err("bad SOS");
            }
            spec_end = 63;
        }
        self.scan = Scan { spec_start, spec_end, succ_high, succ_low };
        Ok(())
    }

    /// `stbi__process_frame_header` (SOF), allocating the component planes.
    fn process_frame_header(&mut self) -> Result<(), Error> {
        let lf = self.s.get16be();
        if lf < 11 {
            return err("bad SOF len");
        }
        if self.s.get8() != 8 {
            return err("JPEG format not supported: 8-bit only");
        }
        self.img_y = self.s.get16be();
        if self.img_y == 0 {
            return err("JPEG format not supported: delayed height");
        }
        self.img_x = self.s.get16be();
        if self.img_x == 0 {
            return err("0 width");
        }
        let c = self.s.get8() as usize;
        if c != 3 && c != 1 {
            return err("bad component count");
        }
        self.img_n = c;
        if lf != 8 + 3 * c as i32 {
            return err("bad SOF len");
        }
        self.rgb = 0;
        const RGB: [i32; 3] = [b'R' as i32, b'G' as i32, b'B' as i32];
        for (i, comp) in self.comp[..c].iter_mut().enumerate() {
            comp.id = self.s.get8() as i32;
            if comp.id != i as i32 + 1 && comp.id != i as i32 {
                // Some encoders label the components 'R', 'G', 'B' (no colour transform).
                if comp.id != RGB[i] {
                    return err("bad component ID");
                }
                self.rgb += 1;
            }
            let q = self.s.get8() as i32;
            comp.h = q >> 4;
            if comp.h == 0 || comp.h > 4 {
                return err("bad H");
            }
            comp.v = q & 15;
            if comp.v == 0 || comp.v > 4 {
                return err("bad V");
            }
            comp.tq = self.s.get8() as usize;
            if comp.tq > 3 {
                return err("bad TQ");
            }
        }

        if (1 << 30) / self.img_x / (self.img_n as i32) < self.img_y {
            return err("Image too large to decode");
        }
        let h_max = self.comp[..c].iter().map(|k| k.h).max().unwrap_or(1).max(1);
        let v_max = self.comp[..c].iter().map(|k| k.v).max().unwrap_or(1).max(1);
        self.img_h_max = h_max;
        self.img_v_max = v_max;
        let mcu_w = h_max * 8;
        let mcu_h = v_max * 8;
        self.img_mcu_x = (self.img_x + mcu_w - 1) / mcu_w;
        self.img_mcu_y = (self.img_y + mcu_h - 1) / mcu_h;
        for comp in &mut self.comp[..c] {
            comp.x = (self.img_x * comp.h + h_max - 1) / h_max;
            comp.y = (self.img_y * comp.v + v_max - 1) / v_max;
            comp.w2 = self.img_mcu_x * comp.h * 8;
            comp.h2 = self.img_mcu_y * comp.v * 8;
            // stb_image reads up to a row's worth past the plane for unusual sampling factors
            // (e.g. 3:2); the padding keeps those reads in bounds.
            comp.data = vec![0; comp.w2 as usize * comp.h2 as usize + self.img_x as usize + 16];
            if self.progressive {
                comp.coeff_w = (comp.w2 + 7) >> 3;
                let coeff_h = (comp.h2 + 7) >> 3;
                comp.coeff = vec![0; comp.coeff_w as usize * coeff_h as usize * 64];
            } else {
                comp.coeff = Vec::new();
                comp.coeff_w = 0;
            }
        }
        Ok(())
    }

    /// `stbi__decode_jpeg_header`: up to and including the frame header.
    fn decode_header(&mut self) -> Result<(), Error> {
        self.bits.marker = MARKER_NONE;
        if self.get_marker() != 0xd8 {
            return err("no SOI");
        }
        let mut m = self.get_marker();
        while !matches!(m, 0xc0..=0xc2) {
            self.process_marker(m)?;
            m = self.get_marker();
            while m == MARKER_NONE {
                // Some files have padding after their blocks.
                if self.s.at_eof() {
                    return err("no SOF");
                }
                m = self.get_marker();
            }
        }
        self.progressive = m == 0xc2;
        self.process_frame_header()
    }

    /// `stbi__decode_jpeg_image`: decodes all scans into the component planes.
    fn decode_image(&mut self) -> Result<(), Error> {
        self.restart_interval = 0;
        self.decode_header()?;
        let mut m = self.get_marker();
        while m != 0xd9 {
            if m == 0xda {
                self.process_scan_header()?;
                self.parse_entropy_coded_data()?;
                if self.bits.marker == MARKER_NONE {
                    // Tolerate zero padding at the end of the entropy-coded data.
                    while !self.s.at_eof() {
                        let x = self.s.get8();
                        if x == 255 {
                            self.bits.marker = self.s.get8();
                            break;
                        } else if x != 0 {
                            return err("junk before marker");
                        }
                    }
                }
            } else {
                self.process_marker(m)?;
            }
            m = self.get_marker();
        }
        self.finish();
        Ok(())
    }

    /// The colour conversion of `load_jpeg_image` with `req_comp = 4`: upsamples the
    /// components and converts to RGBA.
    fn convert_rgba(&self) -> Image {
        let (w, h) = (self.img_x as usize, self.img_y as usize);
        let n = self.img_n;
        let mut out = vec![0u8; w * h * 4];

        struct Resample {
            kind: u8,
            hs: usize,
            vs: usize,
            w_lores: usize,
            ystep: usize,
            ypos: i32,
            line0: usize,
            line1: usize,
        }
        let mut res: Vec<Resample> = (0..n)
            .map(|k| {
                let c = &self.comp[k];
                let hs = (self.img_h_max / c.h) as usize;
                let vs = (self.img_v_max / c.v) as usize;
                let kind = match (hs, vs) {
                    (1, 1) => 0,
                    (1, 2) => 1,
                    (2, 1) => 2,
                    (2, 2) => 3,
                    _ => 4,
                };
                Resample { kind, hs, vs, w_lores: w.div_ceil(hs), ystep: vs >> 1, ypos: 0, line0: 0, line1: 0 }
            })
            .collect();
        // Line buffers, big enough for the upsampled rows (which may overhang the image).
        let mut linebufs: Vec<Vec<u8>> = res.iter().map(|r| vec![0u8; (r.w_lores * r.hs).max(w) + 3]).collect();
        let mut use_linebuf = [false; 4];
        let mut direct = [0usize; 4];

        for out_row in out.chunks_exact_mut(w * 4) {
            for k in 0..n {
                let r = &mut res[k];
                let c = &self.comp[k];
                let y_bot = r.ystep >= (r.vs >> 1);
                let (near, far) = if y_bot { (r.line1, r.line0) } else { (r.line0, r.line1) };
                let in_near = &c.data[near..];
                let in_far = &c.data[far..];
                let buf = &mut linebufs[k];
                match r.kind {
                    0 => {
                        use_linebuf[k] = false;
                        direct[k] = near;
                    }
                    1 => {
                        resample_row_v_2(buf, in_near, in_far, r.w_lores);
                        use_linebuf[k] = true;
                    }
                    2 => {
                        resample_row_h_2(buf, in_near, r.w_lores);
                        use_linebuf[k] = true;
                    }
                    3 => {
                        resample_row_hv_2(buf, in_near, in_far, r.w_lores);
                        use_linebuf[k] = true;
                    }
                    _ => {
                        resample_row_generic(buf, in_near, r.w_lores, r.hs);
                        use_linebuf[k] = true;
                    }
                }
                r.ystep += 1;
                if r.ystep >= r.vs {
                    r.ystep = 0;
                    r.line0 = r.line1;
                    r.ypos += 1;
                    if r.ypos < c.y {
                        r.line1 += c.w2 as usize;
                    }
                }
            }
            let line = |k: usize| -> &[u8] {
                if use_linebuf[k] { &linebufs[k][..w] } else { &self.comp[k].data[direct[k]..direct[k] + w] }
            };
            let y = line(0);
            if n == 3 {
                let (cb, cr) = (line(1), line(2));
                if self.rgb == 3 {
                    for (i, o) in out_row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        *o = [y[i], cb[i], cr[i], 255];
                    }
                } else {
                    ycbcr_to_rgba_row(out_row, y, cb, cr);
                }
            } else {
                for (i, o) in out_row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    *o = [y[i], y[i], y[i], 255];
                }
            }
        }
        Image::from_raw(w, h, 4, out)
    }
}

// ---------------------------------------------------------------------------------------------
// Upsampling and colour conversion

/// `stbi__resample_row_v_2`: vertical 2x, 3:1 blend of the near and far rows.
fn resample_row_v_2(out: &mut [u8], near: &[u8], far: &[u8], w: usize) {
    for i in 0..w {
        out[i] = ((3 * near[i] as i32 + far[i] as i32 + 2) >> 2) as u8;
    }
}

/// `stbi__resample_row_h_2`: horizontal 2x.
fn resample_row_h_2(out: &mut [u8], input: &[u8], w: usize) {
    let inp = |i: usize| input[i] as i32;
    if w == 1 {
        out[0] = input[0];
        out[1] = input[0];
        return;
    }
    out[0] = input[0];
    out[1] = ((inp(0) * 3 + inp(1) + 2) >> 2) as u8;
    let mut i = 1;
    while i < w - 1 {
        let n = 3 * inp(i) + 2;
        out[i * 2] = ((n + inp(i - 1)) >> 2) as u8;
        out[i * 2 + 1] = ((n + inp(i + 1)) >> 2) as u8;
        i += 1;
    }
    out[i * 2] = ((inp(w - 2) * 3 + inp(w - 1) + 2) >> 2) as u8;
    out[i * 2 + 1] = input[w - 1];
}

/// `stbi__resample_row_hv_2` (identical to its SSE2 version): 2x in both directions with
/// 9:3:3:1 weights.
fn resample_row_hv_2(out: &mut [u8], near: &[u8], far: &[u8], w: usize) {
    if w == 1 {
        let v = ((3 * near[0] as i32 + far[0] as i32 + 2) >> 2) as u8;
        out[0] = v;
        out[1] = v;
        return;
    }
    let mut t1 = 3 * near[0] as i32 + far[0] as i32;
    out[0] = ((t1 + 2) >> 2) as u8;
    for i in 1..w {
        let t0 = t1;
        t1 = 3 * near[i] as i32 + far[i] as i32;
        out[i * 2 - 1] = ((3 * t0 + t1 + 8) >> 4) as u8;
        out[i * 2] = ((3 * t1 + t0 + 8) >> 4) as u8;
    }
    out[w * 2 - 1] = ((t1 + 2) >> 2) as u8;
}

/// `stbi__resample_row_generic`: nearest-neighbour horizontal expansion by `hs`.
fn resample_row_generic(out: &mut [u8], near: &[u8], w: usize, hs: usize) {
    for i in 0..w {
        out[i * hs..(i + 1) * hs].fill(near[i]);
    }
}

/// `float2fixed` of the reduced-precision colour conversion.
const fn float2fixed(x: f32) -> i32 {
    ((x * 4096.0 + 0.5) as i32) << 8
}

/// `stbi__YCbCr_to_RGB_row` (identical to its SSE2 version) writing RGBA.
fn ycbcr_to_rgba_row(out: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) {
    for (i, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let y_fixed = ((y[i] as i32) << 20) + (1 << 19);
        let cr = cr[i] as i32 - 128;
        let cb = cb[i] as i32 - 128;
        let r = y_fixed + cr * float2fixed(1.40200);
        let g = y_fixed + cr * -float2fixed(0.71414) + ((cb * -float2fixed(0.34414)) & 0xffff_0000_u32 as i32);
        let b = y_fixed + cb * float2fixed(1.77200);
        *o = [(r >> 20).clamp(0, 255) as u8, (g >> 20).clamp(0, 255) as u8, (b >> 20).clamp(0, 255) as u8, 255];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar `stbi__idct_block`, for checking the SSE2 emulation on ordinary data.
    fn idct_scalar(data: &[i16; 64]) -> [u8; 64] {
        fn idct_1d(s: [i32; 8]) -> ([i32; 4], [i32; 4]) {
            let (p2, p3) = (s[2], s[6]);
            let p1 = (p2 + p3) * f2f(0.5411961);
            let t2 = p1 + p3 * f2f(-1.847759065);
            let t3 = p1 + p2 * f2f(0.765366865);
            let (p2, p3) = (s[0], s[4]);
            let t0 = (p2 + p3) << 12;
            let t1 = (p2 - p3) << 12;
            let x = [t0 + t3, t1 + t2, t1 - t2, t0 - t3];
            let (mut t0, mut t1, mut t2, mut t3) = (s[7], s[5], s[3], s[1]);
            let p3 = t0 + t2;
            let p4 = t1 + t3;
            let p1 = t0 + t3;
            let p2 = t1 + t2;
            let p5 = (p3 + p4) * f2f(1.175875602);
            t0 *= f2f(0.298631336);
            t1 *= f2f(2.053119869);
            t2 *= f2f(3.072711026);
            t3 *= f2f(1.501321110);
            let p1 = p5 + p1 * f2f(-0.899976223);
            let p2 = p5 + p2 * f2f(-2.562915447);
            let p3 = p3 * f2f(-1.961570560);
            let p4 = p4 * f2f(-0.390180644);
            t3 += p1 + p4;
            t2 += p2 + p3;
            t1 += p2 + p4;
            t0 += p1 + p3;
            (x, [t3, t2, t1, t0])
        }
        let mut val = [0i32; 64];
        for i in 0..8 {
            let d: [i32; 8] = std::array::from_fn(|k| data[i + 8 * k] as i32);
            if d[1..].iter().all(|&v| v == 0) {
                for k in 0..8 {
                    val[i + 8 * k] = d[0] << 2;
                }
            } else {
                let (x, t) = idct_1d(d);
                for k in 0..4 {
                    val[i + 8 * k] = (x[k] + 512 + t[k]) >> 10;
                    val[i + 8 * (7 - k)] = (x[k] + 512 - t[k]) >> 10;
                }
            }
        }
        let mut out = [0u8; 64];
        for i in 0..8 {
            let v: [i32; 8] = std::array::from_fn(|k| val[i * 8 + k]);
            let (x, t) = idct_1d(v);
            for k in 0..4 {
                let b = x[k] + 65536 + (128 << 17);
                out[i * 8 + k] = ((b + t[k]) >> 17).clamp(0, 255) as u8;
                out[i * 8 + 7 - k] = ((b - t[k]) >> 17).clamp(0, 255) as u8;
            }
        }
        out
    }

    #[test]
    fn simd_idct_matches_scalar_on_ordinary_blocks() {
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 16) as i32
        };
        for trial in 0..2000 {
            let mut data = [0i16; 64];
            let n = 1 + trial % 64;
            for _ in 0..n {
                let i = (rnd() % 64) as usize;
                let amp = if i == 0 { 1024 } else { 400 };
                data[i] = ((rnd() % (2 * amp)) - amp) as i16;
            }
            let mut out = [0u8; 64];
            idct_block(&mut out, 8, &data);
            assert_eq!(out, idct_scalar(&data), "trial {trial}");
        }
    }

    #[test]
    fn constants_match_stb() {
        assert_eq!(f2f(0.5411961), 2217);
        assert_eq!(f2f(-1.847759065), -7567);
        assert_eq!(float2fixed(1.40200) >> 8, 5743);
    }

    #[test]
    fn rejects_garbage() {
        assert!(!is_jpeg(b"\x89PNG\r\n"));
        assert!(is_jpeg(b"\xff\xd8\xff\xe0"));
        assert!(decode_rgba(b"\xff\xd8\xff\xd9").is_err());
        assert!(decode_rgba(b"").is_err());
        // Truncated after SOI and a bogus segment: an error, never a panic.
        for len in 0..64 {
            let mut v = vec![0xff, 0xd8, 0xff, 0xdb, 0x00, 0x43, 0x00];
            v.extend((0..len).map(|i| i as u8));
            let _ = decode_rgba(&v);
        }
    }
}
