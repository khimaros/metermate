//! turning a gate region into detector input, at main-stream resolution.
//!
//! this exists because of a measurement. the detector is reliable on vehicles
//! 64px wide and unreliable below 48px, and a go-4 on the far side of the
//! street is about 50px in the gate stream. the same vehicle is roughly 200px
//! in the main stream, so cropping the gate's region out of the main stream and
//! detecting on *that* moves far-range sightings from unreliable to reliable.
//!
//! it is also where the full-resolution training crops come from (r6.1).
//!
//! every function here is pure and tested. coordinate mapping between two
//! streams that share a field of view but not an aspect ratio is the classic
//! thing that looks right and is quietly off by a factor.

use crate::gate::Rect;

/// is this region so large that it is the scene rather than a thing in it?
pub fn covers_most_of(region: Rect, frame_w: u32, frame_h: u32, max_fraction: f32) -> bool {
    let frame = (frame_w as f32) * (frame_h as f32);
    if frame <= 0.0 {
        return true;
    }
    (region.w as f32 * region.h as f32) / frame > max_fraction
}

/// what fraction of `subject` is covered by `other`.
#[allow(dead_code)]
pub fn covered_fraction(subject: Rect, other: Rect) -> f32 {
    let x1 = subject.x.max(other.x);
    let y1 = subject.y.max(other.y);
    let x2 = (subject.x + subject.w).min(other.x + other.w);
    let y2 = (subject.y + subject.h).min(other.y + other.h);
    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }
    let area = (subject.w as f32) * (subject.h as f32);
    if area <= 0.0 {
        return 0.0;
    }
    ((x2 - x1) as f32 * (y2 - y1) as f32) / area
}

/// is this box clipped by the edge of the region it was found in?
///
/// the detector can only see inside the crop it was given, so a vehicle
/// extending past that edge comes back cut in half.
pub fn touches_edge(box_in: Rect, container: Rect, slack: u32) -> bool {
    box_in.x <= container.x + slack
        || box_in.y <= container.y + slack
        || box_in.x + box_in.w + slack >= container.x + container.w
        || box_in.y + box_in.h + slack >= container.y + container.h
}

/// grow a box by a fraction of its size, kept inside the frame.
///
/// used to re-crop a clipped detection rather than discard it. an earlier
/// version rejected anything touching its crop edge and threw away ten of every
/// eleven vehicles, which is a poor trade for a harvester whose whole purpose is
/// to accumulate examples. the box may be truncated, but the pixels around it
/// are right there in the full-resolution frame.
pub fn expand(r: Rect, margin: f32, frame_w: u32, frame_h: u32) -> Rect {
    let grow_x = (r.w as f32 * margin) as u32;
    let grow_y = (r.h as f32 * margin) as u32;
    let x = r.x.saturating_sub(grow_x);
    let y = r.y.saturating_sub(grow_y);
    let w = (r.w + grow_x * 2).min(frame_w.saturating_sub(x)).max(1);
    let h = (r.h + grow_y * 2).min(frame_h.saturating_sub(y)).max(1);
    Rect { x, y, w, h }
}

/// do two regions describe the same object? compares centres against size, so
/// a vehicle that drifts a little between frames is still itself.
///
/// shared by the harvest (do not photograph this again yet) and the gate loop
/// (do not re-run the detector on this again yet).
pub fn same_object(a: Rect, b: Rect, tolerance: f32) -> bool {
    let (acx, acy) = (a.x as f32 + a.w as f32 / 2.0, a.y as f32 + a.h as f32 / 2.0);
    let (bcx, bcy) = (b.x as f32 + b.w as f32 / 2.0, b.y as f32 + b.h as f32 / 2.0);
    let reach_x = (a.w.max(b.w) as f32) * tolerance;
    let reach_y = (a.h.max(b.h) as f32) * tolerance;
    (acx - bcx).abs() <= reach_x && (acy - bcy).abs() <= reach_y
}

/// gate pixels per main-stream pixel, one factor per axis.
///
/// the two streams share a field of view but not necessarily an aspect ratio,
/// so this is a pure per-axis scale with no offset -- and the two factors differ
/// exactly when the substream's pixels are not the scene's shape, which is how
/// an anamorphic substream works. neither size is assumed: both come off the
/// wire (r5.6).
pub fn scale(gate: (u32, u32), main: (u32, u32)) -> (f32, f32) {
    (
        main.0 as f32 / gate.0.max(1) as f32,
        main.1 as f32 / gate.1.max(1) as f32,
    )
}

/// map a gate-space rectangle into main-stream pixels, with context added.
///
/// the scale is derived from the frames actually in hand rather than from the
/// camera's nominal geometry. that keeps it correct when the streams are
/// reconfigured, and it lets a single video file stand in for both streams, so
/// the e2e suite can exercise this path without a camera (r5.5).
///
/// the crop is **square**, and that matters more than it looks. the detector's
/// input is square, so a long thin crop is letterboxed with
/// large black margins and scaled by its longest side. a 659x1440 strip became
/// 640x640 at 0.44x, turning a 300px car into a 133px smudge in a mostly black
/// frame: low confidence, loose boxes, and junk classes like `airplane`.
///
/// a square crop needs no padding at all, so every pixel of the input is scene
/// and the vehicle stays as large as the crop size allows.
pub fn to_main(
    region: Rect,
    gate_w: u32,
    gate_h: u32,
    main_w: u32,
    main_h: u32,
    cfg: &crate::config::CropCfg,
) -> Rect {
    let (sx, sy) = scale((gate_w, gate_h), (main_w, main_h));
    let cx = (region.x as f32 + region.w as f32 / 2.0) * sx;
    let cy = (region.y as f32 + region.h as f32 / 2.0) * sy;

    // the motion region is a partial silhouette -- often just a vehicle's
    // leading and trailing edges -- so the crop is sized from its longest side
    // and given generous context rather than hugging it.
    let side = (region.w as f32 * sx).max(region.h as f32 * sy) * (1.0 + cfg.context_margin);
    let side = side.clamp(cfg.min_crop_px as f32, cfg.max_crop_px as f32);
    clamp(cx - side / 2.0, cy - side / 2.0, side, side, main_w, main_h)
}

/// keep a rectangle inside the frame, preferring to slide it rather than shrink
/// it so an object near an edge keeps its full extent.
fn clamp(x: f32, y: f32, w: f32, h: f32, max_w: u32, max_h: u32) -> Rect {
    let w = (w.round() as u32).min(max_w).max(1);
    let h = (h.round() as u32).min(max_h).max(1);
    let x = (x.round().max(0.0) as u32).min(max_w.saturating_sub(w));
    let y = (y.round().max(0.0) as u32).min(max_h.saturating_sub(h));
    Rect { x, y, w, h }
}

/// map a detection from letterboxed crop space back to main-stream pixels.
///
/// the detector's box is far better framed than the motion region that produced
/// it: the gate reports "these pixels changed", which routinely includes a
/// garage door and a shadow. harvesting the detector's box instead is the
/// difference between a picture of a vehicle and a picture of a street.
pub fn detection_to_main(
    corners: (f32, f32, f32, f32),
    crop_main: Rect,
    size: u32,
    main_w: u32,
    main_h: u32,
) -> Rect {
    let scale =
        (size as f32 / crop_main.w.max(1) as f32).min(size as f32 / crop_main.h.max(1) as f32);
    let (dw, dh) = (crop_main.w as f32 * scale, crop_main.h as f32 * scale);
    let (ox, oy) = ((size as f32 - dw) / 2.0, (size as f32 - dh) / 2.0);
    let undo =
        |v: f32, off: f32, origin: u32| origin as f32 + (v - off) / scale.max(f32::MIN_POSITIVE);

    let (x1, y1, x2, y2) = corners;
    let mx1 = undo(x1, ox, crop_main.x).clamp(0.0, main_w as f32);
    let my1 = undo(y1, oy, crop_main.y).clamp(0.0, main_h as f32);
    let mx2 = undo(x2, ox, crop_main.x).clamp(0.0, main_w as f32);
    let my2 = undo(y2, oy, crop_main.y).clamp(0.0, main_h as f32);
    Rect {
        x: mx1 as u32,
        y: my1 as u32,
        w: (mx2 - mx1).max(1.0) as u32,
        h: (my2 - my1).max(1.0) as u32,
    }
}

/// the same box expressed in gate coordinates, for drawing on the preview.
/// the preview shows the gate view because that is the scene a person
/// recognises, not a disembodied crop.
pub fn main_to_gate(r: Rect, gate_w: u32, gate_h: u32, main_w: u32, main_h: u32) -> Rect {
    let (sx, sy) = scale((main_w, main_h), (gate_w, gate_h));
    Rect {
        x: (r.x as f32 * sx) as u32,
        y: (r.y as f32 * sy) as u32,
        w: ((r.w as f32 * sx) as u32).max(1),
        h: ((r.h as f32 * sy) as u32).max(1),
    }
}

/// extract luma from an interleaved rgb frame, into a reused buffer.
///
/// the gate runs on luma; the preview wants the colour. taking one stream in
/// colour and deriving the other costs about a millisecond per frame here,
/// which is far cheaper than a second decode or a grey preview.
pub fn rgb_to_luma(rgb: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(rgb.len() / 3);
    // rec.601 weights in integer arithmetic: the gate compares luma against
    // itself, so exact colourimetry matters less than being consistent and fast.
    for px in rgb.chunks_exact(3) {
        let y = (77 * px[0] as u32 + 150 * px[1] as u32 + 29 * px[2] as u32) >> 8;
        out.push(y as u8);
    }
}

/// shrink an rgb frame, for showing the main stream in the preview.
///
/// the gate frame is luma-only, because differencing needs nothing else, so
/// previewing it gives a grey picture. the main-stream frame is already decoded
/// for crops and is true 16:9, so downscaling it gives both colour and correct
/// geometry. at preview rates this costs a few milliseconds.
pub fn downscale(rgb: &[u8], w: u32, h: u32, dst_w: u32, dst_h: u32) -> Option<Vec<u8>> {
    use fast_image_resize::images::Image;
    use fast_image_resize::{PixelType, ResizeOptions, Resizer};

    let src = Image::from_vec_u8(w, h, rgb.to_vec(), PixelType::U8x3).ok()?;
    let mut dst = Image::new(dst_w, dst_h, PixelType::U8x3);
    Resizer::new()
        .resize(&src, &mut dst, &ResizeOptions::default())
        .ok()?;
    Some(dst.buffer().to_vec())
}

/// copy a sub-rectangle out of an interleaved rgb buffer.
pub fn extract(src: &[u8], src_w: u32, src_h: u32, r: Rect) -> Vec<u8> {
    debug_assert!(
        r.x + r.w <= src_w && r.y + r.h <= src_h,
        "crop outside frame"
    );
    let mut out = Vec::with_capacity((r.w * r.h * 3) as usize);
    for row in r.y..(r.y + r.h).min(src_h) {
        let start = ((row * src_w + r.x) * 3) as usize;
        let end = start + (r.w * 3) as usize;
        out.extend_from_slice(&src[start..end.min(src.len())]);
    }
    out
}

/// fit an rgb crop into a square model input, preserving aspect and padding.
///
/// aspect must be preserved: the detector was trained on undistorted objects,
/// and stretching a crop to a square would deform every vehicle in it.
pub fn letterbox(crop: &[u8], w: u32, h: u32, size: u32) -> Vec<u8> {
    use fast_image_resize::images::Image;
    use fast_image_resize::{PixelType, ResizeOptions, Resizer};

    let scale = (size as f32 / w as f32).min(size as f32 / h as f32);
    let (dw, dh) = (
        ((w as f32 * scale).round() as u32).clamp(1, size),
        ((h as f32 * scale).round() as u32).clamp(1, size),
    );

    let src = match Image::from_vec_u8(w, h, crop.to_vec(), PixelType::U8x3) {
        Ok(i) => i,
        Err(_) => return vec![0; (size * size * 3) as usize],
    };
    let mut dst = Image::new(dw, dh, PixelType::U8x3);
    if Resizer::new()
        .resize(&src, &mut dst, &ResizeOptions::default())
        .is_err()
    {
        return vec![0; (size * size * 3) as usize];
    }

    // centre it on a black field, the same geometry the ingest filter produces.
    let mut out = vec![0u8; (size * size * 3) as usize];
    let (ox, oy) = ((size - dw) / 2, (size - dh) / 2);
    let scaled = dst.buffer();
    for row in 0..dh {
        let from = (row * dw * 3) as usize;
        let to = (((row + oy) * size + ox) * 3) as usize;
        out[to..to + (dw * 3) as usize].copy_from_slice(&scaled[from..from + (dw * 3) as usize]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crops() -> crate::config::CropCfg {
        crate::config::CropCfg::default()
    }
    const MIN_CROP_PX: u32 = crate::config::MIN_CROP_PX;
    const MAX_CROP_PX: u32 = crate::config::MAX_CROP_PX;
    const GATE_W: u32 = 640;
    const GATE_H: u32 = 480;
    const MAIN_W: u32 = 2560;
    const MAIN_H: u32 = 1440;
    // derived from the two sizes rather than compiled in, which is the whole
    // point of this module: the mapping follows the streams (r5.6).
    const GATE_TO_MAIN_X: f32 = MAIN_W as f32 / GATE_W as f32;
    const GATE_TO_MAIN_Y: f32 = MAIN_H as f32 / GATE_H as f32;

    /// the mapping a camera swap has to survive: a 4096x1856 main and a
    /// 1024x464 sub are the same field of view at one factor per axis, and a
    /// 640x480 sub of the same main is the same field of view at two.
    #[test]
    fn the_scale_follows_whatever_the_two_streams_are() {
        let near = |got: (f32, f32), want: (f32, f32)| {
            (got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4
        };
        assert!(near(scale((1024, 464), (4096, 1856)), (4.0, 4.0)));
        assert!(near(scale((640, 480), (4096, 1856)), (6.4, 1856.0 / 480.0)));
        assert!(near(scale((1280, 720), (2560, 1440)), (2.0, 2.0)));
    }

    /// the same vehicle lands in the same place at any pair of sizes, which is
    /// what a box drawn over a car instead of above it comes down to.
    #[test]
    fn a_centre_stays_the_centre_at_any_stream_size() {
        for (gate, main) in [
            ((1024u32, 464u32), (4096u32, 1856u32)),
            ((640, 480), (4096, 1856)),
            ((1280, 720), (2560, 1440)),
        ] {
            let region = Rect {
                x: gate.0 / 2 - 10,
                y: gate.1 / 2 - 10,
                w: 20,
                h: 20,
            };
            let m = to_main(region, gate.0, gate.1, main.0, main.1, &crops());
            let (cx, cy) = (m.x as f32 + m.w as f32 / 2.0, m.y as f32 + m.h as f32 / 2.0);
            let (want_x, want_y) = (main.0 as f32 / 2.0, main.1 as f32 / 2.0);
            assert!(
                (cx - want_x).abs() <= 2.0 && (cy - want_y).abs() <= 2.0,
                "{}x{} -> {}x{} landed at {cx},{cy}, not {:?}",
                gate.0,
                gate.1,
                main.0,
                main.1,
                (want_x, want_y)
            );
        }
    }

    #[test]
    fn a_gate_region_maps_to_the_same_place_in_main_stream_pixels() {
        // dead centre of the 640x480 gate frame is (320, 240); dead centre of
        // the 2560x1440 main frame is (1280, 720). one must map onto the other.
        let region = Rect {
            x: 310,
            y: 230,
            w: 20,
            h: 20,
        };
        let m = to_main(region, GATE_W, GATE_H, MAIN_W, MAIN_H, &crops());
        let (cx, cy) = (m.x + m.w / 2, m.y + m.h / 2);
        assert!((cx as i64 - 1280).abs() <= 2, "x centre drifted: {cx}");
        assert!((cy as i64 - 720).abs() <= 2, "y centre drifted: {cy}");
    }

    /// the point of the whole module: a far-side vehicle too small for the
    /// detector in the gate stream becomes comfortably large in a main crop.
    #[test]
    fn a_far_side_vehicle_gains_enough_pixels_to_be_detectable() {
        // ~50px wide in the gate stream, which measured as unreliable.
        let region = Rect {
            x: 300,
            y: 230,
            w: 50,
            h: 30,
        };
        let m = to_main(region, GATE_W, GATE_H, MAIN_W, MAIN_H, &crops());

        // 50 * 4.0 = 200px of vehicle, plus context. well past the 64px floor.
        assert!(m.w >= 200, "lost resolution instead of gaining it: {m:?}");
        let vehicle_px = 50.0 * GATE_TO_MAIN_X;
        assert!(vehicle_px >= 64.0, "still under the detection floor");
    }

    #[test]
    fn context_is_added_around_the_region() {
        let region = Rect {
            x: 100,
            y: 100,
            w: 100,
            h: 100,
        };
        let m = to_main(region, GATE_W, GATE_H, MAIN_W, MAIN_H, &crops());
        assert!(
            m.w > (100.0 * GATE_TO_MAIN_X) as u32,
            "no horizontal context"
        );
        assert!(m.h > (100.0 * GATE_TO_MAIN_Y) as u32, "no vertical context");
    }

    #[test]
    fn crops_never_escape_the_frame() {
        for region in [
            Rect {
                x: 0,
                y: 0,
                w: 40,
                h: 40,
            }, // top left
            Rect {
                x: 600,
                y: 440,
                w: 40,
                h: 40,
            }, // bottom right
            Rect {
                x: 0,
                y: 0,
                w: 640,
                h: 480,
            }, // the whole frame
        ] {
            let m = to_main(region, GATE_W, GATE_H, MAIN_W, MAIN_H, &crops());
            assert!(m.x + m.w <= MAIN_W, "{m:?} runs off the right");
            assert!(m.y + m.h <= MAIN_H, "{m:?} runs off the bottom");
            assert!(m.w > 0 && m.h > 0, "{m:?} is empty");
        }
    }

    /// the detector input is square, so anything else is padded and shrunk.
    #[test]
    fn crops_are_square_whatever_shape_the_motion_was() {
        for region in [
            Rect {
                x: 100,
                y: 60,
                w: 300,
                h: 40,
            }, // a wide sliver
            Rect {
                x: 100,
                y: 60,
                w: 30,
                h: 300,
            }, // a tall sliver
            Rect {
                x: 200,
                y: 200,
                w: 80,
                h: 80,
            }, // already square
        ] {
            let m = to_main(region, GATE_W, GATE_H, MAIN_W, MAIN_H, &crops());
            assert_eq!(m.w, m.h, "crop is not square for {region:?}: {m:?}");
        }
    }

    /// the regression that made the detector report `airplane` and `skis`: a
    /// tall motion region became a full-height strip of the main stream, the
    /// vehicle inside shrank to a smudge once letterboxed into 640x640, and
    /// real cars stopped being detected at all.
    #[test]
    fn large_regions_are_capped_so_the_vehicle_stays_big_in_the_crop() {
        // 350px tall in the gate is 1050px in the main stream before context.
        let region = Rect {
            x: 100,
            y: 60,
            w: 300,
            h: 350,
        };
        let m = to_main(region, GATE_W, GATE_H, MAIN_W, MAIN_H, &crops());

        assert!(m.h <= MAX_CROP_PX, "crop is a full-height strip: {m:?}");
        assert!(m.w <= MAX_CROP_PX, "crop is a full-width strip: {m:?}");
        // and it still sits over the region it came from.
        let centre_y = m.y + m.h / 2;
        assert!(
            (centre_y as i64 - ((region.y + region.h / 2) as f32 * GATE_TO_MAIN_Y) as i64).abs()
                < 40,
            "cap moved the crop off the region: {m:?}"
        );
    }

    #[test]
    fn tiny_regions_are_grown_to_a_usable_size() {
        let m = to_main(
            Rect {
                x: 320,
                y: 240,
                w: 2,
                h: 2,
            },
            GATE_W,
            GATE_H,
            MAIN_W,
            MAIN_H,
            &crops(),
        );
        assert!(
            m.w >= MIN_CROP_PX && m.h >= MIN_CROP_PX,
            "{m:?} too small to use"
        );
    }

    /// a box the detector found in a crop must land back where it really is.
    /// this is three transforms deep and every one of them is invisible when
    /// wrong: the harvest would quietly collect mis-framed crops.
    #[test]
    fn a_detection_maps_back_to_where_it_came_from() {
        let region = Rect {
            x: 300,
            y: 230,
            w: 50,
            h: 30,
        };
        let crop_main = to_main(region, GATE_W, GATE_H, MAIN_W, MAIN_H, &crops());
        let size = 640u32;

        // a detection filling the whole letterboxed image must map back to the
        // whole crop, give or take rounding.
        let scale = (size as f32 / crop_main.w as f32).min(size as f32 / crop_main.h as f32);
        let (dw, dh) = (crop_main.w as f32 * scale, crop_main.h as f32 * scale);
        let (ox, oy) = ((size as f32 - dw) / 2.0, (size as f32 - dh) / 2.0);

        let got = detection_to_main((ox, oy, ox + dw, oy + dh), crop_main, size, MAIN_W, MAIN_H);
        assert!(
            (got.x as i64 - crop_main.x as i64).abs() <= 2,
            "{got:?} vs {crop_main:?}"
        );
        assert!(
            (got.w as i64 - crop_main.w as i64).abs() <= 2,
            "{got:?} vs {crop_main:?}"
        );
    }

    #[test]
    fn a_detection_in_the_crop_centre_lands_in_the_region_centre() {
        let region = Rect {
            x: 300,
            y: 230,
            w: 50,
            h: 30,
        };
        let crop_main = to_main(region, GATE_W, GATE_H, MAIN_W, MAIN_H, &crops());
        let size = 640u32;
        // a small box at the centre of the model input.
        let c = size as f32 / 2.0;
        let m = detection_to_main(
            (c - 10.0, c - 10.0, c + 10.0, c + 10.0),
            crop_main,
            size,
            MAIN_W,
            MAIN_H,
        );
        let g = main_to_gate(m, GATE_W, GATE_H, MAIN_W, MAIN_H);

        let (rcx, rcy) = (region.x + region.w / 2, region.y + region.h / 2);
        let (gcx, gcy) = (g.x + g.w / 2, g.y + g.h / 2);
        assert!(
            (gcx as i64 - rcx as i64).abs() <= 3,
            "x off: {gcx} vs {rcx}"
        );
        assert!(
            (gcy as i64 - rcy as i64).abs() <= 3,
            "y off: {gcy} vs {rcy}"
        );
    }

    #[test]
    fn luma_is_derived_from_rgb() {
        let mut out = Vec::new();
        // pure white, pure black, and a saturated green: green dominates luma.
        rgb_to_luma(&[255, 255, 255, 0, 0, 0, 0, 255, 0], &mut out);
        assert_eq!(out.len(), 3);
        assert!(out[0] > 250, "white should be bright: {}", out[0]);
        assert_eq!(out[1], 0, "black should be zero");
        assert!(
            (130..=160).contains(&out[2]),
            "green carries most of luma: {}",
            out[2]
        );
    }

    #[test]
    fn luma_reuses_the_buffer_it_is_given() {
        let mut out = vec![9; 99];
        rgb_to_luma(&[10, 10, 10], &mut out);
        assert_eq!(out.len(), 1, "stale contents were not cleared");
    }

    /// crop 6 of the first harvest was a sliver of a jeep, cut off where the
    /// crop ended. a classifier trained on that learns about slivers.
    #[test]
    fn a_box_against_the_crop_edge_is_recognised_as_clipped() {
        let container = Rect {
            x: 100,
            y: 100,
            w: 200,
            h: 200,
        };
        let flush_left = Rect {
            x: 100,
            y: 150,
            w: 50,
            h: 50,
        };
        let inside = Rect {
            x: 140,
            y: 140,
            w: 100,
            h: 100,
        };

        assert!(
            touches_edge(flush_left, container, 2),
            "clipped box accepted"
        );
        assert!(
            !touches_edge(inside, container, 2),
            "whole vehicle rejected"
        );
    }

    /// the protected vehicle sits in the same place all day, and traffic drives
    /// through its bounding box. a generic motion filter cannot exclude it,
    /// because pixels inside the box genuinely change; only knowing where it
    /// parks can (r9.3).
    #[test]
    fn the_protected_space_is_recognised_by_overlap() {
        let van = Rect {
            x: 395,
            y: 129,
            w: 245,
            h: 309,
        };
        let the_van_itself = Rect {
            x: 400,
            y: 140,
            w: 230,
            h: 290,
        };
        let car_in_the_road = Rect {
            x: 120,
            y: 300,
            w: 140,
            h: 70,
        };

        assert!(
            covered_fraction(the_van_itself, van) > 0.9,
            "should be the van"
        );
        assert_eq!(covered_fraction(car_in_the_road, van), 0.0, "not the van");
    }

    #[test]
    fn expanding_a_box_adds_context_and_stays_inside_the_frame() {
        let r = Rect {
            x: 500,
            y: 500,
            w: 200,
            h: 100,
        };
        let g = expand(r, 0.25, MAIN_W, MAIN_H);
        assert!(g.w > r.w && g.h > r.h, "no context added: {g:?}");
        assert!(g.x <= r.x && g.y <= r.y, "did not grow outwards: {g:?}");

        // a box in the corner grows only where there is room.
        let corner = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
        };
        let c = expand(corner, 0.5, MAIN_W, MAIN_H);
        assert_eq!((c.x, c.y), (0, 0));
        assert!(c.x + c.w <= MAIN_W && c.y + c.h <= MAIN_H);

        // and a box filling the frame cannot grow past it.
        let full = Rect {
            x: 0,
            y: 0,
            w: MAIN_W,
            h: MAIN_H,
        };
        let f = expand(full, 0.5, MAIN_W, MAIN_H);
        assert_eq!((f.w, f.h), (MAIN_W, MAIN_H));
    }

    #[test]
    fn extract_copies_the_requested_pixels() {
        // 4x2 rgb image where each pixel encodes its own x in the red channel.
        let (w, h) = (4u32, 2u32);
        let mut src = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                src[((y * w + x) * 3) as usize] = (x * 10) as u8;
            }
        }
        let got = extract(
            &src,
            w,
            h,
            Rect {
                x: 1,
                y: 0,
                w: 2,
                h: 2,
            },
        );
        assert_eq!(got.len(), (2 * 2 * 3) as usize);
        assert_eq!(got[0], 10, "first column wrong");
        assert_eq!(got[3], 20, "second column wrong");
        assert_eq!(got[6], 10, "second row did not start at x=1");
    }

    #[test]
    fn letterbox_preserves_aspect_and_fills_the_square() {
        // a wide crop must be centred vertically with black bands, not stretched.
        let (w, h) = (200u32, 100u32);
        let crop = vec![255u8; (w * h * 3) as usize];
        let out = letterbox(&crop, w, h, 640);
        assert_eq!(out.len(), (640 * 640 * 3) as usize);

        let row_is_black = |row: u32| {
            let s = (row * 640 * 3) as usize;
            out[s..s + 640 * 3].iter().all(|&b| b == 0)
        };
        assert!(row_is_black(0), "top band should be padding");
        assert!(row_is_black(639), "bottom band should be padding");
        assert!(!row_is_black(320), "middle row should carry image");
    }

    #[test]
    fn letterbox_survives_a_degenerate_crop() {
        // a zero-area crop must not panic; it is a bug upstream, not a crash here.
        let out = letterbox(&[], 0, 0, 640);
        assert_eq!(out.len(), (640 * 640 * 3) as usize);
    }
}
