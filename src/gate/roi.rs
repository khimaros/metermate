//! the region of interest: which pixels the gate may look at.
//!
//! the camera overlooks a block strung with utility wires, and they move in
//! wind. excluding them in `mark_changed`, before any pixel reaches the
//! background model, is cheaper than filtering regions afterwards.
//!
//! coordinates are gate pixels: the frame the polygon is drawn on and the frame
//! the mask applies to. nothing here maps to main-stream pixels.

/// even-odd ray casting, against pixel centres so a polygon drawn on a pixel
/// boundary does not include a row depending on which way rounding fell.
pub fn contains(poly: &[[u32; 2]], x: u32, y: u32) -> bool {
    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = (poly[i][0] as f32, poly[i][1] as f32);
        let (xj, yj) = (poly[j][0] as f32, poly[j][1] as f32);
        if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// rasterise a polygon into a per-pixel mask, or `None` if it bounds no area.
/// `None` rather than an empty mask, so the caller decides what no roi means
/// instead of a stray vertex quietly masking the whole frame away.
pub fn mask(poly: &[[u32; 2]], width: u32, height: u32, min_vertices: usize) -> Option<Vec<bool>> {
    if poly.len() < min_vertices {
        return None;
    }
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height {
        for x in 0..width {
            out.push(contains(poly, x, y));
        }
    }
    Some(out)
}

/// what share of the frame a mask admits. worth logging: an roi covering
/// nothing looks exactly like a street where nothing happens.
pub fn covered(mask: &[bool]) -> f32 {
    if mask.is_empty() {
        return 0.0;
    }
    mask.iter().filter(|m| **m).count() as f32 / mask.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_VERTICES: usize = crate::config::ROI_MIN_VERTICES;
    const SUSPICIOUSLY_SMALL: f32 = crate::config::ROI_SUSPICIOUSLY_SMALL;

    #[test]
    fn a_square_contains_its_inside_and_not_its_outside() {
        let sq = vec![[10, 10], [30, 10], [30, 30], [10, 30]];
        assert!(contains(&sq, 20, 20));
        assert!(contains(&sq, 10, 10), "first pixel of the polygon");
        assert!(!contains(&sq, 5, 20));
        assert!(!contains(&sq, 20, 35));
        assert!(!contains(&sq, 30, 30), "far corner is exclusive");
    }

    /// the street runs diagonally with a building cutting in, so the polygon
    /// that matters is concave and a bounding box would re-admit the sky.
    #[test]
    fn a_concave_polygon_excludes_its_notch() {
        let p = vec![[0, 0], [100, 0], [100, 50], [50, 50], [50, 100], [0, 100]];
        assert!(contains(&p, 25, 25), "inside the arm");
        assert!(contains(&p, 25, 75), "inside the leg");
        assert!(!contains(&p, 75, 75), "the notch must stay out");
    }

    #[test]
    fn the_mask_matches_the_polygon_area() {
        let m = mask(&[[0, 0], [10, 0], [10, 10], [0, 10]], 20, 20, MIN_VERTICES).unwrap();
        assert_eq!(m.len(), 400);
        assert_eq!(m.iter().filter(|v| **v).count(), 100);
        assert!(m[0], "origin is inside");
        assert!(!m[19], "far end of the first row is outside");
    }

    /// a polygon that bounds no area must not become a mask that hides the
    /// frame: the symptom is a camera detecting nothing and reporting no error.
    #[test]
    fn a_degenerate_polygon_is_no_roi_rather_than_an_empty_one() {
        assert!(mask(&[], 10, 10, MIN_VERTICES).is_none());
        assert!(mask(&[[1, 1]], 10, 10, MIN_VERTICES).is_none());
        assert!(mask(&[[1, 1], [5, 5]], 10, 10, MIN_VERTICES).is_none());
        assert!(mask(&[[1, 1], [5, 1], [5, 5]], 10, 10, MIN_VERTICES).is_some());
    }

    #[test]
    fn covered_reports_the_share_of_the_frame_watched() {
        let m = mask(&[[0, 0], [5, 0], [5, 10], [0, 10]], 10, 10, MIN_VERTICES).unwrap();
        assert!((covered(&m) - 0.5).abs() < 1e-6, "{}", covered(&m));
        assert_eq!(covered(&[]), 0.0);
    }

    /// coordinates from a different frame size: valid, and covering nothing.
    #[test]
    fn an_roi_far_off_the_frame_covers_almost_nothing() {
        let p = vec![[600, 600], [700, 600], [700, 700], [600, 700]];
        let m = mask(&p, 640, 480, MIN_VERTICES).unwrap();
        assert!(covered(&m) < SUSPICIOUSLY_SMALL, "{}", covered(&m));
    }
}
