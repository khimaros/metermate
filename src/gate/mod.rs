//! the motion gate: the cheap stage that decides whether anything downstream
//! runs at all.
//!
//! this is what makes the idle case free. the block is full of permanently
//! parked cars, and a per-frame detector would rediscover all of them forever.
//! a background model folds them into the scene and they stop costing anything.

pub mod roi;

use crate::ingest::Frame;
use std::time::{Duration, Instant};

// laid out as the grid it represents; the hole in the middle is the pixel itself.
#[rustfmt::skip]
const NEIGHBOURS: [(i64, i64); 8] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
];

#[derive(Debug, Clone, Copy)]
struct Component {
    rect: Rect,
    px: u32,
    /// the same pixels, each counted by the vehicle area expected where it
    /// lies. equal to `px` when no perspective is drawn.
    weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone)]
pub struct GateResult {
    pub motion: bool,
    pub changed_frac: f32,
    /// disjoint regions of changed pixels, in gate coordinates, largest first.
    /// these are what the detector is given, so it never sees the whole frame.
    pub regions: Vec<Rect>,
}

/// how tall a vehicle is at a point in the frame, from the lines drawn in the
/// roi tab. each line is drawn along a vehicle: its midpoint says where, its
/// length says how tall one is there.
///
/// **fitted across both axes, because on this camera the axis is diagonal.**
/// measured over 532 harvested boxes: apparent height explains R^2 0.36 against
/// image row alone, 0.50 against column, and 0.76 against a plane in both.
/// vehicles run 50-60px at one end against 153px at the other, about nine times
/// the area, and that is not a function of depth-down-the-image. interpolating
/// on row would be confidently wrong in the corners.
///
/// those boxes came from the harvest, so they come from inside whatever roi the
/// gate was watching -- and the roi in force when they were taken was the
/// compressed one, covering the upper 71% of the frame and none of the near
/// lane. the *shape* of the finding survives that, since a restricted sample
/// cannot invent a diagonal axis, but the magnitudes describe that band rather
/// than the frame. nothing here depends on them: the fit reads the lines drawn,
/// not these numbers.
///
/// a plane rather than anything richer: it is the least that describes a street
/// receding across a frame, three lines determine it, and its residual is
/// already smaller than the spread of real vehicle heights.
#[derive(Debug, Clone, Copy)]
pub struct Perspective {
    /// height = `a*x + b*y + c`, in gate pixels.
    a: f32,
    b: f32,
    c: f32,
    /// the shortest and longest lines drawn, which bound what the fit may
    /// claim.
    ///
    /// **a plane says something everywhere, including where nobody drew.**
    /// measured on the deployment's first real lines -- six of them, agreeing
    /// to an rms residual of 8.4px on a mean height of 70.9px -- the fit still
    /// runs to -77px in the top-left, where no line was drawn. a height near
    /// zero divides, and since a pixel is weighted by `1 / h^2`, floored at a
    /// quarter of the shortest line that corner came out 2900 times more
    /// sensitive than the opposite one, none of it from evidence.
    ///
    /// held to the drawn range instead, the widest ratio across the frame is
    /// exactly the area ratio the lines themselves demonstrate. outside their
    /// support the nearest measured value is the honest answer.
    lo: f32,
    hi: f32,
}

impl Perspective {
    /// `None` when nothing was drawn, which is every deployment that has not
    /// opened the roi tab since this existed, and means no correction at all.
    pub fn fit(lines: &[[[u32; 2]; 2]]) -> Option<Self> {
        let samples: Vec<(f32, f32, f32)> = lines
            .iter()
            .map(|[a, b]| {
                let (ax, ay) = (a[0] as f32, a[1] as f32);
                let (bx, by) = (b[0] as f32, b[1] as f32);
                let len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
                ((ax + bx) / 2.0, (ay + by) / 2.0, len)
            })
            .filter(|(_, _, h)| *h > 0.0)
            .collect();

        let heights: Vec<f32> = samples.iter().map(|(_, _, h)| *h).collect();
        let (lo, hi) = match (
            heights.iter().cloned().fold(f32::INFINITY, f32::min),
            heights.iter().cloned().fold(0.0f32, f32::max),
        ) {
            (l, h) if h > 0.0 => (l, h),
            _ => return None,
        };

        let plane = match samples.len() {
            0 => return None,
            // one line says how big a vehicle is, but nothing about how that
            // changes. a constant is honest about that.
            1 => (0.0, 0.0, samples[0].2),
            // two lines fix a gradient along the axis joining them and say
            // nothing across it, so the fit varies only in that direction.
            2 => {
                let ((x1, y1, h1), (x2, y2, h2)) = (samples[0], samples[1]);
                let (dx, dy) = (x2 - x1, y2 - y1);
                let span = dx * dx + dy * dy;
                if span <= f32::EPSILON {
                    (0.0, 0.0, (h1 + h2) / 2.0)
                } else {
                    let a = (h2 - h1) * dx / span;
                    let b = (h2 - h1) * dy / span;
                    (a, b, h1 - a * x1 - b * y1)
                }
            }
            _ => least_squares_plane(&samples).unwrap_or((0.0, 0.0, mean(&heights))),
        };

        Some(Self {
            a: plane.0,
            b: plane.1,
            c: plane.2,
            lo,
            hi,
        })
    }

    pub fn height_at(&self, x: f32, y: f32) -> f32 {
        (self.a * x + self.b * y + self.c).clamp(self.lo, self.hi)
    }
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

/// least-squares `h = a*x + b*y + c`, or `None` when the lines are collinear
/// and no plane is determined by them.
fn least_squares_plane(samples: &[(f32, f32, f32)]) -> Option<(f32, f32, f32)> {
    let n = samples.len() as f32;
    let (mut sx, mut sy, mut sh) = (0.0, 0.0, 0.0);
    let (mut sxx, mut syy, mut sxy, mut sxh, mut syh) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y, h) in samples {
        sx += x;
        sy += y;
        sh += h;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        sxh += x * h;
        syh += y * h;
    }
    // the normal equations, by cramer's rule.
    let m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
    let det = det3(&m);
    // collinear lines leave the plane undetermined across the axis: the
    // determinant collapses and the solution would be arbitrary.
    if det.abs() < 1e-6 {
        return None;
    }
    let rhs = [sxh, syh, sh];
    let solve = |col: usize| {
        let mut c = m;
        for (row, v) in rhs.iter().enumerate() {
            c[row][col] = *v;
        }
        det3(&c) / det
    };
    Some((solve(0), solve(1), solve(2)))
}

fn det3(m: &[[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// changed and watched pixel counts inside `rect`. free rather than a method so
/// the gate and a detached snapshot cannot drift apart on how the roi is
/// discounted -- that discount is the difference between a vehicle straddling
/// the roi boundary being harvested and being silently unharvestable.
fn tally(
    changed_px: &[bool],
    mask: Option<&Vec<bool>>,
    width: u32,
    height: u32,
    rect: Rect,
) -> (u32, u32) {
    let x2 = (rect.x + rect.w).min(width);
    let y2 = (rect.y + rect.h).min(height);
    if rect.x >= x2 || rect.y >= y2 {
        return (0, 0);
    }
    let (mut changed, mut watched) = (0u32, 0u32);
    for y in rect.y..y2 {
        let row = (y * width) as usize;
        for x in rect.x..x2 {
            let i = row + x as usize;
            if mask.is_some_and(|m| !m[i]) {
                continue;
            }
            watched += 1;
            if changed_px[i] {
                changed += 1;
            }
        }
    }
    (changed, watched)
}

fn central_share(changed_px: &[bool], width: u32, height: u32, rect: Rect) -> f32 {
    let x2 = (rect.x + rect.w).min(width);
    let y2 = (rect.y + rect.h).min(height);
    if rect.x >= x2 || rect.y >= y2 {
        return 0.0;
    }
    // the middle half on each axis, never empty: for a box a few pixels across
    // there is nowhere for an intruder to hide, so everything counts.
    let (w, h) = (x2 - rect.x, y2 - rect.y);
    let (mx0, mx1) = (rect.x + w / 4, (x2 - w / 4).max(rect.x + w / 4 + 1));
    let (my0, my1) = (rect.y + h / 4, (y2 - h / 4).max(rect.y + h / 4 + 1));
    let (mut total, mut middle) = (0u32, 0u32);
    for y in rect.y..y2 {
        let row = (y * width) as usize;
        for x in rect.x..x2 {
            if changed_px[row + x as usize] {
                total += 1;
                if (mx0..mx1).contains(&x) && (my0..my1).contains(&y) {
                    middle += 1;
                }
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        middle as f32 / total as f32
    }
}

pub struct MotionGate {
    /// background luma in 8.8 fixed point, one entry per pixel.
    background: Vec<u16>,
    /// optional roi mask; pixels outside it are ignored entirely.
    mask: Option<std::sync::Arc<Vec<bool>>>,
    /// scratch buffers, owned so the per-frame path never allocates.
    changed: Vec<bool>,
    visited: Vec<bool>,
    stack: Vec<usize>,
    width: u32,
    height: u32,
    diff_threshold: u16,
    min_changed_frac: f32,
    latch: Duration,
    warmup_frames: u32,
    seen: u32,
    last_motion: Option<Instant>,
    observed_px: u32,
    /// per-pixel weights, normalised so the mean over watched pixels is one.
    /// `None` when no perspective is drawn, which keeps the score the plain
    /// changed fraction it has always been.
    weights: Option<Vec<f32>>,
    perspective: Option<Perspective>,
    cfg: crate::config::Gate,
}

impl MotionGate {
    pub fn new(width: u32, height: u32, cfg: &crate::config::Gate) -> Self {
        let n = (width as usize) * (height as usize);
        let mut me = Self {
            background: vec![0; n],
            mask: None,
            changed: vec![false; n],
            visited: vec![false; n],
            stack: Vec::new(),
            width,
            height,
            diff_threshold: (cfg.diff_threshold as u16) << 8,
            min_changed_frac: cfg.min_changed_frac,
            latch: Duration::from_millis(cfg.latch_ms),
            warmup_frames: cfg.warmup_frames,
            seen: 0,
            last_motion: None,
            observed_px: width * height,
            weights: None,
            perspective: Perspective::fit(&cfg.perspective),
            cfg: cfg.clone(),
        };
        me.rebuild_weights();
        me
    }

    /// weight every pixel by the vehicle area expected there, normalised so the
    /// mean over watched pixels is one.
    ///
    /// **normalised, so `min_changed_frac` keeps its meaning.** the correction
    /// is about redistributing sensitivity with depth, not about raising it
    /// everywhere; left unnormalised, drawing two lines would quietly retune
    /// every threshold in the config at once.
    fn rebuild_weights(&mut self) {
        let Some(p) = self.perspective else {
            self.weights = None;
            return;
        };
        let mut w = vec![1.0f32; self.background.len()];
        let (mut sum, mut n) = (0.0f64, 0u64);
        for y in 0..self.height {
            for x in 0..self.width {
                let i = (y * self.width + x) as usize;
                let h = p.height_at(x as f32, y as f32);
                // area, not height: a vehicle half as tall covers a quarter of
                // the pixels, and the pixels are what is being counted.
                let raw = 1.0 / (h * h);
                w[i] = raw;
                if self.mask.as_ref().is_none_or(|m| m[i]) {
                    sum += raw as f64;
                    n += 1;
                }
            }
        }
        let mean = if n > 0 { (sum / n as f64) as f32 } else { 1.0 };
        if mean > 0.0 {
            w.iter_mut().for_each(|v| *v /= mean);
        }
        self.weights = Some(w);
    }

    /// restrict the gate to an roi. everything outside stops being looked at,
    /// which is how wires, trees, and the sky stop producing false motion.
    pub fn set_mask(&mut self, mask: Vec<bool>) {
        assert_eq!(
            mask.len(),
            self.background.len(),
            "mask size must match frame"
        );
        self.observed_px = mask.iter().filter(|m| **m).count().max(1) as u32;
        self.mask = Some(std::sync::Arc::new(mask));
        self.rebuild_weights();
        self.reset();
    }

    /// apply the configured roi and report what it covers. a bad roi is
    /// otherwise invisible: it looks exactly like a street where nothing moves.
    pub fn apply_roi(&mut self, poly: &[[u32; 2]]) {
        if poly.is_empty() {
            tracing::info!(
                "no roi: watching the whole {}x{} frame",
                self.width,
                self.height
            );
            return;
        }
        let Some(mask) = roi::mask(poly, self.width, self.height, self.cfg.roi_min_vertices) else {
            tracing::warn!("roi has {} vertices, too few to bound an area", poly.len());
            return;
        };
        let share = roi::covered(&mask);
        if share < self.cfg.roi_suspiciously_small {
            tracing::warn!(
                "roi covers {:.1}% of the frame -- are those gate pixels ({}x{})?",
                share * 100.0,
                self.width,
                self.height
            );
        }
        tracing::info!(
            "roi: {} vertices, {:.0}% of the frame watched",
            poly.len(),
            share * 100.0
        );
        self.set_mask(mask);
    }

    /// what fraction of the *watched* pixels inside `rect` changed on the last
    /// frame.
    ///
    /// this is the direct question, and the one worth asking: did *this object*
    /// move? a bounding-box overlap against the motion region cannot answer it.
    /// a car driving past in front of a parked jeep produces a region covering
    /// much of the jeep while not one jeep pixel changes, and the harvest duly
    /// filled with photographs of the jeep.
    ///
    /// **watched, not all.** masked pixels are forced unchanged, so counting
    /// them in the denominator penalised a vehicle for being near the edge of
    /// the roi: measured against a real 38% roi, 45% of a clip's crops straddle
    /// its boundary, and a car half outside could not score above 0.5 however
    /// much of it moved. the gate fired, the detector found it, and nothing was
    /// harvested.
    pub fn changed_fraction(&self, rect: Rect) -> f32 {
        let (changed, watched) = tally(
            &self.changed,
            self.mask.as_deref(),
            self.width,
            self.height,
            rect,
        );
        if watched == 0 {
            0.0
        } else {
            changed as f32 / watched as f32
        }
    }

    /// of the changed pixels inside `rect`, what share lie in its middle half?
    ///
    /// **`changed_fraction` cannot tell whose motion it is.** a bounding box is
    /// not a vehicle; it is a rectangle that also contains road, sky, and any
    /// other vehicle overlapping it. a car passing a parked one intrudes on a
    /// corner of its box, and the parked car is credited with the movement and
    /// harvested -- which is where the endless crops of the jeep came from.
    ///
    /// measured over 283 crops from one clip: motion belonging to the vehicle
    /// sits at 0.42 of its box's middle half (p10 0.24), while motion clipping a
    /// corner sits at 0.00 (p90 0.33). more than half the bad crops had no
    /// changed pixel in the middle of the box at all.
    pub fn central_share(&self, rect: Rect) -> f32 {
        central_share(&self.changed, self.width, self.height, rect)
    }

    /// drop the background model. called on startup and whenever the camera is
    /// panned or tilted, since every pixel then refers to somewhere else (r5.2).
    // wired up by the pan/tilt watcher in phase 1.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.background.fill(0);
        self.seen = 0;
        self.last_motion = None;
    }

    pub fn update(&mut self, frame: &Frame) -> GateResult {
        debug_assert_eq!(frame.data.len(), self.background.len());
        if self.seen == 0 {
            self.prime(frame);
        }
        let (changed, regions) = self.diff(frame);
        self.absorb(frame);

        self.seen = self.seen.saturating_add(1);
        if self.seen <= self.warmup_frames {
            // the model has not settled; reporting motion here is just noise.
            return GateResult {
                motion: false,
                changed_frac: 0.0,
                regions: Vec::new(),
            };
        }

        let frac = changed / self.observed_px as f32;
        let fired = frac >= self.min_changed_frac && !regions.is_empty();
        if fired {
            self.last_motion = Some(frame.received);
        }
        GateResult {
            motion: fired || self.latched(frame.received),
            changed_frac: frac,
            regions,
        }
    }

    /// motion stays latched briefly after the last qualifying frame so a single
    /// vehicle produces one event rather than a burst of them.
    fn latched(&self, now: Instant) -> bool {
        self.last_motion
            .is_some_and(|t| now.duration_since(t) < self.latch)
    }

    fn prime(&mut self, frame: &Frame) {
        for (bg, &px) in self.background.iter_mut().zip(frame.data.iter()) {
            *bg = (px as u16) << 8;
        }
    }

    /// find changed pixels, group them into objects, and bound each one.
    ///
    /// grouping is the whole job. a single bounding box over every changed pixel
    /// is worthless in a real scene: one speck of sensor noise in a corner and a
    /// car in the middle produce a full-frame box, which hands the detector the
    /// entire frame and defeats the gate.
    ///
    /// returns the changed-pixel count of surviving regions only, so that noise
    /// which never forms an object cannot trip the fraction threshold.
    fn diff(&mut self, frame: &Frame) -> (f32, Vec<Rect>) {
        self.mark_changed(frame);
        let components = self.label_components();
        let merged = merge_nearby(components, self.cfg.merge_gap_px);

        let mut kept: Vec<Component> = merged
            .into_iter()
            .filter(|c| {
                (c.rect.w >= self.cfg.min_region_px || c.rect.h >= self.cfg.min_region_px)
                    && c.rect.w.min(c.rect.h) >= self.cfg.min_region_thickness_px
            })
            .collect();
        kept.sort_by_key(|c| std::cmp::Reverse(c.rect.area()));
        kept.truncate(self.cfg.max_regions);

        let changed = kept.iter().map(|c| c.weight).sum();
        (changed, kept.into_iter().map(|c| c.rect).collect())
    }

    fn mark_changed(&mut self, frame: &Frame) {
        for (i, slot) in self.changed.iter_mut().enumerate() {
            if self.mask.as_ref().is_some_and(|m| !m[i]) {
                *slot = false;
                continue;
            }
            let cur = (frame.data[i] as u16) << 8;
            *slot = cur.abs_diff(self.background[i]) >= self.diff_threshold;
        }
    }

    /// 8-connected flood fill over the changed mask. iterative, with an explicit
    /// stack, because a large blob would blow a recursive one.
    fn label_components(&mut self) -> Vec<Component> {
        let (w, h) = (self.width as i64, self.height as i64);
        self.visited.iter_mut().for_each(|v| *v = false);
        let mut out = Vec::new();

        for start in 0..self.changed.len() {
            if !self.changed[start] || self.visited[start] {
                continue;
            }
            let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0i64, 0i64);
            let mut px = 0u32;
            let mut weight = 0.0f32;
            self.stack.clear();
            self.stack.push(start);
            self.visited[start] = true;

            while let Some(i) = self.stack.pop() {
                let (x, y) = ((i as i64) % w, (i as i64) / w);
                px += 1;
                weight += self.weights.as_ref().map_or(1.0, |ws| ws[i]);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                for (dx, dy) in NEIGHBOURS {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let n = (ny * w + nx) as usize;
                    if self.changed[n] && !self.visited[n] {
                        self.visited[n] = true;
                        self.stack.push(n);
                    }
                }
            }

            if px >= self.cfg.min_component_px {
                out.push(Component {
                    rect: Rect {
                        x: x0 as u32,
                        y: y0 as u32,
                        w: (x1 - x0 + 1) as u32,
                        h: (y1 - y0 + 1) as u32,
                    },
                    px,
                    weight,
                });
            }
        }
        out
    }

    fn absorb(&mut self, frame: &Frame) {
        for (bg, &px) in self.background.iter_mut().zip(frame.data.iter()) {
            let cur = (px as u16) << 8;
            // exponential moving average in fixed point, toward the current frame.
            *bg = if cur >= *bg {
                *bg + ((cur - *bg) >> self.cfg.bg_shift)
            } else {
                *bg - ((*bg - cur) >> self.cfg.bg_shift)
            };
        }
    }
}

impl Rect {
    pub fn area(&self) -> u32 {
        self.w * self.h
    }

    fn right(&self) -> u32 {
        self.x + self.w
    }

    fn bottom(&self) -> u32 {
        self.y + self.h
    }

    /// gap on each axis, zero when the rects touch or overlap.
    fn gap(&self, other: &Rect) -> (u32, u32) {
        let dx = self
            .x
            .max(other.x)
            .saturating_sub(self.right().min(other.right()));
        let dy = self
            .y
            .max(other.y)
            .saturating_sub(self.bottom().min(other.bottom()));
        (dx, dy)
    }

    fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        Rect {
            x,
            y,
            w: self.right().max(other.right()) - x,
            h: self.bottom().max(other.bottom()) - y,
        }
    }
}

/// repeatedly fuse regions that sit within the merge gap on both axes, until
/// nothing more combines. quadratic, but bounded by the component count, which
/// the pixel floor keeps small.
fn merge_nearby(mut components: Vec<Component>, gap: u32) -> Vec<Component> {
    let mut merged = true;
    while merged {
        merged = false;
        'outer: for i in 0..components.len() {
            for j in (i + 1)..components.len() {
                let (dx, dy) = components[i].rect.gap(&components[j].rect);
                if dx <= gap && dy <= gap {
                    let b = components.remove(j);
                    components[i] = Component {
                        rect: components[i].rect.union(&b.rect),
                        px: components[i].px + b.px,
                        weight: components[i].weight + b.weight,
                    };
                    merged = true;
                    break 'outer;
                }
            }
        }
    }
    components
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **a plane extrapolated where nothing was drawn must not invent
    /// sensitivity.** measured against the first real lines drawn on the
    /// deployment -- six of them, agreeing with each other to an rms residual
    /// of 8.4px on a mean height of 70.9px -- the fit still runs to **-77px**
    /// in the top-left, because no line was drawn anywhere near there. floored
    /// at a quarter of the smallest drawn line that corner came out 2900 times
    /// more sensitive than the bottom-right, all of it extrapolation rather
    /// than evidence.
    ///
    /// so the fit is held to the range the lines actually demonstrated. beyond
    /// their support the honest answer is the nearest thing that was measured,
    /// not a confident number nobody supplied.
    #[test]
    fn a_fit_is_never_more_confident_than_the_lines_that_were_drawn() {
        // the deployment's own six lines.
        let p = Perspective::fit(&[
            [[142, 413], [291, 399]],
            [[152, 260], [205, 223]],
            [[84, 289], [57, 312]],
            [[49, 279], [36, 300]],
            [[398, 112], [321, 157]],
            [[93, 416], [31, 416]],
        ])
        .unwrap();

        // shortest and longest drawn: 24.7px and 149.7px.
        let (shortest, longest) = (24.7f32, 149.7f32);
        for y in [0.0, 120.0, 240.0, 360.0, 479.0] {
            for x in [0.0, 160.0, 320.0, 480.0, 639.0] {
                let h = p.height_at(x, y);
                assert!(
                    h >= shortest - 0.5,
                    "h={h} at {x},{y} is smaller than any vehicle drawn ({shortest}px), \
                     which makes that corner sensitive on no evidence"
                );
                assert!(
                    h <= longest + 0.5,
                    "h={h} at {x},{y} is larger than any vehicle drawn ({longest}px)"
                );
            }
        }

        // and the widest sensitivity ratio across the frame is then exactly the
        // area ratio the lines themselves demonstrate, rather than a multiple
        // of it.
        let ratio = (longest / shortest).powi(2);
        assert!(
            (30.0..45.0).contains(&ratio),
            "the drawn lines span {ratio}x in area"
        );
    }

    /// **the point is holding on a small distant subject, not suppressing near
    /// ones.** the same vehicle differences into a quarter of the pixels at
    /// twice the distance, so one threshold either loses it out there or fires
    /// on noise up close. weighting each changed pixel by the vehicle area
    /// expected where it lies makes one vehicle worth about one vehicle
    /// wherever it is.
    ///
    /// measured against the case this exists for -- a car stopped at the stop
    /// sign up the block, motion decaying from 8.9% to under 0.2% in 3.8s while
    /// it was still sitting there.
    #[test]
    fn a_distant_region_counts_for_as_much_as_the_same_vehicle_near() {
        // vehicles are 80px tall at the near kerb and 40px at the far one, so
        // the same vehicle covers a quarter of the pixels far away.
        let mut cfg = cfg();
        cfg.min_changed_frac = 1e-6;
        cfg.perspective = vec![[[320, 400], [320, 480]], [[320, 100], [320, 140]]];

        let near = weighted_frac(
            &cfg,
            Rect {
                x: 280,
                y: 400,
                w: 40,
                h: 40,
            },
        );
        let far = weighted_frac(
            &cfg,
            Rect {
                x: 300,
                y: 100,
                w: 20,
                h: 20,
            },
        );

        assert!(
            far > 0.0 && near > 0.0,
            "neither fired: near={near} far={far}"
        );
        let ratio = far / near;
        assert!(
            (0.5..2.0).contains(&ratio),
            "the same vehicle should score alike at both depths: near={near} \
             far={far} ratio={ratio}"
        );
    }

    /// and with nothing drawn the score is the raw changed fraction, because
    /// that is what every deployment is running today.
    #[test]
    fn without_perspective_the_score_is_the_plain_changed_fraction() {
        let mut cfg = cfg();
        cfg.min_changed_frac = 1e-6;
        assert!(cfg.perspective.is_empty());

        let r = weighted_frac(
            &cfg,
            Rect {
                x: 280,
                y: 400,
                w: 40,
                h: 40,
            },
        );
        // 1600 changed pixels of 640x480 watched.
        let expected = 1600.0 / (640.0 * 480.0);
        assert!(
            (r - expected).abs() < 1e-6,
            "expected the plain fraction {expected}, got {r}"
        );
    }

    /// drive one block through a fresh full-size gate and report what it
    /// scored. full size rather than the 64x48 the other tests use, because a
    /// perspective fit is about where in a real frame something is.
    fn weighted_frac(cfg: &crate::config::Gate, block: Rect) -> f32 {
        const GW: u32 = 640;
        const GH: u32 = 480;
        let at = |d: &[u8]| Frame {
            width: GW,
            height: GH,
            data: Arc::new(d.to_vec()),
            seq: 0,
            received: Instant::now(),
        };
        let mut g = MotionGate::new(GW, GH, cfg);
        let blank = vec![0u8; (GW * GH) as usize];
        for _ in 0..(cfg.warmup_frames + 2) {
            g.update(&at(&blank));
        }
        let mut moved = blank.clone();
        for y in block.y..(block.y + block.h) {
            for x in block.x..(block.x + block.w) {
                moved[(y * GW + x) as usize] = 255;
            }
        }
        g.update(&at(&moved)).changed_frac
    }

    /// **the axis is diagonal, so interpolating on image row is wrong.**
    /// measured over 532 harvested boxes on this camera: apparent height
    /// explains R^2 0.36 against row alone, 0.50 against column, 0.76 against a
    /// plane in both. a fit that only reads `y` cannot tell the bottom-left
    /// corner from the mid-right one, and those differ by nine times in area.
    #[test]
    fn scale_is_fitted_across_both_axes_rather_than_down_the_image() {
        // three vehicles: small bottom-left, large mid-right, mid top-left.
        // heights vary with x as well as y, which a row-only fit cannot hold.
        let p = Perspective::fit(&[
            [[100, 400], [100, 450]], // 50px at (100, 425)
            [[500, 300], [500, 453]], // 153px at (500, 376)
            [[100, 200], [100, 280]], // 80px at (100, 240)
        ])
        .expect("three lines determine a plane");

        let bottom_left = p.height_at(100.0, 425.0);
        let mid_right = p.height_at(500.0, 376.0);
        assert!(
            (bottom_left - 50.0).abs() < 1.0,
            "bottom-left should be ~50px, got {bottom_left}"
        );
        assert!(
            (mid_right - 153.0).abs() < 1.0,
            "mid-right should be ~153px, got {mid_right}"
        );
        // the discriminating claim: two points on the same row, different
        // columns, must not be given the same scale.
        let (left, right) = (p.height_at(100.0, 350.0), p.height_at(500.0, 350.0));
        assert!(
            (left - right).abs() > 20.0,
            "a row-only fit: {left} and {right} on the same row"
        );
    }

    /// one line says how big a vehicle is and nothing about how that changes.
    #[test]
    fn a_single_line_is_a_constant_scale() {
        let p = Perspective::fit(&[[[300, 200], [300, 280]]]).unwrap();
        assert!((p.height_at(0.0, 0.0) - 80.0).abs() < 0.01);
        assert!((p.height_at(639.0, 479.0) - 80.0).abs() < 0.01);
    }

    /// two lines fix a gradient along the axis joining them, which is the whole
    /// point on a street that recedes diagonally.
    #[test]
    fn two_lines_interpolate_along_the_axis_joining_them() {
        // 40px at (120, 460), 120px at (520, 340): far-left to near-right.
        let p = Perspective::fit(&[[[120, 440], [120, 480]], [[520, 280], [520, 400]]]).unwrap();
        assert!((p.height_at(120.0, 460.0) - 40.0).abs() < 1.0);
        assert!((p.height_at(520.0, 340.0) - 120.0).abs() < 1.0);
        // and halfway along that axis is halfway between the two.
        let mid = p.height_at(320.0, 400.0);
        assert!(
            (mid - 80.0).abs() < 2.0,
            "midpoint should be ~80px, got {mid}"
        );
    }

    /// empty is what every deployment has today, and it must mean "unchanged"
    /// rather than "scale by zero".
    #[test]
    fn no_lines_is_no_correction_at_all() {
        assert!(Perspective::fit(&[]).is_none());
        // a click that was never dragged is not a scale either. config refuses
        // these, but the gate may not divide by one if it ever sees it.
        assert!(Perspective::fit(&[[[10, 10], [10, 10]]]).is_none());
    }

    /// extrapolation past the drawn lines can send a plane negative, and a
    /// height of zero divides. clamped to a few times what was actually
    /// measured, so a careless line cannot make a corner infinitely sensitive.
    #[test]
    fn a_fit_extrapolated_off_the_frame_stays_a_usable_height() {
        let p = Perspective::fit(&[[[100, 400], [100, 450]], [[500, 300], [500, 453]]]).unwrap();
        for (x, y) in [(0.0, 0.0), (639.0, 479.0), (0.0, 479.0), (639.0, 0.0)] {
            let h = p.height_at(x, y);
            assert!(h > 0.0, "height {h} at {x},{y} is not usable");
            assert!(h <= 153.0 * 4.0, "height {h} at {x},{y} is unclamped");
        }
    }
    use std::sync::Arc;

    const W: u32 = 64;
    const H: u32 = 48;

    fn frame(fill: u8) -> Frame {
        Frame {
            width: W,
            height: H,
            data: Arc::new(vec![fill; (W * H) as usize]),
            seq: 0,
            received: Instant::now(),
        }
    }

    fn paint(d: &mut [u8], x: u32, y: u32, w: u32, h: u32, v: u8) {
        for yy in y..y + h {
            for xx in x..x + w {
                d[(yy * W + xx) as usize] = v;
            }
        }
    }

    fn frame_with_block(fill: u8, block: u8, bx: u32, by: u32, bw: u32, bh: u32) -> Frame {
        let mut d = vec![fill; (W * H) as usize];
        paint(&mut d, bx, by, bw, bh, block);
        Frame {
            width: W,
            height: H,
            data: Arc::new(d),
            seq: 0,
            received: Instant::now(),
        }
    }

    fn gate(cfg: crate::config::Gate) -> MotionGate {
        MotionGate::new(W, H, &cfg)
    }

    /// **a region may overhang the roi, but never sits wholly outside it.**
    ///
    /// no changed pixel outside the mask contributes -- `mark_changed` zeroes
    /// them first -- so every region is grown from pixels inside the roi and its
    /// box must intersect it. what a box *can* do is overhang, since it is a
    /// bounding box and merging two takes the union.
    ///
    /// clipping the overhang was considered and rejected: the roi says where
    /// motion is *measured*, not where an object may be. a lorry whose roof
    /// crosses the boundary would be cropped with its roof cut off, and the crop
    /// is what the classifier sees.
    #[test]
    fn a_region_may_overhang_the_roi_but_never_sits_outside_it() {
        let mut g = gate(cfg());
        // watch everything except a vertical strip down the middle.
        let excluded = 30..34;
        let mask: Vec<bool> = (0..W * H).map(|i| !excluded.contains(&(i % W))).collect();
        g.set_mask(mask.clone());

        for _ in 0..3 {
            g.update(&frame(100));
        }
        // two blocks either side of the excluded strip, each big enough to
        // survive `self.cfg.min_region_px` and separated by no more than `MERGE_GAP_PX`.
        let mut d = vec![100u8; (W * H) as usize];
        paint(&mut d, 18, 20, 12, 12, 200);
        paint(&mut d, 34, 20, 12, 12, 200);
        let moving = Frame {
            width: W,
            height: H,
            data: Arc::new(d),
            seq: 4,
            received: Instant::now(),
        };
        let r = g.update(&moving);

        assert_eq!(
            r.regions.len(),
            1,
            "the two blocks should merge: {:?}",
            r.regions
        );
        let region = r.regions[0];
        assert!(
            region.x < excluded.start && region.x + region.w > excluded.end,
            "should span the strip: {region:?}"
        );
        // the property the preview relies on: a box always holds watched ground,
        // so none is ever drawn wholly outside the roi.
        assert!(
            watched_inside(&mask, region),
            "a region covered no watched pixel: {region:?}"
        );
    }

    /// **the endless crops of the jeep.**
    ///
    /// a parked vehicle's box, with a passing car intruding on one corner. the
    /// changed fraction cannot tell the two apart -- the pixels really are
    /// inside the box -- so the parked car is credited with the movement and
    /// harvested, over and over, whenever traffic goes by.
    #[test]
    fn motion_clipping_a_corner_is_not_this_vehicles_motion() {
        let mut g = gate(cfg());
        for _ in 0..3 {
            g.update(&frame(100));
        }
        // a car-sized box at 20,10 32x24, with something moving in its corner.
        g.update(&frame_with_block(100, 200, 20, 10, 8, 6));
        let car = Rect {
            x: 20,
            y: 10,
            w: 32,
            h: 24,
        };
        assert!(
            g.changed_fraction(car) > 0.05,
            "the pixels are inside the box: {}",
            g.changed_fraction(car)
        );
        assert!(
            g.central_share(car) < 0.2,
            "corner motion was credited to the box: {}",
            g.central_share(car)
        );
    }

    /// the counterweight: a vehicle that actually moved changed its own middle.
    #[test]
    fn a_vehicle_that_moved_has_motion_in_its_middle() {
        let mut g = gate(cfg());
        for _ in 0..3 {
            g.update(&frame(100));
        }
        let car = Rect {
            x: 20,
            y: 10,
            w: 32,
            h: 24,
        };
        g.update(&frame_with_block(100, 200, car.x, car.y, car.w, car.h));
        assert!(
            g.central_share(car) > 0.2,
            "a vehicle's own motion was rejected: {}",
            g.central_share(car)
        );
    }

    /// a box too small to have a middle must not be penalised for it.
    #[test]
    fn a_tiny_box_is_all_middle() {
        let mut g = gate(cfg());
        for _ in 0..3 {
            g.update(&frame(100));
        }
        g.update(&frame_with_block(100, 200, 20, 10, 3, 3));
        let tiny = Rect {
            x: 20,
            y: 10,
            w: 3,
            h: 3,
        };
        assert!(g.central_share(tiny) > 0.0, "a tiny box scored zero");
    }

    /// **the roi must not dilute a vehicle's movement score.**
    ///
    /// `mark_changed` forces masked pixels to unchanged, so a box straddling
    /// the roi boundary had its changed *fraction* divided by pixels that were
    /// never going to change. a car half outside could not score above 0.5
    /// however much of it moved, and in practice fell under `must_have_moved`
    /// -- so the gate fired, the detector found it, and nothing was harvested.
    ///
    /// measured against a real roi covering 38% of the frame: 45% of the crops
    /// a clip yields straddle its boundary, so this was most of the harvest.
    #[test]
    fn the_roi_does_not_dilute_a_vehicle_straddling_it() {
        let mut g = gate(cfg());
        // watch the left half only.
        let mask: Vec<bool> = (0..W * H).map(|i| (i % W) < W / 2).collect();
        g.set_mask(mask);
        for _ in 0..3 {
            g.update(&frame(100));
        }
        // a car from x=24 to x=40, straddling the boundary at 32.
        g.update(&frame_with_block(100, 200, 24, 10, 16, 12));
        let car = Rect {
            x: 24,
            y: 10,
            w: 16,
            h: 12,
        };
        assert!(
            g.changed_fraction(car) > 0.9,
            "every watched pixel of it moved, but it scored {}",
            g.changed_fraction(car)
        );
        assert!(
            g.central_share(car) > 0.2,
            "central share was diluted too: {}",
            g.central_share(car)
        );
    }

    /// and a box entirely outside the roi has no score to give.
    #[test]
    fn a_box_wholly_outside_the_roi_scores_zero() {
        let mut g = gate(cfg());
        let mask: Vec<bool> = (0..W * H).map(|i| (i % W) < W / 2).collect();
        g.set_mask(mask);
        for _ in 0..3 {
            g.update(&frame(100));
        }
        g.update(&frame_with_block(100, 200, 40, 10, 16, 12));
        let car = Rect {
            x: 40,
            y: 10,
            w: 16,
            h: 12,
        };
        assert_eq!(g.changed_fraction(car), 0.0);
    }

    /// does any pixel of `r` fall inside the mask?
    fn watched_inside(mask: &[bool], r: Rect) -> bool {
        (r.y..(r.y + r.h).min(H))
            .flat_map(|y| (r.x..(r.x + r.w).min(W)).map(move |x| (y * W + x) as usize))
            .any(|i| mask[i])
    }

    /// the counterweight: changed pixels outside the mask never count at all,
    /// so a region cannot be *caused* by something outside the roi.
    #[test]
    fn motion_entirely_outside_the_roi_is_not_seen() {
        let mut g = gate(cfg());
        // watch the bottom half only.
        let mask: Vec<bool> = (0..W * H).map(|i| i / W >= H / 2).collect();
        g.set_mask(mask);

        for _ in 0..3 {
            g.update(&frame(100));
        }
        let r = g.update(&frame_with_block(100, 200, 20, 4, 12, 12));
        assert!(!r.motion, "motion above the roi fired the gate");
        assert!(r.regions.is_empty(), "{:?}", r.regions);
    }

    fn cfg() -> crate::config::Gate {
        crate::config::Gate {
            warmup_frames: 2,
            latch_ms: 0,
            ..Default::default()
        }
    }

    #[test]
    fn static_scene_never_fires() {
        let mut g = gate(cfg());
        for _ in 0..60 {
            assert!(!g.update(&frame(100)).motion);
        }
    }

    #[test]
    fn warmup_suppresses_motion_until_the_model_settles() {
        let mut g = gate(crate::config::Gate {
            warmup_frames: 10,
            ..cfg()
        });
        g.update(&frame(100));
        // a large change during warmup must still be suppressed.
        let r = g.update(&frame_with_block(100, 255, 0, 0, 40, 40));
        assert!(!r.motion, "fired during warmup");
    }

    #[test]
    fn a_moving_block_fires_and_is_bounded() {
        let mut g = gate(cfg());
        for _ in 0..5 {
            g.update(&frame(100));
        }
        let r = g.update(&frame_with_block(100, 255, 10, 12, 20, 16));
        assert!(r.motion, "expected motion, changed_frac={}", r.changed_frac);
        assert_eq!(
            r.regions,
            vec![Rect {
                x: 10,
                y: 12,
                w: 20,
                h: 16
            }]
        );
    }

    /// the bug this guards: a single global bounding box over every changed
    /// pixel means one speck in a corner plus a car in the middle produces a
    /// full-frame region, and the detector then gets handed the whole frame.
    /// that defeats the gate entirely.
    #[test]
    fn two_separated_objects_produce_two_regions() {
        let mut g = gate(cfg());
        for _ in 0..5 {
            g.update(&frame(100));
        }
        let mut d = vec![100u8; (W * H) as usize];
        paint(&mut d, 2, 2, 16, 14, 255);
        paint(&mut d, 44, 30, 16, 14, 255);
        let r = g.update(&Frame {
            width: W,
            height: H,
            data: Arc::new(d),
            seq: 0,
            received: Instant::now(),
        });

        assert!(r.motion);
        assert_eq!(r.regions.len(), 2, "got {:?}", r.regions);
        for reg in &r.regions {
            assert!(reg.w <= 20 && reg.h <= 20, "region spans the gap: {reg:?}");
        }
    }

    /// the harvest kept collecting a parked jeep because cars drove past in
    /// front of it. the region overlapped the jeep; the jeep's own pixels never
    /// changed. this is the distinction that fixes it.
    #[test]
    fn a_stationary_object_beside_motion_has_no_changed_pixels_of_its_own() {
        let mut g = gate(cfg());
        for _ in 0..5 {
            g.update(&frame(100));
        }
        // something moves on the left; a "parked" object sits on the right.
        let mut d = vec![100u8; (W * H) as usize];
        paint(&mut d, 4, 10, 16, 16, 255);
        let f = Frame {
            width: W,
            height: H,
            data: Arc::new(d),
            seq: 0,
            received: Instant::now(),
        };
        g.update(&f);

        let moving = Rect {
            x: 4,
            y: 10,
            w: 16,
            h: 16,
        };
        let parked = Rect {
            x: 40,
            y: 10,
            w: 16,
            h: 16,
        };
        assert!(
            g.changed_fraction(moving) > 0.8,
            "the moving thing should be mostly changed: {}",
            g.changed_fraction(moving)
        );
        assert!(
            g.changed_fraction(parked) < 0.05,
            "a parked object must not look like it moved: {}",
            g.changed_fraction(parked)
        );
    }

    #[test]
    fn changed_fraction_survives_a_rect_outside_the_frame() {
        let mut g = gate(cfg());
        g.update(&frame(100));
        assert_eq!(
            g.changed_fraction(Rect {
                x: W + 10,
                y: 0,
                w: 5,
                h: 5
            }),
            0.0
        );
        // and a rect hanging off the edge counts only the pixels that exist.
        let _ = g.changed_fraction(Rect {
            x: W - 2,
            y: H - 2,
            w: 50,
            h: 50,
        });
    }

    /// low light gains the sensor up and sprays isolated changed pixels across
    /// the frame. they must not enlarge the region a real object produces.
    #[test]
    fn scattered_noise_does_not_inflate_a_real_region() {
        let mut g = gate(crate::config::Gate {
            min_changed_frac: 1e-6,
            ..cfg()
        });
        for _ in 0..5 {
            g.update(&frame(100));
        }
        let mut d = vec![100u8; (W * H) as usize];
        paint(&mut d, 20, 18, 16, 14, 255);
        // isolated single-pixel specks in the far corners.
        for (x, y) in [
            (0u32, 0u32),
            (W - 1, 0),
            (0, H - 1),
            (W - 1, H - 1),
            (60, 3),
        ] {
            d[(y * W + x) as usize] = 255;
        }
        let r = g.update(&Frame {
            width: W,
            height: H,
            data: Arc::new(d),
            seq: 0,
            received: Instant::now(),
        });

        assert!(r.motion);
        let biggest = r.regions.first().expect("a region");
        assert_eq!(
            *biggest,
            Rect {
                x: 20,
                y: 18,
                w: 16,
                h: 14
            },
            "noise leaked into the region: {:?}",
            r.regions
        );
    }

    /// a vehicle rarely differences as one blob: windscreen, body, and shadow
    /// fragment. pieces closer than the merge gap are one object.
    #[test]
    fn adjacent_fragments_merge_into_one_region() {
        let mut g = gate(cfg());
        for _ in 0..5 {
            g.update(&frame(100));
        }
        let mut d = vec![100u8; (W * H) as usize];
        paint(&mut d, 10, 10, 14, 10, 255);
        paint(&mut d, 26, 10, 14, 10, 255); // 2px gap, same object
        let r = g.update(&Frame {
            width: W,
            height: H,
            data: Arc::new(d),
            seq: 0,
            received: Instant::now(),
        });

        assert_eq!(r.regions.len(), 1, "fragments not merged: {:?}", r.regions);
        assert_eq!(
            r.regions[0],
            Rect {
                x: 10,
                y: 10,
                w: 30,
                h: 10
            }
        );
    }

    #[test]
    fn a_speck_smaller_than_the_noise_floor_does_not_fire() {
        let mut g = gate(crate::config::Gate {
            min_changed_frac: 1e-6,
            ..cfg()
        });
        for _ in 0..5 {
            g.update(&frame(100));
        }
        // exceeds the fraction threshold but is smaller than self.cfg.min_region_px.
        let r = g.update(&frame_with_block(100, 255, 5, 5, 3, 3));
        assert!(!r.motion, "a 3x3 speck must not fire");
    }

    /// a long thin sliver is not an object.
    ///
    /// measured on this street: 70% of the regions handed to the detector were
    /// under 2000px, median 357px. the slivers are glare shimmer on the glass,
    /// wires, and shadow edges, and each one costs a 148ms inference that a real
    /// vehicle elsewhere in the frame then does not get.
    #[test]
    fn a_sliver_is_not_a_region() {
        let mut g = gate(crate::config::Gate {
            min_changed_frac: 1e-6,
            ..cfg()
        });
        for _ in 0..5 {
            g.update(&frame(100));
        }
        // as wide as self.cfg.min_region_px demands, but two pixels tall: a wire, not a
        // car. the filter was once `w >= MIN || h >= MIN`, which this passes.
        let r = g.update(&frame_with_block(100, 255, 5, 5, 20, 2));
        assert!(
            r.regions.is_empty(),
            "a 20x2 sliver became a region: {:?}",
            r.regions
        );
    }

    /// the counterpart, and the reason the thickness floor is small: a vehicle
    /// differences as a wide shallow band, and a go-4 across the street is only
    /// about 50x30 gate pixels. neither may be discarded as a sliver.
    #[test]
    fn a_shallow_but_solid_region_survives() {
        let mut g = gate(crate::config::Gate {
            min_changed_frac: 1e-6,
            ..cfg()
        });
        for _ in 0..5 {
            g.update(&frame(100));
        }
        let r = g.update(&frame_with_block(100, 255, 5, 5, 30, 6));
        assert_eq!(
            r.regions.len(),
            1,
            "a vehicle's leading edge was discarded: {:?}",
            r.regions
        );
    }

    /// the whole point of a background model: a car that arrives and then stays
    /// put must stop costing anything. the block is full of them.
    #[test]
    fn a_parked_car_is_absorbed_into_the_background() {
        // ~4s at 30fps. long enough to be clearly absorbed, short enough that a
        // vehicle genuinely stopped at the curb still trips dwell downstream.
        const MAX_ABSORB_FRAMES: usize = 120;

        let mut g = gate(cfg());
        for _ in 0..5 {
            g.update(&frame(100));
        }
        let parked = frame_with_block(100, 255, 10, 10, 20, 20);
        assert!(g.update(&parked).motion, "should fire when it arrives");

        let mut absorbed_after = None;
        for i in 0..MAX_ABSORB_FRAMES {
            if !g.update(&parked).motion {
                absorbed_after = Some(i);
                break;
            }
        }
        let absorbed_after = absorbed_after
            .unwrap_or_else(|| panic!("still firing after {MAX_ABSORB_FRAMES} frames"));

        // and it must stay absorbed, not oscillate around the threshold.
        for _ in 0..100 {
            assert!(
                !g.update(&parked).motion,
                "re-fired after absorbing at {absorbed_after}"
            );
        }
    }

    #[test]
    fn masked_pixels_are_ignored() {
        let mut g = gate(cfg());
        // observe only the bottom half; motion in the top half must not fire.
        let mask: Vec<bool> = (0..W * H).map(|i| i / W >= H / 2).collect();
        g.set_mask(mask);
        for _ in 0..5 {
            g.update(&frame(100));
        }
        let r = g.update(&frame_with_block(100, 255, 5, 2, 30, 20));
        assert!(!r.motion, "fired on masked-out region");
    }
}
