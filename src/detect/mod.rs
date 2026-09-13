//! stage-1 detection: find vehicles and people, generically.
//!
//! this stage is deliberately not project-specific. it runs a stock coco model
//! and answers "is there a vehicle here, and where". deciding whether that
//! vehicle is an sfmta go-4 is the classifier's job, because recognition needs
//! far less labelled data as a classification problem than as a detection one.
//!
//! the `person` class is not incidental: the dismount trigger (r1.3) needs it.

use anyhow::{Context, Result};
use std::path::Path;

/// coco class ids we act on. everything else is detected and then ignored,
/// except in the raw detector debug mode, which shows all of them (r8.4).
pub const CLASS_PERSON: usize = 0;
pub const CLASS_CAR: usize = 2;
pub const CLASS_MOTORCYCLE: usize = 3;
pub const CLASS_BUS: usize = 5;
pub const CLASS_TRUCK: usize = 7;

/// a go-4 is a small three-wheeler and coco has no class for it, so it is
/// expected to land in one of these, probably inconsistently. which one, and at
/// what confidence, is exactly what the raw detector mode exists to find out.
///
/// the *default* of a config key rather than a rule (r10.3): a subject need not
/// be a vehicle, and anything discarded here is discarded before stage 2 or the
/// harvest ever sees it.
/// **not `bicycle`.** a go-4 has never been mistaken for one here, and a street
/// with cyclists on it produces a steady trickle of detections that reach the
/// harvest, the tracker and the alert path for a thing enforcement never
/// arrives on. it is still a coco class the model reports and `--list-classes`
/// names, so anyone watching for one can ask for it.
pub const DEFAULT_CLASSES: [usize; 4] = [CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK];

/// the model emits at most this many boxes; it is baked into the exported graph.
/// values per detection row: x1, y1, x2, y2, confidence, class id.
const ROW_STRIDE: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// corners in the model's input coordinate space, which is also what the
    /// preview draws on, so nothing needs mapping back.
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: usize,
}

impl Detection {
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    pub fn label(&self) -> &'static str {
        class_name(self.class_id)
    }
}

/// which detector classes are worth carrying past stage 1.
///
/// this used to be a compiled-in list of vehicles, which quietly decided that
/// metermate could only ever watch for vehicles: anything else was dropped
/// before the classifier or the harvest saw it, so no amount of later
/// configuration could bring it back (r10.3).
///
/// **an empty filter keeps everything.** that is the honest reading of "I did
/// not ask you to exclude anything", and it is what a new subject wants while
/// its crops are still being collected. the default is not empty, though: it is
/// the vehicle list, so nothing changes for anyone who does not ask.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClassFilter {
    ids: Vec<usize>,
}

impl ClassFilter {
    /// resolve coco class names. an unknown name is an error rather than a
    /// silent miss: a typo would otherwise read as "this class never appears",
    /// which looks exactly like a detector that is not working.
    pub fn from_names(names: &[String]) -> Result<Self> {
        let mut ids = Vec::new();
        for name in names {
            let id = COCO_CLASSES
                .iter()
                .position(|c| c.eq_ignore_ascii_case(name))
                .with_context(|| format!("unknown detector class {name:?}"))?;
            ids.push(id);
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(Self { ids })
    }

    pub fn from_ids(ids: impl IntoIterator<Item = usize>) -> Self {
        let mut ids: Vec<usize> = ids.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        Self { ids }
    }

    pub fn keeps(&self, d: &Detection) -> bool {
        self.ids.is_empty() || self.ids.contains(&d.class_id)
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.ids.iter().map(|id| class_name(*id)).collect()
    }

    pub fn is_any(&self) -> bool {
        self.ids.is_empty()
    }
}

pub trait Detector {
    /// `rgb` is `size * size * 3` bytes of rgb24, already letterboxed.
    fn detect(&mut self, rgb: &[u8], size: u32) -> Result<Vec<Detection>>;
    fn input_size(&self) -> u32;
}

pub struct OnnxDetector {
    session: ort::session::Session,
    size: u32,
    min_confidence: f32,
    /// rows the export's nms block can return, which is a property of the graph
    /// rather than a preference: reading past it reads zero padding.
    max_detections: usize,
    /// reused so the per-frame path does not allocate a 4.9 MB tensor each time.
    input: Vec<f32>,
}

impl OnnxDetector {
    pub fn load(
        model: &Path,
        size: u32,
        min_confidence: f32,
        threads: usize,
        max_detections: usize,
    ) -> Result<Self> {
        // ort's builder returns a typed error that carries the builder back, so
        // it cannot be absorbed by anyhow directly. flatten each to a message.
        // the closures cannot be shared: each call site has a different type.
        let session = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("creating onnx session builder")?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("setting detector thread count")?
            .commit_from_file(model)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading detector model {}", model.display()))?;

        Ok(Self {
            session,
            size,
            min_confidence,
            max_detections,
            input: vec![0.0; (size as usize) * (size as usize) * 3],
        })
    }

    /// rgb24 interleaved -> nchw float, scaled to 0..1, which is what the
    /// ultralytics export expects.
    fn fill_input(&mut self, rgb: &[u8]) {
        let px = (self.size as usize) * (self.size as usize);
        for i in 0..px {
            self.input[i] = rgb[i * 3] as f32 / 255.0;
            self.input[px + i] = rgb[i * 3 + 1] as f32 / 255.0;
            self.input[2 * px + i] = rgb[i * 3 + 2] as f32 / 255.0;
        }
    }
}

impl Detector for OnnxDetector {
    fn input_size(&self) -> u32 {
        self.size
    }

    fn detect(&mut self, rgb: &[u8], size: u32) -> Result<Vec<Detection>> {
        anyhow::ensure!(
            size == self.size,
            "frame size {size} != model input {}",
            self.size
        );
        anyhow::ensure!(
            rgb.len() == (size as usize) * (size as usize) * 3,
            "expected {} rgb bytes, got {}",
            (size as usize) * (size as usize) * 3,
            rgb.len()
        );
        self.fill_input(rgb);

        let s = self.size as usize;
        let tensor = ort::value::Tensor::from_array(([1, 3, s, s], self.input.clone()))
            .context("building input tensor")?;
        let outputs = self
            .session
            .run(ort::inputs!["images" => tensor])
            .context("running detector")?;

        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("reading detector output")?;
        Ok(parse_detections(
            data,
            self.min_confidence,
            self.max_detections,
        ))
    }
}

/// decode a `[1, 300, 6]` block of final detections.
///
/// the export bakes **class-agnostic** nms into the graph, so these are already
/// deduplicated and need only thresholding here. rows are zero-padded up to the
/// fixed 300, so a zero confidence means "no more detections", not "a bad one".
fn parse_detections(data: &[f32], min_confidence: f32, max_detections: usize) -> Vec<Detection> {
    let rows = (data.len() / ROW_STRIDE).min(max_detections);
    let mut out = Vec::new();
    for r in 0..rows {
        let row = &data[r * ROW_STRIDE..(r + 1) * ROW_STRIDE];
        let confidence = row[4];
        if confidence < min_confidence {
            continue;
        }
        out.push(Detection {
            x1: row[0],
            y1: row[1],
            x2: row[2],
            y2: row[3],
            confidence,
            class_id: row[5] as usize,
        });
    }
    out
}

pub fn class_name(id: usize) -> &'static str {
    COCO_CLASSES.get(id).copied().unwrap_or("unknown")
}

#[rustfmt::skip]
pub const COCO_CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
];

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_DETECTIONS: usize = crate::config::MAX_DETECTIONS;

    #[test]
    fn padding_rows_are_not_reported_as_detections() {
        // the graph always emits 300 rows; unused ones are zeros. treating those
        // as detections would flood every frame with boxes at the origin.
        let mut data = vec![0.0f32; crate::config::MAX_DETECTIONS * ROW_STRIDE];
        data[..ROW_STRIDE].copy_from_slice(&[10.0, 20.0, 110.0, 80.0, 0.9, 2.0]);

        let got = parse_detections(&data, 0.25, MAX_DETECTIONS);
        assert_eq!(got.len(), 1, "padding leaked through: {got:?}");
        assert_eq!(got[0].class_id, CLASS_CAR);
        assert_eq!(got[0].width(), 100.0);
        assert_eq!(got[0].height(), 60.0);
    }

    #[test]
    fn confidence_threshold_is_applied() {
        let mut data = vec![0.0f32; 3 * ROW_STRIDE];
        data[0..ROW_STRIDE].copy_from_slice(&[0.0, 0.0, 10.0, 10.0, 0.90, 2.0]);
        data[ROW_STRIDE..2 * ROW_STRIDE].copy_from_slice(&[0.0, 0.0, 10.0, 10.0, 0.30, 7.0]);
        data[2 * ROW_STRIDE..].copy_from_slice(&[0.0, 0.0, 10.0, 10.0, 0.10, 0.0]);

        assert_eq!(parse_detections(&data, 0.25, MAX_DETECTIONS).len(), 2);
        assert_eq!(parse_detections(&data, 0.50, MAX_DETECTIONS).len(), 1);
        assert_eq!(parse_detections(&data, 0.95, MAX_DETECTIONS).len(), 0);
    }

    fn of(class_id: usize) -> Detection {
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            confidence: 1.0,
            class_id,
        }
    }

    #[test]
    fn the_default_filter_covers_what_a_go4_might_register_as() {
        // a three-wheeler has no coco class, so it will land somewhere in here.
        let classes = ClassFilter::from_ids(DEFAULT_CLASSES);
        for id in [CLASS_CAR, CLASS_TRUCK, CLASS_MOTORCYCLE, CLASS_BUS] {
            assert!(classes.keeps(&of(id)), "{} should be kept", of(id).label());
        }
        assert!(!classes.keeps(&of(CLASS_PERSON)));
        assert_eq!(of(CLASS_PERSON).label(), "person");
    }

    /// the point of making this configurable: a subject need not be a vehicle,
    /// and anything dropped here never reaches the classifier or the harvest.
    #[test]
    fn a_configured_filter_can_watch_for_something_that_is_not_a_vehicle() {
        let classes = ClassFilter::from_names(&["person".into(), "dog".into()]).unwrap();
        assert!(classes.keeps(&of(CLASS_PERSON)));
        assert!(!classes.keeps(&of(CLASS_CAR)));
        assert_eq!(classes.names(), vec!["person", "dog"]);
    }

    #[test]
    fn an_empty_filter_keeps_everything() {
        let classes = ClassFilter::from_names(&[]).unwrap();
        assert!(classes.is_any());
        for id in [CLASS_PERSON, CLASS_CAR, 79] {
            assert!(
                classes.keeps(&of(id)),
                "{id} was dropped by an empty filter"
            );
        }
        assert_eq!(classes, ClassFilter::from_ids([]));
    }

    /// a typo would otherwise read as "this class never appears", which looks
    /// exactly like a detector that is not working.
    #[test]
    fn an_unknown_class_name_is_an_error_rather_than_a_silent_miss() {
        let err = ClassFilter::from_names(&["car".into(), "go4".into()]).unwrap_err();
        assert!(format!("{err:#}").contains("go4"), "{err:#}");
    }

    #[test]
    fn class_names_are_matched_without_regard_to_case_and_deduplicated() {
        let classes =
            ClassFilter::from_names(&["Car".into(), "car".into(), "TRUCK".into()]).unwrap();
        assert_eq!(classes.names(), vec!["car", "truck"]);
    }

    #[test]
    fn class_names_line_up_with_coco_ids() {
        assert_eq!(class_name(CLASS_PERSON), "person");
        assert_eq!(class_name(CLASS_CAR), "car");
        assert_eq!(class_name(CLASS_TRUCK), "truck");
        assert_eq!(class_name(CLASS_BUS), "bus");
        assert_eq!(class_name(999), "unknown");
    }
}
