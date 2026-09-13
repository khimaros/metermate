//! passages somebody marked in the preview, waiting to be cut into sets.
//!
//! **a selection is a note, not an extraction.** cutting crops out of a clip
//! loads the detector and reads every frame, which is minutes of cpu -- and
//! r3.1 budgets the whole pipeline at 30% of a core on a machine that is
//! watching a street. a page that started that work would spend the deployment
//! to save a person typing, so the page writes down what to do and `--prepare`
//! does it later.
//!
//! the file is the same shape as `labels.txt` for the same reasons: one line
//! per decision, comments allowed, readable and editable with anything. it
//! sits beside the events directory, as the labels sit beside the harvest.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// one clip and the seconds of it worth cropping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub clip: String,
    /// `START-END` as `--dense` takes it: `SS`, `MM:SS` or `HH:MM:SS`, either
    /// end omittable. kept as written rather than parsed to seconds, so what
    /// `--prepare` runs is what somebody marked.
    pub window: String,
}

/// beside the events directory, as `labels.txt` sits beside the harvest.
pub fn path(events: &Path) -> PathBuf {
    events.parent().unwrap_or(events).join("selections.txt")
}

/// what is waiting to be cut, oldest mark first, without duplicates.
///
/// a clip and a window name a passage, and marking the same passage twice is
/// an accident of clicking rather than a request for two copies of it.
pub fn load(path: &Path) -> Result<Vec<Selection>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading selections {}", path.display()))?;
    let mut out: Vec<Selection> = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(clip), Some(window)) = (parts.next(), parts.next()) else {
            continue;
        };
        let want = Selection {
            clip: clip.to_string(),
            window: window.to_string(),
        };
        if !out.contains(&want) {
            out.push(want);
        }
    }
    Ok(out)
}

/// add one, without reading or rewriting the file.
///
/// the same argument the labels file makes: the pipeline is writing this while
/// `--prepare` may be reading it, and an append cannot lose what it did not
/// know about.
pub fn append(path: &Path, clip: &str, window: &str) -> Result<()> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    writeln!(file, "{clip} {window}").with_context(|| format!("appending to {}", path.display()))
}

/// take a mark back, keeping every other line as written.
///
/// rewritten rather than appended to, unlike the labels: a selection is a note
/// about work not yet done, so there is nothing to lose by rewriting and no
/// tombstone worth inventing. only the preview writes this file, and
/// `--prepare` only reads it.
///
/// removing one that is not there is not an error. two clicks on the same `x`
/// is somebody making sure.
pub fn remove(path: &Path, clip: &str, window: &str) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading selections {}", path.display()))?;
    let (mut kept, mut dropped) = (String::new(), 0);
    for line in text.lines() {
        let body = line.split('#').next().unwrap_or("").trim();
        let mut parts = body.split_whitespace();
        let matches = parts.next() == Some(clip) && parts.next() == Some(window);
        if matches {
            dropped += 1;
            continue;
        }
        kept.push_str(line);
        kept.push('\n');
    }
    if dropped > 0 {
        std::fs::write(path, kept).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(dropped)
}

/// what a selection is cut into: one directory per clip and window.
///
/// **the window is in the name.** two passages of one clip are two sets, and
/// naming them only by the clip would make the second overwrite the first --
/// or worse, land in the same directory and be counted as one passage by
/// everything that groups by time.
pub fn set_dir(sets: &Path, subject: &str, s: &Selection) -> PathBuf {
    let stamp = crate::record::parse_name(&s.clip).map_or(0, |(stamp, _, _)| stamp);
    let window = s.window.replace(':', "").replace('-', "to");
    sets.join(subject).join(format!("{stamp}-{window}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrote(body: &str) -> Vec<Selection> {
        let dir = std::env::temp_dir().join(format!("metermate-sel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("selections.txt");
        std::fs::write(&path, body).unwrap();
        let out = load(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        out
    }

    #[test]
    fn the_file_reads_like_the_labels_file() {
        let got = wrote("# a comment\n\n1789-subject-main.mp4 0:03-0:08\n");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].clip, "1789-subject-main.mp4");
        assert_eq!(got[0].window, "0:03-0:08");
    }

    /// clicking select twice on the same card is an accident of clicking, not
    /// a request for two copies of the passage -- and the second copy would be
    /// counted as a separate example by everything downstream.
    #[test]
    fn the_same_passage_marked_twice_is_one_selection() {
        let got = wrote("a-main.mp4 0-1\na-main.mp4 0-1\n");
        assert_eq!(got.len(), 1, "{got:?}");
    }

    /// two windows of one clip are two passages, though: a go-4 that comes
    /// back is not the same example as the first visit.
    #[test]
    fn two_windows_of_one_clip_are_two_selections() {
        let got = wrote("a-main.mp4 0-1\na-main.mp4 5-9\n");
        assert_eq!(got.len(), 2, "{got:?}");
        assert_ne!(
            set_dir(Path::new("sets"), "go4", &got[0]),
            set_dir(Path::new("sets"), "go4", &got[1]),
            "two passages of one clip must not share a set directory"
        );
    }

    /// the comments are somebody's, and a removal is not an invitation to
    /// tidy the file they are in.
    #[test]
    fn taking_one_back_leaves_the_rest_of_the_file_alone() {
        let dir = std::env::temp_dir().join(format!("metermate-unsel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("selections.txt");
        std::fs::write(
            &path,
            "# marked while watching\na-main.mp4 0-1\nb-main.mp4 5-9\n",
        )
        .unwrap();

        assert_eq!(remove(&path, "a-main.mp4", "0-1").unwrap(), 1);
        let left = std::fs::read_to_string(&path).unwrap();
        assert!(left.contains("# marked while watching"), "{left}");
        assert!(left.contains("b-main.mp4 5-9"), "{left}");
        assert!(!left.contains("a-main.mp4"), "{left}");

        // and again, which is two clicks on the same x.
        assert_eq!(remove(&path, "a-main.mp4", "0-1").unwrap(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_set_is_named_for_the_clip_and_the_window() {
        let s = Selection {
            clip: "1789660864671-subject-main.mp4".into(),
            window: "0:03-0:08".into(),
        };
        assert_eq!(
            set_dir(Path::new("sets"), "go4", &s),
            Path::new("sets/go4/1789660864671-003to008")
        );
    }
}
