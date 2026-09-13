//! just enough fragmented-mp4 parsing to fan one stream out to many viewers.
//!
//! the camera's h.264 is remuxed once, by the same ffmpeg that already decodes
//! the main stream for crops. that single byte stream then has to serve any
//! number of browsers, each of which may connect at any moment.
//!
//! a browser cannot simply be handed the stream from wherever it happens to be:
//! it needs the **init segment** first -- the `ftyp` and `moov` boxes, which
//! describe the codec -- and then whole fragments from a boundary. so the stream
//! is split as it arrives, the init kept, and fragments handed out from the next
//! boundary after a viewer arrives.
//!
//! this is deliberately not an mp4 parser. it reads box headers and nothing
//! else, because that is all the question requires.

/// every box begins with a 32-bit big-endian size and a four-character type.
const HEADER: usize = 8;

#[derive(Debug, PartialEq)]
pub enum Chunk {
    /// `ftyp`/`moov`: what a new viewer needs before anything else.
    Init(Vec<u8>),
    /// `moof`+`mdat`: one playable fragment.
    Fragment(Vec<u8>),
}

/// splits a fragmented-mp4 byte stream into init and fragments.
///
/// fed arbitrary sized reads, because that is what a socket delivers: a box may
/// arrive split across several reads, or several boxes may arrive in one.
#[derive(Default)]
pub struct Split {
    buf: Vec<u8>,
    /// init boxes accumulate until the first `moof` proves the header is done.
    init: Vec<u8>,
    init_done: bool,
    /// a fragment is a `moof` and the `mdat` that follows it, emitted together
    /// so a viewer never receives half of one.
    pending: Vec<u8>,
}

impl Split {
    pub fn new() -> Self {
        Self::default()
    }

    /// consume bytes and return whatever complete chunks they finished.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Chunk> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();

        while let Some((size, kind)) = peek_box(&self.buf) {
            if self.buf.len() < size {
                break; // the rest of this box has not arrived yet
            }
            let boxed: Vec<u8> = self.buf.drain(..size).collect();
            match kind.as_slice() {
                // the first moof ends the init segment.
                b"moof" => {
                    if !self.init_done {
                        self.init_done = true;
                        out.push(Chunk::Init(std::mem::take(&mut self.init)));
                    }
                    // a moof with a fragment already pending means the previous
                    // one had no mdat. emit it rather than gluing them together.
                    if !self.pending.is_empty() {
                        out.push(Chunk::Fragment(std::mem::take(&mut self.pending)));
                    }
                    self.pending = boxed;
                }
                b"mdat" if !self.pending.is_empty() => {
                    self.pending.extend_from_slice(&boxed);
                    out.push(Chunk::Fragment(std::mem::take(&mut self.pending)));
                }
                // anything before the first moof is part of the header. anything
                // else after it (`styp`, `sidx`, free space) is not something a
                // viewer needs, and is dropped rather than guessed at.
                _ if !self.init_done => self.init.extend_from_slice(&boxed),
                _ => {}
            }
        }
        out
    }
}

/// read a box header: its total size and its type.
///
/// returns `None` while fewer than eight bytes are buffered. a declared size
/// below the header is nonsense and would loop forever, so it is treated the
/// same as "wait for more" -- the stream is corrupt and stalling beats spinning.
fn peek_box(buf: &[u8]) -> Option<(usize, [u8; 4])> {
    if buf.len() < HEADER {
        return None;
    }
    let size = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let kind = [buf[4], buf[5], buf[6], buf[7]];
    (size >= HEADER).then_some((size, kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = (HEADER + payload.len()) as u32;
        let mut v = size.to_be_bytes().to_vec();
        v.extend_from_slice(kind);
        v.extend_from_slice(payload);
        v
    }

    fn stream() -> Vec<u8> {
        let mut s = mp4_box(b"ftyp", b"isom");
        s.extend(mp4_box(b"moov", b"header"));
        s.extend(mp4_box(b"moof", b"frag1"));
        s.extend(mp4_box(b"mdat", b"pixels1"));
        s.extend(mp4_box(b"moof", b"frag2"));
        s.extend(mp4_box(b"mdat", b"pixels2"));
        s
    }

    /// a viewer needs ftyp and moov before anything, or the browser has no idea
    /// what codec it is being handed.
    #[test]
    fn the_header_boxes_come_out_as_one_init_segment() {
        let got = Split::new().push(&stream());
        let Chunk::Init(init) = &got[0] else {
            panic!("expected init first, got {got:?}");
        };
        assert_eq!(
            init,
            &[mp4_box(b"ftyp", b"isom"), mp4_box(b"moov", b"header")].concat()
        );
    }

    #[test]
    fn each_moof_and_its_mdat_are_emitted_as_one_fragment() {
        let got = Split::new().push(&stream());
        let frags: Vec<_> = got
            .iter()
            .filter_map(|c| match c {
                Chunk::Fragment(f) => Some(f.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(frags.len(), 2, "{got:?}");
        assert_eq!(
            frags[0],
            [mp4_box(b"moof", b"frag1"), mp4_box(b"mdat", b"pixels1")].concat()
        );
        assert_eq!(
            frags[1],
            [mp4_box(b"moof", b"frag2"), mp4_box(b"mdat", b"pixels2")].concat()
        );
    }

    /// a socket does not deliver whole boxes. this is the case that matters and
    /// the one a naive implementation gets wrong.
    #[test]
    fn boxes_split_across_reads_are_reassembled() {
        let s = stream();
        let mut split = Split::new();
        let mut got = Vec::new();
        // one byte at a time: the most hostile chunking there is.
        for b in &s {
            got.extend(split.push(&[*b]));
        }
        assert!(matches!(got[0], Chunk::Init(_)), "{got:?}");
        assert_eq!(
            got.iter()
                .filter(|c| matches!(c, Chunk::Fragment(_)))
                .count(),
            2,
            "{got:?}"
        );
    }

    #[test]
    fn several_boxes_arriving_at_once_are_all_emitted() {
        let mut split = Split::new();
        let got = split.push(&stream());
        assert_eq!(got.len(), 3, "init + two fragments: {got:?}");
    }

    /// nothing may be emitted before it is complete: half an mdat played by a
    /// browser is a decode error, not a slightly short frame.
    #[test]
    fn a_partial_fragment_is_withheld_until_it_is_whole() {
        let s = stream();
        let mut split = Split::new();
        // everything except the last four bytes of the final mdat.
        let got = split.push(&s[..s.len() - 4]);
        assert_eq!(
            got.iter()
                .filter(|c| matches!(c, Chunk::Fragment(_)))
                .count(),
            1,
            "the incomplete fragment leaked: {got:?}"
        );
        let rest = split.push(&s[s.len() - 4..]);
        assert_eq!(rest.len(), 1, "the completed fragment never arrived");
    }

    /// a corrupt length must not spin the reader forever.
    #[test]
    fn a_nonsense_box_size_does_not_loop() {
        let mut split = Split::new();
        let mut bad = 2u32.to_be_bytes().to_vec(); // smaller than a header
        bad.extend_from_slice(b"junk");
        assert!(split.push(&bad).is_empty());
    }
}
