# the judge fixture

`judge` in `src/classify/mod.rs` is the whole decision: every verdict the
pipeline records and every number a training report prints rests on it. a
change to it would move all of them without failing anything else, so
`the_fixture_is_judged_as_written` pins it to hand-built vectors whose verdicts
can be checked by eye.

- `go4/references.txt` and `go4/negatives.txt` are the shipped layout: one
  directory per subject, `<label> <floats>` per line, in four dimensions rather
  than clip's 512 so it can be read by eye. the label inside each file is
  redundant with the directory holding it and kept anyway -- a `negatives.txt`
  full of go-4s would train the exact inversion of the rule, and the reader
  refuses it rather than ignoring the label as a detail.
- `queries.txt` is the same line format, where the label is the **verdict the
  query is expected to receive** rather than what set it belongs to. it is read
  as labelled rows, not as references, because those are different questions.

**the negatives belong to the subject, not to the street.** one shared negative
set could not hold another subject without dragging down the subject it belonged
to, so `judge` scores each subject against its own. this fixture has one subject
and therefore cannot catch a regression in that, which is what
`one_subject_is_a_negative_for_another` in `classify` is for; add a second
subject directory here if the fixture ever needs to cover it.

the fifth query sits exactly between the two clusters. it is labelled `other`
because an ambiguous crop must not be claimed as a sighting -- the margin exists
for that case and this is what pins the behaviour.

verdicts are compared rather than scores, so the fixture does not break on the
last bit of a float.
