# contributing

## build

    make            # alias for make build
    make build      # release binary at target/release/metermate
    make dev        # debug build

## test

    make test       # rust unit tests
    make test-e2e   # python end to end tests
    make precommit  # fmt, clippy, ruff. run before every commit

prefer end to end tests over unit tests. anything that can be tested end to end
should be. unit tests are for pure functions with tricky arithmetic, such as
coordinate mapping.

end to end tests are python, under `tests/e2e`, managed by `uv`. they drive the
real binary with `--source <file>` so no camera is needed, and assert against a
real mqtt broker.

what a test needs that the suite does not already build goes in a helper module
beside the tests rather than copied from another test: `labelling.py`,
`preview.py` and `alerts.py` are those helpers. `alerts.py` builds the clips,
trains a reference set from them and stands up a fake ntfy server, which is how a
test reaches a live confirmation at all.

### writing a bug fix

investigate, form a hypothesis, then write a test. confirm the test fails.
only then write the fix, and confirm the test passes.

## toolchain

`mise` pins the rust and python versions. `mise install` once, then the makefile
picks them up. builds should be reproducible: pin versions, commit lockfiles.

## layout

    src/ingest/     ffmpeg subprocess, camera cgi, ptz status
    src/gate/       roi motion gate
    src/detect/     stage-1 detector behind a trait
    src/classify/   stage-2 classifier behind a trait
    src/track/      association, dwell, temporal confirmation
    src/alert/      mqtt and home assistant discovery
    src/harvest/    crop recorder
    tools/          python offline tooling: roi editor, labeling, training
    tests/e2e/      python end to end tests

## style

- ascii only, everywhere.
- documentation, comments, and command line output are lowercase. caps are for
  acronyms and emphasis.
- comments explain why, not what. no changelog comments. delete dead code rather
  than commenting it out.
- magic values become named constants at the top of the file that uses them, or
  in `src/config.rs` when shared.
- keep functions under 50 lines. prefer pure functions.
- prefer extending an existing component over adding a new one.

## dependencies

keep them few. every new crate needs a reason that cannot be met by ~50 lines of
local code. the hot path must not acquire a dependency that pulls in a c++
toolchain or a python runtime.

## camera

metermate treats the camera as read-only by default (r5.4). anything that writes
camera configuration lives behind an explicit opt-in flag and is never invoked by
tests.

## releases

never mutate version control on the user's behalf. no commits, tags, or pushes
unless asked.
