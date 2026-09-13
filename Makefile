.DEFAULT_GOAL := build
.PHONY: build dev run test test-e2e eval transits score endtoend precommit fmt lint clean \
	models record logo prepare label retrain

CARGO ?= cargo
BIN   := target/release/metermate
E2E   := tests/e2e

build:
	$(CARGO) build --release

dev:
	$(CARGO) build

run: dev
	$(CARGO) run

# fetch and export both models. offline tooling: the runtime never needs python
# or torch, so this deliberately runs in throwaway uv environments.
#
# the embedder was missing from here for a while, which made it undiscoverable:
# `--label` and `--measure` need it and the only way to get it was to know that
# tools/export_embedder.py existed.
models: models/detector.onnx models/embedder.onnx

models/detector.onnx:
	uv run --with ultralytics --with onnx --with onnxslim --python 3.12 \
		python tools/export_model.py

# 351 MB, and the download of torch to produce it is larger again. on a second
# machine, copying the exported file over is usually faster than re-exporting.
models/embedder.onnx:
	uv run --with torch --with transformers --with onnx --python 3.12 \
		python tools/export_embedder.py

# replay a recorded clip and report what was found. a measurement rather than a
# pass/fail test, so it is not part of precommit:
#   make eval CLIP=data/eval/<stamp>-sub.mp4
# stdlib only, so no dependency list is needed.
eval: build
	uv run --python 3.12 python tools/eval.py $(CLIP) $(EVAL_ARGS)

# record both streams at once into data/eval. the pair matters: the eval needs
# the substream the gate saw and the main stream its crops came from, covering
# the same minutes.
#   make record SECS=600
#   make record RECORD_ARGS="--at 19:30"
SECS ?= 600
record:
	uv run --python 3.12 python tools/record.py --secs $(SECS) $(RECORD_ARGS)

# pull the moments a vehicle actually transits the scene out of a clip, for
# hand labelling. STAMP names a recorded pair in data/eval:
#   make transits STAMP=20260912-103138
transits:
	uv run --with onnxruntime --with numpy --with pillow --with scipy --python 3.12 \
		python tools/label.py data/eval/$(STAMP)-sub.mp4 data/eval/$(STAMP)-main.mp4 \
		data/eval/$(STAMP)-events $(TRANSITS_ARGS)

# recall and false positives, from a manifest whose `truth` fields are filled in.
score:
	uv run --python 3.12 python tools/score.py $(LABELS)

# the same labels, asked of the whole pipeline: did a crop of each transit
# actually reach the harvest, and was each crop taken of something moving.
# builds, because it replays the real binary:
#   make endtoend STAMP=20260912-103138
#
# numpy and scipy are for the precision half, which runs its own motion scan
# rather than trusting the labels. `--no-precision` drops both.
endtoend: build
	uv run --with numpy --with scipy --python 3.12 python tools/endtoend.py \
		$(E2E)/fixtures/labels/$(STAMP).json data/eval/$(STAMP)-sub.mp4 $(E2E_ARGS)

# the three steps of a labelling round, in the order they are run.
#
# everything before these happens in the browser: crops are labelled on the
# preview's crops tab and passages are marked on its events tab. these are
# what turns that into a trained file, and they are apart because the costs
# are: `prepare` is a detector pass per marked passage, `label` is a person
# looking at what it cut, `retrain` is embedding and measuring everything
# labelled so far.
#
# **`label` is not optional.** a cut passage arrives unlabelled, and an
# unlabelled crop is counted as street: a retrain straight after a prepare
# scores the go-4 that was just cut as a false positive.
#
#   make prepare    # cut every marked passage into sets/
#   make label      # label what was cut, every set at once, in the browser
#   make retrain    # then train every configured subject over the lot
#
#   make label SUBJECT=waymo SETS=sets/waymo
SUBJECT ?= go4
SETS    ?= sets
prepare: build
	$(BIN) --prepare

label: build
	$(BIN) --label $(SUBJECT) --harvest $(SETS)

retrain: build
	$(BIN) --retrain

# render the mark from the vector.
#
# **assets/logo.svg is the mark**, drawn by hand, and the pngs are renders of
# it -- nothing regenerates the curves. the favicon is compiled into the binary,
# so a rebuild follows this.
logo: assets/logo.png assets/favicon.png

assets/logo.png: assets/logo.svg
	inkscape $< --export-type=png --export-filename=$@ --export-width=512 --export-height=512

assets/favicon.png: assets/logo.svg
	inkscape $< --export-type=png --export-filename=$@ --export-width=64 --export-height=64

test:
	$(CARGO) test

# end to end tests drive the real binary, so they need the release build
test-e2e: build
	cd $(E2E) && uv run pytest -v

fmt:
	$(CARGO) fmt --all

lint:
	$(CARGO) fmt --all -- --check
	$(CARGO) clippy --all-targets -- -D warnings
	cd $(E2E) && uv run ruff check .
	cd $(E2E) && uv run ruff format --check .

# builds too, so the release binary on disk always matches what was just
# linted and tested. leaving it out meant measuring a stale binary twice.
precommit: lint test build

clean:
	$(CARGO) clean
	rm -rf $(E2E)/.pytest_cache $(E2E)/.venv
