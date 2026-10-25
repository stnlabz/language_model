# STN-Labz language_model Makefile
# AI-assisted; build and tests do not train on the operator corpus.
# Training, data preparation, and pipeline are explicit commands only.
# No service restart, Git commands, or background services.
# Only 'make go' clears generated artifacts in its dedicated gru_run directory.

CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Werror -pedantic
PYTHON = python3
DATA ?= stn_language_sentences.txt
RUN ?= corpus
STEPS ?= 5000
TIMEOUT ?= 120s
NICE ?= 10
PROMPT ?= The engineer
SEED ?= 913
WORDS ?= 32
TEMPERATURE ?= 0.65
PENALTY ?= 0.0
BYTES ?= 64
PORT ?= 8766
# Foreground, CPU-only Digit snapshot -> fresh checkpoint workflow.
GO_SOURCE ?= digit_approved_learning.jsonl
GO_DIR := gru_run
GO_TIMEOUT ?= 6h
STATE_IN ?=
STATE_OUT ?= $(RUN)_state_$(STEPS).stn

TRAIN = $(RUN)_train.bin
VALID = $(RUN)_validation.bin
CHECKPOINT ?= word_gru.stn
SAMPLE ?= $(RUN)_word_sample.txt
# Optional, separately approved Q/A corpus for answer-only supervision.
DIALOGUE ?=

.DEFAULT_GOAL := help
.PHONY: help all build test prepare-data train evaluate generate probe pipeline status train-state resume quality compare-sampling api go

help:
	@echo 'Word GRU checkpoint: word_gru.stn'
	@echo 'Build/test: make build | make test'
	@echo 'Prepare: make prepare-data DATA=stn_language_sentences.txt RUN=corpus'
	@echo 'Train: make train RUN=corpus CHECKPOINT=word_gru.stn [DIALOGUE=approved_qa.txt]'
	@echo 'Generate: make generate CHECKPOINT=word_gru.stn PROMPT="The engineer"'
	@echo 'API: make api CHECKPOINT=word_gru.stn PORT=8766'
	@echo 'Pipeline: make pipeline DATA=stn_language_sentences.txt RUN=corpus'
	@echo 'CPU train from approved Digit JSONL with separated Q/A: make go (GO_SOURCE=... GO_TIMEOUT=6h)'
	@echo 'go reuses gru_run/, archives older backups under checkpoint_backups/,'
	@echo 'then rotates the current checkpoint to word_gru.previous.stn and promotes the new one.'
	@echo 'No API restart or service activation is performed.'
	@echo 'Status: make status'
	@echo 'Legacy byte-GRU training/stateful/quality/probe targets are unsupported.'
	@echo 'Existing server API on port 8765 is untouched.'

all: build

build:
	$(CC) $(CFLAGS) prepare.c -o prepare
	$(CC) $(CFLAGS) word_gru.c -lm -o word_gru

test:
	sh tests/test_prepare.sh
	@echo 'Legacy byte-GRU tests are not word-GRU qualification tests.'

status:
	@printf 'DATA=%s\nRUN=%s\nSTEPS=%s\nTRAIN=%s\nVALID=%s\nCHECKPOINT=%s\n' \
		'$(DATA)' '$(RUN)' '$(STEPS)' '$(TRAIN)' '$(VALID)' '$(CHECKPOINT)'

# The existing prepare executable refuses to overwrite train/validation files.
prepare-data: build
	@test -f '$(DATA)' || { echo 'Missing authorized corpus: $(DATA)' >&2; exit 1; }
	@if test -e '$(TRAIN)' || test -e '$(VALID)'; then echo 'Prepared output exists; use a new RUN' >&2; exit 1; fi
	./prepare prepare '$(DATA)' '$(TRAIN)' '$(VALID)'

# Existing checkpoints are never overwritten. timeout and nice
# reduce the impact on other services running on the shared VPS.
train: build
	@test -s '$(TRAIN)' && test -s '$(VALID)' || { echo 'Missing prepared data: run make prepare-data first' >&2; exit 1; }
	@if test -e '$(CHECKPOINT)' || test -e '$(SAMPLE)'; then echo 'Word checkpoint or sample already exists; never overwrite' >&2; exit 1; fi
	@if test -n '$(DIALOGUE)'; then \
	  test -s '$(DIALOGUE)' || { echo 'Approved dialogue corpus missing or empty: $(DIALOGUE)' >&2; exit 1; }; \
	  echo 'Answer-only supervision: enabled (approved dialogue file supplied)'; \
	  nice -n '$(NICE)' timeout '$(TIMEOUT)' ./word_gru train '$(TRAIN)' '$(VALID)' '$(CHECKPOINT)' '$(SAMPLE)' '$(DIALOGUE)'; \
	else \
	  echo 'Answer-only supervision: disabled (no DIALOGUE supplied)'; \
	  nice -n '$(NICE)' timeout '$(TIMEOUT)' ./word_gru train '$(TRAIN)' '$(VALID)' '$(CHECKPOINT)' '$(SAMPLE)'; \
	fi

evaluate:
	@echo 'Word GRU validation loss is reported by make train; no separate word-GRU evaluation command exists.' >&2; exit 2

generate: build
	@test -s '$(CHECKPOINT)' || { echo 'Missing word-GRU checkpoint: $(CHECKPOINT)' >&2; exit 1; }
	./word_gru sample '$(CHECKPOINT)' '$(PROMPT)' '$(WORDS)' '$(SEED)' '$(TEMPERATURE)' '$(PENALTY)'

probe:
	@echo 'Word GRU has no probe command.' >&2; exit 2

# Opt-in workflow; use a unique RUN for each new experiment.
pipeline:
	$(MAKE) prepare-data DATA='$(DATA)' RUN='$(RUN)'
	$(MAKE) train RUN='$(RUN)' STEPS='$(STEPS)' NICE='$(NICE)' TIMEOUT='$(TIMEOUT)'
	@echo 'Word GRU training complete; consult validation loss and sample artifact.'

# Explicit CPU-only retraining; every run reuses exactly one gru_run directory.
# Only generated run artifacts are deleted. A successful run promotes its checkpoint
# to the repository root after preserving the previous root checkpoint.
# The source queue must be outside gru_run and is never deleted by this target.
# No timestamped subdirectories, background services, or API restart.
go:
	@set -eu; \
	$(MAKE) build; \
	test -s '$(GO_SOURCE)' || { echo 'Missing or empty Digit queue: $(GO_SOURCE)' >&2; exit 1; }; \
	case '$(GO_SOURCE)' in gru_run/*|./gru_run/*) \
	  echo 'GO_SOURCE must be outside gru_run' >&2; exit 1;; \
	esac; \
	if test -L '$(GO_DIR)'; then echo 'Refusing symlink at gru_run' >&2; exit 1; fi; \
	if test -e '$(GO_DIR)' && ! test -d '$(GO_DIR)'; then \
	  echo 'gru_run exists but is not a directory' >&2; exit 1; \
	fi; \
	if test -d '$(GO_DIR)'; then \
	  rm -f -- '$(GO_DIR)/snapshot.jsonl' '$(GO_DIR)/corpus.txt' \
	    '$(GO_DIR)/gru_train.bin' '$(GO_DIR)/gru_validation.bin' \
	    '$(GO_DIR)/dialogue_corpus.txt' \
	    '$(GO_DIR)/word_gru.stn' '$(GO_DIR)/word_sample.txt'; \
	fi; \
	mkdir -p -m 0700 '$(GO_DIR)'; \
	cp '$(GO_SOURCE)' '$(GO_DIR)/snapshot.jsonl'; \
	if ! cmp -s '$(GO_SOURCE)' '$(GO_DIR)/snapshot.jsonl'; then \
	  echo 'Source changed during snapshot; refusing inconsistent training input' >&2; exit 1; \
	fi; \
	$(PYTHON) tools/migrate_digit_jsonl.py \
	  --input '$(GO_DIR)/snapshot.jsonl' --output '$(GO_DIR)/corpus.txt'; \
	$(PYTHON) tools/migrate_digit_jsonl.py \
	  --input '$(GO_DIR)/snapshot.jsonl' --output '$(GO_DIR)/dialogue_corpus.txt' --mode dialogue; \
	printf '%s\n' 'NOTICE: Only explicitly approved non-structured exchanges enter answer-only supervision.'; \
	printf '%s\n' 'NOTICE: Sparse dialogue examples cannot establish conversational qualification.'; \
	$(MAKE) prepare-data DATA='$(GO_DIR)/corpus.txt' RUN='$(GO_DIR)/gru'; \
	$(MAKE) train RUN='$(GO_DIR)/gru' CHECKPOINT='$(GO_DIR)/word_gru.stn' \
	  SAMPLE='$(GO_DIR)/word_sample.txt' DIALOGUE='$(GO_DIR)/dialogue_corpus.txt' \
	  NICE='$(NICE)' TIMEOUT='$(GO_TIMEOUT)'; \
	test -s '$(GO_DIR)/word_gru.stn' || { \
	  echo 'Training completed without a nonempty checkpoint' >&2; exit 1; \
	}; \
	if test -L 'word_gru.previous.stn'; then \
	  echo 'Refusing symlink at previous checkpoint' >&2; exit 1; \
	fi; \
	if test -e 'word_gru.previous.stn'; then \
	  test -f 'word_gru.previous.stn' || { echo 'Previous checkpoint is not a regular file' >&2; exit 1; }; \
	  if test -L 'checkpoint_backups' || { test -e 'checkpoint_backups' && ! test -d 'checkpoint_backups'; }; then \
	    echo 'Backup archive path is not a normal directory' >&2; exit 1; \
	  fi; \
	  mkdir -p -m 0700 'checkpoint_backups'; \
	  archive=$(mktemp -d 'checkpoint_backups/previous.XXXXXXXX') || { echo 'Cannot allocate a unique backup archive' >&2; exit 1; }; \
	  if ! mv -- 'word_gru.previous.stn' "$archive/word_gru.stn"; then \
	    echo 'Cannot archive existing rollback checkpoint' >&2; exit 1; \
	  fi; \
	  printf 'Archived older rollback checkpoint: %s\\n' "$archive/word_gru.stn"; \
	fi; \
	if test -L 'word_gru.stn'; then \
	  echo 'Refusing to replace root checkpoint symlink' >&2; exit 1; \
	fi; \
	if test -e 'word_gru.stn'; then \
	  test -f 'word_gru.stn' || { echo 'Root checkpoint is not a regular file' >&2; exit 1; }; \
	  mv -- 'word_gru.stn' 'word_gru.previous.stn'; \
	fi; \
	if ! mv -- '$(GO_DIR)/word_gru.stn' 'word_gru.stn'; then \
	  if test -f 'word_gru.previous.stn' && ! test -e 'word_gru.stn'; then \
	    mv -- 'word_gru.previous.stn' 'word_gru.stn' || true; \
	  fi; \
	  echo 'Checkpoint move failed' >&2; exit 1; \
	fi; \
	echo 'CPU GRU checkpoint moved to ./word_gru.stn'; \
	echo 'Existing root checkpoint backed up to ./word_gru.previous.stn (when present).'; \
	echo 'No running API was restarted.'

# Word-GRU stateful resume and legacy byte-GRU quality tools are not supported.
train-state resume quality compare-sampling:
	@echo 'Unsupported for word_gru.stn; no byte-GRU fallback will run.' >&2; exit 2

# Explicit foreground, loopback-only word-GRU HTTP inference.
api: build
	@test -s '$(CHECKPOINT)' || { echo 'Missing word-GRU checkpoint: $(CHECKPOINT)' >&2; exit 1; }
	$(PYTHON) http_api.py --word-model --model ./word_gru --checkpoint '$(CHECKPOINT)' --port '$(PORT)'
