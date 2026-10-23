# STN-Labz language_model Makefile
# AI-assisted; build and tests do not train on the operator corpus.
# Training, data preparation, and pipeline are explicit commands only.
# No deletes, deployment, Git commands, or background services.

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
STATE_IN ?=
STATE_OUT ?= $(RUN)_state_$(STEPS).stn

TRAIN = $(RUN)_train.bin
VALID = $(RUN)_validation.bin
CHECKPOINT ?= word_gru.stn
SAMPLE ?= $(RUN)_word_sample.txt

.DEFAULT_GOAL := help
.PHONY: help all build test prepare-data train evaluate generate probe pipeline status train-state resume quality compare-sampling api

help:
	@echo 'Word GRU checkpoint: word_gru.stn'
	@echo 'Build/test: make build | make test'
	@echo 'Prepare: make prepare-data DATA=stn_language_sentences.txt RUN=corpus'
	@echo 'Train: make train RUN=corpus CHECKPOINT=word_gru.stn'
	@echo 'Generate: make generate CHECKPOINT=word_gru.stn PROMPT="The engineer"'
	@echo 'API: make api CHECKPOINT=word_gru.stn PORT=8766'
	@echo 'Pipeline: make pipeline DATA=stn_language_sentences.txt RUN=corpus'
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
	nice -n '$(NICE)' timeout '$(TIMEOUT)' ./word_gru train '$(TRAIN)' '$(VALID)' '$(CHECKPOINT)' '$(SAMPLE)'

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

# Word-GRU stateful resume and legacy byte-GRU quality tools are not supported.
train-state resume quality compare-sampling:
	@echo 'Unsupported for word_gru.stn; no byte-GRU fallback will run.' >&2; exit 2

# Explicit foreground, loopback-only word-GRU HTTP inference.
api: build
	@test -s '$(CHECKPOINT)' || { echo 'Missing word-GRU checkpoint: $(CHECKPOINT)' >&2; exit 1; }
	$(PYTHON) http_api.py --word-model --model ./word_gru --checkpoint '$(CHECKPOINT)' --port '$(PORT)'
