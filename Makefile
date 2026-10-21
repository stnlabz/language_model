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
BYTES ?= 64
STATE_IN ?=
STATE_OUT ?= $(RUN)_state_$(STEPS).stn

TRAIN = $(RUN)_train.bin
VALID = $(RUN)_validation.bin
CHECKPOINT = $(RUN)_$(STEPS).stn

.DEFAULT_GOAL := help
.PHONY: help all build test prepare-data train evaluate generate probe pipeline status train-state resume

help:
	@printf '%s\n' \
		'Build/test: make build | make test' \
		'Prepare:    make prepare-data DATA=stn_language_sentences.txt RUN=corpus' \
		'Train:      make train RUN=corpus STEPS=5000' \
		'Evaluate:   make evaluate RUN=corpus STEPS=5000' \
		'Generate:   make generate RUN=corpus STEPS=5000 PROMPT="The engineer"' \
		'Probe:      make probe RUN=corpus STEPS=5000 PROMPT="The engineer"' \
		'Pipeline:   make pipeline DATA=stn_language_sentences.txt RUN=corpus' \
		'Stateful:   make train-state RUN=corpus STEPS=6400' \
		'Resume:     make resume RUN=corpus STEPS=6400 STATE_IN=prior.stn STATE_OUT=next.stn' \
		'Status:     make status' \
		'' \
		'RUN labels experiment artifacts; distinct RUN names avoid collisions.' \
		'No target automatically downloads, overwrites, deploys, or deletes files.' \
		'Only explicit train/pipeline targets train the model.'

all: build

build:
	$(CC) $(CFLAGS) prepare.c -o prepare
	$(CC) $(CFLAGS) model.c -lm -o model

test:
	sh tests/test_prepare.sh
	sh tests/test_model.sh

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
	@if test -e '$(CHECKPOINT)'; then echo 'Checkpoint exists; choose a new RUN or STEPS' >&2; exit 1; fi
	nice -n '$(NICE)' timeout '$(TIMEOUT)' ./model train '$(TRAIN)' '$(VALID)' '$(STEPS)' '$(CHECKPOINT)'

evaluate: build
	@test -s '$(CHECKPOINT)' && test -s '$(TRAIN)' && test -s '$(VALID)' || { echo 'Missing checkpoint or data' >&2; exit 1; }
	$(PYTHON) tests/evaluate_language.py --model ./model --checkpoint '$(CHECKPOINT)' --train '$(TRAIN)' --validation '$(VALID)'

generate: build
	@test -s '$(CHECKPOINT)' || { echo 'Missing checkpoint: $(CHECKPOINT)' >&2; exit 1; }
	./model generate '$(CHECKPOINT)' '$(PROMPT)' '$(BYTES)' '$(SEED)' utf8

probe: build
	@test -s '$(CHECKPOINT)' || { echo 'Missing checkpoint: $(CHECKPOINT)' >&2; exit 1; }
	./model probe '$(CHECKPOINT)' '$(PROMPT)'

# Opt-in workflow; use a unique RUN for each new experiment.
pipeline:
	$(MAKE) prepare-data DATA='$(DATA)' RUN='$(RUN)'
	$(MAKE) train RUN='$(RUN)' STEPS='$(STEPS)' NICE='$(NICE)' TIMEOUT='$(TIMEOUT)'
	$(MAKE) evaluate RUN='$(RUN)' STEPS='$(STEPS)'

# Stateful training uses WINDOW-aligned batches; never reuse a checkpoint path.
train-state: build
	@test -s '$(TRAIN)' && test -s '$(VALID)' || { echo 'Missing prepared data' >&2; exit 1; }
	@if test -e '$(STATE_OUT)'; then echo 'State output exists; choose a new STATE_OUT' >&2; exit 1; fi
	nice -n '$(NICE)' timeout '$(TIMEOUT)' ./model train-state '$(TRAIN)' '$(VALID)' '$(STEPS)' '$(STATE_OUT)'

resume: build
	@test -n '$(STATE_IN)' && test -s '$(STATE_IN)' || { echo 'Set STATE_IN to an existing stateful checkpoint' >&2; exit 1; }
	@test -s '$(TRAIN)' && test -s '$(VALID)' || { echo 'Missing prepared data' >&2; exit 1; }
	@if test -e '$(STATE_OUT)'; then echo 'State output exists; choose a new STATE_OUT' >&2; exit 1; fi
	nice -n '$(NICE)' timeout '$(TIMEOUT)' ./model resume '$(STATE_IN)' '$(TRAIN)' '$(VALID)' '$(STEPS)' '$(STATE_OUT)'
