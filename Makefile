# STN-Labz original micro Transformer, Linux CPU only.
CC = cc
CFLAGS = -std=c11 -O2 -Wall -Wextra -pedantic
DATA ?= digit_approved_learning.jsonl
CORPUS ?= micro_corpus.txt
CHECKPOINT ?= llm.stn
STEPS ?= 5000
PROMPT ?= The engineer
BYTES ?= 64
SEED ?= 913

.DEFAULT_GOAL := help
.PHONY: help build test prepare train generate status learn serve
help:
	@echo 'make build'
	@echo 'make prepare DATA=digit_approved_learning.jsonl CORPUS=micro_corpus.txt'
	@echo 'make train CORPUS=micro_corpus.txt CHECKPOINT=micro_llm.stn STEPS=5000'
	@echo 'make generate CHECKPOINT=micro_llm.stn PROMPT="The engineer"'
	@echo 'make learn DATA=digit_approved_learning.jsonl CHECKPOINT=llm.stn STEPS=5000'
	@echo 'make serve CHECKPOINT=llm.stn'
	@echo 'No automatic service installation or deployment.'
build:
	$(CC) $(CFLAGS) micro_llm.c -lm -o micro_llm
test: build
	./micro_llm 2>/dev/null; test $$? -eq 2
prepare:
	python3 prepare_jsonl.py '$(DATA)' '$(CORPUS)'
train: build
	./micro_llm train '$(CORPUS)' '$(CHECKPOINT)' '$(STEPS)'
generate: build
	./micro_llm generate '$(CHECKPOINT)' '$(PROMPT)' '$(BYTES)' '$(SEED)'
status:
	@echo 'DATA=$(DATA) CORPUS=$(CORPUS) CHECKPOINT=$(CHECKPOINT) STEPS=$(STEPS)'

# Explicit three-component cycle: gru_learn -> llm checkpoint -> inference service.
# Neither command installs nor activates systemd.
learn: build
	python3 gru_learn.py --jsonl '$(DATA)' --checkpoint '$(CHECKPOINT)' --steps '$(STEPS)' --binary ./micro_llm
serve: build
	python3 llm_service.py --checkpoint '$(CHECKPOINT)' --binary ./micro_llm
