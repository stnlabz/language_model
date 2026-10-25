# STN-Labz native C micro LM; Linux GCC/Make only.
CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -pedantic
DATA ?= digit_approved_learning.jsonl
CORPUS ?= micro_corpus.txt
CHECKPOINT ?= llm.stn
STEPS ?= 5000
PROMPT ?= The engineer
BYTES ?= 64
SEED ?= 913
PORT ?= 8765

.DEFAULT_GOAL := help
.PHONY: help build test prepare train generate learn serve status clean
help:
	@echo 'make build'
	@echo 'make clean'
	@echo 'make prepare DATA=digit_approved_learning.jsonl CORPUS=micro_corpus.txt'
	@echo 'make learn DATA=digit_approved_learning.jsonl CHECKPOINT=llm.stn STEPS=5000'
	@echo 'make generate CHECKPOINT=llm.stn PROMPT="The engineer"'
	@echo 'make serve CHECKPOINT=llm.stn PORT=8765'
	@echo 'No Python, auto-deployment, or automatic service installation.'
build: micro_llm gru_learn llm_service
micro_llm: micro_llm.c
	$(CC) $(CFLAGS) micro_llm.c -lm -o micro_llm
gru_learn: gru_learn.c json_fields.h
	$(CC) $(CFLAGS) gru_learn.c -o gru_learn
llm_service: llm_service.c json_fields.h
	$(CC) $(CFLAGS) llm_service.c -o llm_service
tests/native_regression: tests/native_regression.c json_fields.h
	$(CC) $(CFLAGS) tests/native_regression.c -o tests/native_regression
test: build tests/native_regression
	./tests/native_regression
prepare: gru_learn
	@echo 'Corpus preparation is integrated into gru_learn. Use make learn.'
train: micro_llm
	./micro_llm train '$(CORPUS)' '$(CHECKPOINT)' '$(STEPS)'
learn: build
	./gru_learn '$(DATA)' '$(CHECKPOINT)' '$(STEPS)' './micro_llm'
generate: micro_llm
	./micro_llm generate '$(CHECKPOINT)' '$(PROMPT)' '$(BYTES)' '$(SEED)'
serve: build
	./llm_service './micro_llm' '$(CHECKPOINT)' '$(PORT)'
status:
	@echo 'DATA=$(DATA) CHECKPOINT=$(CHECKPOINT) STEPS=$(STEPS)'

# Clean only generated executables; retain JSONL, corpora and model checkpoints.
clean:
	rm -f micro_llm gru_learn llm_service tests/native_regression
