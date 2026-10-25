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
SQLITE_DB ?= /opt/digit/data/digit_memory.db

.DEFAULT_GOAL := help
.PHONY: help build test native-test math-test language-test prose-test sqlite-test sqlite-export sqlite-learn prepare train generate learn serve status clean
help:
	@echo 'make build'
	@echo 'make test (native, mathematical, and language qualification; aggregate PASS/FAIL)'
	@echo 'make clean'
	@echo 'make sqlite-export SQLITE_DB=/opt/digit/data/digit_memory.db (read-only; stdout)'
	@echo 'make sqlite-test (synthetic database; does not access Digit)'
	@echo 'make prose-test (synthetic prose training, held-out loss and generation diagnostic)'
	@echo 'make sqlite-learn SQLITE_DB=... CHECKPOINT=new.stn STEPS=... (explicit training only)'
	@echo 'make prepare DATA=digit_approved_learning.jsonl CORPUS=micro_corpus.txt'
	@echo 'make learn DATA=digit_approved_learning.jsonl CHECKPOINT=llm.stn STEPS=5000'
	@echo 'make generate CHECKPOINT=llm.stn PROMPT="The engineer"'
	@echo 'make serve CHECKPOINT=llm.stn PORT=8765'
	@echo 'No Python, auto-deployment, or automatic service installation.'
build: micro_llm gru_learn llm_service sqlite_corpus
sqlite_corpus: sqlite_corpus.c
	$(CC) $(CFLAGS) sqlite_corpus.c -lsqlite3 -o sqlite_corpus
sqlite-export: sqlite_corpus
	@./sqlite_corpus '$(SQLITE_DB)'
sqlite-learn: sqlite_corpus micro_llm
	@tmp=$$(mktemp /tmp/stn-sqlite-corpus-XXXXXX) || exit 1; \
	trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	./sqlite_corpus '$(SQLITE_DB)' > "$$tmp" && \
	./micro_llm train "$$tmp" '$(CHECKPOINT)' '$(STEPS)'
micro_llm: micro_llm.c
	$(CC) $(CFLAGS) micro_llm.c -lm -o micro_llm
gru_learn: gru_learn.c json_fields.h
	$(CC) $(CFLAGS) gru_learn.c -o gru_learn
llm_service: llm_service.c json_fields.h
	$(CC) $(CFLAGS) llm_service.c -o llm_service
tests/native_regression: tests/native_regression.c json_fields.h
	$(CC) $(CFLAGS) tests/native_regression.c -o tests/native_regression
native-test:
	@$(MAKE) --no-print-directory -B build tests/native_regression
	./tests/native_regression

# Run every test group even when an earlier group fails. Preserve the
# individual exit status and fail the aggregate if any group fails.
test:
	@failed=0; \
	for target in native-test math-test sqlite-test prose-test language-test; do \
		printf '\n=== %s ===\n' "$$target"; \
		if $(MAKE) --no-print-directory "$$target"; then \
			printf 'PASS: %s\n' "$$target"; \
		else \
			printf 'FAIL: %s\n' "$$target"; \
			failed=1; \
		fi; \
	done; \
	if [ "$$failed" -eq 0 ]; then \
		printf '\nALL TEST GROUPS PASS\n'; \
	else \
		printf '\nTEST SUITE FAIL: see individual group results\n' >&2; \
	fi; \
	exit "$$failed"
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
	rm -f micro_llm gru_learn llm_service sqlite_corpus tests/native_regression tests/model_math tests/sqlite_corpus_test

# Mathematical checks and synthetic language qualification; no service startup.
math-test:
	@$(MAKE) --no-print-directory -B tests/model_math
	./tests/model_math
tests/model_math: tests/model_math.c micro_llm.c
	$(CC) $(CFLAGS) tests/model_math.c -lm -o tests/model_math
language-test:
	@echo 'Language qualification: bounded to 180 seconds; training artifacts retained.'
	@timeout -v -k 5s 180s sh ./tests/language_qualification.sh

# Synthetic SQLite regression: exercises approval filter, duplicates, and read-only access.
tests/sqlite_corpus_test: tests/sqlite_corpus_test.c
	$(CC) $(CFLAGS) tests/sqlite_corpus_test.c -lsqlite3 -o tests/sqlite_corpus_test
sqlite-test:
	@$(MAKE) --no-print-directory -B sqlite_corpus tests/sqlite_corpus_test
	./tests/sqlite_corpus_test

# Independent prose learning diagnostic, not a replacement for Q&A qualification.
prose-test:
	@echo 'Prose training diagnostic: bounded to 180 seconds.'
	@timeout -v -k 5s 180s sh ./tests/prose_qualification.sh
