# Digit JSONL -> STN-Labz GRU corpus migration

Scope: explicit, read-only snapshot conversion. No change to Digit, the live
HTTP ingestion queue, the GRU model architecture, checkpoints, or training.

## Source format

The existing /learn endpoint writes one JSON object per line with fields
input, response, source="digit", approved=true. The converter retains
question-and-answer adjacency as a *single* newline-terminated record:

    Question: Why does PHP seem hard? Answer: PHP syntax may be unfamiliar.

That single record is compatible with prepare.c's record-wise deterministic
train/validation partition. Newlines inside fields are normalized to spaces.
The converter rejects malformed, incomplete, unauthorized, or oversized
snapshots. It never overwrites an output file. Duplicate identical exchanges
are collapsed within each snapshot. Each run reports SHA-256 hashes, record
counts and output byte size. **It does not automatically deduplicate against
previous migrations.** Use a new snapshot and a new run label; avoid merging
overlapping exports without separate record-tracking controls.

## Live ingestion / snapshot rule

Digit continues appending to digit_approved_learning.jsonl. Do NOT use the
live file as the converter input. Take an immutable snapshot first while the
source is quiescent or using an operationally consistent snapshot method; a
copy made during a write can end with a partial JSON record. The converter
rejects such partial snapshots instead of silently discarding data.

Linux:

    cp digit_approved_learning.jsonl digit_snapshot_20261021.jsonl
    python3 tools/migrate_digit_jsonl.py \
      --input digit_snapshot_20261021.jsonl \
      --output digit_migration_20261021.txt

Windows CMD, after transferring the consistent snapshot:

    python tools\migrate_digit_jsonl.py --input digit_snapshot_20261021.jsonl --output digit_migration_20261021.txt

The sample snapshot examined during development contained 3,667 records;
3,662 input labels were formatted as Digit Context, Corpus, or Observation
records. They are NOT ordinary conversational questions. All records
marked approved=true are migrated by default, but do not assume every such
record is high-quality natural-language training data.

For a narrower research set:

    python3 tools/migrate_digit_jsonl.py --input digit_snapshot_20261021.jsonl --output digit_dialogue_20261021.txt --mode dialogue

This excludes records whose *input* starts with "Digit Context record ",
"Digit Corpus record ", or "Digit Observation record ". It is a format-based
filter, not a semantic or provenance approval review. Mode dialogue might
have too few records for train/validation splitting; check counts and content.

## Prepare data only after review

Review the output text for quality, privacy, authorization, and source
provenance. If accepted, use the existing preparer with **new output names**:

Linux:

    ./prepare prepare digit_migration_20261021.txt digit_v1_train.bin digit_v1_validation.bin

Windows CMD:

    build\prepare.exe prepare digit_migration_20261021.txt digit_v1_train.bin digit_v1_validation.bin

This creates byte-level train/validation datasets without starting training.

IMPORTANT: The existing stateful GRU resume command verifies the SHA/length
identity of its original training partition (an internal FNV64 checksum).
It will reject a changed corpus. These newly prepared files cannot simply
replace corpus_v1_train.bin in a resume from corpus_state_768000.stn.
Continuation on a new dataset while preserving existing weights is a
separate, not-yet-implemented and separately authorized training step.

Do not promote a checkpoint or feed converted observations into running
inference automatically. The word "approved" in incoming JSONL is a
caller assertion and does not independently verify reliability.
