# Windows CMD unattended build go

The command build go compiles native C11 components with MSVC cl.exe and runs one finite migration and training experiment. No Python, PowerShell, CMake, or Visual Studio project is needed.

## Prerequisites

Place these files in the Windows repository working directory:
- digit_approved_learning.jsonl — a consistent, approved snapshot transferred from Digit. The source may keep growing on its original host.
- corpus_state_768000.stn — original compatible checkpoint.

From Windows Developer CMD:

    git pull origin main
    build go

## Stages

1. Build model.exe, prepare.exe, migrate.exe, and compare_loss.exe with cl.exe.
2. Perform native synthetic corpus prepare/train/eval smoke checks.
3. Copy digit_approved_learning.jsonl into a unique runs/go_... directory.
4. Convert approved dialogue records into Question/Answer examples and Digit Corpus records (including LESSON and interface:/input) into plain statements. Keep Context and Observation records untouched in the frozen JSONL snapshot rather than pretending they are conversations. Reject incomplete or unauthorized rows.
5. Report counts of Corpus statements, dialogues, retained Context, and retained Observations. Require at least five usable lesson/dialogue records; prepare training and validation bytes from those records only.
6. Evaluate original checkpoint on NEW validation data.
7. Adapt original weights on new data for 64,000 positions and save a checkpoint with the NEW training-data hash.
8. Resume three more 64,000-position increments on that same new data; 256,000 positions total for this run.
9. Evaluate after each increment against the same new holdout and track the best held-out loss, keeping all checkpoints.
10. Generate a fixed-seed sample from the lowest-loss checkpoint; write results to the unique run folder. Mark language quality NOT VERIFIED even if training completes.

The original resume checksum guard is preserved; model adapt is a distinct, explicit operation for the changed corpus, and it never changes the source checkpoint.

## Output

Each run writes logs and files in runs/go_... including report.txt, digit_snapshot.jsonl, digit_corpus.txt, digit_train.bin, digit_valid.bin, baseline_eval.log, train_*.log, eval_*.log, state_*.stn, and best_sample.hex.

A failed stage stops the run, retains partial artifacts, and marks report.txt FAILED. Existing checkpoints and the original input remain unchanged. The machine must stay awake, and CMD must remain running. No automatic deployment to the VPS occurs.

## Limits

- CPU-only native training; no GPU backend added.
- Training data requires >=5 complete, approved input records with exactly the input, response, source, approved fields; source must be digit. Output must remain within the existing 16 MiB preparer cap.
- The converter does not remove duplicates. It explicitly separates Corpus statements and dialogues from Context and Observation records. All original records remain in the immutable run snapshot. An approved=true field and a Corpus label are not independent verification of quality or truth.
- The Windows snapshot copy cannot guarantee atomicity if the file is actively written at the same path. Use a consistent transferred snapshot. A truncated final record is rejected.
- A lower held-out next-byte loss does not prove fluency or answer quality. Manual language review remains required.
- New held-out loss is NOT directly comparable to the 1.626341 result on the earlier corpus.
- The pipeline runs one bounded job and exits; it does not poll for new records, restart itself, or alter any agent or systemd service.

The model is not considered conversationally usable merely because training returns success or validation loss improves. This migration preserves the structured data without treating Context/Observations as conversational training. It does not repair GRU language generation or provide an automated semantic quality test.

## Snapshot observed 2026-10-21

Inspected attached 3,667-line snapshot: 1,215 Corpus LESSON records, 6 Corpus interface:/input records, 1,221 Context records, 1,220 Observation records, and 5 dialogue records. The earlier LESSON-only converter failed on the six interface:/input records. Both Corpus categories now migrate as statements, not fabricated questions. Context and Observations stay in the run snapshot. The 1,215 LESSON entries were marked unverified in their input labels; the accepted JSONL approved=true flag does not independently validate their contents. This fix restores migration compatibility; it cannot guarantee meaningful language output.
