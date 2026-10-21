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
4. Convert approved JSONL records into single-line Question/Answer examples; reject incomplete or unauthorized rows.
5. Prepare train and held-out validation bytes using existing prepare.exe.
6. Evaluate original checkpoint on NEW validation data.
7. Adapt original weights on new data for 64,000 positions and save a checkpoint with the NEW training-data hash.
8. Resume three more 64,000-position increments on that same new data; 256,000 positions total for this run.
9. Evaluate after each increment against the same new holdout and track the best held-out loss, keeping all checkpoints.
10. Generate a fixed-seed sample from the best-loss checkpoint; write all results to the unique run folder.

The original resume checksum guard is preserved; model adapt is a distinct, explicit operation for the changed corpus, and it never changes the source checkpoint.

## Output

Each run writes logs and files in runs/go_... including report.txt, digit_snapshot.jsonl, digit_corpus.txt, digit_train.bin, digit_valid.bin, baseline_eval.log, train_*.log, eval_*.log, state_*.stn, and best_sample.hex.

A failed stage stops the run, retains partial artifacts, and marks report.txt FAILED. Existing checkpoints and the original input remain unchanged. The machine must stay awake, and CMD must remain running. No automatic deployment to the VPS occurs.

## Limits

- CPU-only native training; no GPU backend added.
- Training data requires >=5 complete, approved input records with exactly the input, response, source, approved fields; source must be digit. Output must remain within the existing 16 MiB preparer cap.
- The converter does not remove duplicates or filter structured Digit Context/Corpus/Observation records. Review provenance, quality, and repeated content before running.
- The Windows snapshot copy cannot guarantee atomicity if the file is actively written at the same path. Use a consistent transferred snapshot. A truncated final record is rejected.
- A lower held-out next-byte loss does not prove fluency or answer quality. Manual language review remains required.
- New held-out loss is NOT directly comparable to the 1.626341 result on the earlier corpus.
- The pipeline runs one bounded job and exits; it does not poll for new records, restart itself, or alter any agent or systemd service.
