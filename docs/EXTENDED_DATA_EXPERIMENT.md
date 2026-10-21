# Extended Synthetic Text Experiment

**Status:** Proposed for operator-run testing. This document does not claim qualification.

## Data provenance and scope

`samples/synthetic_language_extended.txt` is newly authored synthetic English text for this repository. It contains 100 newline-delimited records (8,734 bytes in the committed UTF-8 source), covering diagnostics, engineering, software, language learning, everyday activities, operations and reporting. Some records use repeated introductory phrasing; the text is not an independently sampled language corpus. No Digit Corpus, Observations, Context, logs, third-party documents or pretrained data were copied.

The existing `samples/synthetic_language.txt` and earlier checkpoints are unchanged.

## Experiment

1. Prepare the extended sample with the existing C11 `prepare` program, saving to **new** filenames `train_extended.bin` and `validation_extended.bin`.
2. Run the unchanged 64-hidden-state GRU for 5,000 training positions, using `prototype_extended_5000.stn` as a **new** checkpoint.
3. Evaluate on `validation_extended.bin`, including the frequency baseline computed from `train_extended.bin`, UTF-8 validity, and next-byte prompt sensitivity.
4. Record training, validation, memory use, run duration, output validity, and failures. Preserve all old results and checkpoints.

Use low CPU scheduling priority and a bounded runtime on the shared VPS. Never run training as a background service. Avoid overwriting files. Stop on errors.

### Commands (from repository root; after compiling current source)

```sh
./prepare prepare samples/synthetic_language_extended.txt train_extended.bin validation_extended.bin
nice -n 10 timeout 120s ./model train train_extended.bin validation_extended.bin 5000 prototype_extended_5000.stn
python3 tests/evaluate_language.py --model ./model --checkpoint prototype_extended_5000.stn --train train_extended.bin --validation validation_extended.bin
```

If the timeout expires or output is absent, do not evaluate a nonexistent checkpoint. Report the limitation.

## Interpretation controls

- Compare each model against the **frequency baseline on the same dataset and validation split**. Raw validation loss across different datasets is not a fair improvement claim.
- This dataset is roughly five times the original source size but remains very small, synthetic, and linguistically narrow.
- The current `model train` command uses **training positions/steps**, not epochs. At 5,000 steps it cycles through the training text, and the exposure count differs from the original dataset.
- The deterministic record-level split does not guarantee independence of shared phrasing or themes; leakage through similar structures may bias the evaluation.
- Better prediction does not establish understanding, accuracy, useful conversation or agent readiness.
- Do not infer that more training data alone repairs model capacity or optimization weaknesses.

**Next gate:** Operator inspects measured results and explicitly accepts or rejects the experiment. Further model changes, ingestion, and deployment require separate authorization.
