# Position versus content diagnosis — 2026-10-25

## Status and provenance

Operator update at 17:55Z: native tests, SQLite isolation/deduplication, regression, mathematics, training and prose PASS; generalization, language and validation FAIL; overall NOT QUALIFIED. SQLite status is operator-reported, not independently tested in this investigation.

The checkout changed during the investigation. The original `qualification-dpkPzY` checkpoint and first diagnostic output directory were removed by external activity. Its earlier results are historical evidence only. The current run uses source HEAD `0153e99`, a frozen copy of `tests/qualification-y7ehSU/model.stn`, and the recorded source/checkpoint SHA-256 values in `tests/position-probe-20261025-current/inputs.sha256`. That qualification log reports 1/8 trained continuations and 0/4 held-out continuations. It is a different training fixture from the previous eight-paraphrase run; do not compare them as the same experiment.

No optimizer updates, new training, architecture edits, deployment, service activation, or GitHub changes were made. Production source and checkpoint hashes remained unchanged during this run.

## Findings

**Confirmed: strong dependence on absolute prompt position, with some content sensitivity. This is not evidence of a purely position-only model, nor proof of the sole remaining root cause.**

The frozen model was evaluated with teacher-forced expected answers (each true preceding answer byte supplied), and with unfiltered greedy generation. Mean negative log likelihood (NLL) is per answer byte; lower is better. Comparing candidate NLLs is diagnostic and is not a calibrated semantic classification score.

| Intervention | Engineer correct-answer NLL | Gardener correct-answer NLL |
|---|---:|---:|
| Original trained short question | 0.580731 | 0.321445 |
| Add one leading space | 5.069516 | 3.226936 |
| Change question mark to period, same length | 0.583807 | 0.325558 |
| Erase subject with equal-length x characters | 0.568875 | 0.431193 |
| Erase other question wording, keep subject and positions | 0.619027 | 0.327470 |

Moving the unchanged question by one byte damages prediction far more than several same-position content changes. A leading space also adds an attended token, so this intervention alone cannot mathematically isolate positional embeddings from all attention effects. The combined controls establish severe positional sensitivity, not exclusive attribution to one tensor.

Equal-length subject swaps keep all byte positions fixed:

- Replace engineer with gardener while retaining the original frame: gardener NLL 3.112357 versus original engineer NLL 0.976232.
- Replace gardener with engineer while retaining the original frame: engineer NLL 4.703745 versus original gardener NLL 0.726340.

These article-preserving swaps introduce an article/subject grammar mismatch (`an gardener`, `a engineer`), so they are diagnostic counterfactuals rather than grammatical language-acceptance tests.

A grammatical same-length control exists in the actual training fixture: gardener and mechanic both use 37-byte prefixes. The mechanic prompt favors its own answer (NLL 0.402949) over the gardener answer (0.555675), and its output begins ` A mechanic`. Content therefore affects prediction; claiming that the model ignores all question content would be false.

Teacher-forced accuracy also masks free-running degeneration:

- Engineer: 25/29 next bytes correct with the true prefix; greedy output ` An engineer des.st..........`.
- Gardener: 24/25 next bytes correct with the true prefix; greedy output ` A gachanicwrepairs merre`.
- Mechanic: 26/29 next bytes correct with the true prefix; greedy output ` A mechanic achairs mars.....`.

This directly shows that a few next-byte mistakes can derail continuation. It does not establish that increasing training steps fixes that behavior.

## Diagnostic correction

`tests/context_utilization.c` previously compared `The morning air is` with `The machine part is`. Their byte lengths differ, so learned absolute positions could explain some of the reported context sensitivity.

The corrected test uses `The fresh bread is` and `The metal plate is`, asserts equal byte lengths, and retains the identical final suffix ` is`. On the frozen current checkpoint:

```
context_sensitivity_l1=0.23349160
context_top_bytes=46,46
```

The contexts change the distribution, but both select a period. Distribution sensitivity is not coherent language generation and is not a qualification gate.

`tests/position_probe.c` provides the reproducible frozen-model matrix: original and reference phrasings, padding by 1–12 bytes, punctuation/case/separator changes, subject/wording erasure, subject swaps, and the grammatical same-length mechanic control. The three `reference_*` phrasings are not asserted to be trained in the current eight-profession fixture.

## Tests and artifacts

- Both diagnostic programs compiled with C11, `-O2 -Wall -Wextra -pedantic`, without warnings after the diagnostic formatting correction.
- All runs use the production forward pass and original checkpoint parameters. No answer replacement or decoding filters.
- Source/checkpoint SHA-256 verification: PASS.
- `git diff --check`: PASS.
- Full CSV, executable snapshot, source snapshot, checkpoint copy, and hashes: `tests/position-probe-20261025-current/`.

Reproduce from the repository root:

```
cc -std=c11 -O2 -Wall -Wextra -pedantic tests/position_probe.c -lm -o tests/position-probe-20261025-current/probe
./tests/position-probe-20261025-current/probe tests/position-probe-20261025-current/frozen.stn
```

These results support a next controlled experiment that balances answer positions across concepts at a fixed training budget, with completely separate held-out wording/positions. That would test whether the training distribution creates the position shortcut before proposing positional-encoding or attention architecture changes. Neither additional training nor an architecture change was performed here. Overall language qualification remains FAIL.
