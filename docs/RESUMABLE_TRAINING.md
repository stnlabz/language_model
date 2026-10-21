# Resumable GRU training — operator-run gate

The original `model train` command and version-1 weight checkpoints are preserved. Stateful checkpoints use version 2 and contain weights, a completed-position counter, a checksum of the training bytes, and their length. The existing `eval`, `generate`, and `probe` commands can read both formats; only version-2 checkpoints can be resumed.

Each stateful training or resume session must use a multiple of 64 training positions (maximum 100,000 positions per invocation). This preserves the existing 64-step gradient update boundaries and permits byte-identical checkpoints for equally sized total training runs (for example 256 vs. 128+128), provided the same data, build, and floating-point environment are used. Training positions wrap around at the end of the training bytes. Each invocation still resets the hidden state at the start of each 64-position update as in the existing trainer.

No optimizer momentum is stored because the current optimizer uses direct clipped gradient updates and does not maintain momentum. Version-1 checkpoints such as `corpus_v1_5000.stn` **cannot be resumed safely**: they contain no trustworthy position or training-data identity. Keep them for evaluation and generation.

On the VPS, after `git pull`, run:

```sh
make test
make train-state RUN=corpus_v1 STEPS=6400 STATE_OUT=corpus_state_6400.stn
make resume RUN=corpus_v1 STEPS=6400 STATE_IN=corpus_state_6400.stn STATE_OUT=corpus_state_12800.stn
make evaluate RUN=corpus_v1 STEPS=12800
```

**Note:** the existing `make evaluate` uses `$(RUN)_$(STEPS).stn`, not `STATE_OUT`. For a stateful checkpoint, run the evaluator directly, or copy no files and invoke:

```sh
python3 tests/evaluate_language.py --model ./model \
  --checkpoint corpus_state_12800.stn \
  --train corpus_v1_train.bin --validation corpus_v1_validation.bin
```

Training is explicitly opt-in. `nice` and `timeout` bound each run on the shared VPS. Outputs must have new, unused paths and are not intentionally overwritten. The checkpoint is written only after a completed training invocation; an interrupted invocation cannot resume its incomplete segment. If interrupted, resume from the previous successfully written state checkpoint.

The data checksum is an integrity check, not a proof of source authorization or provenance. Training quality, sentence generation, runtime performance, and VPS integration are not qualified by implementing resume support.
