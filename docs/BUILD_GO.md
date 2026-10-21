# build go — native Word GRU

The Operator-authorized language-learning replacement uses `word_gru.c`: an original, C11, word-token gated recurrent unit. It trains from the approved Digit JSONL converted into line-oriented lessons by `migrate.c` and partitioned into train/validation by `prepare.c`. No external pretrained weights, inference services or runtime dependencies are used.

Run `build go` in an MSVC Developer CMD. Existing `runs` content is deleted at the start of each run, and results are written to `runs\gru_run` only.

The training run creates `word_gru.stn` (the lowest-held-out-loss weight snapshot including training vocabulary), `word_gru.log`, and `best_sample.txt`. The sample comes from sampling the word GRU distribution; it is not a copied approved sentence or a hardcoded answer. `build go` displays the sample in CMD.

**Qualification boundary:** A successful build, decreasing word-token loss, or valid spelling does **not** prove coherent language. Inspect real generated sentences and classify incoherent output as FAIL. The checkpoint file format is incompatible with the historical byte-level `model.c` checkpoints. This experimental replacement does not change Digit, Ezra, other agents, or deployments.
