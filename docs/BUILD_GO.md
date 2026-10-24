# build go — native Word GRU

The Operator-authorized language-learning replacement uses `word_gru.c`: an original, C11, word-token gated recurrent unit. It trains from the approved Digit JSONL converted into line-oriented lessons by `migrate.c` and partitioned into train/validation by `prepare.c`. No external pretrained weights, inference services or runtime dependencies are used.

Run `build go` in an MSVC Developer CMD. Existing `runs` content is deleted at the start of each run, and results are written to `runs\gru_run` only.

The training run creates `word_gru.stn` (the lowest-held-out-loss weight snapshot including training vocabulary), `word_gru.log`, and `best_sample.txt`. The sample comes from sampling the word GRU distribution; it is not a copied approved sentence or a hardcoded answer. `build go` displays the sample in CMD.

**Qualification boundary:** A successful build, decreasing word-token loss, or valid spelling does **not** prove coherent language. Inspect real generated sentences and classify incoherent output as FAIL. The checkpoint file format is incompatible with the historical byte-level `model.c` checkpoints. This experimental replacement does not change Digit, Ezra, other agents, or deployments.

## Word-order decoding (2026-10-21)

The word GRU retains learned next-word probabilities, but generation now restricts adjacent word choices to transitions actually observed in Digit's training split. Transition frequency and the GRU probability both influence sampling. Only the training split is used for these constraints. This reduces unsupported local word combinations but does not prove whole-sentence grammar, factuality, originality, or meaning. A generated sample must still be reviewed for coherence; an incoherent sample is a FAIL even if the process exits successfully.

## Windows GO diagnostic correction (2026-10-24)

The operator's October 24 training snapshot contained 89,263 approved records, but the native converter classified only 7 as dialogues and 85,087 as statement lessons. The label `lessons` does not establish that those records teach conversational turn-taking. The converter now emits `training_records`, `dialogue_fraction_percent`, and `conversation_data_gate` into `migration.log`. A low dialogue fraction is a warning, not grounds to silently invent, duplicate, or reclassify records; it does not prevent an explicitly requested training experiment.

After word-GRU training, `build go` now generates three additional prompt-conditioned outputs with the *same* `word_gru sample` decoder used by the inference API, saving them as `live_identity.txt`, `live_mission.txt`, and `live_greeting.txt` under `runs\\gru_run`. The prior `best_sample.txt` is still generated and displayed, but is explicitly labeled **training-time trigram constrained**. That decoder relies on training-split word transitions absent from the deployed checkpoint. Never infer live language quality from the training-constrained sentence.

The run retains `language_quality=NOT_VERIFIED` and `training_completed_language_unqualified`. Inspect the new live samples and independently qualify the exact checkpoint before human-authorized deployment; a few readable outputs alone cannot establish factuality, relevance, turn-taking or generalization. This correction neither trains a checkpoint nor deploys it.
