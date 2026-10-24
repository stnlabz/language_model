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

## Separate dialogue provenance (2026-10-24)

Windows `build go` now invokes the native JSONL converter with a separate output file, `runs\\gru_run\\dialogue_corpus.txt`. It contains **only** originally approved records whose inputs are not classified as Digit Corpus, Context, or Observation records. The primary `digit_corpus.txt` retains the current authorized sentence-training content, including those real dialogue pairs, so this change does not silently alter the neural training distribution. No artificial prompts, converted technical assertions, repeated dialogues, or newly asserted approvals are introduced.

The converter writes the same `dialogue_fraction_percent` and `conversation_data_gate` results to `migration.log`. Windows GO copies the insufficient-data finding to `report.txt`, and prints a visible warning before training. When that gate is insufficient, GO remains a **sentence-structure experiment**. The resulting checkpoint remains `language_quality=NOT_VERIFIED` irrespective of validation loss or the quality of its training-time constrained sample. The `live_*.txt` outputs are generated through the production sampling path and expose that distinction.

**Not fixed by data separation:** The current word-level GRU has no explicit question/answer training objective, and the last observed approved queue contained only seven genuine dialogue records. More general lesson statements, additional epochs, or a low validation loss cannot substitute for approved, diverse dialogue pairs and a separately authorized model/training-objective change. The operator must review the isolated dialogue corpus and approve additional genuine training material before conversation readiness can be established. No new model, checkpoint, or agent deployment is performed by this source change.

## Supervised answer objective (2026-10-24)

Windows `build go` now passes the isolated, operator-approved `dialogue_corpus.txt` to `word_gru train` as an optional fifth training argument (after `SAMPLE`). The baseline language training continues to use `digit_train.bin` and `digit_valid.bin`, selecting its best held-out-loss checkpoint as before. The separate dialogue stage fine-tunes that best checkpoint with **one supervised pass through each genuine question/answer pair**. It does not duplicate records or invent answers.

The dialogue tokenizer keeps each `Question: … Answer: …` record together: question-mark punctuation does not reset the GRU between prompt and answer. Its predictive loss is **masked on question-side tokens**, and applied to answer words and the final end-of-record token. Earlier question-side recurrent states can influence answer prediction within the 24-token truncated backpropagation window. The optimizer's moments are reset before the separate objective. In `word_gru.log`, `dialogue_pairs_supervised` and `answer_tokens_supervised` record what actually participated.

This change deliberately preserves the existing checkpoint serialization, general word-level inference interface and agent-side Core/retrieval boundaries. The optional argument retains the prior four-file training invocation for other workflows. The current seven approved dialogues are still sparse; this is an implementation of a conditional objective, **not** proof of useful conversations. The best-validation-loss figure describes the base language checkpoint **before** supervised dialogue fine-tuning. Windows GO keeps `language_quality=NOT_VERIFIED` and prints production-decoder samples for operator review. No deployment or agent modification is authorized by `build go`.

### Prompt conditioning in inference

The word-GRU `sample` command now consumes the live prompt with the same Q/A context tokens used in supervised training: `question [prompt] answer [generated tokens]`. The trained checkpoint must therefore be used with the matching updated `word_gru` binary; existing checkpoint files remain readable without a serialization change, but generated behavior may differ even for old checkpoints. The sampler still uses model probabilities and never retrieves a memorized answer directly.

Windows GO first executes `word_gru test-dialogue`, a deterministic source-only fixture checking the Q/A boundary, answer loss positions, and rejection of missing or empty answers. Its result is written to `dialogue_fixture.log`; failure stops the run before training. This is a parser qualification fixture, not a language qualification test. The operational word GRU and checkpoint must be tested by the Operator on Windows before any decision to replace Digit's running language model.
