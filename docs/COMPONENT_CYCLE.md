# STN-Labz LM component contract

This repository separates three responsibilities, as directed by the Operator.

1. `gru_learn.py`: reads explicitly approved JSONL via `prepare_jsonl.py`, creates temporary prepared text and invokes the native LM trainer; creates a new checkpoint without overwriting existing files. The component name does **not** indicate that the model uses a GRU.
2. `micro_llm.c`: owns the native CPU Transformer trainer, model checkpoint format and inference. Model artifacts use `.stn` format and are version-checked.
3. `llm_service.py`: explicitly started loopback HTTP inference gateway to an existing checkpoint. No learning is performed by the service, and no agent integration or automatic daemon installation is performed.

## Commands

```bash
make build
make learn DATA=digit_approved_learning.jsonl CHECKPOINT=llm.stn STEPS=5000
make generate CHECKPOINT=llm.stn PROMPT="The engineer"
make serve CHECKPOINT=llm.stn
```

Requests when the service is explicitly started:

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The engineer","max_bytes":64,"seed":913}'
```

The learning stage only consumes records accepted by the preparation script. The source JSONL is not changed. New approved learning requires a new explicitly invoked training operation and a new checkpoint name. Service reload/model swap is *not* automatic.

## Acceptance limitation

Component separation does not establish linguistic competence. The original 96-byte byte-level Transformer and limited training data have failed coherent language-generation checks. An ASCII output restriction only prevents invalid characters; it is not a language-quality fix. Do not qualify or deploy the model based only on compilation, falling loss, a healthy HTTP endpoint, or the presence of a checkpoint. Evaluate held-out loss and unseen prompt-response relevance before acceptance. No agent integration, server setup, or automatic systemd changes are authorized by these source files.
