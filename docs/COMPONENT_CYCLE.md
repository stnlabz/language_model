# STN-Labz Native C Learning / Model / Service Cycle

All runtime components are Native C11/POSIX. No Python interpreter, scripts, or HTTP frameworks are required.

1. `gru_learn.c` (`gru_learn`) reads the **approved** JSONL, decodes approved input/response or statement records, writes a temporary text corpus, and invokes the C model trainer. The original JSONL is read-only. The GRU name denotes the learning stage; the underlying language model is a Transformer.
2. `micro_llm.c` (`micro_llm`) trains the native causal Transformer and stores a versioned `.stn` checkpoint; it also provides generation.
3. `llm_service.c` (`llm_service`) explicitly serves an existing checkpoint over a 127.0.0.1-only HTTP interface. It neither trains nor installs a daemon.

Build the source:
```sh
make build
make test
```

Perform new learning only when explicitly requested, using a **new** checkpoint filename:
```sh
make learn DATA=digit_approved_learning.jsonl CHECKPOINT=llm_new.stn STEPS=5000
```

Run local generation or manually start the service:
```sh
make generate CHECKPOINT=llm_new.stn PROMPT="The engineer"
make serve CHECKPOINT=llm_new.stn PORT=8765
```

Service API (only while `make serve` is running):
```sh
curl -sS http://127.0.0.1:8765/health
curl -sS -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The engineer","max_bytes":64,"seed":913}'
```

## Known limitations

The C JSONL converter accepts approved flat JSON records with strings and scalar metadata; malformed/unsupported JSON aborts rather than silently training on corrupted data. Output contains only records meeting the existing approved-data extraction criteria. The learner refuses to overwrite an existing checkpoint.

The native HTTP server is a **local development integration surface**, not an authorized production daemon. It uses a single process and executes the native generator per request; it is loopback only. No existing AI-agent integration was changed.

**Model quality remains unqualified:** past prompt-generation tests failed. Component replacement is not a claim of coherent language learning; acceptance still requires held-out evaluation and useful prompt-conditioned generation. No systemd files or deployment changes are included.
