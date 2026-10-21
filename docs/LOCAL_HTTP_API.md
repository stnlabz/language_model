# Experimental local HTTP inference API

This is an explicitly started, foreground-only, loopback-bound interface to the existing STN-Labz GRU executable. It is not a deployment, agent integration, public service, conversational qualification, or authorization to change Digit/Ezra/RictusD.

## Start (on VPS, after pulling GitHub main)

    make build
    python3 -m unittest discover -s tests -p 'test_http_api.py'
    make api CHECKPOINT=corpus_state_768000.stn PORT=8765

The server binds only to 127.0.0.1 and serves one request at a time. Ctrl-C stops it. Do not expose or reverse-proxy it outside the machine; there is no authentication or TLS. It does not auto-start or install a service.

## Request

    curl -sS -X POST http://127.0.0.1:8765/generate \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"The engineer ","length":128,"seed":913,"temperature":0.65,"penalty":0}'

Returns JSON with text, generated_bytes, model, experimental fields.

GET /health returns a process health indicator; it does not evaluate the checkpoint or verify generation quality. All other paths return 404.

Only /generate accepts POST JSON. Fields: prompt required (nonempty UTF-8 text, at most 1024 bytes, no NUL); optional length 1..256, seed nonzero unsigned 64-bit, temperature 0.25..2.0, penalty 0..2.0. Unknown fields are rejected. Maximum HTTP body is 4096 bytes. Generation invokes the existing model sample subprocess with a 30-second timeout; process errors produce HTTP 502. Generation returns a UTF-8 string decoded from the model's hexadecimal byte output.

Loopback binding is the only access restriction; any local process able to connect can call the API. Do not pass secrets, policy enforcement decisions, or sensitive Corpus/Context data as prompts. The adapter suppresses standard HTTP access logging of prompts, but requests may still be visible to machine administrators or process monitoring tools.

## Qualification

Output remains experimental and may be incoherent or factually wrong. The API does not interpret commands or take agent actions. No files or checkpoints are modified during requests. The original standalone model command is preserved.


## Optional POST /learn ingestion (not training)

Only send Digit examples separately approved for transfer. The handler stores question/answer pairs without modifying GRU weights or checkpoints. It does not independently verify whether the caller's approved=true claim is justified.

Start the API with a private learning queue, **without any token**:

    python3 http_api.py --model ./model --checkpoint corpus_state_768000.stn \
      --port 8765 --learn-file ./digit_approved_learning.jsonl

The queue is created with permissions 0600 if not present. If --learn-file is omitted, the endpoint returns HTTP 503. Keep the queue outside the web root.

Digit must run on the **same VPS** and send:

    curl -i -X POST http://127.0.0.1:8765/learn \
      -H 'Content-Type: application/json' \
      -d '{"input":"Why is PHP hard?","response":"PHP syntax may be unfamiliar.","source":"digit","approved":true}'

The server accepts the exact required JSON fields input, response, source, and approved. input is 1..1024 UTF-8 bytes, response 1..2048 UTF-8 bytes, source must be digit, approved must be true, and total body must fit within 4096 bytes. Extra fields are rejected.

On success, HTTP 202:

    {"accepted":true,"trained":false,"status":"queued_for_review"}

Errors include HTTP 400 for invalid records, 413 for oversized request, 503 when ingestion is disabled, and 502 for write failures.

No X-STN-Learn-Token header, STN_LABZ_LEARN_TOKEN variable, or authentication configuration is needed. **This removes the only /learn authentication barrier:** any local process that can connect to 127.0.0.1:8765 can submit approved=true records. Do not proxy or expose the service publicly. Digit-side example selection and provenance approval remain required.

The endpoint does NOT train, convert JSONL into a corpus, update checkpoints, or activate model versions.
