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

Only use examples you have separately authorized to provide from Digit; its Observations/Corpus/Context are not automatically permitted. The handler stores question/answer pairs without modifying GRU weights, training files, or checkpoints.

Start the API **manually** with a private ingest destination and shared token in the same shell:

    export STN_LABZ_LEARN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    python3 http_api.py --model ./model --checkpoint corpus_state_768000.stn \
      --port 8765 --learn-file ./digit_approved_learning.jsonl

The token must be passed to Digit through a separately approved method; do not commit it or place it in source files or command-line arguments. The queue is created with mode 0600. Store the queue outside any public web root and avoid submitting private conversation data. If --learn-file is omitted, the endpoint returns 503 and does not accept data. Restart the server after changing configuration.

Digit should send an HTTP POST to http://127.0.0.1:8765/learn on the same host, with Content-Type: application/json and X-STN-Learn-Token header containing the token. The exact JSON payload is:

    {
      "input": "Why does PHP seem difficult?",
      "response": "PHP may seem difficult because its syntax and execution model are unfamiliar.",
      "source": "digit",
      "approved": true
    }

All four fields are required, and extra fields are rejected. input must be 1..1024 UTF-8 bytes, response 1..2048 UTF-8 bytes, source must literally be digit, and approved must be true. The complete body must fit within 4096 bytes.

Example after setting the token above:

    curl -sS -X POST http://127.0.0.1:8765/learn \
      -H "X-STN-Learn-Token: $STN_LABZ_LEARN_TOKEN" \
      -H 'Content-Type: application/json' \
      -d '{"input":"Why is PHP hard?","response":"PHP syntax may be unfamiliar.","source":"digit","approved":true}'

On success the server returns HTTP 202 with:

    {"accepted":true,"trained":false,"status":"queued_for_review"}

Other results: 400 invalid record, 403 absent/bad token, 413 body too large, 503 learning disabled, 502 file write failed. The handler is synchronous and the server accepts one request at a time. Do not confuse queued_for_review with verified provenance or approved training.

The server cannot independently prove that a caller's approved=true claim is true. Digit-side selection and human-approved data provenance remain mandatory. This endpoint does **not** start training, convert JSONL into the existing newline-text corpus format, update weights, or activate a new checkpoint. Those are separate future development and operator approval gates.
