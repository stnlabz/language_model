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
