# STN-Labz GRU API as a systemd service

This is an opt-in deployment recipe, NOT an automatic installation. No change is made to Digit, Ezra, or RictusD. The service binds the HTTP API to 127.0.0.1:8765 only, supports existing /health, /generate and /learn endpoints, and **does not train**. The queue is still a JSONL file.

## Before installation

Confirm the model and checkpoint exist, and ensure no other process already occupies port 8765. Stop any existing foreground instance of http_api.py before launching systemd (Ctrl-C in its terminal). Review the unit at deploy/stn-labz-gru-api.service before installing it. It expects the VPS paths /home/stnchain/language_model and /usr/bin/python3; adjust paths before installation if they differ.

Never expose this HTTP port over the public network: neither inference nor learning ingestion requires authentication. Any local process can submit a learning record.

## Install (operator-issued commands)

    cd /home/stnchain/language_model
    git pull origin main
    make build
    python3 -m unittest discover -s tests -p 'test_http_api.py'
    test -s corpus_state_768000.stn

Create the approved learning queue *before* starting the service. If it exists, preserve the existing entries:

    touch /home/stnchain/language_model/digit_approved_learning.jsonl
    chmod 600 /home/stnchain/language_model/digit_approved_learning.jsonl

No token or environment file is required. Then install and start:

    sudo install -m 644 deploy/stn-labz-gru-api.service /etc/systemd/system/stn-labz-gru-api.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now stn-labz-gru-api.service

## Verify

    systemctl status stn-labz-gru-api.service --no-pager
    curl -i http://127.0.0.1:8765/health
    journalctl -u stn-labz-gru-api.service -n 30 --no-pager

Test /learn from the same VPS without an authentication header and verify the queue line count increases. HTTP 202 means stored, not trained.

## Manage and rollback

    sudo systemctl restart stn-labz-gru-api.service
    sudo systemctl stop stn-labz-gru-api.service
    sudo systemctl disable --now stn-labz-gru-api.service

Stopping or disabling does not delete queue records or checkpoints. Avoid running a second foreground API on port 8765 while systemd is active.

## Security and limitations

- The unit runs as stnchain, not root; bind remains hardcoded to loopback.
- There is no secret token or authentication header. Localhost-only binding is the sole network boundary; local processes can submit records.
- ProtectSystem=strict and a narrow ReadWritePaths restrict writes to the existing learning queue. PrivateTmp and NoNewPrivileges add isolation.
- The checkpoint and executable must remain readable to stnchain. Queue ownership must allow stnchain to append.
- This is a single-threaded experimental Python HTTP server. There is no rate limiting, TLS, request authentication on either /learn or /generate, request concurrency, or guaranteed training.
- systemd automatically restarts after failures. It will not restart for an intentional systemctl stop.
- If switching checkpoints later, explicitly review and update the service unit, then daemon-reload and restart; no automatic checkpoint promotion.

## Validation gate

Only unit and documentation are supplied here; this does not claim that systemd was installed or run. Check runtime results before considering deployment accepted.
