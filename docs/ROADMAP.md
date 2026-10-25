# STN-Labz Language Model — Development Roadmap

**Status (2026-10-25):** Native prototype, SQLite lesson export, model training, and checkpoint evaluation are implemented. Milestone 4 (language-learning evaluation) is active; generalization, language, and validation FAIL. **Overall: NOT QUALIFIED.** This roadmap itself grants no implementation, training, deployment, or milestone-advancement authority.

## Mission

Develop one STN-Labz-native language model to provide reusable human-language understanding and generation for Digit, Ezra, and potentially RictusD and future agents, without external pretrained models or remote model services. Each consuming agent retains its separate mission, knowledge, observations, policies, decisions, and authorization boundaries.

## Known starting constraints

- GitHub repository: `stnlabz/language_model`
- Target VPS reported: 2 vCPUs (AMD EPYC 9354P host), 7.7 GiB RAM, ~93 GB disk available; no GPU detected
- Digit and RictusD already run on the VPS and are out of scope for initial model development
- Digit Corpus reported approximately 50 MB, Observations approximately 5 MB; Context is intended as on-demand persistent data
- Subsequent Operator-directed work implemented read-only export of operator-verified SQLite lesson text and native checkpoint training/evaluation. This does not authorize access to Digit's other data or establish production integration, unrestricted provenance validation, or VPS access by developers.

## Ordered candidate milestones — each requires separate Operator authorization

### 1. Prototype specification

Determine the initial language-learning task, model architecture and size, tokenizer approach, source-language implementation, permissible dependencies, resource budget, permitted training corpus, and measurable acceptance criteria. Document explicit failure and rollback boundaries. No training-data access is implied.

**Current evidence:** A native C11 CPU causal transformer with byte vocabulary and local checkpoints has been implemented. This does not establish a complete approved specification or qualification.

**Gate:** Operator approval of the specific specification and implementation scope.

### 2. Original tokenizer and data pipeline

Implement only the approved tokenizer and corpus preparation necessary for the first experiment. Validate data provenance and authorization prior to any import; never silently copy Digit's data.

**Current evidence:** Read-only SQLite exporter with operator-verified lesson filtering, deduplication, explicit record boundaries, and reported isolation/deduplication PASS. The recorded passing tests do not establish completion of every gate check.

**Gate:** Tests for tokenization, reproducibility, invalid inputs, and resource limits; Operator assessment and authorization to advance.

### 3. Minimal trainable language model

Build an original small model and explicit CPU training/inference mechanisms, without pretrained weights or external model inference. Preserve reproducible configuration and training provenance.

**Current evidence:** Native training and checkpoint persistence are implemented; training is reported PASS. Model build (`sqlite-learn`) and model evaluation (`sqlite-evaluate` / `language-test`) are separate, explicitly invoked operations. A saved checkpoint or a completed training run is not a language test.

**Gate:** Demonstrated training and model-generated output, with recorded resource use and limitations; no fluency claim from mere text generation.

### 4. Evaluate language learning

Measure learning against held-out data and check for memorization, repetitive output, invalid generation, and resource exhaustion. Report what was observed, not assumed.

**Current investigation (2026-10-25):** SQLite-backed checkpoint generation remains repetitive and fails language qualification. A separate frozen-checkpoint position/content diagnosis found severe sensitivity to one-byte prompt-position shifts, some sensitivity to content under equal-length controls, and degraded free-running output despite much stronger teacher-forced prediction. Additional analysis of a failing SQLite continuation found an incorrect first predicted byte even before autoregressive feedback. These experiments use different checkpoints and must not be conflated. No single positional-embedding, attention, or training root cause is established. See `docs/POSITION_DIAGNOSIS_20261025.md`.

**Gate:** Operator end-to-end evaluation and explicit acceptance.

### 5. Future engineering-language and policy capability

Only after separate approval, investigate additional authorized code and doctrine material. Policy familiarity must not substitute for deterministic authorization and enforcement mechanisms.

### 6. Future agent integrations

Only after separate approval, design interfaces consistent with Digit/Ezra MVC, existing data models, and module boundaries; consider RictusD independently. Preserve Corpus, Observations, and on-demand Context as agent-side data mechanisms. Do not change any agent or deploy a service as an automatic consequence of earlier milestones.

## Recorded qualification status — Operator update 20261025:1755Z

| Area | Recorded result |
| --- | --- |
| Native test | PASS |
| SQLite and isolation/deduplication | PASS |
| Regression testing | PASS |
| Math | PASS |
| Training | PASS |
| Prose | PASS |
| Generalization | FAIL |
| Language | FAIL |
| Validation | FAIL |
| **Overall** | **NOT QUALIFIED** |

These are Operator-reported results, not tests executed by this roadmap revision. Passing infrastructure and training checks do not establish language competency. Continue investigation and monitor; do not promote the model without demonstrated qualification and Operator acceptance.

## Qualification and release

Follow applicable STN-Labz AI Module Creation Guide, AI LAW, General Orders, version-numbering requirements, and module qualification procedures. A build, test, benchmark, or locally generated sentence is not qualification or release approval. No step begins without Operator authorization. Stop once the authorized outcome is reached.
