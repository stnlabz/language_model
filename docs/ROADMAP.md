# STN-Labz Language Model — Development Roadmap

**Status:** Planning; no model implementation, training, qualification, or deployment authorized by this roadmap.

## Mission

Develop one STN-Labz-native language model to provide reusable human-language understanding and generation for Digit, Ezra, and potentially RictusD and future agents, without external pretrained models or remote model services. Each consuming agent retains its separate mission, knowledge, observations, policies, decisions, and authorization boundaries.

## Known starting constraints

- GitHub repository: `stnlabz/language_model`
- Target VPS reported: 2 vCPUs (AMD EPYC 9354P host), 7.7 GiB RAM, ~93 GB disk available; no GPU detected
- Digit and RictusD already run on the VPS and are out of scope for initial model development
- Digit Corpus reported approximately 50 MB, Observations approximately 5 MB; Context is intended as on-demand persistent data
- No training-data contents, provenance, runtime integration contracts, or VPS access have been inspected or validated

## Ordered candidate milestones — each requires separate Operator authorization

### 1. Prototype specification

Determine the initial language-learning task, model architecture and size, tokenizer approach, source-language implementation, permissible dependencies, resource budget, permitted training corpus, and measurable acceptance criteria. Document explicit failure and rollback boundaries. No training-data access is implied.

**Gate:** Operator approval of the specific specification and implementation scope.

### 2. Original tokenizer and data pipeline

Implement only the approved tokenizer and corpus preparation necessary for the first experiment. Validate data provenance and authorization prior to any import; never silently copy Digit's data.

**Gate:** Tests for tokenization, reproducibility, invalid inputs, and resource limits; Operator assessment and authorization to advance.

### 3. Minimal trainable language model

Build an original small model and explicit CPU training/inference mechanisms, without pretrained weights or external model inference. Preserve reproducible configuration and training provenance.

**Gate:** Demonstrated training and model-generated output, with recorded resource use and limitations; no fluency claim from mere text generation.

### 4. Evaluate language learning

Measure learning against held-out data and check for memorization, repetitive output, invalid generation, and resource exhaustion. Report what was observed, not assumed.

**Gate:** Operator end-to-end evaluation and explicit acceptance.

### 5. Future engineering-language and policy capability

Only after separate approval, investigate additional authorized code and doctrine material. Policy familiarity must not substitute for deterministic authorization and enforcement mechanisms.

### 6. Future agent integrations

Only after separate approval, design interfaces consistent with Digit/Ezra MVC, existing data models, and module boundaries; consider RictusD independently. Preserve Corpus, Observations, and on-demand Context as agent-side data mechanisms. Do not change any agent or deploy a service as an automatic consequence of earlier milestones.

## Qualification and release

Follow applicable STN-Labz AI Module Creation Guide, AI LAW, General Orders, version-numbering requirements, and module qualification procedures. A build, test, benchmark, or locally generated sentence is not qualification or release approval. No step begins without Operator authorization. Stop once the authorized outcome is reached.
