# STN-Labz Language Model — Agent Instructions

## Authority and scope

The STN-Labz human Operator holds final development authority. Follow applicable STN-Labz AI LAW, General Orders, Developers Codex, AI Module Creation Guide (20261016.0), and version-numbering doctrine (20260906.0). Read this file, README.md, and docs/ROADMAP.md before making changes. When instructions conflict or scope is unclear, stop the affected work and request Operator direction.

Approval is limited to the named task. Repository access, a roadmap, a passing test, or completion of a step is not authorization for the next step. Do not change architecture, interfaces, persistence, deployment, agent behavior, or other files without explicit authorization.

## Project objective

Develop one original STN-Labz-owned language model intended for reuse by Digit, Ezra, and potentially RictusD and future agents. No external pretrained models, weights, distillation, remote inference APIs, or third-party language-model services are authorized.

The first research objective is a small, CPU-compatible model for learning and generating human language. Fluent general-purpose conversation is not a first-version acceptance claim.

## Environment and boundaries

The reported target VPS has 2 vCPUs, approximately 7.7 GiB RAM, 93 GB free disk, and no GPU detected. Digit and RictusD operate there. Do not assume VPS or runtime access from repository access. Do not modify or disrupt existing agents, their Core, their models, their databases, or deployments without separate explicit authorization.

Digit's reported Corpus (~50 MB), Observations (~5 MB), and Context (persistent data for on-demand use) are not implicitly approved as training data. Examine provenance, format, permissions, and intended use only when separately authorized. Retrieval on demand is distinct from model training.

## Engineering controls

- Keep solutions small, deterministic where applicable, inspectable, and documented.
- Respect the adopted MVC responsibilities in any later agent integration: Router routes; controllers coordinate interpretation and reasoning; models access data; views generate responses.
- Do not simulate conversational intelligence with hardcoded responses.
- Keep policy enforcement and agent operational authority outside reliance on probabilistic model output alone.
- Follow applicable source annotation, documentation, version numbering, and qualification rules.
- No unauthorized dependencies, background services, persistence, cloud reliance, or speculative enhancements.
- Preserve existing files and behavior; use minimal reviewable changes and disclose every material change.
- Report observed results separately from inferences. Never claim tests, qualification, runtime activation, or release without evidence.

## Mandatory delivery gate

For each authorized step, report changed files, relevant revision identity, checks actually performed, checks not performed, and limitations. When operator testing or acceptance is outstanding, stop and await explicit operator acceptance and separately authorized next work. A roadmap records possible sequence, not standing permission.
