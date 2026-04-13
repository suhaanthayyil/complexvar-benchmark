# Risk Register

## High risk

- IMEx/IntAct effect-label export schema may vary across releases.
- Optional GNN dependencies may require environment-specific installation.
- Strong subset baselines such as FoldX cannot be part of the redistributable
  core pipeline.

## Medium risk

- High-confidence human complex coverage may limit overlap with directly labeled
  interaction-effect datasets.
- Sequence and graph backends may require model downloads not bundled in the
  repo.

## Mitigations

- Keep direct interaction labels as the primary benchmark.
- Fail loudly on missing effect labels and document the chosen workaround.
- Use toy data and deterministic manifests for smoke testing.
- Restrict the main claim to interface variants.
