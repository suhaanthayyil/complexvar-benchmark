# Model Specification

## Sequence-only baseline

- frozen `ESM2 t6 8M` embeddings
- mutation descriptors concatenated to sequence embedding
- small MLP classifier

## Structural baselines

### ddg_proxy_logistic

- distance to interface
- inter-chain contact count
- burial proxy
- solvent exposure proxy
- local degree
- mutation physicochemical deltas

### struct_xgboost

- all `ddg_proxy_logistic` features
- secondary-structure indicators
- chain role
- local interface neighborhood counts

## Graph models

### monomer_gnn

- residue graph on monomer context
- 3-layer `GINE`

### complex_gnn

- matched `GINE` architecture
- inter-chain edges enabled
- same node and edge feature family as monomer where possible
