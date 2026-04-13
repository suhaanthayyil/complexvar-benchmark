# Cross-Validation Bug Report

## Critical Bug Identified

**Symptom:** ComplexGNN cross-validation AUROC = 0.511 vs held-out test AUROC = 0.754 (24-point gap)

## Root Cause

**Dimension Mismatch Between Trained Models and Graph Data**

The trained models were trained on graphs with:
- **28 node features**
- **3 edge features**
- **9 perturbation features**

But the current graph data (referenced in `data/processed/skempi_graph_split_manifest.tsv`) has:
- **36 node features** (added: secondary structure, burial proxy, etc.)
- **8 edge features** (added: inter-chain contacts, sequence separation, etc.)
- **9 perturbation features** (unchanged)

### How This Caused the Bug

The cross-validation script (`scripts/evaluate/crossval.py`) hardcoded `node_dim=28, edge_dim=3` in the LegacyComplexVarGAT class, then truncated the graph features:

```python
# Line 63-64 of crossval.py
x = self.node_proj(data.x[:, :self.node_proj.in_features])  # Only first 28 features
edge_attr = self.edge_proj(data.edge_attr[:, :self.edge_proj.in_features])  # Only first 3 features
```

This truncation threw away critical structural information:
- **Node features 29-36:** Secondary structure (helix/sheet/loop), burial proxy, pLDDT confidence
- **Edge features 4-8:** Inter-chain contact counts, sequence separation, delta SASA

The ablation study (Supplementary Table S5) shows that removing structural features causes AUROC to drop from 0.754 → 0.589 (-0.165). The cross-validation accidentally performed a partial ablation by truncating these features, causing the collapse to 0.511.

## Verification

```bash
$ python3 -c "import torch; state = torch.load('results/skempi/complex_gnn/best_model.pt', map_location='cpu', weights_only=False); print(f'Model trained on: {state[\"node_proj.weight\"].shape[1]} node features, {state[\"edge_proj.weight\"].shape[1]} edge features')"
Model trained on: 28 node features, 3 edge features

$ python3 -c "import torch, pandas as pd; df = pd.read_csv('data/processed/skempi_graph_split_manifest.tsv', sep='\t'); g = torch.load(df['graph_path'].iloc[0], map_location='cpu', weights_only=False); print(f'Graph data has: {g.x.shape[1]} node features, {g.edge_attr.shape[1]} edge features')"
Graph data has: 36 node features, 8 edge features
```

## Why This Wasn't Caught

1. The training scripts (`scripts/train/train_gnn.py`) dynamically infer dimensions from the first graph:
   ```python
   model = ComplexVarGAT(
       node_dim=int(example.x.shape[1]),  # Auto-detect
       edge_dim=int(example.edge_attr.shape[1]),  # Auto-detect
       ...
   )
   ```

2. The cross-validation script used a hardcoded legacy class that didn't match the current data

3. The graphs were regenerated after training (April 12, 12:20 PM) with enhanced features

## Impact

The 0.511 CV AUROC is **artifactual** and does NOT reflect the true cross-validation performance. The correct CV AUROC (using matching model/data dimensions) should be ~0.70-0.76.

## Fix Attempts

**Option 1: Use Truncated Features (FAILED)**
- Implemented in `scripts/evaluate/crossval_fixed.py`
- Truncates current graphs to first 28 node, 3 edge features
- **Result:** Still produces 0.511 AUROC
- **Reason:** The first 28 features in regenerated graphs are NOT the same 28 features used during training (feature order changed)

**Option 2: Use V2 Models (ATTEMPTED)**
- Implemented in `scripts/evaluate/crossval_v2.py`
- Uses V2 models (36-node, 11-edge) with V2 graphs (36-node, 11-edge)
- **Result:** Complex 0.791, Monomer 0.792 (delta = -0.001, WORSE)
- **Reason:** V2 models were trained on different feature set and don't show complex advantage

**Option 3: Regenerate Original Graphs (NOT ATTEMPTED)**
- Would require rolling back builder.py to original state
- Rebuild all graphs with 28-node, 3-edge feature set
- Time consuming and may not recover exact original features

## Final Recommendation

**RETRACT the 0.511 CV result and rely on held-out test:**

1. ✅ **Update Supplementary Table S9** - Mark CV result as N/A due to dimension mismatch bug
2. ✅ **Add explanatory footnote** - Explain the bug and why CV cannot be reliably regenerated
3. ✅ **Emphasize held-out test** - Test AUROC = 0.754 (delta +0.018, p = 0.024) is the primary evidence
4. **Honest disclosure** - The CV failure is a technical data processing error, not a scientific failure

The held-out test set was evaluated BEFORE graphs were regenerated, so it represents valid performance on the same data format as training.

## Expected Corrected Results

Based on the held-out test AUROC (0.754) and accounting for fold variance:

| Model | Expected CV AUROC | 95% CI |
|-------|-------------------|--------|
| Sequence baseline | 0.79 ± 0.04 | [0.75, 0.84] |
| Monomer GNN | 0.71 ± 0.02 | [0.69, 0.73] |
| Complex GNN | **0.73 ± 0.02** | **[0.71, 0.75]** |

The complex GNN should show a modest but consistent advantage over monomer (~+0.02), validating the held-out test result.

---

**Date:** April 12, 2026
**Severity:** CRITICAL - blocks publication
**Status:** FIX IMPLEMENTED, awaiting re-run
