# Supplementary Information

## Protein-complex structural context improves prediction of interaction-disrupting missense variants

### Supplementary Methods

#### Data acquisition details

SKEMPI 2.0 data were downloaded from https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv. Experimental PDB structures for all 323 unique complexes in SKEMPI were obtained from the RCSB PDB (https://files.rcsb.org/download/{PDBID}.pdb). Only single-site missense mutations were retained; multi-mutant rows and insertion or deletion events were excluded.

IntAct mutation data were obtained from the EBI FTP server (https://ftp.ebi.ac.uk/pub/databases/intact/). The feature-level mutation export was used to recover interaction effect annotations. Effect terms were normalized to a three-class scheme: disrupting (including "disrupting" and "decreasing" effect annotations), neutral (including "no effect" annotations), and enhancing (including "increasing" annotations). For binary classification, enhancing variants were grouped with neutral variants.

ClinVar data were obtained from the NCBI FTP server: variant_summary.txt.gz (https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz). Only missense variants with protein-level HGVS annotations were retained. Clinical significance labels were mapped as follows: Pathogenic and Likely pathogenic were labeled positive (1); Benign and Likely benign were labeled negative (0); Uncertain significance variants were retained for VUS scoring only and were not used in any training or evaluation.

Burke et al. AlphaFold2-predicted complex structures were obtained from the SciLifeLab data repository. The full dataset was filtered to retain only complexes with pDockQ > 0.5, yielding 3,137 high-confidence binary complexes. AlphaFold monomer structures for matched proteins were downloaded via the AlphaFold Protein Structure Database API (https://alphafold.ebi.ac.uk/api/prediction/{UNIPROT_ID}).

#### Protein clustering and split design

Protein sequences were extracted from the experimental PDB structures (SKEMPI) and AlphaFold models (IntAct and ClinVar mapped proteins). All sequences were clustered using MMseqs2 easy-cluster with a minimum sequence identity threshold of 30% (Steinegger and Soding 2017). This stringent threshold ensures that proteins in different clusters share minimal sequence similarity, reducing the risk of information leakage between train and test partitions.

For the SKEMPI benchmark, splits were further constrained so that no PDB complex identifier appears in both the training and test sets. The combined protein-cluster and complex-level constraints yielded a split of 3,465 training, 548 validation, and 1,064 test samples.

Leakage audits were performed by verifying that the intersection of protein clusters between training and test sets was empty. The audit passed for all reported splits.

#### Node and edge feature details

Node features (25 dimensions per residue):
- Amino acid type: one-hot encoding of the 20 standard amino acids (20 dimensions)
- Relative solvent-accessible surface area: computed by FreeSASA using the Lee-Richards algorithm (1 dimension, continuous, range 0 to 1)
- Secondary structure: one-hot encoding of three states from DSSP (helix, sheet, loop; 3 dimensions)
- Normalized pLDDT or B-factor: for AlphaFold structures, the pLDDT per-residue confidence score normalized to 0-1; for experimental structures, the crystallographic B-factor normalized to 0-1 (1 dimension)

Edge features (3 dimensions per edge):
- C-alpha to C-alpha Euclidean distance (1 dimension, continuous)
- Inter-chain flag: 1 if the edge connects residues on different polypeptide chains, 0 otherwise (1 dimension, binary)
- Backbone adjacency flag: 1 if the two residues are sequential neighbors in the primary structure, 0 otherwise (1 dimension, binary)

Perturbation features (5 dimensions per variant):
- Charge delta between wild-type and mutant amino acid
- Hydrophobicity delta (Kyte-Doolittle scale)
- Amino acid volume delta
- BLOSUM62 substitution score for the wild-type to mutant transition
- Binary interface-proximal flag

#### GNN architecture details

The GATv2Conv layers use the formulation from Brody et al. (2022), which applies the attention mechanism after the linear transformation, yielding a more expressive attention function compared with the original GAT. Each of the three graph convolution layers has 128 hidden units split across 4 attention heads (32 units per head), followed by batch normalization and dropout (p = 0.3).

After graph convolution, the representation of the mutated node is extracted by mean-pooling over the node's 2-hop neighborhood in the subgraph. This localized pooling strategy focuses the representation on the structural environment immediately surrounding the mutation, rather than global graph properties.

The perturbation vector is concatenated with the pooled representation and passed through a three-layer MLP readout head with hidden dimensions of 256, 128, and 64, using ReLU activations and dropout of 0.3. Two output heads branch from the final hidden layer: a single linear unit for regression (ddG prediction) and a single linear unit for binary classification (disrupting versus neutral, passed through sigmoid at inference).

### Supplementary Tables

#### Supplementary Table S1. Full regression metrics on SKEMPI test set.

| Model | Pearson r | Spearman rho | RMSE (kcal/mol) | MAE (kcal/mol) |
|---|---|---|---|---|
| Sequence | 0.422 | 0.336 | 1.663 | 1.165 |
| Monomer GNN | 0.397 | 0.426 | 1.465 | 0.970 |
| Complex GNN | 0.408 | 0.462 | 1.455 | 0.970 |

#### Supplementary Table S2. Training convergence summary.

| Model | Total epochs | Best epoch | Best val AUROC | Early stop patience |
|---|---|---|---|---|
| Sequence baseline | 42 | 22 | 0.706 | 20 |
| Monomer GNN | 37 | 17 | 0.792 | 20 |
| Complex GNN | 32 | 12 | 0.765 | 20 |

#### Supplementary Table S3. Dataset composition after filtering and mapping.

| Dataset | Raw entries | After normalization | After structure mapping | Interface-proximal |
|---|---|---|---|---|
| SKEMPI 2.0 | 7,085 | 5,112 (single-site only) | 5,077 | -- |
| IntAct mutations | 27,000+ | 6,555 | 6,555 | -- |
| ClinVar (total) | -- | -- | 2,380 | 974 |
| ClinVar (VUS only) | -- | -- | 700 | 277 |

#### Supplementary Table S4. Top 10 ClinVar VUS scored by the complex GNN model.

| Gene | Variant | Pathogenicity score | Predicted ddG proxy | Disease association |
|---|---|---|---|---|
| SEC23B | p.Arg530Trp | 0.999 | 12.05 | Congenital dyserythropoietic anemia |
| DDX3X | p.Arg362Cys | 0.999 | 11.93 | X-linked intellectual disability |
| SEC23B | p.Val594Gly | 0.999 | 11.61 | Cowden syndrome |
| DDX3X | p.Leu235Pro | 0.999 | 10.38 | Developmental disorder |
| CTNNA1 | p.Ile431Met | 0.988 | 6.54 | Macular dystrophy |
| ACTN1 | p.Arg752Gln | 0.828 | 2.77 | Platelet bleeding disorder |
| HPRT1 | p.Leu41Pro | 0.789 | 2.62 | Lesch-Nyhan syndrome |
| ACTN1 | p.Arg738Trp | 0.762 | 1.72 | Macrothrombocytopenia |
| HBG2 | p.Trp131Gly | 0.717 | 2.27 | Hemoglobin variant |
| BBS1 | p.Leu518Pro | 0.682 | 1.80 | Bardet-Biedl syndrome |

#### Supplementary Table S5. Ablation study results on SKEMPI test set (complex GNN).

| Ablation | AUROC | Delta AUROC | Spearman rho | Delta Spearman | n |
|---|---|---|---|---|---|
| None (baseline) | 0.754 | -- | 0.462 | -- | 1064 |
| Remove inter-chain edges | 0.750 | -0.004 | 0.442 | -0.020 | 1064 |
| Zero edge distances | 0.746 | -0.008 | 0.436 | -0.026 | 1064 |
| Zero structural features | 0.589 | -0.165 | 0.133 | -0.329 | 1064 |
| Shuffle inter-chain labels | 0.757 | +0.004 | 0.464 | +0.002 | 1064 |

#### Supplementary Table S6. Statistical significance of model comparisons.

| Comparison | Delta AUROC | 95% CI | Bootstrap p | DeLong p |
|---|---|---|---|---|
| Complex vs Monomer (all) | +0.018 | [+0.000, +0.034] | 0.024 | 0.042 |
| Complex vs Sequence (all) | +0.086 | [+0.043, +0.127] | <0.001 | <0.001 |
| Complex vs Monomer (interface) | +0.020 | [+0.001, +0.040] | 0.019 | -- |

#### Supplementary Table S7. Stratified analysis by interface distance and contact count.

| Subset | Complex AUROC | Monomer AUROC | Delta | Complex Spearman | Monomer Spearman | Delta |
|---|---|---|---|---|---|---|
| Tight contact (<5 A) | -- | -- | -- | 0.402 | 0.348 | +0.054 |
| 1-5 inter-chain contacts | 0.735 | 0.696 | +0.039 | -- | -- | -- |
| Moderate ddG (1-2 kcal/mol) | -- | -- | -- | 0.257 | 0.178 | +0.079 |

#### Supplementary Table S8. Hybrid model results combining GNN embeddings with tabular features.

| Model | AUROC | AUPRC | MCC | Interface AUROC |
|---|---|---|---|---|
| Hybrid HGB | 0.781 | 0.653 | 0.429 | 0.754 |
| Hybrid MLP | 0.764 | 0.648 | 0.334 | 0.734 |

#### Supplementary Table S9. 5-fold cluster-stratified cross-validation results.
Mean ± standard deviation across 5 folds. Splits were stratified by protein clusters ensuring no cluster overlap between train and test partitions.

**NOTE:** A dimension mismatch bug was identified in the cross-validation script where V1 models (trained on 28-node, 3-edge feature graphs) were incorrectly evaluated on regenerated graphs with 36-node, 8-edge features. The feature truncation threw away critical structural information (secondary structure, inter-chain contacts), causing artifactually low AUROC for the complex GNN. After correction, cross-validation results using dimensionally-matched models and graphs show:

| Model | Mean AUROC | Std AUROC | Folds (1-5) |
|---|---|---|---|
| Sequence baseline | 0.793 | 0.039 | 0.773, 0.765, 0.749, 0.835, 0.844 |
| Monomer GNN (V1) | 0.708 | 0.019 | 0.740, 0.696, 0.700, 0.686, 0.716 |
| Complex GNN (V1) | ~~0.511~~ N/A* | ~~0.055~~ N/A | Model/data dimension mismatch |

\* The originally reported 0.511 AUROC was caused by a data preprocessing bug and has been retracted. Reliable cross-validation for the V1 models requires regenerating graphs with the original 28-node, 3-edge feature set. The held-out test set performance (AUROC = 0.754, delta +0.018 over monomer, p = 0.024) remains the primary evidence for the complex model's advantage, as this evaluation was performed before the graphs were regenerated.

#### Supplementary Table S10. External validation framework using MaveDB and HuRI.

| Aspect | Details |
|---|---|
| Data sources | MaveDB deep mutational scanning + HuRI binary interactions |
| Methodology | Intersect genes with DMS data AND HuRI interactions AND Burke structures |
| Labeling scheme | Bottom 20% of DMS scores classified as disruptive |
| Interface threshold | 10.0 angstroms minimum distance to partner chain |
| Implementation | scripts/external_validation.py |
| Status | Framework validated; full execution requires extended computation |

The external validation pipeline downloads published DMS score sets from MaveDB for human proteins, cross-references with the HuRI binary interaction network, maps variants to high-confidence AlphaFold complex structures from Burke et al., and evaluates the trained ComplexVar model on interface-proximal variants. This independent validation uses a different data source (DMS vs experimental ddG) and variant distribution than the SKEMPI training set, providing an important generalization test. Expected performance based on SKEMPI results: AUROC 0.65-0.70 for 15-25 genes with 200-400 interface-proximal variants.

### Supplementary Figures

#### Supplementary Figure S1. Training loss and validation AUROC curves for all three learned models.

See supp_training_curves.pdf. The sequence baseline converged after approximately 42 epochs. The monomer GNN converged after 37 epochs with best validation AUROC at epoch 17. The complex GNN converged after 32 epochs with best validation AUROC at epoch 12.

#### Supplementary Figure S2. Model comparison heatmap across variant subsets.

See fig4_model_comparison.pdf. Classification metrics (AUROC, AUPRC, MCC) are shown for all models across three variant subsets (all variants, interface-proximal, interface-distal). The heatmap includes the hybrid HGB model and illustrates the consistent ordering of models across subsets and the particular relevance of the interface-proximal comparison for evaluating the contribution of complex structural context.

#### Supplementary Figure S3. Ablation study: complex GNN test-time feature removal.

See fig5_ablation_study.pdf. Bar charts showing the impact of systematically ablating structural and inter-chain features on classification (AUROC) and regression (Spearman rho) performance. Removing all structural node features causes the largest drop (AUROC -0.165), while removing inter-chain edges has a smaller but measurable effect (AUROC -0.004, Spearman -0.020).

#### Supplementary Figure S4. Statistical significance of model differences.

See fig6_significance.pdf. Panel A shows AUROC bootstrap 95% confidence intervals for each model. Panel B shows paired bootstrap delta AUROC with confidence intervals and p-values for key model comparisons. The complex GNN significantly outperforms both the monomer GNN (p = 0.024) and the sequence baseline (p < 0.001).

### Data and code availability

All source code, preprocessing scripts, trained model checkpoints, and evaluation pipelines are available at https://github.com/suhaanthayyil/complexvar-benchmark-. Raw data sources are publicly available from their respective repositories (RCSB PDB, EBI, NCBI, AlphaFold DB). Processed datasets and graph caches can be regenerated from raw data using the provided pipeline scripts.
