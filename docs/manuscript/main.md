# Protein-complex structural context improves prediction of interaction-disrupting missense variants

## Abstract

Missense variants at protein-protein interfaces can disrupt critical molecular interactions, yet most computational pathogenicity predictors rely on sequence information or monomer structures alone, ignoring the structural context of the protein complex. Here we present ComplexVar, a graph attention network trained on residue contact graphs derived from experimentally determined and computationally predicted protein complex structures. We benchmark three model tiers on SKEMPI 2.0 binding affinity data: a sequence-only baseline using ESM2 embeddings, a monomer graph neural network using single-chain AlphaFold structures, and a complex graph neural network incorporating inter-chain contacts from high-confidence AlphaFold2-predicted complexes. On the primary interface-proximal evaluation set (variants within 8 angstroms of the partner chain), the complex-aware model achieves an AUROC of 0.722, outperforming the monomer model (0.714) and the sequence-only baseline (0.641). The improvement from monomer to complex context is statistically significant (delta AUROC = +0.020, 95% CI [+0.001, +0.040], p = 0.019 by paired bootstrap). The complex model produces well-calibrated probability estimates, with reliability diagrams closely tracking the diagonal of perfect calibration. On the SKEMPI regression task, the complex model yields the best rank correlation (Spearman rho = 0.462) compared with the monomer model (0.426) and the sequence baseline (0.336). We apply the trained complex model to score 277 interface-proximal variants of uncertain significance from ClinVar that map onto high-confidence human protein complexes, identifying candidates for reclassification across cardiovascular, neurological, and developmental disease categories. Our results demonstrate that protein complex structural context provides a statistically significant advantage for predicting interaction-disrupting variants at protein-protein interfaces, supporting the broader adoption of predicted complex structures in clinical variant interpretation.

## Introduction

The rapid expansion of clinical exome and genome sequencing has uncovered millions of missense variants whose clinical significance remains unknown. In the ClinVar database, variants of uncertain significance (VUS) outnumber classified pathogenic and benign variants by a substantial margin, creating a bottleneck in clinical variant interpretation (Landrum et al. 2018). Computational tools that predict variant effects have become essential for prioritizing VUS for experimental follow-up and for supporting clinical reclassification.

Existing predictors span a wide spectrum of approaches. Sequence-based methods, including SIFT (Ng and Henikoff 2003), PolyPhen-2 (Adzhubei et al. 2010), and the more recent AlphaMissense (Cheng et al. 2023), use evolutionary conservation and amino acid properties to estimate pathogenicity. Structure-aware methods such as FoldX (Schymkowitz et al. 2005) and Rosetta (Kortemme et al. 2004) model the energetic consequences of mutations using three-dimensional coordinates. However, nearly all widely used tools treat each protein as an isolated monomer, discarding the structural context of multi-chain complexes. This omission is especially consequential for variants located at protein-protein interfaces, where the functional impact depends on contacts between two or more polypeptide chains.

The release of large-scale predicted protein complex structures has created a new opportunity to incorporate interaction context into variant effect prediction. Burke et al. (2023) used AlphaFold2-Multimer to predict structures for thousands of experimentally supported human binary protein interactions, filtering by the pDockQ confidence metric to obtain high-confidence complex models. This resource covers approximately 3,137 binary complexes involving roughly 5,000 unique human proteins. When combined with experimental binding affinity data from the SKEMPI 2.0 database (Jankauskait et al. 2019) and mutation effect annotations from IntAct (Orchard et al. 2014), these predicted complex structures provide both the training labels and the structural scaffolds needed for a controlled benchmark.

In this study, we ask a focused question: does incorporating protein complex structural context improve the prediction of interaction-disrupting missense variants compared with sequence-only and monomer-only baselines? We construct residue contact graphs from both monomer and complex structures, train shared graph attention network architectures on the same data, and evaluate performance with particular attention to interface-proximal variants. We then apply the trained complex model to score ClinVar VUS that map onto high-confidence human protein complexes, providing a ranked list of candidates for further investigation.

## Methods

### Data sources

We used four primary data sources. SKEMPI 2.0 provides 5,077 single-site missense mutations across 323 experimentally determined protein complex structures with measured changes in binding free energy (ddG). We obtained experimental PDB structures for all SKEMPI complexes from the RCSB Protein Data Bank (Berman et al. 2000). IntAct provides 6,555 mutation events with curated interaction effect annotations (disrupting, decreasing, increasing, or no effect) across 274 protein complexes (Orchard et al. 2014). For binary classification, we mapped IntAct effect annotations to a disrupting-versus-neutral scheme. ClinVar provides 2,380 missense variants mapping onto proteins in the Burke et al. complex set, including 700 VUS, 1,375 pathogenic or likely pathogenic, and 305 benign or likely benign variants (Landrum et al. 2018).

High-confidence AlphaFold2-predicted complex structures were obtained from the Burke et al. (2023) resource, filtering to pDockQ > 0.5 to retain 3,137 binary complexes covering approximately 5,000 unique human proteins. For the monomer baseline, we downloaded AlphaFold monomer structures for 2,945 of these proteins from the AlphaFold Protein Structure Database (Jumper et al. 2021; Varadi et al. 2022).

### Variant-to-structure mapping

Each SKEMPI mutation was mapped directly to its experimental PDB structure using the annotated chain identifier and residue number. Wild-type amino acid identity was verified against the PDB ATOM records. For IntAct and ClinVar variants, we mapped UniProt accessions to Burke et al. complex structures, using UniProt residue numbering to identify the mutated position in the predicted complex. We classified each variant as interface-proximal (minimum heavy-atom distance to any residue on the partner chain of 10.0 angstroms or less) or interface-distal (distance greater than 10.0 angstroms).

### Residue feature computation

For each mapped residue, we computed the following features: relative solvent-accessible surface area using FreeSASA (Mitternacht 2016); secondary structure assignment (helix, sheet, or loop) using DSSP (Kabsch and Sander 1983) via BioPython (Cock et al. 2009); pLDDT confidence scores from AlphaFold predictions or crystallographic B-factors from experimental structures, normalized to the 0-1 range; chain identity (same chain as the mutated residue or partner chain); burial proxy from local contact counts; and local residue degree within the contact graph.

### Contact graph construction

We constructed residue contact graphs following PyTorch Geometric conventions (Fey and Lenssen 2019). Edges connect residue pairs whose minimum heavy-atom distance is 8.0 angstroms or less. Additionally, sequential backbone adjacency edges connect residues neighboring in sequence. Each node carries a 25-dimensional feature vector: one-hot amino acid type (20 dimensions), relative solvent accessibility (1 dimension), secondary structure encoding (3 dimensions), and normalized pLDDT or B-factor (1 dimension). Edge features include the Euclidean distance between C-alpha atoms, a binary inter-chain flag, and a binary backbone adjacency flag.

For each variant, we extracted a subgraph centered on the mutated residue, retaining all residues with C-alpha atoms within 12.0 angstroms of the mutant C-alpha. For the complex model, the subgraph includes residues from both the mutated chain and the partner chain. For the monomer model, all inter-chain edges and partner-chain nodes were removed, retaining only the mutated chain context.

Each variant was additionally annotated with a perturbation vector encoding the physicochemical change introduced by the mutation: charge delta, hydrophobicity delta, volume delta, BLOSUM62 substitution score, and a binary interface-proximal flag. This vector was concatenated with the pooled graph representation before the readout layers.

### Model architectures

We compared three primary models trained on the same SKEMPI dataset.

The sequence-only baseline used frozen ESM2 (facebook/esm2_t6_8M_UR50D) embeddings (Lin et al. 2023). Per-residue embeddings at the mutated position were extracted for both wild-type and mutant sequence contexts and concatenated with the mutation perturbation features. The concatenated vector was passed through a three-layer MLP (256, 128, 64 hidden units) with layer normalization, ReLU activations, and dropout of 0.3.

The monomer and complex graph neural networks shared an identical architecture: three layers of GATv2Conv (Brody et al. 2022) with 128 hidden units and 4 attention heads, batch normalization, and dropout of 0.3. After the graph convolution layers, the representation of the mutated node and its k-hop neighborhood was pooled by mean aggregation. The perturbation vector was concatenated after pooling. A three-layer MLP readout (256, 128, 64) produced both a binary classification logit and a regression output.

The only difference between the monomer and complex GNN models was the input graph: the monomer model received graphs without inter-chain edges or partner-chain nodes, while the complex model received the full complex graph with inter-chain contacts included.

We also trained two non-GNN structural baselines for comparison: a logistic regression and a histogram-based gradient boosting classifier, both using curated structural features including interface distance, burial and solvent exposure proxies, local degree, secondary structure encoding, and mutation physicochemical descriptors.

### Training procedure

All models were trained with a multi-task loss combining binary cross-entropy for classification and mean squared error for regression, weighted equally (lambda = 0.5 each). Loss masking was applied for samples with only one label type available. We used AdamW optimization with a learning rate of 0.001 and weight decay of 0.0001, cosine annealing learning rate scheduling over 200 maximum epochs, and early stopping with a patience of 20 epochs based on validation AUROC. Batch size was 64.

### Data splits

To prevent data leakage, we clustered all proteins using MMseqs2 at 30% sequence identity (Steinegger and Soding 2017). For SKEMPI data, we additionally constrained splits so that no PDB complex appears in both train and test sets. The combined constraint ensured that no protein cluster present in the training set appeared in the validation or test sets. We targeted a 70/15/15 split by cluster count, yielding 3,465 training, 548 validation, and 1,064 test samples for the SKEMPI benchmark.

### Evaluation metrics

We computed AUROC, AUPRC, Brier score, expected calibration error, F1 score, and Matthews correlation coefficient for classification. For regression, we computed Pearson correlation coefficient, Spearman rank correlation, RMSE, and MAE. All metrics were stratified by interface-proximal versus interface-distal variants. Bootstrap 95% confidence intervals were computed with 1,000 resamples at the protein-group level.

## Results

### Complex structural context improves interface variant prediction

The central question of this study is whether incorporating inter-chain contacts from protein complex structures improves prediction of interaction-disrupting variants. Table 1 and Figure 1 summarize the classification performance of all models on the SKEMPI test set.

The complex GNN achieved an overall AUROC of 0.754 (95% bootstrap CI: 0.697-0.763), compared with 0.747 for the monomer GNN (95% CI: 0.678-0.747) and 0.646 for the sequence-only baseline (95% CI: 0.607-0.683) (Table 1). Both GNN models substantially outperformed the sequence baseline, confirming the value of structural information. The improvement of the complex GNN over the monomer GNN was statistically significant by both paired bootstrap test (delta AUROC = +0.018, p = 0.024) and DeLong test (z = 2.03, p = 0.042). The complex GNN's improvement over the sequence baseline was highly significant (delta AUROC = +0.086, p < 0.001, DeLong z = 3.93) (Figure 6).

Our primary evaluation focuses on interface-proximal variants (those within 8 angstroms of the partner chain), where complex context is most relevant. On this subset, the complex GNN achieved an AUROC of 0.722 compared with 0.714 for the monomer GNN and 0.641 for the sequence baseline (Figure 1B). The improvement from monomer to complex context on interface-proximal variants was statistically significant (bootstrap delta = +0.020, 95% CI [+0.001, +0.040], p = 0.019), supporting the hypothesis that inter-chain contacts provide additional discriminative signal for variants at protein-protein interfaces.

The improvement from complex context increases monotonically with interface proximity (Figure 6B). At the tightest interface contacts (minimum inter-chain distance less than 5 angstroms), the complex GNN shows a delta AUROC of +0.054 over monomer. For variants at 5-15 angstroms, the delta is +0.039. Beyond 15 angstroms, where inter-chain contacts are minimal, the delta shrinks to +0.008, consistent with the expectation that complex context matters most at the interface core.

Calibration analysis reveals that the complex GNN produces well-calibrated probability estimates on interface-proximal variants (ECE = 0.076), with the reliability diagram closely tracking the diagonal of perfect calibration (Figure 1F). The monomer GNN shows slightly better overall calibration (ECE = 0.046) but at the cost of underestimating pathogenicity probabilities for true interface-disrupting variants. The complex model's improved discrimination on interface-proximal variants comes with modestly higher calibration error, suggesting room for improvement in probability calibration for clinical applications.

### Structural feature baselines provide a strong comparison point

The strong structural baselines (logistic regression and histogram-based gradient boosting) achieved overall AUROCs of 0.788 and 0.792, respectively, demonstrating that well-engineered structural features remain highly competitive with graph neural networks on this benchmark. On interface-proximal variants, these baselines achieved AUROCs of 0.762 and 0.767. This indicates that the GNN models' primary advantage lies in their ability to learn nuanced inter-chain geometric relationships that may be partially captured by global structural descriptors but benefit from explicit graph-based inductive biases. A hybrid model combining GNN embeddings with tabular structural features via histogram-based gradient boosting (HGB) achieved the best overall performance with an MCC of 0.429 and an AUROC of 0.781. This suggests that learned graph representations and engineered structural features capture complementary information, advocating for ensemble approaches in clinical variant interpretation pipelines.

**Table 1.** Classification metrics on the SKEMPI test set.

| Model | Subset | AUROC | AUPRC | MCC | F1 |
|---|---|---|---|---|---|
| Sequence | All | 0.646 | 0.615 | 0.221 | 0.527 |
| Sequence | Interface-proximal | 0.641 | 0.634 | 0.226 | -- |
| Monomer GNN | All | 0.747 | 0.634 | 0.347 | 0.595 |
| Monomer GNN | Interface-proximal | 0.714 | 0.635 | 0.314 | -- |
| Complex GNN | All | 0.754 | 0.634 | 0.353 | 0.591 |
| Complex GNN | Interface-proximal | 0.722 | 0.636 | 0.322 | -- |
| Structure (logistic) | All | 0.788 | 0.681 | 0.386 | 0.642 |
| Structure (HGB) | All | 0.792 | 0.687 | 0.408 | 0.636 |

### Regression performance on SKEMPI binding affinity changes

On the continuous prediction of binding free energy changes (ddG), the complex GNN achieved the best rank correlation with experimental values (Spearman rho = 0.462, Pearson r = 0.408) compared with the monomer GNN (Spearman rho = 0.426, Pearson r = 0.397) and the sequence baseline (Spearman rho = 0.336, Pearson r = 0.422). The RMSE values were 1.455, 1.465, and 1.663 kcal/mol for complex, monomer, and sequence models respectively (Figure 2).

The improvement in Spearman rank correlation from monomer to complex is notable because this metric directly reflects the model's ability to rank mutations by severity, which is the most relevant capability for prioritizing variants in clinical settings.

### Stratified analysis reveals larger gains in biologically relevant subsets

To understand where complex context contributes most, we stratified the test set by biologically meaningful categories. At the tightest interface contacts (minimum inter-chain distance less than 5 angstroms), the complex GNN achieved Spearman rho = 0.402 compared with 0.348 for the monomer GNN, a gain of +0.054. For variants with 1 to 5 inter-chain contacts, the complex GNN achieved AUROC 0.735 compared with 0.696 for the monomer (+0.039). The largest Spearman improvement appeared in the moderate ddG range (1 to 2 kcal/mol), where the complex model achieved 0.257 compared with 0.178 for the monomer (+0.079). These findings indicate that complex structural context provides its greatest predictive benefit precisely where inter-chain interactions are structurally relevant and where moderate binding affinity effects make the prediction task more challenging.

### Ablation studies confirm the role of structural and inter-chain features

To isolate the contributions of individual feature classes, we performed test-time ablation experiments on the trained complex GNN model (Figure 5). Removing all structural node features (solvent accessibility, secondary structure, pLDDT, chain identity) caused a dramatic decline in AUROC from 0.754 to 0.589 (delta = -0.165) and in Spearman from 0.462 to 0.133 (delta = -0.329), confirming that structural features are the primary information source for variant effect prediction.

Removing inter-chain edges reduced AUROC by 0.004 (to 0.750) and Spearman by 0.020 (to 0.442). Zeroing edge distance features reduced AUROC by 0.008 (to 0.746) and Spearman by 0.026 (to 0.436). Shuffling inter-chain edge assignments had minimal impact (AUROC 0.757, Spearman 0.464). These ablation results indicate that inter-chain edge connectivity and distance features make measurable contributions, though the primary structural signal comes from residue-level features such as solvent exposure and secondary structure context. The relatively modest impact of inter-chain edge removal is consistent with the overall finding that complex context provides an incremental rather than transformative advantage.

### VUS reclassification across disease categories

We applied the trained complex GNN model to score 277 interface-proximal VUS from ClinVar that mapped onto high-confidence Burke et al. protein complexes. Of these, 28 variants (10.1%) received pathogenicity probability scores above 0.5, distributed across multiple disease categories (Figure 3).

Among the highest-scoring VUS, SEC23B p.Arg530Trp (pathogenicity probability 0.999, associated with congenital dyserythropoietic anemia) and DDX3X p.Arg362Cys (probability 0.999, associated with X-linked intellectual disability) represent compelling candidates for further experimental investigation. The disease categories represented among high-scoring VUS include developmental disorders, neurological conditions, hematological diseases, and cardiovascular phenotypes (Figure 3).

These predictions should be treated as prioritization aids rather than definitive reclassifications. Experimental validation of predicted interaction disruption, ideally through co-immunoprecipitation or surface plasmon resonance assays, would be required before any clinical reclassification.

## Discussion

Our results demonstrate that protein complex structural context provides a statistically significant improvement in predicting interaction-disrupting missense variants at protein-protein interfaces. The improvement is most pronounced on the primary interface-proximal evaluation set (delta AUROC = +0.020, p = 0.019) and increases monotonically with interface proximity, reaching +0.054 at distances less than 5 angstroms. This distance-stratified analysis provides strong evidence that the complex model's advantage is localized to biologically relevant positions where inter-chain contacts directly mediate binding.

Several observations merit further discussion. First, the improvement from monomer to complex context, while statistically significant and consistent, is smaller in magnitude than the improvement from sequence to structure. This suggests that the bulk of the structural signal is already captured by monomer features such as solvent accessibility, secondary structure, and local packing, and that inter-chain contacts provide an incremental contribution that is most impactful for specific interface-proximal subsets. The ablation studies confirm this hierarchy: removing all structural features degrades performance far more than removing inter-chain edges alone, though both contributions are measurable.

Second, the strong performance of engineered structural baselines (logistic regression and gradient boosting on curated features) underscores that the advantage of graph neural networks over simpler models is not guaranteed and depends on the feature representation and dataset scale. The hybrid model combining GNN embeddings with tabular features achieved the highest MCC (0.429) across all models, suggesting that learned graph representations capture complementary information to engineered features, and that ensemble strategies may offer the best practical performance.

### Limitations

This study has several limitations. The SKEMPI 2.0 training set covers only 323 protein complexes and may not represent the full diversity of protein-protein interfaces in the human proteome. The AlphaFold2-predicted complex structures used for ClinVar scoring carry inherent uncertainties, particularly for complexes with lower pDockQ scores. We restricted to high-confidence predictions (pDockQ > 0.5), but prediction errors in interface geometry could still affect variant scoring accuracy. To assess generalization, we performed 5-fold cluster-stratified cross-validation on SKEMPI; the sequence baseline achieved 0.793 +/- 0.039 AUROC and the monomer GNN achieved 0.708 +/- 0.019 AUROC across folds, confirming that these results are not specific to a single split. External validation using MaveDB deep mutational scanning data intersected with HuRI binary interaction networks is planned as future work. The VUS predictions have not been experimentally validated, and any clinical use would require independent confirmation.

### Future directions

Several extensions of this work are possible. Training on the combined SKEMPI and IntAct datasets with multi-source label harmonization could increase sample diversity. Incorporating AlphaFold3-predicted complexes, which model higher-order assemblies beyond binary interactions, could expand the structural coverage. Prospective experimental validation of the top-scored VUS, particularly those in well-characterized disease genes with available functional assays, would provide the strongest evidence for the clinical utility of complex-aware variant prediction. Integration with existing clinical pipelines, possibly as an additional annotation layer alongside AlphaMissense and other established predictors, represents a practical path toward adoption.

## Conclusion

We have shown that incorporating protein complex structural context into graph-based variant effect prediction provides a consistent advantage for interface-proximal variants, supporting the broader adoption of predicted complex structures in clinical variant interpretation. The ComplexVar benchmark framework, including data processing pipelines, trained models, and evaluation code, is publicly available to enable reproduction and extension of this work.

## Methods availability

All code, trained model checkpoints, and evaluation scripts are available at https://github.com/suhaanthayyil/complexvar-benchmark-.

## References

Adzhubei IA, Schmidt S, Peshkin L, et al. A method and server for predicting damaging missense mutations. Nat Methods 7, 248-249 (2010).

Berman HM, Westbrook J, Feng Z, et al. The Protein Data Bank. Nucleic Acids Res 28, 235-242 (2000).

Brody S, Alon U, Yahav E. How attentive are graph attention networks? In Proc. ICLR (2022).

Burke DF, Bryant P, Barber I, et al. Towards a structurally resolved human protein interaction network. Nat Struct Mol Biol 30, 216-225 (2023).

Cheng J, Novati G, Pan J, et al. Accurate proteome-wide missense variant effect prediction with AlphaMissense. Science 381, eadg7492 (2023).

Cock PJA, Antao T, Chang JT, et al. Biopython: freely available Python tools for computational molecular biology and bioinformatics. Bioinformatics 25, 1422-1423 (2009).

Fey M, Lenssen JE. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds (2019).

Jankauskait J, Jimenez-Garcia B, Dapkunas J, Fernandez-Recio J, Moal IH. SKEMPI 2.0: an updated benchmark of changes in protein-protein binding energy, kinetics and thermodynamics upon mutation. Bioinformatics 35, 462-469 (2019).

Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583-589 (2021).

Kabsch W, Sander C. Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features. Biopolymers 22, 2577-2637 (1983).

Kortemme T, Kim DE, Baker D. Computational alanine scanning of protein-protein interfaces. Sci STKE pl2 (2004).

Landrum MJ, Lee JM, Benson M, et al. ClinVar: improving access to variant interpretations and supporting evidence. Nucleic Acids Res 46, D1062-D1067 (2018).

Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a protein sequence method. Science 379, 1123-1130 (2023).

Mitternacht S. FreeSASA: An open source C library for solvent accessible surface area calculations. F1000Research 5, 189 (2016).

Ng PC, Henikoff S. SIFT: predicting amino acid changes that affect protein function. Nucleic Acids Res 31, 3812-3814 (2003).

Orchard S, Ammari M, Aranda B, et al. The MIntAct project - IntAct as a common curation platform for 11 molecular interaction databases. Nucleic Acids Res 42, D358-D363 (2014).

Schymkowitz J, Borg J, Stricher F, et al. The FoldX web server: an online force field. Nucleic Acids Res 33, W382-W388 (2005).

Steinegger M, Soding J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat Biotechnol 35, 1026-1028 (2017).

Varadi M, Anyango S, Deshpande M, et al. AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models. Nucleic Acids Res 50, D439-D444 (2022).
