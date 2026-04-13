rule train_baselines:
    input:
        "results/toy/baseline/predictions.tsv",


rule train_gnn:
    input:
        "results/toy/gnn/README.txt",


rule train_toy_baseline:
    input:
        "data/processed/toy/classification_features.tsv",
        "data/manifests/toy_split_manifest.tsv"
    output:
        "results/toy/baseline/predictions.tsv"
    shell:
        (
            "python -m complexvar.cli train-baseline "
            "--features data/processed/toy/classification_features.tsv "
            "--splits data/manifests/toy_split_manifest.tsv "
            "--target-column label "
            "--group-column protein_group "
            "--output-dir results/toy/baseline"
        )


rule write_gnn_note:
    output:
        "results/toy/gnn/README.txt"
    shell:
        "python -m complexvar.cli write-gnn-note --output {output}"
