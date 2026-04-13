rule all_data_targets:
    input:
        "data/processed/toy/classification_features.tsv",
        "data/manifests/toy_split_manifest.tsv",


rule download:
    input:
        "data/raw/download_manifest.json"


rule normalize:
    input:
        "data/processed/toy/classification_features.tsv",
        "data/processed/toy/regression_features.tsv",


rule labels:
    input:
        "data/manifests/toy_label_manifest.tsv",
        "data/manifests/toy_variant_manifest.tsv",


rule splits:
    input:
        "data/manifests/toy_split_manifest.tsv",


rule make_toy_dataset:
    output:
        "data/processed/toy/classification_features.tsv",
        "data/processed/toy/regression_features.tsv",
        "data/processed/toy/toy_samples.tsv",
    shell:
        "python -m complexvar.cli make-toy-dataset --output-dir data/processed/toy"


rule make_download_manifest:
    output:
        "data/raw/download_manifest.json"
    shell:
        "python -m complexvar.cli write-download-manifest --output {output}"


rule write_toy_manifests:
    input:
        "data/processed/toy/classification_features.tsv"
    output:
        "data/manifests/toy_variant_manifest.tsv",
        "data/manifests/toy_label_manifest.tsv",
        "data/manifests/toy_split_manifest.tsv"
    shell:
        "python -m complexvar.cli write-toy-manifests --input-dir {input} --output-dir data/manifests"
