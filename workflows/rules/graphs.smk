rule graphs:
    input:
        "data/processed/toy/graph_manifest.tsv",


rule build_toy_graphs:
    input:
        "data/processed/toy/classification_features.tsv"
    output:
        "data/processed/toy/graph_manifest.tsv"
    shell:
        "python -m complexvar.cli build-toy-graphs --input-dir data/processed/toy --output {output}"
