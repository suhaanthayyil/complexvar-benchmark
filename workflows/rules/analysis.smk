rule evaluate:
    input:
        "results/toy/baseline/metrics.json",


rule figures:
    input:
        "results/toy/figures/main_comparison.png",


rule manuscript:
    input:
        "manuscript/main_text/manuscript.md",


rule evaluate_toy_baseline:
    input:
        "results/toy/baseline/predictions.tsv"
    output:
        "results/toy/baseline/metrics.json"
    shell:
        "python -m complexvar.cli evaluate-classification --predictions {input} --output {output}"


rule make_toy_figures:
    input:
        "results/toy/baseline/predictions.tsv"
    output:
        "results/toy/figures/main_comparison.png"
    shell:
        "python -m complexvar.cli make-summary-figure --predictions {input} --output {output}"
