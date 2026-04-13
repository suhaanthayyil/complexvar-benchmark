configfile: "configs/workflow.yaml"

include: "workflows/rules/data.smk"
include: "workflows/rules/graphs.smk"
include: "workflows/rules/model.smk"
include: "workflows/rules/analysis.smk"
