import pandas as pd

from complexvar.data.clinvar import classify_clinvar_significance
from complexvar.data.intact import normalize_effect_label, normalize_intact
from complexvar.data.skempi import ddg_to_binary_label, normalize_skempi


def test_clinvar_classification_filters_uncertain_labels():
    assert classify_clinvar_significance("Likely pathogenic") == "positive"
    assert classify_clinvar_significance("Uncertain significance") is None


def test_skempi_binary_label_thresholds():
    assert ddg_to_binary_label(2.0) == 1.0
    assert ddg_to_binary_label(0.2) == 0.0
    assert ddg_to_binary_label(0.9) == 0.0


def test_intact_effect_label_mapping():
    assert normalize_effect_label("disrupting interaction") == "disrupting"
    assert normalize_effect_label("no effect") == "neutral"
    assert normalize_effect_label(pd.NA) is None


def test_skempi_normalization_derives_ddg_from_affinities():
    frame = pd.DataFrame(
        {
            "#Pdb": ["1ABC_A_B"],
            "Mutation(s)_cleaned": ["AA10V"],
            "Affinity_mut_parsed": [1.0e-7],
            "Affinity_wt_parsed": [1.0e-9],
            "Temperature": ["298(assumed)"],
        }
    )
    normalized = normalize_skempi(frame)
    assert normalized.loc[0, "structure_id"] == "1ABC_A_B"
    assert normalized.loc[0, "pdb_id"] == "1ABC"
    assert normalized.loc[0, "binary_label"] == 1.0
    assert normalized.loc[0, "ddg"] > 0


def test_intact_normalization_uses_feature_annotation_fallback():
    frame = pd.DataFrame(
        {
            "#Feature AC": ["EBI-1"],
            "Feature short label": ["P12345:p.Arg10Ala"],
            "Feature annotation": ["MI:0612 (comment): disrupting binding to partner"],
            "Affected protein AC": ["uniprotkb:P12345"],
            "Affected protein symbol": ["GENE1"],
            "Affected protein organism": ["9606 - Homo sapiens"],
            "Interaction AC": ["EBI-100"],
        }
    )
    normalized = normalize_intact(frame)
    assert normalized.loc[0, "protein_accession"] == "P12345"
    assert normalized.loc[0, "effect_label"] == "disrupting"
    assert normalized.loc[0, "binary_label"] == 1.0
