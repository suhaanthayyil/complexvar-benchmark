import pandas as pd

from complexvar.utils.splits import SplitFractions, leakage_summary, make_group_splits


def test_make_group_splits_keeps_groups_together():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "family_group": ["f1"] * 3 + ["f2"] * 3 + ["f3"] * 3 + ["f4"] * 3,
        }
    )
    out = make_group_splits(
        frame,
        group_column="family_group",
        fractions=SplitFractions(train=0.5, val=0.25, test=0.25),
        seed=17,
    )
    grouped = out.groupby("family_group")["split"].nunique()
    assert grouped.max() == 1


def test_leakage_summary_counts_samples_and_groups():
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "protein_group": ["p1", "p2", "p3"],
            "family_group": ["f1", "f1", "f2"],
        }
    )
    summary = leakage_summary(
        frame, split_column="split", fields=["protein_group", "family_group"]
    )
    assert summary["train"]["samples"] == 2
    assert summary["test"]["family_group"] == 1
