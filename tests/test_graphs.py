from pathlib import Path

from complexvar.graphs.builder import build_residue_graph_tables


def test_build_residue_graph_tables_from_minimal_pdb(tmp_path: Path):
    pdb_path = tmp_path / "toy.pdb"
    pdb_lines = [
        (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
            "  1.00 80.00           C"
        ),
        (
            "ATOM      2  CA  LEU A   2       2.000   0.000   0.000"
            "  1.00 81.00           C"
        ),
        (
            "ATOM      3  CA  GLY B   1       0.000   0.000   6.000"
            "  1.00 78.00           C"
        ),
        (
            "ATOM      4  CA  SER B   2       2.000   0.000   6.000"
            "  1.00 79.00           C"
        ),
        "TER",
        "END",
    ]
    pdb_path.write_text(
        "\n".join(pdb_lines) + "\n",
        encoding="utf-8",
    )
    nodes, edges = build_residue_graph_tables(pdb_path)
    assert len(nodes) == 4
    assert not edges.empty
    assert nodes["is_interface"].max() == 1
