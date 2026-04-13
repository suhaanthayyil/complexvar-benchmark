"""Feature engineering utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from complexvar.constants import (
    AA_TO_INDEX,
    AMINO_ACIDS,
    BLOSUM62_MATRIX,
    CHARGE,
    HYDROPHOBICITY,
    POLARITY,
    VOLUME,
)


def amino_acid_one_hot(residue: str) -> list[int]:
    vector = [0] * len(AMINO_ACIDS)
    if residue in AA_TO_INDEX:
        vector[AA_TO_INDEX[residue]] = 1
    return vector


@dataclass(frozen=True)
class MutationDescriptor:
    wildtype: str
    mutant: str

    @property
    def delta_hydrophobicity(self) -> float:
        return HYDROPHOBICITY.get(self.mutant, 0.0) - HYDROPHOBICITY.get(
            self.wildtype, 0.0
        )

    @property
    def delta_volume(self) -> float:
        return VOLUME.get(self.mutant, 0.0) - VOLUME.get(self.wildtype, 0.0)

    @property
    def delta_charge(self) -> float:
        return CHARGE.get(self.mutant, 0.0) - CHARGE.get(self.wildtype, 0.0)

    @property
    def delta_polarity(self) -> float:
        return POLARITY.get(self.mutant, 0.0) - POLARITY.get(self.wildtype, 0.0)

    @property
    def blosum62_score(self) -> float:
        return float(BLOSUM62_MATRIX.get(self.wildtype, {}).get(self.mutant, -4))

    @property
    def changed_to_gly(self) -> int:
        return int(self.mutant == "G")

    @property
    def changed_to_pro(self) -> int:
        return int(self.mutant == "P")

    @property
    def changed_to_cys(self) -> int:
        return int(self.mutant == "C")

    @property
    def unchanged(self) -> int:
        return int(self.wildtype == self.mutant)

    def as_dict(self) -> dict[str, float]:
        return {
            "delta_hydrophobicity": self.delta_hydrophobicity,
            "delta_volume": self.delta_volume,
            "delta_charge": self.delta_charge,
            "delta_polarity": self.delta_polarity,
            "blosum62_score": self.blosum62_score,
            "changed_to_gly": self.changed_to_gly,
            "changed_to_pro": self.changed_to_pro,
            "changed_to_cys": self.changed_to_cys,
            "mutation_unchanged": self.unchanged,
        }


def mutation_descriptor(wildtype: str, mutant: str) -> dict[str, float]:
    return MutationDescriptor(wildtype=wildtype, mutant=mutant).as_dict()


def interface_burial_proxy(
    local_degree: float,
    inter_chain_contacts: float,
    solvent_proxy: float,
    interface_distance: float | None = None,
) -> float:
    distance_bonus = 0.0
    if interface_distance is not None and np.isfinite(interface_distance):
        distance_bonus = max(0.0, 10.0 - float(interface_distance))
    return float(local_degree + inter_chain_contacts + distance_bonus - solvent_proxy)


def solvent_exposure_proxy(local_degree: float, max_degree: float = 12.0) -> float:
    return float(max(0.0, max_degree - float(local_degree)) / max_degree)


def zscore(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    std = arr.std(ddof=0)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std
