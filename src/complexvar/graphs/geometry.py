"""Geometry helpers for residue graphs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import dist


@dataclass(frozen=True)
class ResiduePoint:
    residue_id: str
    chain_id: str
    x: float
    y: float
    z: float

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def pairwise_contacts(
    residues: list[ResiduePoint], cutoff: float
) -> list[tuple[str, str, float, bool]]:
    """Return residue contacts with inter-chain flags."""

    contacts: list[tuple[str, str, float, bool]] = []
    for left, right in combinations(residues, 2):
        separation = dist(left.xyz, right.xyz)
        if separation <= cutoff:
            contacts.append(
                (
                    left.residue_id,
                    right.residue_id,
                    float(separation),
                    left.chain_id != right.chain_id,
                )
            )
    return contacts


def min_inter_chain_distance(
    residue: ResiduePoint, neighbors: list[ResiduePoint]
) -> float | None:
    distances = [
        dist(residue.xyz, other.xyz)
        for other in neighbors
        if residue.chain_id != other.chain_id
    ]
    if not distances:
        return None
    return float(min(distances))
