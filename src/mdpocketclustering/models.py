from dataclasses import dataclass


@dataclass
class ResidueEnergy:
    residue: str
    index: int
    energy: float
    std: float
