from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Mutation:
    chain: str
    resid: int
    wildtype: str
    mutant: str

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.wildtype}{self.resid}{self.mutant}"


@dataclass(frozen=True)
class Component:
    name: str
    category: str  # protein / ligand / ion / cofactor / membrane / solvent
    count: int = 1


@dataclass
class SystemDefinition:
    system_id: str
    protein_name: str

    mutations: list[Mutation] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)

    @property
    def mutation_label(self) -> str:
        if not self.mutations:
            return "WT"
        return "_".join(m.label for m in self.mutations)

    def has_component(self, name: str) -> bool:
        return any(c.name.lower() == name.lower() for c in self.components)


@dataclass
class SimulationFiles:
    topology: Path
    trajectory: Path

    structure: Optional[Path] = None
    log: Optional[Path] = None
    index: Optional[Path] = None
    energy: Optional[Path] = None


@dataclass
class SimulationRun:
    run_id: str
    system: SystemDefinition

    replicate: int
    engine: str = "gromacs"
    seed: Optional[int] = None

    files: Optional[SimulationFiles] = None

    timestep_fs: Optional[float] = None
    total_time_ns: Optional[float] = None
    temperature_k: Optional[float] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.system.system_id}_rep{self.replicate}"

    def exists(self) -> bool:
        if self.files is None:
            return False
        return self.files.topology.exists() and self.files.trajectory.exists()

    def universe(self):
        import MDAnalysis as mda

        return mda.Universe(
            str(self.files.topology),
            str(self.files.trajectory),
        )
