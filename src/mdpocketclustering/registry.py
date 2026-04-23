from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from mdpocketclustering.systems import SimulationRun


@dataclass
class SimulationRegistry:
    """
    Central container for all simulation runs.
    Acts as a lightweight in-memory database for MD analysis.
    """

    runs: List["SimulationRun"] = field(default_factory=list)

    # --------------------------------------------------
    # BASIC OPERATIONS
    # --------------------------------------------------

    def add(self, run: "SimulationRun") -> None:
        self.runs.append(run)

    def extend(self, runs: Iterable["SimulationRun"]) -> None:
        self.runs.extend(list(runs))

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def filter(
        self,
        *,
        protein: Optional[str] = None,
        replicate: Optional[int] = None,
        mutation: Optional[str] = None,
        component: Optional[str] = None,
        min_time_ns: Optional[float] = None,
    ) -> "SimulationRegistry":

        filtered = self.runs

        if protein is not None:
            filtered = [r for r in filtered if r.system.protein_name == protein]

        if replicate is not None:
            filtered = [r for r in filtered if r.replicate == replicate]

        if mutation is not None:
            filtered = [r for r in filtered if r.system.mutation_label == mutation]

        if component is not None:
            filtered = [r for r in filtered if r.system.has_component(component)]

        if min_time_ns is not None:
            filtered = [r for r in filtered if (r.total_time_ns or 0) >= min_time_ns]

        return SimulationRegistry(filtered)

    def to_dataframe(self, include_components: bool = True) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for r in self.runs:
            row = {
                "run_id": r.run_id,
                "replicate": r.replicate,
                "protein": r.system.protein_name,
                "mutation": r.system.mutation_label,
            }

            row.update(
                {
                    "timestep_fs": r.timestep_fs,
                    "total_time_ns": r.total_time_ns,
                    "temperature_k": r.temperature_k,
                }
            )

            energy = getattr(r, "binding_energy", None) or getattr(
                r, "metadata", {}
            ).get("binding_energy")

            if energy is not None:
                row["binding_energy"] = energy

            if include_components:
                for c in r.system.components:
                    row[f"has_{c.name}"] = True
                    row[f"count_{c.name}"] = c.count

            rows.append(row)

        df = pd.DataFrame(rows)

        if include_components:
            component_cols = [c for c in df.columns if c.startswith("has_")]
            df[component_cols] = (
                df[component_cols].fillna(False).infer_objects(copy=False)
            )

        return df

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()

        if "binding_energy" not in df:
            raise ValueError("No binding_energy found in registry runs")

        return df.groupby(["mutation", "replicate"]).agg(
            mean=("binding_energy", "mean"),
            std=("binding_energy", "std"),
            count=("binding_energy", "count"),
        )
