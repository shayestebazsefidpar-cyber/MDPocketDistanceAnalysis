import numpy as np
from MDAnalysis.analysis.base import AnalysisBase


class ProteinRG:
    name = "protein_rg"

    def compute(self, traj, system, stride=20):
        u = traj.u
        protein = u.select_atoms("protein")

        rg_values = []

        for ts in u.trajectory[::stride]:
            rg_values.append(protein.radius_of_gyration())

        return np.array(rg_values)
