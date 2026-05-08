import numpy as np
from MDAnalysis.analysis.base import AnalysisBase


class RadiusOfGyration:
    name = "radius_of_gyration"

    def compute(self, traj, system, stride=20):
        u = traj.u

        protein = u.select_atoms("protein")

        rg_values = []

        for ts in u.trajectory[::stride]:
            rg = protein.radius_of_gyration()
            rg_values.append(rg)

        return np.array(rg_values)
