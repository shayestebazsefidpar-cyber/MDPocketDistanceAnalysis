import numpy as np
from MDAnalysis.analysis.rms import RMSF


class ProteinRMSF:
    name = "protein_rmsf"

    def compute(self, traj, system, stride=20):
        u = traj.u

        ag = u.select_atoms("protein and backbone")

        # align trajectory first (important for RMSF correctness)
        from MDAnalysis.analysis import align
        align.AlignTraj(u, u, select="protein and backbone").run()

        rmsf_calc = RMSF(ag)
        rmsf_calc.run()

        values = rmsf_calc.results.rmsf

        # RMSF is per atom → we reduce to per-frame-like series (optional smoothing view)
        # here we just return per-atom RMSF
        return values
