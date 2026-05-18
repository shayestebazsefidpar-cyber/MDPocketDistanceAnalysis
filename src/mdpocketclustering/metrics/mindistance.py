import numpy as np
from MDAnalysis.analysis.distances import distance_array


class ProteinLigandMinDistance:
    name = "protein_ligand_mindist"

    def __init__(self, ligand_resname="AP1"):
        self.ligand_resname = ligand_resname

    def compute(self, traj, system, stride=20):
        u = traj.u

        protein = u.select_atoms("protein")
        ligand = u.select_atoms(f"resname {self.ligand_resname}")

        if len(ligand) == 0:
            raise ValueError(f"Ligand {self.ligand_resname} not found in system")

        values = []

        for ts in u.trajectory[::stride]:
            d = distance_array(protein.positions, ligand.positions)
            values.append(d.min())

        return np.array(values)
