import numpy as np


class ProteinRMSD:
    name = "protein_rmsd"

    def compute(self, traj, system, stride=20):
        u = traj.u

        ref = u.select_atoms("protein and backbone")
        mob = u.select_atoms("protein and backbone")

        rmsd = []

        for ts in u.trajectory[::stride]:
            diff = mob.positions - ref.positions
            rmsd.append(np.sqrt((diff**2).mean()))

        return np.array(rmsd)


class LigandRMSD:
    name = "ligand_rmsd"

    def compute(self, traj, system, stride=1):
        u = traj.u

        ligand = u.select_atoms("resname LIG")
        ref = ligand.positions.copy()

        rmsd = []

        for ts in u.trajectory[::stride]:
            diff = ligand.positions - ref
            rmsd.append(np.sqrt((diff**2).mean()))

        return np.array(rmsd)
