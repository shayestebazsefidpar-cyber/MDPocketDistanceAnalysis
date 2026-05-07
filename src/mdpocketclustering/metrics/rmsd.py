import numpy as np


class ProteinRMSD:
    name = "protein_rmsd"

    def compute(self, traj, system, stride=20):
        u = traj.u

        ref = u.select_atoms("protein and backbone")
        ref_pos = ref.positions.copy()

        rmsd = []

        for ts in u.trajectory[::stride]:
            mob = u.select_atoms("protein and backbone")

            diff = mob.positions - ref_pos
            rmsd.append(np.sqrt((diff**2).mean()))

        return np.array(rmsd)


class LigandRMSD:
    name = "ligand_rmsd"

    def compute(self, traj, system, stride=20):
        u = traj.u

        ligand = u.select_atoms("resname LIG")
        ref_pos = ligand.positions.copy()

        rmsd = []

        for ts in u.trajectory[::stride]:
            mob = u.select_atoms("resname LIG")

            diff = mob.positions - ref_pos
            rmsd.append(np.sqrt((diff**2).mean()))

        return np.array(rmsd)
