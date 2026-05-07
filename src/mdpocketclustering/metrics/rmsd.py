from MDAnalysis.analysis.rms import RMSD


class ProteinRMSD:
    name = "protein_rmsd"

    def compute(self, traj, system, stride=20):
        u = traj.u

        rmsd_calc = RMSD(u, u, select="protein and backbone", ref_frame=0, step=stride)
        rmsd_calc.run()

        return rmsd_calc.results.rmsd[:, 2]


class LigandRMSD:
    name = "ligand_rmsd"

    def compute(self, traj, system, stride=20):
        u = traj.u

        rmsd_calc = RMSD(u, u, select="resname AP1", ref_frame=0, step=stride)
        rmsd_calc.run()

        return rmsd_calc.results.rmsd[:, 2]
