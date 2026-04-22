import numpy as np

from mdpocketclustering.pipeline import run_verbose


# -------------------------
# Fake simulation object
# -------------------------
class FakeSim:
    def __init__(self, rep):
        self.replicate = rep
        self.has_atp = True
        self.has_mg = True
        self.mutation = "wt"
        self.binding_energy = -10

    def get_topology(self):
        return "fake.pdb"

    def get_trajectory(self):
        return "fake.dcd"


# -------------------------
# fake MDAnalysis Universe
# -------------------------
def fake_universe(*args, **kwargs):

    class AtomGroup:
        def __init__(self):
            self.positions = np.array([[1.0, 2.0, 3.0]])

        def center_of_mass(self):
            return np.array([1.0, 1.0, 1.0])

        def radius_of_gyration(self):
            return 1.0

        def __len__(self):
            return 10

    class Residue:
        def __init__(self):
            self.resid = 1
            self.resname = "ALA"
            self.atoms = AtomGroup()

    class Pocket(AtomGroup):
        def __init__(self):
            super().__init__()
            self.residues = [Residue()]

        def center_of_mass(self):
            return np.array([0.5, 0.5, 0.5])

        def __len__(self):
            return 5

    class Universe:
        def __init__(self):
            self.trajectory = [0, 1]

        def select_atoms(self, sel):
            return AtomGroup()

    return Universe()


# -------------------------
# TEST
# -------------------------
def test_pipeline_runs(monkeypatch):

    import MDAnalysis as mda

    monkeypatch.setattr(mda, "Universe", fake_universe)

    sims = [FakeSim(1), FakeSim(2), FakeSim(3)]

    result = run_verbose(sims, stride=1)

    assert result is not None

    df, summary, occ, residue_df = result

    assert len(df) > 0
    assert "cluster" in df.columns
