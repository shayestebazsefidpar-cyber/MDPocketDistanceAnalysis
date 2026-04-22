import numpy as np

from mdpocketclustering.features import compute_ligand_pocket_distance


class FakeAtoms:
    def center_of_mass(self):
        return np.array([1.0, 2.0, 3.0])

    def __len__(self):
        return 5


class FakePocket(FakeAtoms):
    def radius_of_gyration(self):
        return 1.5  # fake value


class FakeUniverse:
    def __init__(self):
        self.trajectory = [0, 1, 2]

    def select_atoms(self, sel):
        if "protein" in sel:
            return FakePocket()
        return FakeAtoms()


class FakeSim:
    def __init__(self):
        self.has_mg = True
        self.replicate = 1
        self.binding_energy = -10
        self.mutation = "WT"

    def get_topology(self):
        return "fake.pdb"

    def get_trajectory(self):
        return "fake.dcd"


def test_feature_shape(monkeypatch):

    import MDAnalysis as mda

    monkeypatch.setattr(mda, "Universe", lambda *a, **k: FakeUniverse())

    sim = FakeSim()

    X, meta = compute_ligand_pocket_distance(sim, stride=1)

    assert X.shape[1] == 4  # 4 features
    assert len(meta) == len(X)
