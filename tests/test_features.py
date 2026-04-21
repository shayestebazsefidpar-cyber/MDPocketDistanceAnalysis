import numpy as np
from mdpocketclustering.features import compute_ligand_pocket_distance


class FakeSim:
    def __init__(self):
        self.has_mg = True
        self.replicate = 1
        self.binding_energy = -10

    def get_topology(self):
        return "fake.pdb"

    def get_trajectory(self):
        return "fake.dcd"


def test_feature_shape(monkeypatch):

    def fake_universe(*args, **kwargs):
        class U:
            def __init__(self):
                self.trajectory = [0, 1, 2]

            def select_atoms(self, sel):
                class A:
                    def center_of_mass(self):
                        return np.array([1.0, 2.0, 3.0])
                return A()

        return U()

    import MDAnalysis as mda
    monkeypatch.setattr(mda, "Universe", fake_universe)

    sim = FakeSim()

    X, meta = compute_ligand_pocket_distance(sim, stride=1)

    assert X.shape[1] == 1
    assert len(meta) == len(X)
