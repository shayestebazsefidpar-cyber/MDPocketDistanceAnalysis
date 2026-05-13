import pandas as pd
import pytest

from mdpocketclustering.analysis_prolif_registry import run_prolif_for_registry

# -----------------------------
# FAKE OBJECTS
# -----------------------------


class FakeFiles:
    def __init__(self):
        self.topology = "fake.pdb"
        self.trajectory = "fake.dcd"


class FakeMutation:
    def __init__(self):
        self.chain = "A"
        self.wildtype = "WT"
        self.resid = 100
        self.mutant = "M"


class FakeSystem:
    def __init__(self):
        self.mutations = [FakeMutation()]


class FakeRun:
    def __init__(self):
        self.run_id = "TEST_RUN"
        self.files = FakeFiles()
        self.system = FakeSystem()


class FakeRegistry:
    def __init__(self):
        self.runs = [FakeRun()]


class FakeUniverse:
    """
    Minimal MDAnalysis replacement
    """

    def __init__(self, topology, trajectory):
        self.atoms = self
        self.resnames = ["ATP", "MG1"]
        self.trajectory = [0, 1, 2]

    def select_atoms(self, sel):
        return [1, 2, 3]  # non-empty fake selection


class FakeFingerprint:
    """
    Minimal ProLIF replacement
    """

    def run(self, traj, ligand, protein):
        pass

    def to_dataframe(self):
        return pd.DataFrame(
            {
                "interaction": ["HBDonor", "Hydrophobic"],
                "value": [1, 0],
            }
        )


# -----------------------------
# TEST
# -----------------------------


def test_run_prolif_for_registry(monkeypatch):

    import mdpocketclustering.analysis_prolif_registry as mod

    # patch external dependencies
    monkeypatch.setattr(mod.mda, "Universe", FakeUniverse)
    monkeypatch.setattr(mod, "Fingerprint", FakeFingerprint)

    registry = FakeRegistry()

    df = run_prolif_for_registry(registry, stride=1, save_csv=False)

    # -----------------------------
    # ASSERTIONS
    # -----------------------------

    assert isinstance(df, pd.DataFrame)

    # required columns from your pipeline
    assert "run_id" in df.columns
    assert "mutation" in df.columns
    assert "interaction" in df.columns
    assert "value" in df.columns

    # should return data
    assert len(df) > 0

    # check registry mapping
    assert df["run_id"].iloc[0] == "TEST_RUN"
