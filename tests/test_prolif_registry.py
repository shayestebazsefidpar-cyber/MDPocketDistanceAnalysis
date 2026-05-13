import pandas as pd
import pytest

from mdpocketclustering.analysis_prolif_registry import run_prolif_for_registry


# -------------------------
# MOCK OBJECTS
# -------------------------
class MockMutation:
    def __init__(self):
        self.wildtype = "A"
        self.resid = 525


class MockSystem:
    def __init__(self):
        self.mutations = [MockMutation()]


class MockFiles:
    def __init__(self):
        self.topology = "fake_topology.pdb"
        self.trajectory = "fake_trajectory.dcd"


class MockRun:
    def __init__(self, run_id):
        self.run_id = run_id
        self.files = MockFiles()
        self.system = MockSystem()


class MockRegistry:
    def __init__(self):
        self.runs = [MockRun("test_run_1")]


# -------------------------
# TEST
# -------------------------
def test_run_prolif_for_registry_smoke(tmp_path, monkeypatch):

    registry = MockRegistry()

    # -------------------------
    # FIXED FAKE MDANALYSIS UNIVERSE
    # -------------------------
    class FakeAtomGroup:
        def __init__(self):
            # IMPORTANT FIX: MDAnalysis expects .atoms attribute
            self.atoms = self

        def __len__(self):
            return 1

    class FakeTrajectory:
        pass

    class FakeUniverse:
        def __init__(self, top, traj):
            self.top = top
            self.trajectory = FakeTrajectory()

        def select_atoms(self, sel):
            return FakeAtomGroup()

    monkeypatch.setattr("MDAnalysis.Universe", FakeUniverse)

    # -------------------------
    # MOCK PROLIF
    # -------------------------
    class FakeFingerprint:
        def __init__(self):
            pass

        def run(self, traj, lig=None, prot=None):
            pass

        def to_dataframe(self):
            index = [0, 1]

            columns = pd.MultiIndex.from_tuples(
                [
                    ("ARG", 525, "HBDonor"),
                    ("LYS", 199, "PiStacking"),
                ]
            )

            data = [
                [True, False],
                [False, True],
            ]

            return pd.DataFrame(data, index=index, columns=columns)

    monkeypatch.setattr("prolif.Fingerprint", FakeFingerprint)

    # -------------------------
    # RUN FUNCTION
    # -------------------------
    out_csv = tmp_path / "out.csv"

    result = run_prolif_for_registry(
        registry,
        ligand_sel="resname ATP",
        out_csv=str(out_csv),
    )

    # -------------------------
    # ASSERTIONS
    # -------------------------
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

    expected_cols = {
        "Frame",
        "ligand",
        "interaction",
        "value",
        "residue",
        "resid",
        "run_id",
        "mutation",
    }

    assert set(result.columns) == expected_cols

    assert out_csv.exists()
