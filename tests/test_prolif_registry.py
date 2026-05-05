from unittest.mock import MagicMock, patch

import pandas as pd

from mdpocketclustering.analysis_prolif_registry import run_prolif_for_registry


class FakeMutation:
    def __init__(self):
        self.chain = "A"
        self.wildtype = "L"
        self.resid = 123
        self.mutant = "A"


class FakeRun:
    def __init__(self, run_id):
        self.run_id = run_id

        self.files = MagicMock()
        self.files.topology = "fake_topology.pdb"
        self.files.trajectory = "fake_traj.xtc"

        self.system = MagicMock()
        self.system.mutations = [FakeMutation()]


class FakeRegistry:
    def __init__(self):
        self.runs = [FakeRun("run1"), FakeRun("run2")]


@patch("mdpocketclustering.analysis_prolif_registry.mda.Universe")
@patch("mdpocketclustering.analysis_prolif_registry.Fingerprint")
def test_run_prolif_for_registry(mock_fp_class, mock_universe):

    # -------------------------
    # mock Universe
    # -------------------------
    mock_u = MagicMock()
    mock_universe.return_value = mock_u

    mock_u.select_atoms.return_value.__len__.return_value = 5
    mock_u.trajectory = list(range(10))

    # -------------------------
    # mock Fingerprint
    # -------------------------
    mock_fp = MagicMock()
    mock_fp_class.return_value = mock_fp

    fake_df = pd.DataFrame(
        {
            ("AP1", "ALA123", "HBDonor"): [True, False],
            ("AP1", "ALA123", "HBAcceptor"): [False, True],
        }
    )

    mock_fp.to_dataframe.return_value = fake_df

    registry = FakeRegistry()

    df = run_prolif_for_registry(
        registry, ligand_sel="resname AP1 or resname MG1", stride=20, n_jobs=1
    )

    # -------------------------
    # assertions
    # -------------------------
    assert isinstance(df, pd.DataFrame)
    assert "Frame" in df.columns
    assert "ligand" in df.columns
    assert "protein" in df.columns
    assert "interaction" in df.columns
    assert "value" in df.columns
    assert "run_id" in df.columns
    assert "mutation" in df.columns

    # check concatenation from 2 runs
    assert df["run_id"].nunique() == 2
