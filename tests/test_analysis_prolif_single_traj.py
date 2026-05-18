import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from mdpocketclustering.analysis_prolif_single_traj import run_prolif_for_single_traj


# -------------------------
# Fake objects
# -------------------------

class FakeMutation:
    def __init__(self):
        self.chain = "A"
        self.wildtype = "L"
        self.resid = 123
        self.mutant = "A"


class FakeRun:
    def __init__(self):
        self.run_id = "test_run_001"

        self.files = MagicMock()
        self.files.topology = "fake_topology.pdb"
        self.files.trajectory = "fake_traj.xtc"

        self.system = MagicMock()
        self.system.mutations = [FakeMutation()]


# -------------------------
# Test
# -------------------------

@patch("mdpocketclustering.analysis_prolif_single_traj.mda.Universe")
@patch("mdpocketclustering.analysis_prolif_single_traj.Fingerprint")
def test_run_prolif_for_single_traj(mock_fp_class, mock_universe):

    # --- mock Universe ---
    mock_u = MagicMock()
    mock_universe.return_value = mock_u

    # fake selections
    mock_protein = MagicMock()
    mock_ligand = MagicMock()
    mock_ligand.__len__.return_value = 5  # ligand exists

    mock_u.select_atoms.side_effect = [mock_protein, mock_ligand]
    mock_u.trajectory = list(range(10))  # fake frames

    # --- mock Fingerprint ---
    mock_fp = MagicMock()
    mock_fp_class.return_value = mock_fp

    fake_df = pd.DataFrame({
        ("LIG", "ALA123", "HBDonor"): [True, False, True]
    })

    mock_fp.to_dataframe.return_value = fake_df

    # --- run function ---
    fake_run = FakeRun()
    df = run_prolif_for_single_traj(fake_run)

    # -------------------------
    # Assertions
    # -------------------------

    assert isinstance(df, pd.DataFrame)
    assert "run_id" in df.columns
    assert "mutation" in df.columns
    assert df["run_id"].iloc[0] == "test_run_001"
    assert df["mutation"].iloc[0] == "A:L123A"

    # ensure ProLIF was called
    mock_fp.run.assert_called_once()
    mock_fp.to_dataframe.assert_called_once()
