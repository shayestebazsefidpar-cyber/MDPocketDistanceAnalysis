# test_analysis_prolif.py

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mdpocketclustering.analysis_prolif import retrieve_prolif_frame_residue


class DummyAtoms:
    def __init__(self, n_atoms):
        self.n_atoms = n_atoms

    def __len__(self):
        return self.n_atoms


class DummyTrajectory:
    def __getitem__(self, item):
        return item


class DummyUniverse:
    def __init__(self, ligand_atoms=5):
        self.trajectory = DummyTrajectory()
        self.ligand_atoms = ligand_atoms

    def select_atoms(self, query):
        if query == "protein":
            return DummyAtoms(100)
        return DummyAtoms(self.ligand_atoms)


def test_returns_empty_dataframe_when_no_ligand():
    u = DummyUniverse(ligand_atoms=0)

    df = retrieve_prolif_frame_residue(
        u=u,
        run_id="r1",
        mutation="WT",
        ligand_sel="resname ATP",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


@patch("mdpocketclustering.analysis_prolif.Fingerprint")
def test_fingerprint_called_with_expected_interactions(mock_fp_cls):
    u = DummyUniverse()

    mock_fp = MagicMock()
    mock_fp.to_dataframe.return_value = pd.DataFrame({"x": [1]})
    mock_fp_cls.return_value = mock_fp

    retrieve_prolif_frame_residue(
        u=u,
        run_id="r1",
        mutation="WT",
        ligand_sel="resname ATP",
        stride=10,
    )

    args, kwargs = mock_fp_cls.call_args
    interactions = args[0]

    assert "HBDonor" in interactions
    assert "Hydrophobic" in interactions
    assert "MetalDonor" in interactions


@patch("mdpocketclustering.analysis_prolif.Fingerprint")
def test_run_called_with_correct_arguments(mock_fp_cls):
    u = DummyUniverse()

    mock_fp = MagicMock()
    mock_fp.to_dataframe.return_value = pd.DataFrame({"x": [1]})
    mock_fp_cls.return_value = mock_fp

    retrieve_prolif_frame_residue(
        u=u,
        run_id="r1",
        mutation="WT",
        ligand_sel="resname ATP",
        stride=20,
    )

    mock_fp.run.assert_called_once()

    args, kwargs = mock_fp.run.call_args

    assert args[0] == u.trajectory[::20]


@patch("mdpocketclustering.analysis_prolif.Fingerprint")
def test_returns_dataframe(mock_fp_cls):
    u = DummyUniverse()

    expected = pd.DataFrame(
        {
            "frame": [0, 20],
            "interaction": ["HBDonor", "Hydrophobic"],
        }
    )

    mock_fp = MagicMock()
    mock_fp.to_dataframe.return_value = expected
    mock_fp_cls.return_value = mock_fp

    df = retrieve_prolif_frame_residue(
        u=u,
        run_id="r1",
        mutation="WT",
        ligand_sel="resname ATP",
        stride=20,
    )

    pd.testing.assert_frame_equal(df, expected)


@patch("mdpocketclustering.analysis_prolif.Fingerprint")
def test_stride_is_used(mock_fp_cls):
    u = DummyUniverse()

    mock_fp = MagicMock()
    mock_fp.to_dataframe.return_value = pd.DataFrame()
    mock_fp_cls.return_value = mock_fp

    retrieve_prolif_frame_residue(
        u=u,
        run_id="r1",
        mutation="WT",
        ligand_sel="resname ATP",
        stride=50,
    )

    args, kwargs = mock_fp.run.call_args
    assert args[0] == u.trajectory[::50]
