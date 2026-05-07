from mdpocketclustering.analysistrj.single import SingleTrajectoryRunner
from mdpocketclustering.metrics.rmsd import LigandRMSD, ProteinRMSD


def test_protein_rmsd(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    df = runner.run_metric(ProteinRMSD(), stride=20)

    assert len(df) > 0
    assert "protein_rmsd" in df.columns


def test_ligand_rmsd(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    df = runner.run_metric(LigandRMSD(), stride=20)

    assert len(df) > 0
    assert "ligand_rmsd" in df.columns
