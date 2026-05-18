import numpy as np

from mdpocketclustering.analysistrj.single import SingleTrajectoryRunner
from mdpocketclustering.metrics.rmsd import LigandRMSD, ProteinRMSD


def test_protein_rmsd(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    values = runner.run_metric(ProteinRMSD(), stride=20)

    # basic sanity checks
    assert values is not None
    assert len(values) > 0
    assert np.all(values >= 0)


def test_ligand_rmsd(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    values = runner.run_metric(LigandRMSD(), stride=20)

    # ligand may sometimes be missing in some systems → allow skip-safe check
    assert values is not None
    assert len(values) > 0
    assert np.all(values >= 0)
