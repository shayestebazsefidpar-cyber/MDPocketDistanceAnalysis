import numpy as np

from mdpocketclustering.analysistrj.single import SingleTrajectoryRunner
from mdpocketclustering.metrics.rg import ProteinRG


def test_protein_rg(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    values = runner.run_metric(ProteinRG(), stride=20)

    assert values is not None
    assert len(values) > 0
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)
