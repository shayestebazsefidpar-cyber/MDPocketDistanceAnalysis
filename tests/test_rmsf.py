import numpy as np

from mdpocketclustering.analysistrj.single import SingleTrajectoryRunner
from mdpocketclustering.metrics.rmsf import ProteinRMSF


def test_protein_rmsf(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    values = runner.run_metric(ProteinRMSF(), stride=20)

    assert values is not None
    assert isinstance(values, np.ndarray)
    assert len(values) > 0

    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)
