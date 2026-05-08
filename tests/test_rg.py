import numpy as np

from mdpocketclustering.analysistrj.single import SingleTrajectoryRunner
from mdpocketclustering.metrics.rg import ProteinRG


def test_protein_rg(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    values = runner.run_metric(ProteinRG(), stride=20)

    # basic sanity
    assert values is not None
    assert isinstance(values, np.ndarray)
    assert values.size > 0

    # physical sanity
    assert np.all(np.isfinite(values))
    assert np.min(values) >= 0.0

    # extra robustness (important in MD pipelines)
    assert np.std(values) > 0  # ensures trajectory is not constant artifact
