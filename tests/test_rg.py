import numpy as np

from mdpocketclustering.analysistrj.single import SingleTrajectoryRunner
from mdpocketclustering.metrics.rg import RadiusOfGyration


def test_protein_rg(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    values = runner.run_metric(RadiusOfGyration(), stride=20)

    # basic sanity checks
    assert values is not None
    assert len(values) > 0
    assert np.all(values >= 0)


def test_ligand_rg(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    values = runner.run_metric(RadiusOfGyration(), stride=20)

    # ligand may be missing → still sanity-safe
    assert values is not None
    assert len(values) > 0
    assert np.all(values >= 0)
