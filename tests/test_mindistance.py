import numpy as np

from mdpocketclustering.analysistrj.single import SingleTrajectoryRunner
from mdpocketclustering.metrics.mindistance import ProteinLigandMinDistance


def test_protein_ligand_mindist(registry):
    run = registry.runs[0]

    runner = SingleTrajectoryRunner(run)
    values = runner.run_metric(
        ProteinLigandMinDistance(ligand_resname="AP1"),
        stride=20
    )

    assert values is not None
    assert isinstance(values, np.ndarray)
    assert len(values) > 0

    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)
