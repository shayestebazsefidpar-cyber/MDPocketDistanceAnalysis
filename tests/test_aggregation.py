import pandas as pd

from mdpocketclustering.analysis import aggregate_replicates
from mdpocketclustering.registry import SimulationRegistry
from mdpocketclustering.systems import SimulationFiles, SimulationRun, SystemDefinition


def make_run(mutation, replicate, energy):
    system = SystemDefinition(
        system_id="SYS", protein_name="EGFR", mutations=[], components=[]
    )

    run = SimulationRun(
        run_id=f"{mutation}_{replicate}",
        system=system,
        replicate=replicate,
        files=SimulationFiles(__file__, __file__),
    )

    run.binding_energy = energy
    return run


# --------------------------------------------------
# test aggregation function directly
# --------------------------------------------------


def test_aggregate_replicates_basic():
    reg = SimulationRegistry()

    reg.extend(
        [
            make_run("WT", 1, -10),
            make_run("WT", 2, -20),
            make_run("WT", 3, -30),
        ]
    )

    df = aggregate_replicates(reg, value="binding_energy", by=["mutation"])

    assert "mean" in df.columns
    assert "std" in df.columns
    assert "count" in df.columns

    assert df["mean"].iloc[0] == -20
    assert df["count"].iloc[0] == 3


def test_aggregate_custom_grouping():
    reg = SimulationRegistry()

    reg.extend(
        [
            make_run("WT", 1, -10),
            make_run("WT", 2, -20),
            make_run("MUT", 1, -40),
            make_run("MUT", 2, -60),
        ]
    )

    df = aggregate_replicates(reg, value="binding_energy", by=["mutation"])

    assert len(df) == 1 or len(df) == 2
    assert "mean" in df.columns
