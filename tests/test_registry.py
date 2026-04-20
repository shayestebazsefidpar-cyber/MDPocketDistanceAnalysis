import pandas as pd

from mdpocketclustering.registry import SimulationRegistry
from mdpocketclustering.systems import (
    Component,
    Mutation,
    SimulationFiles,
    SimulationRun,
    SystemDefinition,
)


def make_system(name="SYS", mutation_label=None, components=None):
    mutations = []

    if mutation_label:
        mutations = [Mutation("A", 145, "K", "A")]

    return SystemDefinition(
        system_id=name,
        protein_name="EGFR",
        mutations=mutations,
        components=components or [],
    )


def make_run(system, replicate=1, energy=None):
    run = SimulationRun(
        run_id=f"{system.system_id}_{replicate}",
        system=system,
        replicate=replicate,
        files=SimulationFiles(
            topology=__file__,
            trajectory=__file__,
        ),
    )
    run.binding_energy = energy
    return run


def test_registry_add_and_len():
    reg = SimulationRegistry()

    sys = make_system()
    reg.add(make_run(sys))

    assert len(reg) == 1


def test_registry_extend():
    reg = SimulationRegistry()

    sys = make_system()

    runs = [make_run(sys, i) for i in range(3)]
    reg.extend(runs)

    assert len(reg) == 3


def test_filter_by_replicate():
    reg = SimulationRegistry()

    sys = make_system()

    reg.extend(
        [
            make_run(sys, replicate=1),
            make_run(sys, replicate=2),
            make_run(sys, replicate=2),
        ]
    )

    filtered = reg.filter(replicate=2)

    assert len(filtered) == 2
    assert all(r.replicate == 2 for r in filtered)


def test_filter_by_component():
    reg = SimulationRegistry()

    sys1 = make_system(components=[Component("ATP", "ligand")])
    sys2 = make_system(components=[Component("RNA", "ligand")])

    reg.extend(
        [
            make_run(sys1),
            make_run(sys2),
        ]
    )

    filtered = reg.filter(component="ATP")

    assert len(filtered) == 1
    assert filtered.runs[0].system.has_component("ATP")


def test_to_dataframe_basic():
    reg = SimulationRegistry()

    sys = make_system()

    reg.add(make_run(sys, replicate=1, energy=-10.0))
    reg.add(make_run(sys, replicate=2, energy=-20.0))

    df = reg.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "mutation" in df.columns
    assert "replicate" in df.columns
    assert "binding_energy" in df.columns


def test_to_dataframe_components_expansion():
    reg = SimulationRegistry()

    sys = make_system(
        components=[
            Component("ATP", "ligand"),
            Component("MG", "ion", count=2),
        ]
    )

    reg.add(make_run(sys, energy=-5.0))

    df = reg.to_dataframe()

    assert "has_ATP" in df.columns
    assert "has_MG" in df.columns
    assert bool(df["has_ATP"].iloc[0]) is True


def test_summary_statistics():
    reg = SimulationRegistry()

    sys = make_system()

    reg.add(make_run(sys, replicate=1, energy=-10.0))
    reg.add(make_run(sys, replicate=1, energy=-20.0))
    reg.add(make_run(sys, replicate=2, energy=-30.0))

    summary = reg.summary()

    assert "mean" in summary.columns
    assert "std" in summary.columns
    assert "count" in summary.columns
    assert len(summary) > 0
