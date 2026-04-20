from mdpocketclustering.systems import (
    Component,
    Mutation,
    SimulationFiles,
    SimulationRun,
    SystemDefinition,
)


def test_mutation_label():
    mut = Mutation(chain="A", resid=145, wildtype="K", mutant="A")
    assert mut.label == "A:K145A"


def test_system_definition_wildtype():
    system = SystemDefinition(system_id="EGFR_WT", protein_name="EGFR")
    assert system.mutation_label == "WT"


def test_system_definition_multiple_mutations():
    system = SystemDefinition(
        system_id="EGFR_MUT",
        protein_name="EGFR",
        mutations=[
            Mutation("A", 145, "K", "A"),
            Mutation("A", 200, "L", "R"),
        ],
    )

    assert system.mutation_label == "A:K145A_A:L200R"


def test_has_component():
    system = SystemDefinition(
        system_id="SYS1",
        protein_name="Kinase",
        components=[
            Component("ATP", "ligand"),
            Component("MG", "ion", count=2),
        ],
    )

    assert system.has_component("ATP") is True
    assert system.has_component("atp") is True
    assert system.has_component("ADP") is False


def test_simulation_label():
    system = SystemDefinition(system_id="EGFR_WT", protein_name="EGFR")

    run = SimulationRun(run_id="run001", system=system, replicate=3)

    assert run.label() == "EGFR_WT_rep3"


def test_simulation_exists_true(tmp_path):
    top = tmp_path / "topol.tpr"
    traj = tmp_path / "traj.xtc"

    top.touch()
    traj.touch()

    system = SystemDefinition(system_id="SYS1", protein_name="Protein")

    run = SimulationRun(
        run_id="run001",
        system=system,
        replicate=1,
        files=SimulationFiles(topology=top, trajectory=traj),
    )

    assert run.exists() is True


def test_simulation_exists_false(tmp_path):
    system = SystemDefinition(system_id="SYS1", protein_name="Protein")

    run = SimulationRun(
        run_id="run001",
        system=system,
        replicate=1,
        files=SimulationFiles(
            topology=tmp_path / "missing.tpr", trajectory=tmp_path / "missing.xtc"
        ),
    )

    assert run.exists() is False
