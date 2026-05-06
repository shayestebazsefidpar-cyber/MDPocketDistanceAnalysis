import MDAnalysis as mda
import pandas as pd
from prolif import Fingerprint


def run_prolif_for_single_traj(
    run,
    ligand_sel="resname AP1",
    stride=20,
):
    """
    Run ProLIF + convert to long format in one step
    """

    # --- load universe ---
    u = mda.Universe(run.files.topology, run.files.trajectory)

    # --- mutation handling ---
    mutations = run.system.mutations
    mutation = (
        f"{mutations[0].chain}:{mutations[0].wildtype}"
        f"{mutations[0].resid}{mutations[0].mutant}"
        if mutations
        else "WT"
    )

    # --- selections ---
    protein = u.select_atoms("protein")
    ligand = u.select_atoms(ligand_sel)

    if len(ligand) == 0:
        print(f"❌ No ligand found in run {run.run_id}")
        return pd.DataFrame()

    # --- ProLIF ---
    fp = Fingerprint(
        [
            "HBDonor",
            "HBAcceptor",
            "Hydrophobic",
            "PiStacking",
            "PiCation",
            "CationPi",
            "Anionic",
            "Cationic",
            "EdgeToFace",
            "FaceToFace",
            "MetalAcceptor",
            "MetalDonor",
        ]
    )

    fp.run(u.trajectory[::stride], ligand, protein)
    df = fp.to_dataframe()

    # metadata
    meta = {"run_id": run.run_id, "mutation": mutation}

    # remove metadata if exists
    core = df.drop(columns=list(meta.keys()), errors="ignore")

    # ensure MultiIndex columns
    core.columns = pd.MultiIndex.from_tuples(core.columns)

    # stack → long format
    long_df = (
        core.stack(level=[0, 1, 2])
        .reset_index()
        .rename(
            columns={
                "level_0": "Frame",
                "level_1": "ligand",
                "level_2": "protein",
                "level_3": "interaction",
                0: "value",
            }
        )
    )

    # attach metadata (safe + simple)
    for k, v in meta.items():
        long_df[k] = v

    return long_df
