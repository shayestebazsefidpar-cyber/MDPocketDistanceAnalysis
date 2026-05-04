import pandas as pd
import MDAnalysis as mda
from prolif import Fingerprint


def run_prolif_for_single_traj(
    run,
    ligand_sel="resname AP1 or resname MG1",
    stride=20,
):
    """
    Run ProLIF for a single trajectory (registry run object)
    """

    # --- load universe ---
    u = mda.Universe(run.files.topology, run.files.trajectory)

    # --- mutation handling ---
    mutations = run.system.mutations
    mutation = (
        f"{mutations[0].chain}:{mutations[0].wildtype}{mutations[0].resid}{mutations[0].mutant}"
        if mutations else "WT"
    )

    # --- selections ---
    protein = u.select_atoms("protein")
    ligand = u.select_atoms(ligand_sel)

    if len(ligand) == 0:
        print("❌ No ligand found")
        return pd.DataFrame()

    # --- ProLIF fingerprint ---
    fp = Fingerprint([
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
    ])

    # --- run ---
    fp.run(u.trajectory[::stride], ligand, protein)
    df = fp.to_dataframe()

    # --- add metadata ---
    df["run_id"] = run.run_id
    df["mutation"] = mutation

    return df
