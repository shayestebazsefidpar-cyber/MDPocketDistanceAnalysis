import os
import warnings

import MDAnalysis as mda
import pandas as pd
from prolif import Fingerprint
from tqdm import tqdm


def run_prolif_for_registry(
    registry,
    possible_ligands=None,
    stride=20,
    outdir="prolif_output",
    save_csv=True,
):
    """
    Run ProLIF for all trajectories in a registry with automatic ligand detection.
    """

    if possible_ligands is None:
        possible_ligands = ["AP1", "MG1", "ATP", "ADP", "LIG", "MG"]

    os.makedirs(outdir, exist_ok=True)

    all_data = []

    for run in tqdm(registry.runs, desc="ProLIF registry runs"):
        # --- load universe ---
        u = mda.Universe(run.files.topology, run.files.trajectory)

        # --- detect ligand automatically ---
        found = [l for l in possible_ligands if l in set(u.atoms.resnames)]

        if not found:
            warnings.warn(f"[{run.run_id}] no ligand found, skipping")
            continue

        ligand_sel = "resname " + " ".join(found)

        ligand = u.select_atoms(ligand_sel)
        protein = u.select_atoms("protein")

        if len(ligand) == 0:
            warnings.warn(f"[{run.run_id}] empty ligand selection, skipping")
            continue

        # --- mutation ---
        mutations = run.system.mutations
        mutation = (
            f"{mutations[0].chain}:{mutations[0].wildtype}"
            f"{mutations[0].resid}{mutations[0].mutant}"
            if mutations
            else "WT"
        )

        # --- fingerprint ---
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

        for ts in tqdm(u.trajectory[::stride], desc=f"{run.run_id}", leave=False):
            fp.run([ts], ligand, protein)

        df = fp.to_dataframe()

        # --- flatten to long format (simple + stable) ---
        df = df.reset_index()
        df = df.melt(id_vars=df.columns[:3], var_name="interaction", value_name="value")

        df["run_id"] = run.run_id
        df["mutation"] = mutation

        # --- save per run ---
        if save_csv:
            outpath = f"{outdir}/prolif_{run.run_id}.csv"
            df.to_csv(outpath, index=False)
            warnings.warn(f"[{run.run_id}] saved → {outpath}")

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
