from collections import defaultdict

import MDAnalysis as mda
import pandas as pd
from joblib import Parallel, delayed
from prolif import Fingerprint


def run_prolif_for_registry(
    registry,
    ligand_sel="resname AP1 or resname MG1",
    stride=20,
    n_jobs=1,
):
    """
    Full ProLIF pipeline for entire registry (multi-trajectory)
    returns: dict of DataFrames grouped by mutation
    """

    def run_single(run):

        # --- build universe ---
        u = mda.Universe(run.files.topology, run.files.trajectory)

        # --- mutation ---
        muts = run.system.mutations
        mutation = (
            f"{muts[0].chain}:{muts[0].wildtype}{muts[0].resid}{muts[0].mutant}"
            if muts
            else "WT"
        )

        protein = u.select_atoms("protein")
        ligand = u.select_atoms(ligand_sel)

        if len(ligand) == 0:
            print(f"❌ No ligand in run {run.run_id}")
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

        # --- metadata ---
        meta = {
            "run_id": run.run_id,
            "mutation": mutation,
        }

        # --- convert to long ---
        core = df.drop(columns=list(meta.keys()), errors="ignore")
        core.columns = pd.MultiIndex.from_tuples(core.columns)

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

        for k, v in meta.items():
            long_df[k] = v

        return long_df

    if n_jobs == 1:
        results = [run_single(r) for r in registry.runs]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(run_single)(r) for r in registry.runs)

    all_df = pd.concat(results, ignore_index=True)

    return all_df
