import os
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

    # -----------------------------
    # checkpoint folder (global for function)
    # -----------------------------
    checkpoint_dir = "checkpoints_prolif"
    os.makedirs(checkpoint_dir, exist_ok=True)

    def run_single(run):

        checkpoint_file = os.path.join(checkpoint_dir, f"{run.run_id}.csv")

        # -----------------------------
        # SKIP if already computed
        # -----------------------------
        if os.path.exists(checkpoint_file):
            print(f"⏩ Skip {run.run_id}")
            return pd.read_csv(checkpoint_file)

        # -----------------------------
        # load system
        # -----------------------------
        u = mda.Universe(run.files.topology, run.files.trajectory)

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

        # -----------------------------
        # fingerprint setup
        # -----------------------------
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

        meta = {
            "run_id": run.run_id,
            "mutation": mutation,
        }

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

        # -----------------------------
        # SAVE CHECKPOINT
        # -----------------------------
        long_df.to_csv(checkpoint_file, index=False)
        print(f"💾 Saved: {checkpoint_file}")

        return long_df

    # -----------------------------
    # RUN (serial or parallel)
    # -----------------------------
    if n_jobs == 1:
        results = [run_single(r) for r in registry.runs]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(run_single)(r) for r in registry.runs)

    return pd.concat(results, ignore_index=True)
