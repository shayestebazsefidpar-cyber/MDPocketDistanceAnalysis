import os

import MDAnalysis as mda
import pandas as pd
from joblib import Parallel, delayed
from prolif import Fingerprint


def run_prolif_for_registry(
    registry,
    output_dir="prolife_output",
    output_file="prolif_all_runs.csv",
    stride=20,
    n_jobs=1,
):

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # create file with header only if not exists
    if not os.path.exists(output_path):
        pd.DataFrame().to_csv(output_path, index=False)

    def run_single(run):

        print(f"\n🚀 Running: {run.run_id}")

        u = mda.Universe(run.files.topology, run.files.trajectory)

        muts = run.system.mutations
        mutation = (
            f"{muts[0].chain}:{muts[0].wildtype}{muts[0].resid}{muts[0].mutant}"
            if muts
            else "WT"
        )

        protein = u.select_atoms("protein")

        # LIGAND DETECTION

        possible_ligands = ["AP1", "ATP", "ADP", "LIG", "MG", "MG1"]

        found_ligands = [
            lig for lig in possible_ligands if lig in set(u.atoms.resnames)
        ]

        if len(found_ligands) == 0:
            print(f"❌ No ligand in {run.run_id}")
            return None

        ligand_sel = " or ".join([f"resname {lig}" for lig in found_ligands])

        ligand = u.select_atoms(ligand_sel)

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

        # split protein
        long_df["residue"] = long_df["protein"].str.extract(r"([A-Z]+)")
        long_df["resid"] = long_df["protein"].str.extract(r"(\d+)").astype(int)
        long_df = long_df.drop(columns=["protein"])

        for k, v in meta.items():
            long_df[k] = v

        file_exists = os.path.isfile(output_path)

        long_df.to_csv(output_path, mode="a", header=not file_exists, index=False)

        print(f"💾 appended → {run.run_id}")

        return run.run_id

    if n_jobs == 1:
        for r in registry.runs:
            run_single(r)
    else:
        Parallel(n_jobs=n_jobs)(delayed(run_single)(r) for r in registry.runs)

    print(f"\n✅ DONE → {output_path}")
