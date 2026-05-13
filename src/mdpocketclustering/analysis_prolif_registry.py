import os

import MDAnalysis as mda
import pandas as pd
from prolif import Fingerprint
from tqdm import tqdm


def run_prolif_for_registry(
    registry,
    ligand_sel=None,
    out_csv="out.csv",
    checkpoint_csv="checkpoint.csv",
    resume=True,
):

    processed_runs = set()

    if resume and os.path.exists(checkpoint_csv):
        try:
            existing = pd.read_csv(checkpoint_csv)
            processed_runs = set(existing["run_id"].unique())
        except Exception:
            processed_runs = set()

    if not os.path.exists(out_csv):
        pd.DataFrame(
            columns=[
                "Frame",
                "ligand",
                "interaction",
                "value",
                "residue",
                "resid",
                "run_id",
                "mutation",
            ]
        ).to_csv(out_csv, index=False)

    for run in tqdm(registry.runs, desc="ProLIF processing"):
        if run.run_id in processed_runs:
            continue

        u = mda.Universe(run.files.topology, run.files.trajectory)

        selection = ligand_sel if ligand_sel else "resname ATP"
        ligand = u.select_atoms(selection)
        protein = u.select_atoms("protein")

        if len(ligand) == 0:
            continue

        ligand_name = selection.replace("resname", "").strip()

        fp = Fingerprint()
        fp.run(u.trajectory, lig=ligand, prot=protein)

        df = fp.to_dataframe()

        records = []

        if isinstance(df.columns, pd.MultiIndex):
            for frame_idx, row in df.iterrows():
                for col, value in row.items():
                    residue = col[0] if len(col) > 0 else "unknown"
                    resid = col[1] if len(col) > 1 else "unknown"
                    interaction = col[2] if len(col) > 2 else "unknown"

                    records.append(
                        {
                            "Frame": frame_idx,
                            "ligand": ligand_name,
                            "interaction": interaction,
                            "value": bool(value),
                            "residue": residue,
                            "resid": resid,
                            "run_id": run.run_id,
                            "mutation": f"A:{run.system.mutations[0].wildtype}{run.system.mutations[0].resid}",
                        }
                    )

        df_run = pd.DataFrame(records)

        df_run.to_csv(
            out_csv, mode="a", header=False, index=False, encoding="utf-8-sig"
        )

        df_run.to_csv(
            checkpoint_csv,
            mode="a",
            header=not os.path.exists(checkpoint_csv),
            index=False,
            encoding="utf-8-sig",
        )

    return out_csv
