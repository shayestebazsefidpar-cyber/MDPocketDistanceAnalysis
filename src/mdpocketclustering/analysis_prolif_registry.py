import os
import re
import warnings

import MDAnalysis as mda
import pandas as pd
from prolif import Fingerprint
from tqdm import tqdm


def run_prolif_for_registry(
    registry,
    ligand_sel=None,
    stride=100,
    out_dir="prolif_output",
    resume=True,
):

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    output_csv = os.path.join(out_dir, "prolif_output.csv")
    done_file = os.path.join(out_dir, "done_runs.csv")

    warnings.filterwarnings("ignore")

    possible_ligands = ["AP1", "ATP", "ADP", "LIG", "MG", "MG1"]

    if resume and os.path.exists(done_file) and os.path.getsize(done_file) > 0:
        try:
            df_done = pd.read_csv(done_file)
            done_runs = set(df_done["run_id"].astype(str))
        except Exception:
            done_runs = set()
    else:
        done_runs = set()

    if not os.path.exists(output_csv):
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
        ).to_csv(output_csv, index=False)

    for run in tqdm(registry.runs, desc="ProLIF"):
        run_id = str(run.run_id)

        if run_id in done_runs:
            print(f"⏩ skipping {run_id}")
            continue

        try:
            print(f"\n🚀 Processing: {run_id}")

            u = mda.Universe(run.files.topology, run.files.trajectory)
            protein = u.select_atoms("protein")

            ligand = None
            ligand_name = None

            if ligand_sel:
                ligand = u.select_atoms(ligand_sel)
                ligand_name = ligand_sel

            if ligand is None or len(ligand) == 0:
                for lig in possible_ligands:
                    tmp = u.select_atoms(f"resname {lig}")
                    if len(tmp) > 0:
                        ligand = tmp
                        ligand_name = lig
                        break

            if ligand is None or len(ligand) == 0:
                print(f"⚠️ No ligand found in {run_id}")
                continue

            m = run.system.mutations[0] if run.system.mutations else None
            mutation = f"A:{m.wildtype}{m.resid}{m.mutant}" if m else "WT"

            fp = Fingerprint()
            fp.run(u.trajectory[::stride], lig=ligand, prot=protein)

            df = fp.to_dataframe()

            records = []

            if isinstance(df.columns, pd.MultiIndex):
                for frame_idx, row in df.iterrows():
                    for col, value in row.items():
                        raw = str(col[0]).split(".")[0]

                        match = re.match(r"([A-Za-z]+)(\d+)", raw)

                        if match:
                            residue = match.group(1)
                            resid = match.group(2)
                        else:
                            residue = "unknown"
                            resid = "unknown"

                        interaction = col[2] if len(col) > 2 else "unknown"

                        records.append(
                            {
                                "Frame": frame_idx,
                                "ligand": ligand_name,
                                "interaction": interaction,
                                "value": bool(value),
                                "residue": residue,
                                "resid": resid,
                                "run_id": run_id,
                                "mutation": mutation,
                            }
                        )

            if len(records) == 0:
                print(f"⚠️ empty result for {run_id}")
                continue

            df_run = pd.DataFrame(records)

            df_run.to_csv(
                output_csv,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8-sig",
            )

            pd.DataFrame([{"run_id": run_id}]).to_csv(
                done_file, mode="a", header=not os.path.exists(done_file), index=False
            )

            done_runs.add(run_id)

        except Exception as e:
            print(f"❌ Error in {run_id}: {e}")
            continue

    return output_csv
