import MDAnalysis as mda
import MDAnalysis.analysis.prolif
import pandas as pd


def run_prolif_for_registry(registry, ligand_sel=None, out_csv=None):

    all_frames = []

    for run in registry.runs:
        # -------------------------
        # Load system
        # -------------------------
        u = mda.Universe(run.files.topology, run.files.trajectory)

        # -------------------------
        # Ligand selection
        # -------------------------
        selection = ligand_sel if ligand_sel is not None else "resname ATP"
        ligand_atoms = u.select_atoms(selection)

        if len(ligand_atoms) == 0:
            raise ValueError("Ligand not found")

        ligand_name = selection.replace("resname", "").strip()

        # -------------------------
        # ProLIF fingerprint
        # -------------------------
        fp = MDAnalysis.analysis.prolif.Fingerprint(u)
        df = fp.to_dataframe()

        # -------------------------
        # REQUIRED: expand MultiIndex into residue + interaction
        # -------------------------
        if isinstance(df.columns, pd.MultiIndex):
            records = []

            for frame_idx, row in df.iterrows():
                for col, value in row.items():
                    # col = (residue_name, resid, interaction_type)
                    if isinstance(col, tuple):
                        residue = col[0]
                        resid = col[1]
                        interaction = col[2] if len(col) > 2 else "unknown"
                    else:
                        residue = "unknown"
                        resid = "unknown"
                        interaction = str(col)

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

            df = pd.DataFrame(records)

        else:
            # fallback (rare case)
            df = df.reset_index(drop=True)
            df["Frame"] = df.index
            df["ligand"] = ligand_name
            df["interaction"] = "unknown"
            df["value"] = True
            df["residue"] = "unknown"
            df["resid"] = "unknown"
            df["run_id"] = run.run_id
            df["mutation"] = "unknown"

        # -------------------------
        # enforce final column order
        # -------------------------
        df = df[
            [
                "Frame",
                "ligand",
                "interaction",
                "value",
                "residue",
                "resid",
                "run_id",
                "mutation",
            ]
        ]

        all_frames.append(df)

    # -------------------------
    # merge runs
    # -------------------------
    final_df = pd.concat(all_frames, ignore_index=True)

    # -------------------------
    # export CSV
    # -------------------------
    if out_csv:
        final_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return final_df
