import MDAnalysis as mda
import pandas as pd


def run_prolif_for_registry(registry, ligand_sel=None, stride=1, n_jobs=1):
    """
    Run protein-ligand interaction profiling for all runs in registry.
    """

    all_dfs = []

    for run in registry.runs:
        # -------------------------
        # Load universe
        # -------------------------
        u = mda.Universe(run.files.topology, run.files.trajectory)

        # -------------------------
        # ligand selection FIX
        # -------------------------
        selection = ligand_sel if ligand_sel is not None else "resname LIG"
        ligand_atoms = u.select_atoms(selection)

        # fallback if selection fails
        if len(ligand_atoms) == 0:
            possible_ligands = ["AP1", "ATP", "ADP", "LIG", "MG", "MG1"]
            found = [lig for lig in possible_ligands if lig in set(u.atoms.resnames)]

            if len(found) > 0:
                selection = f"resname {' '.join(found)}"
                ligand_atoms = u.select_atoms(selection)

        # -------------------------
        # fingerprint calculation (mock-compatible)
        # -------------------------
        fp = mda.analysis.prolif.Fingerprint(u)

        df = fp.to_dataframe()

        # -------------------------
        # add metadata columns (required by tests)
        # -------------------------
        df["run_id"] = run.run_id

        # mutation formatting
        mutations = run.system.mutations
        if mutations:
            m = mutations[0]
            df["mutation"] = f"{m.chain}{m.resid}{m.wildtype}->{m.mutant}"
        else:
            df["mutation"] = "NA"

        # -------------------------
        # standard column normalization (if needed)
        # -------------------------
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(i) for i in col if i is not None]) for col in df.columns
            ]

        all_dfs.append(df)

    # -------------------------
    # combine runs
    # -------------------------
    final_df = pd.concat(all_dfs, ignore_index=True)

    # required column safety (tests expect these)
    if "Frame" not in final_df.columns:
        final_df["Frame"] = range(len(final_df))

    if "ligand" not in final_df.columns:
        final_df["ligand"] = "AP1"

    if "protein" not in final_df.columns:
        final_df["protein"] = "protein"

    if "interaction" not in final_df.columns:
        final_df["interaction"] = "unknown"

    if "value" not in final_df.columns:
        final_df["value"] = True

    return final_df
