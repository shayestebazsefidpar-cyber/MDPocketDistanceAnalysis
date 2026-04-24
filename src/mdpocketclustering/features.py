import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array


def extract_md_features(u, run_id, mutation, ligand_sel, cutoff=3.5, stride=10):
    """
    ============================================================
    🧠 MOLECULAR DYNAMICS FEATURE EXTRACTION (FAST + CLEAN)
    ============================================================

    PURPOSE:
    Extract compact ML-ready descriptors from MD trajectories
    for binding analysis and WT vs mutant comparison.

    ------------------------------------------------------------
    FEATURES COMPUTED:

    1) avg_contact_fraction
       → fraction of protein atoms within cutoff of ligand

    2) avg_distance
       → mean ligand–protein atom distance

    3) min_distance
       → closest observed ligand–protein contact

    4) contact_density
       → normalized density of atomic contacts

    ------------------------------------------------------------
    BIOPHYSICAL MEANING:

    ↑ contact_fraction  → stronger binding
    ↓ avg_distance      → tighter complex
    ↓ min_distance      → stronger transient contacts
    ↑ contact_density   → richer interaction network

    ============================================================
    """

    lig = u.select_atoms(ligand_sel)
    protein = u.select_atoms("protein")

    n_res = len(protein.residues)

    contact_fraction_list = []
    mean_distance_list = []
    min_distance_list = []
    contact_density_list = []

    # ============================================================
    # TRAJECTORY SAMPLING
    # ============================================================
    for ts in u.trajectory[::stride]:
        lig_pos = lig.positions
        prot_pos = protein.positions

        # pairwise distances (fast C implementation)
        d = distance_array(prot_pos, lig_pos)

        contact_mask = d < cutoff

        # ========================================================
        # FRAME FEATURES
        # ========================================================
        contact_fraction = contact_mask.sum() / n_res
        mean_distance = d.mean()
        min_distance = d.min()
        contact_density = contact_mask.mean()

        contact_fraction_list.append(contact_fraction)
        mean_distance_list.append(mean_distance)
        min_distance_list.append(min_distance)
        contact_density_list.append(contact_density)

    # ============================================================
    # FINAL AGGREGATION (ONE ROW PER RUN)
    # ============================================================
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "mutation": mutation,
                "avg_contact_fraction": np.mean(contact_fraction_list),
                "avg_distance": np.mean(mean_distance_list),
                "min_distance": np.min(min_distance_list),
                "contact_density": np.mean(contact_density_list),
            }
        ]
    )
