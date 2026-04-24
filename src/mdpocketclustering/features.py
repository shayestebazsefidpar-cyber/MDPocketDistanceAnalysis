import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array


def extract_md_features(u, run_id, mutation, ligand_sel, cutoff=3.5, stride=10):
    """
    ============================================================
    🧠 MOLECULAR DYNAMICS FEATURE EXTRACTION (ROBUST VERSION)
    ============================================================

    PURPOSE:
    Extract ML-ready features from MD trajectories for
    binding analysis and PCA/clustering.

    ------------------------------------------------------------
    FEATURES:

    1) avg_contact_fraction → binding strength proxy
    2) avg_distance         → global separation
    3) min_distance         → closest contact
    4) contact_density      → interaction richness

    ------------------------------------------------------------
    SAFETY:
    - Handles missing ligands (empty selections)
    - Avoids empty array crashes
    - Returns NaNs if system invalid

    ============================================================
    """

    lig = u.select_atoms(ligand_sel)
    protein = u.select_atoms("protein")

    # ============================================================
    # 🧠 SAFETY CHECK (IMPORTANT FIX)
    # ============================================================
    if lig.n_atoms == 0 or protein.n_atoms == 0:
        return pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "mutation": mutation,
                    "avg_contact_fraction": np.nan,
                    "avg_distance": np.nan,
                    "min_distance": np.nan,
                    "contact_density": np.nan,
                }
            ]
        )

    n_res = len(protein.residues)

    contact_fraction_list = []
    mean_distance_list = []
    min_distance_list = []
    contact_density_list = []

    # ============================================================
    # 🧠 TRAJECTORY LOOP
    # ============================================================
    for ts in u.trajectory[::stride]:
        lig_pos = lig.positions
        prot_pos = protein.positions

        # skip if ligand disappears in frame
        if lig_pos.size == 0 or prot_pos.size == 0:
            continue

        # pairwise distances
        d = distance_array(prot_pos, lig_pos)

        # skip empty distance arrays
        if d.size == 0:
            continue

        contact_mask = d < cutoff

        contact_fraction = contact_mask.sum() / n_res
        mean_distance = d.mean()
        min_distance = d.min()
        contact_density = contact_mask.mean()

        contact_fraction_list.append(contact_fraction)
        mean_distance_list.append(mean_distance)
        min_distance_list.append(min_distance)
        contact_density_list.append(contact_density)

    # ============================================================
    # 🧠 FINAL OUTPUT (SAFE AGGREGATION)
    # ============================================================
    if len(contact_fraction_list) == 0:
        return pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "mutation": mutation,
                    "avg_contact_fraction": np.nan,
                    "avg_distance": np.nan,
                    "min_distance": np.nan,
                    "contact_density": np.nan,
                }
            ]
        )

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
