import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array


def extract_md_features(u, run_id, mutation, ligand_sel, cutoff=3.5, stride=100):
    """
    ============================================================
    🧠 MOLECULAR DYNAMICS FEATURE EXTRACTION (FAST VERSION)
    ============================================================

    PURPOSE:
    This function extracts compact, machine-learning-ready features
    from molecular dynamics trajectories for each simulation run.

    These features are designed for:
        → PCA (Principal Component Analysis)
        → clustering of conformational states
        → comparison of WT vs mutants
        → fast statistical analysis across replicas

    ------------------------------------------------------------
    METHODOLOGY:

    For each sampled trajectory frame (controlled by stride):

    1) Compute ligand–protein distances using atom-level coordinates
       (distance_array from MDAnalysis, C-optimized)

    2) Define contacts using a cutoff threshold (default = 3.5 Å)

    3) Extract global descriptors describing binding behavior:

        - avg_contact_fraction:
            fraction of protein atoms in contact with ligand

        - avg_distance:
            mean ligand–protein distance

        - min_distance:
            closest approach between ligand and protein

        - contact_density:
            density of atomic contacts below cutoff

    ------------------------------------------------------------
    COMPUTATIONAL STRATEGY:

    - Uses vectorized distance calculations (distance_array)
    - Uses trajectory subsampling (stride) for efficiency
    - Avoids residue-level loops for scalability

    ------------------------------------------------------------
    BIOPHYSICAL INTERPRETATION:

    These features describe:
        → binding strength (contact fraction, min distance)
        → binding stability (average distance)
        → interaction richness (contact density)

    Higher contact fraction and lower distance indicate
    stronger ligand–protein association.

    ============================================================
    """

    lig = u.select_atoms(ligand_sel)
    protein = u.select_atoms("protein")

    n_res = len(protein.residues)

    feat = []

    # ============================================================
    # 🧠 TRAJECTORY SAMPLING (REDUCED COST)
    # ============================================================
    for ts in u.trajectory[::stride]:
        lig_pos = lig.positions

        # compute all pairwise atom distances (vectorized)
        d = distance_array(protein.atoms.positions, lig_pos)

        # binary contact map (distance cutoff criterion)
        contact_mask = d < cutoff

        # ========================================================
        # 🧠 GLOBAL DESCRIPTORS (PCA-READY FEATURES)
        # ========================================================

        n_contacts = contact_mask.sum() / n_res
        mean_dist = d.mean()
        min_dist = d.min()
        contact_density = contact_mask.mean()

        feat.append([n_contacts, mean_dist, min_dist, contact_density])

    feat = np.array(feat)

    # ============================================================
    # 🧠 FINAL OUTPUT (ONE ROW PER SIMULATION)
    # ============================================================
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "mutation": mutation,
                "avg_contact_fraction": feat[:, 0].mean(),
                "avg_distance": feat[:, 1].mean(),
                "min_distance": feat[:, 2].min(),
                "contact_density": feat[:, 3].mean(),
            }
        ]
    )
