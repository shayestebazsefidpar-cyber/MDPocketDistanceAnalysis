import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array


def extract_md_features(
    u, run_id, mutation, ligand_sel, cutoff=3.5, stride=100, protein_mode="CA"
):
    """
    ============================================================
    🧠 MOLECULAR DYNAMICS FEATURE EXTRACTION (Cα / BACKBONE VERSION)
    ============================================================

    PURPOSE:
    This function extracts compact, ML-ready features from MD trajectories
    using a coarse-grained representation of the protein.

    Two supported representations:
        - "CA"       → C-alpha atoms (default, fastest, most common for PCA)
        - "backbone" → N, CA, C, O atoms (slightly more detailed)

    ------------------------------------------------------------
    WHY Cα / BACKBONE?

    - Reduces noise from side-chain fluctuations
    - Improves stability for PCA and clustering
    - Greatly improves computational speed
    - Preserves global protein motion and binding trends

    ------------------------------------------------------------
    METHODOLOGY:

    For each sampled frame (controlled by stride):

    1) Select reduced protein representation (CA or backbone)
    2) Compute ligand–protein distances (vectorized)
    3) Define contacts using cutoff threshold (3.5 Å)
    4) Extract global interaction descriptors

    ------------------------------------------------------------
    FEATURES:

        avg_contact_fraction:
            fraction of protein CA/backbone atoms in contact with ligand

        avg_distance:
            mean ligand–protein distance

        min_distance:
            closest ligand–protein approach

        contact_density:
            fraction of CA/backbone atoms within cutoff

    ------------------------------------------------------------
    BIOPHYSICAL MEANING:

    - High contact_fraction → strong binding
    - Low distance → close binding state
    - High density → stable interaction network

    ============================================================
    """

    lig = u.select_atoms(ligand_sel)

    # ============================================================
    # 🧠 PROTEIN REPRESENTATION SELECTION (KEY CHANGE)
    # ============================================================
    if protein_mode == "CA":
        protein = u.select_atoms("name CA")
    elif protein_mode == "backbone":
        protein = u.select_atoms("backbone")
    else:
        raise ValueError("protein_mode must be 'CA' or 'backbone'")

    n_atoms = len(protein)

    feat = []

    # ============================================================
    # 🧠 TRAJECTORY SAMPLING (FAST STRIDED PROCESSING)
    # ============================================================
    for ts in u.trajectory[::stride]:
        lig_pos = lig.positions

        # vectorized distance calculation
        d = distance_array(protein.positions, lig_pos)

        # contact definition (binary map)
        contact_mask = d < cutoff

        # ========================================================
        # 🧠 GLOBAL FEATURES (PCA-READY)
        # ========================================================
        contact_fraction = contact_mask.sum() / n_atoms
        mean_dist = d.mean()
        min_dist = d.min()
        contact_density = contact_mask.mean()

        feat.append([contact_fraction, mean_dist, min_dist, contact_density])

    feat = np.array(feat)

    # ============================================================
    # 🧠 FINAL OUTPUT (ONE ROW PER SIMULATION)
    # ============================================================
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "mutation": mutation,
                "protein_mode": protein_mode,
                "avg_contact_fraction": feat[:, 0].mean(),
                "avg_distance": feat[:, 1].mean(),
                "min_distance": feat[:, 2].min(),
                "contact_density": feat[:, 3].mean(),
            }
        ]
    )
