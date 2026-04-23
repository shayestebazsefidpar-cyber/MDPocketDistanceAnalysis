import numpy as np


def compute_pocket_contact_number(run, cutoff: float = 8.0, normalize: bool = True):
    """
    ============================================================
    🧠 PSEUDO-POCKET CONTACT ANALYSIS PIPELINE
    ============================================================

    This function estimates ligand–protein pocket compactness
    using a distance-based contact metric.

    Biological meaning:
    - High values → tight binding pocket
    - Low values  → open/weak binding state
    ============================================================
    """

    u = run.universe()

    # --------------------------------------------------
    # Build ligand dynamically from system definition
    # --------------------------------------------------
    ligand_names = []

    if run.system.has_component("ATP"):
        ligand_names.append("AP1")

    if run.system.has_component("MG"):
        ligand_names.append("MG1")

    if len(ligand_names) == 0:
        raise ValueError(f"No ligand defined in run {run.run_id}")

    sel = " or ".join([f"resname {r}" for r in ligand_names])
    ligand = u.select_atoms(sel)

    if len(ligand) == 0:
        raise ValueError(f"No ligand found in run {run.run_id}")

    # --------------------------------------------------
    # Analysis setup
    # --------------------------------------------------
    n_frames = len(u.trajectory)
    start = int(n_frames * 0.6)

    protein = u.select_atoms("protein")

    contact_ts = []

    # --------------------------------------------------
    # Main loop over trajectory
    # --------------------------------------------------
    for i, ts in enumerate(u.trajectory):
        if i < start:
            continue

        lig_center = ligand.center_of_mass()

        distances = np.linalg.norm(protein.positions - lig_center, axis=1)

        contact_atoms = protein[distances < cutoff]

        contact_ts.append(len(contact_atoms))

    contact_ts = np.array(contact_ts)

    # --------------------------------------------------
    # normalization
    # --------------------------------------------------
    if normalize and len(contact_ts) > 0:
        contact_ts = contact_ts / (contact_ts.max() + 1e-8)

    return {
        "run_id": run.run_id,
        "replicate": run.replicate,
        "mutation": run.system.mutation_label,
        "pocket_contact_mean": float(contact_ts.mean())
        if len(contact_ts) > 0
        else None,
        "pocket_contact_std": float(contact_ts.std()) if len(contact_ts) > 0 else None,
        "pocket_contact_ts": contact_ts,
    }
