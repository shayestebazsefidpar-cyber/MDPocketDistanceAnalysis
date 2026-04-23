import numpy as np


def compute_pocket_contact_number(run, cutoff: float = 8.0, normalize: bool = True):

    u = run.universe()

    ligand = u.select_atoms("resname AP1 or resname MG1")

    if len(ligand) == 0:
        raise ValueError(f"No ligand found in run {run.run_id}")

    n_frames = len(u.trajectory)
    start = int(n_frames * 0.8)

    protein = u.select_atoms("protein")

    contact_ts = []

    for i, ts in enumerate(u.trajectory):
        if i < start:
            continue

        lig_center = ligand.center_of_mass()

        distances = np.linalg.norm(protein.positions - lig_center, axis=1)

        contact_atoms = protein[distances < cutoff]

        contact_ts.append(len(contact_atoms))

    contact_ts = np.array(contact_ts)

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
