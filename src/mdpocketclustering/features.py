import numpy as np


def compute_pocket_volume(run, cutoff: float = 8.0, normalize: bool = True):
    u = run.universe()

    ligand = u.select_atoms("resname AP1 or resname MG1")

    pocket_vol = []

    n_frames = len(u.trajectory)
    start = int(n_frames * 0.8)

    protein = u.select_atoms("protein")

    for i, ts in enumerate(u.trajectory):
        if i < start:
            continue

        lig_center = ligand.center_of_mass()

        distances = np.linalg.norm(protein.positions - lig_center, axis=1)

        pocket_atoms = protein[distances < cutoff]

        pocket_vol.append(len(pocket_atoms))

    pocket_vol = np.array(pocket_vol)

    if normalize and len(pocket_vol) > 0:
        pocket_vol = pocket_vol / (pocket_vol.max() + 1e-8)

    return {
        "run_id": run.run_id,
        "replicate": run.replicate,
        "mutation": run.system.mutation_label,
        "pocket_volume_mean": float(pocket_vol.mean()),
        "pocket_volume_std": float(pocket_vol.std()),
        "pocket_volume_ts": pocket_vol,
    }
