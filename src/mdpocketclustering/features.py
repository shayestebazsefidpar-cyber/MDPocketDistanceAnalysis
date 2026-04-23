import numpy as np


def compute_pocket_volume(run, cutoff=8.0):

    u = run.universe()

    pocket_vol = []

    n_frames = len(u.trajectory)
    start = int(n_frames)

    for i, ts in enumerate(u.trajectory):
        if i < start:
            continue

        ligand = u.select_atoms("resname AP1 or resname MG1")
        lig_center = ligand.center_of_mass()

        pocket_atoms = u.select_atoms(f"protein and around {cutoff} point {lig_center}")

        pocket_vol.append(len(pocket_atoms))

    pocket_vol = np.array(pocket_vol)

    return {
        "mean": float(pocket_vol.mean()) if len(pocket_vol) > 0 else None,
        "std": float(pocket_vol.std()) if len(pocket_vol) > 0 else None,
        "ts": pocket_vol,
    }
