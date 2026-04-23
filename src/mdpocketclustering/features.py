import numpy as np


def compute_pocket_volume(run, cutoff: float = 8.0, normalize: bool = True):

    u = run.universe()

    ligand = u.select_atoms("resname AP1 or resname MG1")

    if len(ligand) == 0:
        raise ValueError(f"No ligand found in run {run.run_id}")

    pocket_vol = []

    for ts in u.trajectory:
        pocket_atoms = u.select_atoms(
            f"(protein) and around {cutoff} (resname AP1 or resname MG1)"
        )
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
