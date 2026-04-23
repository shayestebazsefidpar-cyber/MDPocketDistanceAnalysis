import MDAnalysis as mda
import numpy as np


def compute_pocket_volume(run, cutoff: float = 8.0, normalize: bool = True):

    u = mda.Universe(run.tpr_path, run.xtc_path)

    ligand = u.select_atoms("resname AP1 or resname MG1")

    if len(ligand) == 0:
        raise ValueError(f"No ligand found in run {run.run_id}")

    pocket_vol = []

    for ts in u.trajectory:
        # stable pocket definition: distance from ligand atoms
        pocket_atoms = u.select_atoms(
            f"(protein) and around {cutoff} (resname AP1 or resname MG1)"
        )

        vol = len(pocket_atoms)
        pocket_vol.append(vol)

    pocket_vol = np.array(pocket_vol)

    if normalize and len(pocket_vol) > 0:
        pocket_vol = pocket_vol / (np.max(pocket_vol) + 1e-8)

    return {
        "run_id": run.run_id,
        "pocket_volume_ts": pocket_vol,
        "pocket_volume_mean": float(np.mean(pocket_vol)),
        "pocket_volume_std": float(np.std(pocket_vol)),
    }
