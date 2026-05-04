import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_ligand_residue_contacts(
    u,
    run_id,
    mutation,
    ligand_sel,
    cutoff=8.0,
    stride=20,
    return_sparse=True,
):
    """
    Ligand–residue contact fingerprint over trajectory.

    - computes min atom-atom distance per residue
    - sampled every `stride` frames
    - optional sparse output
    - progress bar enabled
    """

    lig = u.select_atoms(ligand_sel)
    protein = u.select_atoms("protein")

    if lig.n_atoms == 0 or protein.n_atoms == 0:
        return pd.DataFrame()

    residues = protein.residues

    # cache atoms (speed boost)
    res_atoms = [res.atoms for res in residues]

    data = []

    frames = u.trajectory[::stride]

    for i, ts in enumerate(tqdm(frames, desc=f"{run_id}-{mutation}")):
        lig_pos = lig.positions

        row = {
            "run_id": run_id,
            "mutation": mutation,
            "frame": ts.frame,
            "time_ps": ts.time if hasattr(ts, "time") else i * stride,
        }

        for res, res_at in zip(residues, res_atoms):
            res_pos = res_at.positions

            # vectorized distance
            dists = np.linalg.norm(res_pos[:, None, :] - lig_pos[None, :, :], axis=-1)

            min_dist = dists.min()

            col = f"{res.resname}{res.resid}"

            if return_sparse:
                if min_dist < cutoff:
                    row[col] = min_dist
            else:
                row[col] = min_dist

        data.append(row)

    return pd.DataFrame(data)
