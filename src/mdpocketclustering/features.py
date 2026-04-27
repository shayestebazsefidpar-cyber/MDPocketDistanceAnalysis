import numpy as np
import pandas as pd


def extract_md_features(
    u, run_id, mutation, ligand_sel, cutoff=8.0, stride=10, return_sparse=True
):
    """
    Ligand–residue interaction fingerprint (MD-based).

    Features:
    - min atom-atom distance per residue per frame
    - cutoff-based sparsification (optional)
    """

    lig = u.select_atoms(ligand_sel)
    protein = u.select_atoms("protein")

    if lig.n_atoms == 0 or protein.n_atoms == 0:
        return pd.DataFrame()

    residues = protein.residues
    data = []

    frames = u.trajectory[::stride]

    for i, ts in enumerate(frames):
        lig_pos = lig.positions

        row = {
            "run_id": run_id,
            "mutation": mutation,
            "frame": i,
        }

        for res in residues:
            res_pos = res.atoms.positions

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
