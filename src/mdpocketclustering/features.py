import numpy as np
import pandas as pd


def run_global_occupancy(u, run_id, mutation, ligand_sel, cutoff=3.5, bin_size=1000):

    lig = u.select_atoms(ligand_sel)
    protein = u.select_atoms("protein")

    n_res = len(protein.residues)
    n_frames = len(u.trajectory)

    frame_occ = []

    res_atoms = [r.atoms for r in protein.residues]

    for start in range(0, n_frames, bin_size):
        bin_occ = []

        for i in range(bin_size):
            idx = start + i
            if idx >= n_frames:
                break

            u.trajectory[idx]

            lig_com = lig.center_of_mass()

            res_coms = np.array([r.center_of_mass() for r in res_atoms])

            dists = np.linalg.norm(res_coms - lig_com, axis=1)

            n_contacts = np.sum(dists < cutoff)

            occupancy = n_contacts / n_res

            bin_occ.append(occupancy)

        if len(bin_occ) > 0:
            frame_occ.append(np.mean(bin_occ))

    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "mutation": mutation,
                "global_occupancy_%": 100 * np.mean(frame_occ),
            }
        ]
    )
