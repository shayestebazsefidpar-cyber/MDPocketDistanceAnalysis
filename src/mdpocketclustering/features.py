import numpy as np
import pandas as pd


def run_global_occupancy(u, run_id, mutation, ligand_sel, cutoff=3.5, bin_size=100):

    # select ligand atoms (AP1, MG1, etc.)
    lig = u.select_atoms(ligand_sel)

    # select all protein atoms
    protein = u.select_atoms("protein")

    # total number of frames in trajectory
    n_frames = len(u.trajectory)

    # store occupancy per bin (smoothed over time)
    frame_occ = []

    # pre-store atoms of each residue (faster than selecting every frame)
    res_atoms = [res.atoms for res in protein.residues]

    # total number of residues in protein
    n_res = len(res_atoms)

    # loop over trajectory in chunks (bins of 100 frames)
    for start in range(0, n_frames, bin_size):
        bin_occ = []  # occupancy values inside this bin

        # loop over frames inside each bin
        for i in range(bin_size):
            idx = start + i

            # stop if trajectory ends
            if idx >= n_frames:
                break

            # move trajectory to current frame
            u.trajectory[idx]

            # compute ligand center of mass (reference point)
            lig_com = lig.center_of_mass()

            # count how many residues are in contact in this frame
            n_contacts = 0

            # loop over residues
            for atoms in res_atoms:
                # compute distances of all atoms in residue to ligand COM
                d = np.linalg.norm(atoms.positions - lig_com, axis=1)

                # if ANY atom is within cutoff → residue is in contact
                if np.any(d < cutoff):
                    n_contacts += 1

            # normalize by total residues → fraction of contacted residues
            occupancy = n_contacts / n_res

            bin_occ.append(occupancy)

        # average occupancy within this bin
        if len(bin_occ) > 0:
            frame_occ.append(np.mean(bin_occ))

    # final output: average over all bins
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "mutation": mutation,
                "global_occupancy_%": 100 * np.mean(frame_occ),
            }
        ]
    )
