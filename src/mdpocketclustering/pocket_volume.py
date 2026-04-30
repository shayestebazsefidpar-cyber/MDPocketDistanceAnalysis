import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


def pocket_volume_analysis(
    u,                 # MDAnalysis Universe (trajectory + topology)
    run_id,            # simulation/run identifier
    mutation,          # mutation label (e.g., WT, A123V)
    pocket_sel,        # atom selection defining pocket region
    protein_sel="protein",  # selection for protein atoms
    spacing=0.5,       # grid resolution (voxel size, Å)
    buffer=4.0,        # padding around pocket (Å)
    probe_radius=1.4,  # probe size (solvent radius, Å)
    stride=20,         # frame sampling interval
    debug=False,       # enable debug output
):
    """
    Pure voxel-based pocket volume 

    - grid around pocket COM
    - protein exclusion via distance cutoff
    - voxel counting
    """

    pocket = u.select_atoms(pocket_sel)
    protein = u.select_atoms(protein_sel)

    if pocket.n_atoms == 0:
        raise ValueError("No pocket atoms found.")

    volumes = []

    # IMPORTANT: iterate correctly
    traj = u.trajectory[::stride]

    for ts in tqdm(traj, desc=f"{run_id}-{mutation}"):

        # no need to reassign u.trajectory[ts.frame] ❌

        # -------------------------
        # center of pocket
        # -------------------------
        center = pocket.center_of_mass()

        # -------------------------
        # grid definition
        # -------------------------
        x = np.arange(center[0] - buffer, center[0] + buffer, spacing)
        y = np.arange(center[1] - buffer, center[1] + buffer, spacing)
        z = np.arange(center[2] - buffer, center[2] + buffer, spacing)

        grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        grid = grid.reshape(-1, 3)

        # -------------------------
        # distance to protein atoms
        # -------------------------
        dist = cdist(grid, protein.positions)
        min_dist = dist.min(axis=1)

        # -------------------------
        # cavity definition (POVME-like idea)
        # inside pocket = far from protein
        # -------------------------
        cavity = (min_dist > probe_radius)

        cavity = cavity.reshape(len(x), len(y), len(z))

        voxel_volume = spacing ** 3
        volume = cavity.sum() * voxel_volume

        volumes.append(volume)

        if debug:
            print(f"{run_id}-{mutation} frame={ts.frame} volume={volume:.2f}")

    return np.array(volumes)
