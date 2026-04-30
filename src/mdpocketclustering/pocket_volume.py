import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm


def pocket_volume_analysis(
    u,
    run_id,
    mutation,
    pocket_sel,
    protein_sel="protein",
    spacing=0.5,
    buffer=4.0,
    probe_radius=1.4,
    stride=20,
    debug=False,
):
    """
    Voxel-based pocket volume estimation.

    Returns:
        DataFrame with frame-wise pocket volume.
    """

    pocket = u.select_atoms(pocket_sel)
    protein = u.select_atoms(protein_sel)

    if pocket.n_atoms == 0:
        raise ValueError("No pocket atoms found.")

    volumes = []

    traj = u.trajectory[::stride]

    for i, ts in enumerate(tqdm(traj, desc=f"{run_id}-{mutation}")):
        center = pocket.center_of_mass()

        x = np.arange(center[0] - buffer, center[0] + buffer, spacing)
        y = np.arange(center[1] - buffer, center[1] + buffer, spacing)
        z = np.arange(center[2] - buffer, center[2] + buffer, spacing)

        grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        grid = grid.reshape(-1, 3)

        dist = cdist(grid, protein.positions)
        min_dist = dist.min(axis=1)

        cavity = min_dist > probe_radius
        cavity = cavity.reshape(len(x), len(y), len(z))

        voxel_volume = spacing**3
        volume = cavity.sum() * voxel_volume

        volumes.append(volume)

        if debug:
            print(f"{run_id}-{mutation} frame={ts.frame} volume={volume:.2f}")

    # ✅ IMPORTANT FIX: return DataFrame (not numpy array)
    return pd.DataFrame(
        {
            "frame": np.arange(len(volumes)) * stride,
            "volume_A3": np.array(volumes),
            "run_id": run_id,
            "mutation": mutation,
        }
    )
