import MDAnalysis as mda
import numpy as np


def compute_ligand_pocket_distance(sim, stride=100):

    u = mda.Universe(sim.get_topology(), sim.get_trajectory())

    lig_sel = "resname AP1" if sim.has_mg else "resname LIG"
    lig = u.select_atoms(lig_sel)

    pocket = u.select_atoms(f"protein and around 8 {lig_sel}")

    X = []
    meta = []

    for i, ts in enumerate(u.trajectory[::stride]):
        d = np.linalg.norm(lig.center_of_mass() - pocket.center_of_mass())

        X.append(d)

        frame_id = getattr(ts, "frame", i)

        meta.append(
            {
                "replicate": sim.replicate,
                "frame": frame_id,
                "binding_energy": sim.binding_energy,
            }
        )

    return np.array(X).reshape(-1, 1), meta
