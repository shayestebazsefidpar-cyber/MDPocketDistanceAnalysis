import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.core.topology import Topology
from MDAnalysis.core.universe import Universe

# import your function
from mdpocketclustering.contact_fingerprint import compute_ligand_residue_contacts


def make_fake_universe(n_res=10, n_atoms_per_res=5, n_lig_atoms=3, n_frames=5):

    # --- topology ---
    atoms = []
    resids = []
    resnames = []

    atom_id = 1

    for r in range(n_res):
        for a in range(n_atoms_per_res):
            atoms.append(atom_id)
            resids.append(r + 1)
            resnames.append("ALA")
            atom_id += 1

    # ligand atoms
    for a in range(n_lig_atoms):
        atoms.append(atom_id)
        resids.append(999)
        resnames.append("LIG")
        atom_id += 1

    n_atoms = len(atoms)

    topology = mda.topology.Topology.from_dict(
        {
            "atoms": {
                "ids": atoms,
                "names": ["C"] * n_atoms,
                "resnames": resnames,
                "resids": resids,
            }
        }
    )

    u = Universe(topology)

    # --- fake trajectory positions ---
    u.load_new(np.random.rand(n_frames, n_atoms, 3) * 30.0)

    return u
