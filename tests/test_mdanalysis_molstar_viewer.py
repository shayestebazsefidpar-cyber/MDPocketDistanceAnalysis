import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.core.universe import Universe

from mdpocketclustering.mdanalysis_molstar_viewer import show_mda_frame


def test_show_mda_frame_runs():

    n_atoms = 50
    n_residues = 10
    n_frames = 5

    coords = np.random.rand(n_frames, n_atoms, 3) * 10

    # ============================================================
    # 🧬 TOPOLOGY (FIXED ONLY IN TEST)
    # ============================================================

    u = Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=np.repeat(np.arange(n_residues), 5),
        trajectory=True,
    )

    # atom-level attributes
    u.add_TopologyAttr("name", ["CA"] * n_atoms)

    # residue-level attributes
    u.add_TopologyAttr("resname", ["ALA"] * n_residues)
    u.add_TopologyAttr("resid", np.arange(1, n_residues + 1))

    # segment-level attribute (IMPORTANT FIX)
    u.add_TopologyAttr("segid", ["A"])  # ✅ FIXED (ONLY 1 VALUE)

    # trajectory
    u.load_new(coords, format=MemoryReader)

    # call your function (UNCHANGED)
    view = show_mda_frame(u, frame=2)

    assert view is not None
