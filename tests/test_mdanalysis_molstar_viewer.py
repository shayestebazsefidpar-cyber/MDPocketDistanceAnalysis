import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.core.universe import Universe
from ipymolstar import PDBeMolstar


# ============================================================
# 🎯 YOUR VIEWER FUNCTION (clean version used in pipeline)
# ============================================================
def show_mda_molstar_frame(universe, frame, selection="all"):
    """
    Show MDAnalysis frame in Mol* without PDBWriter (no warnings).
    """

    universe.trajectory[frame]
    atoms = universe.select_atoms(selection)

    coords = atoms.positions
    resnames = atoms.resnames
    resids = atoms.resids
    names = atoms.names
    segids = atoms.segids

    pdb_lines = []

    for i, (coord, resn, resid, name, seg) in enumerate(
        zip(coords, resnames, resids, names, segids), start=1
    ):
        x, y, z = coord

        pdb_lines.append(
            "ATOM  {i:5d} {name:^4s} {resn:>3s} {chain:1s}"
            "{resid:4d}    "
            "{x:8.3f}{y:8.3f}{z:8.3f}"
            "  1.00  0.00           {elem:>2s}".format(
                i=i,
                name=name[:4],
                resn=resn,
                chain=seg if len(seg) == 1 else "A",
                resid=resid,
                x=x,
                y=y,
                z=z,
                elem=name[0],
            )
        )

    pdb_string = "\n".join(pdb_lines).encode("utf-8")

    view = PDBeMolstar(
        custom_data={
            "data": pdb_string,
            "format": "pdb",
            "binary": False,
        },
        hide_controls_icon=True,
        hide_expand_icon=True,
        hide_settings_icon=True,
        hide_selection_icon=True,
        hide_animation_icon=True,
        hide_water=True,
        hide_carbs=True,
    )

    return view


# ============================================================
# 🧪 FAKE MD SYSTEM (NO FILES)
# ============================================================

n_atoms = 50
n_frames = 5

# fake trajectory: (frames, atoms, xyz)
coords = np.random.rand(n_frames, n_atoms, 3) * 10


# ============================================================
# 🧬 BUILD FAKE TOPOLOGY
# ============================================================

u = Universe.empty(
    n_atoms,
    n_residues=10,
    atom_resindex=np.repeat(np.arange(10), 5),
    trajectory=True
)

u.add_TopologyAttr("name", ["CA"] * n_atoms)
u.add_TopologyAttr("resname", ["ALA"] * 10)
u.add_TopologyAttr("resid", np.repeat(np.arange(1, 11), 5))
u.add_TopologyAttr("segid", ["A"] * n_atoms)


# ============================================================
# 🎞️ ATTACH FAKE TRAJECTORY
# ============================================================

u.load_new(coords, format=MemoryReader)


# ============================================================
# 🔬 TEST VIEWER
# ============================================================

view = show_mda_molstar_frame(u, frame=2)
view
