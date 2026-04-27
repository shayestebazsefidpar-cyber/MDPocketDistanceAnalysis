import numpy as np
from ipymolstar import PDBeMolstar


def show_mda_frame(universe, frame, selection="all"):
    """
    Zero-warning MDAnalysis → Mol* viewer (no PDBWriter used).
    """

    universe.trajectory[frame]
    atoms = universe.select_atoms(selection)

    coords = atoms.positions
    resnames = atoms.resnames
    resids = atoms.resids
    names = atoms.names
    chainids = atoms.segids

    pdb_lines = []

    for i, (coord, resn, resid, name, chain) in enumerate(
        zip(coords, resnames, resids, names, chainids), start=1
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
                chain=chain if len(chain) == 1 else "A",
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
