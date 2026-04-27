import tempfile
import warnings

import numpy as np
from ipymolstar import PDBeMolstar
from MDAnalysis import MissingDataWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=MissingDataWarning)


def show_mda_frame(universe, frame, selection="all"):
    """
    MDAnalysis → temporary PDB → Mol* viewer (robust Jupyter-safe version).
    """

    universe.trajectory[frame]
    atoms = universe.select_atoms(selection)

    # write temp PDB
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        atoms.write(tmp.name)
        tmp_path = tmp.name

    # IMPORTANT: read file content instead of relying on URL
    with open(tmp_path, "r") as f:
        pdb_string = f.read()

    return PDBeMolstar(
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
