import pandas as pd
from prolif import Fingerprint


def retrieve_prolif_frame_residue(
    u,
    run_id,
    mutation,
    ligand_sel,
    stride=20,
    debug=False,
):
    """
    Extract residue-level protein-ligand interactions per frame
    directly from MDAnalysis trajectory using ProLIF.
    """

    protein = u.select_atoms("protein")
    ligand = u.select_atoms(ligand_sel)

    if len(ligand) == 0:
        print("❌ No ligand found")
        return pd.DataFrame()

    #  correct interaction names (from your environment)
    fp = Fingerprint(
        [
            "HBDonor",
            "HBAcceptor",
            "Hydrophobic",
            "PiStacking",
            "PiCation",
            "CationPi",
            "Anionic",
            "Cationic",
            "EdgeToFace",
            "FaceToFace",
            "MetalAcceptor",
            "MetalDonor",
        ]
    )

    rows = []

    fp.run(u.trajectory[:stride], ligand, protein)
    df = fp.to_dataframe()

    return df
