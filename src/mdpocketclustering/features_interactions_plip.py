import tempfile

import pandas as pd
from plip.exchange.report import BindingSiteReport
from plip.structure.preparation import PDBComplex


def retrieve_plip_interactions(
    u, run_id, mutation, ligand_sel, frame_id=0, debug=False
):
    """Extract PLIP protein-ligand interactions from a single frame."""

    protein = u.select_atoms("protein")
    ligand = u.select_atoms(ligand_sel)

    if len(ligand) == 0:
        print("❌ No ligand found")
        return pd.DataFrame()

    atoms = protein + ligand

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        atoms.write(tmp.name)
        pdb_path = tmp.name

    if debug:
        print("PDB path:", pdb_path)

    mol = PDBComplex()
    mol.load_pdb(pdb_path)

    # Characterize and analyze all ligands
    for lig in mol.ligands:
        mol.characterize_complex(lig)
    mol.analyze()  # Required to populate interaction_sets [web:60][web:61]

    results = []

    interaction_types = [
        "hydrophobic",
        "hbond",
        "saltbridge",
        "pistacking",
        "pication",
        "halogen",
        "metal",
        "waterbridge",
    ]

    for site_id, site in mol.interaction_sets.items():
        report = BindingSiteReport(site)

        for itype in interaction_types:
            feats = getattr(report, f"{itype}_features", None)
            if feats and len(feats) > 0:
                results.append(
                    {
                        "run_id": run_id,
                        "mutation": mutation,
                        "frame": frame_id,
                        "ligand_site": site_id,
                        "interaction_type": itype,
                        "count": len(feats),
                    }
                )

    return pd.DataFrame(results)
