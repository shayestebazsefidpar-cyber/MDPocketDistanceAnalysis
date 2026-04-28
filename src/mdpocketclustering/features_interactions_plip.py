import tempfile

import pandas as pd
from plip.exchange.report import BindingSiteReport
from plip.structure.preparation import PDBComplex


def retrieve_plip_interactions(
    u, run_id, mutation, ligand_sel, frame_id=0, debug=False
):

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

    results = []

    for lig in mol.ligands:
        mol.characterize_complex(lig)

    for site_id, site in mol.interaction_sets.items():
        report = BindingSiteReport(site)

        for itype in [
            "hydrophobic",
            "hbond",
            "saltbridge",
            "pistacking",
            "pication",
            "halogen",
            "metal",
            "waterbridge",
        ]:
            feats = getattr(report, f"{itype}_features", None)

            if not feats:
                continue

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
