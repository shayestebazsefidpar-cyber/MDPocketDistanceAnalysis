import tempfile

import MDAnalysis as mda
from plip.exchange.report import BindingSiteReport
from plip.structure.preparation import PDBComplex


def retrieve_plip_interactions(
    u,
    run_id,
    mutation,
    ligand_sel,
    stride=10,
):

    results = []

    for ts in u.trajectory[::stride]:
        protein_atoms = u.select_atoms("protein")
        ligand_atoms = u.select_atoms(ligand_sel)

        atoms = protein_atoms + ligand_atoms

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            atoms.write(tmp.name)
            pdb_path = tmp.name

        protlig = PDBComplex()
        protlig.load_pdb(pdb_path)

        for ligand in protlig.ligands:
            protlig.characterize_complex(ligand)

        for key, site in protlig.interaction_sets.items():
            binding_site = BindingSiteReport(site)

            for t in [
                "hydrophobic",
                "hbond",
                "saltbridge",
                "pistacking",
                "pication",
                "halogen",
                "metal",
                "waterbridge",
            ]:
                features = getattr(binding_site, f"{t}_features", None)

                if not features:
                    continue

                if ":" in key:
                    resname, chain, resid = key.split(":")
                else:
                    resname, chain, resid = key, None, None

                site_type = "ligand" if resname in ligand_sel else "protein"

                results.append(
                    {
                        "run_id": run_id,
                        "mutation": mutation,
                        "frame": ts.frame,
                        "ligand_site": key,
                        "resname": resname,
                        "chain": chain,
                        "resid": resid,
                        "site_type": site_type,
                        "interaction_type": t,
                    }
                )

    return results
