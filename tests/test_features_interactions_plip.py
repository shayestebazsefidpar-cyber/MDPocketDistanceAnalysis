import warnings

from MDAnalysis import MissingDataWarning

warnings.filterwarnings("ignore")
# import function from your module
from mdpocketclustering.features_interactions_plip import retrieve_plip_interactions


def main():
    topology = "topology.pdb"
    trajectory = "trajectory.xtc"

    u = mda.Universe(topology, trajectory)
    ligand_sel = "resname AP1 or resname MG1"

    results = retrieve_plip_interactions(
        u=u, run_id="test_run", mutation="WT", ligand_sel=ligand_sel, stride=50
    )

    print("\n✅ Number of interaction records:", len(results))

    if len(results) > 0:
        print("\n🔹 Sample result:")
        print(results[0])

    import pandas as pd

    df = pd.DataFrame(results)

    print("\n📊 DataFrame head:")
    print(df.head())


if __name__ == "__main__":
    main()
