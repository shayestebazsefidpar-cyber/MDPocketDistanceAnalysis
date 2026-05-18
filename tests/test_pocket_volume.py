import warnings

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis import MissingDataWarning

warnings.filterwarnings("ignore", category=MissingDataWarning)

from mdpocketclustering.pocket_volume import pocket_volume_analysis


def main():
    topology = "topology.pdb"
    trajectory = "trajectory.xtc"

    u = mda.Universe(topology, trajectory)

    pocket_sel = "resname AP1 or resname MG1"

    stride = 50

    volumes = pocket_volume_analysis(
        u=u,
        run_id="test_run",
        mutation="WT",
        pocket_sel=pocket_sel,
        protein_sel="protein",
        spacing=0.4,
        buffer=4.0,
        probe_radius=1.4,
        stride=stride,
        debug=False,
    )

    print("\n✅ Number of frames analyzed:", len(volumes))

    # ---- add time (important!)
    time_ps = np.arange(len(volumes)) * stride * u.trajectory.dt

    df = pd.DataFrame(
        {"frame": np.arange(len(volumes)), "time_ps": time_ps, "volume_A3": volumes}
    )

    print("\n📊 DataFrame head:")
    print(df.head())

    print("\n📈 Stats:")
    print(df.describe())

    # ---- save output (very useful)
    df.to_csv("pocket_volume.csv", index=False)
    print("\n💾 Saved to pocket_volume.csv")


if __name__ == "__main__":
    main()
