import numpy as np
import pandas as pd

from mdpocketclustering.core.trajectory import Trajectory


class SingleTrajectoryRunner:
    def __init__(self, run):
        self.run = run
        self.traj = Trajectory(run).load()

    def run_metric(self, metric, stride=1):
        values = metric.compute(self.traj, self.run.system, stride=stride)

        return pd.DataFrame(
            {"time": np.arange(len(values)) * stride, metric.name: values}
        )

        return pd.DataFrame({"time": np.arange(len(values)), metric.name: values})
