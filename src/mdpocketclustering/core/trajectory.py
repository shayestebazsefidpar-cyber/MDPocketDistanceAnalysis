import MDAnalysis as mda


class Trajectory:
    def __init__(self, run):
        self.run = run
        self.u = None

    def load(self):
        self.u = mda.Universe(
            str(self.run.files.topology), str(self.run.files.trajectory)
        )
        return self

    def select(self, sel):
        return self.u.select_atoms(sel)
