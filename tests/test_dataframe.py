from mdpocketclustering.dataframe import build_dataframe


def test_build_dataframe():
    class FakeSim:
        def __init__(self):
            self.mutation = "WT"
            self.replicate = 1
            self.has_atp = True
            self.has_mg = False
            self.binding_energy = -10

    df = build_dataframe([FakeSim()])

    assert "binding_energy" in df.columns
    assert len(df) == 1
