def make_fake_universe():

    u = MagicMock()

    # fake trajectory
    u.trajectory = [0, 1, 2, 3, 4]

    # fake atom selection
    ligand = MagicMock()
    ligand.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])

    pocket = MagicMock()

    # simulate fluctuating pocket size (open/close)
    pocket_sizes = [50, 80, 120, 60, 30]
    pocket_iter = iter(pocket_sizes)

    def fake_select_atoms(*args, **kwargs):
        m = MagicMock()
        m.__len__.return_value = next(pocket_iter)
        return m

    u.select_atoms = fake_select_atoms

    return u
