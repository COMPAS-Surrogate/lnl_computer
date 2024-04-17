import numpy as np

from lnl_computer.mock_data import (
    McZGrid,
    MockData,
    MockObservation,
    generate_mock_data,
)


def test_reproducible_dataset(tmpdir):
    mock_datapaths = [f"{tmpdir}/test_data{i}" for i in range(2)]
    kwgs = dict(duration=1, sf_params=dict(aSF=0.01, dSF=4.70, mu_z=-0.23))
    data = []
    for path in mock_datapaths:
        np.random.seed(42)
        data.append(generate_mock_data(outdir=path, **kwgs))

    # ensure the McZ grid is the same
    grid0 = McZGrid.from_h5(data[0].mcz_grid_filename)
    grid1 = McZGrid.from_h5(data[1].mcz_grid_filename)
    assert np.allclose(grid0.rate_matrix, grid1.rate_matrix)

    # ensure the observations are the same
    obs0 = MockObservation.from_npz(data[0].observations_filename)
    obs1 = MockObservation.from_npz(data[1].observations_filename)
    assert np.allclose(obs0.mcz, obs1.mcz)

    # ensure the truth is the same
    truth0 = data[0].truth
    truth1 = data[1].truth
    assert truth0 == truth1
