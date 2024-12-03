import json

import numpy as np

from lnl_computer.cli.main import make_mock_obs
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
    obs0 = MockObservation.load(data[0].observations_filename)
    obs1 = MockObservation.load(data[1].observations_filename)
    assert np.allclose(obs0.weights, obs1.weights)

    # ensure the reference_param is the same
    truth0 = data[0].truth
    truth1 = data[1].truth
    assert truth0 == truth1


def test_mock_obs_lnl(tmpdir, mock_data: MockData):
    cosmo_params = dict(aSF=0.01, dSF=4.70, mu_z=-0.23)
    obs_fname = f"{tmpdir}/mock_obs.npz"
    make_mock_obs(
        compas_h5_path=mock_data.compas_filename,
        sf_sample=cosmo_params,
        duration=1,
        fname=obs_fname,
    )
    truths_fn = f"{tmpdir}/reference_param.json"
    # load the reference_param
    with open(truths_fn, "r") as f:
        truth = json.load(f)
    assert isinstance(
        truth["lnl"], float
    ), f"truth['lnl'] not a float: {truth['lnl']}"

    # load the observation
    obs = MockObservation.load(obs_fname)
    # assert dict compare
    assert obs.cosmological_parameters["aSF"] == cosmo_params["aSF"]
    assert obs.cosmological_parameters["dSF"] == cosmo_params["dSF"]
