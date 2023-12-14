import os
import unittest
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.observation.mock_observation import MockObservation

# from lnl_computer.plotting.gif_generator import

PLOT = False


def test_mcz_grid_lnl(test_datapath, tmp_path):
    # fails as testfile is too small --> not enough DCOs
    grd = McZGrid.from_compas_output(
        test_datapath,
        cosmological_parameters=dict(
            aSF=0.01, dSF=4.70, mu_z=-0.23, sigma_z=0.0
        ),
        n_bootstrapped_matrices=2,
        chirp_mass_bins=np.linspace(3, 40, 50),
        redshift_bins=np.linspace(0, 0.6, 100),
    )
    tmp_fn = os.path.join(tmp_path, "mcz_grid.h5")
    grd.save(fname=tmp_fn)
    new_uni = McZGrid.from_h5(tmp_fn)
    assert np.allclose(grd.chirp_mass_bins, new_uni.chirp_mass_bins)
    grd.plot().savefig(os.path.join(tmp_path, "mcz_grid.png"))

    obs = MockObservation.from_mcz_grid(grd)
    obs.save(os.path.join(tmp_path, "mock_obs.npz"))
    obs = MockObservation.from_npz(os.path.join(tmp_path, "mock_obs.npz"))
    obs.plot().savefig(os.path.join(tmp_path, "mock_obs.png"))

    lnl, unc = grd.get_lnl(obs.mcz)
    assert lnl > -np.inf
    assert unc != np.nan

