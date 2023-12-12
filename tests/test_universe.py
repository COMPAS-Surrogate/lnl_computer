import os
import unittest
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from lnl_computer.cosmic_integration.universe import Universe

# from lnl_computer.plotting.gif_generator import

PLOT = False


def test_universe(test_datapath, tmp_path):
    # fails as testfile is too small --> not enough DCOs
    uni = Universe.from_compas_output(
        test_datapath,
        cosmological_parameters=dict(aSF=0.01, dSF=4.70, mu_z=-0.23, sigma_z=0.)
    )
    tmp_fn = os.path.join(tmp_path, "universe.h5")
    uni.save(fname=tmp_fn)
    new_uni = Universe.from_h5(tmp_fn)
    assert np.allclose(uni.chirp_mass_bins, new_uni.chirp_mass_bins)
    mock_uni = uni.sample_observations(n_obs=1000)
    if PLOT:
        uni.plot().savefig(os.path.join(tmp_path, "universe.png"))
        mock_uni.plot().savefig(os.path.join(tmp_path, "mock_universe.png"))

