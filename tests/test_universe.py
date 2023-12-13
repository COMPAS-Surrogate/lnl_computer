import os
import unittest
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from lnl_computer.cosmic_integration.mcz_grid import McZGrid

# from lnl_computer.plotting.gif_generator import

PLOT = False


def test_mcz_grid(test_datapath, tmp_path):
    # fails as testfile is too small --> not enough DCOs
    uni = McZGrid.from_compas_output(
        test_datapath,
        cosmological_parameters=dict(aSF=0.01, dSF=4.70, mu_z=-0.23, sigma_z=0.)
    )
    tmp_fn = os.path.join(tmp_path, "mcz_grid.h5")
    uni.save(fname=tmp_fn)
    new_uni = McZGrid.from_h5(tmp_fn)
    assert np.allclose(uni.chirp_mass_bins, new_uni.chirp_mass_bins)
    mock_uni = uni.sample_observations(n_obs=1000)
    if PLOT:
        uni.plot().savefig(os.path.join(tmp_path, "mcz_grid.png"))
        mock_uni.plot().savefig(os.path.join(tmp_path, "mock_mcz_grid.png"))

#
#
# if __name__ == "__main__":
#     PATH = "/Users/avaj0001/Documents/projects/compas_dev/quasir_compass_blocks/data/COMPAS_Output.h5"
#     clean = True
#     outdir = "out"
#     for i in range(10):
#         np.random.seed(i)
#         uni_file = f"{outdir}/v{i}.h5"
#         uni = McZGrid.from_compas_output(
#             PATH,
#             n_bootstrapped_matrices=10,
#             outdir=outdir,
#             redshift_bins=np.linspace(0, 0.6, 100),
#             chirp_mass_bins=np.linspace(3, 40, 50),
#         )
#         uni.save(fname=uni_file)
#         uni = McZGrid.from_h5(uni_file)
#
#     uni = McZGrid.from_h5(f"{outdir}/v0.h5")
#     mock_pop = MockObservation.sample_possible_event_matrix(uni)
#     mock_pop.save(f"{outdir}/mock_pop.npz")
#     mock_pop = MockObservation.from_npz(f"{outdir}/mock_pop.npz")
#     fig = mock_pop.plot()
#     fig.savefig(f"{outdir}/mock_pop.png")
#
#     # uni_binned = mcz_grid.from_npz(uni_file)
#     # uni_binned.plot_detection_rate_matrix(outdir=outdir)
#     # mock_pop = uni_binned.sample_possible_event_matrix()
#     # mock_pop.plot(outdir=outdir)
