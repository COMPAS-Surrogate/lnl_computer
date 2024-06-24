from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..cosmic_integration.mcz_grid import McZGrid
from ..cosmic_integration.star_formation_paramters import DEFAULT_DICT
from ..logger import logger
from .observation import Observation


class MockObservation(Observation):
    @classmethod
    def from_mcz_grid(cls, mcz_grid: McZGrid, duration=None, n_obs=None):
        w = generate_mock_obs_weights(mcz_grid, duration, n_obs)
        n_obs, n_mc_bins, n_z_bins = w.shape
        return cls(
            w,
            mcz_grid.chirp_mass_bins,
            mcz_grid.redshift_bins,
            label=f"MockObs(n={n_obs}, bins=[{n_mc_bins}, {n_z_bins}]",
            cosmological_parameters=mcz_grid.cosmological_parameters,
        )

    def from_compas_h5(
        cls,
        compas_h5_fname,
        duration=None,
        n_obs=None,
        cosmological_parameters=DEFAULT_DICT,
    ):
        mcz_grid = McZGrid.from_compas_output(
            compas_h5_fname, cosmological_parameters
        )
        mcz_grid.bin_data()
        return cls.from_compas_h5(mcz_grid, duration, n_obs)


def generate_mock_obs_weights(
    mcz_grid: McZGrid, duration: float, n_obs: int = None
):
    mcz_samples = _sample_events_from_mcz_grid(
        mcz_grid, duration=duration, n_obs=n_obs
    )
    w = np.zeros((len(mcz_samples), *mcz_grid.rate_matrix.shape))
    for i, (mc, z) in enumerate(mcz_samples):
        mc_bin, z_bin = mcz_grid.get_matrix_bin_idx(mc, z)
        w[i, mc_bin, z_bin] += 1
    return w


def _sample_events_from_mcz_grid(
    mcz_grid: McZGrid,
    duration: float,
    n_obs: float = None,
) -> np.ndarray:
    """
    Sample Mc-Z pairs from the mcz_grid.

    Sample using the detection rate as weights (i.e. sample more from higher detection rate regions).
    #TODO: implement sample_using_emcee (sample from the detection rate distribution using poisson distributions)
    #TODO: draw from the detection rate distribution using poisson distributions

    :param mcz_grid: The mcz_grid to sample from
    :param duration: The duration of the observation (in years), to convert rate->number of events
    :param n_obs: The number of observations to sample (could be a float, _will_not_ be rounded to an int)

    :return: np.ndarray of shape (n_obs, 2) where each row is (mc, z)
    """
    n_obs = mcz_grid.n_detections(duration) if n_obs is None else n_obs
    logger.info(
        f"Sampling {n_obs:.1f} events ({duration}yrs) from mcz_grid[{mcz_grid}]"
    )
    df = _mcz_to_df(mcz_grid)
    if np.sum(df.rate) > 0:
        n_events = df.sample(
            weights=df.rate, n=int(n_obs), random_state=0, replace=True
        )
    else:
        n_events = df.sample(n=int(n_obs), random_state=0)

    return n_events[["mc", "z"]].values


def _mcz_to_df(mcz_grid) -> pd.DataFrame:
    """The mcz_grid as a pandas dataframe with columns (mc, z, rate), sorted by rate (high to low)"""

    z, mc = mcz_grid.redshift_bins, mcz_grid.chirp_mass_bins
    rate = mcz_grid.rate_matrix.ravel()
    zz, mcc = np.meshgrid(z, mc)
    df = pd.DataFrame({"z": zz.ravel(), "mc": mcc.ravel(), "rate": rate})
    df = df.sort_values("rate", ascending=False)

    # drop nans and log the number of rows dropped
    n_nans = df.isna().any(axis=1).sum()
    if n_nans > 0:
        logger.warning(f"Dropping {n_nans}/{len(df)} rows with nan values")
        df = df.dropna()

    # check no nan in dataframe
    if df.isna().any().any():
        logger.error("Nan values in dataframe")

    return df
