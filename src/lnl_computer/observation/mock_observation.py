import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..cosmic_integration.mcz_grid import McZGrid
from ..cosmic_integration.star_formation_paramters import DEFAULT_DICT
from ..logger import logger
from .observation import Observation


class MockObservation(Observation):
    def __init__(
        self,
        rate2d: np.ndarray,
        mcz: np.ndarray,
        mcz_grid: McZGrid,
        duration: float,
    ):
        """
        A mock observation of a population of BBHs.

        :param rate2d: The detection rate matrix
        :param mcz: The Mc-Z pairs of the mock population (shape (n_events, 2)), where each row is (mc, z)
        :param mcz_grid: The mcz_grid that the mock population was sampled from
        """
        self.rate2d = rate2d
        self.mcz = mcz
        self.mcz_grid = mcz_grid
        self.outdir = mcz_grid.outdir
        self.duration = duration

    @classmethod
    def from_mcz_grid(
        cls,
        mcz_grid: McZGrid,
        duration: float,
        n_obs: int = None,
    ) -> "MockObservation":
        """Make a fake detection matrix with the same shape as the mcz_grid"""

        rate2d = np.zeros(mcz_grid.rate_matrix.shape)
        event_mcz = _sample_events_from_mcz_grid(
            mcz_grid, duration=duration, n_obs=n_obs
        )
        for mc, z in event_mcz:
            mc_bin, z_bin = mcz_grid.get_matrix_bin_idx(mc, z)
            rate2d[mc_bin, z_bin] += 1

        return MockObservation(
            rate2d=rate2d, mcz=event_mcz, mcz_grid=mcz_grid, duration=duration
        )

    @classmethod
    def from_compas_h5(cls, compas_h5_fname: str, duration: float, **kwargs):
        kwargs["cosmological_parameters"] = kwargs.get(
            "cosmological_parameters", DEFAULT_DICT
        )
        mcz_grid = McZGrid.from_compas_output(compas_h5_fname, **kwargs)
        mcz_grid.bin_data()
        return cls.from_mcz_grid(mcz_grid, duration=duration)

    def plot(self, fname=None) -> plt.Figure:
        fig = self.mcz_grid.plot()
        axes = fig.get_axes()
        axes[0].scatter(
            self.mcz[:, 1],
            self.mcz[:, 0],
            s=15,
            c="dodgerblue",
            marker="*",
            alpha=0.95,
        )
        fig.suptitle(f"Mock population ({self.n_events} blue stars)")
        axes[1].set_title(self.mcz_grid.param_str, fontsize=7)

        if fname:
            fig.savefig(fname)

        return fig

    def __repr__(self) -> str:
        return f"<MockObservation({self.n_events} events, {self.duration}yrs)>"

    @property
    def label(self) -> str:
        return f"mock_observation_{self.mcz_grid.label}"

    def __dict__(self):
        mcz_grid_data = self.mcz_grid.__dict__()
        mcz_grid_data["bootstrapped_rate_matrices"] = None
        return {
            "rate2d": self.rate2d,
            "mcz": self.mcz,
            "duration": self.duration,
            **mcz_grid_data,
        }

    @classmethod
    def from_npz(cls, fname: str) -> "MockObservation":
        data = dict(np.load(fname, allow_pickle=True))
        return cls(
            rate2d=data["rate2d"],
            mcz=data["mcz"],
            mcz_grid=McZGrid.from_dict(data),
            duration=data["duration"],
        )


def _sample_events_from_mcz_grid(
    mcz_grid: McZGrid,
    duration: float,
    n_obs: float = None,
) -> np.ndarray:
    """
    Sample Mc-Z pairs from the mcz_grid.

    Sample using the detection rate as weights (i.e. sample more from higher detection rate regions).
    #TODO: implement sample_using_emcee (sample from the detection rate distribution using poisson distributions)
    # FIXME: draw from the detection rate distribution using poisson distributions

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
