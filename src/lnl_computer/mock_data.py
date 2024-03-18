"""Mocking utilities for testing (mocks COMPAS populations, and MCZ obserations)."""
import json
import os
from typing import Dict

from compas_python_utils.cosmic_integration.binned_cosmic_integrator.bbh_population import (
    generate_mock_bbh_population_file,
)

from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.observation.mock_observation import MockObservation


def generate_mock_data(outdir: str, sf_params: Dict[str, float] = None):
    """Generate mock datasets for testing."""
    return MockData.generate_mock_datasets(outdir=outdir, sf_params=sf_params)


def load_mock_data(outdir: str):
    """Load mock datasets for testing."""
    return MockData(outdir)


class MockData(object):
    """Mocking utilities."""

    def __init__(self, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

    @property
    def compas_filename(self):
        return os.path.join(self.outdir, "mock_COMPAS_output.h5")

    @property
    def observations_filename(self):
        return os.path.join(self.outdir, "mock_MZC_obs.npz")

    @property
    def mcz_grid_filename(self):
        return os.path.join(self.outdir, "mock_MZC_output.h5")

    @classmethod
    def generate_mock_datasets(
        cls, outdir: str, sf_params: Dict[str, float] = None
    ):
        self = cls(outdir)
        if not os.path.exists(self.compas_filename):
            generate_mock_bbh_population_file(filename=self.compas_filename)

        if not os.path.exists(self.mcz_grid_filename):
            McZGrid.generate_n_save(
                self.compas_filename,
                sf_sample=sf_params,
                fname=self.mcz_grid_filename,
            )

        if not os.path.exists(self.observations_filename):
            obs = MockObservation.from_mcz_grid(self.mcz_grid)
            obs.save(self.observations_filename)
        return self

    @property
    def mcz_grid(self) -> McZGrid:
        return McZGrid.from_h5(self.mcz_grid_filename)

    @property
    def observations(self) -> MockObservation:
        return MockObservation.from_npz(self.observations_filename)

    @property
    def truth(self) -> Dict:
        return _get_true_params(self)


def _get_true_params(mock_data: MockData):
    fn = f"{mock_data.outdir}/truth.json"
    if os.path.exists(fn):
        with open(fn, "r") as f:
            return json.load(f)
    else:
        grid = mock_data.mcz_grid
        cosmo_params = grid.cosmological_parameters
        true_params = dict(
            aSF=cosmo_params["aSF"],
            dSF=cosmo_params["dSF"],
            sigma0=cosmo_params["sigma0"],
            muz=cosmo_params["muz"],
        )
        lnl = (
            grid.lnl(
                mcz_obs=mock_data.observations.mcz,
                duration=1,
                compas_h5_path=mock_data.compas_filename,
                sf_sample=true_params,
                n_bootstraps=0,
            )[0]
            * -1
        )
        true_params["lnl"] = lnl
        with open(fn, "w") as f:
            json.dump(true_params, f)
        return true_params
