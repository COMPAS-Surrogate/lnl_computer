"""Mocking utilities for testing (mocks COMPAS populations, and MCZ obserations)."""
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
