import os

import numpy as np
import pytest

from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.mock_data import MockData
from lnl_computer.observation.lvk_observation import LVKObservation
from lnl_computer.observation.mock_observation import MockObservation

PLOT = False


def test_lnl(mock_data: MockData):
    lnl, unc = mock_data.mcz_grid.get_lnl(mock_data.observations)
    assert lnl > -np.inf
    assert unc != np.nan

    # NO-BOOTSTRAPS LNL
    lnl, unc = McZGrid.lnl(
        mcz_obs=mock_data.observations,
        compas_h5_path=mock_data.compas_filename,
        sf_sample=dict(aSF=0.01, dSF=4.70, mu_z=-0.23),
        n_bootstraps=0,
    )
    assert lnl > -np.inf
    assert np.isnan(unc)
    expected_lnl = 10647.1
    np.testing.assert_allclose(
        lnl,
        expected_lnl,
        atol=200,
        err_msg=f"lnl={lnl:.1f} not close to {expected_lnl}",
    )


@pytest.mark.skip(reason="Takes too long")
def test_lnl_nan(mock_data: MockData, tmp_path: str):
    # ensure not getting a nan!
    lnl, unc = McZGrid.lnl(
        mcz_obs=mock_data.observations,
        compas_h5_path=mock_data.compas_filename,
        sf_sample=dict(aSF=0.01, dSF=4.70, mu_z=-0.01, sigma_0=0.0),
        n_bootstraps=0,
        save_plots=True,
        outdir=f"{tmp_path}/nan_lnl",
    )
    assert not np.isnan(lnl)
    assert np.isnan(unc)


def test_duration(mock_data: MockData):
    # ensure duration is used
    mock_d1 = MockObservation.from_mcz_grid(mock_data.mcz_grid, duration=1)
    mock_d2 = MockObservation.from_mcz_grid(mock_data.mcz_grid, duration=2)
    assert mock_d1.n_events < mock_d2.n_events


def test_ogc4_lnl(monkeypatched_mcz_grid: McZGrid, tmp_path: str):
    obs = LVKObservation.from_ogc4_data()
    (
        lnl,
        _,
    ) = monkeypatched_mcz_grid.get_lnl(mcz_obs=obs)
    assert lnl > -np.inf
