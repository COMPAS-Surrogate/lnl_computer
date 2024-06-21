import glob
import os

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.mock_data import MockData, generate_mock_data

HERE = os.path.dirname(__file__)
TEST_DIR = os.path.join(HERE, "test_data")


@pytest.fixture
def mock_data() -> MockData:
    np.random.seed(42)
    return generate_mock_data(outdir=TEST_DIR, duration=1)


@pytest.fixture
def tmp_path() -> str:
    """Temporary directory."""
    pth = os.path.join(HERE, "out_tmp")
    os.makedirs(pth, exist_ok=True)
    return pth


@pytest.fixture
def monkeypatched_mcz_grid(mock_data: MockData, monkeypatch) -> McZGrid:
    model = mock_data.mcz_grid
    norm2d = multivariate_normal(
        [0.2, 20], np.array([[1.0, 0.0], [0.0, 20.0]])
    )
    z_bins, mc_bins = model.redshift_bins, model.chirp_mass_bins
    Z, MC = np.meshgrid(z_bins, mc_bins)
    model.rate_matrix = norm2d.pdf(np.stack([Z, MC], axis=-1))
    return model
