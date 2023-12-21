import glob
import os

import pytest

from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.mock_data import MockData, generate_mock_data

HERE = os.path.dirname(__file__)
TEST_DIR = os.path.join(HERE, "test_data")


@pytest.fixture
def mock_data() -> MockData:
    return generate_mock_data(outdir=TEST_DIR)


@pytest.fixture
def tmp_path():
    """Temporary directory."""
    pth = os.path.join(HERE, "out_tmp")
    os.makedirs(pth, exist_ok=True)
    return pth
