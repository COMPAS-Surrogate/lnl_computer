import glob
import os

import pytest
from compas_python_utils.cosmic_integration.binned_cosmic_integrator.bbh_population import (
    generate_mock_bbh_population_file,
)

HERE = os.path.dirname(__file__)
TEST_DIR = os.path.join(HERE, "test_data")
TEST_FILE = os.path.join(TEST_DIR, "mock_COMPAS_output.h5")


@pytest.fixture
def test_datapath():
    """Test data."""
    if os.path.exists(TEST_FILE):
        return TEST_FILE

    os.makedirs(TEST_DIR, exist_ok=True)
    generate_mock_bbh_population_file(filename=TEST_FILE)
    return TEST_FILE


@pytest.fixture
def tmp_path():
    """Temporary directory."""
    pth = os.path.join(HERE, "out_tmp")
    os.makedirs(pth, exist_ok=True)
    return pth