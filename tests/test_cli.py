import os
import shutil

import pandas as pd
import pytest
from click.testing import CliRunner

from lnl_computer.cli.cli import (
    cli_batch_lnl_generation,
    cli_combine_lnl_data,
    cli_make_mock_obs,
    cli_make_sf_table,
    make_mock_obs,
    make_sf_table,
)
from lnl_computer.mock_data import MockData
from lnl_computer.observation.mock_observation import MockObservation


def test_cli_make_sf_table(tmp_path):
    fname = f"{tmp_path}/parameter_table.csv"
    runner = CliRunner()
    n = 2**5
    params = ["aSF", "dSF"]
    out = runner.invoke(
        cli_make_sf_table,
        ["-p", params[0], "-p", params[1], "-n", n, "-f", fname],
    )
    print(out.stdout)
    assert out.exit_code == 0
    df = pd.read_csv(fname)
    assert len(df) == n
    assert set(df.columns) == set(params)


def test_cli_make_mock_obs(mock_data: MockData, tmp_path):
    fname = f"{tmp_path}/mock_obs.npz"
    runner = CliRunner()
    asf = 0.01
    out = runner.invoke(
        cli_make_mock_obs,
        [
            mock_data.compas_filename,
            "--sf_sample",
            f"aSF:{asf}",
            "--fname",
            fname,
        ],
    )
    print(out.stdout)
    assert out.exit_code == 0
    obs = MockObservation.from_npz(fname)
    assert obs.mcz_grid.cosmological_parameters["aSF"] == asf


def test_cli_batch_lnl_generation(mock_data, tmp_path):
    # STEP 1: make a parameter table
    tmp_path = f"{tmp_path}/lnl_generation"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path, exist_ok=True)
    sf_fname = f"{tmp_path}/parameter_table.csv"
    make_sf_table(parameters=["aSF", "dSF"], n=2, fname=sf_fname)
    sf_parm = pd.read_csv(sf_fname).to_dict("records")[0]

    # STEP 2: generate mock observations
    mock_fname = f"{tmp_path}/mock_obs.npz"
    make_mock_obs(
        mock_data.compas_filename, sf_parm, fname=mock_fname, duration=1
    )

    # STEP 3: generate lnl data
    runner = CliRunner()
    out = runner.invoke(
        cli_batch_lnl_generation,
        [
            mock_fname,
            mock_data.compas_filename,
            sf_fname,
            "--n_bootstraps",
            1,
            "--no_plots",
            "--outdir",
            tmp_path,
        ],
    )
    assert out.exit_code == 0

    # STEP 4: combine lnl data
    comb_fname = f"{tmp_path}/combined_lnl_data.csv"
    out = runner.invoke(
        cli_combine_lnl_data, [tmp_path, "--fname", comb_fname]
    )
    data = pd.read_csv(comb_fname)
    assert len(data) == 2
    assert out.exit_code == 0
