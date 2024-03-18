import json
import os
from typing import Dict, List, Union

import click
import pandas as pd
from compas_python_utils.cosmic_integration.binned_cosmic_integrator.bbh_population import (
    generate_mock_bbh_population_file,
)

from .main import (
    batch_lnl_generation,
    combine_lnl_data,
    make_mock_obs,
    make_sf_table,
)


@click.command("make_sf_table")
@click.option(
    "--parameters",
    "-p",
    type=str,
    multiple=True,
    default=["aSF", "dSF", "muz", "sigma0"],
    help="List of parameters to generate mcz-grids for",
)
@click.option(
    "--n",
    "-n",
    type=int,
    default=50,
    help="Number of samples to generate",
)
@click.option(
    "--grid_parameterspace",
    "-g",
    type=bool,
    is_flag=True,
    help="Whether to grid the parameter space",
)
@click.option(
    "--fname",
    "-f",
    type=str,
    default="parameter_table.csv",
    help="Output filename (must be a .csv)",
)
def cli_make_sf_table(
    parameters,
    n,
    grid_parameterspace,
    fname,
) -> None:
    """Parses the table of parameters to generate mcz-grids for."""
    make_sf_table(
        parameters=parameters,
        n=n,
        grid_parameterspace=grid_parameterspace,
        fname=fname,
    )


DEFAULT_SF_PARAMETERS = dict(
    aSF=0.01,
    bSF=2.77,
    cSF=2.9,
    dSF=4.7,
    muz=-0.23,
    sigma0=0.39,
)


@click.command(name="make_mock_obs")
@click.argument("compas_h5_path", type=str)
@click.option(
    "--sf_sample",
    type=str,
    show_default=True,
    default="aSF:0.01 dSF:4.7 muz:-0.23 sigma0:0.39",
    help="Star formation parameters",
)
@click.option(
    "--fname",
    type=str,
    default="mock_observation.npz",
    help="Output filename (must be a .npz)",
)
def cli_make_mock_obs(
    compas_h5_path: str,
    sf_sample: Union[Dict, str],
    fname: str = "mock_observation.npz",
) -> "MockObservation":
    """Generate a set of 'mock' observations for the sf-sample and compas output file (COMPAS_H5_PATH)."""
    return make_mock_obs(compas_h5_path, sf_sample, fname=fname)


@click.command(name="batch_lnl_generation")
@click.argument("mcz_obs", type=click.Path(exists=True))
@click.argument(
    "compas_h5_path",
    type=str,
)
@click.argument(
    "parameter_table", type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--n_bootstraps",
    default=100,
    help="Number of bootstraps to generate for each parameter set (used for error estimation)",
    type=int,
    show_default=True,
)
@click.option(
    "--plots/--no_plots",
    default=True,
    help="Save diagnostic plots for each parameter set",
    is_flag=True,
    show_default=True,
)
@click.option(
    "--outdir",
    default="out_mcz_grids",
    help="Outdir for mcz-grids",
    type=str,
    show_default=True,
)
def cli_batch_lnl_generation(
    mcz_obs: str,
    compas_h5_path: str,
    parameter_table: Union[pd.DataFrame, str],
    n_bootstraps: int = 100,
    plots: bool = True,
    outdir: str = "out_mcz_grids",
) -> None:
    """
    Given observations (MCZ_OBS, npz-file), COMPAS output (COMPAS_H5_PATH, h5 file),
    and a table of SF parameters (PARAMETER_TABLE, csv file), generate McZ grids and compute likelihoods.

    The likelihoods are saved to OUTDIR/*_lnl.csv
    """
    batch_lnl_generation(
        mcz_obs,
        compas_h5_path,
        parameter_table,
        n_bootstraps,
        plots,
        outdir,
    )


@click.command("combine_lnl_data")
@click.argument("outdir", default="out_mcz_grids", type=str)
@click.option(
    "--fname", default="", type=str, help="Output filename (must be a .csv)"
)
def cli_combine_lnl_data(
    outdir: str = "out_mcz_grids",
    fname: str = "",
) -> None:
    """
    Combine the likelihood data in 'OUTDIR/*_lnl.csv' -> FNAME

    OUTDIR: Output directory with likelihood files (csvs)
    """
    combine_lnl_data(outdir=outdir, fname=fname)


@click.command("mock_compas_output")
@click.option(
    "--fname",
    type=str,
    default="mock_compas_output.h5",
    help="Output filename (must be a .h5)",
)
def cli_make_mock_compas_output(fname: str):
    """Generate a mock COMPAS output file at FNAME"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    generate_mock_bbh_population_file(fname=fname)
    click.echo(f"Mock COMPAS output saved to {fname}")
