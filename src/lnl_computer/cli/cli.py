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
    default=["aSF", "dSF", "mu_z", "sigma_z"],
)
@click.option(
    "--n",
    "-n",
    type=int,
    default=50,
)
@click.option(
    "--grid_parameterspace",
    "-g",
    type=bool,
    is_flag=True,
)
@click.option(
    "--fname",
    "-f",
    type=str,
    default="parameter_table.csv",
)
def cli_make_sf_table(
    parameters,
    n,
    grid_parameterspace,
    fname,
) -> None:
    """Parses the table of parameters to generate mcz-grids for.

    :param parameters: list of parameters to generate mcz-grids for
    :param parameter_table: path to pandas dataframe csv containing cosmic integration parameters (or dataframe itself)
    :param n: number of samples to generate
    :param custom_ranges: custom ranges for parameters
    :param grid_parameterspace: whether to grid the parameter space
    :param outdir: output directory for mcz-grids
    :return: pandas dataframe containing cosmic integration parameters
    """
    make_sf_table(
        parameters=parameters,
        n=n,
        grid_parameterspace=grid_parameterspace,
        fname=fname,
    )


@click.command(name="make_mock_obs")
@click.argument("compas_h5_path", type=str)
@click.argument(
    "sf_sample", type=str, default="aSF:0.01 dSF:0.01 muz:0.01 sigma0:0.01"
)
@click.option(
    "--fname", type=str, help="Output filename", default="mock_observation.npz"
)
def cli_make_mock_obs(
    compas_h5_path: str,
    sf_sample: Union[Dict, str],
    fname: str = "mock_observation.npz",
) -> "MockObservation":
    """Generate a detection matrix for a given set of star formation parameters
    :param compas_h5_path:
    :param sf_sample: Dict of star formation parameters, or a string like "k1:v1 k2:v2"->{"k1":float(v1)", "k2":float(v2)}
    :param fname: mcgrid-fname
    """
    return make_mock_obs(compas_h5_path, sf_sample, fname=fname)


@click.command(name="batch_lnl_generation")
@click.argument(
    "mcz_obs", type=click.Path(exists=True)
)  # Assuming the path to the npz file
@click.argument("compas_h5_path", type=str)
@click.argument(
    "parameter_table", type=click.Path(exists=True, dir_okay=False)
)  # Assuming it's a CSV file
@click.option(
    "--n_bootstraps",
    default=100,
    help="Number of bootstraps to generate for each parameter set",
    type=int,
)
@click.option(
    "--save_images/--no_save_images",
    default=True,
    help="Save images of the generated mcz-grids",
    is_flag=True,
)
@click.option(
    "--outdir",
    default="out_mcz_grids",
    help="Output directory for mcz-grids",
    type=str,
)
def cli_batch_lnl_generation(
    mcz_obs: str,
    compas_h5_path: str,
    parameter_table: Union[pd.DataFrame, str],
    n_bootstraps: int = 100,
    save_images: bool = True,
    outdir: str = "out_mcz_grids",
) -> None:
    """
    Generate a set of COMPAS Mc-Z detection rate matrices
    :param compas_h5_path: path to COMPAS h5 file
    :param parameter_table: path to pandas dataframe containing cosmic integration parameters
    :param n_bootstraps: number of bootstraps to generate for each parameter set
    :param save_images: save images of the generated mcz-grids
    :param outdir: output directory for mcz-grids
    :return: None
    """
    batch_lnl_generation(
        mcz_obs,
        compas_h5_path,
        parameter_table,
        n_bootstraps,
        save_images,
        outdir,
    )


@click.command("combine_lnl_data")
@click.argument("outdir", default="out_mcz_grids", type=str)
@click.option("--fname", default="", type=str)
def cli_combine_lnl_data(
    outdir: str = "out_mcz_grids",
    fname: str = "",
) -> None:
    """
    Combine the likelihood data in 'OUTDIR/*_lnl.csv' -> FNAME

    combine_lnl_data OUDIR --fname FNAME
    """
    combine_lnl_data(outdir=outdir, fname=fname)


@click.command("mock_compas_output")
@click.option("--fname", type=str, default="mock_compas_output.h5")
def cli_make_mock_compas_output(fname: str):
    """Generate a mock COMPAS output file at FNAME"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    generate_mock_bbh_population_file(fname=fname)
    click.echo(f"Mock COMPAS output saved to {fname}")
