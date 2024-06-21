import os
import uuid
from argparse import ArgumentParser
from functools import partial
from glob import glob
from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Union

import click
import corner
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from ..cosmic_integration.mcz_grid import McZGrid
from ..cosmic_integration.star_formation_paramters import (
    draw_star_formation_samples,
)
from ..logger import logger
from ..observation.mock_observation import MockObservation
from ..observation.observation import Observation


def make_sf_table(
    parameters: List[str] = None,
    n: int = 50,
    grid_parameterspace=False,
    fname: str = "parameter_table.csv",
) -> pd.DataFrame:
    """Parses the table of parameters to generate mcz-grids for.

    :param parameters: list of parameters to generate mcz-grids for
    :param parameter_table: path to pandas dataframe csv containing cosmic integration parameters (or dataframe itself)
    :param n: number of samples to generate
    :param custom_ranges: custom ranges for parameters
    :param grid_parameterspace: whether to grid the parameter space
    :param outdir: output directory for mcz-grids
    :return: pandas dataframe containing cosmic integration parameters
    """
    logger.info(f"Generating {n} samples of {parameters}")
    parameters = parameters or ["aSF", "dSF", "muz", "sigma0"]
    parameter_table = pd.DataFrame(
        draw_star_formation_samples(
            n,
            parameters=parameters,
            as_list=True,
            custom_ranges=None,
            grid=grid_parameterspace,
        )
    )
    # TODO: DOUBLE CHECK WITH JEFF -- Muz, sigma_z, sigma_0???

    parameter_table.to_csv(fname, index=False)
    fig = corner.corner(
        parameter_table.values,
        labels=parameters,
        truths=None,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=False,
        no_fill_contours=False,
        fill_contours=False,
    )
    fig.savefig(fname.replace(".csv", ".png"))
    logger.info(f"Parameter table saved to {fname}")


def make_mock_obs(
    compas_h5_path: str,
    sf_sample: Union[Dict, str],
    duration: float,
    fname: str = "mock_observation.npz",
    save_plots: bool = False,
) -> "MockObservation":
    """Generate a detection matrix for a given set of star formation parameters
    :param compas_h5_path:
    :param sf_sample: Dict of star formation parameters, or a string like "k1:v1 k2:v2"->{"k1":float(v1)", "k2":float(v2)}
    :param fname: mcgrid-fname
    """
    assert fname.endswith(".npz"), "fname must end with .npz"
    if isinstance(sf_sample, str):
        sf_sample = dict([s.split(":") for s in sf_sample.split(" ")])
        sf_sample = {k: float(v) for k, v in sf_sample.items()}

    mcz_grid = McZGrid.generate_n_save(
        compas_h5_path=compas_h5_path,
        sf_sample=sf_sample,
        duration=duration,
        save_plots=False,
    )
    obs = MockObservation.from_mcz_grid(mcz_grid, duration=duration)

    obs.save(fname)

    # save truth-json
    lnl = mcz_grid.get_lnl(duration=duration, mcz_obs=obs)
    truth_fname = os.path.dirname(fname) + "/truth.json"
    truth_data = dict(duration=duration, lnl=lnl[0], **sf_sample)
    logger.info(f"MockObs truth: {truth_data}")
    _write_json(data=truth_data, fname=truth_fname)
    logger.info(
        f"Mock observation saved to {fname} and truths to {truth_fname}"
    )


def _write_json(data, fname):
    import json

    with open(fname, "w") as f:
        json.dump(data, f)


def batch_lnl_generation(
    mcz_obs: Union[Observation, str],
    compas_h5_path: str,
    parameter_table: Union[pd.DataFrame, str],
    duration: float = 1,
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
    os.makedirs(outdir, exist_ok=True)
    if isinstance(mcz_obs, str):
        mcz_obs = MockObservation.from_npz(mcz_obs)
    if isinstance(parameter_table, str):
        parameter_table = pd.read_csv(parameter_table)
    param_names = parameter_table.columns.tolist()
    n = len(parameter_table)
    n_proc = _get_n_workers()
    if n < n_proc:
        n_proc = n

    kwargs = dict(
        mcz_obs=mcz_obs,
        duration=duration,
        compas_h5_path=compas_h5_path,
        save_plots=save_images,
        outdir=outdir,
        n_bootstraps=n_bootstraps,
        fname="",
    )
    logger.info(
        f"Generating mcz-grids (with {n_proc} threads for {n} samples of {param_names}) [{kwargs}]"
    )
    kwargs["mcz_obs"] = mcz_obs
    # setting up args for process_map
    _lnl_func = partial(McZGrid.lnl, **kwargs)
    process_map(
        _lnl_func,
        parameter_table.to_dict("records"),
        max_workers=n_proc,
        chunksize=n_proc,
    )
    combine_lnl_data(outdir=outdir)


def combine_lnl_data(
    outdir: str = "out_mcz_grids",
    fname: str = "",
) -> None:
    """
    Combine the likelihood data from the generated mcz-grids into a single file
    :param mcz_obs: Observation object containing the mcz data
    :param parameter_table: path to pandas dataframe containing cosmic integration parameters
    :param n_bootstraps: number of bootstraps to generate for each parameter set
    :param outdir: output directory for mcz-grids
    :return: None
    """
    files = glob(f"{outdir}/*_lnl.csv")
    if fname == "":
        fname = f"{outdir}/combined_lnl_data.csv"
    logger.info(f"Compiling {len(files)} LnL values to {outdir}/{fname}")
    df = pd.concat([pd.read_csv(f) for f in files])
    df.to_csv(fname, index=False)


def _get_n_workers():
    """Get the number of workers for parallel processing"""
    total_cpus_available = cpu_count()
    num_workers = 4
    if total_cpus_available > 64:
        num_workers = 16
    elif total_cpus_available > 32:
        num_workers = 8
    elif total_cpus_available < 16:
        num_workers = 4
    logger.warning(
        f"Using {num_workers}/{total_cpus_available} workers for parallel processing"
        "[Total number of CPUs not used to avoid memory issues]"
    )
    # TODO: check -- are we still having memory issues?
    return num_workers
