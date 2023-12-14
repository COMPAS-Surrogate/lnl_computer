import os

import pandas as pd
from glob import glob
from typing import Dict, Union
from tqdm.contrib.concurrent import process_map
import click
from multiprocessing import cpu_count

from ..cosmic_integration.mcz_grid import McZGrid
from ..observation.observation import Observation
from ..observation.mock_observation import MockObservation
from ..logger import logger



from typing import Dict, Union
import click

from ..cosmic_integration.mcz_grid import McZGrid
from ..observation.mock_observation import MockObservation
from ..logger import logger

import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Union

import pandas as pd

from ..logger import logger
from ..cosmic_integration.star_formation_paramters import draw_star_formation_samples
import click


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
    parameters = parameters or ["aSF", "dSF", "mu_z", "sigma_z"]
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
    logger.info(f"Parameter table saved to {fname}")




def make_mock_obs(
        compas_h5_path: str,
        sf_sample: Union[Dict, str],
        fname: str = "mock_observation.npz",
) -> "MockObservation":
    """ Generate a detection matrix for a given set of star formation parameters
    :param compas_h5_path:
    :param sf_sample: Dict of star formation parameters, or a string like "k1:v1 k2:v2"->{"k1":float(v1)", "k2":float(v2)}
    :param fname: mcgrid-fname
    """
    assert fname.endswith(".npz"), ("fname must end with .npz")
    if isinstance(sf_sample, str):
        sf_sample = dict([s.split(":") for s in sf_sample.split(" ")])
        sf_sample = {k: float(v) for k, v in sf_sample.items()}

    mcz_grid = McZGrid.generate_n_save(
        compas_h5_path=compas_h5_path,
        sf_sample=sf_sample,
        save_plots=False,
    )
    obs = MockObservation.from_mcz_grid(mcz_grid)
    obs.save(fname)
    logger.info(f"Mock observation saved to {fname}")



def batch_lnl_generation(
        mcz_obs: Union[Observation, str],
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
    os.makedirs(outdir, exist_ok=True)
    if isinstance(mcz_obs, str):
        mcz_obs = MockObservation.from_npz(mcz_obs).mcz
    if isinstance(parameter_table, str):
        parameter_table = pd.read_csv(parameter_table)
    param_names = parameter_table.columns.tolist()
    n = len(parameter_table)
    n_proc = _get_n_workers()
    if n < n_proc:
        n_proc = n
    logger.info(
        f"Generating mcz-grids (with {n_proc} threads for {n} samples of {param_names})"
    )
    # setting up args for process_map
    args = dict(
        mcz_obs=[mcz_obs] * n,
        duration=[1] * n,
        compas_h5_path=[compas_h5_path] * n,
        sf_params=parameter_table.to_dict("records"),
        save_plots=[save_images] * n,
        outdir=[outdir] * n,
        fname=[f"{outdir}/uni_{i}.npz" for i in range(n)],
        n_bootstraps=[n_bootstraps] * n,
    ).values()
    process_map(
        McZGrid.lnl, *args, max_workers=n_proc, chunksize=n_proc
    )
    combine_lnl_data(outdir=outdir)



def combine_lnl_data(
        outdir: str = "out_mcz_grids",
        fname: str = 'combined_lnl_data.csv',
) -> None:
    """
    Combine the likelihood data from the generated mcz-grids into a single file
    :param mcz_obs: Observation object containing the mcz data
    :param parameter_table: path to pandas dataframe containing cosmic integration parameters
    :param n_bootstraps: number of bootstraps to generate for each parameter set
    :param outdir: output directory for mcz-grids
    :return: None
    """
    files = glob(f'{outdir}/*_lnl.csv')

    logger.info(f"Compiling {len(files)} LnL values to {outdir}/{fname}")
    df = pd.concat([pd.read_csv(f) for f in files])
    df.to_csv(f"{outdir}/{fname}", index=False)


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
