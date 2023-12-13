"""Module to run COMPAS simulations and generate data for surrogate model"""

import os

import pandas as pd
from tqdm.contrib.concurrent import process_map

from .star_formation_paramters import draw_star_formation_samples
from .mcz_grid import McZGrid
from ..logger import logger
from typing import List, Dict, Tuple, Optional, Union

from ..utils import get_num_workers, make_gif


def generate_set_of_matrices(
        compas_h5_path: str,
        parameters: List[str] = None,
        parameter_table: Union[pd.DataFrame, str] = None,
        n: int = 50,
        custom_ranges: Dict[str, Tuple[float, float]] = None,
        grid_parameterspace=False,
        save_images: bool = True,
        outdir: str = "out_mcz_grids",

):
    """
    Generate a set of COMPAS detection rate matricies
    :param compas_h5_path: Path to COMPAS h5 file
    :param n: number of matricies to generate
    :param save_images: save images of the matricies
    :param outdir: dir to save data and images
    :param parameters: parameters to draw from for matrix [aSF, dSF, muz, sigma0]
    :return:
    """
    os.makedirs(outdir, exist_ok=True)
    param_table = _parse_parameter_table(parameters, parameter_table, n, custom_ranges, grid_parameterspace)
    parameters = param_table.columns.tolist()

    n = len(param_table)

    n_proc = get_num_workers()
    logger.info(
        f"Generating mcz-grids (with {n_proc} threads for {n} samples of parameters {parameters})"
    )

    args = (
        [compas_h5_path] * n,
        param_table[parameters].to_dict("records"),
        [save_images] * n,
        [outdir] * n,
        [f"{outdir}/uni_{i}.npz" for i in range(n)]  # fnames
    )
    process_map(McZGrid.generate_n_save, *args, max_workers=n_proc, chunksize=n_proc)

    if save_images:
        make_gif(
            os.path.join(outdir, "plot_mczgrid_*.png"),
            os.path.join(outdir, "mczgrid.gif"),
            duration=100,
            loop=True,
        )


def _parse_parameter_table(
        parameters: List[str] = None,
        parameter_table: Union[pd.DataFrame, str] = None,
        n: int = 50,
        custom_ranges: Dict[str, Tuple[float, float]] = None,
        grid_parameterspace=False,
        outdir: str = "out_mcz_grids",
) -> pd.DataFrame:
    if isinstance(parameter_table, pd.DataFrame):
        pass

    elif isinstance(parameter_table, str) and os.path.isfile(parameter_table):
        parameter_table = pd.read_csv(parameter_table)

    else:
        parameters = parameters or ["aSF", "dSF", "mu_z", "sigma_z"]
        parameter_table = pd.DataFrame(draw_star_formation_samples(
            n,
            parameters=parameters,
            as_list=True,
            custom_ranges=custom_ranges,
            grid=grid_parameterspace,
        ))
        # TODO: DOUBLE CHECK WITH JEFF -- Muz, sigma_z, sigma_0???

    fname = f"{outdir}/parameter_table.csv"
    parameter_table.to_csv(fname, index=False)
    return parameter_table
