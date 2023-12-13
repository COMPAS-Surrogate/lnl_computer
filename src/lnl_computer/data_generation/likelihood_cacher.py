import os
import random
from itertools import repeat
from typing import List, Optional

import h5py
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ..cosmic_integration.mcz_grid import McZGrid
from ..observation.mock_observation import MockObservation
from ..liklelihood import LikelihoodCache, ln_likelihood
from ..logger import logger

from ..utils import get_num_workers, horizontal_concat


def _get_lnl_and_param_uni(uni: McZGrid, observed_mcz: np.ndarray) -> np.ndarray:
    lnl = ln_likelihood(
        mcz_obs=observed_mcz,
        model_prob_func=uni.prob_of_mcz,
        n_model=uni.n_detections(),
        detailed=False,
    )
    logger.debug(f"Processed {uni} lnl={lnl}.")
    return np.array([lnl, *uni.param_list])


def _get_lnl_and_param_from_npz(npz_fn: str, observed_mcz: np.ndarray) -> np.ndarray:
    uni = McZGrid.from_npz(npz_fn)
    return _get_lnl_and_param_uni(uni, observed_mcz)


def _get_lnl_and_param_from_h5(h5_path: h5py.File, idx: int, observed_mcz: np.ndarray) -> np.ndarray:
    uni = McZGrid.from_hdf5(h5py.File(h5_path, "r"), idx)
    return _get_lnl_and_param_uni(uni, observed_mcz)


def compute_and_cache_lnl(
        mock_population: MockObservation,
        cache_lnl_file: str,
        h5_path: Optional[str] = "",
        mcz_grid_paths: Optional[List] = None,
) -> LikelihoodCache:
    """
    Compute likelihoods given a Mock Population and mcz_grids (either stored in a h5 or the paths to the mcz_grid files).
    """
    if mcz_grid_paths is not None:
        n = len(mcz_grid_paths)
        args = (
            _get_lnl_and_param_from_npz,
            mcz_grid_paths,
            repeat(mock_population.mcz),
        )
    elif h5_path is not None:
        n = len(h5py.File(h5_path, "r")["parameters"])
        args = (
            _get_lnl_and_param_from_h5,
            repeat(h5_path),
            range(n),
            repeat(mock_population.mcz),
        )
    else:
        raise ValueError("Must provide either hf5_path or mcz_grid_paths")

    logger.info(f"Starting LnL computation for {n} mcz_grids")

    try:
        lnl_and_param_list = np.array(
            process_map(
                *args,
                desc="Computing likelihoods",
                max_workers=get_num_workers(),
                chunksize=100,
                total=n,
            )
        )
    except Exception as e:
        lnl_and_param_list = np.array(
            [
                _get_lnl_and_param_from_h5(h5_path, i, mock_population.mcz)
                for i in tqdm(range(n))
            ]
        )
    true_lnl = ln_likelihood(
        mcz_obs=mock_population.mcz,
        model_prob_func=mock_population.mcz_grid.prob_of_mcz,
        n_model=mock_population.mcz_grid.n_detections(),
    )
    lnl_cache = LikelihoodCache(
        lnl=lnl_and_param_list[:, 0],
        params=lnl_and_param_list[:, 1:],
        true_params=mock_population.param_list,
        true_lnl=true_lnl,
    )
    lnl_cache.save(cache_lnl_file)
    mock_population.save(f"{os.path.dirname(cache_lnl_file)}/mock_uni.npz")
    logger.success(f"Saved {cache_lnl_file}")
    return lnl_cache


def get_training_lnl_cache(
        outdir,
        n_samp=None,
        det_matrix_h5=None,
        mcz_grid_id=None,
        mock_uni=None,
        clean=False,
) -> LikelihoodCache:
    """
    Get the likelihood cache --> used for training the surrogate
    Specify the det_matrix_h5 and mcz_grid_id to generate a new cache

    :param outdir: outdir to store the cache (stored as OUTDIR/cache_lnl.npz)
    :param n_samp: number of samples to save in the cache (all samples used if None)
    :param det_matrix_h5: the detection matrix used to generate the lnl cache
    :param mcz_grid_id: the mcz_grid id used to generate the lnl cache
    """
    cache_file = f"{outdir}/cache_lnl.npz"
    if clean and os.path.exists(cache_file):
        logger.info(f"Removing cache {cache_file}")
        os.remove(cache_file)
    if os.path.exists(cache_file):
        logger.info(f"Loading cache from {cache_file}")
        lnl_cache = LikelihoodCache.from_npz(cache_file)
    else:
        os.makedirs(outdir, exist_ok=True)
        h5_file = h5py.File(det_matrix_h5, "r")
        total_n_det_matricies = len(h5_file["detection_matricies"])

        if mock_uni is None:
            if mcz_grid_id is None:
                mcz_grid_id = random.randint(0, total_n_det_matricies)
            assert (
                    mcz_grid_id < total_n_det_matricies
            ), f"mcz_grid id {mcz_grid_id} is larger than the number of det matricies {total_n_det_matricies}"
            mock_uni = McZGrid.from_hdf5(h5_file, mcz_grid_id)
        else:
            assert isinstance(mock_uni, McZGrid)

        mock_population = mock_uni.sample_possible_event_matrix()
        mock_population.plot(save=True, fname=f"{outdir}/injection.png")
        logger.info(
            f"Generating cache {cache_file} using {det_matrix_h5} and mcz_grid {mcz_grid_id}:{mock_population}"
        )
        lnl_cache = compute_and_cache_lnl(
            mock_population, cache_file, h5_path=det_matrix_h5
        )

    plt_fname = cache_file.replace(".npz", ".png")
    lnl_cache.plot(fname=plt_fname, show_datapoints=True)
    train_plt_fname = plt_fname.replace(".png", "_training.png")
    if n_samp is not None:
        lnl_cache = lnl_cache.sample(n_samp)
    lnl_cache.plot(fname=train_plt_fname, show_datapoints=True)
    horizontal_concat(
        [plt_fname, train_plt_fname], f"{outdir}/cache_pts.png", rm_orig=False
    )

    return lnl_cache
