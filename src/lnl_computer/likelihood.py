from typing import Callable, Tuple, Union

import numpy as np

from .observation.observation import Observation



def ln_poisson_likelihood(
    n_obs: float, n_model: float, ignore_factorial=True
) -> float:
    """
    Computes LnL(N_obs | N_model) = N_obs * ln(N_model) - N_model - ln(N_obs!)

    :param n_obs: number of observed events
    :param n_model: number of events predicted by the model
    :param ignore_factorial: ignore the factorial term in the likelihood

    # TODO: Why are we ignoring the factorial term?? It was in Ilya's notes from 2023, but unsure why...

    :return: the log likelihood
    """
    if n_model <= 0:
        return -np.inf
    lnl = n_obs * np.log(n_model) - n_model
    if ignore_factorial is False:
        lnl += -np.log(np.math.factorial(n_obs))
    return lnl


def ln_mcz_grid_likelihood(
    obs_weights: np.ndarray, model_prob_grid: np.ndarray
) -> float:
    """
    Computes LnL(mc, z | model)
        =  sum_n  ln sum_i  p(mc_i, z_i | model) * wi,n
        (for N_obs events, and i {mc,z} bins)


    for even_idx in range(n_events):
        p_event = 0
        for mc_idx in range(n_mc):
            for z_idx in range(n_z):
                w = obs_weights[even_idx, mc_idx, z_idx]
                p_event += w * model_prob_grid[mc_idx, z_idx]
        lnl += np.log(p_event)

    """
    p_events = np.einsum("nij,ij->n", obs_weights, model_prob_grid)
    return np.nansum(np.log(p_events))


def ln_likelihood(
    obs: Observation,
    model: "McZGrid",
    detailed=False,
) -> Union[float, Tuple[float, float, float, float]]:
    """
    Compute the log likelihood of the model given the observation

    :param obs: the observation
    :param model: the McZ-grid model
    :param detailed: return detailed likelihood components

    :return: the log likelihood
    if detailed:
        return [lnl, poisson_lnl, mcz_lnl, model_n_detections]

    """

    # unpack the model into the grid and the number of detections
    model_grid = model.get_prob_grid(obs.duration)
    model_n_detections = model.n_detections(obs.duration)

    # compute the likelihood
    poisson_lnl = ln_poisson_likelihood(obs.n_events, model_n_detections)
    mcz_lnl = ln_mcz_grid_likelihood(obs.weights, model_grid)
    lnl = poisson_lnl + mcz_lnl
    if detailed:
        return np.array([lnl, poisson_lnl, mcz_lnl, model_n_detections])
    return lnl

