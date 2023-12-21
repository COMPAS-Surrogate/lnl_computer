from typing import Dict, List, Union

import numpy as np
from bilby.core.prior import PriorDict, Uniform
from scipy.stats import qmc
from scipy.stats.qmc import LatinHypercube

DEFAULT_SF_PARAMETERS = dict(
    aSF=0.01,
    bSF=2.77,
    cSF=2.9,
    dSF=4.7,
    muz=-0.23,
    sigma0=0.39,
)

STAR_FORMATION_RANGES = dict(
    muz=[-0.5, -0.001],  # Jeff's alpha
    sigma0=[0.1, 0.6],  # Jeff's sigma
    aSF=[0.005, 0.015],
    dSF=[4.2, 5.2],
)
LATEX_LABELS = dict(
    muz=r"$\mu_z$",
    sigma0=r"$\sigma_0$",
    aSF=r"$\rm{SF}[a]$",
    dSF=r"$\rm{SF}[d]$",
)


class MyPriorDict(PriorDict):
    @property
    def bounds(self):
        return np.array(
            [[self[p].minimum, self[p].maximum] for p in self.keys()]
        )

    @property
    def n_params(self):
        return len(self.keys())

    def sample_val(self):
        return np.array([self[p].sample() for p in self.keys()]).reshape(1, -1)


def get_star_formation_prior(parameters=None) -> MyPriorDict:
    if parameters is None:
        parameters = list(STAR_FORMATION_RANGES.keys())
    pri = dict()
    for p in parameters:
        pri[p] = Uniform(
            *STAR_FORMATION_RANGES[p], name=p, latex_label=LATEX_LABELS[p]
        )
    return MyPriorDict(pri)


def draw_star_formation_samples(
    n: int = 1000,
    parameters: List[str] = None,
    as_list=False,
    custom_ranges: Dict = None,
    grid: bool = False,
) -> Union[Dict[str, np.ndarray], List[Dict]]:
    """Draw samples from the star formation parameters.
    Returns a dictionary of arrays, or a list of dictionaries if as_list is True.
    """
    if parameters is None:
        parameters = list(STAR_FORMATION_RANGES.keys())
    assert all(
        [p in STAR_FORMATION_RANGES for p in parameters]
    ), "Invalid parameters"
    num_dim = len(parameters)

    ranges = STAR_FORMATION_RANGES.copy()
    if custom_ranges is not None:
        ranges.update(custom_ranges)
    parameter_ranges = np.array([ranges[p] for p in parameters])
    lower_bound = parameter_ranges[:, 0]
    upper_bound = parameter_ranges[:, 1]

    if grid:
        n_per_dim = int(np.ceil(n ** (1 / num_dim)))
        grid = np.meshgrid(
            *[np.linspace(*r, n_per_dim) for r in parameter_ranges]
        )
        samples = np.vstack([g.ravel() for g in grid]).T

    else:
        sampler = LatinHypercube(d=num_dim)
        unscaled_samples = sampler.random(n)
        samples = qmc.scale(
            unscaled_samples, l_bounds=lower_bound, u_bounds=upper_bound
        )

    dict_of_params = {p: samples[:, i] for i, p in enumerate(parameters)}
    if as_list:
        return [
            dict(zip(dict_of_params, t)) for t in zip(*dict_of_params.values())
        ]
    return dict_of_params
