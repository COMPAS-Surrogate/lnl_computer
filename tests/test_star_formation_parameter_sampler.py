import os

import pandas as pd

from lnl_computer.cosmic_integration.star_formation_paramters import (
    draw_star_formation_samples,
    get_star_formation_prior,
)

PLOT = True


def test_prior():
    p = get_star_formation_prior(parameters=["aSF", "dSF"])
    assert p.bounds.shape == (2, 2)
    p = get_star_formation_prior()
    assert p.bounds.shape == (p.n_params, 2)
    assert p.sample_val().shape == (1, p.n_params)


def test_sampler(tmp_path):
    parameters = ["aSF", "dSF"]
    samples_1000 = draw_star_formation_samples(1000, parameters)
    assert samples_1000["aSF"].shape == (1000,)
    assert samples_1000["dSF"].shape == (1000,)

    samples_30 = pd.DataFrame(
        draw_star_formation_samples(30, parameters, as_list=True)
    )

    samples_grid = pd.DataFrame(
        draw_star_formation_samples(50, parameters, grid=True)
    )

    samples_all = draw_star_formation_samples(1000)

    if PLOT:
        import matplotlib.pyplot as plt

        plt.plot(
            samples_1000["aSF"],
            samples_1000["dSF"],
            ".",
            color="tab:red",
            zorder=0,
        )
        plt.plot(
            samples_30["aSF"],
            samples_30["dSF"],
            ".",
            color="tab:blue",
            zorder=1,
        )
        plt.plot(
            samples_grid["aSF"],
            samples_grid["dSF"],
            "s",
            color="tab:green",
            zorder=1,
            alpha=0.5,
        )
        plt.grid()
        plt.savefig(os.path.join(tmp_path, "test_sf_samples.png"))
