import os

import matplotlib.pyplot as plt
import numpy as np

from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.observation.lvk_observation import LVKObservation


def test_lvk_observation(monkeypatched_mcz_grid: McZGrid, tmp_path: str):
    obs = LVKObservation.from_ogc4_data()
    plt_dir = tmp_path + "/obs"
    os.makedirs(plt_dir, exist_ok=True)
    plot_lvk_observation(obs, monkeypatched_mcz_grid, plt_dir, 0)


def plot_lvk_observation(
    obs: LVKObservation, model: McZGrid, plt_dir: str, idx=0
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    z_bins, mc_bins = model.redshift_bins, model.chirp_mass_bins
    Z, MC = np.meshgrid(z_bins, mc_bins)

    w = obs.weights[idx]
    prob = model.rate_matrix * w

    axes[0].pcolor(Z, MC, w)
    axes[1].pcolor(
        Z,
        MC,
        model.rate_matrix,
    )
    axes[2].pcolor(
        Z,
        MC,
        prob,
    )
    axes[0].set_title("Obs")
    axes[1].set_title("Model")
    axes[2].set_title("Obs*Model")
    for i, m in enumerate([w, model.rate_matrix, prob]):
        labl = f"Sum: {np.sum(m):.2e}"
        axes[i].text(
            0.05,
            0.95,
            labl,
            ha="left",
            va="top",
            color="white",
            transform=axes[i].transAxes,
        )
    plt.savefig(plt_dir + f"/lvk_observation_{idx}.png")
