import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


class Observation:
    def __init__(
        self,
        weights: np.ndarray,
        mc_bins: np.array,
        z_bins: np.array,
        duration: float,  # in years
        label: str = "",
        cosmological_parameters: Dict[str, float] = None,
    ):
        self.weights = weights
        self.mc_bins = mc_bins
        self.z_bins = z_bins
        self.duration = duration
        self.n_events, self.n_mc_bins, self.n_z_bins = weights.shape
        assert len(mc_bins) == self.n_mc_bins
        assert len(z_bins) == self.n_z_bins
        self.label = label
        self.cosmological_parameters = cosmological_parameters

    def __repr__(self):
        return f"Obs({self.weights_str(self.weights)}, {self.duration} years)"

    def __str__(self):
        return self.__repr__()

    def __dict__(self):
        d = {
            "weights": self.weights,
            "mc_bins": self.mc_bins,
            "z_bins": self.z_bins,
            "duration": self.duration,
        }
        if self.cosmological_parameters:
            params = self.cosmological_parameters
            params = np.array(
                [(k, v) for k, v in params.items()],
                dtype=[("key", "U10"), ("value", "f8")],
            )
            d["cosmological_parameters"] = params
        d["label"] = self.label
        return d

    def save(self, fname: str = "", outdir: str = ""):
        if outdir:
            fname = os.path.join(outdir, f"{self.label}.npz")
        np.savez(fname, **self.__dict__())

    @classmethod
    def from_dict(cls, data: Dict):
        if "cosmological_parameters" in data:
            params = data["cosmological_parameters"]
            if isinstance(params, np.ndarray):
                params = {k: v for k, v in params}
            data["cosmological_parameters"] = params
        if isinstance(data["label"], np.ndarray):
            data["label"] = data["label"].item()
        if isinstance(data["duration"], np.ndarray):
            data["duration"] = data["duration"].item()
        return cls(**data)

    @classmethod
    def load(cls, fname: str):
        data = np.load(fname)
        loaded_data = {k: data[k] for k in data.files}
        if "label" not in loaded_data:
            loaded_data["label"] = os.path.basename(fname).split(".")[0]
        return cls.from_dict(loaded_data)

    def plot(self, fname=None, ax=None, title=None) -> plt.Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        fig = ax.get_figure()

        Z, MC = np.meshgrid(self.z_bins, self.mc_bins)
        cbar = ax.pcolor(Z, MC, self.weights.sum(axis=0), cmap="inferno")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\mathcal{M}_{\rm src}\, [M_{\odot}]$")
        cbar = fig.colorbar(cbar, ax=ax)
        cbar.set_label(r"$\sum_{\rm events} w[i, z, \mathcal{M}]$")
        if title is None:
            title = str(self)
        ax.set_title(title)
        if fname is not None:
            fig.savefig(fname)
        return fig

    @staticmethod
    def weights_str(weights):
        ns, mcs, zs = weights.shape
        return f"n={ns}, bins=[{mcs}, {zs}]"
