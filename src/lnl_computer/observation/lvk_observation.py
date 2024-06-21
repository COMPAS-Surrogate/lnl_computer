import numpy as np
from ogc4_interface.population_mcz import PopulationMcZ

from .observation import Observation


class LVKObservation(Observation):
    def __init__(self, weights: np.ndarray):
        self._weights = weights
        self._n_events, self._n_mc_bins, self._n_z_bins = weights.shape

    @classmethod
    def from_ogc4(cls, pastro_threshold=0.95):
        data = PopulationMcZ.load(pastro_threshold=pastro_threshold)
        w = data.weights
        # change from (n_events, z_bins, mc_bins) to (n_events, mc_bins, z_bins)
        w = np.moveaxis(w, (0, 1, 2), (0, 2, 1))
        return cls(w)

    @property
    def n_events(self) -> int:
        return self._n_events

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def label(self) -> str:
        return "LVKObs(n={}, bins=[{}, {}]".format(
            self._n_events, self._n_z_bins, self._n_mc_bins
        )
