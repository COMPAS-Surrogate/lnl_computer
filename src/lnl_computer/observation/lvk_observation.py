import numpy as np
from ogc4_interface.population_mcz import PopulationMcZ

from .observation import Observation


class LVKObservation(Observation):
    @classmethod
    def from_ogc4_data(
        cls,
        pastro_threshold=0.95,
        observing_runs=["O3a", "O3b"],
        filter_valid_mcz=True,
    ) -> "LVKObservation":
        data = PopulationMcZ.load(
            pastro_threshold=pastro_threshold,
            observing_runs=observing_runs,
            filter_valid_mcz=filter_valid_mcz,
        )
        w = data.weights
        # change from (n_events, z_bins, mc_bins) to (n_events, mc_bins, z_bins)
        w = np.moveaxis(w, (0, 1, 2), (0, 2, 1))
        ne, nmc, nz = w.shape
        return cls(
            w,
            mc_bins=data.mc_bins,
            z_bins=data.z_bins,
            label=f"LVKObs({cls.weights_str(w)})]",
        )

    def __repr__(self):
        return "LVK" + super().__repr__()
