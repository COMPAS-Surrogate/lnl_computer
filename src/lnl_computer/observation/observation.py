from abc import ABC, abstractmethod

import numpy as np


class Observation(ABC):
    @property
    @abstractmethod
    def n_events(self) -> int:
        pass

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """
        The weights of each event
        shape: (n_events, z_bins, mc_bins)
        """
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        pass

    def __repr__(self):
        return self.label
