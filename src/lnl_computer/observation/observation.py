from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt


class Observation(ABC):

    def __init__(self, mcz: np.ndarray):
        """
        An observation of a population of BBHs.

        :param mcz: Shape (n_events, 2), where each row is (mc, z), corresponding to the chirp mass and redshift of each event
        """
        self.mcz = mcz
        self.outdir = ""

    @property
    def n_events(self) -> int:
        return len(self.mcz)

    @property
    @abstractmethod
    def label(self) -> str:
        pass

    @abstractmethod
    def __dict__(self):
        pass


    @abstractmethod
    def plot(self) -> plt.Figure:
        pass

    def save(self, fname: str = ""):
        if fname == "":
            fname = f"{self.outdir}/{self.label}.npz"
        np.savez(fname, **self.__dict__())

    @classmethod
    @abstractmethod
    def from_npz(cls, path: str) -> "Observation":
        pass