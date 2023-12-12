from typing import Dict, List

import numpy as np
import xarray as xr

from dataclasses import dataclass


@dataclass
class DataCache:
    """Class to hold X (input) and y (output) data.

    x: xr.Dataset input data
    y: xr.Dataset output data

    Has the following functions:
    - save(fname): save data to netcdf file
    - load(fname): load data from netcdf file
    - varying_params: the names of the varying parameters (X)
    - train_test_split(frac_train): split data into train and test sets based on frac_train (returns 2 DataCache objects)
    """
    x: xr.Dataset
    y: xr.Dataset

    @classmethod
    def from_dataset(cls, d: xr.Dataset):
        """Create a DataCache from an xarray dataset."""
        x = d[d.attrs["in_params"]]
        y = d[d.attrs["out_params"]]
        return cls(x, y)

    @classmethod
    def from_dict(cls, x: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> "DataCache":
        """Create a DataCache from a dictionary of numpy arrays."""
        coords = {"sample": np.arange(len(x[list(x.keys())[0]]))}
        x = xr.Dataset({k: (["sample"], v) for k, v in x.items()}, coords=coords)
        y = xr.Dataset({k: (["sample"], v) for k, v in y.items()}, coords=coords)
        return cls(x, y)

    def save(self, fname: str) -> None:
        """Save data to netcdf file."""
        self.data.to_netcdf(fname)

    @classmethod
    def load(cls, fname: str) -> "DataCache":
        """Load data from netcdf file."""
        return cls.from_dataset(xr.open_dataset(fname))

    @property
    def data(self) -> xr.Dataset:
        """Return the data."""
        if not hasattr(self, "_data"):
            d = xr.merge([self.x, self.y])
            d.attrs["in_params"] = list(self.x.data_vars)
            d.attrs["out_params"] = list(self.y.data_vars)
            self._data = d
        return self._data

    def train_test_split(self, frac_train: float = 0.8) -> List["DataCache"]:
        """Split data into train and test sets based on frac_train (returns 2 DataCache objects)."""
        idx = np.arange(len(self))
        np.random.shuffle(idx)
        train_idx = idx[: int(frac_train * len(self))]
        test_idx = idx[int(frac_train * len(self)):]

        train_data = DataCache.from_dataset(self.data.isel(sample=train_idx))
        test_data = DataCache.from_dataset(self.data.isel(sample=test_idx))
        return [
            DataCache(train_data.x, train_data.y),
            DataCache(test_data.x, test_data.y),
        ]

    def __len__(self):
        return len(self.x.sample)

    @property
    def x_array(self):
        """Return the input data as a numpy array.
        Each row is a sample, each column is a parameter.
        """
        return self.x.to_array().values.T

    @property
    def y_array(self):
        """Return the input data as a numpy array.
        Each row is a sample, each column is a parameter.
        """
        return self.y.to_array().values.T

    @property
    def in_shape(self):
        return self.x_array.shape[1:]
