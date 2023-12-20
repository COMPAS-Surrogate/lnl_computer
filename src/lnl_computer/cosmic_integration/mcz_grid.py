import os
import shutil
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from compas_python_utils.cosmic_integration.binned_cosmic_integrator.detection_matrix import (
    DetectionMatrix,
)

from ..logger import logger
from .star_formation_paramters import DEFAULT_SF_PARAMETERS
from ..likelihood import ln_likelihood


class McZGrid(DetectionMatrix):
    """
    Represents a detection matrix in (m_c, z) space.

    This class is a wrapper around the DetectionMatrix class, with some extra functionality.

    :param compas_path: The path to the COMPAS hdf5 file
    :param cosmological_parameters: The cosmological parameters to use (dict)
    :param rate_matrix: The detection rate matrix
    :param chirp_mass_bins: The chirp mass bins
    :param redshift_bins: The redshift bins
    :param n_systems: The number of systems in the COMPAS file
    :param n_bbh: The number of BBH systems in the COMPAS file
    :param outdir: The output directory to save the detection matrix to
    :param bootstrapped_rate_matrices: The bootstrapped rate matrices


    Important methods:
    - sample_observations
    - prob_of_mcz

    """

    @classmethod
    def from_dict(cls, data: Dict) -> "McZGrid":
        """Create a mcz_grid object from a dictionary"""
        obj_types = [
            "compas_path",
            "cosmological_parameters",
            "n_systems",
            "n_bbh",
        ]
        for key in data:
            if key in obj_types:
                data[key] = data[key].item()
        det_matrix = cls(
            compas_path=data["compas_path"],
            cosmological_parameters=data["cosmological_parameters"],
            rate_matrix=data["rate_matrix"],
            chirp_mass_bins=data["chirp_mass_bins"],
            redshift_bins=data["redshift_bins"],
            n_systems=data["n_systems"],
            n_bbh=data["n_bbh"],
            outdir=data.get("outdir", "."),
            bootstrapped_rate_matrices=data.get(
                "bootstrapped_rate_matrices", None
            ),
        )
        logger.debug(f"Loaded cached det_matrix with: {det_matrix.param_str}")
        return det_matrix

    @classmethod
    def from_hdf5(
            cls,
            h5file: Union[h5py.File, str],
            idx: int = None,
            search_param: Dict[str, float] = {},
    ):
        """Create a mcz_grid object from a hdf5 file (dont run cosmic integrator)"""
        data = {}
        common_keys = [
            "compas_h5_path",
            "n_systems",
            "redshifts",
            "chirp_masses",
        ]

        h5file_opened = False
        if isinstance(h5file, str):
            h5file = h5py.File(h5file, "r")
            h5file_opened = True

        for key in common_keys:
            data[key] = h5file.attrs.get(key, None)
            if data[key] is None:
                logger.warning(
                    f"Could not find {key} in hdf5 file. Attributes avail: {h5file.attrs.keys()}"
                )

        if idx is None:
            params = h5file["parameters"]
            search_val = [
                search_param["aSF"],
                search_param["bSF"],
                search_param["cSF"],
                search_param["dSF"],
                search_param["muz"],
                search_param["sigma0"],
            ]
            # get index of the closest match
            idx = np.argmin(np.sum((params[:] - search_val) ** 2, axis=1))
            logger.debug(f"Found closest match at index {idx}")

        data["detection_rate"] = h5file["detection_matricies"][idx]
        params = h5file["parameters"][idx]
        data["SF"] = params[:4]
        data["muz"] = params[4]
        data["sigma0"] = params[5]
        mcz_grid = cls(**data)
        logger.debug(f"Loaded cached det_matrix with: {mcz_grid.param_str}")

        if h5file_opened:
            h5file.close()

        return mcz_grid

    def save(self, fname="") -> None:
        """Save the mcz_grid object to a npz file, return the filename"""
        super().save()
        if fname != "":
            shutil.move(self.default_fname, fname)

    def get_matrix_bin_idx(self, mc: float, z: float) -> Tuple[int, int]:
        mc_bin = np.argmin(np.abs(self.chirp_mass_bins - mc))
        z_bin = np.argmin(np.abs(self.redshift_bins - z))
        return mc_bin, z_bin

    def prob_of_mcz(self, mc: float, z: float, duration: float = 1.0) -> float:
        mc_bin, z_bin = self.get_matrix_bin_idx(mc, z)
        return self.rate_matrix[mc_bin, z_bin] / self.n_detections(duration)

    def get_bootstrapped_grid(self, i: int) -> "McZGrid":
        """Creates a new Uni using the ith bootstrapped rate matrix"""
        assert (
                i < self.n_bootstraps
        ), f"i={i} is larger than the number of bootstraps {self.n_bootstraps}"
        return McZGrid(
            compas_path=self.compas_path,
            cosmological_parameters=self.cosmological_parameters,
            rate_matrix=self.bootstrapped_rate_matrices[i],
            chirp_mass_bins=self.chirp_mass_bins,
            redshift_bins=self.redshift_bins,
            n_systems=self.n_systems,
            n_bbh=self.n_bbh,
            outdir=self.outdir,
        )

    def n_detections(self, duration: float = 1.0) -> float:
        """Calculate the number of detections in a given duration (in years)"""
        return np.nansum(self.rate_matrix) * duration

    def get_lnl(self, mcz_obs: np.ndarray, duration: float = 1.0) -> Tuple[float, float]:
        """Get Lnl+/-unc from the mcz_obs"""
        lnl = ln_likelihood(
            mcz_obs=mcz_obs,
            model_prob_func=self.prob_of_mcz,
            n_model=self.n_detections(duration)
        )
        bootstrapped_lnls = []
        for i in range(self.n_bootstraps):
            bootstrap_mcz_grid = self.get_bootstrapped_grid(i)
            bootstrapped_lnls.append(
                ln_likelihood(
                    mcz_obs=mcz_obs,
                    model_prob_func=bootstrap_mcz_grid.prob_of_mcz,
                    n_model=bootstrap_mcz_grid.n_detections(duration),
                )
            )
        return lnl, np.std(np.array(bootstrapped_lnls))

    def __dict__(self) -> Dict:
        return self.to_dict()

    def __repr__(self) -> str:
        return f"<mcz_grid: [{self.n_systems} systems], {self.param_str}>"


    @property
    def param_str(self):
        return "_".join([f"{k}_{v:.10f}" for k, v in self.cosmological_parameters.items()])

    @property
    def label(self) -> str:
        return super().label.replace("detmatrix", "mczgrid")

    @property
    def default_fname(self) -> str:
        return f"{self.outdir}/{self.label}.h5"

    @property
    def param_list(self) -> np.array:
        return np.array(self.cosmological_parameters.values()).flatten()

    @property
    def param_names(self) -> List[str]:
        return list(self.cosmological_parameters.keys())

    @property
    def n_bootstraps(self) -> int:
        if self.bootstrapped_rate_matrices is None:
            return 0
        return len(self.bootstrapped_rate_matrices)

    @classmethod
    def generate_n_save(
            cls,
            compas_h5_path: str,
            sf_sample: Dict,
            save_plots: bool = False,
            outdir=None,
            fname="",
            n_bootstraps=0,
    ) -> "McZGrid":
        """ Generate a detection matrix for a given set of star formation parameters
        :param compas_h5_path:
        :param sf_sample: Dict of star formation parameters
        :param save_plots: Bool to save plots
        :param outdir: outdir for plots + mcz_grid
        :param fname: mcgrid-fname (if empty, will not save)
        :param n_bootstraps: N
        :return:
        """
        if os.path.isfile(fname):
            logger.warning(f"Skipping {fname} generation as it already exists")
            return cls.from_h5(fname)
        if fname != "" and not fname.endswith(".h5"):
            logger.error(f"fname must end with .h5, got {fname}")

        mcz_grid = cls.from_compas_output(
            compas_path=compas_h5_path,
            cosmological_parameters=dict(
                aSF=sf_sample.get("aSF", DEFAULT_SF_PARAMETERS["aSF"]),
                dSF=sf_sample.get("dSF", DEFAULT_SF_PARAMETERS["dSF"]),
                mu_z=sf_sample.get("muz", DEFAULT_SF_PARAMETERS["muz"]),
                sigma_0=sf_sample.get(
                    "sigma0", DEFAULT_SF_PARAMETERS["sigma0"]
                ),
            ),
            max_detectable_redshift=0.6,
            redshift_bins=np.linspace(0, 0.6, 100),
            chirp_mass_bins=np.linspace(3, 40, 50),
            outdir=outdir,
            save_plots=save_plots,
            n_bootstrapped_matrices=n_bootstraps,
        )
        if isinstance(fname, str):
            mcz_grid.save(fname=fname)
        return mcz_grid

    @classmethod
    def lnl(cls, mcz_obs: np.ndarray, duration=1, *args) -> Tuple[float, float]:
        """Return the LnL(sf_sample|mcz_obs)+/-unc

        Also saves the Lnl+/-unc and params to a csv file

        :param mcz_obs: The observed mcz values
        :param duration: The duration of the observation (in years)
        :param args: Arguments to pass to generate_n_save
        :return: The LnL value
        """
        model = cls.generate_n_save(*args)
        lnl, unc = model.get_lnl(mcz_obs=mcz_obs, duration=duration)
        # save lnl data to csv
        df = pd.DataFrame(dict(
            lnl=lnl,
            unc=unc,
            **model.cosmological_parameters,
        ), index=[0])

        fname = args[-2].replace(".h5", "_lnl.csv") if args[-2] != "" else f"{model.outdir}/{model.label}_lnl.csv"
        df.to_csv(fname, index=False)
        return lnl, unc
