[![Coverage Status](https://coveralls.io/repos/github/COMPAS-Surrogate/lnl_computer/badge.svg)](https://coveralls.io/github/COMPAS-Surrogate/lnl_computer)
[![PyPI version](https://badge.fury.io/py/compas-surrogate.lnl_computer.svg?icon=si%3Apython)](https://badge.fury.io/py/compas-surrogate.lnl_computer)

# [COMPAS Detection Likelihood computer](https://github.com/COMPAS-Surrogate/lnl_computer)

Utils to run the COMPAS cosmic-integration code in batches and compute the LnL given SF params and set of observations.

More specifically, this package:
- runs COMPAS's `cosmic-integrator` code for SF params
- saves the detection rate matrices given the SF params
- computes the LnL (and bootstrapped uncertainty) given a set of observations
- saves the LnL and uncertainty for the SF params and observations

## Installation

```bash
pip install lnl_computer@git+https://github.com/COMPAS-Surrogate/lnl_computer.git
```

## Example

```python
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.mock_data import generate_mock_data


SF_SAMPLE = dict(aSF=0.01, dSF=4.70, mu_z=-0.23, sigma_z=0.0)

mock_data = generate_mock_data(outdir='temp', sf_params=SF_SAMPLE)
lnl, unc = McZGrid.lnl(
        mcz_obs=mock_data.observations.mcz,
        compas_h5_path=mock_data.compas_filename,
        sf_sample=SF_SAMPLE,
        n_bootstraps=0,
        outdir='temp',
        save_plots=False, # Set to True to save plots for diagnostics
    )
```

![](https://user-images.githubusercontent.com/15642823/227399574-3945c7da-564d-46da-8a0f-de830ebcc0e8.png)

## CLI Interface

```bash
!batch_lnl_generation --help
Usage: batch_lnl_generation [OPTIONS] MCZ_OBS COMPAS_H5_PATH PARAMETER_TABLE

  Given observations (MCZ_OBS, npz-file), COMPAS output (COMPAS_H5_PATH, h5
  file), and a table of SF parameters (PARAMETER_TABLE, csv file), generate
  McZ grids and compute likelihoods.

  The likelihoods are saved to OUTDIR/*_lnl.csv

Options:
  --n_bootstraps INTEGER  Number of bootstraps to generate for each parameter
                          set (used for error estimation)  [default: 100]
  --plots / --no_plots    Save diagnostic plots for each parameter set
                          [default: plots]
  --outdir TEXT           Outdir for mcz-grids  [default: out_mcz_grids]
  --help                  Show this message and exit.
```
```bash
!combine_lnl_data --help
Usage: combine_lnl_data [OPTIONS] [OUTDIR]

  Combine the likelihood data in 'OUTDIR/*_lnl.csv' -> FNAME

  OUTDIR: Output directory with likelihood files (csvs)

Options:
  --fname TEXT  Output filename (must be a .csv)
  --help        Show this message and exit.
```
```bash
!make_mock_obs --help
Usage: make_mock_obs [OPTIONS] COMPAS_H5_PATH

  Generate a set of 'mock' observations for the sf-sample and compas output
  file (COMPAS_H5_PATH).

Options:
  --sf_sample TEXT  Star formation parameters  [default: aSF:0.01 dSF:0.01
                    muz:0.01 sigma0:0.01]
  --fname TEXT      Output filename (must be a .npz)
  --help            Show this message and exit.
```
```bash
!make_sf_table --help
Usage: make_sf_table [OPTIONS]

  Parses the table of parameters to generate mcz-grids for.

Options:
  -p, --parameters TEXT      List of parameters to generate mcz-grids for
  -n, --n INTEGER            Number of samples to generate
  -g, --grid_parameterspace  Whether to grid the parameter space
  -f, --fname TEXT           Output filename (must be a .csv)
  --help                     Show this message and exit.
```
```bash
!make_mock_compas_output --help
Usage: make_mock_compas_output [OPTIONS]

  Generate a mock COMPAS output file at FNAME

Options:
  --fname TEXT  Output filename (must be a .h5)
  --help        Show this message and exit.
```
