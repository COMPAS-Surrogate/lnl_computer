[![Coverage Status](https://coveralls.io/repos/github/COMPAS-Surrogate/lnl_computer/badge.svg)](https://coveralls.io/github/COMPAS-Surrogate/lnl_computer)
# COMPAS Detection Likelihood computer

Utils to run the COMPAS cosmic-integration code in large batches and compute the LnL given a Mc-z detection matrix.

Given metallicity and star formation params, and a detection rate matrix this helps

- run COMPAS's `cosmic-integrator` code
- saves the detection rate matrices given the params
- computes the LnL (and bootstrapped uncertainty)
- saves the LnL and uncertainty

## CLI Interface:

```
batch_lnl_generation <-- runs the cosmic integrator for a set of params + saves the LnL 
combine_lnl_data <-- combines the LnL data from multiple runs
make_mock_obs <-- generates a Mc-z detection matrix and a set of observations of 'observed' BBHs params
make_sf_table <-- generates a table of star formation params
make_mock_compas_output <-- generates a mock COMPAS output file
```

## Installation

- Clone the COMPAS repo + install the COMPAS py tools (pip install -e .)
- Clone this repo + install package (pip install -e .)

