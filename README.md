# COMPAS Detection Likelihood computer
Utils to run the COMPAS cosmic-integration code in large batches and compute the LnL given a Mc-z detection matrix.

Given metallicity and star formation params, and a detection rate matrix this helps 
- run COMPAS's `cosmic-integrator` code
- saves the detection rate matrices given the params
- computes the LnL (and bootstrapped uncertainty)
- saves the LnL and uncertainty

## Installation
- Clone the COMPAS repo + install the COMPAS py tools (pip install -e .)
- Clone this  repo + install package (pip install -e .)

