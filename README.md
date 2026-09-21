# ISOSNOW - ISOFIT Snow Surface Model

## Overview

This diverges from ISOFIT dev branch on 31 July 2026, but relevant PRs/patches will continue to be incorperated (see change log). The ISOFIT Snow Surface model branch is used to estimate snow, topographic, and atmospheric parameters directly from TOA radiance data using BRDF snow models. This branch also computes the `_snow` and `_snow_uncert` image files, which contain important information like albedo and canopy-adjusted fractional snow covered area. 

## Installation

RT and other ancillary data must be downloaded/installed similar to ISOFIT. There is more information here: https://isofit.github.io/isofit

isosnow-40 branch can be installed by:

```
git clone https://github.com/brentwilder/isofit.git
git checkout isosnow-40
cd /to/branch
pip install -e .
```

## Usage

- See `./isosnow_scripts/run.sh` for running the model on an image

- The low-rank model files for PV, NPV, and Soil are in the `isosnow_data` folder and must be copied over into the home `~/.isofit/data` directory prior to running.


## Change log

- 21 September 2026: Update AOD prior to 0.05 +/- 0.005. Bring in fsnow, grain size, cosi, water vapor heuristics prior to OE for small speed/stability gain. Decrease ftol from 0.01 to 1e-7.

- 18 September 2026: Bring in some prior beliefs about the static cos(i) from the DEM (Dozier et al., 2022). This is a function of per-pixel slope and aspect uncertainties.

- 17 September 2026: Bring in weights based on example in Richter 1998 (Correction of satellite imagery over mountainous terrain) for adjacency range. This is similar to what we showed in Alex's Springer Series. And then also, stopped computing Sa_inv every iteration because it is always the same in the snow model. Finally, S_hat is fully corrected now, and utilizes Sa (allowing us to use informative priors if we need at any point).

- 16 September 2026: Adding sqrt transformation for grain radius to reduce interpolation error

- 10 September 2026: Integrated analytical Jacobian to account for cos_i, instead of relying on numerical, 2-point method (EMIT speed test: 11.22 spectra/sec to 20.99 spectra/sec). Also, fixed bug to assign average rfl to nodata for adjacency calc (prior to aggregating). 

- 9 September 2026: ISOFIT PR-1026, inversion windows patch (EMIT)

- 2 September 2026: ISOFIT PR-1012, MODTRAN TP7 codes

- 2 September 2026: ISOFIT PR-1022, Background topo updates

- 2 September 2026: ISOFIT PR-979, enables setting priors in the surface JSON


## Additional notes

- `COS_I` is always set to be solved (instead of "flat" or "dem"), and is fully hooked up between surface and atmosphere RT.

- Post-processing requires the albedo LUT and a canopy fraction dataset. Canopy fraction data must match dimensions of input data exactly.
