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

- 16 September 2026: Adding sqrt transformation for grain radius to reduce interpolation error

- 10 September 2026: Integrated analytical Jacobian to account for cos_i, instead of relying on numerical, 2-point method (EMIT speed test: 11.22 spectra/sec to 20.99 spectra/sec). Also, fixed bug to assign average rfl to nodata for adjacency calc (prior to aggregating). 

- 9 September 2026: ISOFIT PR-1026, inversion windows patch (EMIT)

- 2 September 2026: ISOFIT PR-1012, MODTRAN TP7 codes

- 2 September 2026: ISOFIT PR-1022, Background topo updates

- 2 September 2026: ISOFIT PR-979, enables setting priors in the surface JSON


## Additional notes

- Currently `S_hat` does not use any information from `Sa` because we use typically use uninformative priors in the snow model

- `COS_I` is always set to be solved (instead of "flat" or "dem"), and is fully hooked up between surface and atmosphere RT.

- Post-processing requires the albedo LUT and a canopy fraction dataset. Canopy fraction data must match dimensions of input data exactly.
