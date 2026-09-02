# ISOSNOW - ISOFIT Snow Surface Model


## Overview

- This diverges from ISOFIT dev branch on around 31 July 2026

- Plan to bring in major updates from future ISOFIT PRs as needed (shown in change log)


## Usage

- See `./isofit_scripts`

- The low-rank model files for PV, NPV, and Soil are in the `isosnow_data` folder and must be copied over into the home `~/.isofit/data` directory prior to running.

## Change log

- 2 September 2026: ISOFIT PR-1012, MODTRAN TP7 codes

- 2 September 2026: ISOFIT PR-1022, Background topo updates

- 2 September 2026: ISOFIT PR-979, enables setting priors in the surface JSON


## Other (maybe) helpful notes

- Least squares uses 2-point method , skips custom Jac that ISOFIT has implemented.

- `COS_I` is always set to be solved (instead of "flat" or "dem"), and is fully hooked up between surface and atmosphere RT.

