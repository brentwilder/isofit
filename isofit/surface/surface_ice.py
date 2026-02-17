#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: Niklas Bohn, urs.n.bohn@jpl.nasa.gov


import numpy as np
import xarray as xr

from .surface_multicomp import MultiComponentSurface
from isofit.core.common import eps, load_wavelen, VectorInterpolator
from isofit.configs import Config
from isofit.data import env


class IceSurface(MultiComponentSurface):
    """
    TODO: experimental, may not be completed depending on our works,

    """

    def __init__(self, full_config: Config):

        super().__init__(full_config)

        # entire state vector
        self.statevec_names = [
            "cosi",  # topography
            "Grain_radius",
            "Algae",
            "Dust",
            "Liquid_water",  # ice properties (grain is actually bubble radius)
        ]
        self.scale = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        self.init = [
            0.8,
            20.0,
            0.0,
            0.0,
            0.0,
        ]
        self.bounds = np.array(
            [
                [0.01, 1.0],  # cosi
                [10.0, 25000.0],  # radius
                [0.0, 6e4],  # algae cells per mL
                [0.0, 1.0],  # dust concentration      #TODO DEBUG
                [0.0, 50.0],  # liquid water
            ]
        )

        self.n_state = len(self.statevec_names)
        self.idx_surface = np.arange(len(self.statevec_names))

        self.x_fixed = None

        # load DISORT data to Jouni's interp class
        # NOTE: need to make this path based on sensoorr
        config = full_config.forward_model.instrument
        if config.wavelength_file is not None:
            self.wl, self.fwhm = load_wavelen(config.wavelength_file)

        # NOTE: this is imperfect. And also, for APEX the exact FWHM WL vary based on flight.
        # Therefore, need to find better way or remember to pre-convolve for each new APEX we process.
        if len(self.wl) == 425:
            disort_sensor = "ANG"
        if len(self.wl) == 230:
            disort_sensor = "PRISMA"
        if len(self.wl) == 285:
            disort_sensor = "EMIT"
        if len(self.wl) == 299:
            disort_sensor = "APEX"

        # disort_path = env.path("data", f"disort_ice_lut_{disort_sensor}.nc")
        disort_path = "/Users/bawilder/Code/snow/LUT/qaanaaq_ice_lut_ANG.nc"
        ds = xr.load_dataset(disort_path)
        grid = [
            ds["sza"].values,
            ds["grain_radius"].values,
            ds["algae_conc"].values,
            ds["dust_conc"].values,
            ds["lwc"].values,
            ds["ice_thickness"].values,
        ]

        self.g_hdrf = VectorInterpolator(
            grid_input=grid, data_input=ds["hdrf"].values, version="mlg"
        )
        self.g_brdf = VectorInterpolator(
            grid_input=grid, data_input=ds["brdf"].values, version="mlg"
        )

        if self.wl is not None:
            self.n_wl = len(self.wl)

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""
        mu = self.init
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x."""

        # flat priors
        f = np.ones_like(self.scale) * 1e6
        Cov = np.diag(f)

        return Cov

    def fit_params(self, meas, L_atm, L_dir, L_dif, sphalb, geom):
        """Given a Radiance estimate, fit a state vector."""
        # xopt = least_squares(self.err_obj, self.init, jac='2-point', **self.LS_Params,
        #                     args=(meas, geom, L_atm, L_dir, L_dif, sphalb))

        return self.init

    def calc_rfl(self, x_surface, geom, L_down_dir=None, L_down_dif=None):
        """
        Returns BRDF and HDRF


        """

        # catch potential invalid values before LUT
        if x_surface[0] >= 1.0:
            x_surface[0] = 1.0
        if x_surface[0] <= 0.01:
            x_surface[0] = 0.01
        if x_surface[1] <= 10.0:
            x_surface[1] = 10.0
        if x_surface[1] >= 25000.0:
            x_surface[1] = 25000.0
        if x_surface[2] >= 6e4:
            x_surface[2] = 6e4
        if x_surface[2] <= 0.0:
            x_surface[2] = 0.0
        if x_surface[3] >= 1000.0:
            x_surface[3] = 1000.0
        if x_surface[3] <= 0.0:
            x_surface[3] = 0.0
        if x_surface[4] >= 50.0:
            x_surface[4] = 50.0
        if x_surface[4] <= 0.0:
            x_surface[4] = 0.0

        # correct for RAA way DISORT is expecting it.
        # disort_raa = 180 - raa
        ice_dir_dir = self.g_brdf(
            np.array(
                [
                    np.degrees(np.arccos(x_surface[0])),
                    x_surface[1],
                    x_surface[2],
                    x_surface[3],
                    x_surface[4],
                    4.5,
                ]
            )
        )

        ice_dif_dir = self.g_hdrf(
            np.array(
                [
                    np.degrees(np.arccos(x_surface[0])),
                    x_surface[1],
                    x_surface[2],
                    x_surface[3],
                    x_surface[4],
                    4.5,
                ]
            )
        )

        # geom.snow_ref = ice_dif_dir
        return ice_dir_dir, ice_dif_dir

    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""

        _, lamb = self.calc_rfl(x_surface, geom)

        return lamb

    def drdn_dsurface(self, x_surface, geom, L_down_dir=None, L_down_dif=None):
        """Derivative of radiance with respect to
        full surface vector . NOTE: assumes very simple finite diff.
        This is computed once for Seps matrix for initial condition.
        """

        # first the radiance at the current state vector
        rho_dir, rho_dif = self.calc_rfl(x_surface, geom)
        svf = geom.svf
        cosi = x_surface[0]
        rdn = L_down_dir * rho_dir * cosi + L_down_dif * rho_dif * svf

        # perturb each element of the surface state vector (finite difference)
        drdn_dsurface = []

        x_surfaces_perturb = x_surface + np.eye(len(x_surface)) * eps

        for x_surface_perturb in x_surfaces_perturb:
            rho_dir_p, rho_dif_p = self.calc_rfl(x_surface_perturb, geom)

            # get new angles
            cosi_perturb = x_surface_perturb[0]

            rdn_perturb = (rho_dir_p * L_down_dir * cosi_perturb) + (
                rho_dif_p * L_down_dif * svf
            )

            drdn_dsurface.append((rdn_perturb - rdn) / eps)

        drdn_dsurface = np.array(drdn_dsurface).T

        return drdn_dsurface

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface."""

        dLs = np.zeros((self.n_wl, self.n_state), dtype=float)

        return dLs

    def summarize(self, x_surface, geom):
        return "cosi: %5.3f, grain-radius: %5.3f, liquid-water: %5.3f, dust: %5.3f" % (
            x_surface[0] + x_surface[1],
            x_surface[2],
            x_surface[3],
            x_surface[4],
        )
