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
import math
import xarray as xr
import pickle
import pandas as pd

from .surface_multicomp import MultiComponentSurface
from isofit.core.common import eps, resample_spectrum, load_wavelen, VectorInterpolator
from isofit.configs import Config
from isofit.data import env


class SnowSurface(MultiComponentSurface):
    """

    Snow model is radiative transfer model that combines optical properties 
    based on Mie theory and multiscattering calculations using the 
    multistream DIScrete Ordinate Radiative Transfer (DISORT) code.

    This method solves for topography, atmosphere, snow properties, fractional covers, and fractional types.

    Accurate background reflectances are assumed within the apply-oe pipeline of this branch. 


    """

    def __init__(self, full_config: Config):

        super().__init__(full_config)

        # entire state vector
        self.statevec_names = (['empty','cos_i', # topography (2)
                                'Grain_radius', 'Liquid_water', 'Dust', 'Algae', # snow properties (4)
                                'z_snow', 'z_pv', 'z_npv', 'z_soil',  # pixel surface fractional covers (4)
                                'veg_rank','npv_rank', 'soil_rank', #endmember parameters (3)
                                ])
        self.scale = [1., 1., 
                      1., 1., 1.,1.,
                      1., 1., 1., 1.,
                      1., 1.,1.,
                      ]
        self.init = [1., 0.5,
                     500., 0., 0.,0., 
                     0., 0., 0., 0., 
                     0., 0., 0.,
                     ]
        self.bounds = np.array([
            [-1.,1.],              # Sine aspect (BLANK)
            [0., 1.],              # Cosine aspect  (cos_i actually)
            [30., 1500.],          # grain radius
            [0., 25.],             # liquid water fraction
            [0.0, 4000.],          # dust concentration
            [0.0, 6e5],            # algae cells per mL          
            [-5., 5.],             # z snow
            [-5., 5.],             # z veg
            [-5., 5.],             # z npv
            [-5., 5.],             # z soil
            [-3., 3.],             # pv rank
            [-3., 3.],             # npv rank 
            [-3., 3.],             # soil  rank     
        ])

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

        disort_sensor = "ENMAP"
        disort_path = env.path("data", f"disort_snow_lut_{disort_sensor}.nc")
        ds = xr.load_dataset(disort_path)
        grid = [
            ds['sza'].values,
            ds['vza'].values,
            ds['raa'].values,
            ds['grain_radius'].values,
            ds['algae_conc'].values,
            ds['dust_conc'].values,
            ds['lwc'].values,
        ]

        self.g_hdrf = VectorInterpolator(
            grid_input=grid,
            data_input=ds['hdrf'].values,
            version="mlg"
        )

        self.g_brdf = VectorInterpolator(
            grid_input=grid,
            data_input=ds['brdf'].values,
            version="mlg"
        )
        
        self.g_alb = VectorInterpolator(
            grid_input=grid, data_input=ds["a_diff"].values, version="mlg"
        )



        # Load in Endmembers data
        with open(env.path("data", f"pv_{disort_sensor}.pkl"), 'rb') as f:
            self.pv = pickle.load(f)
        with open(env.path("data", f"npv_{disort_sensor}.pkl"), 'rb') as f:
            self.npv = pickle.load(f)       
        with open(env.path("data", f"soil_{disort_sensor}.pkl"), 'rb') as f:
            self.soil = pickle.load(f)    

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
        """Given a Radiance estimate, fit a state vector.
        
        """
        #xopt = least_squares(self.err_obj, self.init, jac='2-point', **self.LS_Params,
        #                     args=(meas, geom, L_atm, L_dir, L_dif, sphalb))

        return self.init


    def calc_rfl(self, x_surface, geom, L_down_dir=None, L_down_dif=None):
        """
        Returns BRDF and HDRF
        
        
        """

        # catch potential invalid values before LUT
        if x_surface[0] >= 1.0:
            x_surface[0] = 1.0
        if x_surface[1] >= 1.0:
            x_surface[1] = 1.0
        if x_surface[0] <= -1.0:
            x_surface[0] = -1.0
        if x_surface[1] <= 0.0:
            x_surface[1] = 1e-6
        if x_surface[2] <= 30.0:
            x_surface[2] = 30.0
        if x_surface[2] >= 1500.0:
            x_surface[2] = 1500.0
        if x_surface[3] >= 25.0:
            x_surface[3] = 25.0
        if x_surface[3] <= 0.0:
            x_surface[3] = 0.0
        if x_surface[4] >= 4000.0:
            x_surface[4] = 4000.0
        if x_surface[4] <= 0.0:
            x_surface[4] = 0.0
        if x_surface[5] >= 6e5:
            x_surface[5] = 6e5
        if x_surface[5] <= 0.0:
            x_surface[5] = 0.0
            
        # calculate all relevant angles
        vza = geom.observer_zenith
        raa = geom.relative_azimuth
        vaa = geom.observer_azimuth
        sza = geom.solar_zenith
        saa = geom.solar_azimuth
        slope = geom.slope

        # EnMAP hack / if want to fully go to cosi only not aspect.
        cosv = np.cos(np.radians(vza))
        cosi = x_surface[1]
        cosi = max(0.06, min(cosi, 1.0)) 
        
        #cosi, cosv = self.calc_new_angles(x_surface,
        #                                  sza, vza, 
        #                                  saa, vaa,
        #                                  slope, geom)
        
        # correct for RAA way DISORT is expecting it.
        disort_raa = 180 - raa
        snow_dir_dir = self.g_brdf(np.array([np.degrees(np.arccos(cosi)), np.degrees(np.arccos(cosv)), 
                                             disort_raa, x_surface[2], x_surface[5], x_surface[4], x_surface[3]]))
        
        snow_dif_dir = self.g_hdrf(np.array([np.degrees(np.arccos(cosi)), np.degrees(np.arccos(cosv)), 
                                             disort_raa, x_surface[2], x_surface[5], x_surface[4], x_surface[3]]))

        # Endmembers
        rho_pv = self.reconstruct_reflectance(np.array([x_surface[10]]), self.pv)
        rho_npv = self.reconstruct_reflectance(np.array([x_surface[11]]), self.npv)
        rho_soil = self.reconstruct_reflectance(np.array([x_surface[12]]), self.soil)

        # apply soft max condition for sum-to-1 fractional covers condition...
        z = np.array([x_surface[6], x_surface[7], x_surface[8], x_surface[9]])
        f = np.exp(z) / np.sum(np.exp(z))
        rho_dir_dir = snow_dir_dir*f[0] + f[1]*rho_pv + f[2]*rho_npv + f[3]*rho_soil
        rho_dif_dir = snow_dif_dir*f[0] + f[1]*rho_pv + f[2]*rho_npv + f[3]*rho_soil



        return rho_dir_dir, rho_dif_dir



    def calc_snow_albedo(self, x_surface, geom, L_down_dir=None, L_down_dif=None):
        """
        Returns broadband snow albedos
        """
        # catch potential invalid values before LUT
        if x_surface[0] >= 1.0:
            x_surface[0] = 1.0
        if x_surface[1] >= 1.0:
            x_surface[1] = 1.0
        if x_surface[0] <= -1.0:
            x_surface[0] = -1.0
        if x_surface[1] <= -1.0:
            x_surface[1] = -1.0
        if x_surface[2] <= 30.0:
            x_surface[2] = 30.0
        if x_surface[2] >= 1500.0:
            x_surface[2] = 1500.0
        if x_surface[3] >= 25.0:
            x_surface[3] = 25.0
        if x_surface[3] <= 0.0:
            x_surface[3] = 0.0
        if x_surface[4] >= 4000.0:
            x_surface[4] = 4000.0
        if x_surface[4] <= 0.0:
            x_surface[4] = 0.0
        if x_surface[5] >= 6e5:
            x_surface[5] = 6e5
        if x_surface[5] <= 0.0:
            x_surface[5] = 0.0

        # calculate all relevant angles
        vza = geom.observer_zenith
        raa = geom.relative_azimuth
        vaa = geom.observer_azimuth
        sza = geom.solar_zenith
        saa = geom.solar_azimuth
        slope = geom.slope

        cosv = np.cos(np.radians(vza))
        cosi = x_surface[1]
        cosi = max(0.06, min(cosi, 1.0)) 
        
        #cosi, cosv = self.calc_new_angles(x_surface, sza, vza, saa, vaa, slope, geom)

        # correct for RAA way DISORT is expecting it.
        disort_raa = 180 - raa

        # TODO
        # temp try/except to work for both versions of the DISORT LUT, to be removed soon..
        # a_dir = self.g_ad(np.array([np.degrees(np.arccos(cosi)), np.degrees(np.arccos(cosv)),
        #                                    disort_raa, x_surface[2], x_surface[5], x_surface[4], x_surface[3]]))

        # a_dif = self.g_ah(np.array([np.degrees(np.arccos(cosi)), np.degrees(np.arccos(cosv)),
        #                                    disort_raa, x_surface[2], x_surface[5], x_surface[4], x_surface[3]]))

        # This is tmp until fully go to next DISORT version
        a_dir = a_dif = self.g_alb(
            np.array(
                [
                    np.degrees(np.arccos(cosi)),
                    np.degrees(np.arccos(cosv)),
                    disort_raa,
                    x_surface[2],
                    x_surface[5],
                    x_surface[4],
                    x_surface[3],
                ]
            )
        )

        # Compute diffuse fraction
        L_total = L_down_dif + L_down_dir
        k = L_down_dif / (L_down_dif + L_down_dir + 1e-12)
        alb_blue = (1 - k) * a_dir + k * a_dif

        # integrate (numpy changed trapz it seems in 2.0?)
        total_albedo = np.trapezoid(alb_blue * L_total, dx=1) / np.trapezoid(
            L_total + 1e-12, dx=1
        )
        direct_albedo = np.trapezoid(a_dir * L_down_dir, dx=1) / np.trapezoid(
            L_down_dir + 1e-12, dx=1
        )
        diffuse_albedo = np.trapezoid(a_dif * L_down_dif, dx=1) / np.trapezoid(
            L_down_dif + 1e-12, dx=1
        )

        return total_albedo, direct_albedo, diffuse_albedo


    def reconstruct_reflectance(self, x_surface, mu_V_tuple):
        """
        TODO
        """
        mu=mu_V_tuple[0]
        V=mu_V_tuple[1]
        rfl = V.T @ x_surface + mu
        rfl[rfl<0] = 0.01

        if np.isnan(rfl).any():
            rfl = np.full_like(rfl, fill_value=200.0)

        return rfl


    def calc_new_angles(self, x_surface, sza, vza, saa, vaa, slope, geom):
        """
        Calculates cosi and cosv using aspect solved for in the inversion
        """
        # solve for aspect via sin(aspect) and cos(aspect)
        aspect = np.degrees(math.atan2(x_surface[0], x_surface[1]))
        if (aspect < 0.0):
            aspect += 360.0
        aspect = np.radians(aspect)

        cosi = (np.sin(np.radians(sza)) * np.sin(np.radians(slope)) *
                np.cos(np.radians(saa) - aspect) +
                np.cos(np.radians(sza)) * np.cos(np.radians(slope)))

        cosv = (np.sin(np.radians(vza)) * np.sin(np.radians(slope)) *
                np.cos(np.radians(vaa) - aspect) +
                np.cos(np.radians(vza)) * np.cos(np.radians(slope)))


        # check if pixel is cast in shadow from other terrain, fall back to static aspect data.
        if geom.shadow == 0:
            cosi=0.0
            cosv = (np.sin(np.radians(vza)) * np.sin(np.radians(slope)) *
                np.cos(np.radians(vaa) - geom.aspect) +
                np.cos(np.radians(vza)) * np.cos(np.radians(slope)))


        # capped at 0.06 bc disort lut max zenith was 87 degrees
        cosi = max(0.06, min(cosi, 1.0)) 
        cosv = max(0.06, min(cosv, 1.0)) 


        return cosi, cosv



    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""

        _ , lamb = self.calc_rfl(x_surface, geom)

        return lamb



    def drdn_dsurface(self, x_surface, geom, L_down_dir=None, L_down_dif=None):
        """Derivative of radiance with respect to
        full surface vector . NOTE: assumes very simple finite diff. 
        This is computed once for Seps matrix for initial condition. 
        """

        # first the radiance at the current state vector
        rho_dir, rho_dif = self.calc_rfl(x_surface, geom)
        svf = geom.svf

        cosi=x_surface[1]
        #cosi,_ = self.calc_new_angles(x_surface, 
        #                                        geom.solar_zenith, 
        #                                        geom.observer_zenith,
        #                                        geom.solar_azimuth, 
        #                                        geom.observer_azimuth, geom.slope)

        rdn = L_down_dir*rho_dir*cosi + L_down_dif*rho_dif*svf

        # perturb each element of the surface state vector (finite difference)
        drdn_dsurface = []

        x_surfaces_perturb = x_surface + np.eye(len(x_surface)) * eps

        for x_surface_perturb in x_surfaces_perturb:
            rho_dir_p,rho_dif_p = self.calc_rfl(x_surface_perturb, geom)

            # get new angles
            cosi_perturb = x_surface_perturb[1]
            #cosi_perturb,_ = self.calc_new_angles(x_surface_perturb, 
            #                                      geom.solar_zenith, 
            #                                      geom.observer_zenith,
            #                                      geom.solar_azimuth, 
            #                                      geom.observer_azimuth, geom.slope)

            rdn_perturb = ((rho_dir_p * L_down_dir * cosi_perturb) + 
                           (rho_dif_p * L_down_dif * svf))
            
            drdn_dsurface.append((rdn_perturb - rdn) / eps)

        drdn_dsurface = np.array(drdn_dsurface).T

        return drdn_dsurface
    



    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface."""

        dLs = np.zeros((self.n_wl, self.n_state), dtype=float)

        return dLs
    


    def summarize(self, x_surface, geom):
        return 'cosA: %5.3f, grain-radius: %5.3f, liquid-water: %5.3f, dust: %5.3f' % (
          +  x_surface[1], x_surface[2], x_surface[3], x_surface[4])