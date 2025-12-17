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
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

import logging
import os
from datetime import datetime

import numpy as np


class Geometry:
    """The geometry of the observation, all we need to calculate sensor,
    surface, and solar positions."""

    def __init__(
        self,
        obs: np.array = None,
        loc: np.array = None,
        dt: datetime = None,
        esd: np.array = None,
        svf: np.array = None,
        shadow: np.array = None,
        slope: float = None,
        modtran_adjustments: tuple = (None, None),
        endmember_data: tuple = (None, None, None, None),   
        bkg_terms: tuple = (None, None)    
    ):
        # Set some benign defaults...
        self.observer_zenith = (
            0  # REVIEW: pytest/test_geometry asserts 0, change to None?
        )
        self.observer_azimuth = (
            0  # REVIEW: pytest/test_geometry asserts 0, change to None?
        )
        self.solar_zenith = None
        self.slope = None
        self.solar_azimuth = None
        self.observer_altitude_km = None
        self.surface_elevation_km = None
        self.earth_sun_distance = None
        self.esd_factor = 1.

        # set for initial write?
        self.a_total = 0.0
        self.a_direct = 0.0
        self.a_diffuse = 0.0

        if esd is None:
            logging.warning(
                "Earth sun distance not provided. Proceeding without might cause some inaccuracies down the line"
            )
            esd = np.ones((366, 2))
            esd[:, 0] = np.arange(1, 367, 1)
        self.earth_sun_distance_reference = esd

        self.svf = svf

        # check on svf
        if self.svf<0.0 or self.svf>1.0:
            self.svf=1.0 # assume nan data and assume safe assumption of 1.

        self.cos_i = None

        self.endmember_data = endmember_data

        if shadow is not None:
            self.shadow = shadow

        if slope is not None:
            self.slope = slope

        # allow for K to come from 2point method.
        self.K = None
        self.MAPE = None

        # unpack endmember data
        # user selected for now for background.
        self.pv = self.endmember_data[0]
        self.npv = self.endmember_data[1]
        self.soil = self.endmember_data[2]
        self.surface_StartWL = self.endmember_data[3]

        # The 'obs' object is observation metadata that follows a historical
        # AVIRIS-NG format.  It arrives to our initializer in the form of
        # a list-like object...
        if obs is not None:
            self.path_length_km = obs[0] / 1000
            self.observer_azimuth = obs[1]  # 0 to 360 clockwise from N
            self.observer_zenith = obs[2]  # 0 to 90 from zenith
            self.solar_azimuth = obs[3]  # 0 to 360 clockwise from N
            self.solar_zenith = obs[4]  # 0 to 90 from zenith
            if np.isnan(self.slope): # slope data not given, take from loc file
                self.slope = obs[6] # slope in degrees
            self.aspect = obs[7] # aspect in degrees
            self.cos_i = obs[8]  # cosine of eSZA
            # calculate relative to-sun azimuth
            delta_phi = np.abs(self.solar_azimuth - self.observer_azimuth)
            self.relative_azimuth = np.minimum(delta_phi, 360 - delta_phi)  # 0 to 180
        
        # The 'loc' object is a list-like object that optionally contains
        # latitude and longitude information about the surface being
        # observed.
        self.latitude = None
        self.longitude = None
        if loc is not None:
            self.surface_elevation_km = loc[2] / 1000.0
            self.latitude = loc[1]  # Northing
            self.longitude = loc[0]  # Westing
            if self.longitude < 0:
                self.longitude = 360.0 - self.longitude

        if loc is not None and obs is not None:
            self.observer_altitude_km = (
                self.surface_elevation_km
                + self.path_length_km * np.cos(np.deg2rad(self.observer_zenith))
            )

        if dt is not None:
            self.esd_factor = 1.

        # apply modtran adjustment  here
        # BW
        if modtran_adjustments != (None, None):
            self.interp_h2o_upper_bound, self.interp_aot_lower_bound = modtran_adjustments[0], modtran_adjustments[1]
            self.h2o_upper_bound = self.interp_h2o_upper_bound(self.surface_elevation_km)
            self.aot_lower_bound = self.interp_aot_lower_bound(self.surface_elevation_km)
        
        # Background terms
        self.rho_e = bkg_terms[0] # assuming 1km
        self.rho_terrain = bkg_terms[1] # assuming 0.5km


    def get_esd_factor(self, date_time: datetime):
        """Get distance ratio from sun based on time of year, relative to day 1
        Args:
            date_time: datetime to search

        Returns:
            float: ratio of earth sun distnace based on datetime.
        """

        return float(1)

    def check_coszen_and_cos_i(self, coszen):
        coszen = np.cos(np.deg2rad(self.solar_zenith)) if np.isnan(coszen) else coszen

        # Local solar zenith angle as a function of surface slope and aspect
        cos_i = self.cos_i if self.cos_i is not None else coszen

        return coszen, cos_i
