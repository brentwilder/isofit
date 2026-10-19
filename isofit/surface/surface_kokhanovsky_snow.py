#! /usr/bin/env python3
#
#  Copyright 2020 California Institute of Technology
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
from __future__ import annotations

import numpy as np
import pickle
from scipy import interpolate

from isofit.core.common import svd_inv_sqrt
from isofit.core.units import micron_to_nm
from isofit.surface.surface import Surface

from isofit.data import env



class KokhanovskySnowSurface(Surface):
    """TODO
    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)
        config = full_config.forward_model.surface
        self.terrain_style = config.terrain_style
        self.max_slope = config.max_slope
        self.solve_mixed_pixel = True


        self.statevec_names = (["COS_I", # topography 
                                "GRAIN_RADIUS", "LWC", "DUST_CONC", "ALGAE_CONC", # snow properties (4)
                                "FRACTIONAL_DATA", "FRACTIONAL_PV", "FRACTIONAL_NPV", "FRACTIONAL_SOIL",  # pixel surface fractional covers (4)
                                "PV_LOWRANK","NPV_LOWRANK", "SOIL_LOWRANK", #endmember parameters (3)
                                ])
        self.scale = [1., 
                      1., 1., 1.,1.,
                      1., 1., 1., 1.,
                      1., 1.,1.,
                      ]
        self.init = [0.5,
                     500., 0., 0.,0., 
                     0., 0., 0., 0., 
                     0., 0., 0.,
                     ]
        self.bounds = np.array([
            [1e-8, 1.],            # cosi
            [30., 5000.],          # grain radius
            [0., 35.],             # liquid water fraction
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


        # First, stash important lengths and indices from the LUT
        self.n_wl = len(self.wl)
        self.n_state = len(self.statevec_names)
        self.idx_lamb = np.arange(self.n_wl)
        self.idx_surface = np.arange(len(self.statevec_names))
        
        # Stash dynamic indices for low-rank parameters
        self.idx_pv_lr = self.statevec_names.index("PV_LOWRANK") if "PV_LOWRANK" in self.statevec_names else None
        self.idx_npv_lr = self.statevec_names.index("NPV_LOWRANK") if "NPV_LOWRANK" in self.statevec_names else None
        self.idx_soil_lr = self.statevec_names.index("SOIL_LOWRANK") if "SOIL_LOWRANK" in self.statevec_names else None

        # Then, assign the priors and optimizaton parameters from the surface config
        self.init, self.bounds, self.scale, self.mean, self.sigma = [], [], [], [], []

        for name in self.statevec_names:
            state_config = getattr(config.statevector, name)
            self.init.append(float(state_config.get("init")))
            bound_val = state_config.get("bounds")
            self.bounds.append([float(bound_val[0]), float(bound_val[1])])
            self.scale.append(float(state_config.get("scale")))
            self.mean.append(float(state_config.get("prior_mean")))
            self.sigma.append(float(state_config.get("prior_sigma")))

        self.init = np.array(self.init, dtype=float)
        self.bounds = np.array(self.bounds, dtype=float)
        self.scale = np.array(self.scale, dtype=float)
        self.mean = np.array(self.mean, dtype=float)
        self.sigma = np.array(self.sigma, dtype=float)

        # Cache some important computations
        # NOTE for now this assumes no off diagonal elements
        Cov = np.diag(self.sigma**2)
        Cov_normalized = Cov / np.mean(np.diag(Cov))
        self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized = svd_inv_sqrt(
            Cov_normalized
        )

        # NOTE LUTSurface currently is not compatible with analytical line
        self.analytical_iv_idx = np.arange(self.n_state)

        if self.use_background_rfl:
            self.drdn_drfl = self.drdn_drfl_heterogeneous_bkg
        else:
            self.drdn_drfl = self.drdn_drfl_homogeneous_bkg

        # BW 
        # Specific to isosnow (Snow surface model)
        # We go with a set of low-rank models (1param) for each em (pv, npv, soil)
        # This allows it to shift slightly but not too much
        # Specifically for this is that algae is less likely to be used to absorb in the 
        # vis for bad fit of PV. It will fluctuate this term instead.

        if len(self.wl) == 425:
            disort_sensor = "ANG"
        if len(self.wl) == 285:
            disort_sensor = "EMIT"

        with open(env.path("data", f"pv_{disort_sensor}.pkl"), "rb") as f:
            self.pv = pickle.load(f)
        with open(env.path("data", f"npv_{disort_sensor}.pkl"), "rb") as f:
            self.npv = pickle.load(f)       
        with open(env.path("data", f"soil_{disort_sensor}.pkl"), "rb") as f:
            self.soil = pickle.load(f)    

        self.idx_em_rfls = [
            self.statevec_names.index("FRACTIONAL_DATA"),
            self.statevec_names.index("FRACTIONAL_PV"),
            self.statevec_names.index("FRACTIONAL_NPV"),
            self.statevec_names.index("FRACTIONAL_SOIL"),
        ]

        self.cos_i_idx = 0

        ##################################################################
        ##################################################################
        # Caching ART specific items below  . . . . 
        ##################################################################
        ##################################################################

        self.model_wl = self.wl*1e-9

        # Get ice imag and real indices
        ice_data = np.load(env.path("data", "rfidx_ice.npz"))
        ice_wl = micron_to_nm(ice_data["wvl"])
        self.k_ice = interpolate.interp1d(ice_wl, ice_data["im_Pic16"], kind="slinear")(self.wl)
        self.r_ice = interpolate.interp1d(ice_wl, ice_data["re_Pic16"], kind="slinear")(self.wl)

        # Refractive index of water
        # options are: 'k_22c', 'k_minus8c', 'k_minus25c', 'k_minus7c', 'k_25c_H', 'k_20c', 'k_25c_S
        k_wi = np.genfromtxt(env.path("data", "iop", "k_liquid_water_ice.csv"), 
                             delimiter=",", names=True, encoding="utf-8-sig")
        k_wi_idx = ~np.isnan(k_wi["wl_20c"]) & ~np.isnan(k_wi["k_20c"])
        self.k_water = np.interp(x=self.wl, xp=k_wi["wl_20c"][k_wi_idx], fp=k_wi["k_20c"][k_wi_idx])

        # Algae constants
        # from Dauchet et al. [2015]
        self.abs_xsc = snow_algae_chevrollier_2023(self.wl)
        self.x_w = 0.78 # water fraction
        self.cell_radius = 7 # micrometer
        self.cell_volume = (4/3 * np.pi * self.cell_radius**3) * 1e-18 #micro to m3
        self.density_algae = 1400
        self.density_ice = 917.0
        self.k_alg = self.x_w * self.k_water + self.abs_xsc * self.model_wl / (np.pi * 4) / self.cell_volume
        self.mae_alg = (8 / np.pi) / self.model_wl * self.k_alg / self.density_algae 
        self.algae_conc_sf = (1e6 * self.cell_volume * (self.density_algae / self.density_ice))

        # Mineral dust constants
        self.dust_density = 2.65 # g cm-3
        self.aae = 4.1 # fixed based on PM2_5 Libya in Caponi 2017
        self.mae400 = 110 # fixed based on PM2_5 Libya in Caponi 2017
        self.mae_dust = self.mae400 * (self.model_wl / 400e-9)**(-self.aae)
        self.dust_conc_sf = 1e-6 * ((self.density_ice / 1000) / (self.dust_density * (8 / np.pi)))

        # Custom phase and r0 params
        # NOTE: These are linked to the g0 and B0 below
        # Updated params for different shape can be estimated by fitting to DISORT across principal plane
        self.p_0 = -1.90946652e+00
        self.p_1 = 1.58752445e-02
        self.p_2 = 6.83972969e+00
        self.p_3 = 5.42025234e-03

        self.r0_0 = -1.11176131e+00
        self.r0_1 = -7.12680819e-01
        self.r0_2 = 9.02713485e+00

        self.g0 = 0.82 # Middle range of Robledano et al. 2023
        self.B0 = self.r_ice**2 #B=n2 from Robledano et al. 2023

        # original relation for reference
        #p = 11.1 * np.exp(-0.087 * geom.theta) + 1.1 * np.exp(-0.014 * geom.theta)
        #r0 = (1.247 + 1.186 * (cos_i + geom.cos_v) + 5.157 * cos_i * geom.cos_v + p) / 4.0 / (cos_i + geom.cos_v)

        # Not used here in the reflectance model... but for the albedo output we assume similar relation as TARTES, 
        # we assume mu_s = ((7 / 3 - 1) / 2) for the white sky albedo term.
        # and black sky albedo is just the planar albedo with the appropriate cosi

        ##################################################################
        ##################################################################
        ##################################################################
        ##################################################################


    def update_heuristic_prior_means(self, x_surface, geom):
        return x_surface

    def xa(self, x_surface, geom):
        """Mean of prior distribution."""

        return self.init

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x."""

        variance = pow(self.sigma, 2)
        Sa_unnormalized = np.diag(variance)

        return Sa_unnormalized, self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized

    def Sb(self):
        """Uncertainty due to unmodeled variables."""
        return np.diagflat(np.power(self.bval, 2))

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""
        return self.init

    def calc_rfl(self, x_surface, geom):
        """TODO
        """
        rho_dir_dir = self.aart(x_surface, geom, diffuse=False)
        rho_dif_dir = self.aart(x_surface, geom, diffuse=True)

        # Apply softmax for fractional components
        f = self.softmax(np.array(x_surface[self.idx_em_rfls]))

        # Reconstruct endmembers
        rho_pv = self.reconstruct_reflectance(np.array([x_surface[self.idx_pv_lr]]), self.pv)
        rho_npv = self.reconstruct_reflectance(np.array([x_surface[self.idx_npv_lr]]), self.npv)
        rho_soil = self.reconstruct_reflectance(np.array([x_surface[self.idx_soil_lr]]), self.soil)
        em_matrix = np.column_stack([rho_pv, rho_npv, rho_soil])

        # Apply linear mixture
        rho_dir_dir = rho_dir_dir * f[0] + np.dot(em_matrix, f[1:])
        rho_dif_dir = rho_dif_dir * f[0] + np.dot(em_matrix, f[1:])

        return rho_dir_dir, rho_dif_dir



    def aart(self, x_surface, geom, diffuse):
        """
        This is a custom implementation, with phase function and r0 that were fit to DISORT simulations.

        We assumed a similar grain shape as described in Robledano et al. 2023, with:

        g0 = 0.82
        B0 = n^2

        """
        # Ensure optimization stays in bounds
        x_surface = [max(float(b[0]), min(float(val), float(b[1]))) for val, b in zip(x_surface, self.bounds)]

        # NOTE For now assuming order like this
        cos_i, grain_radius, lwc, dust_conc, algae_conc = x_surface[:5]

        # Update scattering angle
        # NOTE 180 is scattering back, sun behind sensor
        # 0 is sensor facing sun
        sini = np.sin(np.arccos(cos_i))
        self.theta = np.degrees(np.arccos(-cos_i*geom.cos_v + sini*geom.sinv*geom.cos_raa)) 

        #  SSA to diameter
        #d = 6 / (917*ssa)

        # convert to how model expects
        d = grain_radius * 2 * 1e-6
        dust_conc = dust_conc * self.dust_conc_sf
        lwc = lwc / 100.0
        algae_conc = algae_conc * self.algae_conc_sf
        
        # Composite refractive index
        k_eff = (1-lwc)*self.k_ice + lwc*self.k_water

        # r0 - semi-infinite non-absorbing snow layer
        p = self.p_0 * np.exp(-self.p_1 * geom.theta) + self.p_2 * np.exp(-self.p_3 * geom.theta)
        r0 = (self.r0_0 + self.r0_1 * (cos_i + geom.cos_v) + self.r0_2 * cos_i * geom.cos_v + p) / 4.0 / (cos_i + geom.cos_v)

        # Compute r - spherical albedo (considering different polutants)
        beta1 = 2.0 / d * self.mae_dust * dust_conc
        beta2 = 2.0 / d * self.mae_alg * algae_conc
        gamma_i = (4 * np.pi * k_eff) / self.model_wl
        l =  d / ((9*(1-self.g0)) / (16*self.B0))
        r = np.exp(-np.sqrt(((gamma_i + (beta1+beta2)) * l)))

        # Escape function - f
        u1 = 0.6*cos_i + 1. / 3. + np.sqrt(cos_i) / 3.
        u2 = 0.6*geom.cos_v + 1. / 3. + np.sqrt(geom.cos_v) / 3.
        f = (u1 * u2) / r0

        # Directional Reflectance (rho_s), r0 infinit factor, r spherical albedo, and escape function
        rho_s = r0 * r**f

        # Diffuse-direct
        if diffuse:
            rho_s = r**u2

        return rho_s


    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""
        _ , rho_dif = self.calc_rfl(x_surface, geom)
        return rho_dif

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


    def softmax(self, z):
        "Used to maintain sum-to-1 condition and positive fractional covers"
        return np.exp(z) / np.sum(np.exp(z))

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        return self.dlamb_dsurface(x_surface, geom)

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian rfl with respect to state vector."""

        x_surface = np.array(x_surface, dtype=float)
        dlamb = np.zeros((self.n_wl, self.n_state))

        _, grain_radius, lwc, dust_conc, algae_conc = x_surface[:5]

        d = grain_radius * 2 * 1e-6
        dust_conc = dust_conc * self.dust_conc_sf
        lwc = lwc / 100.0
        algae_conc = algae_conc * self.algae_conc_sf
        
        k_eff = (1 - lwc) * self.k_ice + lwc * self.k_water

        beta1 = 2.0 / d * self.mae_dust * dust_conc
        beta2 = 2.0 / d * self.mae_alg * algae_conc
        gamma_i = (4 * np.pi * k_eff) / self.model_wl
        
        C_l = (9 * (1 - self.g0)) / (16 * self.B0)
        l = d / C_l

        X = gamma_i + beta1 + beta2
        S = np.sqrt(X * l)
        
        u2 = 0.6 * geom.cos_v + 1.0 / 3.0 + np.sqrt(geom.cos_v) / 3.0
        
        rho_snow = np.exp(-u2 * S)

        # Common term for the partials below
        drho_dinner = -u2 * rho_snow / (2.0 * S + 1e-12)

        # Grain radius
        dd_dgr = 2 * 1e-6
        dX_dd = -(beta1 + beta2) / d
        dl_dd = 1.0 / C_l
        dinner_dd = dX_dd * l + X * dl_dd
        drho_dgr = drho_dinner * dinner_dd * dd_dgr

        # Liquid water
        dk_eff_dlwc = (self.k_water - self.k_ice) * 0.01
        dgamma_dlwc = (4 * np.pi * dk_eff_dlwc) / self.model_wl
        dinner_dlwc = dgamma_dlwc * l
        drho_dlwc = drho_dinner * dinner_dlwc

        # Dust
        dbeta1_ddust = 2.0 / d * self.mae_dust * self.dust_conc_sf
        dinner_ddust = dbeta1_ddust * l
        drho_ddust = drho_dinner * dinner_ddust

        # Algae
        dbeta2_dalgae = 2.0 / d * self.mae_alg * self.algae_conc_sf
        dinner_dalgae = dbeta2_dalgae * l
        drho_dalgae = drho_dinner * dinner_dalgae

        # Fractional covers
        f = np.exp(x_surface[self.idx_em_rfls]) / np.sum(np.exp(x_surface[self.idx_em_rfls]))
        df_dz = np.diag(f) - np.outer(f, f)

        rho_pv = self.reconstruct_reflectance(np.array([x_surface[self.idx_pv_lr]]), self.pv)
        rho_npv = self.reconstruct_reflectance(np.array([x_surface[self.idx_npv_lr]]), self.npv)
        rho_soil = self.reconstruct_reflectance(np.array([x_surface[self.idx_soil_lr]]), self.soil)
        em_matrix = np.column_stack([rho_pv, rho_npv, rho_soil])

        dlamb[:, 0] = 0.0 
        dlamb[:, 1] = drho_dgr * f[0]    
        dlamb[:, 2] = drho_dlwc * f[0]    
        dlamb[:, 3] = drho_ddust * f[0] 
        dlamb[:, 4] = drho_dalgae * f[0] 

        components = np.column_stack([rho_snow, em_matrix])
        for idx_pos, state_idx in enumerate(self.idx_em_rfls):
            dlamb[:, state_idx] = np.dot(components, df_dz[:, idx_pos])

        if self.idx_pv_lr is not None:
            V_pv = self.pv[1].ravel() if self.pv[1].ndim > 1 else self.pv[1]
            dlamb[:, self.idx_pv_lr] = V_pv * f[1]

        if self.idx_npv_lr is not None:
            V_npv = self.npv[1].ravel() if self.npv[1].ndim > 1 else self.npv[1]
            dlamb[:, self.idx_npv_lr] = V_npv * f[2]

        if self.idx_soil_lr is not None:
            V_soil = self.soil[1].ravel() if self.soil[1].ndim > 1 else self.soil[1]
            dlamb[:, self.idx_soil_lr] = V_soil * f[3]

        return dlamb


    def drdn_drfl_heterogeneous_bkg(
        self, L_tot, s_alb, rho_dif_dir, L_dir_dir, L_dir_dif, L_dif_dir, L_dif_dif
    ):
        """Partial derivative of radiance with respect to
        surface reflectance, treating dir-dif and dif-dif as constants."""
        return L_dir_dir + (L_dif_dir / (1.0 - s_alb * rho_dif_dir))

    def drdn_drfl_homogeneous_bkg(
        self, L_tot, s_alb, rho_dif_dir, L_dir_dir, L_dir_dif, L_dif_dir, L_dif_dif
    ):
        """Partial derivative of radiance with respect to
        surface reflectance (dir-dir = dif-dir = dir-dif = dif-dif)."""
        return L_tot / ((1.0 - s_alb * rho_dif_dir) ** 2)

    def calc_Ls(self, x_surface, geom):
        """Emission of surface, as a radiance."""

        return np.zeros(self.n_wl, dtype=float)

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to
        state vector, calculated at x_surface."""

        dLs = np.zeros((self.n_wl, self.n_state), dtype=float)

        return dLs

    def drdn_dLs(self, t_total_up):
        """Partial derivative of radiance with respect to
        surface emission"""

        return t_total_up

    def dcosi_dsurface(
        self,
        x_surface,
        geom,
        rho_dir_dir,
        rho_dif_dir,
        r,
        L_dir_dir,
        L_dif_dir,
    ):
        """Partial derivative of TOA radiance with respect to cosi."""

        cosi = max(x_surface[self.cos_i_idx], 1e-6)
        b = 1.0
        t_down_dir = r["transm_down_dir"]

        # Hay's model
        A = (b * t_down_dir) / geom.coszen
        B = (1.0 - b * t_down_dir) * geom.skyview_factor
        denominator = (A * cosi) + B

        # Direct and diffuse components
        d_rdn_dir = (L_dir_dir / cosi) * rho_dir_dir
        d_rdn_dif = (L_dif_dir * (A / denominator)) * rho_dif_dir

        return d_rdn_dir + d_rdn_dif

    def drdn_dsurface(
        self,
        x_surface,
        geom,
        r,
        rho_dif_dir,
        drfl_dsurface,
        dLs_dsurface,
        s_alb,
        t_total_up,
        L_tot,
        L_dir_dir=None,
        L_dir_dif=None,
        L_dif_dir=None,
        L_dif_dif=None,
    ):
        """Derivative of radiance with respect to
        full surface vector"""

        drdn_dLs = t_total_up

        drdn_dsurface = np.zeros(drfl_dsurface.shape)
        drdn_drfl = self.drdn_drfl(
            L_tot,
            s_alb,
            rho_dif_dir,
            L_dir_dir=L_dir_dir,
            L_dir_dif=L_dir_dif,
            L_dif_dir=L_dif_dir,
            L_dif_dif=L_dif_dif,
        )

        # Construct the output matrix:
        # Dimensions should be (len(RT.wl), len(x_surface))
        # which is correctly handled by the instrument resampling
        drdn_dsurface[:, : self.n_wl] = np.multiply(
            drdn_drfl[:, np.newaxis], drfl_dsurface[:, : self.n_wl]
        )

        # Cosi
        if self.cos_i_idx is not None:
            drdn_dsurface[:, self.cos_i_idx] += self.dcosi_dsurface(
                x_surface,
                geom,
                rho_dif_dir,
                rho_dif_dir,
                r,
                L_dir_dir,
                L_dif_dir,
            )

        # Get the derivative w.r.t. surface emission
        drdn_dLs = np.multiply(self.drdn_dLs(t_total_up)[:, np.newaxis], dLs_dsurface)

        return np.add(drdn_dsurface, drdn_dLs)

    def analytical_model(
        self,
        L_tot,
        geom,
        s_alb=None,
        L_dir_dir=None,
        L_dir_dif=None,
        L_dif_dir=None,
        L_dif_dif=None,
    ):
        """
        Linearization of the surface reflectance terms to use in the
        AOE inner loop (see Susiluoto, 2025). We set the quadratic
        spherical albedo term to a constant background, which
        simplifies the linearization
        background - s * rho_bg

        NOTE FOR SURFACE_LUT:
        To avoid confusion this does not output anything.
        """
        pass

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        if len(x_surface) < 1:
            return ""

        return "Surface: " + " ".join(
            [f"{n}: {v:5.4f}" for n, v in zip(self.statevec_names, x_surface)]
        )









########################################
# INTERPOLATED DATA FROM Chevrollier2023
########################################

# Interpolated from snow_algae_empirical_Chevrollier2023
algae_abs_wavelengths = np.array([
    340., 345., 350., 355., 360., 365., 370., 375., 380., 385., 390., 395., 400., 405.,
    410., 415., 420., 425., 430., 435., 440., 445., 450., 455., 460., 465., 470., 475.,
    480., 485., 490., 495., 500., 505., 510., 515., 520., 525., 530., 535., 540., 545.,
    550., 555., 560., 565., 570., 575., 580., 585., 590., 595., 600., 605., 610., 615.,
    620., 625., 630., 635., 640., 645., 650., 655., 660., 665., 670., 675., 680., 685.,
    690., 695., 700., 705., 710., 715., 720., 725., 730., 735., 740., 745., 750., 755.,
    760., 765., 770., 775., 780., 785., 790., 795., 800., 805., 810., 815., 820., 825.,
    830., 835., 840., 845., 850.])

algae_abs_data = np.array([
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77335262e-10, 1.77335262e-10,
    1.77335262e-10, 1.77335262e-10, 1.77268582e-10, 1.77201902e-10,
    1.72238472e-10, 1.67275043e-10, 1.54499561e-10, 1.41724079e-10,
    1.21583405e-10, 1.01442732e-10, 8.44938366e-11, 6.75449410e-11,
    5.86893966e-11, 4.98338523e-11, 4.52856453e-11, 4.07374384e-11,
    3.76331693e-11, 3.45289002e-11, 3.16364558e-11, 2.87440114e-11,
    2.91436459e-11, 2.95432803e-11, 3.14390655e-11, 3.33348508e-11,
    3.46587407e-11, 3.59826307e-11, 4.00603803e-11, 4.41381300e-11,
    4.01574201e-11, 3.61767102e-11, 2.48784552e-11, 1.35802002e-11,
    7.20517045e-12, 8.30140716e-13, 4.58160528e-13, 8.61803404e-14,
    8.76886425e-14, 8.91969445e-14, 8.59923354e-14, 8.27877263e-14,
    1.92606610e-14, -4.42664044e-14, -9.37278041e-14, -1.43189204e-13,
    -1.74431051e-13, -2.05672899e-13, -2.14995734e-13, -2.24318569e-13,
    -1.96513499e-13, -1.68708428e-13, -1.02131249e-13, -3.55540688e-14,
    -1.77770344e-14, -1.88221673e-27, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00])


def snow_algae_chevrollier_2023(wls):
    return interpolate.interp1d(algae_abs_wavelengths, algae_abs_data, kind="slinear", bounds_error=False, fill_value=0.0)(wls)