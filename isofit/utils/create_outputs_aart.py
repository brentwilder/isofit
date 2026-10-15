from os.path import abspath, join
import xarray as xr
import numpy as np
from scipy import interpolate
from spectral.io import envi

from isofit import ray
from isofit.core.fileio import initialize_output, write_bil_chunk
from isofit.core.common import VectorInterpolator, envi_header
from isofit.core.units import micron_to_nm
from isofit.surface.surface_kokhanovsky_snow import snow_algae_chevrollier_2023

from isofit.data import env

FSNOW_THRESHOLD = 0.0

B_R = 2.7

SNOW_BANDS = [
    "FSCA",
    "SNOW_ALBEDO_TOTAL",
    "SNOW_ALBEDO_DIRECT",
    "SNOW_ALBEDO_DIFFUSE",
    "SNOW_ALBEDO_VIS",
    "SNOW_ALBEDO_IR",
    "F_SNOW",
    "GRAIN_RADIUS",
    "LIQUID_WATER",
    "DUST_CONCENTRATION",
    "ALGAE_CONCENTRATION",
    "COS_I",
]

STATE_BANDS = [
    "ALGAE_CONC",
    "COS_I",
    "DUST_CONC",
    "FRACTIONAL_DATA",
    "FRACTIONAL_NPV",
    "FRACTIONAL_PV",
    "FRACTIONAL_SOIL",
    "GRAIN_RADIUS",
    "LWC",
    "NPV_LOWRANK",
    "PV_LOWRANK",
    "SOIL_LOWRANK",
    "AOT550",
    "H2OSTR",
]


def snow_model_outputs_aart(
    input_loc,
    input_obs,
    paths,
    veg_fraction_file,
    modtran_file,
    dayofyear, 
    n_cores,
    nodata_value=-9999,
):
    """TODO"""

    loc = envi.open(envi_header(input_loc), input_loc).open_memmap()
    rows, cols, _ = loc.shape
    del loc

    rdn_fname = paths.fid + "_rdn"
    outpath = abspath(join(paths.output_directory, rdn_fname.replace("_rdn", "_snow")))
    outpath_uncert = abspath(join(paths.output_directory, rdn_fname.replace("_rdn", "_snow_uncert")))

    for path in [outpath, outpath_uncert]:
        _ = initialize_output(
            output_metadata={
                "data type": 4,
                "file type": "ENVI Standard",
                "byte order": 0,
                "no data value": nodata_value,
                "lines": rows,
                "samples": cols,
                "interleave": "bil",
                "band names": SNOW_BANDS,
            },
            outpath=path,
            out_shape=(rows, len(SNOW_BANDS), cols),
            bands=f"{len(SNOW_BANDS)}",
            description="TODO",
        )

    m = xr.load_dataset(modtran_file)
    solar_irr_full = m.solar_irr.values
    raw_wl = m.wl.values

    transm_down_dir = m['transm_down_dir'].transpose('AOT550', 'H2OSTR', 'solar_zenith', 'surface_elevation_km', 'wl').values
    transm_down_dif = m['transm_down_dif'].transpose('AOT550', 'H2OSTR', 'solar_zenith', 'surface_elevation_km', 'wl').values
    grid_modtran = [m['AOT550'].values, m['H2OSTR'].values, m['solar_zenith'].values, m['surface_elevation_km'].values]

    g_t_down_dir = VectorInterpolator(grid_input=grid_modtran, data_input=transm_down_dir, version="mlg")
    g_t_down_dif = VectorInterpolator(grid_input=grid_modtran, data_input=transm_down_dif, version="mlg")

    ice_path = env.path("data", "rfidx_ice.npz")
    water_path = env.path("data", "iop", "k_liquid_water_ice.csv")

    wargs = [
        ray.put(g_t_down_dir),
        ray.put(g_t_down_dif),
        ray.put(solar_irr_full),
        ray.put(raw_wl),
        input_loc,
        input_obs,
        paths,
        veg_fraction_file,
        nodata_value,
        dayofyear,
        outpath,
        outpath_uncert,
        ice_path,
        water_path,
    ]
    
    workers = ray.util.ActorPool([SnowWorker.remote(*wargs) for _ in range(n_cores)])

    line_breaks = np.linspace(0, rows, n_cores * 4, dtype=int)
    line_breaks = [(line_breaks[n], line_breaks[n + 1]) for n in range(len(line_breaks) - 1)]

    list(workers.map_unordered(lambda a, b: a.run_chunks.remote(b), line_breaks))


@ray.remote(num_cpus=1)
class SnowWorker(object):
    def __init__(
        self,
        g_t_down_dir,
        g_t_down_dif,
        solar_irr_full,
        raw_wl,
        input_loc: str,
        input_obs: str,
        paths,
        veg_fraction_file: str,
        nodata_value: float,
        dayofyear: float, 
        outpath: str,
        outpath_uncert: str,
        ice_path,
        water_path,

    ):
        self.nodata_value = nodata_value
        self.rfl_outpath = outpath
        self.uncert_outpath = outpath_uncert

        self.g_t_down_dir = g_t_down_dir
        self.g_t_down_dif = g_t_down_dif
        self.solar_irr_full = solar_irr_full
        self.raw_wl = raw_wl
        self.wl_art = np.arange(280, 4000.0001, 10.0)
        self.wavelengths = self.wl_art * 1e-9
        self.local_i_diffuse = ((7 / 3 - 1) / 2)

        # Get ice imag and real indices
        ice_data = np.load(ice_path)
        ice_wl = micron_to_nm(ice_data["wvl"])
        self.k_ice_coarse  = interpolate.interp1d(ice_wl, ice_data["im_Pic16"], kind="slinear")(self.wl_art)
        self.r_ice_coarse = interpolate.interp1d(ice_wl, ice_data["re_Pic16"], kind="slinear")(self.wl_art)

        # Refractive index of water
        # options are: 'k_22c', 'k_minus8c', 'k_minus25c', 'k_minus7c', 'k_25c_H', 'k_20c', 'k_25c_S
        k_wi = np.genfromtxt(water_path,
                              delimiter=",", names=True, encoding="utf-8-sig")
        k_wi_idx = ~np.isnan(k_wi["wl_20c"]) & ~np.isnan(k_wi["k_20c"])
        self.k_water_coarse = np.interp(x=self.wl_art, xp=k_wi["wl_20c"][k_wi_idx], fp=k_wi["k_20c"][k_wi_idx])

        # Get algae data
        self.abs_xsc_coarse = snow_algae_chevrollier_2023(self.wl_art)


        self.g0 = 0.82
        self.B0 = self.r_ice_coarse**2
        self.eps = (9.0 * (1.0 - self.g0)) / (16.0 * self.B0)
        self.aae = 4.1
        self.mae400 = 110.0
        self.mae = self.mae400 * (self.wavelengths / 400e-9)**(-self.aae)

        x_w = 0.78
        self.cell_radius = 7.0
        self.cell_volume = (4.0 / 3.0 * np.pi * self.cell_radius**3) * 1e-18
        k_alg = x_w * self.k_water_coarse + self.abs_xsc_coarse * self.wavelengths / (np.pi * 4.0) / self.cell_volume
        density_algae = 1400.0
        self.mae_alg = (8.0 / np.pi) / self.wavelengths * k_alg / density_algae
        self.density_ice = 917.0
        self.density_algae = density_algae

        self.loc = envi.open(envi_header(input_loc), input_loc).open_memmap(interleave="bip")
        self.obs = envi.open(envi_header(input_obs), input_obs).open_memmap(interleave="bip")

        state_img = envi.open(envi_header(paths.state_working_path), paths.state_working_path)
        state_metadata_bands = state_img.metadata.get("band names")
        self.state = state_img.open_memmap(interleave="bip")

        self.uncert = envi.open(envi_header(paths.uncert_working_path), paths.uncert_working_path).open_memmap(interleave="bip")
        self.svf = envi.open(envi_header(paths.svf_working_path), paths.svf_working_path).open_memmap(interleave="bip")[:, :, 0].copy()

        try:
            self.canopy = envi.open(envi_header(veg_fraction_file), veg_fraction_file).open_memmap(interleave="bip")[:, :, 0].copy()
        except:
            self.canopy = np.zeros_like(self.svf)

        self.canopy[self.canopy < 0.0] = 0.0
        self.canopy[np.isnan(self.canopy)] = 0.0
        if np.nanmax(self.canopy) > 1.1:
            self.canopy = self.canopy / 100.0
        if np.nanmax(self.svf) > 1.1:
            self.svf = self.svf / 100.0

        self.state_idx = {band.strip().upper(): i for i, band in enumerate(state_metadata_bands)}
        
        self.algae_idx = self.state_idx.get("ALGAE_CONC")
        self.grain_idx = self.state_idx.get("GRAIN_RADIUS")
        self.dust_idx = self.state_idx.get("DUST_CONC")
        self.fsnow_idx = self.state_idx.get("FRACTIONAL_DATA")
        self.cosi_idx = self.state_idx.get("COS_I")
        self.lwc_idx = self.state_idx.get("LWC")
        self.aot_idx = self.state_idx.get("AOT550")
        self.h2o_idx = self.state_idx.get("H2OSTR")

        self.fsca_sidx = SNOW_BANDS.index("FSCA")
        self.total_sidx = SNOW_BANDS.index("SNOW_ALBEDO_TOTAL")
        self.dir_sidx = SNOW_BANDS.index("SNOW_ALBEDO_DIRECT")
        self.diff_sidx = SNOW_BANDS.index("SNOW_ALBEDO_DIFFUSE")
        self.vis_sidx = SNOW_BANDS.index("SNOW_ALBEDO_VIS")
        self.ir_sidx = SNOW_BANDS.index("SNOW_ALBEDO_IR")
        self.fsnow_sidx = SNOW_BANDS.index("F_SNOW")
        self.grain_sidx = SNOW_BANDS.index("GRAIN_RADIUS")
        self.lwc_sidx = SNOW_BANDS.index("LIQUID_WATER")
        self.dust_sidx = SNOW_BANDS.index("DUST_CONCENTRATION")
        self.algae_sidx = SNOW_BANDS.index("ALGAE_CONCENTRATION")
        self.cosi_sidx = SNOW_BANDS.index("COS_I")
    
        self.n_lines = self.state.shape[0]
        self.n_samples = self.state.shape[1]

    def aart_albedo(self, grain_radius, dust_conc, algae_conc, lwc, cos_sza, cos_vza, theta):
        
        # Expand for broadcasting for faster finite difference
        dust_conc = (dust_conc * 1e-6)[:, np.newaxis]
        lwc = (lwc / 100.0)[:, np.newaxis]
        d = (grain_radius * 2.0 * 1e-6)[:, np.newaxis]

        k_eff = (1.0 - lwc) * self.k_ice_coarse + lwc * self.k_water_coarse

        k_alg = 0.78 * self.k_water_coarse + self.abs_xsc_coarse * self.wavelengths  / (np.pi * 4.0) / self.cell_volume
        mae_alg = (8.0 / np.pi) / self.wavelengths  * k_alg / self.density_algae

        dust_conc = dust_conc * (0.917 / (2.65 * (8.0 / np.pi)))
        algae_conc = (algae_conc * 1e6 * self.cell_volume)[:, np.newaxis]
        algae_conc = algae_conc * (self.density_algae / self.density_ice)

        p = -1.90946652e+00 * np.exp(-1.58752445e-02 * theta) + 6.83972969e+00 * np.exp(-5.42025234e-03 * theta)
        r0 = (-1.11176131e+00 - 7.12680819e-01 * (cos_sza + cos_vza) + 9.02713485e+00 * cos_sza * cos_vza + p) / 4.0 / (cos_sza + cos_vza)

        beta1 = 2.0 / d * self.mae * dust_conc
        beta2 = 2.0 / d * mae_alg * algae_conc
        gamma_i = (4.0 * np.pi * k_eff) / self.wavelengths 
        
        l_param = d / self.eps
        term = (gamma_i + (beta1 + beta2)) * l_param
        S = np.sqrt(np.maximum(term, 1e-12))
        r = np.exp(-S)

        u1 = 0.6 * cos_sza + 1.0 / 3.0 + np.sqrt(cos_sza) / 3.0
        return r**u1

    def run_chunks(self, line_breaks: tuple) -> None:
        start_line, stop_line = line_breaks
        chunk_shape = (stop_line - start_line, self.n_samples, len(SNOW_BANDS))
        
        output_snow = np.full(chunk_shape, self.nodata_value, dtype=np.float32)
        output_snow_uncert = np.zeros(chunk_shape, dtype=np.float32)

        sub_state = self.state[start_line:stop_line, :, :]
        sub_uncert = self.uncert[start_line:stop_line, :, :]
        sub_svf = self.svf[start_line:stop_line, :]
        sub_canopy = self.canopy[start_line:stop_line, :]
        sub_loc = self.loc[start_line:stop_line, :, :]
        sub_obs = self.obs[start_line:stop_line, :, :]

        zero_mask = np.all(sub_state == 0.0, axis=-1)
        output_snow[zero_mask, :] = self.nodata_value

        f_snow_vals = sub_state[..., self.fsnow_idx]
        grain_vals = sub_state[..., self.grain_idx]
        dust_vals = sub_state[..., self.dust_idx]
        algae_vals = sub_state[..., self.algae_idx]
        lwc_vals = sub_state[..., self.lwc_idx]

        output_snow[..., self.fsnow_sidx] = f_snow_vals
        output_snow[..., self.grain_sidx] = grain_vals
        output_snow[..., self.lwc_sidx] = lwc_vals
        output_snow[..., self.dust_sidx] = dust_vals
        output_snow[..., self.algae_sidx] = algae_vals
        output_snow[..., self.cosi_sidx] = sub_state[..., self.cosi_idx]

        output_snow_uncert[..., self.grain_sidx] = sub_uncert[..., self.grain_idx]
        output_snow_uncert[..., self.lwc_sidx] = sub_uncert[..., self.lwc_idx]
        output_snow_uncert[..., self.dust_sidx] = sub_uncert[..., self.dust_idx]
        output_snow_uncert[..., self.algae_sidx] = sub_uncert[..., self.algae_idx]
        output_snow_uncert[..., self.cosi_sidx] = sub_uncert[..., self.state_idx.get("COS_I")]
        output_snow_uncert[..., self.fsnow_sidx] = sub_uncert[..., self.fsnow_idx]

        low_snow_mask = f_snow_vals < FSNOW_THRESHOLD
        for b_idx in [self.grain_sidx, self.lwc_sidx, self.dust_sidx, self.algae_sidx, self.cosi_sidx]:
            output_snow[low_snow_mask, b_idx] = self.nodata_value
            output_snow_uncert[low_snow_mask, b_idx] = 0.0

        theta_v_prime = np.arctan(B_R * np.tan(np.radians(sub_obs[..., 2])))
        theta_s_prime = np.radians(90 - np.degrees(np.arctan((B_R * np.tan(np.radians(90 - sub_obs[..., 6]))))))
        phi_v_prime = np.radians(sub_obs[..., 1] - sub_obs[..., 7])
        vgf = (1 - sub_canopy) ** ((np.cos(theta_s_prime)) / (np.cos(phi_v_prime) * np.sin(theta_v_prime) * np.sin(theta_s_prime) + np.cos(theta_v_prime) * np.cos(theta_s_prime)))

        fsca_denom = np.clip(1.0 - vgf, 1e-6, 1.0)
        output_snow[..., self.fsca_sidx] = np.clip(f_snow_vals / fsca_denom, 0.0, 1.0)
        output_snow_uncert[..., self.fsca_sidx] = sub_uncert[..., self.fsnow_idx]

        r_indices, c_indices = np.where(f_snow_vals >= FSNOW_THRESHOLD)
        if len(r_indices) == 0:
            return

        for r, c in zip(r_indices, c_indices):
            _alt_v = sub_loc[r, c, 2] / 1000.0
            _grain_v = grain_vals[r, c]
            _algae_v = algae_vals[r, c]
            _dust_v = dust_vals[r, c]
            _lwc_v = lwc_vals[r, c]
            _sza_v = sub_obs[r, c, 4]
            _cosi_v = sub_state[r, c, self.cosi_idx]
            _svf_v = sub_svf[r, c]

            u_grain = sub_uncert[r, c, self.grain_idx]
            u_algae = sub_uncert[r, c, self.algae_idx]
            u_dust = sub_uncert[r, c, self.dust_idx]
            u_lwc = sub_uncert[r, c, self.lwc_idx]

            # NOTE: can use FWHM if needed
            # But tests showed 10 nm to give similar performance at broadband resolution
            # Assuming a fixed AOD and h2o for albedo retrieval
            lut_array = np.array([0.02, 0.1, _sza_v, _alt_v])
            L_down_dir = self.g_t_down_dir(lut_array) * self.solar_irr_full * np.cos(np.radians(_sza_v)) / np.pi
            L_down_dif = self.g_t_down_dif(lut_array) * self.solar_irr_full * np.cos(np.radians(_sza_v)) / np.pi

            L_dif = L_down_dif * _svf_v
            cosi_clamped = max(_cosi_v, 1e-8)
            L_dir = L_down_dir / np.cos(np.radians(_sza_v)) * cosi_clamped
            L_total = L_dir + L_dif

            dg = max(u_grain * 1e-4, 1e-5)
            dal = max(u_algae * 1e-4, 1e-6)
            dd = max(u_dust * 1e-4, 1e-6)
            dl = max(u_lwc * 1e-4, 1e-6)

            grains = np.array([_grain_v, _grain_v + dg, _grain_v, _grain_v, _grain_v])
            algaes = np.array([_algae_v, _algae_v, _algae_v + dal, _algae_v, _algae_v])
            dusts = np.array([_dust_v, _dust_v, _dust_v, _dust_v + dd, _dust_v])
            lwcs = np.array([_lwc_v, _lwc_v, _lwc_v, _lwc_v, _lwc_v + dl])

            # batch aart call
            alb_dd_batch = self.aart_albedo(grains, dusts, algaes, lwcs, cosi_clamped, cos_vza=1.0, theta=180.0)
            alb_hd_batch = self.aart_albedo(grains, dusts, algaes, lwcs, self.local_i_diffuse, cos_vza=1.0, theta=180.0)

            k = L_dif / (L_dif + L_dir + 1e-12)
            alb_blue_batch = (1.0 - k) * alb_dd_batch + k * alb_hd_batch

            totals = np.trapz(alb_blue_batch * L_total, self.wl_art, axis=1) / np.trapz(L_total + 1e-12, self.wl_art)
            directs = np.trapz(alb_dd_batch * L_dir, self.wl_art, axis=1) / np.trapz(L_dir + 1e-12, self.wl_art)
            diffuses = np.trapz(alb_hd_batch * L_dif, self.wl_art, axis=1) / np.trapz(L_dif + 1e-12, self.wl_art)

            mask_vis = (self.wl_art >= 280.) & (self.wl_art <= 700.)
            vis_denom = np.trapz((L_total + 1e-12)[mask_vis], self.wl_art[mask_vis])
            viss = np.trapz((alb_blue_batch * L_total)[:, mask_vis], self.wl_art[mask_vis], axis=1) / vis_denom

            mask_ir = (self.wl_art >= 700.) & (self.wl_art <= 4000.)
            ir_denom = np.trapz((L_total + 1e-12)[mask_ir], self.wl_art[mask_ir])
            irs = np.trapz((alb_blue_batch * L_total)[:, mask_ir], self.wl_art[mask_ir], axis=1) / ir_denom

            albedos = np.column_stack([totals, directs, diffuses, viss, irs]) 

            output_snow[r, c, self.total_sidx] = albedos[0][0]
            output_snow[r, c, self.dir_sidx] = albedos[0][1]
            output_snow[r, c, self.diff_sidx] = albedos[0][2]
            output_snow[r, c, self.vis_sidx] = albedos[0][3]
            output_snow[r, c, self.ir_sidx] = albedos[0][4]

            j_grain = (albedos[1] - albedos[0]) / dg
            j_algae = (albedos[2] - albedos[0]) / dal
            j_dust  = (albedos[3] - albedos[0]) / dd
            j_lwc   = (albedos[4] - albedos[0]) / dl

            jacobians = np.column_stack([j_grain, j_algae, j_dust, j_lwc])
            u = np.array([u_grain, u_algae, u_dust, u_lwc])
            albedo_uncertainties = np.sqrt(np.sum((jacobians * u) ** 2, axis=1))

            albedo_sidxs = [self.total_sidx, self.dir_sidx, self.diff_sidx, self.vis_sidx, self.ir_sidx]
            for i, sidx in enumerate(albedo_sidxs):
                output_snow_uncert[r, c, sidx] = albedo_uncertainties[i]

        write_bil_chunk(
            np.swapaxes(output_snow, 1, 2),
            self.rfl_outpath,
            start_line,
            (self.n_lines, len(SNOW_BANDS), self.n_samples),
        )

        write_bil_chunk(
            np.swapaxes(output_snow_uncert, 1, 2),
            self.uncert_outpath,
            start_line,
            (self.n_lines, len(SNOW_BANDS), self.n_samples),
        )