from os.path import abspath, join

import xarray as xr
import numpy as np
from spectral.io import envi

from isofit import ray
from isofit.core.common import envi_header
from isofit.core.fileio import initialize_output, write_bil_chunk
from isofit.core.common import VectorInterpolator, eps

FSNOW_THRESHOLD = 0.75

B_R=2.7

ALBEDO_AOT_MIN = 0.01 + eps
ALBEDO_AOT_MAX = 0.6 - eps

ALBEDO_H2O_MIN = 0.05 + eps
ALBEDO_H2O_MAX = 0.50001 - eps

ALBEDO_ALT_MIN = 0.0 + eps
ALBEDO_ALT_MAX = 5.0 - eps

ALBEDO_ALT_MIN = 0.0 + eps
ALBEDO_ALT_MAX = 5.0 - eps

ALBEDO_GRAIN_MIN = 30.0 + eps
ALBEDO_GRAIN_MAX = 1500.0 - eps

ALBEDO_ALGAE_MIN = 0.0 + eps
ALBEDO_ALGAE_MAX = 6e5 - eps

ALBEDO_DUST_MIN = 0.0 + eps
ALBEDO_DUST_MAX = 4000.0 - eps

ALBEDO_LWC_MIN = 0.0 + eps
ALBEDO_LWC_MAX = 25.0 - eps

ALBEDO_SZA_MIN = 0.0 + eps
ALBEDO_SZA_MAX = 70.0 - eps

ALBEDO_COSI_MIN = 1e-3 + eps
ALBEDO_COSI_MAX = 70.0 - eps

ALBEDO_SVF_MIN = 0.5 + eps
ALBEDO_SVF_MAX = 1.0 - eps

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
    "EOF_1",
    "EOF_2",
    "EOF_3",
]


def snow_model_outputs(
    input_loc,
    input_obs,
    paths,
    veg_fraction_file,
    albedo_lut,
    dayofyear, 
    n_cores,
    nodata_value=-9999,
):
    """
    TODO
    """

    loc = envi.open(envi_header(input_loc), input_loc).open_memmap()
    rows, cols, _ = loc.shape
    del loc

    # Set output paths
    rdn_fname = paths.fid + "_rdn"
    outpath = abspath(join(paths.output_directory, rdn_fname.replace("_rdn", "_snow")))
    outpath_uncert = abspath(join(paths.output_directory, rdn_fname.replace("_rdn", "_snow_uncert")))

    # Initialize the outputs
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
        outpath=outpath,
        out_shape=(rows, len(SNOW_BANDS), cols),
        bands=f"{len(SNOW_BANDS)}",
        description="DISORT derived snow surface properties fit to TOA radiance",
    )

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
        outpath=outpath_uncert,
        out_shape=(rows, len(SNOW_BANDS), cols),
        bands=f"{len(SNOW_BANDS)}",
        description="DISORT derived snow surface properties fit to TOA radiance uncertainty",
    )

    # Load Albedo LUT 
    ds = xr.load_dataset(albedo_lut)
    grid = [
        ds["aot"].values,
        ds["h2o"].values,
        ds["altitude"].values,
        ds["grain_radius"].values,
        ds["algae_conc"].values,
        ds["dust_conc"].values,
        ds["lwc"].values,
        ds["toa_sza"].values,
        ds["cosi"].values,
        ds["svf"].values,
    ]
    
    combined_data = np.stack(
        [
            ds["a_total"].values,
            ds["a_direct"].values,
            ds["a_diffuse"].values,
            ds["a_vis"].values,
            ds["a_ir"].values,
            # Total derivatives
            ds["da_total_daot"].values,
            ds["da_total_dh2o"].values,
            ds["da_total_dgrain"].values,
            ds["da_total_dalgae"].values,
            ds["da_total_ddust"].values,
            ds["da_total_dlwc"].values,
            ds["da_total_dcosi"].values,
            ds["da_total_dsvf"].values,
            # Direct derivatives
            ds["da_direct_daot"].values,
            ds["da_direct_dh2o"].values,
            ds["da_direct_dgrain"].values,
            ds["da_direct_dalgae"].values,
            ds["da_direct_ddust"].values,
            ds["da_direct_dlwc"].values,
            ds["da_direct_dcosi"].values,
            ds["da_direct_dsvf"].values,
            # Diffuse derivatives
            ds["da_diffuse_daot"].values,
            ds["da_diffuse_dh2o"].values,
            ds["da_diffuse_dgrain"].values,
            ds["da_diffuse_dalgae"].values,
            ds["da_diffuse_ddust"].values,
            ds["da_diffuse_dlwc"].values,
            ds["da_diffuse_dcosi"].values,
            ds["da_diffuse_dsvf"].values,
            # VIS derivatives
            ds["da_vis_daot"].values,
            ds["da_vis_dh2o"].values,
            ds["da_vis_dgrain"].values,
            ds["da_vis_dalgae"].values,
            ds["da_vis_ddust"].values,
            ds["da_vis_dlwc"].values,
            ds["da_vis_dcosi"].values,
            ds["da_vis_dsvf"].values,
            # IR derivatives
            ds["da_ir_daot"].values,
            ds["da_ir_dh2o"].values,
            ds["da_ir_dgrain"].values,
            ds["da_ir_dalgae"].values,
            ds["da_ir_ddust"].values,
            ds["da_ir_dlwc"].values,
            ds["da_ir_dcosi"].values,
            ds["da_ir_dsvf"].values,
        ],
        axis=-1,
    )
    G = VectorInterpolator(grid_input=grid, data_input=combined_data, version="mlg")
    ds.close()

    wargs = [
        ray.put(G),
        input_loc,
        input_obs,
        paths,
        veg_fraction_file,
        nodata_value,
        dayofyear,
        outpath,
        outpath_uncert,
    ]
    
    workers = ray.util.ActorPool([SnowWorker.remote(*wargs) for _ in range(n_cores)])

    line_breaks = np.linspace(0, rows, n_cores * 4, dtype=int)
    line_breaks = [(line_breaks[n], line_breaks[n + 1]) for n in range(len(line_breaks) - 1)]

    list(workers.map_unordered(lambda a, b: a.run_chunks.remote(b), line_breaks))


@ray.remote(num_cpus=1)
class SnowWorker(object):
    def __init__(
        self,
        G: VectorInterpolator,
        input_loc: str,
        input_obs: str,
        paths,
        veg_fraction_file: str,
        nodata_value: float,
        dayofyear: float, 
        outpath: str,
        outpath_uncert: str,
    ):
        self.nodata_value = nodata_value
        self.rfl_outpath = outpath
        self.uncert_outpath = outpath_uncert

        self.G = G

        self.loc = envi.open(envi_header(input_loc), input_loc).open_memmap(interleave="bip")
        self.obs = envi.open(envi_header(input_obs), input_obs).open_memmap(interleave="bip")

        state_img = envi.open(envi_header(paths.state_working_path), paths.state_working_path)
        state_metadata_bands = state_img.metadata.get("band names")
        self.state = state_img.open_memmap(interleave="bip")

        self.uncert = envi.open(envi_header(paths.uncert_working_path), paths.uncert_working_path).open_memmap(interleave="bip")
        
        self.svf = envi.open(envi_header(paths.svf_working_path), paths.svf_working_path).open_memmap(interleave="bip").squeeze().copy()
        self.canopy = envi.open(envi_header(veg_fraction_file), veg_fraction_file).open_memmap(interleave="bip").squeeze().copy()

        # NOTE a lot of the times these canopy will have nan for non-vegetated areas
        # and so we will populate this with zeros
        self.canopy[self.canopy<0.0] = 0.0
        self.canopy[np.isnan(self.canopy)] = 0.0

        if np.nanmax(self.canopy) > 1.1 :
            self.canopy = self.canopy / 100.0
        
        if np.nanmax(self.svf) > 1.1 :
            self.svf = self.svf / 100.0

        # Grab terrain error from literature
        # NOTE this cosi error is not used but keeping this here in case helpful
        dozier_2022_cosi_himachal_pradesh = [
            (355, 0.117),  # Dec 21
            (37,  0.111),  # Feb 6
            (59,  0.105),  # Feb 28
            (79,  0.097),  # Mar 20
            (100, 0.089),  # Apr 10
            (123, 0.089),  # May 3
            (172, 0.078),  # Jun 21
        ]
        dozier_doy_vals = [x[0] for x in dozier_2022_cosi_himachal_pradesh]
        dozier_error_vals = [x[1] for x in dozier_2022_cosi_himachal_pradesh]
        self.cosi_error_literature = float(np.interp(dayofyear, 
                                                      dozier_doy_vals, 
                                                      dozier_error_vals, 
                                                      left=0.117, 
                                                      right=0.078))
        self.svf_error_literature = 0.0404

        self.state_idx = {band.strip().upper(): i for i, band in enumerate(state_metadata_bands)}
        
        self.algae_idx = self.state_idx.get("ALGAE_CONC")
        self.grain_idx = self.state_idx.get("GRAIN_RADIUS")
        self.dust_idx = self.state_idx.get("DUST_CONC")
        self.fsnow_idx = self.state_idx.get("FRACTIONAL_DATA")
        self.fpv_idx = self.state_idx.get("FRACTIONAL_PV")
        self.fnpv_idx = self.state_idx.get("FRACTIONAL_NPV")
        self.fsoil_idx = self.state_idx.get("FRACTIONAL_SOIL")
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

    def run_chunks(self, line_breaks: tuple) -> None:
        start_line, stop_line = line_breaks
        chunk_shape = (stop_line - start_line, self.n_samples, len(SNOW_BANDS))
        
        output_snow = np.full(chunk_shape, self.nodata_value, dtype=np.float32)
        output_snow_uncert = np.full(chunk_shape, self.nodata_value, dtype=np.float32)
    
        sub_state = self.state[start_line:stop_line, :, :]
        sub_uncert = self.uncert[start_line:stop_line, :, :]
        sub_svf = self.svf[start_line:stop_line, :]
        sub_canopy = self.canopy[start_line:stop_line, :]
        sub_loc = self.loc[start_line:stop_line, :, :]
        sub_obs = self.obs[start_line:stop_line, :, :]

        f_snow_vals = sub_state[..., self.fsnow_idx]
        grain_vals = sub_state[..., self.grain_idx]
        dust_vals = sub_state[..., self.dust_idx]
        algae_vals = sub_state[..., self.algae_idx]
        lwc_vals = sub_state[..., self.lwc_idx]

        output_snow[..., self.fsnow_sidx] = f_snow_vals
        output_snow[..., self.grain_sidx] = grain_vals
        output_snow[..., self.lwc_sidx] = sub_state[..., self.lwc_idx]
        output_snow[..., self.dust_sidx] = dust_vals
        output_snow[..., self.algae_sidx] = sub_state[..., self.algae_idx]
        output_snow[..., self.cosi_sidx] = sub_state[..., self.cosi_idx]

        output_snow_uncert[..., self.grain_sidx] = sub_uncert[..., self.grain_idx]
        output_snow_uncert[..., self.lwc_sidx] = sub_uncert[..., self.lwc_idx]
        output_snow_uncert[..., self.dust_sidx] = sub_uncert[..., self.dust_idx]
        output_snow_uncert[..., self.algae_sidx] = sub_uncert[..., self.algae_idx]
        output_snow_uncert[..., self.cosi_sidx] = sub_uncert[..., self.cosi_idx]

        # Exclude state (except for f_snow) where there is low snow
        low_snow_mask = f_snow_vals < FSNOW_THRESHOLD
        exclude_bands_snow = [
            self.grain_sidx,
            self.lwc_sidx,
            self.dust_sidx,
            self.algae_sidx,
            self.cosi_sidx,
        ]
        for b_idx in exclude_bands_snow:
            output_snow[low_snow_mask, b_idx] = self.nodata_value
            output_snow_uncert[low_snow_mask, b_idx] = self.nodata_value

        # Perform fSCA calculation
        # GO-VGF EQN
        fshade = 0.0 # This term in the eqn can be ignored because we solve optimally for cos_i.
        # Typically fshade is present because photometric shade is unaccounted for in the retrieval..
        # In our retrieval since we solve for this, thre is no need to post-correct fSCA.
        theta_v_prime = np.arctan(B_R * np.tan(np.radians(sub_obs[..., 2])))
        theta_s_prime= np.radians(90 - np.degrees(np.arctan((B_R * np.tan(np.radians(90-sub_obs[..., 6]))))))
        phi_v_prime = np.radians(sub_obs[..., 1]-sub_obs[..., 7])
        vgf = (1-sub_canopy) ** ((np.cos(theta_s_prime)) / (np.cos(phi_v_prime)*np.sin(theta_v_prime)*np.sin(theta_s_prime)+np.cos(theta_v_prime)*np.cos(theta_s_prime)))

        fsca_denom = (1 - fshade - vgf) # denom seperated out bc canopy_cover can cause to be greater than 1.
        fsca_denom = np.clip(fsca_denom, 1e-6, 1.0)
        fsca_val = f_snow_vals / fsca_denom
        output_snow[..., self.fsca_sidx] = np.clip(fsca_val, 0.0, 1.0)

        r_indices, c_indices = np.where((f_snow_vals >= FSNOW_THRESHOLD))

        for r, c in zip(r_indices, c_indices):
   
            # NOTE Clipping to max of albedo LUT generated on 3 Sep 2026
            _aot_v = np.clip(sub_state[r, c, self.aot_idx], ALBEDO_AOT_MIN, ALBEDO_AOT_MAX)
            _h2o_v = np.clip(sub_state[r, c, self.h2o_idx], ALBEDO_H2O_MIN, ALBEDO_H2O_MAX)
            _alt_v = np.clip(sub_loc[r, c, 2] / 1000.0, ALBEDO_ALT_MIN, ALBEDO_ALT_MAX)
            _grain_v = np.clip(grain_vals[r, c], ALBEDO_GRAIN_MIN, ALBEDO_GRAIN_MAX)
            _algae_v = np.clip(algae_vals[r, c], ALBEDO_ALGAE_MIN, ALBEDO_ALGAE_MAX)
            _dust_v = np.clip(dust_vals[r, c], ALBEDO_DUST_MIN, ALBEDO_DUST_MAX)
            _lwc_v = np.clip(lwc_vals[r, c], ALBEDO_LWC_MIN, ALBEDO_LWC_MAX)
            _sza_v = np.clip(sub_obs[r, c, 4], ALBEDO_SZA_MIN, ALBEDO_SZA_MAX)
            _cosi_v = np.clip(sub_state[r, c, self.cosi_idx], ALBEDO_COSI_MIN, ALBEDO_COSI_MAX)
            _svf_v = np.clip(sub_svf[r, c], ALBEDO_SVF_MIN, ALBEDO_SVF_MAX)

            interp_g = self.G(np.array([_aot_v, _h2o_v, _alt_v, _grain_v, _algae_v, _dust_v, _lwc_v, _sza_v, _cosi_v, _svf_v]))

            a = interp_g[0:5]
            d = interp_g[5:]

            # Storing interpolated albedo from LUT
            output_snow[r, c, self.total_sidx] = a[0]
            output_snow[r, c, self.dir_sidx] = a[1]
            output_snow[r, c, self.diff_sidx] = a[2]
            output_snow[r, c, self.vis_sidx] = a[3]
            output_snow[r, c, self.ir_sidx] = a[4]

            # Error propagation for albedo based on OE uncertainty
            u = np.array([
                sub_uncert[r, c, self.aot_idx],
                sub_uncert[r, c, self.h2o_idx],
                sub_uncert[r, c, self.grain_idx],
                sub_uncert[r, c, self.algae_idx],
                sub_uncert[r, c, self.dust_idx],
                sub_uncert[r, c, self.lwc_idx],
                sub_uncert[r, c, self.cosi_idx],
                self.svf_error_literature # svf , static assumed upper end from Dozier 2022
            ])

            # NOTE on assumptions:
            # 1. assumes independence and omits covariance
            # 2. gaussian error distribution
            # 3. local linearity
            for i, idx_s in enumerate([self.total_sidx, self.dir_sidx, self.diff_sidx, self.vis_sidx, self.ir_sidx]):
                output_snow_uncert[r, c, idx_s] = np.sqrt(np.sum(((d[i * 8 : (i + 1) * 8]) * u) ** 2))


        write_bil_chunk(
            np.swapaxes(output_snow, 1, 2),
            self.rfl_outpath,
            start_line,
            (self.n_lines, len(SNOW_BANDS), self.n_samples),
        )

        # Save surface snow property uncertainty
        write_bil_chunk(
            np.swapaxes(output_snow_uncert, 1, 2),
            self.uncert_outpath,
            start_line,
            (self.n_lines, len(SNOW_BANDS), self.n_samples),
        )