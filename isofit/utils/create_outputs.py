import xarray as xr
import numpy as np
from spectral.io import envi

from isofit.core.common import envi_header
from isofit.core.common import VectorInterpolator


def snow_model_outputs(input_loc, paths, veg_fraction_file, albedo_lut, mean_to_sensor_zenith):
    """
    - fSCA calculation based on canopy correction
    - Removes of all data (except fSCA) where f_snow is less than 0.75.. And where grain size is outside of the bounds..
    - uses the albedo calculators (Direct, Diffuse, Total, VIS, NIR) on the remaining data

    Outputs into a new _snow.hdr
    snow.hdr = [fSCA, snow_albedo_direct, snow_albedo_diffuse, snow_albedo_total, snow_albedo_nir, snow_albedo_vis, f_snow, grain_radius, liquid_water, dust_concentration, algae_concentration, cos_i]
    snow_uncert.hdr = [fSCA(-9999),  snow_albedo_direct, snow_albedo_diffuse, snow_albedo_total, snow_albedo_nir, snow_albedo_vis, f_snow(-9999), grain_radius, liquid_water, dust_concentration, algae_concentration, cos_i]

    """

    # Load the loc, state, and uncert files
    loc = envi.open(envi_header(input_loc), input_loc).open_memmap()
    state = envi.open(envi_header(paths.state_working_path), paths.state_working_path).open_memmap()
    uncert = envi.open(envi_header(paths.uncert_working_path), paths.uncert_working_path).open_memmap()

    # Load ancillary data needed: skyview, canopy fraction, 
    svf = envi.open(envi_header(paths.svf_working_path), paths.svf_working_path).open_memmap().squeeze()
    canopy = envi.open(envi_header(veg_fraction_file), veg_fraction_file).open_memmap().squeeze()

    # Load albedo LUT
    ds = xr.load_dataset(albedo_lut)
    grid = [
        ds['altitude'].values,
        ds['grain_radius'].values,
        ds['dust_conc'].values,
        ds['toa_sza'].values,
        ds['cosi'].values,
        ds['svf'].values,
    ]

    combined_data = np.stack([
        #ds['a_total'].values,
        ds['a_direct'].values,
        ds['a_diffuse'].values,
        #ds['da_total_dgrain'].values,
        #ds['da_total_ddust'].values,
        #ds['da_total_dcosi'].values,
        #ds['da_total_dsvf'].values,
        #ds['da_direct_dcosi'].values,
        #ds['da_diffuse_dsvf'].values,
    ], axis=-1)

    g_all = VectorInterpolator(grid_input=grid, data_input=combined_data, version="rg")



    # Initialize the output



    # Before loop, removal of data where f_snow  is less than 0.75 and grain size is less than 30.01 or greater than 1499.99
    # Or where dust value is greater than 3999.99



    
    # For every pixel, do the albedo LUT (uncert too) , fSCA correction .. 



    return
