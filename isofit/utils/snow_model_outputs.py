import argparse
import xarray as xr
import numpy as np
import pickle
from spectral import *
import math
import rasterio as rio
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

from isofit.core.common import resample_spectrum, VectorInterpolator



def calc_new_angles(sin_asp, cos_asp, sza, vza, saa, vaa, slope):
    '''
    Angles solved for in the inversion.
    '''
    # solve for aspect via sin(aspect) and cos(aspect)
    aspect = np.degrees(math.atan2(sin_asp, cos_asp))
    if (aspect < 0.0):
        aspect += 360.0

    # then use this information to solve for cosi and cosv
    cosi = np.sin(sza*(np.pi/180))*np.sin(slope*(np.pi/180))*np.cos(saa*(np.pi/180)-aspect*(np.pi/180))+np.cos(sza*(np.pi/180))*np.cos(slope*(np.pi/180))
    cosv = np.sin(vza*(np.pi/180))*np.sin(slope*(np.pi/180))*np.cos(vaa*(np.pi/180)-aspect*(np.pi/180))+np.cos(vza*(np.pi/180))*np.cos(slope*(np.pi/180))

    # Make any needed corrections to the result
    if cosi <= 0.05:
        cosi = 0.05
    if cosv <= 0.05:
        cosv = 0.05
    if cosi >= 1.0:
        cosi = 1.0
    if cosv >= 1.0:
        cosv = 1.0

    return cosi, cosv


def rho_to_rdn(rho, coszen, solar_irr):
    """Function to convert a transmittance vector to radiance.

    Args:
        rho:       input data vector in transmittance
        coszen:    cosine of solar zenith angle
        solar_irr: solar irradiance vector (optional)

    Returns:
        Data vector converted to radiance
    """
    return (solar_irr * coszen) / np.pi * rho


def incoming_flux(transms, thetas, sky_view, mus, alb_snow, slp, aot, h2o, alt, vza):
    transm_down_dir= transms[0]
    transm_down_dif = transms[1]   

    coszen = np.cos(np.radians(thetas))
    lut_array = np.array([aot, h2o, alt, vza])

    L_down_dir = transm_down_dir(lut_array) * solar_irr * coszen / np.pi
    L_down_dif = transm_down_dif(lut_array) * solar_irr * coszen / np.pi

    #plt.scatter(wl,L_down_dir)
    #plt.show()

    #alb_snow = 0.80 #NOTE TODO
    # background reflectance is treated as surrounding snow albedo

    # neighbor slope diffuse irradiance
    ct = max(0,((1 + np.cos(np.radians(slp))) / 2 ) - sky_view)
    L_slope = (L_down_dif+L_down_dir) * alb_snow * ct

    # Adjust to local terrain 
    L_down_dif = L_down_dif * sky_view
    L_down_dir = L_down_dir / coszen * mus

    # combine
    L_down = L_down_dir + L_down_dif + L_slope


    return L_down





if __name__ == '__main__':

    # Set req args
    parser = argparse.ArgumentParser(description='Model Snow Albedo')                
    parser.add_argument('--states', type=str, required=True,
                        help='Path to state vector file (ending in .hdr)')
    parser.add_argument('--uncert', type=str, required=True,
                        help='Path to posterior uncertainty file (ending in .hdr)')
    parser.add_argument('--svf', type=str, required=True,
                        help='Path to sky view fraction file.') 
    parser.add_argument('--slope', type=str, required=True,
                        help='Path to slope file.') 
    parser.add_argument('--shadow', type=str, required=True,
                        help='Path to shadow file.') 
    parser.add_argument('--canopy', type=str, required=True,
                        help='Path to canopy cover file.') 
    parser.add_argument('--obs', type=str, required=True,
                        help='Path to OBS file (ending in .hdr)')
    parser.add_argument('--loc', type=str, required=True,
                        help='Path to LOC file (ending in .hdr)')   
    parser.add_argument('--modtran', type=str, required=True,
                        help='Path to modtran lut (NetCDF)')
    parser.add_argument('--output', type=str, required=True,
                        help='output dir')


    # Parse args
    args = parser.parse_args()
    state_file = args.states
    uncert_file = args.uncert
    svf_file = args.svf
    slope_file = args.slope
    shadow_file = args.shadow
    canopy_file = args.canopy
    obs_file = args.obs
    loc_file = args.loc
    modtran_file = args.modtran
    output_dir = args.output
    

    # Load MODTRAN LUT used for terrain correction of snow reflectance.
    m = xr.load_dataset(modtran_file)
    solar_irr = m.solar_irr.values
    wl = m.wl.values
    transm_down_dir = m['transm_down_dir'].transpose('AOT550', 'H2OSTR', 'surface_elevation_km','observer_zenith', 'wl').values
    transm_down_dif = m['transm_down_dif'].transpose('AOT550', 'H2OSTR', 'surface_elevation_km','observer_zenith', 'wl').values
    m_grid = [
        m['AOT550'].values,
        m['H2OSTR'].values,
        m['surface_elevation_km'].values,
        m['observer_zenith'].values,        
    ]

    wl_mask = (m.wl.values > 350.) & (m.wl.values < 2500.)
    wl = m.wl.values[wl_mask]
    solar_irr = solar_irr[wl_mask]

    # Trim LUTs along wavelength dimension
    transm_down_dir = transm_down_dir[..., wl_mask]
    transm_down_dif = transm_down_dif[..., wl_mask]

    # Now create interpolators
    g_t_down_dir = VectorInterpolator(
        grid_input=m_grid,
        data_input=transm_down_dir,
        version="mlg"
    )
    g_t_down_dif = VectorInterpolator(
        grid_input=m_grid,
        data_input=transm_down_dif,
        version="mlg"
    )

   # load DISORT data to Jouni's interp class
   # NOTE: will need to sub this out for correct
    ds = xr.load_dataset("/Users/bawilder/Code/snow/LUT/disort_snow_lut.nc")
    sensor_wl = ds['wavelength'].values
    grid = [
        ds['sza'].values,
        ds['vza'].values,
        ds['raa'].values,
        ds['grain_radius'].values,
        ds['algae_conc'].values,
        ds['dust_conc'].values,
        ds['lwc'].values,
    ]

    g_adif = VectorInterpolator(
        grid_input=grid,
        data_input=ds['a_diff'].values,
        version="mlg"
    )

    print('MODTRAN WL RANGE:')
    print(wl)

    svf = np.squeeze(rio.open(svf_file).read()).astype(float)
    slope = rio.open(slope_file).read(1).astype(float) 
    shadow = np.load(shadow_file)
   

    #canopy_cover = rio.open(canopy_file).read(1).astype(float) 
    #canopy_cover[np.isnan(canopy_cover)] = 0.
    canopy_cover = 0 # NOTE: basing this OE uncertainty will replace canopy cover need.
    # if wanted fSCA need to come back to this. but for now i just want albedo where uncertainty is small.

    state = envi.open(state_file)
    state = state.open_memmap(writable=False).astype(float).copy()

    uncert = envi.open(uncert_file)
    uncert = uncert.open_memmap(writable=False).astype(float).copy()

    # band names = { 0 path length, 1 to-sensor azimuth, 
    # 2 to-sensor zenith, 3  to-sun azimuth, 4  to-sun zenith,
    # 5  phase, 6  slope, 7  aspect, 8  cosine i, 9  UTC time, 10  earth-sun distance}
    obs = envi.open(obs_file)
    loc = envi.open(loc_file)
    obs = obs.open_memmap(writeable=False).copy()
    loc = loc.open_memmap(writeable=False).copy()

    zsnow = state[:,:,6]
    zveg = state[:,:,7]
    znpv = state[:,:,8]
    zsoil = state[:,:,9]

    # apply soft max condition for sum-to-1 fractional covers condition...
    fsnow = np.full_like(state[:,:,0], fill_value=np.nan)
    for i in range(fsnow.shape[0]):
        for j in range(fsnow.shape[1]):
            zi = np.array([zsnow[i,j], zveg[i,j], znpv[i,j], zsoil[i,j]])
            fi = np.exp(zi) / np.sum(np.exp(zi))
            fsnow[i,j] = fi[0]

    ####
    # Inn this section, similar to SPIRES, we only allow dust, grain size OE retrievals if pass certain threshold
    # TODO: just 
    fsnow_threshold = 0.75

    grain = state[:,:,2]
    lw = state[:,:,3]
    dust = state[:,:,4]
    algae = state[:,:,5]

    grain[fsnow <= fsnow_threshold] = np.nan   
    lw[fsnow <= fsnow_threshold] = np.nan  
    dust[fsnow <= fsnow_threshold] = np.nan   
    algae[fsnow <= fsnow_threshold] = np.nan  

    ####

    # initiate broadband array
    # load in initial dataset for reference
    with rio.open(svf_file) as src:
        ras_data = src.read()
        ras_meta = src.profile
        ras_meta['nodata'] = -9999


    # set as zeros (less than fsca min will be set to zero)
    broadband_arr = np.zeros_like(svf)

    tmp = []
    
    print('loaded all data starting loop...')

    def process_pixel(i, j):
        # do not compute albedo if fsca below a certain threshold.
        if fsnow[i,j] < 0.75:
            return (i, j, -9999)
        if np.isnan(grain[i,j]) or np.isnan(dust[i,j]) or np.isnan(algae[i,j]) or np.isnan(lw[i,j]):
            return (i,j,-9999)
        try:
            states_ij = state[i, j, :]
            obs_ij = obs[i, j, :]
            loc_ij = loc[i, j, :]
            vaa = obs_ij[1]
            vza = obs_ij[2]
            saa = obs_ij[3]
            sza = obs_ij[4]
            alt = loc_ij[2] / 1000.
            aot = states_ij[13]
            h2o = states_ij[14]
            #slope = obs_ij[6]
            slope_ij = slope[i,j]
            grain_ij = grain[i, j]
            lwc_ij = lw[i, j]
            algae_ij = algae[i, j]
            dust_ij = dust[i, j]
            shadow_ij =shadow[i,j]

            cosi, cosv = calc_new_angles(states_ij[0], states_ij[1], sza, vza, saa, vaa, slope_ij)
            cosi = shadow_ij * cosi
            cosi = np.clip(cosi, 0.06,1.0)
            cosv = np.clip(cosv,0.06,1.0)

            delta_phi = np.abs(saa - vaa)
            raa = np.minimum(delta_phi, 360 - delta_phi)
            disort_raa = 180 - raa

            alb_snow = g_adif(np.array([np.degrees(np.arccos(cosi)), np.degrees(np.arccos(cosv)), 
                                        disort_raa, grain_ij, algae_ij, dust_ij, lwc_ij]))
            alb_snow = np.interp(wl, sensor_wl, alb_snow)

            #m, thetas, sky_view, mus, alb_snow, slp,     aot, h2o, alt, vza
            L_total = incoming_flux((g_t_down_dir,g_t_down_dif), sza, svf[i, j], cosi, alb_snow, slope_ij,
                                    aot, h2o, alt, 180-vza)
            broadband = np.trapezoid(alb_snow * L_total, dx=1) / np.trapezoid(L_total, dx=1)

            #if i==13 and j==12:
            #    # i, j = 13, 12
            #    tmp.append([wl, alb_snow , L_total])
            #    np.save('tmp.npy', tmp)

            return (i, j, broadband)

        except Exception as e:
            print(f"Failed at pixel ({i},{j}): {e}")
            return (i, j, -9999)

    # Parallel processing
    results = Parallel(n_jobs=-3, verbose=10)(
        delayed(process_pixel)(i, j) 
        for i in range(state.shape[0]) 
        for j in range(state.shape[1])
    )

    # Populate results into the array
    for i, j, broadband in results:
        broadband_arr[i, j] = broadband

    # write file
    with rio.open(f'{output_dir}/BA.tif', 'w', **ras_meta) as dst:
        dst.write(broadband_arr, 1)
