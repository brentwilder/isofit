import xarray as xr
import numpy as np
import os
from joblib import Parallel, delayed
from isofit.core.common import VectorInterpolator


disort_file = "/Users/bawilder/Code/snow/LUT/disort_snow_lut_ANG.nc"
modtran_file = "/Users/bawilder/Documents/KINGS/MODTRAN/modtran_blue_hook.nc"
output_file = "/Users/bawilder/Code/snow/LUT/broadband_albedo_lut.nc"
n_jobs = 13  
# NOTE: static AOT and H2o
altitude = np.arange(0, 5.001, 0.5)     # km
grain_radius = np.linspace(30, 1500, 20)
dust_conc = np.arange(0, 4000.001, 1000)
toa_sza = np.arange(0, 60.001, 5)
cosi = np.arange(0, 1.001, 0.05)
svf = np.arange(0, 1.0001, 0.25)
aot = 0.05


ds = xr.load_dataset(disort_file)
sensor_wl = ds['wavelength'].values
grid_disort = [
    ds['sza'].values,
    ds['vza'].values,
    ds['raa'].values,
    ds['grain_radius'].values,
    ds['algae_conc'].values,
    ds['dust_conc'].values,
    ds['lwc'].values,
]
g_adif = VectorInterpolator(grid_input=grid_disort, data_input=ds['a_diff'].values, version="mlg")


m = xr.load_dataset(modtran_file)
solar_irr = m.solar_irr.values
wl = m.wl.values
transm_down_dir = m['transm_down_dir'].transpose('AOT550', 'solar_zenith', 'surface_elevation_km', 'wl').values
transm_down_dif = m['transm_down_dif'].transpose('AOT550', 'solar_zenith', 'surface_elevation_km', 'wl').values
grid_modtran = [m['AOT550'].values, m['solar_zenith'].values, m['surface_elevation_km'].values]

wl_mask = (wl > 350.) & (wl < 2500.)
wl = wl[wl_mask]
solar_irr = solar_irr[wl_mask]
transm_down_dir = transm_down_dir[..., wl_mask]
transm_down_dif = transm_down_dif[..., wl_mask]

g_t_down_dir = VectorInterpolator(grid_input=grid_modtran, data_input=transm_down_dir, version="mlg")
g_t_down_dif = VectorInterpolator(grid_input=grid_modtran, data_input=transm_down_dif, version="mlg")


shape = (len(altitude), len(grain_radius), len(dust_conc),
         len(toa_sza), len(cosi), len(svf))


def compute_point(alt, grain_ij, dust_ij, theta, cosi, sv):
    lut_array = np.array([aot, theta, alt])
    L_down_dir = g_t_down_dir(lut_array) * solar_irr * np.cos(np.radians(theta)) / np.pi
    L_down_dif = g_t_down_dif(lut_array) * solar_irr * np.cos(np.radians(theta)) / np.pi

    if cosi<0.06:
        local_inc_angle = np.degrees(np.arccos(0.06))
    else:
        local_inc_angle = np.degrees(np.arccos(cosi))

    alb_snow = np.interp(
        wl,
        sensor_wl,
        g_adif(np.array([local_inc_angle, 0.0, 180.0, grain_ij, 0, dust_ij, 0]))
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        L_slope = (L_down_dif + L_down_dir) * alb_snow * 0.0
        L_dif = L_down_dif * sv
        L_dir = L_down_dir / np.cos(np.radians(theta)) * cosi
        L_total = L_dif + L_dir + L_slope
        denom = np.trapezoid(L_total, dx=1)
        broadband = np.trapezoid(alb_snow * L_total, dx=1) / denom

        L_slope_dir = (L_down_dir) * alb_snow * 0.0
        L_total_dir = L_dir + L_slope_dir
        denom_direct = np.trapezoid(L_total_dir, dx=1)
        direct = np.trapezoid(alb_snow * L_total_dir, dx=1) / denom_direct

        L_slope_dif = (L_down_dif) * alb_snow * 0.0
        L_total_dif = L_dif + L_slope_dif
        denom_dif = np.trapezoid(L_total_dif, dx=1)
        diffuse = np.trapezoid(alb_snow * L_total_dif, dx=1) / denom_dif

    return broadband, direct, diffuse


grid_list = []
for alt in altitude:
    for grain_ij in grain_radius:
        for dust_ij in dust_conc:
            for theta in toa_sza:
                for local_inc_angle in cosi:
                    for sv in svf:
                        grid_list.append((alt, grain_ij, dust_ij, theta, local_inc_angle, sv))

print(f"Computing {len(grid_list)} LUT entries using {n_jobs} cores...")


results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
    delayed(compute_point)(alt, grain_ij, dust_ij, theta, local_inc_angle, sv)
    for alt, grain_ij, dust_ij, theta, local_inc_angle, sv in grid_list
)

# Unpack results
a_total = np.zeros(shape)
a_direct = np.zeros(shape)
a_diffuse = np.zeros(shape)

k = 0
for i_alt, alt in enumerate(altitude):
    for i_grain, grain in enumerate(grain_radius):
        for i_dust, dust in enumerate(dust_conc):
            for i_sza, sza in enumerate(toa_sza):
                for i_cosi, ci in enumerate(cosi):
                    for i_svf, sv in enumerate(svf):
                        broadband, direct, diffuse = results[k]
                        a_total[i_alt, i_grain, i_dust, i_sza, i_cosi, i_svf] = broadband
                        a_direct[i_alt, i_grain, i_dust, i_sza, i_cosi, i_svf] = direct
                        a_diffuse[i_alt, i_grain, i_dust, i_sza, i_cosi, i_svf] = diffuse
                        k += 1


lut_ds = xr.Dataset(
    {
        "a_total": (["altitude", "grain_radius", "dust_conc", "toa_sza", "cosi", "svf"], a_total),
        "a_direct": (["altitude", "grain_radius", "dust_conc", "toa_sza", "cosi", "svf"], a_direct),
        "a_diffuse": (["altitude", "grain_radius", "dust_conc", "toa_sza", "cosi", "svf"], a_diffuse),
    },
    coords={
        "altitude": altitude,
        "grain_radius": grain_radius,
        "dust_conc": dust_conc,
        "toa_sza": toa_sza,
        "cosi": cosi,
        "svf": svf,
    },
    attrs={"AOT": aot, "description": "Broadband albedos of DISORT Snow Surface. To be used for STARS derived snow properties."}
)

lut_ds.to_netcdf(output_file)

