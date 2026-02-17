import numpy as np
from spectral import *
import scipy
import multiprocessing as mp
import rasterio as rio
from scipy import interpolate
import matplotlib.pyplot as plt


# run in parallel
def run_ray_pool(list_args, n_cpu):
    with mp.Pool(n_cpu) as pool:
        results = pool.map(ray_trace_mp, list_args)
    return results


# Define wrapper function
def ray_trace_mp(args):
    return ray_trace(*args)


def ray_trace(i, j, dem_arr, sza, saa, pix_size):
    """
    loop through all pixels in parallel and compute simple ray tracing

    needs dem raster
    and obs array

    """

    i_lim = dem_arr.shape[0]
    j_lim = dem_arr.shape[1]

    # Compute projected solar directions
    tan_theta_e = np.tan(np.radians(90 - sza))
    tan_sundir = -1 * np.tan(
        np.radians(saa)
    )  # direction flipped because negative is pointing right

    # Assumes image is taken mid-day so mover_i(row-wise / N-S) will always be larger than mover_j
    y_mover = np.arange(0, 100.1, 1)
    x_mover = np.round(y_mover * tan_sundir).astype(int)

    # Southern hemisphere (need to move back up the image instead of down the rows)
    if saa > 270 or saa < 90:
        y_mover = y_mover * -1

    y = i + y_mover
    x = j + x_mover

    # Make sure y_mover and x_mover do not put us outside image range
    ix_remove = np.argwhere((y >= i_lim - 1) | (x >= j_lim - 1) | (y < 0) | (x < 0))
    y = np.round(np.delete(y, ix_remove))
    x = np.round(np.delete(x, ix_remove))

    # Extract the values along the line, linear
    zi = scipy.ndimage.map_coordinates(dem_arr, np.vstack((y, x)), order=1)

    # ray from sun
    h = (
        np.sqrt(((y - i) * pix_size) ** 2 + ((x - j) * pix_size) ** 2)
    ) * tan_theta_e + dem_arr[i, j]

    #  create finer resolution
    try:
        interp_func = interpolate.interp1d(
            np.arange(len(zi)), zi, kind="cubic", fill_value="extrapolate"
        )
    except:
        return [i, j, 0]

    fine_zi = interp_func(
        np.linspace(0, len(zi) - 1, 10 * len(zi))
    )  # Upsample by a factor of 10
    fine_h = np.interp(np.linspace(0, len(zi) - 1, 10 * len(zi)), np.arange(len(zi)), h)

    intersections = fine_zi - fine_h

    idx_intersections = np.where(np.diff(np.sign(intersections)))[
        0
    ]  # Change in sign =intersection

    if len(idx_intersections) > 1:
        first_idx = idx_intersections[1]
        last_idx = idx_intersections[-1]

        dx = np.abs(first_idx - last_idx) / 10 * pix_size
        dz = np.abs(fine_zi[last_idx] - fine_zi[first_idx])

        L = np.sum(np.sqrt(dx**2 + dz**2))

    else:
        L = 0  # No valid intersection

    return [i, j, L]


if __name__ == "__main__":
    # inputs
    n_cpu = 12
    output = (
        "/Users/bawilder/Documents/SNOW/LIDAR/Alaska23/dsm_0.5m_small_raytracing.tif"
    )
    dsm_path = "/Users/bawilder/Documents/SNOW/LIDAR/Alaska23/dsm_0.5m_small.tif"
    obs_path = "/Users/bawilder/Code/mm1-alaska/ang_input/apr7/ang20230407t215924_rdn_v2aa4_obs.hdr"
    # inputs

    obs = envi.open(obs_path)
    obs = obs.open_memmap(writeable=True)

    # mean sza and saa
    sza_array = obs[:, :, 4].flatten()
    saa_array = obs[:, :, 3].flatten()
    sza_array = sza_array[sza_array >= 0]
    sza_array = sza_array[sza_array <= 90]
    saa_array = saa_array[saa_array >= -180]

    sza = np.nanmean(sza_array)
    saa = np.nanmean(saa_array)

    print(sza, saa)

    dsm = rio.open(dsm_path)
    crs = dsm.crs
    pix_size, _ = dsm.res
    meta = dsm.meta
    west, south, east, north = dsm.bounds
    dsm = np.squeeze(dsm.read()).astype(float)

    # ray_trace(650,620, dsm, sza, saa, pix_size)

    args = []
    for a in range(dsm.shape[0]):
        for b in range(dsm.shape[1]):
            args.append([a, b, dsm, sza, saa, pix_size])
    ray_trace_results = run_ray_pool(args, n_cpu=n_cpu)

    # Then, fills the array structure.
    ray_array = np.empty_like(dsm)
    for z in ray_trace_results:
        ray_array[z[0], z[1]] = z[2]

    # write to array
    with rio.open(output, "w", **meta) as ds:
        ds.write(ray_array, 1)
