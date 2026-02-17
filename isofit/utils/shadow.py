import numpy as np
from spectral import *
import scipy
import multiprocessing as mp
from spectral.io import envi
import os

from isofit.core.common import envi_header


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

    """

    # start = time.time()

    i_lim = dem_arr.shape[0]
    j_lim = dem_arr.shape[1]

    # Compute projected solar directions
    tan_theta_e = np.tan(np.radians(90 - sza))
    tan_sundir = -1 * np.tan(
        np.radians(saa)
    )  # direction flipped because negative is pointing right

    # Assumes image is taken mid-day so mover_i(row-wise / N-S) will always be larger than mover_j
    # starts with high sampling and decreases resolution with increasing distance
    y_mover = np.arange(0, 300.1, 1)
    x_mover = np.round(y_mover * tan_sundir).astype(int)

    # Southern hemisphere (need to move back up the image instead of down the rows)
    if saa > 270 or saa < 90:
        y_mover = y_mover * -1

    if sza == 0.0:  # NaN data, skip
        return [i, j, -9999]

    # Run ray tracing
    y = i + y_mover
    x = j + x_mover

    # make sure y_mover and x_mover did not put outside img range
    ix_remove = np.argwhere((y >= i_lim - 1) | (x >= j_lim - 1) | (y < 0) | (x < 0))
    y = np.round(np.delete(y, ix_remove))
    x = np.round(np.delete(x, ix_remove))

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(dem_arr, np.vstack((y, x)), order=1)

    # Create the sun ray, from the pixel
    h = (
        np.sqrt(((y - i) * pix_size) ** 2 + ((x - j) * pix_size) ** 2)
    ) * tan_theta_e + dem_arr[i, j]

    if ((h[1:] < zi[1:]).any()) == True:
        return [i, j, 0]
    else:
        return [i, j, 1]

    return


def run(input_loc, input_obs, pix_size, n_cpus):

    def flatten_and_stats(array_2d):
        array_1d = array_2d.flatten()
        array_1d = array_1d[(array_1d >= 0) & (array_1d <= 360)]
        mean = np.nanmean(array_1d)
        return mean

    sza = flatten_and_stats(envi.open(envi_header(input_obs), input_obs)[:, :, 4])
    saa = flatten_and_stats(envi.open(envi_header(input_obs), input_obs)[:, :, 3])
    elev = envi.open(envi_header(input_loc), input_loc)[:, :, 2]
    elev[elev < -999] = np.nan

    if len(elev.shape) > 2:
        elev = elev.squeeze()

    rows, cols = elev.shape[0], elev.shape[1]

    list_args = []
    for i in range(rows):
        for j in range(cols):
            list_args.append((i, j, elev, sza, saa, pix_size))

    results = run_ray_pool(list_args, n_cpus)

    shadow_arr = np.full((rows, cols), 1, dtype=np.int8)
    for i, j, val in results:
        shadow_arr[i, j] = val

    np.save(str(os.path.join(os.path.dirname(input_loc), "shadow.npy")), shadow_arr)

    return


# if __name__ == "__main__":
#    input_loc = '/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/ang20240815t160909_008_L1B_ORT_7b1b5d77_LOC'
#    input_obs = '/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/ang20240815t160909_008_L1B_ORT_7b1b5d77_OBS'
#    pix_size = 6.6
#    n_cpus = 12
#    run(input_loc=input_loc,
#        input_obs=input_obs,
#        pix_size=pix_size,
#        n_cpus=n_cpus)
