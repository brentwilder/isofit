import numpy as np
import ray
from scipy.ndimage import uniform_filter
import os
from glob import glob
from spectral.io import envi

from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.configs import configs
from isofit.inversion.inverse_simple import invert_algebraic

def bkg_heuristic_estimate(working_directory):
    """NOTE: assumes NaN to be 0.25 background"""

    # weights based on example in Richter 1998, `Correction of satellite imagery over mountainous terrain`
    radii_frac = [0.0, 0.45, 0.65, 0.80, 0.90, 1.0]
    weights = [0.24, 0.24, 0.22, 0.15, 0.15]

    @ray.remote
    def invert_chunk(row_chunk, cols, config, fm):
  
        rad_img = envi.open(config.input.measured_radiance_file + '.hdr', config.input.measured_radiance_file)
        rad_mm = rad_img.open_memmap(interleave='bip', writable=False)
        
        loc_img = envi.open(config.input.loc_file + '.hdr', config.input.loc_file)
        loc_mm = loc_img.open_memmap(interleave='bip', writable=False)

        obs_img = envi.open(config.input.obs_file + '.hdr', config.input.obs_file)
        obs_mm = obs_img.open_memmap(interleave='bip', writable=False)
        
        from isofit.core.fileio import IO
        tmp_io = IO(config, fm)
        esd_reference = tmp_io.esd
        del tmp_io
        
        n_bands = len(fm.surface.idx_lamb)
        rfl_chunk = np.zeros((len(row_chunk), len(cols), n_bands), dtype=np.float32)

        x_center = fm.init.copy()
        x_surface, x_RT, x_instrument = fm.unpack(x_center)
        
        for h2oname in ["H2OSTR", "h2o"]:
            if h2oname in fm.RT.statevec_names:
                x_RT[fm.RT.statevec_names.index(h2oname)] = 0.5

        for i, r in enumerate(row_chunk):
            for j, c in enumerate(cols):
                # Detach from memmap safely for Ray/Scipy
                meas_pixel = np.array(rad_mm[r, c, :], dtype=np.float32)
                obs_pixel = np.array(obs_mm[r, c, :], dtype=np.float32)
                loc_pixel = np.array(loc_mm[r, c, :], dtype=np.float32)
                
                if meas_pixel[0] < -250.0 or np.isnan(meas_pixel[0]):
                    rfl_chunk[i, j, :] = 0.25
                    continue

                else:
                    local_geom = Geometry(esd=esd_reference, svf=1.0, obs=obs_pixel, loc=loc_pixel, slope=0.0)
                    
                    rfl_est, _, _ = invert_algebraic(
                        surface=fm.surface,
                        RT=fm.RT,
                        instrument=fm.instrument,
                        x_surface=x_surface,
                        x_RT=x_RT,
                        x_instrument=x_instrument,
                        meas=meas_pixel,
                        geom=local_geom
                    )
                    rfl_chunk[i, j, :] = rfl_est


        return row_chunk, rfl_chunk

    def calc_rho_e(cube, max_radius_km, pixel_size_m, radii_frac=None, weights=None, terrain=False):
        max_r_px = int(round(max_radius_km * 1000 / pixel_size_m))

        if terrain:
            r = max_r_px
            size = 2 * r + 1
            avg = uniform_filter(cube, size=(size, size, 1), mode='nearest')
            return avg

        radii_px = [int(round(r * max_r_px)) for r in radii_frac]
        weighted_avg = np.zeros_like(cube)

        for i in range(len(weights)):
            r_in = radii_px[i]
            r_out = radii_px[i + 1]

            size_out = 2 * r_out + 1
            area_out = size_out ** 2
            avg_outer = uniform_filter(cube, size=(size_out, size_out, 1), mode='nearest')

            if r_in > 0:
                size_in = 2 * r_in + 1
                area_in = size_in ** 2
                avg_inner = uniform_filter(cube, size=(size_in, size_in, 1), mode='nearest')
                annulus_avg = (avg_outer * area_out - avg_inner * area_in) / (area_out - area_in)
            else:
                annulus_avg = avg_outer

            weighted_avg += weights[i] * annulus_avg

        return weighted_avg
    
    config = configs.create_new_config(glob(os.path.join(working_directory, "config", "") + "*_isofit.json")[0])
    
    fm = ForwardModel(config)
    rad_img = envi.open(config.input.measured_radiance_file + '.hdr', config.input.measured_radiance_file)
    rows = int(rad_img.metadata['lines'])
    cols = int(rad_img.metadata['samples'])
    range_cols = range(cols)
    pixel_size = config.implementation.pixel_size

    dir_path = os.path.dirname(config.input.loc_file)
    rho_e_path = os.path.join(dir_path, "rho_e.npy")
    rho_terrain_path = os.path.join(dir_path, "rho_terrain.npy")

    if not os.path.exists(rho_e_path):
        chunk_size = 50
        row_chunks = [list(range(i, min(i + chunk_size, rows))) for i in range(0, rows, chunk_size)]
        
        params = [ray.put(obj) for obj in [range_cols, config, fm]]
        futures = [invert_chunk.remote(chunk, *params) for chunk in row_chunks]

        rfl_cube = np.zeros((rows, cols, len(fm.surface.idx_lamb)), dtype=np.float32)
        for row_chunk, rfl_chunk in ray.get(futures):
            for i, r in enumerate(row_chunk):
                rfl_cube[r, :, :] = rfl_chunk[i, :, :]

        @ray.remote
        def calc_rho_e_ray(*args, **kwargs):
            return calc_rho_e(*args, **kwargs)

        rho_e_future = calc_rho_e_ray.remote(rfl_cube, 1.0, pixel_size, radii_frac, weights, terrain=False)
        rho_terrain_future = calc_rho_e_ray.remote(rfl_cube, 0.5, pixel_size, terrain=True)
        rho_e, rho_terrain = ray.get([rho_e_future, rho_terrain_future])

        rho_e = np.copy(rho_e)
        rho_terrain = np.copy(rho_terrain)
        rho_e[np.isnan(rho_e)] = 0.25
        rho_terrain[np.isnan(rho_terrain)] = 0.25


        np.save(rho_e_path, rho_e.astype(np.float16))
        np.save(rho_terrain_path, rho_terrain.astype(np.float16))

    del fm
    return None