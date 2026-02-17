import numpy as np
import ray
from scipy.ndimage import uniform_filter
import os
from glob import glob

from isofit.inversion.inverse_simple import invert_algebraic, invert_simple
from isofit.core.fileio import IO
from isofit.core.forward import ForwardModel
from isofit.configs import configs


def bkg_heuristic_estimate(working_directory):
    """NOTE: assumes NaN to be 0.25 background"""

    # weights based on example in Richter 1998, `Correction of satellite imagery over mountainous terrain`
    # TODO: this does not apply currently for airborne retrieval.
    radii_frac = [0.0, 0.45, 0.65, 0.80, 0.90, 1.0]
    weights = [0.24, 0.24, 0.22, 0.15, 0.15]

    @ray.remote
    def invert_chunk(row_chunk, cols, config, fm):
        io = IO(config, fm)
        n_bands = len(fm.surface.idx_lamb)
        rfl_chunk = np.zeros((len(row_chunk), len(cols), n_bands), dtype=np.float32)

        # estimate on center of chunk
        center_r = row_chunk[len(row_chunk) // 2]
        center_c = cols[len(cols) // 2]
        center_data = io.get_components_at_index(center_r, center_c, bkg_solve=True)

        # Simple inversion at center
        x_center = invert_simple(fm, center_data.meas, center_data.geom)
        _, _, x_instr = fm.unpack(fm.init.copy())

        # Iterate accross chunk
        for i, r in enumerate(row_chunk):
            for c in cols:
                input_data = io.get_components_at_index(r, c, bkg_solve=True)
                if input_data is None or input_data.meas is None:
                    rfl_chunk[i, c, :] = np.nan
                    continue

                rfl_est, _, _ = invert_algebraic(
                    fm.surface,
                    fm.RT,
                    fm.instrument,
                    x_center[fm.idx_surface],
                    x_center[fm.idx_RT],
                    x_instr,
                    input_data.meas,
                    input_data.geom,
                )
                rfl_chunk[i, c, :] = rfl_est

        return row_chunk, rfl_chunk

    def calc_rho_e(
        cube, max_radius_km, pixel_size_m, radii_frac=None, weights=None, terrain=False
    ):
        max_r_px = int(round(max_radius_km * 1000 / pixel_size_m))

        # Terrain case: all contained within 0.45 km ring (approx. 0.5 km)
        if terrain:
            r = max_r_px
            size = 2 * r + 1
            avg = uniform_filter(cube, size=(size, size, 1), mode="nearest")
            return np.nan_to_num(avg, nan=0.25)

        # Dif-dif and dir-dif case. 1km.
        radii_px = [int(round(r * max_r_px)) for r in radii_frac]
        weighted_avg = np.zeros_like(cube)

        for i in range(len(weights)):
            r_in = radii_px[i]
            r_out = radii_px[i + 1]

            size_out = 2 * r_out + 1
            area_out = size_out**2
            avg_outer = uniform_filter(
                cube, size=(size_out, size_out, 1), mode="nearest"
            )

            if r_in > 0:
                size_in = 2 * r_in + 1
                area_in = size_in**2
                avg_inner = uniform_filter(
                    cube, size=(size_in, size_in, 1), mode="nearest"
                )
                annulus_avg = (avg_outer * area_out - avg_inner * area_in) / (
                    area_out - area_in
                )
            else:
                annulus_avg = avg_outer

            weighted_avg += weights[i] * annulus_avg

        return np.nan_to_num(weighted_avg, nan=0.25)

    config = configs.create_new_config(
        glob(os.path.join(working_directory, "config", "") + "*_isofit.json")[0]
    )

    fm = ForwardModel(config)
    io = IO(config, fm)
    rows = io.n_rows
    cols = io.n_cols
    range_rows = range(rows)
    range_cols = range(cols)
    pixel_size = config.implementation.pixel_size

    dir_path = os.path.dirname(config.input.loc_file)
    rho_e_path = os.path.join(dir_path, "rho_e.npy")
    rho_terrain_path = os.path.join(dir_path, "rho_terrain.npy")

    if not os.path.exists(rho_e_path):
        # Chunk size
        chunk_size = 50
        row_chunks = [
            list(range(i, min(i + chunk_size, rows)))
            for i in range(0, rows, chunk_size)
        ]

        params = [ray.put(obj) for obj in [range_cols, config, fm]]
        futures = [invert_chunk.remote(chunk, *params) for chunk in row_chunks]

        # Collect results and combine
        rfl_cube = np.zeros((rows, cols, len(fm.surface.idx_lamb)), dtype=np.float32)
        for row_chunk, rfl_chunk in ray.get(futures):
            for i, r in enumerate(row_chunk):
                rfl_cube[r, :, :] = rfl_chunk[i, :, :]

        @ray.remote
        def calc_rho_e_ray(*args, **kwargs):
            return calc_rho_e(*args, **kwargs)

        # 1km for rho_e and 0.5 km for rho_terrain
        rho_e_future = calc_rho_e_ray.remote(
            rfl_cube, 1.0, pixel_size, radii_frac, weights, terrain=False
        )
        rho_terrain_future = calc_rho_e_ray.remote(
            rfl_cube, 0.5, pixel_size, terrain=True
        )
        rho_e, rho_terrain = ray.get([rho_e_future, rho_terrain_future])

        # save to float 16 because it is sufficent for bkg solve
        np.save(rho_e_path, rho_e.astype(np.float16))
        np.save(rho_terrain_path, rho_terrain.astype(np.float16))

    del io
    del fm

    return
