import xarray as xr
import numpy as np
from joblib import Parallel, delayed

from isofit.core.common import resample_spectrum

def load_sensor_spec(filepath):
    """
    Loads sensor data
    """
    data = np.loadtxt(filepath)
    wl = data[:, 1]
    fwhm = data[:, 2] 
    return wl, fwhm



def resample_disort_lut(
    input_nc_path,
    output_nc_path,
    sensor_spec_file,
    n_cpus,
):
    # Load sensor spec from file
    sensor_wl, sensor_fwhm = load_sensor_spec(sensor_spec_file)

    # Load the DISORT LUT
    ds = xr.load_dataset(input_nc_path)
    disort_wl = ds['wavelength'].values

   # Parallel resampling function
    def resample_variable(data):
        shape = data.shape[:-1]
        data_reshaped = data.reshape(-1, data.shape[-1])

        resampled = Parallel(n_jobs=n_cpus)(
            delayed(resample_spectrum)(spectrum, disort_wl, sensor_wl, sensor_fwhm)
            for spectrum in data_reshaped
        )

        return np.stack(resampled).reshape(*shape, len(sensor_wl))

    brdf_resampled = resample_variable(ds['brdf'].values)
    hdrf_resampled = resample_variable(ds['hdrf'].values)
    a_diff_resampled = resample_variable(ds['a_diff'].values)

    # save to a new file
    ds_out = xr.Dataset(
        {
            "brdf": (ds['brdf'].dims[:-1] + ('wavelength',), brdf_resampled),
            "hdrf": (ds['hdrf'].dims[:-1] + ('wavelength',), hdrf_resampled),
            "a_diff": (ds['a_diff'].dims[:-1] + ('wavelength',), a_diff_resampled)
        },
        coords={**{k: ds.coords[k] for k in ds.coords if k != 'wavelength'},
                "wavelength": sensor_wl}
    )

    ds_out.attrs["description"] = "Resampled DISORT LUT for sensor"
    ds_out.attrs["units"] = "wavelength in nanometers"
    ds_out.to_netcdf(output_nc_path)




if __name__ == "__main__":
    ########
    # Paths
    input_nc = "/Users/bawilder/Code/snow/LUT/qaanaaq_ice_lut.nc"
    sensor_file = "/Users/bawilder/Code/isofit-snow/pipeline/ang-wave.txt"
    output_nc = "/Users/bawilder/Code/snow/LUT/qaanaaq_ice_lut_ANG.nc"
    ########


    resample_disort_lut(input_nc, output_nc, sensor_file, n_cpus=12)
