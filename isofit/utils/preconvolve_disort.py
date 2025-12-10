import xarray as xr
import numpy as np
from joblib import Parallel, delayed



def resample_spectrum(
    x: np.array,
    wl: np.array,
    wl2: np.array,
    fwhm2: np.array,
    fill: bool = False,
    srf_file: str = None,
) -> np.array:
    """Resample a spectrum to a new wavelength / FWHM.
       Assumes Gaussian SRFs.

    Args:
        x: radiance vector
        wl: sample starting wavelengths
        wl2: wavelengths to resample to
        fwhm2: full-width-half-max at resample resolution
        fill: boolean indicating whether to fill in extrapolated regions
        ### sc Adding for non-Gaussian SRF ###
        srf_file: SRF for the sensor if not assuming Gaussian

    Returns:
        np.array: interpolated radiance vector

    """
    # sc including if else to add non-Gaussian SRF
    # Probably a better way than this with file paths, etc.
    if srf_file is None:
        H = np.array(
            [
                spectral_response_function(wl, wi, fwhmi / 2.355)
                for wi, fwhmi in zip(wl2, fwhm2)
            ]
        )
        H[np.isnan(H)] = 0
    else:
        # Loading in user-supplied srf file and assigning variables
        srf_data = xr.open_dataset(srf_file)
        sensor_rsr, rsr_wls = srf_data.RSR.data, srf_data.wavelength.data

        # Grabbing indices of rsr wavelengths which match sensor wavelengths (wl2)
        idx = [np.argwhere(abs(rsr_wls - wl2[i]) < 0.1)[0] for i in range(len(wl2))]
        idx = np.asarray(idx).flatten()

        # Getting RSR at sensor spectral resolution
        rsr_channel_res = sensor_rsr[:, idx]

        # Normalize H to unit length
        H = rsr_channel_res / np.sum(rsr_channel_res, axis=1)[:, np.newaxis]
        H[np.isnan(H)] = 0

    dims = len(x.shape)
    if fill:
        if dims > 1:
            raise Exception("resample_spectrum(fill=True) only works with vectors")

        x = x.reshape(-1, 1)
        xnew = np.dot(H, x).ravel()
        good = np.isfinite(xnew)
        for i, xi in enumerate(xnew):
            if not good[i]:
                nearest_good_ind = np.argmin(abs(wl2[good] - wl2[i]))
                xnew[i] = xnew[nearest_good_ind]
        return xnew
    else:
        # Matrix
        if dims > 1:
            return np.dot(H, x.T).T

        # Vector
        else:
            x = x.reshape(-1, 1)
            return np.dot(H, x).ravel()


def load_spectrum(spectrum_file: str) -> (np.array, np.array):
    """Load a single spectrum from a text file with initial columns giving
       wavelength and magnitude, respectively.

    Args:
        spectrum_file: file to load spectrum from

    Returns:
        np.array: spectrum values
        np.array: wavelengths, if available in the file

    """

    spectrum = np.loadtxt(spectrum_file)
    if spectrum.ndim > 1:
        spectrum = spectrum[:, :2]
        wavelengths, spectrum = spectrum.T
        if wavelengths[0] < 100:
            wavelengths = wavelengths * 1000.0  # convert microns -> nm if needed
        return spectrum, wavelengths
    else:
        return spectrum, None


def spectral_response_function(response_range: np.array, mu: float, sigma: float):
    """Calculate the spectral response function.

    Args:
        response_range: signal range to calculate over
        mu: mean signal value
        sigma: signal variation

    Returns:
        np.array: spectral response function

    """

    u = (response_range - mu) / abs(sigma)
    y = (1.0 / (np.sqrt(2.0 * np.pi) * abs(sigma))) * np.exp(-u * u / 2.0)
    srf = y / y.sum()
    return srf





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

    brdf_resampled = resample_variable(ds['r_dd'].values)
    hdrf_resampled = resample_variable(ds['r_hd'].values)
    a_dir_resampled = resample_variable(ds['a_dd'].values)
    a_diff_resampled = resample_variable(ds['a_hd'].values)

    # save to a new file
    ds_out = xr.Dataset(
        {
            "r_dd": (ds['r_dd'].dims[:-1] + ('wavelength',), brdf_resampled),
            "r_hd": (ds['r_hd'].dims[:-1] + ('wavelength',), hdrf_resampled),
            "a_dd": (ds['a_dd'].dims[:-1] + ('wavelength',), a_dir_resampled),
            "a_hd": (ds['a_hd'].dims[:-1] + ('wavelength',), a_diff_resampled)
        },
        coords={**{k: ds.coords[k] for k in ds.coords if k != 'wavelength'},
                "wavelength": sensor_wl}
    )

    ds_out.attrs["description"] = "Resampled DISORT LUT for EMIT sensor. Optically thick snow layer."
    ds_out.attrs["units"] = "wavelength in nanometers"
    ds_out.to_netcdf(output_nc_path)




if __name__ == "__main__":
    ########
    # Paths
    input_nc = "/Users/bawilder/Code/snow/LUT/disort_snow_lut.nc"
    sensor_file = "/Users/bawilder/Code/melt-metrics/bw-wave/emit-wave.txt"
    output_nc = "/Users/bawilder/Code/snow/LUT/disort_snow_lut_EMIT.nc"
    ########


    resample_disort_lut(input_nc, output_nc, sensor_file, n_cpus=12)
