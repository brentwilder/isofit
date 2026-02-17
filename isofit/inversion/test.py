import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("/Users/bawilder/Code/isofit-PRs/local/a3/output/R_F1/libradtran_mus/lut_full/lut.nc")

# Define the terms to plot
terms = ['sphalb', 'rhoatm', 'dif-dir', 'dif-dif']

# Select a single point in the parameter space to view the spectrum
# We'll use .isel() to pick the first index of every dimension except 'wl'
sample = ds[terms].isel(
    AERFRAC_2=1, 
    H2OSTR=1, 
    observer_zenith=0, 
    relative_azimuth=0, 
    surface_elevation_km=0
)

plt.figure(figsize=(10, 6))

for term in terms:
    plt.plot(sample.wl, sample[term], label=term)

plt.title("Coupled Radiative Transfer Terms")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Value (Transmission/Reflectance)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

ds.close()