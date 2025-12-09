import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import interpolate
import os


def load_eco_data(path, WAVE, FWHM):
    baddata=0
    d = pd.read_csv(f'{path}', skiprows=20, sep=r'\s+', header=None)
    d.columns = ['Wavelength', 'Reflectance']
    if np.min(d['Wavelength']) > 0.5:
        baddata = 1
    interp_type = 'linear'
    if 'soil' in path:
        interp_type = 'nearest'
    rho_i = interpolate.interp1d(d['Wavelength'], d['Reflectance'], kind=interp_type, fill_value='extrapolate')
    rho_i = (rho_i(WAVE)).T / 100
    if baddata==1:
        rho_i = rho_i*0 - 9999
    return rho_i

def collect_spectra_by_type(folder, types, WAVE, FWHM):
    grouped = {t: [] for t in types}
    for fname in os.listdir(folder):
        if fname.lower().endswith('.spectrum.txt'):  
            lower_fname = fname.lower()
            if 'vegetation' in lower_fname and 'vswir' in lower_fname and 'tree' in lower_fname:
                t = 'vegetation'
                rho = load_eco_data(os.path.join(folder, fname), WAVE, FWHM)
                grouped[t].append(rho)
            if 'soil' in lower_fname:
                try:
                    t = 'soil'
                    rho = load_eco_data(os.path.join(folder, fname), WAVE, FWHM)
                    if rho[0] >0:
                        grouped[t].append(rho)
                except:
                    print(fname, 'did not work')
            if 'nonphotosyntheticvegetation' in lower_fname and 'vswir' in lower_fname:
                t = 'nonphotosyntheticvegetation'
                rho = load_eco_data(os.path.join(folder, fname), WAVE, FWHM)
                grouped[t].append(rho)
    dfs = {}
    for t in types:
        if grouped[t]:
            df = pd.DataFrame(np.column_stack(grouped[t]), columns=[f"rho_{i}" for i in range(len(grouped[t]))])
            df.insert(0, 'Wavelength', WAVE)
            dfs[t] = df
        else:
            print(f"No spectra found for type: {t}")
    return dfs

def build_pca_model(spectral_data,set_k, variance_threshold=0.99):
    mu = spectral_data.mean(axis=0)
    centered_data = spectral_data - mu
    pca = PCA()
    pca.fit(centered_data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    k = np.searchsorted(cumulative_variance, variance_threshold) + 1
    k = set_k
    V = pca.components_[:k]
    eigenvalues = pca.explained_variance_[:k]
    V = V * np.sqrt(eigenvalues)[:, np.newaxis]

    return mu, V 

def reconstruct_reflectance(x_rfl, V, mu):
    return V.T @ x_rfl + mu

def main(sensor_file, SENSOR, eco_dir, types, output_dir):

    data = np.loadtxt(sensor_file)
    WAVE = data[:, 1] / 1000
    FWHM = data[:, 2] / 1000

    spectra_dfs = collect_spectra_by_type(eco_dir, types, WAVE, FWHM)
    for t, df in spectra_dfs.items():
        df.to_csv(f"{output_dir}/{t}_spectra.csv", index=False)

    npv = f'{output_dir}/nonphotosyntheticvegetation_spectra.csv'
    pv = f'{output_dir}/vegetation_spectra.csv'
    soil = f'{output_dir}/soil_spectra.csv'

    df = pd.read_csv(soil)
    df = df.set_index('Wavelength')
    soil_spectra = df.values.T
    mu_soil, V_soil  = build_pca_model(soil_spectra,1)
    with open(f'{output_dir}/soil_{SENSOR}.pkl', 'wb') as f:
        pickle.dump((mu_soil, V_soil), f)

    df = pd.read_csv(pv)
    df = df.set_index('Wavelength')
    pv_spectra = df.values.T
    mu_pv, V_pv  = build_pca_model(pv_spectra,1)
    with open(f'{output_dir}/pv_{SENSOR}.pkl', 'wb') as f:
        pickle.dump((mu_pv, V_pv ), f)

    df = pd.read_csv(npv)
    df = df.set_index('Wavelength')
    npv_spectra = df.values.T
    mu_npv, V_npv  = build_pca_model(npv_spectra,1)
    with open(f'/{output_dir}/npv_{SENSOR}.pkl', 'wb') as f:
        pickle.dump((mu_npv, V_npv), f)

if __name__ == "__main__":

    SENSOR='ANG'
    sensor_file = "/Users/bawilder/Code/isofit-snow/pipeline/ang-wave.txt"
    output_dir = "/Users/bawilder/.isofit/data/"
    #output_dir = "/Users/bawilder/Documents/AK_REVIEW_PAPER/emit_data"  
    #output_dir = "/Users/bawilder/Documents/AK_REVIEW_PAPER/PRISMA_data"
    #output_dir = "/Users/bawilder/Documents/AK_REVIEW_PAPER/ang_data"


    ECO = '/Users/bawilder/Code/snow/data/ecospeclib-all/'
    TYPES = ['vegetation', 'soil', 'nonphotosyntheticvegetation'] 

    main(sensor_file, SENSOR, ECO, TYPES , output_dir)
