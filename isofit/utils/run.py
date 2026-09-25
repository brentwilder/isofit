import glob
import os
import numpy as np
from spectral.io import envi

albedo_pairs_dir = '/Users/bawilder/Code/isofit-PRs/isofit/test/emit_albedo_pairs'

site_coords = {
    'dozier': (37.64315005500472, -119.02909564511828),
    'sbsp': (37.906883, -107.726265),
    'tablerock': (40.12498, -105.2368),
    'table_rock': (40.12498, -105.2368),
    'grandmesa': (39.050802, -108.061435),
    'grand_mesa': (39.050802, -108.061435),
    'snotel335': (39.80364, -105.77786),
    'snotel365': (45.89107, -110.93851),
    'snotel737': (39.01467, -107.04933),
    'snotel825': (40.53740, -106.67655),
    'niwot': (40.0543, -105.5824),
    'cper': (40.81550, -104.74560),
    'neonng': (46.76970, -100.91540),
}

for site_folder in os.listdir(albedo_pairs_dir):
  if site_folder.startswith('.'):
    continue
  site_path = os.path.join(albedo_pairs_dir, site_folder)
  if not os.path.isdir(site_path):
    continue
    
  site_key = site_folder.lower()
  if site_key not in site_coords:
    continue
    
  lat, lon = site_coords[site_key]

  for date_str in os.listdir(site_path):
    if date_str.startswith('.'):
      continue
    # Target the 'data' subdirectory where the raw inputs live
    scene_dir = os.path.join(site_path, date_str, 'data')
    if not os.path.exists(scene_dir):
      continue
      
    loc_hdrs = glob.glob(os.path.join(scene_dir, '*_loc.hdr')) + glob.glob(os.path.join(scene_dir, '*_LOC.hdr'))
    for hdr_path in loc_hdrs:
      bin_path = hdr_path[:-4]
      if not os.path.exists(bin_path):
        continue
      
      img = envi.open(hdr_path, bin_path)
      mmap = img.open_memmap(interleave='bip', writable=True)
      
      mmap[:, :, 0] = lon
      mmap[:, :, 1] = lat
      mmap.flush()
      print(f"Forced update LOC: {site_folder}/{date_str} -> Lat: {lat}, Lon: {lon}")