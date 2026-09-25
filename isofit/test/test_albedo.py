from datetime import datetime
import glob
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from spectral.io import envi

from isofit.test import albedo

albedo_pairs_dir = "/Users/bawilder/Code/isofit-PRs/isofit/test/emit_albedo_pairs"

obs_data_dict = {
    "dozier": albedo.dozier_obs_data,
    "sbsp": albedo.sbsp_obs_data,
    "tablerock": albedo.tablerock_obs_data,
    "grandmesa": albedo.grandmesa_obs_data,
    "snotel335": albedo.snotel335_obs_data,
    "snotel365": albedo.snotel365_obs_data,
    "snotel737": albedo.snotel737_obs_data,
    "snotel825": albedo.snotel825_obs_data,
    "niwot": albedo.niwot_obs_data,
    "cper": albedo.cper_obs_data,
    "neonng": albedo.neonng_obs_data,
}

obs_data = {(s, d): {"val": v, "site": s} for s, src in obs_data_dict.items() for d, v in src.items()}
site_name_map = {k: k for k in obs_data_dict}


scene_records = []
if os.path.exists(albedo_pairs_dir):
  for base_folder in os.listdir(albedo_pairs_dir):
    site_key = site_name_map.get(base_folder)
    site_dir = os.path.join(albedo_pairs_dir, base_folder)
    if not site_key or not os.path.isdir(site_dir):
      continue

    for d in os.listdir(site_dir):
      if not re.match(r"^\d{8}$", d) or (site_key, d) not in obs_data:
        continue

      output_dir = os.path.join(site_dir, d, "output")
      snow_hdr = glob.glob(os.path.join(output_dir, "*_snow.hdr"))
      if snow_hdr:
        scene_records.append({
            "date": d,
            "obs": obs_data[(site_key, d)]["val"],
            "site": site_key,
            "snow_hdr": snow_hdr[0],
            "uncert_hdr": glob.glob(os.path.join(output_dir, "*_snow_uncert.hdr")) or [None],
        })




obs_list, pred_list, uncert_list, site_list, matched_scenes = [], [], [], [], []

for rec in scene_records:
  snow_data = envi.open(rec["snow_hdr"]).open_memmap(interleave="bip")
  snow_val = snow_data[0, 0, 1]

  u_val = np.nan
  if rec["uncert_hdr"][0]:
    u_val = envi.open(rec["uncert_hdr"][0]).open_memmap(interleave="bip")[0, 0, 1]

  if snow_val != -9999 and not np.isnan(snow_val):
    obs_list.append(rec["obs"])
    pred_list.append(snow_val)
    uncert_list.append(u_val if not np.isnan(u_val) else 0.0)
    site_list.append(rec["site"])
    matched_scenes.append(rec)

obs_arr, pred_arr, uncert_arr, site_arr = np.array(obs_list), np.array(pred_list), np.array(uncert_list), np.array(site_list)
n_pts = len(obs_arr)



rmse = np.sqrt(np.mean((pred_arr - obs_arr) ** 2))
mean_bias = np.mean(pred_arr - obs_arr)
r2 = 1 - (np.sum((obs_arr - pred_arr) ** 2) / np.sum((obs_arr - np.mean(obs_arr)) ** 2)) 

reg_slope, reg_intercept = np.polyfit(obs_arr, pred_arr, 1)
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")










plt.figure(figsize=(7, 7))

plot_color = {
    "dozier": ("royalblue", "The Jeff Dozier Study Site"),
    "sbsp": ("darkorange", "Senator Beck Study Plot"),
    "tablerock": ("forestgreen", "SurfRad-Table Rock"),
    "grandmesa": ("purple", "Grand Mesa Study Plot"),
    "snotel335": ("black", "SNOTEL-335"),
    "snotel365": ("red", "SNOTEL-365"),
    "snotel737": ("magenta", "SNOTEL-737"),
    "snotel825": ("teal", "SNOTEL-825"),
    "niwot": ("gray", "NEON-Niwot"),
    "cper": ("darkturquoise", "NEON-CPER"),
    "neonng": ("olivedrab", "NEON-NG"),
}

line_x = np.linspace(0.4, 1.0, 100)
plt.plot(line_x, line_x * 1.10, color="black", linestyle="-", linewidth=1.2, label="+/- 10% absorption")
plt.plot(line_x, line_x * 0.90, color="black", linestyle="-", linewidth=1.2)

for site, (color, label) in plot_color.items():
  mask = site_arr == site
  if np.any(mask):
    plt.errorbar(obs_arr[mask], pred_arr[mask], xerr=albedo.ASSUMED_ERROR, yerr=uncert_arr[mask],
                 fmt="o", color=color, ecolor=color, elinewidth=1.0, capsize=2,
                 markeredgecolor="k", markersize=7, zorder=3, label=label)

min_val = min(obs_arr.min(), pred_arr.min()) - 0.07
max_val = max(obs_arr.max(), pred_arr.max()) + 0.07

plt.plot([min_val, max_val], [min_val, max_val], color="gray", linestyle="--", linewidth=1.5, zorder=2, label="1:1 line")

reg_x = np.array([min_val, max_val])
plt.plot(reg_x, reg_slope * reg_x + reg_intercept, color="red", linestyle="-", linewidth=0.5, alpha=0.7, zorder=4, label=f"Regression: y={reg_slope:.2f}x+{reg_intercept:.2f}")

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.gca().set_aspect("equal", adjustable="box")

plt.xlabel("Observed broadband snow albedo [-]", fontsize=12)
plt.ylabel("Modeled broadband snow albedo [-]", fontsize=12)
plt.title(f"Run: {run_timestamp}", fontsize=12)


plt.gca().text(0.05, 0.95, 
               f"n = {n_pts}\nRMSE = {rmse:.4f}\nBias = {mean_bias:.4f}\n$R^2$ = {r2:.4f}\nSlope = {reg_slope:.3f}\nIntercept = {reg_intercept:.3f}", 
               transform=plt.gca().transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"))

plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()











f_snow_arr = np.array([envi.open(rec["snow_hdr"]).open_memmap(interleave="bip")[0, 0, 6] for rec in matched_scenes])
absorption_error = np.abs(pred_arr - obs_arr) / obs_arr

valid_fit_mask = (~np.isnan(f_snow_arr)) & (~np.isnan(absorption_error))
x_fit, y_fit = f_snow_arr[valid_fit_mask], absorption_error[valid_fit_mask]

sort_idx = np.argsort(x_fit)
x_sorted, y_sorted = x_fit[sort_idx], y_fit[sort_idx]

lin_coeffs = np.polyfit(x_sorted, y_sorted, 1)
y_lin_fit = np.polyval(lin_coeffs, x_sorted)
r2_lin = 1 - (np.sum((y_sorted - y_lin_fit) ** 2) / np.sum((y_sorted - np.mean(y_sorted)) ** 2))

plt.figure(figsize=(8, 6))
plt.scatter(x_sorted, y_sorted, color="teal", edgecolors="k", s=65, zorder=3, label="Data points")
plt.plot(x_sorted, y_lin_fit, color="darkblue", 
         linewidth=2, linestyle="-", 
         label=f"y = {lin_coeffs[0]:.3f}x + {lin_coeffs[1]:.3f} ($R^2$ = {r2_lin:.3f})")

plt.xlabel("f_snow [-]", fontsize=12)
plt.ylabel("Absolute absorption error [-]", fontsize=12)
plt.legend(loc="best", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()