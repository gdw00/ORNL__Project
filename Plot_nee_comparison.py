import matplotlib
matplotlib.use('Agg') 
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['grid.linewidth'] = 0.8


def compute_nee(nc_path):

    with xr.open_dataset(nc_path) as ds:
        area_km2 = ds['area'].astype(np.float64)
        landfrac = ds['landfrac'].astype(np.float64)
        NEE = ds['NEE'].astype(np.float64)  #  gC/m²/s
        
        secs_per_yr = 3600.0 * 24.0 * 365  
        nee_total = (NEE * area_km2 * landfrac * secs_per_yr).sum(dim=("lat", "lon")).item()
    return nee_total


ai_spinup_file = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_spinup/e3sm_run/20250408_trendytest_ICB1850CNPRDCTCBC/run/20250408_trendytest_ICB1850CNPRDCTCBC.elm.h0.0101-01-01-00000.nc"
normal_spinup_file = "/gpfs/wolf2/cades/cli185/proj-shared/zdr/sparsegrid/test_cases/20240903_TRENDY_f09_ICB1850CNPRDCTCBC/run/20240903_TRENDY_f09_ICB1850CNPRDCTCBC.elm.h0.0401-01-01-00000.nc"


nee_ai = compute_nee(ai_spinup_file)
nee_norm = compute_nee(normal_spinup_file)
nee_diff = nee_ai - nee_norm


bars = [nee_ai, nee_norm, nee_diff]
labels = ["NEE (AI Spinup)", "NEE (Normal Spinup)", "Difference (AI - Normal)"]
colors = ["tab:blue", "tab:orange", "tab:green"]

fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(bars))
bar_width = 0.6
bars_plot = ax.bar(x, bars, width=bar_width, color=colors)


y_range = max(bars) - min(bars)
offset = y_range * 0.02


for i, v in enumerate(bars):
    va = 'bottom' if v >= 0 else 'top'
    y_text = v + offset if v >= 0 else v - offset
    ax.text(x[i], y_text, f"{v:.5f}", ha='center', va=va, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("NEE (summed over lat, lon) [ton C / year]")
ax.set_title("NEE Comparison and Difference (AI Spinup vs Normal Spinup)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("./4_new_results_NEE_ALL/nee_comparison_bar_chart_scientific3.png", dpi=300)
plt.show()
