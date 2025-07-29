import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import argparse
import glob


#variables = ['CWDC', 'DEADSTEMC', 'GPP', 'HR', 'NEE', 'SOIL1C', 'SOIL2C', 'SOIL3C', 'SOIL4C', 'TLAI', 'TOTCOLC', 'TOTSOMC']
variables = ['GPP']

def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    return max(files, key=os.path.getmtime)


def get_display_unit_and_scaled(data, var, ds):
    # Try to get area, landfrac, landmask
    area = ds["area"].values if "area" in ds.variables else 1.0
    landfrac = ds["landfrac"].values if "landfrac" in ds.variables else 1.0
    landmask = ds["landmask"].values if "landmask" in ds.variables else None
    units = var.attrs.get("units", "")
    display_unit = units
    # Mask
    if landmask is not None:
        mask = (landmask == 1)
        data = np.where(mask, data, np.nan)
    # Area/landfrac scaling
    weighted = data * area * landfrac
    # Unit conversion
    if units.endswith("gC/m^2/s"):
        weighted = weighted * 365 * 24 * 3600  # gC/m^2/s to gC/m^2/year, then to tonC/km^2/year
        display_unit = "tonC/km^2/year"
    elif units.endswith("gC/m^2"):
        weighted = weighted  # gC/m^2 to tonC/km^2
        display_unit = "tonC/km^2"
    return weighted, display_unit


def plot_variable_comparison(var, data1, data2, lon, lat, label1, label2, save_path, unit1, unit2):
    diff = data1 - data2
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)
    proj = ccrs.PlateCarree()
    vmin_1 = float(np.nanmin(data1))
    vmax_1 = float(np.nanmax(data1))
    vmin_2 = float(np.nanmin(data2))
    vmax_2 = float(np.nanmax(data2))
    vmin_orig = min(vmin_1, vmin_2)
    vmax_orig = max(vmax_1, vmax_2)
    diff_abs_max = float(np.nanmax(np.abs(diff)))
    # Plot 1
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    im1 = ax1.pcolormesh(lon, lat, data1, transform=proj, vmin=vmin_orig, vmax=vmax_orig, cmap='viridis')
    ax1.set_title(f'{var} - {label1}', fontsize=14, fontweight='bold')
    ax1.set_global()
    gl1 = ax1.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7, pad=0.05)
    cbar1.set_label(f'{unit1}', fontsize=12)
    # Plot 2
    ax2 = fig.add_subplot(gs[1, 0], projection=proj)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax2.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    im2 = ax2.pcolormesh(lon, lat, data2, transform=proj, vmin=vmin_orig, vmax=vmax_orig, cmap='viridis')
    ax2.set_title(f'{var} - {label2}', fontsize=14, fontweight='bold')
    ax2.set_global()
    gl2 = ax2.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7, pad=0.05)
    cbar2.set_label(f'{unit2}', fontsize=12)
    # Plot diff
    ax3 = fig.add_subplot(gs[2, 0], projection=proj)
    ax3.add_feature(cfeature.COASTLINE)
    ax3.add_feature(cfeature.BORDERS)
    ax3.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax3.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    norm = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0, vmax=diff_abs_max)
    im3 = ax3.pcolormesh(lon, lat, diff, transform=proj, norm=norm, cmap='RdBu_r')
    ax3.set_title(f'{var} - Difference ({label1} - {label2})', fontsize=14, fontweight='bold')
    ax3.set_global()
    gl3 = ax3.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl3.top_labels = False
    gl3.right_labels = False
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.7, pad=0.05)
    cbar3.set_label(f'Δ{var}', fontsize=12)
    plt.suptitle(f'Variable {var} Comparison Analysis', fontsize=18, fontweight='bold', y=0.96)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare variables between NetCDF files.')
    parser.add_argument('--latest', type=str, default=None, help='Path to the latest result NetCDF file (file 1).')
    parser.add_argument('--previous', type=str, default=None, help='Path to a previous result NetCDF file (file 2).')
    parser.add_argument('--model', type=str, default='model_780_h0_results.nc', help='Path to the model reference NetCDF file.')
    parser.add_argument('--search_pattern', type=str, default='../*.nc', help='Glob pattern to search for NetCDF files.')
    args = parser.parse_args()

    # Find latest file if not provided
    if args.latest is None:
        args.latest = get_latest_file(args.search_pattern)
        print(f"Auto-selected latest file: {args.latest}")
    if args.previous is None:
        nc_files = sorted(glob.glob(args.search_pattern))
        prev_candidates = [f for f in nc_files if f != args.latest]
        if not prev_candidates:
            raise FileNotFoundError("No previous file found.")
        args.previous = prev_candidates[-1]
        print(f"Auto-selected previous file: {args.previous}")
    model_file = args.model
    # Output folders
    latest_model_dir = 'latest_model'
    latest_reference_dir = 'latest_reference'
    os.makedirs(latest_model_dir, exist_ok=True)
    os.makedirs(latest_reference_dir, exist_ok=True)
    # Open datasets
    ds_latest = xr.open_dataset(args.latest)
    ds_prev = xr.open_dataset(args.previous)
    ds_model = xr.open_dataset(model_file)
    lon = ds_latest['lon']
    lat = ds_latest['lat']
    print(f"\nComparing variables: {variables}\n")
    for var in variables:
        if var not in ds_latest.data_vars or var not in ds_model.data_vars:
            print(f"Variable {var} not found in latest/model, skipping...")
        else:
            v_latest = ds_latest[var]
            v_model = ds_model[var]
            data_latest = v_latest.isel(time=0).values if 'time' in v_latest.dims else v_latest.values
            data_model = v_model.isel(time=0).values if 'time' in v_model.dims else v_model.values
            data_latest_scaled, unit_latest = get_display_unit_and_scaled(data_latest, v_latest, ds_latest)
            data_model_scaled, unit_model = get_display_unit_and_scaled(data_model, v_model, ds_model)
            save_path = os.path.join(latest_model_dir, f'{var}_comparison.png')
            plot_variable_comparison(var, data_latest_scaled, data_model_scaled, lon, lat, 'Latest', 'Model', save_path, unit_latest, unit_model)
        if var not in ds_latest.data_vars or var not in ds_prev.data_vars:
            print(f"Variable {var} not found in latest/previous, skipping...")
        else:
            v_latest = ds_latest[var]
            v_prev = ds_prev[var]
            data_latest = v_latest.isel(time=0).values if 'time' in v_latest.dims else v_latest.values
            data_prev = v_prev.isel(time=0).values if 'time' in v_prev.dims else v_prev.values
            data_latest_scaled, unit_latest = get_display_unit_and_scaled(data_latest, v_latest, ds_latest)
            data_prev_scaled, unit_prev = get_display_unit_and_scaled(data_prev, v_prev, ds_prev)
            save_path = os.path.join(latest_reference_dir, f'{var}_comparison.png')
            plot_variable_comparison(var, data_latest_scaled, data_prev_scaled, lon, lat, 'Latest', 'Previous', save_path, unit_latest, unit_prev)
    ds_latest.close()
    ds_prev.close()
    ds_model.close()
    print("\nAll variable comparison plots are complete!")
    print(f"Images saved in: {latest_model_dir}/ and {latest_reference_dir}/")

if __name__ == '__main__':
    main()