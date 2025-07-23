import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm


variables = ['CWDC', 'DEADSTEMC', 'GPP', 'HR', 'NEE', 'SOIL1C', 'SOIL2C', 'SOIL3C', 'SOIL4C', 'TLAI', 'TOTCOLC', 'TOTSOMC']


file_500 = 'data/20250408_trendytest_ICB1850CNPRDCTCBC.elm.h0.0501-01-01-00000.nc'
file_780 = 'data/20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0.0781-01-01-00000.nc'


save_path_template = '500-780/{variable}_comparison.png'


output_dir = '500-780'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")


print("Reading data files...")
try:
    ds_500 = xr.open_dataset(file_500)
    ds_780 = xr.open_dataset(file_780)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the data files are in the correct 'data/' directory.")
    exit()


lon = ds_500['lon']
lat = ds_500['lat']


print(f"\n Starting to plot {len(variables)} variable comparisons...")
for i, var in enumerate(variables, 1):
    print(f"\n [{i}/{len(variables)}] Processing variable: {var}")
    
    if var not in ds_500.data_vars or var not in ds_780.data_vars:
        print(f" Variable {var} not found in one of the data files, skipping...")
        continue
    

    data_500 = ds_500[var].isel(time=0) if 'time' in ds_500[var].dims else ds_500[var]
    data_780 = ds_780[var].isel(time=0) if 'time' in ds_780[var].dims else ds_780[var]
    

    diff = data_500 - data_780
    
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)
    
    proj = ccrs.PlateCarree()
    
    vmin_500 = float(np.nanmin(data_500))
    vmax_500 = float(np.nanmax(data_500))
    vmin_780 = float(np.nanmin(data_780))
    vmax_780 = float(np.nanmax(data_780))

    vmin_orig = min(vmin_500, vmin_780)
    vmax_orig = max(vmax_500, vmax_780)
    
    diff_abs_max = float(np.nanmax(np.abs(diff)))
    
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    im1 = ax1.pcolormesh(lon, lat, data_500, transform=proj, 
                         vmin=vmin_orig, vmax=vmax_orig, cmap='viridis')
    ax1.set_title(f'{var} - Year 500', fontsize=14, fontweight='bold')
    ax1.set_global()
    
    gl1 = ax1.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False

    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7, pad=0.05)
    cbar1.set_label(f'Units for {var}', fontsize=12)
    

    ax2 = fig.add_subplot(gs[1, 0], projection=proj)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax2.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    im2 = ax2.pcolormesh(lon, lat, data_780, transform=proj,
                         vmin=vmin_orig, vmax=vmax_orig, cmap='viridis')
    ax2.set_title(f'{var} - Year 780', fontsize=14, fontweight='bold')
    ax2.set_global()
    

    gl2 = ax2.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False

    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7, pad=0.05)
    cbar2.set_label(f'Units for {var}', fontsize=12)
    

    ax3 = fig.add_subplot(gs[2, 0], projection=proj)
    ax3.add_feature(cfeature.COASTLINE)
    ax3.add_feature(cfeature.BORDERS)
    ax3.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax3.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    norm = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0, vmax=diff_abs_max)
    im3 = ax3.pcolormesh(lon, lat, diff, transform=proj, 
                         norm=norm, cmap='RdBu_r')
    ax3.set_title(f'{var} - Difference (Year 500 - Year 780)', fontsize=14, fontweight='bold')
    ax3.set_global()
    

    gl3 = ax3.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl3.top_labels = False
    gl3.right_labels = False

    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.7, pad=0.05)
    cbar3.set_label(f'Δ{var}', fontsize=12)
    

    plt.suptitle(f'Variable {var} Comparison Analysis', fontsize=18, fontweight='bold', y=0.96)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    

    save_path = save_path_template.format(variable=var)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {save_path}")


ds_500.close()
ds_780.close()

print("\nAll variable comparison plots are complete!")
try:
    num_files = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"📂 Image location: {output_dir}/")
    print(f"📊 Total images generated: {num_files}")
except NameError:
    pass