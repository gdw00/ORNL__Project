############plot NEE, if plot GPP, just set variable = 'NEE'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np

def ds_to_df(ds, variable):

    data = ds[variable]
    if 'time' in data.dims:
        data = data.isel(time=0)
    data = data.squeeze()
    lat = ds['lat'].values
    lon = ds['lon'].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    values = data.values.flatten()
    return pd.DataFrame({
        'Latitude': lat2d.flatten(),
        'Longitude': lon2d.flatten(),
        variable: values
    })

# 1. Specify file paths

ai_spinup_file     = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_spinup/e3sm_run/20250408_trendytest_ICB1850CNPRDCTCBC/run/20250408_trendytest_ICB1850CNPRDCTCBC.elm.h0.0421-01-01-00000.nc"
normal_spinup_file = "/autofs/nccsopen-svm1_home/daweigao/0_dataset/20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0.0781-01-01-00000.nc"

# 2. Open datasets
ds_ai     = xr.open_dataset(ai_spinup_file)
ds_normal = xr.open_dataset(normal_spinup_file)

# 3. Convert to DataFrames
variable   = 'NEE'
# variable   = 'GPP'
df_normal  = ds_to_df(ds_normal, variable)
df_ai      = ds_to_df(ds_ai,     variable)

# 4. Merge on grid cells
df_merged = pd.merge(
    df_normal, df_ai,
    on=['Latitude', 'Longitude'],
    suffixes=('_normal', '_ai')
)

# 5. Prepare scatter data
x = df_merged[f"{variable}_normal"].values
y = df_merged[f"{variable}_ai"].values
mask = np.isfinite(x) & np.isfinite(y)
x_clean = x[mask]
y_clean = y[mask]
n_points = len(x_clean)

# 6. Plot & save to file (no plt.show())
plt.figure(figsize=(8, 6))
plt.scatter(x_clean, y_clean, alpha=0.5, label="Data points")
plt.xlabel(f"{variable} from Normal Spinup")
plt.ylabel(f"{variable} from AI Spinup")
plt.title(f"Scatter Plot of {variable}: AI Spinup vs Normal Spinup")
plt.legend()
plt.grid(True)
plt.text(
    0.05, 0.95, f"Total points: {n_points}",
    transform=plt.gca().transAxes,
    verticalalignment='top', fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8)
)
plt.tight_layout()
plt.savefig(f"./5_plot_400vs700/{variable}_scatter.png", dpi=300)
plt.close()

# 7. Cleanup
ds_ai.close()
ds_normal.close()
print('done')




#################plot tlai
import netCDF4 as nc
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# File paths
ai_spinup_file = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_spinup/e3sm_run/20250408_trendytest_ICB1850CNPRDCTCBC/run/20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0421-01-01-00000.nc"
normal_spinup_file = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc"

# Variable to plot
variable = 'tlai'

# PFT names (based on NetCDF global attributes)
pft_names = {
    0: 'Not Vegetated',
    1: 'Needleleaf Evergreen Temperate Tree',
    2: 'Needleleaf Evergreen Boreal Tree',
    3: 'Needleleaf Deciduous Boreal Tree',
}

# Function to extract data for specific PFTs
def extract_pft_data(file_path, variable, max_pfts=4):

    with nc.Dataset(file_path, 'r') as ds:
        var_data = ds.variables[variable][:] 
        pft_index = ds.variables['pfts1d_itypveg'][:] 
        gridcell_index = ds.variables['pfts1d_gridcell_index'][:] 
        lat = ds.variables['pfts1d_lat'][:] 
        lon = ds.variables['pfts1d_lon'][:] 
        print('lat', len(lat))
        print('lon', len(lon))
        
        df = pd.DataFrame({
            'Longitude': lon,
            'Latitude': lat,
            variable: var_data,
            'PFT_index': pft_index,
            'Gridcell_index': gridcell_index
        })
        
        pft_dfs = {}
        for pft in range(max_pfts):
            pft_df = df[df['PFT_index'] == pft].reset_index(drop=True)
            print(f"PFT {pft}: {len(pft_df)} points")
            if not pft_df.empty:
                pft_dfs[pft] = pft_df
        
        unit = ds.variables[variable].units if hasattr(ds.variables[variable], 'units') else ""
    
    return pft_dfs, unit

# Modified plot_comparison_map function (unchanged)
def plot_comparison_map(
    df_truth, df_pred, 
    variable, 
    pft_index, 
    model1_name="AI_spinup", 
    model2_name="normal_spinup", 
    unit="", 
    save_path="comparison.png"
):
    min_value = min(df_truth[variable].min(), df_pred[variable].min())
    max_value = max(df_truth[variable].max(), df_pred[variable].max())
    df_truth = df_truth.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)
    min_len = min(len(df_pred), len(df_truth))
    df_pred = df_pred.iloc[:min_len]
    df_truth = df_truth.iloc[:min_len]
    difference = df_pred[variable] - df_truth[variable]
    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))
    
    pft_name = pft_names.get(pft_index, f'PFT {pft_index}')
    print(f" {variable}  (PFT {pft_index}: {pft_name}):")
    print(f"   {model1_name} - Min: {df_pred[variable].min()}, Max: {df_pred[variable].max()}")
    print(f"   {model2_name} - Min: {df_truth[variable].min()}, Max: {df_truth[variable].max()}")
    print(f"   Difference - Min: {difference.min()}, Max: {difference.max()}")
    print(f"   len: {len(df_pred)}")
    
    diff_df = pd.DataFrame({
        'Longitude': df_truth['Longitude'],
        'Latitude': df_truth['Latitude'],
        'Difference': difference
    })
    print(diff_df.head(10).round(6))
    
    plt.figure(figsize=(12, 15))
    plt.subplot(3, 1, 1)
    scatter1 = plt.scatter(
        df_pred['Longitude'], 
        df_pred['Latitude'], 
        c=df_pred[variable], 
        cmap='viridis', 
        s=10, 
        vmin=min_value, 
        vmax=max_value
    )
    cbar1 = plt.colorbar(scatter1, label=unit)
    cbar1.ax.text(1.05, 1.2, f"Max: {max_value:.6f}", transform=cbar1.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar1.ax.text(1.05, -0.05, f"Min: {min_value:.6f}", transform=cbar1.ax.transAxes, ha='center', fontsize=12, color='black')
    plt.title(f"{variable} - {model1_name} (PFT {pft_index}: {pft_name})", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    scatter2 = plt.scatter(
        df_truth['Longitude'], 
        df_truth['Latitude'], 
        c=df_truth[variable], 
        cmap='viridis', 
        s=10, 
        vmin=min_value, 
        vmax=max_value
    )
    cbar2 = plt.colorbar(scatter2, label=unit)
    cbar2.ax.text(1.05, 1.2, f"Max: {max_value:.6f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar2.ax.text(1.05, -0.05, f"Min: {min_value:.6f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')
    plt.title(f"{variable} - {model2_name} (PFT {pft_index}: {pft_name})", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    scatter3 = plt.scatter(
        df_truth['Longitude'], 
        df_truth['Latitude'], 
        c=difference, 
        cmap='RdBu', 
        s=10, 
        vmin=diff_min, 
        vmax=diff_max
    )
    cbar3 = plt.colorbar(scatter3, label=unit)
    cbar3.ax.text(1.05, 1.2, f"Max: {difference.max():.6f}", transform=cbar3.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar3.ax.text(1.05, -0.05, f"Min: {difference.min():.6f}", transform=cbar3.ax.transAxes, ha='center', fontsize=12, color='black')
    plt.title(f"{variable} - Difference ({model1_name} - {model2_name}) (PFT {pft_index}: {pft_name})", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Map plot saved to {save_path}")

# New function to plot scatter plot
def plot_scatter(
    df_truth, df_pred, 
    variable, 
    pft_index, 
    model1_name="AI_spinup", 
    model2_name="normal_spinup", 
    unit="", 
    save_path="scatter.png"
):
    """
    为特定 PFT 绘制散点图，横轴为 normal_spinup，纵轴为 AI_spinup。
    
    参数：
    df_truth (pd.DataFrame): normal_spinup 数据
    df_pred (pd.DataFrame): AI_spinup 数据
    variable (str): 变量名
    pft_index (int): PFT 索引
    model1_name (str): AI_spinup 模型名称
    model2_name (str): normal_spinup 模型名称
    unit (str): 单位
    save_path (str): 输出文件路径
    """
    # Merge DataFrames on Longitude, Latitude, and Gridcell_index
    df_merged = pd.merge(
        df_truth, df_pred,
        on=['Longitude', 'Latitude', 'Gridcell_index'],
        suffixes=('_normal', '_ai')
    )
    
    # Prepare scatter data
    x = df_merged[f"{variable}_normal"].values
    y = df_merged[f"{variable}_ai"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n_points = len(x_clean)
    
    # Print scatter statistics
    pft_name = pft_names.get(pft_index, f'PFT {pft_index}')
    print(f" Scatter Plot  (PFT {pft_index}: {pft_name}):")
    print(f"   Valid points after merge: {n_points}")
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_clean, y_clean, alpha=0.5, s=10, label="Data points")
    plt.xlabel(f"{variable} from {model2_name} ({unit})")
    plt.ylabel(f"{variable} from {model1_name} ({unit})")
    plt.title(f"Scatter Plot of {variable}: {model1_name} vs {model2_name}\n(PFT {pft_index}: {pft_name})")
    plt.legend()
    plt.grid(True)
    plt.text(
        0.05, 0.95, f"Total points: {n_points}",
        transform=plt.gca().transAxes,
        verticalalignment='top', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Add 1:1 line for reference
    min_val = min(x_clean.min(), y_clean.min())
    max_val = max(x_clean.max(), y_clean.max())
    # plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {save_path}")

# Main execution
def main():
    # Extract data
    ai_pft_dfs, unit_pred = extract_pft_data(ai_spinup_file, variable)
    truth_pft_dfs, unit_truth = extract_pft_data(normal_spinup_file, variable)
    
    # Use the unit from one of the files
    unit = unit_pred or unit_truth or "unitless"
    
    # Plot for each PFT
    for pft_index in [1, 2, 3]:
        if pft_index in ai_pft_dfs and pft_index in truth_pft_dfs:
            df_pred = ai_pft_dfs[pft_index]
            df_truth = truth_pft_dfs[pft_index]
            pft_name = pft_names.get(pft_index, f'PFT {pft_index}')
            
            # Plot comparison map
            map_save_path = f"./5_plot_400vs700/comparison_{variable}_pft{pft_index}.png"
            plot_comparison_map(
                df_truth=df_truth,
                df_pred=df_pred,
                variable=variable,
                pft_index=pft_index,
                unit=unit,
                save_path=map_save_path
            )
            
            # Plot scatter
            scatter_save_path = f"./5_plot_400vs700/scatter_{variable}_pft{pft_index}.png"
            plot_scatter(
                df_truth=df_truth,
                df_pred=df_pred,
                variable=variable,
                pft_index=pft_index,
                unit=unit,
                save_path=scatter_save_path
            )
        else:
            print(f"PFT {pft_index} ingore plot")

if __name__ == '__main__':
    main()




#################plot soil3c
import netCDF4 as nc
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# File paths
ai_spinup_file = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_spinup/e3sm_run/20250408_trendytest_ICB1850CNPRDCTCBC/run/20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0421-01-01-00000.nc"
normal_spinup_file = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc"

# Variable to plot
variable = 'soil3c_vr'

# Function to extract SOIL3C_vr data for specific levgrnd levels, taking first column per gridcell
def extract_soil_data(file_path, variable, max_levels=3):

    with nc.Dataset(file_path, 'r') as ds:
        # Extract variables
        try:
            var_data = ds.variables[variable][:]  # SOIL3C_vr (column, levgrnd)
        except KeyError:
            raise KeyError(f"Variable '{variable}' not found in {file_path}")
        
        try:
            lat = ds.variables['cols1d_lat'][:]  # Latitude
            lon = ds.variables['cols1d_lon'][:]  # Longitude
            gridcell_index = ds.variables['cols1d_gridcell_index'][:]  # Gridcell index
        except KeyError as e:
            raise KeyError(f"Required variable {str(e)} not found in {file_path}")
        
        # Create synthetic column index
        n_columns = var_data.shape[0]
        col_index = np.arange(n_columns)  # 0 to n_columns-1
        
        # Debugging output
        print(f"File: {file_path}")
        print(f"  {variable} shape: {var_data.shape}")
        print(f"  cols1d_lat length: {len(lat)}")
        print(f"  cols1d_lon length: {len(lon)}")
        print(f"  cols1d_gridcell_index length: {len(gridcell_index)}")
        print(f"  Unique gridcell_index values: {len(np.unique(gridcell_index))}")
        
        # Ensure max_levels does not exceed available levgrnd
        n_levels = var_data.shape[1]
        if max_levels > n_levels:
            print(f"Warning: max_levels ({max_levels}) exceeds available levels ({n_levels}). Using {n_levels}.")
            max_levels = n_levels
        
        # Create DataFrames for each level, taking the first column per gridcell
        level_dfs = {}
        for level in range(max_levels):
            # Extract data for specific level
            level_data = var_data[:, level]  # Data for level (column,)
            
            # Create DataFrame at column level
            df = pd.DataFrame({
                'Longitude': lon,
                'Latitude': lat,
                'Gridcell_index': gridcell_index,
                variable: level_data,
                'Column_index': col_index
            })
            
            # Sort by Column_index to ensure we take the first column (smallest index)
            df = df.sort_values(by=['Gridcell_index', 'Column_index'])
            
            # Take the first column for each Gridcell_index
            df_first = df.groupby('Gridcell_index').first().reset_index()
            
            # Remove rows with NaN values in variable
            df_first = df_first.dropna(subset=[variable]).reset_index(drop=True)
            print(f"  Level {level}: {len(df_first)} valid gridcell points (first column)")
            
            if not df_first.empty:
                level_dfs[level] = df_first
        
        # Get unit
        unit = ds.variables[variable].units if hasattr(ds.variables[variable], 'units') else "unitless"
    
    return level_dfs, unit

# Function to plot scatter for specific levgrnd
def plot_soil_scatter(
    df_truth, df_pred, 
    variable, 
    level_index, 
    model1_name="AI_spinup", 
    model2_name="normal_spinup", 
    unit="", 
    save_path="scatter.png"
):
    """
    Plot scatter plot for a specific soil layer, x-axis: normal_spinup, y-axis: AI_spinup.
    
    Args:
        df_truth (pd.DataFrame): normal_spinup data (first column per gridcell)
        df_pred (pd.DataFrame): AI_spinup data (first column per gridcell)
        variable (str): Variable name
        level_index (int): Soil layer index
        model1_name (str): Name of AI_spinup model
        model2_name (str): Name of normal_spinup model
        unit (str): Unit of variable
        save_path (str): Path to save plot
    """
    # Merge DataFrames on Gridcell_index
    df_merged = pd.merge(
        df_truth, df_pred,
        on=['Gridcell_index'],
        suffixes=('_normal', '_ai')
    )
    
    # Verify lat/lon consistency with tolerance
    df_merged = df_merged[
        (np.abs(df_merged['Latitude_normal'] - df_merged['Latitude_ai']) < 1e-5) &
        (np.abs(df_merged['Longitude_normal'] - df_merged['Longitude_ai']) < 1e-5)
    ]
    
    # Prepare scatter data
    x = df_merged[f"{variable}_normal"].values
    y = df_merged[f"{variable}_ai"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n_points = len(x_clean)
    
    # Print scatter statistics
    print(f" Scatter Plot Statistics (Level {level_index}):")
    print(f"   Valid points after merge: {n_points}")
    if n_points > 0:
        print(f"   {model2_name} - Min: {x_clean.min():.4f}, Max: {x_clean.max():.4f}")
        print(f"   {model1_name} - Min: {y_clean.min():.4f}, Max: {y_clean.max():.4f}")
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_clean, y_clean, alpha=0.5, s=10, label="Data points")
    plt.xlabel(f"{variable} from {model2_name} ({unit})")
    plt.ylabel(f"{variable} from {model1_name} ({unit})")
    plt.title(f"Scatter Plot of {variable}: {model1_name} vs {model2_name}\n(Level {level_index}, First Column per Gridcell)")
    plt.grid(True)
    plt.text(
        0.05, 0.95, f"Total points: {n_points}",
        transform=plt.gca().transAxes,
        verticalalignment='top', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Add 1:1 line
    if n_points > 0:
        min_val = min(x_clean.min(), y_clean.min())
        max_val = max(x_clean.max(), y_clean.max())
        # plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {save_path}")

# Main execution
def main():
    # Extract data
    try:
        ai_level_dfs, unit_pred = extract_soil_data(ai_spinup_file, variable)
        truth_level_dfs, unit_truth = extract_soil_data(normal_spinup_file, variable)
    except KeyError as e:
        print(f"Error extracting data: {e}")
        return
    
    # Use the unit from one of the files
    unit = unit_pred or unit_truth or "unitless"
    
    # Plot for each level (0, 1, 2)
    for level_index in [0, 1, 2]:
        if level_index in ai_level_dfs and level_index in truth_level_dfs:
            df_pred = ai_level_dfs[level_index]
            df_truth = truth_level_dfs[level_index]
            save_path = f"./5_plot_400vs700/scatter_{variable}_levgrnd{level_index}_firstcol.png"
            
            plot_soil_scatter(
                df_truth=df_truth,
                df_pred=df_pred,
                variable=variable,
                level_index=level_index,
                unit=unit,
                save_path=save_path
            )
        else:
            print(f"Level {level_index} data missing, skipping plot.")

if __name__ == '__main__':
    main()