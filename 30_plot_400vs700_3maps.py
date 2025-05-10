############plot NEE, if plot GPP, just set variable = 'NEE'
import xarray as xr 
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# File paths (example)

ai_spinup_file     = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_spinup/e3sm_run/20250408_trendytest_ICB1850CNPRDCTCBC/run/20250408_trendytest_ICB1850CNPRDCTCBC.elm.h0.0421-01-01-00000.nc"
normal_spinup_file = "/autofs/nccsopen-svm1_home/daweigao/0_dataset/20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0.0781-01-01-00000.nc"

ds_ai = xr.open_dataset(ai_spinup_file)
ds_normal = xr.open_dataset(normal_spinup_file)

def ds_to_df(ds, variable):
    data = ds[variable]
    if 'time' in data.dims:
        print("存在时间维度")
        print("时间维度长度:", data.sizes['time'])
        data = data.isel(time=0)
    else:
        print("不存在时间维度")
    data = data.squeeze()
    
    # Assume the netCDF file contains 'lat' and 'lon' variables
    lat = ds['lat'].values
    lon = ds['lon'].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    values = data.values.flatten()
    df = pd.DataFrame({
        'Latitude': lat2d.flatten(),
        'Longitude': lon2d.flatten(),
        variable: values
    })
    return df

def plot_comparison_map(
    df_truth, df_pred, 
    variable, 
    model1_name="AI_spinup", 
    model2_name="normal_spinup", 
    unit="",  # New parameter for unit
    save_path_template="comparison_{variable}.png"
):
    """
    Plots three maps: one for the prediction values, one for the truth values,
    and one for their difference. Adds unit text on the far right of each colorbar.
    
    df_truth: DataFrame for the truth (or reference) model
    df_pred: DataFrame for the prediction model
    variable: variable name to compare
    model1_name: name for df_pred model
    model2_name: name for df_truth model
    unit: unit string to display on the colorbar (e.g., "gC m⁻² day⁻¹")
    save_path_template: template for saving the figure
    """

    # Calculate a unified color range
    min_value = min(df_truth[variable].min(), df_pred[variable].min())
    max_value = max(df_truth[variable].max(), df_pred[variable].max())

    # Ensure indices match
    df_truth = df_truth.reset_index(drop=True)
    df_truth_filtered = df_truth.iloc[:len(df_pred)]  
    difference = df_pred[variable] - df_truth_filtered[variable]

    # Calculate color range for difference (using symmetric range)
    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))

    print(f"📌 {variable} 统计信息:")
    print(f"  🔹 {model1_name} - Min: {df_pred[variable].min()}, Max: {df_pred[variable].max()}")
    print(f"  🔹 {model2_name} - Min: {df_truth[variable].min()}, Max: {df_truth[variable].max()}")
    print(f"  🔹 Difference - Min: {difference.min()}, Max: {difference.max()}")

    diff_df = pd.DataFrame({
        'Longitude': df_truth['Longitude'],
        'Latitude': df_truth['Latitude'],
        'Difference': difference
    })

    # Print first 10 rows of diff_df rounded to 6 decimal places
    print(diff_df.head(10).round(6))
    
    pd.set_option('display.float_format', '{:.6f}'.format)

    plt.figure(figsize=(12, 15))

    # 1) Subplot for model1 (df_pred)
    plt.subplot(3, 1, 1)
    scatter1 = plt.scatter(
        df_truth['Longitude'], 
        df_truth['Latitude'], 
        c=df_pred[variable], 
        cmap='viridis', 
        s=10, 
        vmin=min_value, 
        vmax=max_value
    )
    cbar1 = plt.colorbar(scatter1, label=unit)
    cbar1.ax.text(
        1.05, 1.2, f"Max: {max_value:.6f}", 
        transform=cbar1.ax.transAxes, 
        ha='center', 
        fontsize=12, 
        color='black'
    )
    cbar1.ax.text(
        1.05, -0.05, f"Min: {min_value:.6f}", 
        transform=cbar1.ax.transAxes, 
        ha='center', 
        fontsize=12, 
        color='black'
    )

    plt.title(f"{variable} - {model1_name}", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # 2) Subplot for model2 (df_truth)
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
    cbar2.ax.text(
        1.05, 1.2, f"Max: {max_value:.6f}", 
        transform=cbar2.ax.transAxes, 
        ha='center', 
        fontsize=12, 
        color='black'
    )
    cbar2.ax.text(
        1.05, -0.05, f"Min: {min_value:.6f}", 
        transform=cbar2.ax.transAxes, 
        ha='center', 
        fontsize=12, 
        color='black'
    )

    plt.title(f"{variable} - {model2_name}", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # 3) Subplot for the difference (model1 - model2)
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
    cbar3.ax.text(
        1.05, 1.2, f"Max: {difference.max():.6f}", 
        transform=cbar3.ax.transAxes, 
        ha='center', 
        fontsize=12, 
        color='black'
    )
    cbar3.ax.text(
        1.05, -0.05, f"Min: {difference.min():.6f}", 
        transform=cbar3.ax.transAxes, 
        ha='center', 
        fontsize=12, 
        color='black'
    )

    plt.title(f"{variable} - Difference ({model1_name} - {model2_name})", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Save the figure
    save_path = save_path_template.format(variable=variable)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# --------------------- Usage Example ---------------------
# Suppose the variable is "GPP"
variable = 'NEE'
# variable = 'GPP'


unit_string = r"gC m$^{-2}$ s$^{-1}$"
# unit_string = r""
df_normal = ds_to_df(ds_normal, variable)
df_ai = ds_to_df(ds_ai, variable)

# Here, we treat normal_spinup as the "truth" (reference) and AI_spinup as the prediction.
plot_comparison_map(
    df_truth=df_normal,
    df_pred=df_ai,
    variable=variable,
    model1_name="AI_spinup(100 years)",
    model2_name="normal_spinup",
    unit=unit_string,
    save_path_template="./5_plot_400vs700/{variable}_comparison_map.png"
)







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
    # Add more as needed
}

# Function to extract data for specific PFTs
def extract_pft_data(file_path, variable, max_pfts=5):

    with nc.Dataset(file_path, 'r') as ds:
        # Extract variables
        var_data = ds.variables[variable][:]  # TLAI data
        pft_index = ds.variables['pfts1d_itypveg'][:]  # PFT type index
        gridcell_index = ds.variables['pfts1d_gridcell_index'][:]  # Gridcell index
        lat = ds.variables['pfts1d_lat'][:]  # Latitude
        lon = ds.variables['pfts1d_lon'][:]  # Longitude
        print('lat',len(lat))
        print('lon',len(lon))
        
        # Create DataFrame for all data
        df = pd.DataFrame({
            'Longitude': lon,
            'Latitude': lat,
            variable: var_data,
            'PFT_index': pft_index
        })
        
        # Split by PFT
        pft_dfs = {}
        for pft in range(max_pfts):
            pft_df = df[df['PFT_index'] == pft].reset_index(drop=True)
            print(f"PFT {pft}: {len(pft_df)} points")
            if not pft_df.empty:
                pft_dfs[pft] = pft_df
        
        # Get unit
        unit = ds.variables[variable].units if hasattr(ds.variables[variable], 'units') else ""
    
    return pft_dfs, unit

# Modified plot_comparison_map function
def plot_comparison_map(
    df_truth, df_pred, 
    variable, 
    pft_index, 
    model1_name="AI_spinup", 
    model2_name="normal_spinup", 
    unit="", 
    save_path="comparison.png"
):

    # Calculate a unified color range
    min_value = min(df_truth[variable].min(), df_pred[variable].min())
    max_value = max(df_truth[variable].max(), df_pred[variable].max())

    # Ensure indices match
    df_truth = df_truth.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)
    
    # Ensure the same length
    min_len = min(len(df_pred), len(df_truth))
    df_pred = df_pred.iloc[:min_len]
    df_truth = df_truth.iloc[:min_len]
    
    # Calculate difference
    difference = df_pred[variable] - df_truth[variable]

    # Calculate color range for difference (symmetric)
    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))

    # Print statistics
    pft_name = pft_names.get(pft_index, f'PFT {pft_index}')
    print(f"📌 {variable}  (PFT {pft_index}: {pft_name}):")
    print(f"  🔹 {model1_name} - Min: {df_pred[variable].min()}, Max: {df_pred[variable].max()}")
    print(f"  🔹 {model2_name} - Min: {df_truth[variable].min()}, Max: {df_truth[variable].max()}")
    print(f"  🔹 Difference - Min: {difference.min()}, Max: {difference.max()}")
    print(f"  🔹 len: {len(df_pred)}")

    # Create DataFrame for difference
    diff_df = pd.DataFrame({
        'Longitude': df_truth['Longitude'],
        'Latitude': df_truth['Latitude'],
        'Difference': difference
    })
    print(diff_df.head(10).round(6))

    # Create figure
    plt.figure(figsize=(12, 15))

    # Subplot for model1 (df_pred)
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

    # Subplot for model2 (df_truth)
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

    # Subplot for the difference
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

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

# Main execution
def main():
    # Extract data
    ai_pft_dfs, unit_pred = extract_pft_data(ai_spinup_file, variable)
    truth_pft_dfs, unit_truth = extract_pft_data(normal_spinup_file, variable)
    
    # Use the unit from one of the files
    unit = unit_pred or unit_truth or "unitless"
    
    # Plot for each PFT
    for pft_index in range(5):  # PFT 0, 1, 2
    # for pft_index in [1, 2, 3]:
        if pft_index in ai_pft_dfs and pft_index in truth_pft_dfs:
            df_pred = ai_pft_dfs[pft_index]
            df_truth = truth_pft_dfs[pft_index]
            pft_name = pft_names.get(pft_index, f'PFT {pft_index}')
            save_path = f"./5_plot_400vs700/comparison_{variable}_pft{pft_index}.png"
            
            plot_comparison_map(
                df_truth=df_truth,
                df_pred=df_pred,
                variable=variable,
                pft_index=pft_index,
                unit=unit,
                save_path=save_path
            )
        else:
            print(f"PFT {pft_index} ingore the data")

if __name__ == '__main__':
    main()







######################plot soil3c_vr
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
    """
    Extract SOIL3C_vr data from NetCDF file, take the first column per gridcell, split by levgrnd layer.
    
    Args:
        file_path (str): Path to NetCDF file
        variable (str): Variable name (e.g., 'SOIL3C_vr')
        max_levels (int): Maximum soil layers to extract (0 to max_levels-1)
    
    Returns:
        level_dfs (dict): Dict of DataFrames per levgrnd layer, one column per gridcell
        unit (str): Variable unit
    """
    with nc.Dataset(file_path, 'r') as ds:
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
        
        n_columns = var_data.shape[0]
        col_index = np.arange(n_columns)  # 0 to n_columns-1
        
        print(f"File: {file_path}")
        print(f"  {variable} shape: {var_data.shape}")
        print(f"  cols1d_lat length: {len(lat)}")
        print(f"  cols1d_lon length: {len(lon)}")
        print(f"  cols1d_gridcell_index length: {len(gridcell_index)}")
        print(f"  Unique gridcell_index values: {len(np.unique(gridcell_index))}")
        
        n_levels = var_data.shape[1]
        if max_levels > n_levels:
            print(f"Warning: max_levels ({max_levels}) exceeds available levels ({n_levels}). Using {n_levels}.")
            max_levels = n_levels
        
        level_dfs = {}
        for level in range(max_levels):
            level_data = var_data[:, level]  # Data for level (column,)
            
            df = pd.DataFrame({
                'Longitude': lon,
                'Latitude': lat,
                'Gridcell_index': gridcell_index,
                variable: level_data,
                'Column_index': col_index
            })
            
            df = df.sort_values(by=['Gridcell_index', 'Column_index'])
            df_first = df.groupby('Gridcell_index').first().reset_index()
            df_first = df_first.dropna(subset=[variable]).reset_index(drop=True)
            print(f"  Level {level}: {len(df_first)} valid gridcell points (first column)")
            
            if not df_first.empty:
                level_dfs[level] = df_first
        
        unit = ds.variables[variable].units if hasattr(ds.variables[variable], 'units') else "unitless"
    
    return level_dfs, unit

# Function to plot comparison map based on your reference
def plot_comparison_map(
    df_truth, df_pred, 
    variable, 
    level_index, 
    model1_name="AI_spinup", 
    model2_name="normal_spinup", 
    unit="", 
    save_path="comparison.png"
):
    """
    为特定土壤层绘制对比图表（AI_spinup、normal_spinup 和差值）。
    
    参数：
    df_truth (pd.DataFrame): normal_spinup 数据
    df_pred (pd.DataFrame): AI_spinup 数据
    variable (str): 变量名
    level_index (int): 土壤层索引
    model1_name (str): AI_spinup 模型名称
    model2_name (str): normal_spinup 模型名称
    unit (str): 单位
    save_path (str): 输出文件路径
    """
    # Calculate a unified color range
    min_value = min(df_truth[variable].min(), df_pred[variable].min())
    max_value = max(df_truth[variable].max(), df_pred[variable].max())

    # Ensure indices match
    df_truth = df_truth.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)
    
    # Ensure the same length
    min_len = min(len(df_pred), len(df_truth))
    df_pred = df_pred.iloc[:min_len]
    df_truth = df_truth.iloc[:min_len]
    
    # Calculate difference
    difference = df_pred[variable] - df_truth[variable]

    # Calculate color range for difference (symmetric)
    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))

    # Print statistics
    print(f"📌 {variable} 统计信息 (Level {level_index}):")
    print(f"  🔹 {model1_name} - Min: {df_pred[variable].min():.6f}, Max: {df_pred[variable].max():.6f}")
    print(f"  🔹 {model2_name} - Min: {df_truth[variable].min():.6f}, Max: {df_truth[variable].max():.6f}")
    print(f"  🔹 Difference - Min: {difference.min():.6f}, Max: {difference.max():.6f}")
    print(f"  🔹 数据点数: {len(df_pred)}")

    # Create DataFrame for difference
    diff_df = pd.DataFrame({
        'Longitude': df_truth['Longitude'],
        'Latitude': df_truth['Latitude'],
        'Difference': difference
    })
    print(diff_df.head(10).round(6))

    # Create figure
    plt.figure(figsize=(12, 15))

    # Subplot for model1 (df_pred)
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
    plt.title(f"{variable} - {model1_name} (Level {level_index})", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Subplot for model2 (df_truth)
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
    plt.title(f"{variable} - {model2_name} (Level {level_index})", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Subplot for the difference
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
    plt.title(f"{variable} - Difference ({model1_name} - {model2_name}) (Level {level_index})", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

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
    
    # Plot maps for each level (0, 1, 2)
    for level_index in [0, 1, 2]:
        if level_index in ai_level_dfs and level_index in truth_level_dfs:
            df_pred = ai_level_dfs[level_index]
            df_truth = truth_level_dfs[level_index]
            save_path = f"./5_plot_400vs700/maps_{variable}_levgrnd{level_index}.png"
            
            plot_comparison_map(
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