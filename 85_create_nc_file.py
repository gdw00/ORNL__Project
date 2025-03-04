
##get data
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
import xarray as xr
from scipy.interpolate import griddata
from netCDF4 import Dataset
from collections import defaultdict
import random



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.jit.load("LSTM_model_62.pt")
model.eval()  
model.to(device)  


input_files = sorted(glob.glob('/mnt/DATA/dawei/3_oak_all/11_LSTM_add_restart/1_global_data/1_training_data_batch_*.pkl'))
df_list = []

for file in input_files:
    print(f"Loading {file}...")
    try:
        batch_df = pd.read_pickle(file)
        df_list.append(batch_df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

df = pd.concat(df_list, ignore_index=True)
print("All data loaded.")
print("Columns in df:", df.columns)


print("Before Interpolation:")
print("Mean:", np.nanmean(df["H2OSOI_10CM"]))
print("Std:", np.nanstd(df["H2OSOI_10CM"]))
points = np.column_stack((df["Longitude"], df["Latitude"]))
values = df["H2OSOI_10CM"].values.flatten()

valid_mask = ~np.isnan(values)
points_valid = points[valid_mask]
values_valid = values[valid_mask]

df["H2OSOI_10CM_filled"] = griddata(points_valid, values_valid, (df["Longitude"], df["Latitude"]), method="nearest")
print("Original Min H2OSOI_10CM_filled:", np.nanmin(df["H2OSOI_10CM_filled"]))
print("Original Max H2OSOI_10CM_filled:", np.nanmax(df["H2OSOI_10CM_filled"]))

# df["H2OSOI_10CM"] = df["H2OSOI_10CM"].fillna(df["H2OSOI_10CM_filled"])
df["H2OSOI_10CM"] = np.where(np.isnan(df["H2OSOI_10CM"]), df["H2OSOI_10CM_filled"], df["H2OSOI_10CM"])

df["H2OSOI_10CM_original"] = df["H2OSOI_10CM"].copy()
df["H2OSOI_10CM"] = np.where(np.isnan(df["H2OSOI_10CM"]), df["H2OSOI_10CM_filled"], df["H2OSOI_10CM"])
unchanged_values = (df["H2OSOI_10CM_original"].dropna() == df["H2OSOI_10CM"].dropna()).all()

df.drop(columns=["H2OSOI_10CM_filled"], inplace=True)
df.drop(columns=["H2OSOI_10CM_original"], inplace=True)

# print("After Interpolation:")
# df["H2OSOI_10CM_filled"] = griddata(points_valid, values_valid, (df["Longitude"], df["Latitude"]), method="nearest")
# df["H2OSOI_10CM"] = df["H2OSOI_10CM"].fillna(df["H2OSOI_10CM_filled"])
# print("Mean:", np.nanmean(df["H2OSOI_10CM"]))
# print("Std:", np.nanstd(df["H2OSOI_10CM"]))
#####end
print(f"Filtered dataset: {len(df)} samples remaining after removing NaN H2OSOI values.")

x_list_columns_2d = ['soil3c_vr', 'soil4c_vr', 'cwdc_vr']
y_list_columns_2d = ['Y_soil3c_vr', 'Y_soil4c_vr', 'Y_cwdc_vr']
x_list_columns_1d = ['deadcrootc', 'deadstemc', 'tlai']
y_list_columns_1d = ['Y_deadcrootc', 'Y_deadstemc', 'Y_tlai']
list_columns_2d = x_list_columns_2d + y_list_columns_2d
list_columns_1d = x_list_columns_1d + y_list_columns_1d
for col in y_list_columns_1d:
    if col in df.columns:
        flattened_values = np.concatenate(df[col].dropna().values).flatten()  # Flatten list values
        print(f"{col}: min={flattened_values.min():.12f}, max={flattened_values.max():.12f}")
    else:
        print(f"Warning: {col} not found in the dataframe.")


for col in x_list_columns_1d + y_list_columns_1d:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x[:17] if isinstance(x, list) else x)

print("Columns truncated to first 17 elements.")
max_length_1d = df[list_columns_1d].applymap(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).max().max()


for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)


max_rows_2d = df[list_columns_2d].applymap(lambda x: x.shape[0] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()
max_cols_2d = df[list_columns_2d].applymap(lambda x: x.shape[1] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()


for col in list_columns_1d:
    df[col] = df[col].apply(lambda x: np.pad(np.array(x), (0, max_length_1d - len(x)), mode='constant') if isinstance(x, (list, np.ndarray)) else np.zeros(max_length_1d))

for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.pad(x, ((0, max_rows_2d - x.shape[0]), (0, max_cols_2d - x.shape[1])), mode='constant') if isinstance(x, np.ndarray) and x.ndim == 2 else np.zeros((max_rows_2d, max_cols_2d)))






list_data_x_1d = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_1d}
list_data_y_1d = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_1d}

list_data_x_2d = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_2d}
list_data_y_2d = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_2d}


for col in x_list_columns_1d:
    print(f"{col} type: {type(list_data_x_1d[col])}, dtype: {list_data_x_1d[col].dtype}, shape: {list_data_x_1d[col].shape}")  
for col in y_list_columns_1d:
    print(f"{col} type: {type(list_data_y_1d[col])}, dtype: {list_data_y_1d[col].dtype}, shape: {list_data_y_1d[col].shape}")  

for col in x_list_columns_2d:
    print(f"{col} type: {type(list_data_x_2d[col])}, dtype: {list_data_x_2d[col].dtype}, shape: {list_data_x_2d[col].shape}")  
for col in y_list_columns_2d:
    print(f"{col} type: {type(list_data_y_2d[col])}, dtype: {list_data_y_2d[col].dtype}, shape: {list_data_y_2d[col].shape}")  


# List of columns for time-series data
time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
# static_columns = [col for col in df.columns if col not in time_series_columns and not col.startswith('Y_')]
static_columns = [
    col for col in df.columns 
    if col not in time_series_columns 
    and not col.startswith('Y_') 
    and col not in x_list_columns_2d 
    and col not in x_list_columns_1d
]
print("Static Columns (filtered):", static_columns)
# target_columns = [col for col in df.columns if col.startswith('Y_')]
target_columns = [
    col for col in df.columns 
    if col.startswith('Y_') 
    and col not in y_list_columns_2d 
    and col not in y_list_columns_1d
]

print("Target Columns (filtered):", target_columns)
for col in time_series_columns:
    print('processisng: ',col)
    df[col] = df[col].apply(
        lambda x: np.pad(x[:1476], (0, max(0, 1476 - len(x))), 'constant') if isinstance(x, list) else np.zeros(1476, dtype=np.float32)
    )


time_series_data = np.stack([np.column_stack(df[col]) for col in time_series_columns], axis=-1)
scaler_time_series = StandardScaler()
time_series_data = time_series_data.reshape(-1, len(time_series_columns))  # Flatten for normalization
time_series_data = scaler_time_series.fit_transform(time_series_data)
time_series_data = time_series_data.reshape(-1, 1476, len(time_series_columns))  # Reshape back

scaler_static = StandardScaler()
static_data = scaler_static.fit_transform(df[static_columns].values)

scaler_target = StandardScaler()
target_data = scaler_target.fit_transform(df[target_columns].values)
print('Normalize the static and target-end')
print('Convert data to PyTorch tensors')
# Convert data to PyTorch tensors
time_series_data = torch.tensor(time_series_data, dtype=torch.float32).to(device)
static_data = torch.tensor(static_data, dtype=torch.float32).to(device)
target_data = torch.tensor(target_data, dtype=torch.float32).to(device)


for col in list_columns_1d:
    min_val = df[col].apply(lambda x: np.min(x) if isinstance(x, (list, np.ndarray)) else 0).min()
    max_val = df[col].apply(lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) else 0).max()
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")

scalers_1d = {} 
for col in list_columns_1d:
    scaler = StandardScaler()
    

    col_data = np.vstack(df[col].values)  
    

    scaled_data = scaler.fit_transform(col_data)

    scalers_1d[col] = scaler

    df[col] = list(scaled_data)

scalers_2d = {}
for col in list_columns_2d:
    reshaped_data = np.vstack(df[col].values).reshape(-1, 1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(reshaped_data)
    scalers_2d[col] = scaler  

    reshaped_scaled_data = scaled_data.reshape(len(df), max_rows_2d, max_cols_2d)

    df[col] = list(reshaped_scaled_data)

for col in list_columns_2d:
    min_val = df[col].apply(lambda x: np.min(x)).min()
    max_val = df[col].apply(lambda x: np.max(x)).max()
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")
for col in list_columns_1d:
    min_val = df[col].apply(lambda x: np.min(x)).min()
    max_val = df[col].apply(lambda x: np.max(x)).max()
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")
x_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_2d}
x_list_columns_1d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_1d}
y_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_2d}
y_list_columns_1d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_1d}



########################get prediction
batch_size = 16
predictions_scalar = []
predictions_vector = []
predictions_matrix = []
with torch.no_grad():
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        time_series_batch = time_series_data[i:end_idx].to(device)
        static_batch = static_data[i:end_idx].to(device)
        columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in x_list_columns_1d_tensors.items()}
        columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in x_list_columns_2d_tensors.items()}
        y_columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in y_list_columns_1d_tensors.items()}
        y_columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in y_list_columns_2d_tensors.items()}
        

        columns_1d_batch_tensor = torch.cat([tensor for tensor in columns_1d_batch.values()], dim=1).to(device)
        columns_2d_batch_tensor = torch.cat([tensor.unsqueeze(1) for tensor in columns_2d_batch.values()], dim=1).to(device)


        scalar_pred, vector_pred, matrix_pred = model(time_series_batch, static_batch, columns_1d_batch_tensor, columns_2d_batch_tensor)


        predictions_scalar.append(scalar_pred.cpu().numpy())
        predictions_vector.append(vector_pred.cpu().numpy())
        predictions_matrix.append(matrix_pred.cpu().numpy())
predictions_scalar = np.concatenate(predictions_scalar, axis=0)  # (N, 5)
predictions_vector = np.concatenate(predictions_vector, axis=0)  # (N, 3, 34)
predictions_matrix = np.concatenate(predictions_matrix, axis=0)  # (N, 3, 18, 15)

predictions_scalar_np = scaler_target.inverse_transform(predictions_scalar)
predictions_df = pd.DataFrame(predictions_scalar_np, columns=target_columns)
# predictions_original_scale = scaler_target.inverse_transform(predictions.values)

predictions_1d_dict = {}  

for idx, col in enumerate(x_list_columns_1d):
    y_col = 'Y_' + col 

    if col in scalers_1d:  

        predictions_1d_dict[y_col] = scalers_1d[y_col].inverse_transform(predictions_vector[:, idx, :])



predictions_1d_df = pd.DataFrame({col: predictions_1d_dict[col].tolist() for col in predictions_1d_dict})


predictions_2d_dict = {} 


for idx, col in enumerate(x_list_columns_2d):  
    y_col = 'Y_' + col  

    if col in scalers_2d:  
        scaler = scalers_2d[y_col]  

        pred_reshaped = predictions_matrix[:, idx, :, :].reshape(-1, 1)
        predictions_2d_dict[y_col] = scaler.inverse_transform(pred_reshaped).reshape(-1, max_rows_2d, max_cols_2d)


predictions_2d_df = pd.DataFrame({col: predictions_2d_dict[col].tolist() for col in predictions_2d_dict})
print("!!!!!!!!!!data done!")





##########start to write the code



new_values = {}

input_nc = "./4_nc_file/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0401-01-01-00000.nc"
output_nc = "./4_nc_file/updated_20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0401-01-01-00000.nc"


# variables_scalar = ["GPP", "HR", "AR", "NPP", "Col_fire_Closs"] 
variables_1d = ["deadcrootc", "deadstemc", "tlai"]                
variables_2d = ["soil3c_vr", "soil4c_vr", "cwdc_vr"]              


with Dataset(input_nc, "r") as src:

    original_shapes = {var: src.variables[var].shape for var in ( variables_1d + variables_2d)}

print(original_shapes)





new_values = {}  


with Dataset(input_nc, "r") as src:

    gridcell_lat = src.variables['grid1d_lat'][:]
    gridcell_lon = src.variables['grid1d_lon'][:]


    pft_lat = src.variables['pfts1d_lat'][:]
    pft_lon = src.variables['pfts1d_lon'][:]


    pft_gridcell_index = src.variables['pfts1d_gridcell_index'][:]  # (pft,)


    original_shapes = {var: src.variables[var].shape for var in variables_1d}




gridcell_pft_map = defaultdict(list)  
for pft_idx, gridcell_idx in enumerate(pft_gridcell_index):
    gridcell_pft_map[gridcell_idx - 1].append(pft_idx)

for i, (grid_idx, pft_indices) in enumerate(list(gridcell_pft_map.items())[:5]):
    print(f"  🔹 Gridcell {grid_idx}: {len(pft_indices)}  PFT")


for var in variables_1d:

    original_shape = original_shapes[var]  # (pft,)


    new_values[var] = np.zeros(original_shape, dtype=np.float32)  


    predicted_values = np.array(predictions_1d_df[f"Y_{var}"].tolist(), dtype=np.float32)


    for grid_idx, pft_indices in gridcell_pft_map.items():

        pft_count = len(pft_indices) 
        max_replace = min(17, pft_count)  

        new_values[var][pft_indices[:max_replace]] = predicted_values[grid_idx, :max_replace]

        if pft_count > max_replace:
            new_values[var][pft_indices[max_replace:]] = 0





new_values_2d = {}


with Dataset(input_nc, "r") as src:

    column_gridcell_index = src.variables['cols1d_gridcell_index'][:]  # (column,)


    original_shapes_2d = {var: src.variables[var].shape for var in variables_2d}



gridcell_column_map = defaultdict(list)
for column_idx, gridcell_idx in enumerate(column_gridcell_index):
    gridcell_column_map[gridcell_idx - 1].append(column_idx)

for i, (grid_idx, column_indices) in enumerate(list(gridcell_column_map.items())[:5]):
    print(f"  🔹 Gridcell {grid_idx}: {len(column_indices)} 个 columns")


for var in variables_2d:

    original_shape = original_shapes_2d[var]  # (column, levgrnd)


    new_values_2d[var] = np.zeros(original_shape, dtype=np.float32)


    predicted_values = np.array(predictions_2d_df[f"Y_{var}"].tolist(), dtype=np.float32)

  


    for grid_idx, column_indices in gridcell_column_map.items():


        if len(column_indices) > 0:
            first_column_idx = column_indices[0] 
            new_values_2d[var][first_column_idx, :] = predicted_values[grid_idx, 0, :]  




output_nc = "./4_nc_file/updated_20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0401-01-01-00000.nc"

with Dataset(input_nc, "r") as src:
    with Dataset(output_nc, "w", format="NETCDF4") as dst:


        for name, dimension in src.dimensions.items():
            dst.createDimension(name, (None if dimension.isunlimited() else len(dimension)))

        for name, variable in src.variables.items():
            var_dtype = variable.dtype
            var_dims = variable.dimensions
            dst_var = dst.createVariable(name, var_dtype, var_dims)


            dst_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})

            if name in new_values:
                dst_var[:] = new_values[name]
            elif name in new_values_2d:
                dst_var[:, :] = new_values_2d[name]
            else:
                dst_var[:] = variable[:]  

print("\n done:", output_nc)
