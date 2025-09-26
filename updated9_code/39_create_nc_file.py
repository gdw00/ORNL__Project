import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import glob
from config import Config
from sklearn.preprocessing import StandardScaler
import xarray as xr
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from netCDF4 import Dataset
from collections import defaultdict
from scipy.spatial import KDTree

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torch.jit.load("./LSTM_model_62.pt")
model.eval()
model.to(device)

# Load and concatenate dataset from pickle files
input_files = sorted(glob.glob(Config.TRAIN_INPUT_GLOB))
df_list = []
for file in input_files:
    try:
        batch_df = pd.read_pickle(file)
        df_list.append(batch_df)
    except Exception as e:
        print(f"Error loading {file}: {e}")
df = pd.concat(df_list, ignore_index=True)

# Drop unnecessary columns
cols_to_drop = Config.COLS_TO_DROP
df = df.drop(columns=cols_to_drop, errors='ignore')

# Interpolate NaN values in specified _vr variables using spatial nearest neighbors
vars_to_interpolate = ['sminn_vr', 'smin_nh4_vr', 'Y_sminn_vr', 'Y_smin_nh4_vr']
for col in vars_to_interpolate:
    if col in df.columns:
        bad_mask = df[col].apply(lambda x: np.isnan(np.sum(x)))
        good_mask = ~bad_mask
        if not bad_mask.any():
            continue
        points_good = df.loc[good_mask, ['Longitude', 'Latitude']].values
        points_bad = df.loc[bad_mask, ['Longitude', 'Latitude']].values
        kdtree_good_points = KDTree(points_good)
        distances, nearest_indices_in_good_set = kdtree_good_points.query(points_bad, k=1)
        good_df_indices = df.index[good_mask]
        bad_df_indices = df.index[bad_mask]
        for i, bad_idx in enumerate(bad_df_indices):
            nearest_pos = nearest_indices_in_good_set[i]
            nearest_good_df_idx = good_df_indices[nearest_pos]
            df.at[bad_idx, col] = df.at[nearest_good_df_idx, col]

# Interpolate NaN values in H2OSOI_10CM
points = np.column_stack((df["Longitude"], df["Latitude"]))
values = df["H2OSOI_10CM"].values.flatten()
valid_mask = ~np.isnan(values)
points_valid = points[valid_mask]
values_valid = values[valid_mask]
h2osoi_filled = griddata(points_valid, values_valid, (df["Longitude"], df["Latitude"]), method="nearest")
df["H2OSOI_10CM"] = np.where(np.isnan(df["H2OSOI_10CM"]), h2osoi_filled, df["H2OSOI_10CM"])

# Define feature lists from config
x_list_columns_2d = Config.X_LIST_COLUMNS_2D
y_list_columns_2d = Config.Y_LIST_COLUMNS_2D
x_list_columns_1d = Config.X_LIST_COLUMNS_1D
y_list_columns_1d = Config.Y_LIST_COLUMNS_1D

list_columns_2d = x_list_columns_2d + y_list_columns_2d
list_columns_1d = x_list_columns_1d + y_list_columns_1d

# Truncate 1D variables
for col in x_list_columns_1d + y_list_columns_1d:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x[1:17] if isinstance(x, (list, np.ndarray)) else x)

# Process 2D variables
cols_2d = x_list_columns_2d + y_list_columns_2d
vars_to_reshape = Config.VARS_TO_RESHAPE
for col in cols_2d:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, (list, tuple)) else x)
        if col in vars_to_reshape:
            df[col] = df[col].apply(lambda x: x.reshape(-1, 1) if isinstance(x, np.ndarray) and x.ndim == 1 else x)
        df[col] = df[col].apply(lambda x: x[0:1, :10] if isinstance(x, np.ndarray) and x.ndim == 2 else x)

# Calculate max dimensions for padding
max_length_1d = df[list_columns_1d].applymap(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).max().max()
max_rows_2d = df[list_columns_2d].applymap(lambda x: x.shape[0] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()
max_cols_2d = df[list_columns_2d].applymap(lambda x: x.shape[1] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()

# Pad all list-like variables to uniform length/shape
for col in list_columns_1d:
    df[col] = df[col].apply(lambda x: np.pad(np.array(x), (0, max_length_1d - len(x)), mode='constant') if isinstance(x, (list, np.ndarray)) else np.zeros(max_length_1d))
for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.pad(x, ((0, max_rows_2d - x.shape[0]), (0, max_cols_2d - x.shape[1])), mode='constant') if isinstance(x, np.ndarray) and x.ndim == 2 else np.zeros((max_rows_2d, max_cols_2d)))

# Define feature groups for model input
time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
static_columns = [
    col for col in df.columns
    if col not in time_series_columns
    and not col.startswith('Y_')
    and col not in x_list_columns_2d
    and col not in x_list_columns_1d
    and not col.startswith('pft_')
]
target_columns = [
    col for col in df.columns
    if col.startswith('Y_')
    and col not in y_list_columns_2d
    and col not in y_list_columns_1d
]

# Process and normalize time-series data
for col in time_series_columns:
    df[col] = df[col].apply(
        lambda x: (
            np.pad(x[:1476], (0, max(0, 1476 - len(x))), 'constant')[-240:]
            if isinstance(x, list)
            else np.zeros(240, dtype=np.float32)
        )
    )
time_series_data = np.stack([np.column_stack(df[col]) for col in time_series_columns], axis=-1)
scaler_time_series = MinMaxScaler(feature_range=(0, 1))
time_series_data = time_series_data.reshape(-1, len(time_series_columns))
time_series_data = scaler_time_series.fit_transform(time_series_data)
time_series_data = time_series_data.reshape(-1, 240, len(time_series_columns))

# Normalize static and target data
scaler_static = MinMaxScaler(feature_range=(0, 1))
static_data = scaler_static.fit_transform(df[static_columns].values)
scaler_target = MinMaxScaler(feature_range=(0, 1))
target_data = scaler_target.fit_transform(df[target_columns].values)

# Convert to PyTorch tensors
time_series_data = torch.tensor(time_series_data, dtype=torch.float32)
static_data = torch.tensor(static_data, dtype=torch.float32)
target_data = torch.tensor(target_data, dtype=torch.float32)

# Process and normalize PFT data
pft_columns = [col for col in df.columns if col.startswith('pft_')]
pft_data = np.stack(df[pft_columns].apply(lambda row: np.concatenate(row), axis=1).values)
scaler_pft = MinMaxScaler()
pft_data = scaler_pft.fit_transform(pft_data)
pft_data = torch.tensor(pft_data, dtype=torch.float32)

# Normalize 1D and 2D list-like features
scalers_1d = {}
for col in list_columns_1d:
    scaler = MinMaxScaler(feature_range=(0, 1))
    col_data = np.vstack(df[col].values)
    scaled_data = scaler.fit_transform(col_data)
    scalers_1d[col] = scaler
    df[col] = list(scaled_data)

scalers_2d = {}
for col in list_columns_2d:
    reshaped_data = np.vstack(df[col].values).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(reshaped_data)
    scalers_2d[col] = scaler
    reshaped_scaled_data = scaled_data.reshape(len(df), max_rows_2d, max_cols_2d)
    df[col] = list(reshaped_scaled_data)

# Create final feature tensors
x_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_2d}
x_list_columns_1d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_1d}
y_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_2d}
y_list_columns_1d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_1d}

# Generate predictions
batch_size = 16
predictions_scalar, predictions_vector, predictions_matrix = [], [], []
with torch.no_grad():
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        time_series_batch = time_series_data[i:end_idx].to(device)
        static_batch = static_data[i:end_idx].to(device)
        pft_data_batch = pft_data[i:end_idx].to(device)
        
        columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in x_list_columns_1d_tensors.items()}
        columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in x_list_columns_2d_tensors.items()}
        
        columns_1d_batch_tensor = torch.cat(list(columns_1d_batch.values()), dim=1).to(device)
        columns_2d_batch_tensor = torch.cat([t.unsqueeze(1) for t in columns_2d_batch.values()], dim=1).to(device)

        scalar_pred, vector_pred, matrix_pred = model(time_series_batch, static_batch, columns_1d_batch_tensor, columns_2d_batch_tensor, pft_data_batch)
        vector_pred = torch.maximum(vector_pred, torch.tensor(0.0, device=vector_pred.device))
        matrix_pred = torch.maximum(matrix_pred, torch.tensor(0.0, device=matrix_pred.device))
        
        predictions_scalar.append(scalar_pred.cpu().numpy())
        predictions_vector.append(vector_pred.cpu().numpy())
        predictions_matrix.append(matrix_pred.cpu().numpy())

predictions_scalar = np.concatenate(predictions_scalar, axis=0)
predictions_vector = np.concatenate(predictions_vector, axis=0)
predictions_matrix = np.concatenate(predictions_matrix, axis=0)

# Inverse transform predictions to original scale
predictions_scalar_np = scaler_target.inverse_transform(predictions_scalar)
predictions_df = pd.DataFrame(predictions_scalar_np, columns=target_columns)

predictions_1d_dict = {}
for idx, col in enumerate(y_list_columns_1d):
    if col in scalers_1d:
        predictions_1d_dict[col] = scalers_1d[col].inverse_transform(predictions_vector[:, idx, :])
predictions_1d_df = pd.DataFrame({col: data.tolist() for col, data in predictions_1d_dict.items()})

predictions_2d_dict = {}
for idx, col in enumerate(y_list_columns_2d):
    if col in scalers_2d:
        scaler = scalers_2d[col]
        pred_reshaped = predictions_matrix[:, idx, :, :].reshape(-1, 1)
        predictions_2d_dict[col] = scaler.inverse_transform(pred_reshaped).reshape(-1, max_rows_2d, max_cols_2d)
predictions_2d_df = pd.DataFrame({col: data.tolist() for col, data in predictions_2d_dict.items()})

print("Prediction dataframes created.")

# Write predictions to a new NetCDF file
input_nc = Config.INPUT_NC
output_nc = Config.OUTPUT_NC
variables_1d_to_write = Config.VARIABLES_1D
variables_2d_to_write = Config.VARIABLES_2D
new_values = {}
new_values_2d = {}

with Dataset(input_nc, "r") as src:
    pft_gridcell_index = src.variables['pfts1d_gridcell_index'][:]
    original_shapes_1d = {var: src.variables[var].shape for var in variables_1d_to_write}
    original_nc_data = {var: src.variables[var][:] for var in src.variables}
    
    column_gridcell_index = src.variables['cols1d_gridcell_index'][:]
    original_shapes_2d = {var: src.variables[var].shape for var in variables_2d_to_write}
    original_nc_data_2d = {var: src.variables[var][:] for var in variables_2d_to_write}

# Map gridcells to their PFTs
gridcell_pft_map = defaultdict(list)
for pft_idx, gridcell_idx in enumerate(pft_gridcell_index):
    gridcell_pft_map[gridcell_idx - 1].append(pft_idx)

# Prepare 1D variable data for writing
for var in variables_1d_to_write:
    original_shape = original_shapes_1d[var]
    new_values[var] = original_nc_data[var].copy()
    predicted_values = np.array(predictions_1d_df[f"Y_{var}"].tolist(), dtype=np.float32)

    for grid_idx, pft_indices in gridcell_pft_map.items():
        if grid_idx >= predicted_values.shape[0]:
            continue
        pft_count = len(pft_indices)
        max_replace = min(17, pft_count)
        
        new_values[var][pft_indices[0]] = 0
        if max_replace > 1:
            new_values[var][pft_indices[1:max_replace]] = predicted_values[grid_idx, :max_replace]
        if pft_count > max_replace:
            new_values[var][pft_indices[max_replace:]] = 0

# Map gridcells to their columns
gridcell_column_map = defaultdict(list)
for column_idx, gridcell_idx in enumerate(column_gridcell_index):
    gridcell_column_map[gridcell_idx - 1].append(column_idx)

# Prepare 2D variable data for writing
for var in variables_2d_to_write:
    original_shape = original_shapes_2d[var]
    new_values_2d[var] = original_nc_data_2d[var].copy()
    predicted_values = np.array(predictions_2d_df[f"Y_{var}"].tolist(), dtype=np.float32)

    for grid_idx, column_indices in gridcell_column_map.items():
        if grid_idx >= predicted_values.shape[0]:
            continue
        if len(column_indices) > 0:
            first_column_idx = column_indices[0]
            if new_values_2d[var].ndim == 2:
                num_columns = new_values_2d[var].shape[1]
                temp = np.zeros(num_columns, dtype=np.float32)
                num_to_fill = min(10, num_columns)
                temp[:num_to_fill] = predicted_values[grid_idx, 0, :num_to_fill]
                new_values_2d[var][first_column_idx, :] = temp
            elif new_values_2d[var].ndim == 1:
                predicted_scalar = predicted_values[grid_idx, 0, 0]
                new_values_2d[var][first_column_idx] = predicted_scalar

# Write the new NetCDF file
with Dataset(input_nc, "r") as src, Dataset(output_nc, "w", format="NETCDF4") as dst:
    for name, dimension in src.dimensions.items():
        dst.createDimension(name, (None if dimension.isunlimited() else len(dimension)))
    
    for name, variable in src.variables.items():
        dst_var = dst.createVariable(name, variable.dtype, variable.dimensions)
        dst_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})

        if name in new_values:
            dst_var[:] = new_values[name]
        elif name in new_values_2d:
            data_to_write = new_values_2d[name]
            if len(dst_var.dimensions) == 2:
                dst_var[:, :] = data_to_write
            else:
                dst_var[:] = data_to_write
        else:
            dst_var[:] = variable[:]

print(f"New NetCDF file successfully created: {output_nc}")