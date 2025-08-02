import glob
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.jit.load("LSTM_model_62.pt")
model.eval()
model.to(device)

input_files = sorted(glob.glob('/mnt/DATA/dawei/3_oak_all/31_summer_paper/5_add_all_8_cnp/0_dataset/1_training_data_batch_*.pkl'))

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

print(df.info())
print(df.info())
cols_to_drop = [
    'H2OSFC', 'H2OSNO', 'H2OSOI_LIQ', 'H2OSOI_ICE', 'LAKE_SOILC', 'H2OCAN',
    'TH2OSFC', 'T_GRND', 'T_GRND_R', 'T_GRND_U', 'T_LAKE', 'T_SOISNO', 
    'TS_TOPO', 'taf', 'T_VEG', 'T10_VALUE',
    'Y_H2OSFC', 'Y_H2OSNO', 'Y_H2OSOI_LIQ', 'Y_H2OSOI_ICE', 'Y_LAKE_SOILC', 'Y_H2OCAN',
    'Y_TH2OSFC', 'Y_T_GRND', 'Y_T_GRND_R', 'Y_T_GRND_U', 'Y_T_LAKE', 'Y_T_SOISNO', 
    'Y_TS_TOPO', 'Y_taf', 'Y_T_VEG', 'Y_T10_VALUE',
]
df = df.drop(columns=cols_to_drop, errors='ignore')
print(df.info())



print("\n" + "="*80)
print("Starting spatial nearest neighbor interpolation for _vr variables...")

vars_to_interpolate = ['sminn_vr', 'smin_nh4_vr', 'Y_sminn_vr', 'Y_smin_nh4_vr']

for col in vars_to_interpolate:
    if col in df.columns:
        print(f"\n--- Processing column: '{col}' ---")
        
        bad_mask = df[col].apply(lambda x: np.isnan(np.sum(x)))
        good_mask = ~bad_mask

        if not bad_mask.any():
            print("  No NaN samples found in this column, no interpolation needed.")
            continue

        print(f"  - Found {bad_mask.sum()} bad samples, {good_mask.sum()} good samples.")

        points_good = df.loc[good_mask, ['Longitude', 'Latitude']].values
        points_bad = df.loc[bad_mask, ['Longitude', 'Latitude']].values

        kdtree_good_points = KDTree(points_good)
        
        distances, nearest_indices_in_good_set = kdtree_good_points.query(points_bad, k=1)

        good_df_indices = df.index[good_mask]
        bad_df_indices = df.index[bad_mask]

        print("  - Starting replacement with nearest neighbor valid arrays...")
        num_replaced = 0
        for i, bad_idx in enumerate(bad_df_indices):
            nearest_pos = nearest_indices_in_good_set[i]
            nearest_good_df_idx = good_df_indices[nearest_pos]
            
            df.at[bad_idx, col] = df.at[nearest_good_df_idx, col]
            num_replaced += 1
            
        print(f"  - Successfully replaced {num_replaced} bad samples.")

    else:
        print(f"  Warning: Column '{col}' not found in DataFrame.")

print("\n" + "="*80)
print("All specified variable interpolation repair completed!")

print("\nStarting final verification to ensure all NaN have been cleared...")
all_clear = True
for col in vars_to_interpolate:
    if col in df.columns:
        has_nan = df[col].apply(lambda x: np.isnan(np.sum(x))).any()
        if has_nan:
            print(f"  Error: Column '{col}' still contains NaN after interpolation!")
            all_clear = False
        else:
            print(f"  Column '{col}' has been successfully cleaned.")
if all_clear:
    print("\nCongratulations! All NaN in target variables have been successfully repaired through interpolation!")
print("="*80 + "\n")

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
df["H2OSOI_10CM"] = np.where(np.isnan(df["H2OSOI_10CM"]), df["H2OSOI_10CM_filled"], df["H2OSOI_10CM"])
df["H2OSOI_10CM_original"] = df["H2OSOI_10CM"].copy()
df["H2OSOI_10CM"] = np.where(np.isnan(df["H2OSOI_10CM"]), df["H2OSOI_10CM_filled"], df["H2OSOI_10CM"])
unchanged_values = (df["H2OSOI_10CM_original"].dropna() == df["H2OSOI_10CM"].dropna()).all()
print(f"Are uninterpolated region values completely consistent: {unchanged_values}")
df.drop(columns=["H2OSOI_10CM_filled"], inplace=True)
df.drop(columns=["H2OSOI_10CM_original"], inplace=True)
print("Shape of df['H2OSOI_10CM']:", df["H2OSOI_10CM"].shape)
print("Interpolation completed, NaN values have been filled!")
print(f"Filtered dataset: {len(df)} samples remaining after removing NaN H2OSOI values.")
cols_to_drop = [
]

df = df.drop(columns=cols_to_drop, errors='ignore')
print("Deleted unnecessary features, current dataset contains columns:", df.columns.tolist())

x_list_columns_2d = [
    'soil3c_vr', 'soil4c_vr', 'cwdc_vr' ,'cwdn_vr' ,'secondp_vr' ,'cwdp' ,'totcolp','totlitc','cwdp_vr',
    'soil1c_vr', 'soil1n_vr', 'soil1p_vr',
    'soil2c_vr', 'soil2n_vr', 'soil2p_vr',
    'soil3n_vr', 'soil3p_vr',
    'soil4n_vr', 'soil4p_vr',
    'litr1c_vr', 'litr2c_vr', 'litr3c_vr',
    'litr1n_vr', 'litr2n_vr', 'litr3n_vr',
    'litr1p_vr', 'litr2p_vr', 'litr3p_vr',
    'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr',
    'labilep_vr', 'occlp_vr', 'primp_vr'
]

y_list_columns_2d = [
    'Y_soil3c_vr', 'Y_soil4c_vr', 'Y_cwdc_vr','Y_cwdn_vr' ,'Y_secondp_vr','Y_cwdp' ,'Y_totcolp','Y_totlitc','Y_cwdp_vr',
    'Y_soil1c_vr', 'Y_soil1n_vr', 'Y_soil1p_vr',
    'Y_soil2c_vr', 'Y_soil2n_vr', 'Y_soil2p_vr',
    'Y_soil3n_vr', 'Y_soil3p_vr',
    'Y_soil4n_vr', 'Y_soil4p_vr',
    'Y_litr1c_vr', 'Y_litr2c_vr', 'Y_litr3c_vr',
    'Y_litr1n_vr', 'Y_litr2n_vr', 'Y_litr3n_vr',
    'Y_litr1p_vr', 'Y_litr2p_vr', 'Y_litr3p_vr',
    'Y_sminn_vr', 'Y_smin_no3_vr', 'Y_smin_nh4_vr',
    'Y_labilep_vr', 'Y_occlp_vr', 'Y_primp_vr'
]

x_list_columns_1d = [
    'deadcrootc', 'deadstemc', 'tlai' ,'totvegc' ,'deadstemn' ,'deadcrootn' ,'deadstemp' ,'deadcrootp' ,
    'leafc','leafc_storage' ,'frootc' ,'frootc_storage',
    'leafn', 'leafn_storage', 'frootn','frootn_storage',
    'leafp', 'leafp_storage', 'frootp','frootp_storage',
    'livestemc', 'livestemc_storage', 
    'livestemn', 'livestemn_storage', 
    'livestemp', 'livestemp_storage',
    'deadcrootc_storage', 'deadstemc_storage', 
    'livecrootc', 'livecrootc_storage', 
    'deadcrootn_storage', 'deadstemn_storage', 
    'livecrootn', 'livecrootn_storage', 
    'deadcrootp_storage', 'deadstemp_storage', 
    'livecrootp', 'livecrootp_storage'
]

y_list_columns_1d = [
    'Y_deadcrootc', 'Y_deadstemc', 'Y_tlai','Y_totvegc' ,'Y_deadstemn' ,'Y_deadcrootn' ,'Y_deadstemp' ,
    'Y_deadcrootp' ,'Y_leafc','Y_leafc_storage' ,'Y_frootc' ,'Y_frootc_storage',
    'Y_leafn', 'Y_leafn_storage', 'Y_frootn','Y_frootn_storage',
    'Y_leafp', 'Y_leafp_storage', 'Y_frootp','Y_frootp_storage',
    'Y_livestemc', 'Y_livestemc_storage', 
    'Y_livestemn', 'Y_livestemn_storage', 
    'Y_livestemp', 'Y_livestemp_storage',
    'Y_deadcrootc_storage', 'Y_deadstemc_storage', 
    'Y_livecrootc', 'Y_livecrootc_storage', 
    'Y_deadcrootn_storage', 'Y_deadstemn_storage', 
    'Y_livecrootn', 'Y_livecrootn_storage', 
    'Y_deadcrootp_storage', 'Y_deadstemp_storage', 
    'Y_livecrootp', 'Y_livecrootp_storage'
]

list_columns_2d = x_list_columns_2d + y_list_columns_2d
list_columns_1d = x_list_columns_1d + y_list_columns_1d
for col in y_list_columns_1d:
    if col in df.columns:
        flattened_values = np.concatenate(df[col].dropna().values).flatten()
        print(f"{col}: min={flattened_values.min():.12f}, max={flattened_values.max():.12f}")
    else:
        print(f"Warning: {col} not found in the dataframe.")

for col in x_list_columns_1d + y_list_columns_1d:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x[1:17] if isinstance(x, list) else x)

print("Columns truncated to first 17 elements.")
max_length_1d = df[list_columns_1d].applymap(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).max().max()

for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

print("Starting 2D data processing pipeline...")

cols_2d = x_list_columns_2d + y_list_columns_2d
vars_to_reshape = ['cwdp', 'totcolp','totlitc', 'Y_cwdp', 'Y_totcolp','Y_totlitc']

for col in cols_2d:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, (list, tuple)) else x)

        if col in vars_to_reshape:
            df[col] = df[col].apply(lambda x: x.reshape(-1, 1) if isinstance(x, np.ndarray) and x.ndim == 1 else x)

        df[col] = df[col].apply(lambda x: x[:, :10] if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] >= 10 else x)

print("2D data processing pipeline finished.")
max_rows_2d = df[list_columns_2d].applymap(lambda x: x.shape[0] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()
max_cols_2d = df[list_columns_2d].applymap(lambda x: x.shape[1] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()

print(f"Max 1D length: {max_length_1d}")
print(f"Max 2D shape: ({max_rows_2d}, {max_cols_2d})")

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

print("All list variables have been successfully converted to PyTorch tensors!")
time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
static_columns = [
    col for col in df.columns 
    if col not in time_series_columns 
    and not col.startswith('Y_') 
    and col not in x_list_columns_2d 
    and col not in x_list_columns_1d
    and not col.startswith('pft_')  
]
print("Static Columns (filtered):", static_columns)
target_columns = [
    col for col in df.columns 
    if col.startswith('Y_') 
    and col not in y_list_columns_2d 
    and col not in y_list_columns_1d
]

print("Target Columns (filtered):", target_columns)
for col in time_series_columns: 
    print('processing:', col)
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

scaler_static = MinMaxScaler(feature_range=(0, 1))
static_data = scaler_static.fit_transform(df[static_columns].values)

scaler_target = MinMaxScaler(feature_range=(0, 1))
target_data = scaler_target.fit_transform(df[target_columns].values)
print('Normalize the static and target-end')
print('Convert data to PyTorch tensors')
time_series_data = torch.tensor(time_series_data, dtype=torch.float32).to(device)
static_data = torch.tensor(static_data, dtype=torch.float32).to(device)
target_data = torch.tensor(target_data, dtype=torch.float32).to(device)



print("\nStarting column-wise standardization of 1D features (StandardScaler)...")

pft_columns = [col for col in df.columns if col.startswith('pft_')]
pft_data = np.stack(df[pft_columns].apply(lambda row: np.concatenate(row), axis=1).values)

scaler_pft = MinMaxScaler()
pft_data = scaler_pft.fit_transform(pft_data)

pft_data = torch.tensor(pft_data, dtype=torch.float32)

print("\nData range before normalization for `list_columns_1d`:")
for col in list_columns_1d:
    min_val = df[col].apply(lambda x: np.min(x) if isinstance(x, (list, np.ndarray)) else 0).min()
    max_val = df[col].apply(lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) else 0).max()
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")

scalers_1d = {}
for col in list_columns_1d:
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    col_data = np.vstack(df[col].values)
    
    scaled_data = scaler.fit_transform(col_data)
    
    scalers_1d[col] = scaler

    df[col] = list(scaled_data)
scalers_2d = {}

print("\nStarting StandardScaler normalization for 2D features...")
print("\nData range before normalization for `list_columns_2d`:")
for col in list_columns_2d:
    min_val = df[col].apply(lambda x: np.min(x) if isinstance(x, (list, np.ndarray)) else 0).min()
    max_val = df[col].apply(lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) else 0).max()
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")
for col in list_columns_2d:
    print(f"\nProcessing 2D feature column: {col}")

    print(f"  - Shape before normalization `df[{col}].values`: {[x.shape for x in df[col].values[:5]]} (first 5 examples)")

    try:
        reshaped_data = np.vstack(df[col].values).reshape(-1, 1)
    except Exception as e:
        print(f"vstack failed, error: {e}")
        continue

    print(f"  - `reshaped_data` shape: {reshaped_data.shape}")

    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(reshaped_data)
        scalers_2d[col] = scaler
        print(f"  - `scaled_data` shape: {scaled_data.shape}")
    except Exception as e:
        print(f"Normalization failed, error: {e}")
        continue

    try:
        reshaped_scaled_data = scaled_data.reshape(len(df), max_rows_2d, max_cols_2d)
        print(f"  - `reshaped_scaled_data` shape: {reshaped_scaled_data.shape}")
    except Exception as e:
        print(f"Reshape failed, error: {e}")
        continue

    try:
        df[col] = list(reshaped_scaled_data)
        print(f"Normalization successful, df[{col}] assignment completed")
    except Exception as e:
        print(f"Assignment back to df[{col}] failed, error: {e}")

print("\nAll 2D features have been successfully normalized!")

print("\nData range check after normalization:")
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

batch_size = 16
predictions_scalar = []
predictions_vector = []
predictions_matrix = []
with torch.no_grad():
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        time_series_batch = time_series_data[i:end_idx].to(device)
        static_batch = static_data[i:end_idx].to(device)
        pft_data_batch = pft_data[i:end_idx].to(device)
        columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in x_list_columns_1d_tensors.items()}
        columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in x_list_columns_2d_tensors.items()}
        y_columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in y_list_columns_1d_tensors.items()}
        y_columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in y_list_columns_2d_tensors.items()}
        
        columns_1d_batch_tensor = torch.cat([tensor for tensor in columns_1d_batch.values()], dim=1).to(device)
        columns_2d_batch_tensor = torch.cat([tensor.unsqueeze(1) for tensor in columns_2d_batch.values()], dim=1).to(device)

        scalar_pred, vector_pred, matrix_pred = model(time_series_batch, static_batch, columns_1d_batch_tensor, columns_2d_batch_tensor, pft_data_batch)
        vector_pred = torch.maximum(vector_pred, torch.tensor(0.0, device=vector_pred.device))
        matrix_pred = torch.maximum(matrix_pred, torch.tensor(0.0, device=matrix_pred.device))
        predictions_scalar.append(scalar_pred.cpu().numpy())
        predictions_vector.append(vector_pred.cpu().numpy())
        predictions_matrix.append(matrix_pred.cpu().numpy())
predictions_scalar = np.concatenate(predictions_scalar, axis=0)
predictions_vector = np.concatenate(predictions_vector, axis=0)
predictions_matrix = np.concatenate(predictions_matrix, axis=0)

predictions_scalar_np = scaler_target.inverse_transform(predictions_scalar)
predictions_df = pd.DataFrame(predictions_scalar_np, columns=target_columns)
predictions_1d_dict = {}

for idx, col in enumerate(x_list_columns_1d):
    y_col = 'Y_' + col

    if col in scalers_1d:
        predictions_1d_dict[y_col] = scalers_1d[y_col].inverse_transform(predictions_vector[:, idx, :])
    else:
        print(f"Scaler not found for {col}, skipping inverse normalization")

predictions_1d_df = pd.DataFrame({col: predictions_1d_dict[col].tolist() for col in predictions_1d_dict})
print("predictions_1d_df.shape:", predictions_1d_df.shape)

predictions_2d_dict = {}
print("\nInverse normalizing `y_list_columns_2d` predicted values:")

if not isinstance(scalers_2d, dict):
    raise TypeError("scalers_2d should be a dict, but is now: ", type(scalers_2d))

print(f"Available scalers_2d keys: {list(scalers_2d.keys())}")

for idx, col in enumerate(x_list_columns_2d):
    y_col = 'Y_' + col

    if col in scalers_2d:
        scaler = scalers_2d[y_col]
        pred_reshaped = predictions_matrix[:, idx, :, :].reshape(-1, 1)
        predictions_2d_dict[y_col] = scaler.inverse_transform(pred_reshaped).reshape(-1, max_rows_2d, max_cols_2d)
    else:
        print(f"Scaler not found for {col}, skipping inverse normalization")

predictions_2d_df = pd.DataFrame({col: predictions_2d_dict[col].tolist() for col in predictions_2d_dict})
print("!!!!!!!!!!data done!")
print("predictions_2d_df.shape:", predictions_2d_df.shape)

def analyze_written_data_stats(title, df_to_analyze, variables):
    print("\n" + "="*70)
    print(f"Precise Analysis (Data to be Written to File Only): {title}")
    print("="*70)
    
    for var in variables:
        print(f"\n--- Analyzing Variable: {var} ---")
        if var in df_to_analyze.columns:
            try:
                all_values = np.concatenate([x[0] for x in df_to_analyze[var].values])
                
                max_val = np.max(all_values)
                min_val = np.min(all_values)
                sum_val = np.sum(all_values)
                mean_val = np.mean(all_values)

                print(f"  Max: {max_val}")
                print(f"  Min: {min_val}")
                print(f"  Sum: {sum_val}")
                print(f"  Mean: {mean_val}")

            except Exception as e:
                print(f"  Could not analyze variable '{var}'. Error: {e}")
        else:
            print(f"  Variable '{var}' not found in the DataFrame.")
    print("="*70 + "\n")

variables_to_inspect = ['Y_labilep_vr', 'Y_occlp_vr', 'Y_primp_vr', 'Y_secondp_vr','Y_soil3c_vr']

analyze_written_data_stats("Model Predictions (Row 0 Only)", predictions_2d_df, variables_to_inspect)

def analyze_remaining_data_stats(title, df_to_analyze, variables):
    print("\n" + "="*70)
    print(f"Analysis of Remaining Data (Rows 1-17): {title}")
    print("="*70)
    
    for var in variables:
        print(f"\n--- Analyzing Variable: {var} ---")
        if var in df_to_analyze.columns:
            try:
                all_values = np.concatenate([np.ravel(x[1:]) for x in df_to_analyze[var].values])
                
                if all_values.size == 0:
                    print("  (No remaining data to analyze)")
                    continue

                max_val = np.max(all_values)
                min_val = np.min(all_values)
                sum_val = np.sum(all_values)
                mean_val = np.mean(all_values)

                print(f"  Max: {max_val}")
                print(f"  Min: {min_val}")
                print(f"  Sum: {sum_val}")
                print(f"  Mean: {mean_val}")

            except Exception as e:
                print(f"  Could not analyze variable '{var}'. Error: {e}")
        else:
            print(f"  Variable '{var}' not found in the DataFrame.")
    print("="*70 + "\n")

variables_to_inspect = ['Y_labilep_vr', 'Y_occlp_vr', 'Y_primp_vr', 'Y_secondp_vr','Y_soil3c_vr']

analyze_remaining_data_stats("Model Predictions (Remaining 17 Rows)", predictions_2d_df, variables_to_inspect)

def analyze_and_print_stats(title, df_to_analyze, variables):
    print("\n" + "="*70)
    print(f"Detailed Analysis (All Rows): {title}")
    print("="*70)
    
    for var in variables:
        print(f"\n--- Analyzing Variable: {var} ---")
        if var in df_to_analyze.columns:
            try:
                all_values = np.concatenate([np.ravel(x) for x in df_to_analyze[var].values])
                
                max_val = np.max(all_values)
                min_val = np.min(all_values)
                sum_val = np.sum(all_values)
                mean_val = np.mean(all_values)

                print(f"  Max: {max_val}")
                print(f"  Min: {min_val}")
                print(f"  Sum: {sum_val}")
                print(f"  Mean: {mean_val}")

            except Exception as e:
                print(f"  Could not analyze variable '{var}'. Error: {e}")
        else:
            print(f"  Variable '{var}' not found in the DataFrame.")
    print("="*70 + "\n")

variables_to_inspect = ['Y_labilep_vr', 'Y_occlp_vr', 'Y_primp_vr', 'Y_secondp_vr','Y_soil3c_vr']

analyze_and_print_stats("Model Predictions (All Rows)", predictions_2d_df, variables_to_inspect)


new_values = {}
input_nc = "/mnt/DATA/dawei/3_oak_all/30_base_18_add_pft_variables/4_nc_file/20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc"
output_nc = "./4_nc_file/updated11_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc"
variables_1d = [
    'deadcrootc', 'deadstemc', 'tlai' ,'totvegc' ,'deadstemn' ,'deadcrootn' ,'deadstemp' ,'deadcrootp' ,
    'leafc','leafc_storage' ,'frootc' ,'frootc_storage',
    'leafn', 'leafn_storage', 'frootn','frootn_storage',
    'leafp', 'leafp_storage', 'frootp','frootp_storage',
    'livestemc', 'livestemc_storage', 
    'livestemn', 'livestemn_storage', 
    'livestemp', 'livestemp_storage',
    'deadcrootc_storage', 'deadstemc_storage', 
    'livecrootc', 'livecrootc_storage', 
    'deadcrootn_storage', 'deadstemn_storage', 
    'livecrootn', 'livecrootn_storage', 
    'deadcrootp_storage', 'deadstemp_storage', 
    'livecrootp', 'livecrootp_storage'
]
variables_2d = [
    'soil3c_vr', 'soil4c_vr', 'cwdc_vr' ,'cwdn_vr' ,'secondp_vr' ,'cwdp' ,'totcolp','totlitc','cwdp_vr',
    'soil1c_vr', 'soil1n_vr', 'soil1p_vr',
    'soil2c_vr', 'soil2n_vr', 'soil2p_vr',
    'soil3n_vr', 'soil3p_vr',
    'soil4n_vr', 'soil4p_vr',
    'litr1c_vr', 'litr2c_vr', 'litr3c_vr',
    'litr1n_vr', 'litr2n_vr', 'litr3n_vr',
    'litr1p_vr', 'litr2p_vr', 'litr3p_vr',
    'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr',
    'labilep_vr', 'occlp_vr', 'primp_vr'
]

with Dataset(input_nc, "r") as src:
    original_shapes = {var: src.variables[var].shape for var in ( variables_1d + variables_2d)}

print("\nOriginal NetCDF variable shapes:")
print(original_shapes)

new_values = {}

with Dataset(input_nc, "r") as src:
    gridcell_lat = src.variables['grid1d_lat'][:]
    gridcell_lon = src.variables['grid1d_lon'][:]

    pft_lat = src.variables['pfts1d_lat'][:]
    pft_lon = src.variables['pfts1d_lon'][:]

    pft_gridcell_index = src.variables['pfts1d_gridcell_index'][:]

    original_shapes = {var: src.variables[var].shape for var in variables_1d}
    original_nc_data = {var: src.variables[var][:] for var in src.variables}

print("\nRead NetCDF variable information completed!")
print(f"Total {len(np.unique(pft_gridcell_index))} gridcells, {len(pft_lat)} pfts")

gridcell_pft_map = defaultdict(list)
for pft_idx, gridcell_idx in enumerate(pft_gridcell_index):
    gridcell_pft_map[gridcell_idx - 1].append(pft_idx)

print(f"Statistics completed, number of PFTs under each gridcell recorded!")
print(f"Example: Number of PFTs for first 5 gridcells:")
for i, (grid_idx, pft_indices) in enumerate(list(gridcell_pft_map.items())[:5]):
    print(f"  Gridcell {grid_idx}: {len(pft_indices)} PFTs")

for var in variables_1d:
    original_shape = original_shapes[var]

    new_values[var] = original_nc_data[var].copy()

    predicted_values = np.array(predictions_1d_df[f"Y_{var}"].tolist(), dtype=np.float32)

    print(f"\nVariable {var} processing started: original shape {original_shape}, predicted values shape {predicted_values.shape}")

    for grid_idx, pft_indices in gridcell_pft_map.items():
        if grid_idx >= predicted_values.shape[0]:
            print(f"Warning: grid_idx {grid_idx} exceeds range, skipping assignment")
            continue
        pft_count = len(pft_indices)
        max_replace = min(17, pft_count)

        new_values[var][pft_indices[0]] = 0
        if max_replace > 1:
            new_values[var][pft_indices[1:max_replace]] = predicted_values[grid_idx, :max_replace]

        if pft_count > max_replace:
            new_values[var][pft_indices[max_replace:]] = 0

    if new_values[var].shape != original_shape:
        raise ValueError(f"Variable {var} predicted shape {new_values[var].shape} does not match original shape {original_shape}!")

    print(f"Variable {var} processing completed!\n")

random_gridcells = random.sample(list(gridcell_pft_map.keys()), 5)
print("\nRandomly checking data update status for 5 gridcells:")
for grid_idx in random_gridcells:
    pft_indices = gridcell_pft_map[grid_idx]
    print(f"Gridcell {grid_idx}: {len(pft_indices)} PFTs")
    
    for var in variables_1d:
        assigned_values = new_values[var][pft_indices[:20]]
        print(f"  {var} first 20 updated values: {assigned_values[:20]} ...")

print("\nvariables_1d processing completed, first 17 pfts replaced, remaining filled with 0 (based on gridcell-pft structure)!")



new_values_2d = {}

with Dataset(input_nc, "r") as src:
    column_gridcell_index = src.variables['cols1d_gridcell_index'][:]

    original_shapes_2d = {var: src.variables[var].shape for var in variables_2d}
    original_nc_data_2d = {var: src.variables[var][:] for var in variables_2d}

print("\nRead NetCDF variable information completed!")
print(f"Total {len(np.unique(column_gridcell_index))} gridcells, {len(column_gridcell_index)} columns")

gridcell_column_map = defaultdict(list)
for column_idx, gridcell_idx in enumerate(column_gridcell_index):
    gridcell_column_map[gridcell_idx - 1].append(column_idx)

print(f"Statistics completed, number of columns under each gridcell recorded!")
print(f"Example: Number of columns for first 5 gridcells:")
for i, (grid_idx, column_indices) in enumerate(list(gridcell_column_map.items())[:5]):
    print(f"  Gridcell {grid_idx}: {len(column_indices)} columns")
special_four_vars = ['labilep_vr', 'occlp_vr', 'primp_vr', 'secondp_vr']
print(f"Special treatment (update all columns) for: {special_four_vars}")
for var in variables_2d:
    original_shape = original_shapes_2d[var]

    new_values_2d[var] = original_nc_data_2d[var].copy()

    predicted_values = np.array(predictions_2d_df[f"Y_{var}"].tolist(), dtype=np.float32)

    print(f"\nVariable {var} processing started: original shape {original_shape}, predicted values shape {predicted_values.shape}")

    if var in special_four_vars:
        print(f"  Updating variable: {var} -> using [SEQUENTIAL MAPPING] logic")
        
        for grid_idx, column_indices in gridcell_column_map.items():
            if grid_idx >= predicted_values.shape[0] or not column_indices: continue

            for i, column_idx in enumerate(column_indices):
                
                if i >= 18: 
                    break 

                if new_values_2d[var].ndim == 2:
                    original_layer_count = new_values_2d[var].shape[1]
                    temp_vector = np.zeros(original_layer_count, dtype=np.float32)
                    num_to_fill = min(10, original_layer_count)
                    
                    prediction_slice = predicted_values[grid_idx, i, :num_to_fill]
                    temp_vector[:num_to_fill] = prediction_slice
                    new_values_2d[var][column_idx, :] = temp_vector

                elif new_values_2d[var].ndim == 1:
                    predicted_scalar = predicted_values[grid_idx, i, 0]
                    new_values_2d[var][column_idx] = predicted_scalar

    else:

        for grid_idx, column_indices in gridcell_column_map.items():
            if grid_idx >= predicted_values.shape[0]:
                print(f"Warning: grid_idx {grid_idx} exceeds range, skipping assignment")
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
    if new_values_2d[var].shape != original_shape:
        raise ValueError(f"Variable {var} predicted shape {new_values_2d[var].shape} does not match original shape {original_shape}!")

    print(f"Variable {var} processing completed!\n")

print("\nvariables_2d processing completed, **only first column replaced, others filled with 0** (based on gridcell-column structure)!")



with Dataset(input_nc, "r") as src:
    with Dataset(output_nc, "w", format="NETCDF4") as dst:
        print("\nCreating new NetCDF file (keeping original structure, only updating variable values)...")

        for name, dimension in src.dimensions.items():
            dst.createDimension(name, (None if dimension.isunlimited() else len(dimension)))

        for name, variable in src.variables.items():
            var_dtype = variable.dtype
            var_dims = variable.dimensions
            dst_var = dst.createVariable(name, var_dtype, var_dims)

            dst_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})

            if name in new_values:
                print(f"Updating variable: {name} (1D)")
                print(f"{name} (1D) statistics: min={new_values[name].min()}, max={new_values[name].max()}")
                dst_var[:] = new_values[name]
            elif name in new_values_2d:
                data_to_write = new_values_2d[name]
                print(f"Updating variable: {name}")
                print(f"{name} statistics: min={data_to_write.min():.4f}, max={data_to_write.max():.4f}")

                if len(dst_var.dimensions) == 2:
                    dst_var[:, :] = data_to_write
                elif len(dst_var.dimensions) == 1:
                    dst_var[:] = data_to_write
                else:
                    dst_var[:] = data_to_write
            else:
                dst_var[:] = variable[:]

print("\nNetCDF file has been successfully updated and saved:", output_nc)