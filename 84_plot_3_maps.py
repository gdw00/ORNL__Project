
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import os




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

#####H2OSOI_10CM start
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


df.drop(columns=["H2OSOI_10CM_filled"], inplace=True)
df.drop(columns=["H2OSOI_10CM_original"], inplace=True)

#####end


# df = df.dropna(subset=['H2OSOI_10CM'])
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
predictions_scalar = np.concatenate(predictions_scalar, axis=0) 
predictions_vector = np.concatenate(predictions_vector, axis=0)  
predictions_matrix = np.concatenate(predictions_matrix, axis=0)  

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


print(f"Filtered dataset: {len(df)} samples remaining after removing NaN H2OSOI values.")

for col in x_list_columns_1d + y_list_columns_1d:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x[:17] if isinstance(x, list) else x)

for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)


max_rows_2d = df[list_columns_2d].applymap(lambda x: x.shape[0] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()
max_cols_2d = df[list_columns_2d].applymap(lambda x: x.shape[1] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()



for col in list_columns_1d:
    df[col] = df[col].apply(lambda x: np.pad(np.array(x), (0, max_length_1d - len(x)), mode='constant') if isinstance(x, (list, np.ndarray)) else np.zeros(max_length_1d))

for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.pad(x, ((0, max_rows_2d - x.shape[0]), (0, max_cols_2d - x.shape[1])), mode='constant') if isinstance(x, np.ndarray) and x.ndim == 2 else np.zeros((max_rows_2d, max_cols_2d)))


def plot_comparison_map(df, predictions, variable, title,save_path_template):
    """

    """

    min_value = min(df[variable].min(), predictions[variable].min())
    max_value = max(df[variable].max(), predictions[variable].max())


    df = df.reset_index(drop=True)
    df_filtered = df.iloc[:len(predictions)] 
    difference = predictions[variable] - df_filtered[variable]


    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))


    print(f"  🔹 Predictions - Min: {predictions[variable].min()}, Max: {predictions[variable].max()}")
    print(f"  🔹 Ground Truth - Min: {df[variable].min()}, Max: {df[variable].max()}")
    print(f"  🔹 Difference - Min: {difference.min()}, Max: {difference.max()}")

    plt.figure(figsize=(12, 15)) 


    plt.subplot(3, 1, 1)
    scatter1 = plt.scatter(
        df['Longitude'], df['Latitude'], c=predictions[variable], cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    cbar1 = plt.colorbar(scatter1, label=f"{variable} (Prediction - Original Scale)")
    cbar1.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar1.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar1.ax.text(1.05, -0.02, f"Min: {min_value:.2f}", transform=cbar1.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - Prediction", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)


    plt.subplot(3, 1, 2)
    scatter2 = plt.scatter(
        df['Longitude'], df['Latitude'], c=df[variable], cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    cbar2 = plt.colorbar(scatter2, label=f"{variable} (Ground Truth - Original Scale)")
    cbar2.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar2.ax.text(1.05, -0.02, f"Min: {min_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - Ground Truth", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    scatter3 = plt.scatter(
        df['Longitude'], df['Latitude'], c=difference, cmap='RdBu', s=10, vmin=diff_min, vmax=diff_max
    )
    cbar3 = plt.colorbar(scatter3, label=f"{variable} Difference (Prediction - Ground Truth)")
    cbar3.ax.text(1.05, 1.02, f"Max: {difference.max():.2f}", transform=cbar3.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar3.ax.text(1.05, -0.02, f"Min: {difference.min():.2f}", transform=cbar3.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - Difference", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)


    save_path = save_path_template.format(variable=variable)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

save_path_1 = './1_plot_5_gridcell/4_{variable}_comparison_map_with_difference.png'
for variable in target_columns:

    plot_comparison_map(df, predictions_df, variable, title=f"{variable} Comparison",save_path_template=save_path_1)



####################################1d


def p2_plot_comparison_map(df, predictions, variable, title,save_path_template,i):
   

    df_values = np.vstack(df[variable].values)[:, i]  
    predictions_values = np.vstack(predictions[variable].values)[:, i]  


    min_value = min(df_values.min(), predictions_values.min())
    max_value = max(df_values.max(), predictions_values.max())


    difference = predictions_values - df_values
    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))


    plt.figure(figsize=(12, 15))  


    plt.subplot(3, 1, 1)
    scatter1 = plt.scatter(
        df['Longitude'], df['Latitude'], c=predictions_values, cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    cbar1 = plt.colorbar(scatter1, label=f"{variable} (Prediction - Original Scale)")
    cbar1.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar1.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar1.ax.text(1.05, -0.02, f"Min: {min_value:.2f}", transform=cbar1.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - Prediction", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)


    plt.subplot(3, 1, 2)
    scatter2 = plt.scatter(
        df['Longitude'], df['Latitude'], c=df_values, cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    cbar2 = plt.colorbar(scatter2, label=f"{variable} (Ground Truth - Original Scale)")
    cbar2.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar2.ax.text(1.05, -0.02, f"Min: {min_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - Ground Truth", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)


    plt.subplot(3, 1, 3)
    scatter3 = plt.scatter(
        df['Longitude'], df['Latitude'], c=difference, cmap='RdBu', s=10, vmin=diff_min, vmax=diff_max
    )
    cbar3 = plt.colorbar(scatter3, label=f"{variable} Difference (Prediction - Ground Truth)")
    cbar3.ax.text(1.05, 1.02, f"Max: {difference.max():.2f}", transform=cbar3.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar3.ax.text(1.05, -0.02, f"Min: {difference.min():.2f}", transform=cbar3.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - Difference", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)


    save_path = save_path_template.format(variable=variable, i = i)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# save_path_2 = './2_plot_3_pft/4_{variable}_{i}_comparison_map_with_difference.png'
# for variable in y_list_columns_1d:

#     y_pred = np.vstack(predictions_1d_df[variable].values)
#     # y_true = np.vstack(ground_truth_1d_df[var].values)
#     num_features = y_pred.shape[1]  
#     for i in range(num_features):


#         p2_plot_comparison_map(df, predictions_1d_df, variable, title=f"{variable}_{i} Comparison",save_path_template=save_path_2,i=i)

# print("done**")


###################################2d


def p3_plot_comparison_map(df, predictions, variable, title, save_path_template, layer):


    


    df_values =   np.array(df[variable].tolist())[:, 0, layer] 
    predictions_values = np.array(predictions[variable].tolist())[:, 0, layer]  

    min_value = min(df_values.min(), predictions_values.min())
    max_value = max(df_values.max(), predictions_values.max())


    difference = predictions_values - df_values
    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))


    plt.figure(figsize=(12, 15))  


    plt.subplot(3, 1, 1)
    scatter1 = plt.scatter(
        df['Longitude'], df['Latitude'], c=predictions_values, cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    cbar1 = plt.colorbar(scatter1, label=f"{variable} (Prediction - Original Scale)")
    cbar1.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar1.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar1.ax.text(1.05, -0.02, f"Min: {min_value:.2f}", transform=cbar1.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - Prediction", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)


    plt.subplot(3, 1, 2)
    scatter2 = plt.scatter(
        df['Longitude'], df['Latitude'], c=df_values, cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    cbar2 = plt.colorbar(scatter2, label=f"{variable} (Ground Truth - Original Scale)")
    cbar2.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar2.ax.text(1.05, -0.02, f"Min: {min_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - Ground Truth", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)


    plt.subplot(3, 1, 3)
    scatter3 = plt.scatter(
        df['Longitude'], df['Latitude'], c=difference, cmap='RdBu', s=10, vmin=diff_min, vmax=diff_max
    )
    cbar3 = plt.colorbar(scatter3, label=f"{variable} Difference (Prediction - Ground Truth)")
    cbar3.ax.text(1.05, 1.02, f"Max: {difference.max():.2f}", transform=cbar3.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar3.ax.text(1.05, -0.02, f"Min: {difference.min():.2f}", transform=cbar3.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - Difference", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    save_path = save_path_template.format(variable=variable, layer=layer)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# save_path_3 = './3_plot_3_column/4_{variable}_comparison_map_with_difference.png'
# for variable in y_list_columns_2d:

#     plot_comparison_map(df, predictions_2d_df, variable, title=f"{variable} Comparison",save_path_template=save_path_3)

# print("done**")
