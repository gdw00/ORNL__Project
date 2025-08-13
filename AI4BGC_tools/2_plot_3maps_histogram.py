import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = torch.jit.load("LSTM_model_62_best.pt")
model.eval() 
model.to(device) 

input_files = sorted(glob.glob('./dataset/1_training_data_batch_*.pkl'))
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

df["H2OSOI_10CM"] = np.where(np.isnan(df["H2OSOI_10CM"]), df["H2OSOI_10CM_filled"], df["H2OSOI_10CM"])

df["H2OSOI_10CM_original"] = df["H2OSOI_10CM"].copy()
df["H2OSOI_10CM"] = np.where(np.isnan(df["H2OSOI_10CM"]), df["H2OSOI_10CM_filled"], df["H2OSOI_10CM"])

df.drop(columns=["H2OSOI_10CM_filled"], inplace=True)
df.drop(columns=["H2OSOI_10CM_original"], inplace=True)
print("Shape of df['H2OSOI_10CM']:", df["H2OSOI_10CM"].shape)



# cols_to_drop = [
#     'soil1p_vr', 'avail_retransp', 'plant_pdemand', 'leafp_storage',
#     'Y_soil1p_vr', 'Y_avail_retransp', 'Y_plant_pdemand', 'Y_leafp_storage'
# ]
# df = df.drop(columns=cols_to_drop, errors='ignore')

x_list_columns_2d = [
    'soil3c_vr', 'soil4c_vr', 'cwdc_vr'
    
]
y_list_columns_2d = [
    'Y_soil3c_vr', 'Y_soil4c_vr', 'Y_cwdc_vr'
]

x_list_columns_1d = ['deadcrootc', 'deadstemc', 'tlai']
y_list_columns_1d = ['Y_deadcrootc', 'Y_deadstemc', 'Y_tlai']
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
max_length_1d = df[list_columns_1d].map(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).max().max()


for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

cols_2d = x_list_columns_2d + y_list_columns_2d
for col in cols_2d:
    df[col] = df[col].apply(lambda x: x[:, :10] if isinstance(x, np.ndarray) and x.shape[1] >= 10 else x)

max_rows_2d = df[list_columns_2d].map(lambda x: x.shape[0] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()
max_cols_2d = df[list_columns_2d].map(lambda x: x.shape[1] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()


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



time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']

static_columns = [
    col for col in df.columns 
    if col not in time_series_columns 
    and not col.startswith('Y_') 
    and col not in x_list_columns_2d 
    and col not in x_list_columns_1d
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
scaler_time_series = MinMaxScaler()
time_series_data = time_series_data.reshape(-1, len(time_series_columns)) 
time_series_data = scaler_time_series.fit_transform(time_series_data)
time_series_data = time_series_data.reshape(-1, 240, len(time_series_columns)) 

scaler_static = MinMaxScaler()
static_data = scaler_static.fit_transform(df[static_columns].values)

scaler_target = MinMaxScaler()
target_data = scaler_target.fit_transform(df[target_columns].values)
print('Normalize the static and target-end')
print('Convert data to PyTorch tensors')
# Convert data to PyTorch tensors
time_series_data = torch.tensor(time_series_data, dtype=torch.float32).to(device)
static_data = torch.tensor(static_data, dtype=torch.float32).to(device)
target_data = torch.tensor(target_data, dtype=torch.float32).to(device)




scalers_1d = {}  
for col in list_columns_1d:
    scaler = MinMaxScaler()
    
    col_data = np.vstack(df[col].values)
    
    scaled_data = scaler.fit_transform(col_data)
    
    scalers_1d[col] = scaler

    df[col] = list(scaled_data)

scalers_2d = {}

for col in list_columns_2d:
    reshaped_data = np.vstack(df[col].values).reshape(-1, 1)
    scaler = MinMaxScaler()
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

predictions_1d_dict = {} 

for idx, col in enumerate(x_list_columns_1d):
    y_col = 'Y_' + col 

    if col in scalers_1d: 
        predictions_1d_dict[y_col] = scalers_1d[y_col].inverse_transform(predictions_vector[:, idx, :])



predictions_1d_df = pd.DataFrame({col: predictions_1d_dict[col].tolist() for col in predictions_1d_dict})


predictions_2d_dict = {} 


print(f" Available scalers_2d keys: {list(scalers_2d.keys())}")

for idx, col in enumerate(x_list_columns_2d): 
    y_col = 'Y_' + col  

    if col in scalers_2d:  
        scaler = scalers_2d[y_col]  

        pred_reshaped = predictions_matrix[:, idx, :, :].reshape(-1, 1)
        predictions_2d_dict[y_col] = scaler.inverse_transform(pred_reshaped).reshape(-1, max_rows_2d, max_cols_2d)


predictions_2d_df = pd.DataFrame({col: predictions_2d_dict[col].tolist() for col in predictions_2d_dict})
print("!!!!!!!!!!data done!")
####################################################################################################################

input_files = sorted(glob.glob('./dataset/1_training_data_batch_*.pkl'))
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
# cols_to_drop = [
#     'OCCLUDED_P', 'SECONDARY_P', 'LABILE_P', 'APATITE_P'
# ]
# df = df.drop(columns=cols_to_drop, errors='ignore')



for col in x_list_columns_1d + y_list_columns_1d:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x[1:17] if isinstance(x, list) else x)

for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

max_rows_2d = df[list_columns_2d].map(lambda x: x.shape[0] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()
max_cols_2d = df[list_columns_2d].map(lambda x: x.shape[1] if isinstance(x, np.ndarray) and x.ndim == 2 else 0).max().max()

for col in list_columns_1d:
    df[col] = df[col].apply(lambda x: np.pad(np.array(x), (0, max_length_1d - len(x)), mode='constant') if isinstance(x, (list, np.ndarray)) else np.zeros(max_length_1d))

for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.pad(x, ((0, max_rows_2d - x.shape[0]), (0, max_cols_2d - x.shape[1])), mode='constant') if isinstance(x, np.ndarray) and x.ndim == 2 else np.zeros((max_rows_2d, max_cols_2d)))



def plot_comparison_map(df, predictions, variable, title,save_path_template):

    min_value = min(df[variable].min(), predictions[variable].min())
    max_value = max(df[variable].max(), predictions[variable].max())

    df = df.reset_index(drop=True)
    df_filtered = df.iloc[:len(predictions)] 
    difference = predictions[variable] - df_filtered[variable]

    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))

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
    cbar2 = plt.colorbar(scatter2, label=f"{variable} (ELM simulation result - Original Scale)")
    cbar2.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar2.ax.text(1.05, -0.02, f"Min: {min_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - ELM simulation result", fontsize=16)
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
def get_latitude_group(lat):

    abs_lat = abs(lat)
    if abs_lat < 23:
        return "Tropical"
    elif abs_lat < 40:
        return "Subtropical"
    elif abs_lat < 66:
        return "Temperate"
    else:
        return "Frigid"

def plot_difference_histogram_by_latitude(df, predictions, variable, save_path_template):


    df = df.reset_index(drop=True)
    n = len(predictions)
    df_filtered = df.iloc[:n]

    differences = predictions[variable] - df_filtered[variable]

    lat_groups = df_filtered['Latitude'].apply(get_latitude_group)

    diff_df = pd.DataFrame({
        'difference': differences,
        'lat_group': lat_groups
    })

    group_ranges = {
        "Tropical": "(<23°)",
        "Subtropical": "(23°-40°)",
        "Temperate": "(40°-66°)",
        "Frigid": "(>=66°)"
    }
    
    groups = ["Tropical", "Subtropical", "Temperate", "Frigid"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, group in enumerate(groups):
        group_data = diff_df[diff_df['lat_group'] == group]['difference']
        axes[i].hist(group_data, bins=30, color='blue', alpha=0.7)

        group_title = f"{group} {group_ranges[group]} (n={len(group_data)})"
        axes[i].set_title(group_title, fontsize=12)
        axes[i].set_xlabel("Prediction Difference (Prediction - ELM)")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True)
    
    plt.suptitle(f"Histogram of Prediction Differences for {variable} by Latitude Groups", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    

    save_path = save_path_template.format(variable=variable)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Histogram saved to: {save_path}")

# Create output directory if it doesn't exist
os.makedirs("./1_plot_5_gridcell", exist_ok=True)

save_path_1 = './1_plot_5_gridcell/4_{variable}_comparison_map_with_difference.png'
for variable in target_columns:
    print(f"plot {variable}")
    plot_comparison_map(df, predictions_df, variable, title=f"{variable} Comparison",save_path_template=save_path_1)


save_path_hist = './1_plot_5_gridcell/{variable}_difference_histogram.png'
for variable in target_columns:
    print(f"{variable} ")
    plot_difference_histogram_by_latitude(df, predictions_df, variable, save_path_template=save_path_hist)


######################1d

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
    cbar2 = plt.colorbar(scatter2, label=f"{variable} (ELM simulation result - Original Scale)")
    cbar2.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')
    cbar2.ax.text(1.05, -0.02, f"Min: {min_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=12, color='black')

    plt.title(f"{title} - ELM simulation result", fontsize=16)
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

save_path_2 = './2_plot_3_pft/4_{variable}_{i}_comparison_map_with_difference.png'
for variable in y_list_columns_1d:
    print(f"plot {variable}...")
    y_pred = np.vstack(predictions_1d_df[variable].values)

    num_features = y_pred.shape[1] 
    for i in range(num_features):
        print(f"  - Processing the {i}th feature dimension of {variable}")

        p2_plot_comparison_map(df, predictions_1d_df, variable, title=f"{variable}_{i} Comparison",save_path_template=save_path_2,i=i)




def get_latitude_group(lat):

    abs_lat = abs(lat)
    if abs_lat < 23:
        return "Tropical"
    elif abs_lat < 40:
        return "Subtropical"
    elif abs_lat < 66:
        return "Temperate"
    else:
        return "Frigid"

def p2_plot_difference_histogram(df, predictions, variable, title, save_path_template, i):

    df_values = np.vstack(df[variable].values)[:, i]            
    predictions_values = np.vstack(predictions[variable].values)[:, i]  
    differences = predictions_values - df_values

    latitudes = df['Latitude'].values
    hist_df = pd.DataFrame({
        'difference': differences,
        'Latitude': latitudes
    })

    hist_df['lat_group'] = hist_df['Latitude'].apply(get_latitude_group)


    group_ranges = {
        "Tropical": "(<23°)",
        "Subtropical": "(23°-40°)",
        "Temperate": "(40°-66°)",
        "Frigid": "(>=66°)"
    }
    groups = ["Tropical", "Subtropical", "Temperate", "Frigid"]


    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for j, group in enumerate(groups):
        group_data = hist_df[hist_df['lat_group'] == group]['difference']
        axes[j].hist(group_data, bins=30, color='blue', alpha=0.7)
        axes[j].set_title(f"{group} {group_ranges[group]} (n={len(group_data)})", fontsize=12)
        axes[j].set_xlabel("Prediction Difference (Prediction - ELM)")
        axes[j].set_ylabel("Frequency")
        axes[j].grid(True)

    plt.suptitle(f"Histogram of {variable} (dim {i}) Difference (Prediction - ELM) by Latitude Groups", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = save_path_template.format(variable=variable, i=i)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Histogram saved to: {save_path}")


save_path_hist = './2_plot_3_pft/{variable}_{i}_difference_histogram.png'
for variable in y_list_columns_1d:
    y_pred = np.vstack(predictions_1d_df[variable].values)
    num_features = y_pred.shape[1] 
    for i in range(num_features):
        print(f"Processing {variable} dimension {i} histogram...")
        p2_plot_difference_histogram(df, predictions_1d_df, variable,
                                     title=f"{variable}_{i} Histogram",
                                     save_path_template=save_path_hist, i=i)

# ###################################2d


results_list = []

for var in y_list_columns_2d:
    print(f"\n===== variable {var} =====")

    pred_arr = np.array(predictions_2d_df[var].tolist())  

    pred_sel = pred_arr[:, 0, :]

    pred_sum_per_layer = pred_sel.sum(axis=0)

    pred_total_sum = pred_sel.sum()
    

    gt_arr = np.array(df[var].tolist()) 
    gt_sel = gt_arr[:, 0, :]               
    gt_sum_per_layer = gt_sel.sum(axis=0)
    gt_total_sum = gt_sel.sum()

    

    for layer_index in range(pred_sum_per_layer.shape[0]):
        results_list.append({
            "variable": var,
            "layer": layer_index,
            "pred_sum": pred_sum_per_layer[layer_index],
            "gt_sum": gt_sum_per_layer[layer_index]
        })

    results_list.append({
        "variable": var,
        "layer": "total",
        "pred_sum": pred_total_sum,
        "gt_sum": gt_total_sum
    })


results_df = pd.DataFrame(results_list)


results_csv_path = "y_list_columns_2d_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\n results saved {results_csv_path}")



def p3_plot_comparison_map(df, predictions, variable, title_prefix, save_path_template, layer):

    df_values =  np.array(df[variable].tolist())[:, 0, layer]
    predictions_values = np.array(predictions[variable].tolist())[:, 0, layer]  


    min_value = min(df_values.min(), predictions_values.min())
    max_value = max(df_values.max(), predictions_values.max())


    difference = predictions_values - df_values
    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))
    processed_title_component = str(title_prefix)
    if processed_title_component.startswith("Y_"):
        processed_title_component = processed_title_component[2:] 


    if processed_title_component:  
        processed_title_component = processed_title_component[0].upper() + processed_title_component[1:]



    fig_width = 12
    fig_height = 5  

    plt.figure(figsize=(fig_width, fig_height))
    scatter1 = plt.scatter(
        df['Longitude'], df['Latitude'], c=predictions_values, cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    cbar1 = plt.colorbar(scatter1)
    cbar1.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar1.ax.transAxes, ha='center', fontsize=15, color='black')
    cbar1.ax.text(1.05, -0.05, f"Min: {min_value:.2f}", transform=cbar1.ax.transAxes, ha='center', fontsize=15, color='black')
    plt.title(f"{processed_title_component} - Prediction", fontsize=20)
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    plt.grid(True)
    plt.tight_layout()  
    save_path_pred = save_path_template.format(variable=variable, layer=layer, type='prediction')
    os.makedirs(os.path.dirname(save_path_pred), exist_ok=True)
    plt.savefig(save_path_pred, dpi=300)
    plt.close()
    print(f"   Prediction map saved to: {save_path_pred}")

    plt.figure(figsize=(fig_width, fig_height))
    scatter2 = plt.scatter(
        df['Longitude'], df['Latitude'], c=df_values, cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    cbar2 = plt.colorbar(scatter2)
    cbar2.ax.text(1.05, 1.02, f"Max: {max_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=15, color='black')
    cbar2.ax.text(1.05, -0.05, f"Min: {min_value:.2f}", transform=cbar2.ax.transAxes, ha='center', fontsize=15, color='black')
    plt.title(f"{processed_title_component} - ELM simulation result", fontsize=20)
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    plt.grid(True)
    plt.tight_layout() 
    save_path_true = save_path_template.format(variable=variable, layer=layer, type='ground_truth')
    os.makedirs(os.path.dirname(save_path_true), exist_ok=True)
    plt.savefig(save_path_true, dpi=300)
    plt.close()
    print(f"   Ground truth map saved to: {save_path_true}")

    plt.figure(figsize=(fig_width, fig_height))
    scatter3 = plt.scatter(
        df['Longitude'], df['Latitude'], c=difference, cmap='RdBu', s=10, vmin=diff_min, vmax=diff_max
    )
    cbar3 = plt.colorbar(scatter3)
    cbar3.ax.text(1.05, 1.02, f"Max: {difference.max():.2f}", transform=cbar3.ax.transAxes, ha='center', fontsize=15, color='black')
    cbar3.ax.text(1.05, -0.05, f"Min: {difference.min():.2f}", transform=cbar3.ax.transAxes, ha='center', fontsize=15, color='black')
    plt.title(f"{processed_title_component} - Difference", fontsize=20)
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    save_path_diff = save_path_template.format(variable=variable, layer=layer, type='difference')
    os.makedirs(os.path.dirname(save_path_diff), exist_ok=True)
    plt.savefig(save_path_diff, dpi=300)
    plt.close()
    print(f"Difference map saved to: {save_path_diff}")


save_path_3 = './6_plot_separate_0_9_depth/4_{variable}_layer_{layer}_{type}.pdf'


selected_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

for variable in y_list_columns_2d:
    print(f"\n Processing variable: {variable}")

    y_pred_all = np.array(predictions_2d_df[variable].tolist())  

    num_samples, num_columns, num_layers = y_pred_all.shape
    print(f" Shape of variable {variable}: {num_samples} samples, {num_columns} columns, {num_layers} layers")
    y_pred = y_pred_all[:, 0, :] 


    for layer in selected_layers:
        if layer >= num_layers:
            print(f"{variable}[Layer {layer}] is out of range, skipping!")
            continue

        print(f"\n Processing {variable} - Depth Layer {layer}")

        y_pred_layer = y_pred[:, layer]

        p3_plot_comparison_map(df, predictions_2d_df, variable, title_prefix=f"{variable} - Layer {layer}",
                                    save_path_template=save_path_3, layer=layer)





#####################################



def get_latitude_group(lat):
    abs_lat = abs(lat)
    if abs_lat < 23:
        return "Tropical"
    elif abs_lat < 40:
        return "Subtropical"
    elif abs_lat < 66:
        return "Temperate"
    else:
        return "Frigid"
def p3_plot_difference_histogram(df, predictions, variable, save_path_template, layer, csv_save_path_template):
    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    df_values = np.array(df[variable].tolist())[:, 0, layer]
    predictions_values = np.array(predictions[variable].tolist())[:, 0, layer]
    differences = predictions_values - df_values

    latitudes = df['Latitude'].values
    hist_df = pd.DataFrame({
        'Latitude': latitudes,
        'Difference': differences
    })
    hist_df['LatGroup'] = hist_df['Latitude'].apply(get_latitude_group)

    group_ranges = {
        "Tropical": "(<23°)",
        "Subtropical": "(23°-40°)",
        "Temperate": "(40°-66°)",
        "Frigid": "(>=66°)"
    }
    groups = ["Tropical", "Subtropical", "Temperate", "Frigid"]

    for group in groups:
        group_data = hist_df[hist_df['LatGroup'] == group]


        csv_path = csv_save_path_template.format(variable=variable, layer=layer, group=group)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        group_data.to_csv(csv_path, index=False)
        print(f"CSV data saved to: {csv_path}")

        plt.figure(figsize=(6, 5))
        plt.hist(group_data['Difference'], bins=30, color='blue', alpha=0.7)
        plt.title(f"{group} {group_ranges[group]} (n={len(group_data)})", fontsize=20)
        plt.xlabel("Prediction Difference")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()

        save_path = save_path_template.format(variable=variable, layer=layer, group=group)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Histogram saved to: {save_path}")

save_path_hist_2d = './histograms/{variable}_layer_{layer}_{group}_difference_histogram_P.pdf'
save_path_csv_2d = './histogram_csvs/{variable}_layer_{layer}_{group}_difference_data_P.csv'
selected_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for variable in y_list_columns_2d:
    print(f"\n Processing variable: {variable}")
    y_pred_all = np.array(predictions_2d_df[variable].tolist())
    num_samples, num_columns, num_layers = y_pred_all.shape
    print(f"Variable {variable} shape: {num_samples} samples, {num_columns} columns, {num_layers} layers")

    for layer in selected_layers:
        if layer >= num_layers:
            print(f" {variable} [Layer {layer}] is out of range, skipping!")
            continue
        print(f"  Plotting histograms for {variable} - Layer {layer}")
        p3_plot_difference_histogram(
            df,
            predictions_2d_df,
            variable,
            save_path_template=save_path_hist_2d,
            csv_save_path_template=save_path_csv_2d,
            layer=layer
        )

print("done")
