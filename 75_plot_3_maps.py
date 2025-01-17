import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler


# model = torch.jit.load("LSTM_model.pt")
model = torch.jit.load("LSTM_model_62.pt")
model.eval() 
model.to("cpu")  

input_files = sorted(glob.glob('/mnt/DATA/dawei/0_oak/1_global_data/1_training_data_batch_*.pkl'))
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


time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
static_columns = [col for col in df.columns if col not in time_series_columns and not col.startswith('Y_')]
target_columns = [col for col in df.columns if col.startswith('Y_')]

for col in time_series_columns:
    df[col] = df[col].apply(
        lambda x: np.pad(x[:1476], (0, max(0, 1476 - len(x))), 'constant') if isinstance(x, list) else np.zeros(1476)
    )

time_series_data = np.stack([np.column_stack(df[col]) for col in time_series_columns], axis=-1)

scaler_time_series = StandardScaler()
time_series_data = time_series_data.reshape(-1, len(time_series_columns))
time_series_data = scaler_time_series.fit_transform(time_series_data)
time_series_data = time_series_data.reshape(-1, 1476, len(time_series_columns))

scaler_static = StandardScaler()
static_data = scaler_static.fit_transform(df[static_columns].values)

scaler_target = StandardScaler()
target_data = scaler_target.fit_transform(df[target_columns].values)

time_series_data = torch.tensor(time_series_data, dtype=torch.float32)
static_data = torch.tensor(static_data, dtype=torch.float32)

batch_size = 16
predictions = []

with torch.no_grad():
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        time_series_batch = time_series_data[i:end_idx]
        static_batch = static_data[i:end_idx]
        predictions.append(model(time_series_batch, static_batch).numpy())

predictions = np.concatenate(predictions, axis=0)
predictions = pd.DataFrame(predictions, columns=target_columns)


predictions_original_scale = scaler_target.inverse_transform(predictions.values)
predictions_original_scale = pd.DataFrame(predictions_original_scale, columns=target_columns)



raw_data = df[['Longitude', 'Latitude']].copy()
raw_data['SOIL3C_GroundTruth'] = df['Y_SOIL3C']
raw_data['SOIL4C_GroundTruth'] = df['Y_SOIL4C']
print("Columns in df:", df.columns)
print("Columns in predictions:", predictions.columns)

raw_data['SOIL3C_Prediction'] = predictions_original_scale['Y_SOIL3C']
raw_data['SOIL4C_Prediction'] = predictions_original_scale['Y_SOIL4C']
raw_data['SOIL3C_Difference'] = predictions_original_scale['Y_SOIL3C'] - df['Y_SOIL3C']
raw_data['SOIL4C_Difference'] = predictions_original_scale['Y_SOIL4C'] - df['Y_SOIL4C']


csv_file_path = "SOIL3C_SOIL4C_raw_data.csv"
raw_data.to_csv(csv_file_path, index=False)
print(f"Raw data saved to {csv_file_path}")


def plot_comparison_map(df, predictions, variable, title):

    min_value = min(df[variable].min(), predictions[variable].min())
    max_value = max(df[variable].max(), predictions[variable].max())

    difference = predictions[variable] - df[variable]
    diff_min = -max(abs(difference.min()), abs(difference.max()))
    diff_max = max(abs(difference.min()), abs(difference.max()))

    plt.figure(figsize=(12, 15)) 


    plt.subplot(3, 1, 1)
    scatter1 = plt.scatter(
        df['Longitude'], df['Latitude'], c=predictions[variable], cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    plt.colorbar(scatter1, label=f"{variable} (Prediction - Original Scale)")
    plt.title(f"{title} - Prediction", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    scatter2 = plt.scatter(
        df['Longitude'], df['Latitude'], c=df[variable], cmap='viridis', s=10, vmin=min_value, vmax=max_value
    )
    plt.colorbar(scatter2, label=f"{variable} (Ground Truth - Original Scale)")
    plt.title(f"{title} - Ground Truth", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    scatter3 = plt.scatter(
        df['Longitude'], df['Latitude'], c=difference, cmap='RdBu', s=10, vmin=diff_min, vmax=diff_max
    )
    plt.colorbar(scatter3, label=f"{variable} Difference (Prediction - Ground Truth)")
    plt.title(f"{title} - Difference", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"2_{variable}_comparison_map_with_difference.png", dpi=300)  
    # plt.show()

for variable in target_columns:
    print(f"Plot {variable}...")
    plot_comparison_map(df, predictions_original_scale, variable, title=f"{variable} Comparison")
