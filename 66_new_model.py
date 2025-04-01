import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import glob
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_files = sorted(glob.glob('./1_global_data/1_training_data_batch_*.pkl'))

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

import numpy as np
import torch

x_list_columns_2d = [
    'soil3c_vr', 'soil4c_vr', 'cwdc_vr'
]

y_list_columns_2d = [
    'Y_soil3c_vr', 'Y_soil4c_vr', 'Y_cwdc_vr'
]

x_list_columns_1d = [
    'deadcrootc', 'deadstemc', 'tlai'
]

y_list_columns_1d = [
    'Y_deadcrootc', 'Y_deadstemc', 'Y_tlai'
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

print("Columns truncated to elements 2 through 17.")

max_length_1d = df[list_columns_1d].applymap(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).max().max()

for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

cols_2d = x_list_columns_2d + y_list_columns_2d
for col in cols_2d:
    df[col] = df[col].apply(lambda x: x[:, :10] if isinstance(x, np.ndarray) and x.shape[1] >= 10 else x)

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

print("All list variables successfully converted to PyTorch tensor!")

print("2D input feature shapes:")
for col in x_list_columns_2d:
    tensor = list_data_x_2d[col]
    print(f"{col} shape: {tensor.shape}")

print("2D target feature shapes:")
for col in y_list_columns_2d:
    tensor = list_data_y_2d[col]
    print(f"{col} shape: {tensor.shape}")

print("All list variables successfully converted to NumPy arrays and updated in DataFrame!")
print(df.info())

df = df.dropna(subset=['H2OSOI_10CM'])
print(f"Filtered dataset: {len(df)} samples remaining after removing NaN H2OSOI values.")

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
    print('processisng: ',col)
    df[col] = df[col].apply(
        lambda x: np.pad(x[:1476], (0, max(0, 1476 - len(x))), 'constant') if isinstance(x, list) else np.zeros(1476, dtype=np.float32)
    )

print("start shuffle")
df = shuffle(df, random_state=42).reset_index(drop=True)
print("end shuffle")
print('onvert the time-series data into a 3D numpy-start')
time_series_data = np.stack([np.column_stack(df[col]) for col in time_series_columns], axis=-1)

print('onvert the time-series data into a 3D numpy-end')
print('Normalize the time-series-start')
scaler_time_series = MinMaxScaler(feature_range=(0, 1))
time_series_data = time_series_data.reshape(-1, len(time_series_columns))
time_series_data = scaler_time_series.fit_transform(time_series_data)
time_series_data = time_series_data.reshape(-1, 1476, len(time_series_columns))

print('Normalize the time-series-end')
print('Normalize the static and target-start')
print("Static columns:", static_columns)
print(df[static_columns].head())
for col in static_columns:
    if df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
        print(f"Column '{col}' contains sequences or arrays.")

scaler_static = MinMaxScaler(feature_range=(0, 1))
static_data = scaler_static.fit_transform(df[static_columns].values)

scaler_target = MinMaxScaler(feature_range=(0, 1))
target_data = scaler_target.fit_transform(df[target_columns].values)
print('Normalize the static and target-end')
print('Convert data to PyTorch tensors')

if 'Y_GPP' in target_columns:
    gpp_idx = target_columns.index('Y_GPP')
    gpp_values = target_data[:, gpp_idx]
    print(f"Normalized GPP min: {gpp_values.min():.6f}, max: {gpp_values.max():.6f}")
else:
    print("GPP column not found in target_columns")

time_series_data = torch.tensor(time_series_data, dtype=torch.float32)

static_data = torch.tensor(static_data, dtype=torch.float32)
target_data = torch.tensor(target_data, dtype=torch.float32)

print('Split data into train and test sets')
train_size = int(0.8 * len(df))
test_size = len(df) - train_size

groundtruth_y_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_2d}
groundtruth_y_list_columns_1d_tensors = {
    col: torch.stack([torch.tensor(np.array(x, dtype=np.float32)) for x in df[col].values])
    for col in y_list_columns_1d
}

groundtruth_test_y_list_columns_2d = {col: tensor[train_size:].to(device) for col, tensor in groundtruth_y_list_columns_2d_tensors.items()}
groundtruth_test_y_list_columns_1d = {col: tensor[train_size:].to(device) for col, tensor in groundtruth_y_list_columns_1d_tensors.items()}
for col in groundtruth_test_y_list_columns_1d:
    tensor_data = groundtruth_test_y_list_columns_1d[col]

    if isinstance(tensor_data, torch.Tensor):
        tensor_data = tensor_data.cpu().numpy()

    min_val = np.min(tensor_data)
    max_val = np.max(tensor_data)

    print(f"- {col}: min = {min_val:.12f}, max = {max_val:.12f}")

print("Test set 'test_y_list_columns_1d' min/max values check complete!")

print("\nStandardizing 1D features (StandardScaler)...")

print("\nPre-normalization 'list_columns_1d' data ranges:")
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

print("All 1D variables normalized!")

print("\nStandardScaler normalization complete!")

print("\nPost-normalization 'list_columns_1d' data ranges:")
for col in list_columns_1d:
    mean_val = df[col].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else 0).mean()
    std_val = df[col].apply(lambda x: np.std(x) if isinstance(x, (list, np.ndarray)) else 0).mean()
    print(f"  - {col}: mean = {mean_val:.4f}, std = {std_val:.4f}")

print("All 1D features standardized!")

print("1D features Min-Max normalization complete!")

scalers_2d = {}

print("\nStandardizing 2D features...")
print("\nPre-normalization 'list_columns_2d' data ranges:")
for col in list_columns_2d:
    min_val = df[col].apply(lambda x: np.min(x) if isinstance(x, (list, np.ndarray)) else 0).min()
    max_val = df[col].apply(lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) else 0).max()
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")
for col in list_columns_2d:
    print(f"\nProcessing 2D feature column: {col}")

    print(f"  - Pre-normalization `df[{col}].values` shapes: {[x.shape for x in df[col].values[:5]]} (first 5 examples)")

    try:
        reshaped_data = np.vstack(df[col].values).reshape(-1, 1)
    except Exception as e:
        print(f"Failed vstack, error: {e}")
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
        print(f"Normalization successful, df[{col}] updated")
    except Exception as e:
        print(f"Assignment back to df[{col}] failed, error: {e}")

print("\nAll 2D features successfully normalized!")

print("\nPost-normalization data range check:")
for col in list_columns_2d:
    min_val = df[col].apply(lambda x: np.min(x)).min()
    max_val = df[col].apply(lambda x: np.max(x)).max()
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")
for col in list_columns_1d:
    min_val = df[col].apply(lambda x: np.min(x)).min()
    max_val = df[col].apply(lambda x: np.max(x)).max()
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")

print("1D and 2D feature standardization complete!")

x_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_2d}
x_list_columns_1d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_1d}
y_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_2d}
y_list_columns_1d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_1d}

for col, tensor in x_list_columns_2d_tensors.items():
    print(f"{col} - min: {tensor.min().item()}, max: {tensor.max().item()}")

for col, tensor in x_list_columns_1d_tensors.items():
    print(f"{col} - min: {tensor.min().item()}, max: {tensor.max().item()}")

train_x_list_columns_2d = {col: tensor[:train_size].to(device) for col, tensor in x_list_columns_2d_tensors.items()}
train_x_list_columns_1d = {col: tensor[:train_size].to(device) for col, tensor in x_list_columns_1d_tensors.items()}
train_y_list_columns_2d = {col: tensor[:train_size].to(device) for col, tensor in y_list_columns_2d_tensors.items()}
train_y_list_columns_1d = {col: tensor[:train_size].to(device) for col, tensor in y_list_columns_1d_tensors.items()}
print("Checking data types of x_list_columns_2d_tensors:")
for col, tensor in x_list_columns_2d_tensors.items():
    print(f"{col} type: {type(tensor)}")

print("\nChecking data types of train_x_list_columns_2d:")
for col, tensor in train_x_list_columns_2d.items():
    print(f"{col} type: {type(tensor)}")

train_time_series = time_series_data[:train_size]
train_static = static_data[:train_size]
train_target = target_data[:train_size]

test_x_list_columns_2d = {col: tensor[train_size:].to(device) for col, tensor in x_list_columns_2d_tensors.items()}
test_x_list_columns_1d = {col: tensor[train_size:].to(device) for col, tensor in x_list_columns_1d_tensors.items()}
test_y_list_columns_2d = {col: tensor[train_size:].to(device) for col, tensor in y_list_columns_2d_tensors.items()}
test_y_list_columns_1d = {col: tensor[train_size:].to(device) for col, tensor in y_list_columns_1d_tensors.items()}
test_time_series = time_series_data[train_size:]
test_static = static_data[train_size:]
test_target = target_data[train_size:]
print(f"test_target Min: {test_target.min().item()}, Max: {test_target.max().item()}")
print("\nTest set 'test_y_list_columns_1d' pre-normalization data ranges:")

train_time_series = train_time_series.to(device)
train_static = train_static.to(device)
train_target = train_target.to(device)
print('data preparation done!')
print('data preparation done!')

print("Checking time_series_data for NaN or Inf...")
print(torch.isnan(time_series_data).any())
print(torch.isinf(time_series_data).any())

print("Checking static_data for NaN or Inf...")
print(torch.isnan(static_data).any())
print(torch.isinf(static_data).any())

print("Checking target_data for NaN or Inf...")
print(torch.isnan(target_data).any())
print(torch.isinf(target_data).any())
nan_counts = pd.DataFrame(df[static_columns].isna().sum(), columns=["NaN Count"])
print(nan_counts[nan_counts["NaN Count"] > 0])

class CombinedModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, num_static_features,
                 num_1d_features, num_2d_features, fc_hidden_size, batch_size):
        super(CombinedModel, self).__init__()

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, batch_first=True)

        self.fc_static = nn.Linear(num_static_features, fc_hidden_size)

        self.fc_1d = nn.Linear(num_1d_features, fc_hidden_size)

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, fc_hidden_size)
        )

        self.feature_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=160, nhead=4, batch_first=True),
            num_layers=2
        )

        self.shared_fc = nn.Linear(160, 128)
        self.fc_scalar_branch = nn.Linear(160, 128)
        self.fc_vector_branch = nn.Linear(160, 128)
        self.fc_matrix_branch = nn.Linear(160, 128)

        self.fc_scalar = nn.Linear(128, 9)
        self.fc_vector = nn.Linear(128, 3 * 16)
        self.fc_matrix = nn.Linear(128, 3 * (18 * 10))

        self.log_sigma_scalar = nn.Parameter(torch.zeros(1))
        self.log_sigma_vector = nn.Parameter(torch.zeros(1))
        self.log_sigma_matrix = nn.Parameter(torch.zeros(1))

        self.batch_size = batch_size

    def forward(self, time_series_data, static_data, list_1d_data, list_2d_data):
        lstm_out, _ = self.lstm(time_series_data)
        lstm_out = lstm_out[:, -1, :]

        static_out = torch.relu(self.fc_static(static_data))

        list_1d_out = torch.relu(self.fc_1d(list_1d_data))

        list_2d_out = self.conv2d(list_2d_data)

        combined = torch.cat((lstm_out, static_out, list_1d_out, list_2d_out), dim=1)
        combined = combined.unsqueeze(1)
        combined = self.feature_fusion(combined)
        combined = combined.squeeze(1)

        scalar_features = torch.relu(self.fc_scalar_branch(combined))
        vector_features = torch.relu(self.fc_vector_branch(combined))
        matrix_features = torch.relu(self.fc_matrix_branch(combined))
        batch_size_dynamic = matrix_features.shape[0]
        scalar_output = self.fc_scalar(scalar_features)
        vector_output = F.softplus(self.fc_vector(vector_features)).view(batch_size_dynamic, 3, 16)
        matrix_output = F.softplus(self.fc_matrix(matrix_features)).view(batch_size_dynamic, 3, 18, 10)

        return scalar_output, vector_output, matrix_output

    def compute_loss(self, scalar_pred, vector_pred, matrix_pred, target_scalar, target_vector, target_matrix):
        loss_scalar = criterion_scalar(scalar_pred, target_scalar)
        loss_vector = criterion_vector(vector_pred, target_vector)
        loss_matrix = criterion_matrix(matrix_pred, target_matrix)

        log_sigma_scalar = torch.clamp(self.log_sigma_scalar, -2.05, 0)
        log_sigma_vector = torch.clamp(self.log_sigma_vector, -2.05, 0)
        log_sigma_matrix = torch.clamp(self.log_sigma_matrix, -2.05, 0)

        sigma_scalar = torch.exp(log_sigma_scalar)
        sigma_vector = torch.exp(log_sigma_vector)
        sigma_matrix = torch.exp(log_sigma_matrix)

        weighted_loss_scalar = (1 / (2 * sigma_scalar**2)) * loss_scalar + 0.5 * torch.log(1 + sigma_scalar**2)
        weighted_loss_vector = (1 / (2 * sigma_vector**2)) * loss_vector + 0.5 * torch.log(1 + sigma_vector**2)
        weighted_loss_matrix = (1 / (2 * sigma_matrix**2)) * loss_matrix + 0.5 * torch.log(1 + sigma_matrix**2)

        penalty_vector = torch.mean(torch.relu(-vector_pred))
        penalty_matrix = torch.mean(torch.relu(-matrix_pred))

        total_loss = weighted_loss_scalar + weighted_loss_vector + weighted_loss_matrix + 0.1 * (penalty_vector + penalty_matrix)
        return total_loss

lstm_input_size = len(time_series_columns)
lstm_hidden_size = 64
num_static_features = static_data.shape[1]
num_1d_features = 48
num_2d_features = 18 * 10
fc_hidden_size = 32
batch_size = 16

model = CombinedModel(
    lstm_input_size=lstm_input_size,
    lstm_hidden_size=lstm_hidden_size,
    num_static_features=num_static_features,
    num_1d_features=num_1d_features,
    num_2d_features=num_2d_features,
    fc_hidden_size=fc_hidden_size,
    batch_size=batch_size
).to(device)

criterion_scalar = nn.MSELoss()
criterion_vector = nn.MSELoss()
criterion_matrix = nn.MSELoss()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15
batch_size = 16
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for i in range(0, train_size, batch_size):
        end_idx = min(i + batch_size, train_size)
        time_series_batch = train_time_series[i:end_idx]
        static_batch = train_static[i:end_idx]
        target_batch = train_target[i:end_idx]

        columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in train_x_list_columns_1d.items()}
        columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in train_x_list_columns_2d.items()}
        y_columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in train_y_list_columns_1d.items()}
        y_columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in train_y_list_columns_2d.items()}

        columns_1d_batch_tensor = torch.cat([tensor for tensor in columns_1d_batch.values()], dim=1)
        columns_2d_batch_tensor = torch.cat([tensor.unsqueeze(1) for tensor in columns_2d_batch.values()], dim=1)
        y_columns_1d_batch_tensor = torch.cat([tensor for tensor in y_columns_1d_batch.values()], dim=1)
        y_columns_2d_batch_tensor = torch.cat([tensor.unsqueeze(1) for tensor in y_columns_2d_batch.values()], dim=1)

        optimizer.zero_grad()
        scalar_pred, vector_pred, matrix_pred  = model(time_series_batch, static_batch, columns_1d_batch_tensor, columns_2d_batch_tensor)
        
        batch_size_y = y_columns_1d_batch_tensor.shape[0]
        y_columns_1d_batch_tensor = y_columns_1d_batch_tensor.view(batch_size_y, 3, 16)

        loss = model.compute_loss(scalar_pred, vector_pred, matrix_pred, target_batch, y_columns_1d_batch_tensor, y_columns_2d_batch_tensor)

        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / (train_size / batch_size)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for j in range(0, test_size, batch_size):
            end_idx = min(j + batch_size, test_size)
            time_series_batch = test_time_series[j:end_idx].to(device)
            static_batch = test_static[j:end_idx].to(device)
            target_batch = test_target[j:end_idx].to(device)
            
            columns_1d_batch = {col: tensor[j:end_idx] for col, tensor in train_x_list_columns_1d.items()}
            columns_2d_batch = {col: tensor[j:end_idx] for col, tensor in train_x_list_columns_2d.items()}
            y_columns_1d_batch = {col: tensor[j:end_idx] for col, tensor in train_y_list_columns_1d.items()}
            y_columns_2d_batch = {col: tensor[j:end_idx] for col, tensor in train_y_list_columns_2d.items()}

            columns_1d_batch_tensor = torch.cat([tensor for tensor in columns_1d_batch.values()], dim=1)
            columns_2d_batch_tensor = torch.cat([tensor.unsqueeze(1) for tensor in columns_2d_batch.values()], dim=1)
            y_columns_1d_batch_tensor = torch.cat([tensor for tensor in y_columns_1d_batch.values()], dim=1)
            y_columns_2d_batch_tensor = torch.cat([tensor.unsqueeze(1) for tensor in y_columns_2d_batch.values()], dim=1)

            scalar_pred, vector_pred, matrix_pred = model(time_series_batch, static_batch, columns_1d_batch_tensor, columns_2d_batch_tensor)

            batch_size_y = y_columns_1d_batch_tensor.shape[0]
            y_columns_1d_batch_tensor = y_columns_1d_batch_tensor.view(batch_size_y, 3, 16)

            loss = model.compute_loss(scalar_pred, vector_pred, matrix_pred, target_batch, y_columns_1d_batch_tensor, y_columns_2d_batch_tensor)
            val_loss += loss.item()

        val_loss /= (test_size / batch_size)
        val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Scalar Weight: {(1 / (2 * torch.exp(model.log_sigma_scalar)**2)).item():.4f}")
    print(f"Vector Weight: {(1 / (2 * torch.exp(model.log_sigma_vector)**2)).item():.4f}")
    print(f"Matrix Weight: {(1 / (2 * torch.exp(model.log_sigma_matrix)**2)).item():.4f}")

losses_df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Train Loss': train_losses,
    'Validation Loss': val_losses
})
losses_df.to_csv("training_validation_losses2.csv", index=False)
print("Training and validation losses saved as 'training_validation_losses2.csv'")

model.eval()
predictions_scalar = []
predictions_vector = []
predictions_matrix = []

with torch.no_grad():
    for i in range(0, test_size, batch_size):
        end_idx = min(i + batch_size, test_size)
        time_series_batch = test_time_series[i:end_idx].to(device)
        static_batch = test_static[i:end_idx].to(device)
        columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in test_x_list_columns_1d.items()}
        columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in test_x_list_columns_2d.items()}
        y_columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in test_y_list_columns_1d.items()}
        y_columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in test_y_list_columns_2d.items()}
        
        columns_1d_batch_tensor = torch.cat([tensor for tensor in columns_1d_batch.values()], dim=1)
        columns_2d_batch_tensor = torch.cat([tensor.unsqueeze(1) for tensor in columns_2d_batch.values()], dim=1)  

        scalar_pred, vector_pred, matrix_pred = model(time_series_batch, static_batch, columns_1d_batch_tensor, columns_2d_batch_tensor)

        predictions_scalar.append(scalar_pred.cpu().numpy())
        predictions_vector.append(vector_pred.cpu().numpy())
        predictions_matrix.append(matrix_pred.cpu().numpy())

predictions_scalar = np.concatenate(predictions_scalar, axis=0)
predictions_vector = np.concatenate(predictions_vector, axis=0)
predictions_matrix = np.concatenate(predictions_matrix, axis=0)

mse_scalar = mean_squared_error(test_target.cpu().numpy(), predictions_scalar)
test_loss_scalar = np.sqrt(mse_scalar)

test_y_list_columns_1d_tensor = torch.cat([tensor for tensor in test_y_list_columns_1d.values()], dim=1).cpu().numpy()
test_y_list_columns_2d_tensor = torch.cat([tensor.unsqueeze(1) for tensor in test_y_list_columns_2d.values()], dim=1).cpu().numpy()

mse_vector = mean_squared_error(test_y_list_columns_1d_tensor.reshape(-1, 3 * 16), 
                                predictions_vector.reshape(-1, 3 * 16))
mse_matrix = mean_squared_error(test_y_list_columns_2d_tensor.reshape(-1, 3 * (18 * 10)), 
                                predictions_matrix.reshape(-1, 3 * (18 * 10)))

print(f"Test Loss (Scalar): {test_loss_scalar:.4f}")
print(f"Test Loss (Vector): {mse_vector:.4f}")
print(f"Test Loss (Matrix): {mse_matrix:.4f}")

predictions_scalar_np = scaler_target.inverse_transform(predictions_scalar)
ground_truth_scalar_np = scaler_target.inverse_transform(test_target.cpu().numpy())

predictions_df = pd.DataFrame(predictions_scalar_np, columns=target_columns)
ground_truth_df = pd.DataFrame(ground_truth_scalar_np, columns=target_columns)

predictions_df.to_csv("predictions_original_scale2.csv", index=False)
ground_truth_df.to_csv("ground_truth_original_scale2.csv", index=False)

print("Predictions saved as 'predictions_original_scale2.csv'")
print("Ground truth saved as 'ground_truth_original_scale2.csv'")

predictions_1d_dict = {}
ground_truth_1d_dict = {}

print("\nInverse transform 'y_list_columns_1d' predictions:")

print(f"Available keys in test_y_list_columns_1d: {list(test_y_list_columns_1d.keys())}")

for idx, col in enumerate(x_list_columns_1d):
    y_col = 'Y_' + col

    if col in scalers_1d:
        predictions_1d_dict[y_col] = scalers_1d[y_col].inverse_transform(predictions_vector[:, idx, :])
    else:
        print(f"No scaler found for {col}, skipping inverse transform")
for y_col in groundtruth_test_y_list_columns_1d:
    ground_truth_1d_dict[y_col] = groundtruth_test_y_list_columns_1d[y_col].cpu().numpy() if isinstance(groundtruth_test_y_list_columns_1d[y_col], torch.Tensor) else test_y_list_columns_1d[y_col]

print("\nSaving 'ground_truth' without inverse transform!")

predictions_1d_df = pd.DataFrame({col: predictions_1d_dict[col].tolist() for col in predictions_1d_dict})
ground_truth_1d_df = pd.DataFrame({col: ground_truth_1d_dict[col].tolist() for col in ground_truth_1d_dict})
print("\nGround Truth normalized data ranges:")
for y_col in ground_truth_1d_dict:
    groundtruth_min = np.min(ground_truth_1d_dict[y_col])
    groundtruth_max = np.max(ground_truth_1d_dict[y_col])
    print(f"- {y_col}: min = {groundtruth_min:.12f}, max = {groundtruth_max:.12f}")
print("1D ground truth min/max value check complete!")

predictions_1d_df.to_csv("predictions_1d_original_scale.csv", index=False)
ground_truth_1d_df.to_csv("ground_truth_1d_original_scale.csv", index=False)

print("Predictions (1D) saved as 'predictions_1d_original_scale.csv'")
print("Ground truth (1D) saved as 'ground_truth_1d_original_scale.csv'")

print("\nInverse transformed 'predictions_1d_dict' data ranges:")
for col in predictions_1d_dict:
    min_val = np.min(predictions_1d_dict[col])
    max_val = np.max(predictions_1d_dict[col])
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")

print("\nInverse transformed 'ground_truth_1d_dict' data ranges:")
for col in ground_truth_1d_dict:
    min_val = np.min(ground_truth_1d_dict[col])
    max_val = np.max(ground_truth_1d_dict[col])
    print(f"  - {col}: min = {min_val:.12f}, max = {max_val:.12f}")

print("1D feature inverse transform min/max check complete!")

predictions_2d_dict = {}
ground_truth_2d_dict = {}

print("\nInverse transform 'y_list_columns_2d' predictions:")

if not isinstance(scalers_2d, dict):
    raise TypeError("Error: 'scalers_2d' should be a 'dict', but is: ", type(scalers_2d))

print(f"Available scalers_2d keys: {list(scalers_2d.keys())}")

for idx, col in enumerate(x_list_columns_2d):
    y_col = 'Y_' + col

    if col in scalers_2d:
        scaler = scalers_2d[y_col]

        pred_reshaped = predictions_matrix[:, idx, :, :].reshape(-1, 1)
        predictions_2d_dict[y_col] = scaler.inverse_transform(pred_reshaped).reshape(-1, max_rows_2d, max_cols_2d)
    else:
        print(f"No scaler found for {col}, skipping inverse transform")
for y_col in groundtruth_test_y_list_columns_2d:
    ground_truth_2d_dict[y_col] = groundtruth_test_y_list_columns_2d[y_col].cpu().numpy() if isinstance(groundtruth_test_y_list_columns_2d[y_col], torch.Tensor) else test_y_list_columns_1d[y_col]

predictions_2d_df = pd.DataFrame({col: predictions_2d_dict[col].tolist() for col in predictions_2d_dict})
ground_truth_2d_df = pd.DataFrame({col: ground_truth_2d_dict[col].tolist() for col in ground_truth_2d_dict})

predictions_2d_df.to_csv("predictions_2d_original_scale.csv", index=False)
ground_truth_2d_df.to_csv("ground_truth_2d_original_scale.csv", index=False)

print("Predictions (2D) saved as 'predictions_2d_original_scale.csv'")
print("Ground truth (2D) saved as 'ground_truth_2d_original_scale.csv'")

print("All 2D features inverse transformed!")

print("All 2D features inverse transformed!")

print("\nInverse transformed 'predictions_2d_dict' data ranges:")
for col in predictions_2d_dict:
    min_val = np.min(predictions_2d_dict[col])
    max_val = np.max(predictions_2d_dict[col])
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")

print("\nInverse transformed 'ground_truth_2d_dict' data ranges:")
for col in ground_truth_2d_dict:
    min_val = np.min(ground_truth_2d_dict[col])
    max_val = np.max(ground_truth_2d_dict[col])
    print(f"  - {col}: min = {min_val:.4f}, max = {max_val:.4f}")

print("2D feature inverse transform min/max check complete!")

scripted_model = torch.jit.script(model)
scripted_model.save("LSTM_model_62.pt")
print("Model successfully scripted and saved!")

plt.figure(figsize=(10, 6))

plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', linewidth=2)

plt.xlabel('Epoch', fontsize=18, weight='bold')
plt.ylabel('Mean Squared Error (MSE) Loss', fontsize=18, weight='bold')
plt.title('Training and Validation Loss vs. Epoch', fontsize=20, weight='bold')
plt.legend(fontsize=18, loc='upper right', frameon=True)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('training_validation_loss.png', dpi=300)
