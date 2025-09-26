import glob
from config import Config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import os

os.makedirs("./results", exist_ok=True)

input_files = sorted(glob.glob(Config.TRAIN_INPUT_GLOB))
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

cols_to_drop = Config.COLS_TO_DROP
df = df.drop(columns=cols_to_drop, errors='ignore')

# Remove samples with NaN values in specified columns if any exist
vars_to_scan = ['sminn_vr', 'smin_nh4_vr', 'Y_sminn_vr', 'Y_smin_nh4_vr']
all_unique_bad_indexes = set()
for col in vars_to_scan:
    if col in df.columns:
        bad_indexes = df.index[df[col].apply(lambda x: np.isnan(np.sum(x)))].tolist()
        if bad_indexes:
            all_unique_bad_indexes.update(bad_indexes)

if all_unique_bad_indexes:
    print(f"Dropping {len(all_unique_bad_indexes)} samples containing NaN values.")
    df.drop(index=list(all_unique_bad_indexes), inplace=True)
    df.reset_index(drop=True, inplace=True)

x_list_columns_2d = Config.X_LIST_COLUMNS_2D
y_list_columns_2d = Config.Y_LIST_COLUMNS_2D
x_list_columns_1d = Config.X_LIST_COLUMNS_1D
y_list_columns_1d = Config.Y_LIST_COLUMNS_1D

list_columns_2d = x_list_columns_2d + y_list_columns_2d
list_columns_1d = x_list_columns_1d + y_list_columns_1d

for col in x_list_columns_1d + y_list_columns_1d:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x[1:17] if isinstance(x, list) else x)

max_length_1d = df[list_columns_1d].applymap(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).max().max()

# 2D data processing pipeline
cols_2d = x_list_columns_2d + y_list_columns_2d
vars_to_reshape = Config.VARS_TO_RESHAPE

for col in cols_2d:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, (list, tuple)) else x)
        if col in vars_to_reshape:
            df[col] = df[col].apply(lambda x: x.reshape(-1, 1) if isinstance(x, np.ndarray) and x.ndim == 1 else x)
        df[col] = df[col].apply(lambda x: x[0:1, :10] if isinstance(x, np.ndarray) and x.ndim == 2 else x)

max_rows_2d = 0
max_cols_2d = 0
for col in list_columns_2d:
    if col in df.columns:
        for x in df[col]:
            if isinstance(x, np.ndarray) and x.ndim == 2:
                if x.shape[0] > max_rows_2d:
                    max_rows_2d = x.shape[0]
                if x.shape[1] > max_cols_2d:
                    max_cols_2d = x.shape[1]

for col in list_columns_1d:
    df[col] = df[col].apply(lambda x: np.pad(np.array(x), (0, max_length_1d - len(x)), mode='constant') if isinstance(x, (list, np.ndarray)) else np.zeros(max_length_1d))

for col in list_columns_2d:
    df[col] = df[col].apply(lambda x: np.pad(x, ((0, max_rows_2d - x.shape[0]), (0, max_cols_2d - x.shape[1])), mode='constant') if isinstance(x, np.ndarray) and x.ndim == 2 else np.zeros((max_rows_2d, max_cols_2d)))

df = df.dropna(subset=['H2OSOI_10CM'])
print(f"Filtered dataset: {len(df)} samples remaining after removing NaN H2OSOI values.")

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

for col in time_series_columns:
    df[col] = df[col].apply(
        lambda x: (
            np.pad(x[:1476], (0, max(0, 1476 - len(x))), 'constant')[-240:]
            if isinstance(x, list)
            else np.zeros(240, dtype=np.float32)
        )
    )

df = shuffle(df, random_state=42).reset_index(drop=True)

time_series_data = np.stack([np.column_stack(df[col]) for col in time_series_columns], axis=-1)
scaler_time_series = MinMaxScaler(feature_range=(0, 1))
time_series_data = time_series_data.reshape(-1, len(time_series_columns))
time_series_data = scaler_time_series.fit_transform(time_series_data)
time_series_data = time_series_data.reshape(-1, 240, len(time_series_columns))

scaler_static = MinMaxScaler(feature_range=(0, 1))
static_data = scaler_static.fit_transform(df[static_columns].values)

scaler_target = MinMaxScaler(feature_range=(0, 1))
target_data = scaler_target.fit_transform(df[target_columns].values)

time_series_data = torch.tensor(time_series_data, dtype=torch.float32)
static_data = torch.tensor(static_data, dtype=torch.float32)
target_data = torch.tensor(target_data, dtype=torch.float32)

train_size = int(0.8 * len(df))
test_size = len(df) - train_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

pft_columns = [col for col in df.columns if col.startswith('pft_')]
pft_data = np.stack(df[pft_columns].apply(lambda row: np.concatenate(row), axis=1).values)
scaler_pft = MinMaxScaler()
pft_data = scaler_pft.fit_transform(pft_data)
pft_data = torch.tensor(pft_data, dtype=torch.float32)
train_pft = pft_data[:train_size].to(device)
test_pft = pft_data[train_size:].to(device)

groundtruth_y_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_2d}
groundtruth_y_list_columns_1d_tensors = {
    col: torch.stack([torch.tensor(np.array(x, dtype=np.float32)) for x in df[col].values])
    for col in y_list_columns_1d
}

groundtruth_test_y_list_columns_2d = {col: tensor[train_size:].to(device) for col, tensor in groundtruth_y_list_columns_2d_tensors.items()}
groundtruth_test_y_list_columns_1d = {col: tensor[train_size:].to(device) for col, tensor in groundtruth_y_list_columns_1d_tensors.items()}

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

x_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_2d}
x_list_columns_1d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in x_list_columns_1d}
y_list_columns_2d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_2d}
y_list_columns_1d_tensors = {col: torch.stack([torch.tensor(x, dtype=torch.float32) for x in df[col].values]) for col in y_list_columns_1d}

train_x_list_columns_2d = {col: tensor[:train_size].to(device) for col, tensor in x_list_columns_2d_tensors.items()}
train_x_list_columns_1d = {col: tensor[:train_size].to(device) for col, tensor in x_list_columns_1d_tensors.items()}
train_y_list_columns_2d = {col: tensor[:train_size].to(device) for col, tensor in y_list_columns_2d_tensors.items()}
train_y_list_columns_1d = {col: tensor[:train_size].to(device) for col, tensor in y_list_columns_1d_tensors.items()}

train_time_series = time_series_data[:train_size].to(device)
train_static = static_data[:train_size].to(device)
train_target = target_data[:train_size].to(device)

test_x_list_columns_2d = {col: tensor[train_size:].to(device) for col, tensor in x_list_columns_2d_tensors.items()}
test_x_list_columns_1d = {col: tensor[train_size:].to(device) for col, tensor in x_list_columns_1d_tensors.items()}
test_y_list_columns_2d = {col: tensor[train_size:].to(device) for col, tensor in y_list_columns_2d_tensors.items()}
test_y_list_columns_1d = {col: tensor[train_size:].to(device) for col, tensor in y_list_columns_1d_tensors.items()}
test_time_series = time_series_data[train_size:].to(device)
test_static = static_data[train_size:].to(device)
test_target = target_data[train_size:].to(device)

class PFT_CNN(nn.Module):
    def __init__(self, num_pft_features, output_size):
        super(PFT_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(32 * 4, output_size)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.conv_layers(x)

class CombinedModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, num_static_features,
                 num_1d_features, num_2d_features, fc_hidden_size, batch_size, fc_pft_size,
                 num_columns_1d: int, num_columns_2d: int):
        super(CombinedModel, self).__init__()

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc_static = nn.Linear(num_static_features, 64)
        self.fc_1d = nn.Linear(num_1d_features, fc_hidden_size)
        self.fc_pft = PFT_CNN(pft_data.shape[1], fc_pft_size)

        self.num_columns_1d: int = int(num_columns_1d)
        self.num_columns_2d: int = int(num_columns_2d)

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.num_columns_2d, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(128 * 1 * 2, fc_hidden_size)
        )
        self.num_tokens = 4
        self.token_dim = (192 + fc_pft_size) // self.num_tokens

        self.feature_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=4, batch_first=True),
            num_layers=2
        )

        self.fc_scalar_branch = nn.Linear(192 + fc_pft_size, 128)
        self.fc_vector_branch = nn.Linear(192 + fc_pft_size, 128)
        self.fc_matrix_branch = nn.Linear(192 + fc_pft_size, 128)

        self.fc_scalar = nn.Linear(128, 4)
        self.fc_vector = nn.Linear(128, self.num_columns_1d * 16)
        self.fc_matrix = nn.Linear(128, self.num_columns_2d * (1 * 10))

        self.batch_size = batch_size

    def forward(self, time_series_data, static_data, list_1d_data, list_2d_data, pft_data):
        lstm_out, _ = self.lstm(time_series_data)
        lstm_out = lstm_out[:, -1, :]
        static_out = torch.relu(self.fc_static(static_data))
        list_1d_out = torch.relu(self.fc_1d(list_1d_data))
        list_2d_out = self.conv2d(list_2d_data)
        pft_out = self.fc_pft(pft_data)
        
        combined = torch.cat((lstm_out, static_out, list_1d_out, list_2d_out, pft_out), dim=1)
        batch_size_dynamic = combined.shape[0]
        
        self.token_dim = combined.shape[1] // self.num_tokens
        combined_tokens = combined.view(batch_size_dynamic, self.num_tokens, self.token_dim)
        
        fused_tokens = self.feature_fusion(combined_tokens)
        combined_fused = fused_tokens.view(batch_size_dynamic, -1)

        scalar_features = torch.relu(self.fc_scalar_branch(combined_fused))
        vector_features = torch.relu(self.fc_vector_branch(combined_fused))
        matrix_features = torch.relu(self.fc_matrix_branch(combined_fused))

        scalar_output = self.fc_scalar(scalar_features)
        vector_output = F.softplus(self.fc_vector(vector_features)).view(batch_size_dynamic, self.num_columns_1d, 16)
        matrix_output = F.softplus(self.fc_matrix(matrix_features)).view(batch_size_dynamic, self.num_columns_2d, 1, 10)

        return scalar_output, vector_output, matrix_output

lstm_input_size = len(time_series_columns)
lstm_hidden_size = 64
num_static_features = static_data.shape[1]
temp_1d_tensor = torch.cat([tensor for tensor in x_list_columns_1d_tensors.values()], dim=1)
num_1d_features = temp_1d_tensor.shape[1]
num_2d_features = 1 * 10
fc_hidden_size = 32
batch_size = 16
token_dim = 64
fc_pft_size = 16

model = CombinedModel(
    lstm_input_size=lstm_input_size,
    lstm_hidden_size=lstm_hidden_size,
    num_static_features=num_static_features,
    num_1d_features=num_1d_features,
    num_2d_features=num_2d_features,
    fc_hidden_size=fc_hidden_size,
    batch_size=batch_size,
    fc_pft_size=fc_pft_size,
    num_columns_1d=Config.num_all_columns_1D,
    num_columns_2d=Config.num_all_columns_2D
).to(device)

criterion_scalar = nn.MSELoss()
criterion_vector = nn.MSELoss()
criterion_matrix = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
train_losses = []
val_losses = []
W1, W2, W3, W4 = 1, 1, 1, 1

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for i in range(0, train_size, batch_size):
        end_idx = min(i + batch_size, train_size)
        time_series_batch = train_time_series[i:end_idx]
        static_batch = train_static[i:end_idx]
        target_batch = train_target[i:end_idx]
        pft_batch = train_pft[i:end_idx]

        columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in train_x_list_columns_1d.items()}
        columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in train_x_list_columns_2d.items()}
        y_columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in train_y_list_columns_1d.items()}
        y_columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in train_y_list_columns_2d.items()}

        columns_1d_batch_tensor = torch.cat(list(columns_1d_batch.values()), dim=1)
        columns_2d_batch_tensor = torch.cat([t.unsqueeze(1) for t in columns_2d_batch.values()], dim=1)
        y_columns_1d_batch_tensor = torch.cat(list(y_columns_1d_batch.values()), dim=1)
        y_columns_2d_batch_tensor = torch.cat([t.unsqueeze(1) for t in y_columns_2d_batch.values()], dim=1)

        optimizer.zero_grad()
        scalar_pred, vector_pred, matrix_pred = model(time_series_batch, static_batch, columns_1d_batch_tensor, columns_2d_batch_tensor, pft_batch)

        batch_size_y = y_columns_1d_batch_tensor.shape[0]
        y_columns_1d_batch_tensor = y_columns_1d_batch_tensor.view(batch_size_y, Config.num_all_columns_1D, 16)

        loss_scalar = criterion_scalar(scalar_pred, target_batch)
        loss_vector = criterion_vector(vector_pred, y_columns_1d_batch_tensor)
        loss_matrix = criterion_matrix(matrix_pred, y_columns_2d_batch_tensor)

        NPP = scalar_pred[:, target_columns.index('Y_NPP')]
        GPP = scalar_pred[:, target_columns.index('Y_GPP')]
        AR = scalar_pred[:, target_columns.index('Y_AR')]
        C2 = torch.mean((NPP - (GPP - AR)) ** 2)

        loss = W1 * loss_scalar + W2 * loss_vector + W3 * loss_matrix + W4 * C2
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
            time_series_batch = test_time_series[j:end_idx]
            static_batch = test_static[j:end_idx]
            target_batch = test_target[j:end_idx]
            pft_batch = test_pft[j:end_idx]
            
            columns_1d_batch = {col: tensor[j:end_idx] for col, tensor in test_x_list_columns_1d.items()}
            columns_2d_batch = {col: tensor[j:end_idx] for col, tensor in test_x_list_columns_2d.items()}
            y_columns_1d_batch = {col: tensor[j:end_idx] for col, tensor in test_y_list_columns_1d.items()}
            y_columns_2d_batch = {col: tensor[j:end_idx] for col, tensor in test_y_list_columns_2d.items()}

            columns_1d_batch_tensor = torch.cat(list(columns_1d_batch.values()), dim=1)
            columns_2d_batch_tensor = torch.cat([t.unsqueeze(1) for t in columns_2d_batch.values()], dim=1)
            y_columns_1d_batch_tensor = torch.cat(list(y_columns_1d_batch.values()), dim=1)
            y_columns_2d_batch_tensor = torch.cat([t.unsqueeze(1) for t in y_columns_2d_batch.values()], dim=1)

            scalar_pred, vector_pred, matrix_pred = model(time_series_batch, static_batch, columns_1d_batch_tensor, columns_2d_batch_tensor, pft_batch)
            
            batch_size_y = y_columns_1d_batch_tensor.shape[0]
            y_columns_1d_batch_tensor = y_columns_1d_batch_tensor.view(batch_size_y, Config.num_all_columns_1D, 16)

            loss_scalar = criterion_scalar(scalar_pred, target_batch)
            loss_vector = criterion_vector(vector_pred, y_columns_1d_batch_tensor)
            loss_matrix = criterion_matrix(matrix_pred, y_columns_2d_batch_tensor)
            NPP = scalar_pred[:, target_columns.index('Y_NPP')]
            GPP = scalar_pred[:, target_columns.index('Y_GPP')]
            AR = scalar_pred[:, target_columns.index('Y_AR')]
            C2 = torch.mean((NPP - (GPP - AR)) ** 2)

            loss = W1 * loss_scalar + W2 * loss_vector + W3 * loss_matrix + W4 * C2
            val_loss += loss.item()

    val_loss /= (test_size / batch_size)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

losses_df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Train Loss': train_losses,
    'Validation Loss': val_losses
})
losses_df.to_csv("./results/training_validation_losses.csv", index=False)
print("Training and validation losses saved as 'training_validation_losses.csv'")

model.eval()
predictions_scalar, predictions_vector, predictions_matrix = [], [], []

with torch.no_grad():
    for i in range(0, test_size, batch_size):
        end_idx = min(i + batch_size, test_size)
        time_series_batch = test_time_series[i:end_idx]
        static_batch = test_static[i:end_idx]
        pft_batch = test_pft[i:end_idx]

        columns_1d_batch = {col: tensor[i:end_idx] for col, tensor in test_x_list_columns_1d.items()}
        columns_2d_batch = {col: tensor[i:end_idx] for col, tensor in test_x_list_columns_2d.items()}
        
        columns_1d_batch_tensor = torch.cat(list(columns_1d_batch.values()), dim=1)
        columns_2d_batch_tensor = torch.cat([t.unsqueeze(1) for t in columns_2d_batch.values()], dim=1)

        scalar_pred, vector_pred, matrix_pred = model(time_series_batch, static_batch, columns_1d_batch_tensor, columns_2d_batch_tensor, pft_batch)

        predictions_scalar.append(scalar_pred.cpu().numpy())
        predictions_vector.append(vector_pred.cpu().numpy())
        predictions_matrix.append(matrix_pred.cpu().numpy())

predictions_scalar = np.concatenate(predictions_scalar, axis=0)
predictions_vector = np.concatenate(predictions_vector, axis=0)
predictions_matrix = np.concatenate(predictions_matrix, axis=0)

mse_scalar = mean_squared_error(test_target.cpu().numpy(), predictions_scalar)
test_loss_scalar = np.sqrt(mse_scalar)

test_y_list_columns_1d_tensor = torch.cat(list(test_y_list_columns_1d.values()), dim=1).cpu().numpy()
test_y_list_columns_2d_tensor = torch.cat([t.unsqueeze(1) for t in test_y_list_columns_2d.values()], dim=1).cpu().numpy()

mse_vector = mean_squared_error(
    test_y_list_columns_1d_tensor.reshape(-1, Config.num_all_columns_1D * 16),
    predictions_vector.reshape(-1, Config.num_all_columns_1D * 16)
)
mse_matrix = mean_squared_error(
    test_y_list_columns_2d_tensor.reshape(-1, Config.num_all_columns_2D * (1 * 10)),
    predictions_matrix.reshape(-1, Config.num_all_columns_2D * (1 * 10))
)

print(f"Test Loss (Scalar): {test_loss_scalar:.4f}")
print(f"Test Loss (Vector): {mse_vector:.4f}")
print(f"Test Loss (Matrix): {mse_matrix:.4f}")

predictions_scalar_np = scaler_target.inverse_transform(predictions_scalar)
ground_truth_scalar_np = scaler_target.inverse_transform(test_target.cpu().numpy())

predictions_df = pd.DataFrame(predictions_scalar_np, columns=target_columns)
ground_truth_df = pd.DataFrame(ground_truth_scalar_np, columns=target_columns)

predictions_df.to_csv("./results/predictions_original_scale.csv", index=False)
ground_truth_df.to_csv("./results/ground_truth_original_scale.csv", index=False)
print("Scalar predictions and ground truth saved.")

predictions_1d_dict = {}
for idx, col in enumerate(y_list_columns_1d):
    if col in scalers_1d:
        predictions_1d_dict[col] = scalers_1d[col].inverse_transform(predictions_vector[:, idx, :])

ground_truth_1d_dict = {
    col: groundtruth_test_y_list_columns_1d[col].cpu().numpy()
    for col in groundtruth_test_y_list_columns_1d
}

predictions_1d_df = pd.DataFrame({col: data.tolist() for col, data in predictions_1d_dict.items()})
ground_truth_1d_df = pd.DataFrame({col: data.tolist() for col, data in ground_truth_1d_dict.items()})

predictions_1d_df.to_csv("./results/predictions_1d_original_scale.csv", index=False)
ground_truth_1d_df.to_csv("./results/ground_truth_1d_original_scale.csv", index=False)
print("1D predictions and ground truth saved.")

predictions_2d_dict = {}
for idx, col in enumerate(y_list_columns_2d):
    if col in scalers_2d:
        scaler = scalers_2d[col]
        pred_reshaped = predictions_matrix[:, idx, :, :].reshape(-1, 1)
        predictions_2d_dict[col] = scaler.inverse_transform(pred_reshaped).reshape(-1, max_rows_2d, max_cols_2d)

ground_truth_2d_dict = {
    col: groundtruth_test_y_list_columns_2d[col].cpu().numpy()
    for col in groundtruth_test_y_list_columns_2d
}

predictions_2d_df = pd.DataFrame({col: data.tolist() for col, data in predictions_2d_dict.items()})
ground_truth_2d_df = pd.DataFrame({col: data.tolist() for col, data in ground_truth_2d_dict.items()})

predictions_2d_df.to_csv("./results/predictions_2d_original_scale.csv", index=False)
ground_truth_2d_df.to_csv("./results/ground_truth_2d_original_scale.csv", index=False)
print("2D predictions and ground truth saved.")

scripted_model = torch.jit.script(model)
scripted_model.save("LSTM_model.pt")
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
plt.savefig('./results/training_validation_loss.png', dpi=300)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")