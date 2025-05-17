import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


def accuracy(y_true, y_pred, threshold=0.05):
    correct = np.abs(y_true - y_pred) <= threshold
    return np.mean(correct)

# Paths
input_dir  = r'your path'
output_dir = r'your path'

# List of files
input_files  = sorted(glob.glob(os.path.join(input_dir,  '*.csv')))
output_files = sorted(glob.glob(os.path.join(output_dir, '*.csv')))
assert len(input_files) == len(output_files), "Input and output file counts do not match"

data_in_list  = []
data_out_list = []
file_names    = []

# Reading data
for inp_file, out_file in zip(input_files, output_files):
    try:
        inp = pd.read_csv(inp_file,  sep=',', header=0).fillna(0.0).astype(float).values.flatten()
        out = pd.read_csv(out_file, sep=',', header=0).fillna(0.0).astype(float).values.flatten()
    except Exception as e:
        print(f"Error reading {inp_file} or {out_file}: {e}")
        continue
    data_in_list.append(inp)
    data_out_list.append(out)
    file_names.append(os.path.basename(inp_file))

# Padding outputs to the same length
max_len = max(map(len, data_out_list))
data_out_list = [np.pad(o, (0, max_len - len(o)), 'constant') for o in data_out_list]

# Convert to numpy arrays
X_np = np.vstack(data_in_list)
Y_np = np.vstack(data_out_list)

# Standardization of inputs and outputs
input_scaler  = StandardScaler()
output_scaler = StandardScaler()

X_np = input_scaler.fit_transform(X_np)
Y_np = output_scaler.fit_transform(Y_np)

# Convert to tensors
X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(Y_np, dtype=torch.float32)

print("X.shape:", X.shape, "Y.shape:", Y.shape)

# Splitting the dataset
X_train, X_test, Y_train, Y_test, train_files, test_files = train_test_split(
    X, Y, file_names, test_size=0.3, random_state=42
)

# Neural network model
class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, output_dim)
        )
    def forward(self, x):
        return self.model(x)

input_dim  = X.shape[1]
output_dim = Y.shape[1]
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FullyConnectedNet(input_dim, output_dim).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=4, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  Y_test ), batch_size=4)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Training loop
num_epochs = 100
train_losses = []
test_losses  = []

for epoch in range(num_epochs):
    model.train()
    lt = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        lt += loss.item()*bx.size(0)
    lt /= len(train_loader.dataset)
    train_losses.append(lt)

    model.eval()
    lv = 0.0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            lv += criterion(model(bx), by).item()*bx.size(0)
    lv /= len(test_loader.dataset)
    test_losses.append(lv)

    scheduler.step(lv)
    if (epoch+1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}  Train: {lt:.6f}  Test: {lv:.6f}")

# Plotting loss
plt.plot(train_losses, label='Train')
plt.plot(test_losses,  label='Test')
plt.xlabel('Epoch'); plt.ylabel('MSE')
plt.legend(); plt.grid(True); plt.show()

# Column names as in real output
columns = ['a', 'NE', 'TE', 'DN', 'HE', 'XI', 'CN', 'CE', 'q', 'UPL', 'POH', 'Cond', 'j||', 'QNE']

# Prediction for a single file
model.eval()
with torch.no_grad():
    single_X = X_test[0].unsqueeze(0).to(device)  # Take only the first test file
    pred = model(single_X).cpu().numpy()

# Inverse transform prediction
pred_unscaled = output_scaler.inverse_transform(pred)

# Print test file name
print(f"Test file: {test_files[0]}")

# Round predicted values to 3 decimal places
pred_unscaled_rounded = np.round(pred_unscaled, 3)

# Save in correct format
pred_df = pd.DataFrame(pred_unscaled_rounded.reshape(-1, len(columns)), columns=columns)

# Output path
output_path = r'C:\Users\Admin\Desktop\AstraStuff\predictions_test.csv'
pred_df.to_csv(output_path, sep=',', index=False)

print(f"Results for file {test_files[0]} saved to {output_path}")

# Quality evaluation
true_unscaled = output_scaler.inverse_transform(Y_test[0].cpu().numpy().reshape(1, -1))

mse   = mean_squared_error(true_unscaled.flatten(), pred_unscaled.flatten())
rmse  = np.sqrt(mse)
mae   = mean_absolute_error(true_unscaled.flatten(), pred_unscaled.flatten())
r2    = r2_score(true_unscaled.flatten(), pred_unscaled.flatten())

# Metrics for neural network
print("Single file metrics:")
print(f"MSE : {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"R2  : {r2:.6f}")


y_true = true_unscaled.flatten()
y_pred = pred_unscaled.flatten()

pearson_coef, pearson_pval = pearsonr(y_true, y_pred)
print(f"Pearson R: {pearson_coef:.6f}, p-value: {pearson_pval:.3e}")
