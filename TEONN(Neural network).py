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

# Пути
input_dir  = r'your path'
output_dir = r'your path'

# Список файлов
input_files  = sorted(glob.glob(os.path.join(input_dir,  '*.csv')))
output_files = sorted(glob.glob(os.path.join(output_dir, '*.csv')))
assert len(input_files) == len(output_files), "Входные и выходные файлы в разном количестве"

data_in_list  = []
data_out_list = []
file_names    = []

# Считываем данные
for inp_file, out_file in zip(input_files, output_files):
    try:
        inp = pd.read_csv(inp_file,  sep=',', header=0).fillna(0.0).astype(float).values.flatten()
        out = pd.read_csv(out_file, sep=',', header=0).fillna(0.0).astype(float).values.flatten()
    except Exception as e:
        print(f"Ошибка при чтении {inp_file} или {out_file}: {e}")
        continue
    data_in_list.append(inp)
    data_out_list.append(out)
    file_names.append(os.path.basename(inp_file))

# Padding выходов до одной длины
max_len = max(map(len, data_out_list))
data_out_list = [np.pad(o, (0, max_len - len(o)), 'constant') for o in data_out_list]

# Превращаем в numpy
X_np = np.vstack(data_in_list)
Y_np = np.vstack(data_out_list)

# Стандартизация входа и выхода
input_scaler  = StandardScaler()
output_scaler = StandardScaler()

X_np = input_scaler.fit_transform(X_np)
Y_np = output_scaler.fit_transform(Y_np)

# Перевод в тензоры
X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(Y_np, dtype=torch.float32)

print("X.shape:", X.shape, "Y.shape:", Y.shape)

# Делим данные
X_train, X_test, Y_train, Y_test, train_files, test_files = train_test_split(
    X, Y, file_names, test_size=0.3, random_state=42
)

# Модель
class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            #nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            
            #nn.Softplus(),
            nn.ReLU(),
            #nn.Dropout(0.2),

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

# Обучение
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

# График потерь
plt.plot(train_losses, label='Train')
plt.plot(test_losses,  label='Test')
plt.xlabel('Epoch'); plt.ylabel('MSE')
plt.legend(); plt.grid(True); plt.show()


# Названия колонок как в настоящем output
columns = ['a', 'NE', 'TE', 'DN', 'HE', 'XI', 'CN', 'CE', 'q', 'UPL', 'POH', 'Cond', 'j||', 'QNE']

# Предсказание для одного файла
model.eval()
with torch.no_grad():
    single_X = X_test[0].unsqueeze(0).to(device)  # Берем только первый файл из теста
    pred = model(single_X).cpu().numpy()

# Обратная стандартизация предсказания
pred_unscaled = output_scaler.inverse_transform(pred)

# Выводим название тестового файла
print(f"Тестовый файл: {test_files[0]}")

# Округляем предсказанные значения до 3 знаков после запятой
pred_unscaled_rounded = np.round(pred_unscaled, 3)

# Сохраняем в правильном формате
pred_df = pd.DataFrame(pred_unscaled_rounded.reshape(-1, len(columns)), columns=columns)

# Путь сохранения
output_path = r'C:\Users\Admin\Desktop\AstraStuff\predictions_test.csv'
pred_df.to_csv(output_path, sep=',', index=False)

print(f"Результаты для файла {test_files[0]} сохранены в {output_path}")

# Оценка качества
true_unscaled = output_scaler.inverse_transform(Y_test[0].cpu().numpy().reshape(1, -1))

mse   = mean_squared_error(true_unscaled.flatten(), pred_unscaled.flatten())
rmse  = np.sqrt(mse)
mae   = mean_absolute_error(true_unscaled.flatten(), pred_unscaled.flatten())
r2    = r2_score(true_unscaled.flatten(), pred_unscaled.flatten())

# Метрики для нейронной сети


# ----------------------
print("Single file metrics:")
print(f"MSE : {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"R2  : {r2:.6f}")



y_true = true_unscaled.flatten()
y_pred = pred_unscaled.flatten()


pearson_coef, pearson_pval = pearsonr(y_true, y_pred)
print(f"Pearson R: {pearson_coef:.6f}, p-value: {pearson_pval:.3e}")

# ---------------------------
