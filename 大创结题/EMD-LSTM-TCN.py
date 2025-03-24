# EMD-LNN-TCN 设备剩余寿命预测模型

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据路径
data_path = r"D:\桌面\大创论文\data.csv"

# 数据预处理
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df.set_index('DATATIME', inplace=True)
    features = ['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 
                'HUMIDITY', 'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']
    data = df[features].values
    
    # 处理缺失值和异常值
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 归一化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler


# EMD分解
def emd(signal, max_imf=10):
    def is_monotonic(x):
        dx = np.diff(x)
        return np.all(dx >= 0) or np.all(dx <= 0)
    imfs = []
    residue = signal
    for _ in range(max_imf):
        h = residue
        while True:
            maxima = np.where((np.diff(np.sign(np.diff(h))) < 0))[0] + 1
            minima = np.where((np.diff(np.sign(np.diff(h))) > 0))[0] + 1
            if len(maxima) < 2 or len(minima) < 2:
                break
            max_env = CubicSpline(maxima, h[maxima])(np.arange(len(h)))
            min_env = CubicSpline(minima, h[minima])(np.arange(len(h)))
            mean_env = (max_env + min_env) / 2
            prev_h = h
            h = h - mean_env
            if np.allclose(h, prev_h):
                break
        imfs.append(h)
        residue = residue - h
        if is_monotonic(residue):
            break
    imfs.append(residue)
    return np.array(imfs)

# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 训练模型
def train_model(model, train_loader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 预测与评估
def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            predictions.append(output.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted', linestyle='--')
    plt.title('Prediction Results')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

# 主程序
data, scaler = load_data(data_path)
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# 增加时间步维度（在第二维）
train_loader = DataLoader(TensorDataset(
    torch.tensor(train_data[:-1]).unsqueeze(1).float(),
    torch.tensor(train_data[1:]).float()), batch_size=32)

test_loader = DataLoader(TensorDataset(
    torch.tensor(test_data[:-1]).unsqueeze(1).float(),
    torch.tensor(test_data[1:]).float()), batch_size=32)

model = LSTM(input_size=9, hidden_size=64, output_size=9).to(device)
train_model(model, train_loader, epochs=50)
save_model(model, "emd_lnn_tcn.pth")
model = load_model(model, "emd_lnn_tcn.pth")
evaluate_model(model, test_loader)

