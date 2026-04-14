import os
import glob
import time
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

# ==============================
# 基本設定
# ==============================
TEST_SIZE = 0.2
WINDOW_SIZE = 100
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 128
HIDDEN_SIZE = 64
N_LAYERS = 2  # Transformer encoder layers
STEP_COUNT = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_LEARN_MODEL = True
IS_STEP_COUNT = True

# 緯度経度のグリッド設定
WIDTH_LAT = {"START":39.75, "GOAL":40.15,  "GRID":80, "LENGTH":0.005}
WIDTH_LON = {"START":116.15,"GOAL":116.55, "GRID":80, "LENGTH":0.005}
lat_len = round((WIDTH_LAT["GOAL"] - WIDTH_LAT["START"]) / WIDTH_LAT["GRID"], 5)
lon_len = round((WIDTH_LON["GOAL"] - WIDTH_LON["START"]) / WIDTH_LON["GRID"], 5)
WIDTH_LAT_SIZE = WIDTH_LAT["GRID"] + 1
WIDTH_LON_SIZE = WIDTH_LON["GRID"] + 1

START = time.time()

# ==============================
# Transformer Encoder モデル
# ==============================
class TransformerDataset(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, n_output=STEP_COUNT*2, nhead=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_output = n_output

        # 入力線形変換 (Embedding)
        self.input_linear = torch.nn.Linear(input_dim, hidden_size)

        # Positional Encoding
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, WINDOW_SIZE, hidden_size))

        # Transformer Encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 出力層
        self.fc_out = torch.nn.Linear(hidden_size, n_output)
        self.relu = torch.nn.ReLU()

        self.to(DEVICE)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_linear(x) + self.pos_embedding  # Embedding + Positional Encoding
        x = self.transformer(x)                       # Transformer Encoder
        cls_token = x[:, -1, :]                       # 最後の時刻の特徴量をCLS的に使用
        y = self.relu(cls_token)
        y = self.fc_out(y)
        return y

# ==============================
# データ読み込み・前処理
# ==============================
def read_data(data_file):
    names = ('Latitude', 'Longitude', 'All_0', 'Altitude', 'Date_num', 'Date', 'Time')
    df_act = pd.read_csv(data_file, sep=',', header=None, names=names)
    df_act['Time'] = pd.to_datetime(df_act['Date'] + ' ' + df_act['Time'])
    df_act.drop(columns=['All_0', 'Date_num', 'Date'], inplace=True)
    df_act.loc[(df_act['Altitude'] == -777), 'Altitude'] = None
    df_act.sort_values('Time', inplace=True)
    df_act = df_act.reset_index(drop=True)

    df_act['Latitude']  = ((df_act['Latitude'] - WIDTH_LAT["START"]) // lat_len).astype(int)
    df_act['Longitude'] = ((df_act['Longitude'] - WIDTH_LON["START"]) // lon_len).astype(int)

    # 配列化
    df_extraction = np.zeros((len(df_act), 2))
    df_extraction[:, 0] = df_act['Latitude'].values
    df_extraction[:, 1] = df_act['Longitude'].values

    # Grid範囲外を削除
    df_extraction = df_extraction[
        (df_extraction[:, 0] >= 0) & (df_extraction[:, 0] < WIDTH_LAT["GRID"]) &
        (df_extraction[:, 1] >= 0) & (df_extraction[:, 1] < WIDTH_LON["GRID"])
    ]
    return df_extraction

def add_terminal(df_datas, add_num=1):
    li_terminal_value = [[WIDTH_LAT["GRID"], WIDTH_LON["GRID"]] for _ in range(add_num)]
    df_datas = [np.append(df_data, li_terminal_value, axis=0) for df_data in df_datas]
    return df_datas

# ==============================
# DataLoader作成
# ==============================
def make_loader(df_datas, n_dim, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, shuffle=True):
    n_data = sum(len(df) - window_size - STEP_COUNT for df in df_datas if len(df) > window_size + STEP_COUNT)
    if n_data <= 0:
        return {"created": False, "size": 0}

    data = np.zeros((n_data, window_size, n_dim))
    labels = np.zeros((n_data, STEP_COUNT, n_dim))

    n_index = 0
    for df_data in df_datas:
        for i in range(len(df_data) - window_size - STEP_COUNT):
            data[n_index] = df_data[i:i+window_size]
            labels[n_index] = df_data[i+window_size:i+window_size+STEP_COUNT]
            n_index += 1

    data = torch.tensor(data, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return {"created": True, "size": n_data, "data": data, "labels": labels, "loader": loader}

# ==============================
# 学習ループ
# ==============================
def learn_model(model, train_loader):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs.view(len(outputs), STEP_COUNT, 2)

            labels[:, :, 0] /= WIDTH_LAT_SIZE
            labels[:, :, 1] /= WIDTH_LON_SIZE

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.6f}")

        loss_history.append(loss.item())
    return model, loss_history

# ==============================
# 推論・評価
# ==============================
def test_result(model, loader, exactly=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs.view(len(outputs), STEP_COUNT, 2)

            # 元のスケールに戻す
            outputs[:, :, 0] *= WIDTH_LAT_SIZE
            outputs[:, :, 1] *= WIDTH_LON_SIZE

            outputs = torch.round(outputs)

            if exactly:
                for i in range(len(outputs)):
                    for j in range(STEP_COUNT):
                        correct += int(torch.all(torch.abs(outputs[i][j] - labels[i][j]) <= 1))
            else:
                for i in range(len(outputs)):
                    for j in range(STEP_COUNT):
                        correct += int(torch.all(outputs[i][j] == labels[i][j]))

            total += len(outputs) * STEP_COUNT

    answer_rate = 100 * correct / total
    print(f"テストの正解率: {answer_rate:.2f}%")
    return answer_rate

# ==============================
# メイン関数
# ==============================
def main():
    path = "output_folder"
    os.makedirs(path, exist_ok=True)

    # データ読み込み
    data_folders = glob.glob(os.path.join(path, "Data") + "/*")
    all_user_results = []

    for data_path in data_folders:
        user_name = os.path.basename(data_path)
        data_files = glob.glob(os.path.join(data_path, "Trajectory") + "/*")
        df_datas = [read_data(f) for f in data_files]
        df_datas = add_terminal(df_datas)

        # Train/Test 分割
        split_idx = int(len(df_datas[0])*(1-TEST_SIZE))
        train_data = [df[:split_idx] for df in df_datas]
        test_data  = [df[split_idx:] for df in df_datas]

        n_dim = df_datas[0].shape[1]
        model = TransformerDataset(n_dim).to(DEVICE)
        summary(model)

        train_loader = make_loader(train_data, n_dim)["loader"]
        test_loader = make_loader(test_data, n_dim)["loader"]

        if IS_LEARN_MODEL:
            model, _ = learn_model(model, train_loader)

        rate = test_result(model, test_loader)
        all_user_results.append((user_name, rate))

    print("全ユーザー結果:", all_user_results)

if __name__ == "__main__":
    main()
