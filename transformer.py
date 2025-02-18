import math
import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error


# ---------------------------
# Dataset: Tạo sequence từ dữ liệu lưu lượng giao thông
# ---------------------------
class TrafficDataset(Dataset):
    def __init__(self, data, window_size):
        """
        Args:
            data (numpy array): Dữ liệu đã được chuẩn hóa, shape (n_samples, 1)
            window_size (int): Số bước lịch sử để dự báo bước kế tiếp.
        """
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.window_size]
        target = self.data[idx + self.window_size]
        return torch.FloatTensor(seq), torch.FloatTensor(target)


# ---------------------------
# Positional Encoding: Thêm thông tin vị trí vào embedding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model: Số chiều của embedding.
            dropout: Tỷ lệ dropout.
            max_len: Chiều dài tối đa của sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x có shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------
# Transformer Model cho Time Series Forecasting
# ---------------------------
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=8, num_layers=2, dropout=0.1, window_size=64, output_dim=1):
        """
        Args:
            input_dim: Số chiều đầu vào (1 đối với univariate).
            d_model: Số chiều embedding của Transformer.
            nhead: Số head trong multi-head attention.
            num_layers: Số lớp TransformerEncoder.
            dropout: Tỷ lệ dropout.
            window_size: Độ dài của sequence đầu vào.
            output_dim: Số chiều đầu ra (1 đối với univariate forecast).
        """
        super(TransformerTimeSeries, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=window_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        x = self.pos_encoder(x)  # -> (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # -> (seq_len, batch, d_model)
        x = self.transformer_encoder(x)  # -> (seq_len, batch, d_model)
        out = x[-1, :, :]  # Lấy output của bước cuối cùng: (batch, d_model)
        out = self.fc(out)  # -> (batch, output_dim)
        return out


# ---------------------------
# Main function: Đọc dữ liệu, huấn luyện, đánh giá và plot kết quả training
# ---------------------------
def main(args):
    # Đọc dữ liệu từ CSV
    data = pd.read_csv(args.file_path, parse_dates=["timestamp"])
    values = data["hourly_traffic_count"].values.reshape(-1, 1)

    # Chuẩn hóa dữ liệu về khoảng [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values)

    window_size = args.time_step  # Dùng time_step làm window size
    dataset = TrafficDataset(values_scaled, window_size)

    # Chia dữ liệu thành train và test (80% train, 20% test)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình Transformer với các siêu tham số từ argument
    model = TransformerTimeSeries(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        window_size=window_size
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    train_loss_history = []
    test_loss_history = []

    for epoch in range(1, args.train_epoch + 1):
        model.train()
        train_losses = []
        for seq, target in train_loader:
            seq = seq.to(device)  # (batch, window_size, 1)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch}/{args.train_epoch} - Train Loss: {avg_train_loss:.4f}")

        # Đánh giá trên tập test
        model.eval()
        test_losses = []
        with torch.no_grad():
            for seq, target in test_loader:
                seq = seq.to(device)
                target = target.to(device)
                output = model(seq)
                loss = criterion(output, target)
                test_losses.append(loss.item())
        avg_test_loss = np.mean(test_losses)
        test_loss_history.append(avg_test_loss)
        print(f"Epoch {epoch}/{args.train_epoch} - Test Loss: {avg_test_loss:.4f}")

        # Lưu checkpoint nếu loss test giảm
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with Test Loss: {avg_test_loss:.4f}")

    # Vẽ đồ thị loss của quá trình training và test
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(test_loss_history, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Test Loss over Epochs")
    plt.legend()
    plt.savefig(args.plot_path)
    plt.show()

    # Đánh giá trên tập test để tính các metrics
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for seq, target in test_loader:
            seq = seq.to(device)
            output = model(seq)
            predictions.extend(output.cpu().numpy())
            actuals.extend(target.cpu().numpy())

    # Chuyển đổi giá trị về thang ban đầu
    predictions = scaler.inverse_transform(np.array(predictions))
    actuals = scaler.inverse_transform(np.array(actuals))

    r2 = r2_score(actuals, predictions)
    mse_value = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse_value)

    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse_value:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Lưu metrics vào file JSON
    metrics = {
        "epoch": args.train_epoch,
        "r2_score": r2,
        "mse": mse_value,
        "rmse": rmse
    }
    with open(args.metrics_path, "w") as f:
        json.dump(metrics, f)

    # Lưu model cuối cùng
    torch.save(model.state_dict(), args.model_path)
    print("Final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer for Traffic Flow Forecasting")
    parser.add_argument('--file_path', type=str, default="resource/train27303.csv", help="Path to CSV dataset")
    parser.add_argument('--metrics_path', type=str, default="res/Transformer/transformer_metrics.json",
                        help="Path to save metrics JSON")
    parser.add_argument('--plot_path', type=str, default="res/Transformer/transformer_plot.png",
                        help="Path to save loss plot")
    parser.add_argument('--model_path', type=str, default="res/Transformer/transformer_model.pth",
                        help="Path to save final model")
    parser.add_argument('--checkpoint_path', type=str, default="res/Transformer/transformer_checkpoint.pth",
                        help="Path to save checkpoint model")
    parser.add_argument('--time_step', type=int, default=64, help="Number of historical steps (window size)")
    parser.add_argument('--train_epoch', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    # Các siêu tham số cho Transformer
    parser.add_argument('--d_model', type=int, default=64, help="Embedding dimension of Transformer")
    parser.add_argument('--nhead', type=int, default=8, help="Number of heads in multi-head attention")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of Transformer encoder layers")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    main(args)
