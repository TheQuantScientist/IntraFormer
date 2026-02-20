import os
import warnings
import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT = "/home/nckh2/qa/IntraFormer/data"

STOCK_FILES = {
    'AAPL':  f'{DATA_ROOT}/AAPL_1d_full.csv',
    'ABBV':  f'{DATA_ROOT}/ABBV_1d_full.csv',
    'AMD':   f'{DATA_ROOT}/AMD_1d_full.csv',
    'AMGN':  f'{DATA_ROOT}/AMGN_1d_full.csv',
    'AMZN':  f'{DATA_ROOT}/AMZN_1d_full.csv',
    'AVGO':  f'{DATA_ROOT}/AVGO_1d_full.csv',
    'AXP':   f'{DATA_ROOT}/AXP_1d_full.csv',
    'BAC':   f'{DATA_ROOT}/BAC_1d_full.csv',
    'BLK':   f'{DATA_ROOT}/BLK_1d_full.csv',
    'BMY':   f'{DATA_ROOT}/BMY_1d_full.csv',
    'C':     f'{DATA_ROOT}/C_1d_full.csv',
    'DHR':   f'{DATA_ROOT}/DHR_1d_full.csv',
    'GOOGL': f'{DATA_ROOT}/GOOGL_1d_full.csv',
    'GS':    f'{DATA_ROOT}/GS_1d_full.csv',
    'INTC':  f'{DATA_ROOT}/INTC_1d_full.csv',
    'JNJ':   f'{DATA_ROOT}/JNJ_1d_full.csv',
    'JPM':   f'{DATA_ROOT}/JPM_1d_full.csv',
    'LLY':   f'{DATA_ROOT}/LLY_1d_full.csv',
    'META':  f'{DATA_ROOT}/META_1d_full.csv',
    'MRK':   f'{DATA_ROOT}/MRK_1d_full.csv',
    'MS':    f'{DATA_ROOT}/MS_1d_full.csv',
    'MSFT':  f'{DATA_ROOT}/MSFT_1d_full.csv',
    'NVDA':  f'{DATA_ROOT}/NVDA_1d_full.csv',
    'ORCL':  f'{DATA_ROOT}/ORCL_1d_full.csv',
    'PFE':   f'{DATA_ROOT}/PFE_1d_full.csv',
    'SCHW':  f'{DATA_ROOT}/SCHW_1d_full.csv',
    'SPGI':  f'{DATA_ROOT}/SPGI_1d_full.csv',
    'TMO':   f'{DATA_ROOT}/TMO_1d_full.csv',
    'UNH':   f'{DATA_ROOT}/UNH_1d_full.csv',
    'WFC':   f'{DATA_ROOT}/WFC_1d_full.csv',
}

SEQ_LEN       = 60
BATCH_SIZE    = 128
EPOCHS        = 1500              # reduced for simple models
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
PATIENCE      = 120
MIN_DELTA     = 5e-7
TEST_DAYS     = 182
VAL_FRACTION  = 0.20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FN = nn.MSELoss()

FEATURES = ['open', 'high', 'low', 'close', 'volume']
CLOSE_IDX = FEATURES.index('close')  # 3

PREFIX = "stock_traditional_per_stock_"

# ─── Models ─────────────────────────────────────────────────────────────────────

class CNNLSTM(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch, seq_len, features) → transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)           # (B, C, L)
        x = self.conv(x)                # (B, 128, L)
        x = x.transpose(1, 2)           # (B, L, 128)
        x, (h_n, c_n) = self.lstm(x)
        # Take last time step
        x = x[:, -1, :]                 # (B, 256)
        return self.fc(x).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1]).squeeze(-1)


class GRU(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.gru = nn.GRU(input_size, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(o[:, -1]).squeeze(-1)


class CLAM(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 200, 2, batch_first=True)
        self.attn = nn.Linear(200, 1)
        self.fc = nn.Linear(200, 1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        w = torch.softmax(self.attn(x), dim=1)
        x = (w * x).sum(dim=1)
        return self.fc(x).squeeze(-1)


MODELS = {
    'CNN-LSTM': CNNLSTM,
    'LSTM':     LSTM,
    'GRU':      GRU,
    'CLAM':     CLAM
}

# ─── Dataset ────────────────────────────────────────────────────────────────────

class OneStepCloseDataset(Dataset):
    def __init__(self, scaled_data: np.ndarray, seq_length: int):
        self.scaled_data = scaled_data
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.scaled_data) - self.seq_length)

    def __getitem__(self, idx):
        x = self.scaled_data[idx:idx + self.seq_length]
        y = self.scaled_data[idx + self.seq_length, CLOSE_IDX]
        return torch.from_numpy(x).float(), torch.tensor(y).float()

# ─── Train function ─────────────────────────────────────────────────────────────

def train_model(model_class, train_loader, val_loader, epochs, device):
    model = model_class().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_batch_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Noise augmentation
            x = x + torch.randn_like(x) * 0.015

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = LOSS_FN(pred, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            train_loss_total += loss.item()
            train_batch_count += 1

        avg_train_loss = train_loss_total / train_batch_count if train_batch_count > 0 else float('inf')

        model.eval()
        val_loss_total = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss_total += LOSS_FN(pred, y).item()
                val_batch_count += 1

        avg_val_loss = val_loss_total / val_batch_count if val_batch_count > 0 else float('inf')

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"[{epoch+1:4d}] train: {avg_train_loss:.7f}  val: {avg_val_loss:.7f}")

        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        torch.cuda.empty_cache()
        gc.collect()

    if best_state is not None:
        model.load_state_dict(best_state)
        print("→ Restored best weights")

    return model

# ─── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate_model(model, full_scaled, seq_len, batch_size, device, close_scaler, test_df, symbol):
    dataset = OneStepCloseDataset(full_scaled, seq_len)
    if len(dataset) == 0:
        return None, None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).cpu().numpy()
            preds.append(p)
            trues.append(y.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    start = max(0, len(full_scaled) - seq_len - TEST_DAYS)
    p_test = preds[start : start + TEST_DAYS]
    t_test = trues[start : start + TEST_DAYS]

    if len(p_test) == 0:
        return None, None

    # Scaled
    mse_s  = np.mean((p_test - t_test)**2)
    rmse_s = math.sqrt(mse_s) if mse_s > 0 else 0.0
    mae_s  = np.mean(np.abs(p_test - t_test))

    # Real
    p_real = close_scaler.inverse_transform(p_test.reshape(-1, 1)).ravel()
    t_real = close_scaler.inverse_transform(t_test.reshape(-1, 1)).ravel()

    mse_r  = np.mean((p_real - t_real)**2)
    rmse_r = math.sqrt(mse_r) if mse_r > 0 else 0.0
    mae_r  = np.mean(np.abs(p_real - t_real))
    mape   = np.mean(np.abs((t_real - p_real) / (t_real + 1e-8))) * 100

    metrics = {
        'symbol': symbol,
        'rmse_scaled': round(rmse_s, 4),
        'mae_scaled': round(mae_s, 4),
        'rmse_real': round(rmse_r, 4),
        'mae_real': round(mae_r, 4),
        'mape_pct': round(mape, 2)
    }

    pred_df = pd.DataFrame({
        'symbol': symbol,
        'date': test_df.index[-len(p_real):],
        'true_close': t_real,
        'pred_close': p_real
    })

    return metrics, pred_df

# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_metrics = []
    all_preds = []

    for symbol, path in STOCK_FILES.items():
        print(f"\n{'='*70}\nProcessing {symbol}\n{'='*70}")

        try:
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')[FEATURES].sort_index().ffill().bfill()
        except Exception as e:
            print(f"Failed to load {symbol}: {e}")
            continue

        if len(df) < SEQ_LEN + TEST_DAYS + 100:
            print(f"Skipping {symbol} — insufficient data")
            continue

        test_df = df.iloc[-TEST_DAYS:]
        pre_test = df.iloc[:-TEST_DAYS]
        val_size = int(len(pre_test) * VAL_FRACTION)
        val_df = pre_test.iloc[-val_size:]
        train_df = pre_test.iloc[:-val_size]

        scalers = {f: MinMaxScaler().fit(train_df[[f]]) for f in FEATURES}
        close_scaler = scalers['close']

        train_s = np.hstack([scalers[f].transform(train_df[[f]]) for f in FEATURES]).astype(np.float32)
        val_s   = np.hstack([scalers[f].transform(val_df[[f]]) for f in FEATURES]).astype(np.float32)
        test_s  = np.hstack([scalers[f].transform(test_df[[f]]) for f in FEATURES]).astype(np.float32)
        full_s  = np.concatenate([train_s, val_s, test_s])

        train_ds = OneStepCloseDataset(train_s, SEQ_LEN)
        val_ds   = OneStepCloseDataset(val_s, SEQ_LEN)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"Skipping {symbol} — empty dataset after split")
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        for name, ModelCls in MODELS.items():
            print(f"  → {name}")
            model = train_model(ModelCls, train_loader, val_loader, EPOCHS, DEVICE)
            metrics, df_pred = evaluate_model(model, full_s, SEQ_LEN, BATCH_SIZE, DEVICE, close_scaler, test_df, symbol)

            if metrics:
                metrics['model'] = name
                all_metrics.append(metrics)
                if df_pred is not None:
                    df_pred['model'] = name
                    all_preds.append(df_pred)

            torch.cuda.empty_cache()
            gc.collect()

    # Save
    output_dir = "/home/nckh2/qa"
    os.makedirs(output_dir, exist_ok=True)

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, f"{PREFIX}metrics.csv"), index=False)
        print(f"Saved metrics → {output_dir}/{PREFIX}metrics.csv")

    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_csv(os.path.join(output_dir, f"{PREFIX}predictions.csv"), index=False)
        print(f"Saved predictions → {output_dir}/{PREFIX}predictions.csv")

    print("\nDone.")