import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import time
import threading
import subprocess

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT = "/home/nckh2/qa/IntraFormer/data"

stock_files = {
    'AAPL': f'{DATA_ROOT}/AAPL_1d_full.csv',
}

SEQ_LEN       = 60
BATCH_SIZE    = 128
EPOCHS        = 1500
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
PATIENCE      = 120
MIN_DELTA     = 5e-7
TEST_DAYS     = 182
VAL_FRACTION  = 0.20

ENSEMBLE_MEMBERS = 3
RANDOM_SEEDS     = [42, 1337, 2025]  # Slightly updated seeds for freshness

FEATURES = ['open', 'high', 'low', 'close', 'volume']
CLOSE_IDX = FEATURES.index('close')  # 3

EMISSION_FACTOR = 0.521  # kg CO2 per kWh for Vietnam

# ─── GPU Power Monitor ──────────────────────────────────────────────────────────

class GPUPowerMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.powers = []
        self.running = False
        self.thread = None

    def _monitor(self):
        while self.running:
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
                power = float(output.strip().decode('utf-8').split('\n')[0])  # Assume single GPU
                self.powers.append(power)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_average_power(self):
        if self.powers:
            return sum(self.powers) / len(self.powers)
        return 0.0

# ─── Reversible Instance Normalization ──────────────────────────────────────────

class EquityRevNorm(nn.Module):
    """
    Reversible Instance Normalization for equity time series.
    Normalizes per instance (batch item) and supports reversible denormalization.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        shape = (1, 1, num_features)
        if affine:
            self.weight = nn.Parameter(torch.ones(shape))
            self.bias   = nn.Parameter(torch.zeros(shape))
        else:
            self.register_buffer('weight', torch.ones(shape))
            self.register_buffer('bias',   torch.zeros(shape))

    def normalize(self, x):
        mean  = x.mean(dim=1, keepdim=True).detach()
        stdev = x.std(dim=1, keepdim=True).detach() + self.eps
        x_norm = (x - mean) / stdev
        return x_norm * self.weight + self.bias, mean, stdev

    def denormalize(self, x, mean, stdev):
        x = (x - self.bias) / (self.weight + self.eps)
        return x * stdev + mean

# ─── Core Model ─────────────────────────────────────────────────────────────────

class EquityCloseSeqFormer(nn.Module):
    """
    Equity Close Sequence Former: Channel-sequential transformer for next-day close prediction.
    Processes OHLCV channels as a sequence of tokens (no patching).
    """
    def __init__(self, num_channels=5, seq_length=60, d_model=256, nhead=8,
                 num_layers=3, dropout=0.12, use_rev_norm=True):
        super().__init__()
        self.use_rev_norm = use_rev_norm
        self.num_channels = num_channels
        self.seq_length = seq_length

        if use_rev_norm:
            self.rev_norm = EquityRevNorm(num_channels, affine=True)

        self.time_projection = nn.Linear(seq_length, d_model)
        self.pos_embedding   = nn.Parameter(torch.randn(1, num_channels, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_stack = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.close_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, src):
        # src: (B, L, C) → batch, time, channels
        B, L, C = src.shape

        if self.use_rev_norm:
            x, mean, stdev = self.rev_norm.normalize(src)
        else:
            x, mean, stdev = src, None, None

        # Project time dimension → (B*C, 1, d_model)
        x = x.permute(0, 2, 1).reshape(B * C, L)
        x = self.time_projection(x.unsqueeze(1))

        # Reshape + add pos embed → treat channels as sequence
        x = x.view(B, C, -1) + self.pos_embedding

        x = self.transformer_stack(x)

        # Predict from close channel token
        close_rep = x[:, CLOSE_IDX, :]
        pred = self.close_predictor(close_rep).squeeze(-1)

        # Denormalize if used
        if self.use_rev_norm:
            pred_full = torch.zeros(B, 1, C, device=pred.device)
            pred_full[:, 0, CLOSE_IDX] = pred
            pred_full = self.rev_norm.denormalize(pred_full, mean, stdev)
            pred = pred_full[:, 0, CLOSE_IDX]

        return pred

# ─── Dataset ────────────────────────────────────────────────────────────────────

class EquityOneStepCloseDataset(Dataset):
    def __init__(self, scaled_data: np.ndarray, seq_length: int):
        self.scaled_data = scaled_data
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.scaled_data) - self.seq_length)

    def __getitem__(self, idx):
        x = self.scaled_data[idx:idx + self.seq_length]
        y = self.scaled_data[idx + self.seq_length, CLOSE_IDX]
        return torch.from_numpy(x).float(), torch.tensor(y).float()

# ─── Training a single ensemble member ──────────────────────────────────────────

def train_equity_ensemble_member(model, train_loader, val_loader, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_batch_count = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            bx = bx + torch.randn_like(bx) * 0.015
            pred = model(bx)
            loss = loss_fn(pred, by)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss_total += loss.item()
            train_batch_count += 1

        avg_train_loss = train_loss_total / train_batch_count if train_batch_count > 0 else float('inf')

        model.eval()
        val_loss_total = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss_total += loss_fn(pred, by).item()
                val_batch_count += 1
        avg_val_loss = val_loss_total / val_batch_count if val_batch_count > 0 else float('inf')

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"[{epoch+1:4d}] train: {avg_train_loss:.7f}  val: {avg_val_loss:.7f}  lr: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        torch.cuda.empty_cache()
        gc.collect()

    if best_weights is not None:
        model.load_state_dict(best_weights)
        print("→ Restored best weights from memory")

    return model

# ─── Inference (without error metrics) ──────────────────────────────────────────

def perform_ensemble_inference(ensemble_members, full_scaled, seq_len, batch_size, device):
    dataset = EquityOneStepCloseDataset(full_scaled, seq_len)
    if len(dataset) == 0:
        print("Insufficient data for inference")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_predictions = []
    for member in ensemble_members:
        member.eval()
        member_preds = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(device)
                pred = member(bx)
                member_preds.append(pred.cpu().numpy())
        all_predictions.append(np.concatenate(member_preds))

    # Just compute consensus to simulate full inference
    _ = np.mean(all_predictions, axis=0)

# ─── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    symbol = 'AAPL'
    filepath = stock_files[symbol]
    print(f"\n{'='*70}\nProcessing equity: {symbol}\n{'='*70}")

    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()[FEATURES].ffill().bfill()
    except Exception as e:
        print(f"Failed to load {symbol}: {e}")
        exit()

    if len(df) < SEQ_LEN + TEST_DAYS + 100:
        print(f"Insufficient samples for {symbol}")
        exit()

    test_segment   = df.iloc[-TEST_DAYS:]
    pre_test_data  = df.iloc[:-TEST_DAYS]
    val_size       = int(len(pre_test_data) * VAL_FRACTION)
    val_segment    = pre_test_data.iloc[-val_size:]
    train_segment  = pre_test_data.iloc[:-val_size]

    scalers = {col: MinMaxScaler().fit(train_segment[[col]]) for col in FEATURES}
    close_scaler = scalers['close']

    def normalize_data(frame):
        return np.hstack([scalers[col].transform(frame[[col]]) for col in FEATURES]).astype(np.float32)

    train_scaled = normalize_data(train_segment)
    val_scaled   = normalize_data(val_segment)
    test_scaled  = normalize_data(test_segment)
    full_scaled_data = np.concatenate([train_scaled, val_scaled, test_scaled])

    train_dataset = EquityOneStepCloseDataset(train_scaled, SEQ_LEN)
    val_dataset   = EquityOneStepCloseDataset(val_scaled,   SEQ_LEN)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"Empty dataset after split for {symbol}")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # ── Measure Training Overhead ───────────────────────────────────────────
    print("→ Starting training measurement")
    train_monitor = GPUPowerMonitor(interval=0.5)
    train_start_time = time.time()
    train_monitor.start()

    ensemble_members = []
    for idx in range(ENSEMBLE_MEMBERS):
        print(f"→ Training member {idx+1}/{ENSEMBLE_MEMBERS} (seed={RANDOM_SEEDS[idx]})")
        torch.manual_seed(RANDOM_SEEDS[idx])
        member_model = EquityCloseSeqFormer(
            num_channels=len(FEATURES), seq_length=SEQ_LEN,
            d_model=256, nhead=8, num_layers=3, dropout=0.12, use_rev_norm=True
        ).to(device)

        member_model = train_equity_ensemble_member(
            member_model, train_loader, val_loader, EPOCHS, device
        )
        ensemble_members.append(member_model)

    train_monitor.stop()
    train_duration_s = time.time() - train_start_time
    train_avg_power_w = train_monitor.get_average_power()
    train_energy_j = train_avg_power_w * train_duration_s
    train_energy_kwh = train_energy_j / 3_600_000
    train_co2_kg = train_energy_kwh * EMISSION_FACTOR
    emission_rate_g_per_kwh = EMISSION_FACTOR * 1000  # gCO2/kWh

    print("\nTraining Metrics:")
    print(f"Duration (s): {train_duration_s:.2f}")
    print(f"Energy (kWh): {train_energy_kwh:.6f}")
    print(f"CO2 eq (kg): {train_co2_kg:.6f}")
    print(f"Emission rate (gCO2eq/kWh): {emission_rate_g_per_kwh:.2f}")

    # ── Measure Inference Overhead ──────────────────────────────────────────
    print("\n→ Starting inference measurement")
    infer_monitor = GPUPowerMonitor(interval=0.5)
    infer_start_time = time.time()
    infer_monitor.start()

    perform_ensemble_inference(
        ensemble_members, full_scaled_data, SEQ_LEN, BATCH_SIZE*2, device
    )

    infer_monitor.stop()
    infer_duration_s = time.time() - infer_start_time
    infer_avg_power_w = infer_monitor.get_average_power()
    infer_energy_j = infer_avg_power_w * infer_duration_s
    infer_energy_kwh = infer_energy_j / 3_600_000
    infer_co2_kg = infer_energy_kwh * EMISSION_FACTOR

    print("\nInference Metrics:")
    print(f"Duration (s): {infer_duration_s:.2f}")
    print(f"Energy (kWh): {infer_energy_kwh:.6f}")
    print(f"CO2 eq (kg): {infer_co2_kg:.6f}")
    print(f"Emission rate (gCO2eq/kWh): {emission_rate_g_per_kwh:.2f}")

    print("\nCompleted. No outputs saved to disk.")