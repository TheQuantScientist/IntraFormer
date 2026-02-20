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

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT = "/home/nckh2/qa/IntraFormer/data"

stock_files = {
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

# Ablation configurations
ABLATIONS = [
    {'name': 'full_model', 'use_rev_norm': True, 'num_layers': 3, 'dropout': 0.12, 'use_pos_emb': True},
    {'name': 'no_revnorm', 'use_rev_norm': False, 'num_layers': 3, 'dropout': 0.12, 'use_pos_emb': True},
    {'name': 'no_posemb', 'use_rev_norm': True, 'num_layers': 3, 'dropout': 0.12, 'use_pos_emb': False},
    {'name': 'fewer_layers', 'use_rev_norm': True, 'num_layers': 1, 'dropout': 0.12, 'use_pos_emb': True},
    {'name': 'no_dropout', 'use_rev_norm': True, 'num_layers': 3, 'dropout': 0.0, 'use_pos_emb': True},
]

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
                 num_layers=3, dropout=0.12, use_rev_norm=True, use_pos_emb=True):
        super().__init__()
        self.use_rev_norm = use_rev_norm
        self.use_pos_emb = use_pos_emb
        self.num_channels = num_channels
        self.seq_length = seq_length

        if use_rev_norm:
            self.rev_norm = EquityRevNorm(num_channels, affine=True)

        self.time_projection = nn.Linear(seq_length, d_model)
        if use_pos_emb:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_channels, d_model) * 0.02)
        else:
            self.pos_embedding = None

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
        x = x.view(B, C, -1)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding

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


# ─── Ensemble Evaluation ────────────────────────────────────────────────────────

def evaluate_ensemble_close_prediction(ensemble_members, full_scaled, seq_len, batch_size, device, scaler_close, test_data, symbol, ablation_name):
    dataset = EquityOneStepCloseDataset(full_scaled, seq_len)
    if len(dataset) == 0:
        print(f"Skipping {symbol} - insufficient data")
        return {}, pd.DataFrame()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_predictions = []
    ground_truth = []
    for member in ensemble_members:
        member.eval()
        member_preds = []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                pred = member(bx)
                member_preds.append(pred.cpu().numpy())
                if len(all_predictions) == 0:
                    ground_truth.append(by.numpy())
        all_predictions.append(np.concatenate(member_preds))

    # Consensus = mean of ensemble
    consensus_pred = np.mean(all_predictions, axis=0)
    ground_truth = np.concatenate(ground_truth)

    start_idx = max(0, len(full_scaled) - seq_len - TEST_DAYS)
    pred_slice = consensus_pred[start_idx:start_idx + TEST_DAYS]
    true_slice = ground_truth[start_idx:start_idx + TEST_DAYS]

    if len(pred_slice) == 0:
        return {}, pd.DataFrame()

    mse_scaled  = np.mean((pred_slice - true_slice)**2)
    rmse_scaled = math.sqrt(mse_scaled) if mse_scaled > 0 else 0.0
    mae_scaled  = np.mean(np.abs(pred_slice - true_slice))

    pred_real = scaler_close.inverse_transform(pred_slice.reshape(-1,1)).ravel()
    true_real = scaler_close.inverse_transform(true_slice.reshape(-1,1)).ravel()

    mse_real  = np.mean((pred_real - true_real)**2)
    rmse_real = math.sqrt(mse_real) if mse_real > 0 else 0.0
    mae_real  = np.mean(np.abs(pred_real - true_real))
    mape_pct  = np.mean(np.abs((true_real - pred_real)/(true_real + 1e-8))) * 100

    metrics = {
        'ablation': ablation_name,
        'symbol': symbol,
        'test_days': len(pred_slice),
        'rmse_scaled': round(rmse_scaled, 4),
        'mae_scaled': round(mae_scaled, 4),
        'rmse_real': round(rmse_real, 4),
        'mae_real': round(mae_real, 4),
        'mape_pct': round(mape_pct, 2)
    }

    pred_df = pd.DataFrame({
        'ablation': ablation_name,
        'symbol': symbol,
        'date': test_data.index[-len(pred_real):],
        'true_close': true_real,
        'pred_close': pred_real
    })

    return metrics, pred_df


# ─── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    all_predictions_dfs = []
    all_metrics_list = []

    for ablation in ABLATIONS:
        ablation_name = ablation['name']
        print(f"\n{'#'*70}\nStarting ablation: {ablation_name}\n{'#'*70}")

        for symbol, filepath in stock_files.items():
            print(f"\n{'='*70}\nProcessing equity: {symbol} for ablation {ablation_name}\n{'='*70}")

            try:
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()[FEATURES].ffill().bfill()
            except Exception as e:
                print(f"Failed to load {symbol}: {e}")
                continue

            if len(df) < SEQ_LEN + TEST_DAYS + 100:
                print(f"Skipping {symbol} - insufficient samples")
                continue

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
                print(f"Skipping {symbol} - empty dataset after split")
                continue

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
            val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

            # Train ensemble members
            ensemble_members = []
            for idx in range(ENSEMBLE_MEMBERS):
                print(f"→ Training member {idx+1}/{ENSEMBLE_MEMBERS} (seed={RANDOM_SEEDS[idx]})")
                torch.manual_seed(RANDOM_SEEDS[idx])
                member_model = EquityCloseSeqFormer(
                    num_channels=len(FEATURES), seq_length=SEQ_LEN,
                    d_model=256, nhead=8,
                    num_layers=ablation['num_layers'],
                    dropout=ablation['dropout'],
                    use_rev_norm=ablation['use_rev_norm'],
                    use_pos_emb=ablation['use_pos_emb']
                ).to(device)

                member_model = train_equity_ensemble_member(
                    member_model, train_loader, val_loader, EPOCHS, device
                )
                ensemble_members.append(member_model)

            # Evaluate ensemble
            metrics_dict, pred_dataframe = evaluate_ensemble_close_prediction(
                ensemble_members, full_scaled_data, SEQ_LEN, BATCH_SIZE*2, device,
                close_scaler, test_segment, symbol, ablation_name
            )

            if metrics_dict:
                all_metrics_list.append(metrics_dict)
                all_predictions_dfs.append(pred_dataframe)
                print(f"Evaluation completed for {symbol} in ablation {ablation_name}")

    # ── Save consolidated outputs ───────────────────────────────────────────────
    output_dir = "/home/nckh2/qa"
    os.makedirs(output_dir, exist_ok=True)

    if all_predictions_dfs:
        pd.concat(all_predictions_dfs, ignore_index=True).to_csv(
            os.path.join(output_dir, "IntraFormer_ablation_all_equity_true_vs_pred.csv"), index=False
        )
        print(f"\nSaved predictions: {output_dir}/IntraFormer_ablation_all_equity_true_vs_pred.csv")

    if all_metrics_list:
        pd.DataFrame(all_metrics_list).to_csv(
            os.path.join(output_dir, "IntraFormer_ablation_all_equity_metrics.csv"), index=False
        )
        print(f"Saved metrics: {output_dir}/IntraFormer_ablation_all_equity_metrics.csv")

    print("\nCompleted. No model checkpoints were saved to disk.")