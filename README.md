# IntraFormer – Channel-Sequential Transformer for Next-Day Equity Close Prediction

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3.9+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License: MIT">
</p>

![Model Architecture](graphs/IntraFormer.png)

**IntraFormer** is a lightweight, channel-first transformer architecture designed specifically for **next-day closing price prediction** on daily equity data (OHLCV).

Instead of treating time steps as the sequence (common in most time-series transformers), IntraFormer **treats the OHLCV channels as the sequence** and projects the time dimension — a design choice that has shown surprisingly strong empirical performance on individual stock prediction tasks.

## Key Features

- Channel-sequential Transformer encoder (no patching)
- **Reversible Instance Normalization** (RevIN) tailored for financial time series
- Optional positional embeddings on channels
- Ensemble of 3 independently trained models (different seeds)
- Ablation suite comparing RevIN, positional embeddings, layer count, dropout
- Realistic walk-forward backtesting with transaction costs (5 bps)
- Comparison scripts vs traditional deep models (LSTM/GRU/CNN-LSTM/CLAM)
- Carbon footprint estimation example (training + inference energy)

## Project Structure

```text
thequantscientist-intraformer/
├── README.md
├── LICENSE                  # MIT
└── src/
    ├── IntraFormer.py       # Main single-stock training & evaluation script
    ├── ablation.py          # Ablation study: RevIN / pos emb / layers / dropout
    ├── backtesting.py       # Portfolio-level walk-forward backtest
    ├── computation.py       # Energy & CO₂ footprint measurement example
    ├── traditional_benchmark.py   # LSTM / GRU / CNN-LSTM / CLAM baselines
    └── transformer_benchmark.py   # Autoformer / FEDformer / Informer / iTransformer
    └── ultimate_results.csv   # (example output – not versioned)
```

## Quick Start

### 1. Requirements

```bash
pip install torch pandas numpy scikit-learn
# optional: for transformer benchmarks
pip install einops
```

### 2. Data Preparation

Place daily OHLCV CSV files in `data/` with format:

```csv
Date,open,high,low,close,volume
2020-01-02,74.06,75.15,73.80,75.09,135480400
...
```

Example filename: `AAPL_1d_full.csv`

Update paths in the scripts if needed.

### 3. Run main experiment (30 stocks, ensemble of 3)

```bash
cd src
python IntraFormer.py
```

→ Produces predictions and metrics in `~/qa/IntraFormer_all_equity_*`

### 4. Run ablation study

```bash
python ablation.py
```

→ Compares 5 configurations across all stocks

### 5. Run backtest

After generating predictions:

```bash
python backtesting.py
```

Shows long-only, top-N, and long-short portfolio performance with realistic costs.

## Model Architecture (simplified)

```text
Input:  (B, 60 days, 5 channels: OHLCV)

→ Reversible Instance Normalization (per sample)
→ Linear projection: 60 → d_model=256   (per channel)
→ Add learnable channel positional embedding   (optional)
→ TransformerEncoder (3 layers, 8 heads, GELU FFN)
→ Take close-channel representation (index 3)
→ MLP head → scalar next-day close prediction
→ Denormalize (if RevIN used)
```

## Results Highlights (your data)

- Best ablation usually includes **RevIN + positional embeddings + 3 layers**
- IntraFormer frequently outperforms LSTM/GRU baselines by 10–30% in RMSE/MAE on many names
- Portfolio strategies show positive Sharpe even after transaction costs in several configurations

## Citation

If you find this work useful in your research, feel free to cite:

```bibtex
@misc{intraformer2026,
  author = {Nguyen Quoc Anh},
  title  = {IntraFormer: Channel-Sequential Transformer for Daily Equity Close Prediction},
  year   = {2026},
  url    = {https://github.com/thequantscientist/intraformer}
}
```

