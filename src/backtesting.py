import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PRED_CSV   = "/home/nckh2/qa/IntraFormer_all_equity_true_vs_pred.csv"
DATA_DIR   = Path("/home/nckh2/qa/IntraFormer/data")
# List of symbols (you can also read from directory or from pred file)
SYMBOLS = [
    'AAPL','ABBV','AMD','AMGN','AMZN','AVGO','AXP','BAC','BLK','BMY',
    'C','DHR','GOOGL','GS','INTC','JNJ','JPM','LLY','META','MRK',
    'MS','MSFT','NVDA','ORCL','PFE','SCHW','SPGI','TMO','UNH','WFC'
]

# ── Config ───────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 50000.0
MIN_PRED_RET    = 0.0035          # minimum predicted intraday return to take signal
MAX_EXPOSURE    = 1.0
TRANS_COST_BPS  = 5.0             # 5 bps = 0.05%
N_RANK          = 15

# ── Data loading & preparation ──────────────────────────────────────────────
# 1. Load predictions (long format)
df_pred = pd.read_csv(PRED_CSV, parse_dates=['date'])
df_pred = df_pred.rename(columns={
    'date': 'Date',
    'symbol': 'symbol',
    'true_close': 'true_close',
    'pred_close': 'pred_close'
}).set_index(['symbol', 'Date'])

# 2. Load true OHLC data and combine
dfs = []
for sym in SYMBOLS:
    path = DATA_DIR / f"{sym}_1d_full.csv"
    if not path.exists():
        print(f"Missing: {path}")
        continue
    df_sym = pd.read_csv(path, parse_dates=['Date'])
    df_sym['symbol'] = sym
    dfs.append(df_sym)

df_true = pd.concat(dfs, ignore_index=True)
df_true = df_true[['symbol', 'Date', 'open', 'close']].copy()
df_true = df_true.set_index(['symbol', 'Date'])

# 3. Join predictions + true data
df = df_true.join(df_pred, how='inner').reset_index()
df = df.sort_values(['Date', 'symbol']).copy()

# Create previous close (per symbol)
df['close_prev'] = df.groupby('symbol')['close'].shift(1)

# Drop first day per symbol (no prev close)
df = df.dropna(subset=['close_prev']).copy()

# Quick check
print("Date range:", df['Date'].min(), "→", df['Date'].max())
print("Symbols in backtest:", sorted(df['symbol'].unique()))
print(f"Rows: {len(df):,}")

# ── Helpers ──────────────────────────────────────────────────────────────────
def ret_full(row):
    """ Close-to-close return (previous close → today's close) """
    prev = row['close_prev']
    cl   = row['close']
    return (cl - prev) / prev if prev > 0 else 0.0

def ret_intraday(row):
    """ Open-to-close return (today's intraday move) """
    opn = row['open']
    cl  = row['close']
    return (cl - opn) / opn if opn > 0 else 0.0

# ── Signal functions ─────────────────────────────────────────────────────────
def sig_long_only_all(row):
    """ Long everything predicted to rise > threshold from open """
    sigs = []
    grp = row.groupby('symbol')  # actually single row, but future-proof
    for sym, r in grp:
        if r['open'].iloc[0] <= 0: continue
        pr = (r['pred_close'].iloc[0] - r['open'].iloc[0]) / r['open'].iloc[0]
        if pr > MIN_PRED_RET:
            sigs.append((sym, 1.0, True))
    if not sigs:
        return []
    w = 1.0 / len(sigs)
    return [(sym, w, True) for sym, _, _ in sigs]


def sig_top_n_mom(row):
    """ Long top-N by predicted intraday return """
    pred = {}
    for sym, r in row.groupby('symbol'):
        if r['open'].iloc[0] <= 0: continue
        pr = (r['pred_close'].iloc[0] - r['open'].iloc[0]) / r['open'].iloc[0]
        pred[sym] = pr

    if not pred:
        return []

    top = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:N_RANK]
    valid = [(sym, 1.0, True) for sym, r in top if r > MIN_PRED_RET]
    if not valid:
        return []

    w = 1.0 / len(valid)
    return [(sym, w, True) for sym, _, _ in valid]


def sig_long_short(row):
    """ Long top-N / short bottom-N by predicted close-to-close return """
    pred = {}
    for sym, r in row.groupby('symbol'):
        prev = r['close_prev'].iloc[0]
        if prev <= 0: continue
        pr = (r['pred_close'].iloc[0] - prev) / prev
        pred[sym] = pr

    if not pred:
        return []

    srt = sorted(pred.items(), key=lambda x: x[1], reverse=True)

    longs  = [(sym, 1.0, True)  for sym, r in srt[:N_RANK]  if r >  MIN_PRED_RET]
    shorts = [(sym, -1.0, False) for sym, r in srt[-N_RANK:] if r < -MIN_PRED_RET]

    all_sig = longs + shorts
    if not all_sig:
        return []

    gross = sum(abs(w) for _, w, _ in all_sig)
    norm = 1.0 / gross if gross > 0 else 0
    return [(sym, w * norm, lng) for sym, w, lng in all_sig]


# ── Simulation ───────────────────────────────────────────────────────────────
def run_strategy(sig_func, name, ret_func):
    capital = INITIAL_CAPITAL
    peak = capital
    max_dd = 0.0
    daily_rets = []
    trade_days = 0
    coin_track = defaultdict(lambda: {'pnl': 0.0, 'days': 0, 'wins': 0, 'gross_rets': []})

    # Group by date (each iteration = one trading day)
    for date, day_df in df.groupby('Date', sort=True):
        signals = sig_func(day_df)

        if not signals:
            dr = 0.0
        else:
            trade_days += 1
            tot_abs = sum(abs(w) for _, w, _ in signals)
            scale = min(tot_abs, MAX_EXPOSURE) / tot_abs if tot_abs > 0 else 0.0
            dr = 0.0

            for sym, w, is_long in signals:
                # Find row for this symbol on this day
                row = day_df[day_df['symbol'] == sym].iloc[0]
                gr = ret_func(row)
                signed = gr if is_long else -gr
                tcost = (TRANS_COST_BPS / 10000) * abs(w)   # one-way cost
                net = signed - tcost
                contrib = net * abs(w) * scale
                dr += contrib

                # Track per symbol
                ct = coin_track[sym]
                ct['pnl'] += contrib * capital
                ct['days'] += 1
                ct['gross_rets'].append(signed)
                if signed > 0:
                    ct['wins'] += 1

        capital *= (1 + dr)
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        daily_rets.append(dr)

    # ── Statistics ───────────────────────────────────────────────────────────
    tot_ret = (capital / INITIAL_CAPITAL) - 1
    sharpe = (np.mean(daily_rets) / np.std(daily_rets)) * np.sqrt(182) if np.std(daily_rets) > 0 else 0   # ≈252 trading days
    calmar = tot_ret / max_dd if max_dd > 0 else 0

    print(f"\n{name}")
    print(f" Total Return : {tot_ret:8.2%}")
    print(f" Sharpe       : {sharpe:8.2f}")
    print(f" Max DD       : {max_dd:8.2%}")
    print(f" Calmar       : {calmar:8.2f}")
    print(f" Final $      : {capital:9,.0f}")
    print(f" Trade days   : {trade_days:4d} / {df['Date'].nunique()}")

    # Per-symbol stats
    rows = []
    for sym, st in coin_track.items():
        if st['days'] == 0: continue
        hit = st['wins'] / st['days'] if st['days'] > 0 else 0
        avg_gr = np.mean(st['gross_rets']) if st['gross_rets'] else 0
        cum_gr = np.prod(1 + np.array(st['gross_rets'])) - 1 if st['gross_rets'] else 0
        rows.append({
            'Symbol': sym,
            'PnL $': round(st['pnl'], 1),
            'Days': st['days'],
            'Hit Rate': f"{hit:5.1%}",
            'Avg ret/day': f"{avg_gr:6.2%}",
            'Cum ret': f"{cum_gr:6.2%}",
        })

    if rows:
        pdf = pd.DataFrame(rows).sort_values('PnL $', ascending=False).set_index('Symbol')
        print("\nPer-symbol performance:")
        print(pdf.round(2))
    else:
        print(" No positions taken")

    return {
        'name': name,
        'Total Return': tot_ret,
        'Sharpe': sharpe,
        'Max DD': max_dd,
        'Calmar': calmar,
        'Final $': capital
    }


# ── Run all strategies ───────────────────────────────────────────────────────
strategies = [
    (sig_long_only_all, "1. Intraday Long-Only (all > thresh)", ret_intraday),
    (sig_top_n_mom,     "2. Intraday Top-N Predicted Return",   ret_intraday),
    (sig_long_short,    "3. Long-Short (top/bottom N pred)",   ret_full),
]

summary_rows = []
for sig_f, name, ret_f in strategies:
    res = run_strategy(sig_f, name, ret_f)
    summary_rows.append(res)

# ── Summary table ────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SUMMARY COMPARISON")
sum_df = pd.DataFrame(summary_rows).set_index('name')
sum_df = sum_df[['Total Return', 'Sharpe', 'Max DD', 'Calmar', 'Final $']]
sum_df['Total Return'] = sum_df['Total Return'].apply(lambda x: f"{x:7.2%}")
sum_df['Max DD']       = sum_df['Max DD'].apply(lambda x: f"{x:7.2%}")
print(sum_df.round(2))
print("═"*70)