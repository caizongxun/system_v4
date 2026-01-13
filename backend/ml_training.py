"""ML training pipeline for System V4.

This script loads OHLCV data from the HuggingFace dataset
`zongowo111/v2-crypto-ohlcv-data`, builds engineered features
(momentum, volatility and custom composite indicators),
constructs trade labels based on ATR stop loss and 1:1.5 risk-reward,
then trains a baseline model to predict whether to enter long, short or stay flat
on the next bar given the last fully closed bar.

The entire flow is modular to facilitate debugging and future optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Dict, Optional
import time

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
SUPPORTED_SYMBOLS: List[str] = [
    "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT",
    "BALUSDT", "BATUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "COMPUSDT",
    "CRVUSDT", "DOGEUSDT", "DOTUSDT", "ENJUSDT", "ENSUSDT", "ETCUSDT",
    "ETHUSDT", "FILUSDT", "GALAUSDT", "GRTUSDT", "IMXUSDT", "KAVAUSDT",
    "LINKUSDT", "LTCUSDT", "MANAUSDT", "MATICUSDT", "MKRUSDT", "NEARUSDT",
    "OPUSDT", "SANDUSDT", "SNXUSDT", "SOLUSDT", "SPELLUSDT", "UNIUSDT",
    "XRPUSDT", "ZRXUSDT",
]
SUPPORTED_TIMEFRAMES: List[str] = ["15m", "1h", "1d"]


class TradeLabel(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = 2


@dataclass
class ATRConfig:
    period: int = 14
    stop_atr: float = 1.0
    rr_ratio: float = 1.5  # risk:reward = 1:1.5


# ==============================
# Data Loading
# ==============================


def load_klines(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data for a symbol and timeframe from HuggingFace.

    The dataset structure follows:
    klines/{SYMBOL}/{BASE}_{TIMEFRAME}.parquet
    """
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol: {symbol}")
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    base = symbol.replace("USDT", "")
    filename = f"{base}_{timeframe}.parquet"
    path_in_repo = f"klines/{symbol}/{filename}"

    print(f"  Downloading {path_in_repo}...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=path_in_repo,
        repo_type="dataset",
    )

    df = pd.read_parquet(local_path)

    # Ensure datetime index on open_time
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")

    df = df.sort_index()
    print(f"  Loaded {len(df)} bars")
    return df


# ==============================
# Feature Engineering
# ==============================


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def add_momentum_and_volatility_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add momentum and volatility features based on % returns and ATR."""
    df = df.copy()

    # Percentage returns
    df["ret_close"] = df["close"].pct_change()

    # Momentum: rolling sum of returns
    df["mom_return"] = df["ret_close"].rolling(window).sum()

    # Volatility: rolling standard deviation of returns
    df["vol_return"] = df["ret_close"].rolling(window).std()

    # ATR (used both as feature and for label construction)
    df["atr"] = compute_atr(df, period=window)

    return df


def add_custom_composite_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create 1-2 custom composite indicators from basic signals.

    Idea:
    - trend_strength: ratio of close position within high-low range
    - momentum_vol_score: combine momentum and volatility into one score
    """
    df = df.copy()

    # Avoid division by zero
    range_hl = (df["high"] - df["low"]).replace(0, np.nan)

    # Trend strength: how close close is to high vs low
    df["trend_strength"] = ((df["close"] - df["low"]) / range_hl).clip(0, 1)

    # Momentum-volatility composite
    mom_norm = (df["mom_return"] - df["mom_return"].rolling(100).mean()) / (
        df["mom_return"].rolling(100).std() + 1e-8
    )
    vol_norm = (df["vol_return"] - df["vol_return"].rolling(100).mean()) / (
        df["vol_return"].rolling(100).std() + 1e-8
    )

    df["momentum_vol_score"] = 0.7 * mom_norm + 0.3 * vol_norm

    return df


# ==============================
# Label Construction (Fixed)
# ==============================


def build_trade_labels(
    df: pd.DataFrame,
    atr_cfg: ATRConfig,
    debug: bool = False,
) -> pd.Series:
    """Label each bar t as LONG, SHORT or FLAT.

    For each bar t (using price of bar t for entry):
    - Compute ATR_t
    - Long: entry at close_t, SL = close_t - ATR_t, TP = close_t + ATR_t * rr
    - Short: entry at close_t, SL = close_t + ATR_t, TP = close_t - ATR_t * rr

    We simulate future bars (t+1, t+2, ...) until either TP or SL is hit.
    Only TP hits result in a label (LONG or SHORT). Bars covered by an open
    trade are not allowed to open new trades (non-overlapping constraint).

    Outcome mapping:
    - If long TP hit first (before any SL): label LONG at t
    - If short TP hit first (before any SL): label SHORT at t
    - Else: FLAT
    """
    df = df.copy()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    n = len(df)

    labels = np.full(n, TradeLabel.FLAT, dtype=int)
    trade_end_indices = np.full(n, -1, dtype=int)

    i = 0
    debug_count = 0
    while i < n - 1:
        # Skip if already covered by previous trade
        if trade_end_indices[i] > i:
            i += 1
            continue

        if np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        entry = close[i]
        risk = atr_cfg.stop_atr * atr[i]
        reward = atr_cfg.rr_ratio * risk

        long_sl = entry - risk
        long_tp = entry + reward

        short_sl = entry + risk
        short_tp = entry - reward

        # Simulate future path to find TP or SL
        j = i + 1
        long_tp_hit = None
        long_sl_hit = None
        short_tp_hit = None
        short_sl_hit = None

        max_lookahead = min(50, n - i - 1)

        while j < i + 1 + max_lookahead:
            bar_high = high[j]
            bar_low = low[j]

            # Long: check TP first, then SL (TP takes priority if both in same bar)
            if long_tp_hit is None and bar_high >= long_tp:
                long_tp_hit = j
            elif long_sl_hit is None and bar_low <= long_sl:
                long_sl_hit = j

            # Short: check TP first, then SL
            if short_tp_hit is None and bar_low <= short_tp:
                short_tp_hit = j
            elif short_sl_hit is None and bar_high >= short_sl:
                short_sl_hit = j

            # Early exit: both sides have hit something
            if (long_tp_hit is not None or long_sl_hit is not None) and \
               (short_tp_hit is not None or short_sl_hit is not None):
                break

            j += 1

        # Decide label: only label LONG or SHORT if TP is hit before SL
        outcome = TradeLabel.FLAT
        trade_close_idx = None

        # Check if long TP hit before long SL
        long_is_profitable = long_tp_hit is not None and \
                             (long_sl_hit is None or long_tp_hit <= long_sl_hit)

        # Check if short TP hit before short SL
        short_is_profitable = short_tp_hit is not None and \
                              (short_sl_hit is None or short_tp_hit <= short_sl_hit)

        if long_is_profitable and short_is_profitable:
            # Both are profitable, pick the one that hits first
            if long_tp_hit <= short_tp_hit:
                outcome = TradeLabel.LONG
                trade_close_idx = long_tp_hit
            else:
                outcome = TradeLabel.SHORT
                trade_close_idx = short_tp_hit
        elif long_is_profitable:
            outcome = TradeLabel.LONG
            trade_close_idx = long_tp_hit
        elif short_is_profitable:
            outcome = TradeLabel.SHORT
            trade_close_idx = short_tp_hit

        labels[i] = int(outcome)

        if debug and debug_count < 10 and outcome != TradeLabel.FLAT:
            print(f"  Bar {i}: entry={entry:.2f}, atr={atr[i]:.2f}, "
                  f"long_tp={long_tp:.2f} (hit@{long_tp_hit}), "
                  f"long_sl={long_sl:.2f} (hit@{long_sl_hit}), "
                  f"short_tp={short_tp:.2f} (hit@{short_tp_hit}), "
                  f"short_sl={short_sl:.2f} (hit@{short_sl_hit}), "
                  f"label={TradeLabel(outcome).name}")
            debug_count += 1

        if outcome == TradeLabel.FLAT:
            i += 1
        else:
            # Mark the range of this trade as covered
            if trade_close_idx is not None:
                for idx in range(i, trade_close_idx + 1):
                    trade_end_indices[idx] = trade_close_idx
                i = trade_close_idx + 1
            else:
                i += 1

    return pd.Series(labels, index=df.index, name="label")


# ==============================
# Dataset Assembly
# ==============================


def build_feature_dataframe(
    df: pd.DataFrame,
    atr_cfg: ATRConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """End-to-end feature and label construction for a single symbol/timeframe."""
    print("  Computing momentum and volatility features...")
    df_feat = add_momentum_and_volatility_features(df)
    
    print("  Building custom composite indicators...")
    df_feat = add_custom_composite_indicators(df_feat)

    # Drop early rows with NaNs from rolling calculations
    df_feat = df_feat.dropna().copy()
    print(f"  After dropping NaNs: {len(df_feat)} bars")

    # Build labels
    print("  Constructing trade labels...")
    labels = build_trade_labels(df_feat, atr_cfg=atr_cfg, debug=True)

    # Align and drop rows without labels
    df_feat = df_feat.loc[labels.index]

    # Use previous bar features to predict next bar decision
    # Shift labels backward by 1: label at t corresponds to decision based on bar t-1
    labels_shifted = labels.shift(-1).dropna()
    df_feat = df_feat.loc[labels_shifted.index]

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "ret_close", "mom_return", "vol_return", "atr",
        "trend_strength", "momentum_vol_score",
    ]

    X = df_feat[feature_cols].copy()
    y = labels_shifted.astype(int)
    return X, y


# ==============================
# Model Training
# ==============================


def train_baseline_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Train a RandomForest classifier with class weight balancing."""
    print("  Splitting into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    print("  Training RandomForest (n_estimators=200, max_depth=8)...")
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
        verbose=1,
    )
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed:.2f}s")

    print("\n  Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print("="*60)
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[int(TradeLabel.FLAT), int(TradeLabel.LONG), int(TradeLabel.SHORT)],
            target_names=[
                TradeLabel.FLAT.name,
                TradeLabel.LONG.name,
                TradeLabel.SHORT.name,
            ],
            zero_division=0,
        )
    )
    
    print("\nConfusion Matrix:")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"           Pred FLAT  Pred LONG  Pred SHORT")
    print(f"Actual FLAT   {cm[0,0]:6d}    {cm[0,1]:6d}     {cm[0,2]:6d}")
    print(f"Actual LONG   {cm[1,0]:6d}    {cm[1,1]:6d}     {cm[1,2]:6d}")
    print(f"Actual SHORT  {cm[2,0]:6d}    {cm[2,1]:6d}     {cm[2,2]:6d}")
    
    print("\nFeature Importance (Top 10):")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:20s}: {row['importance']:.4f}")

    return model


# ==============================
# Entry Point
# ==============================


def main():
    symbol = "BTCUSDT"
    timeframe = "15m"

    print(f"Loading data for {symbol} {timeframe}...")
    df = load_klines(symbol, timeframe)

    atr_cfg = ATRConfig(period=14, stop_atr=1.0, rr_ratio=1.5)

    print("\nBuilding features and labels...")
    X, y = build_feature_dataframe(df, atr_cfg)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Label distribution:\n{y.value_counts().sort_index().to_string()}")
    print(f"Label percentages:")
    for label_val in [0, 1, 2]:
        pct = (y == label_val).sum() / len(y) * 100
        print(f"  {TradeLabel(label_val).name:5s}: {pct:6.2f}%")

    print("\nTraining baseline model...")
    _ = train_baseline_model(X, y)


if __name__ == "__main__":
    main()
