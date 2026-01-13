"""ML training pipeline for System V4.

This script loads OHLCV data from the HuggingFace dataset
`zongowo111/v2-crypto-ohlcv-data`, builds engineered features
(momentum, volatility and custom composite indicators),
constructs trade labels based on Relative Strength (comparing upside vs downside moves),
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
class LabelConfig:
    """Configuration for Relative Strength label construction."""
    future_bars: int = 10
    ratio: float = 1.2  # ratio > 1: require upside/downside to be ratio*other_side to trigger


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

    # ATR (used as feature for volatility)
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
# Label Construction: Relative Strength
# ==============================


def build_relative_strength_labels(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
    debug: bool = False,
) -> pd.Series:
    """Label each bar t as LONG, SHORT or FLAT using Relative Strength.

    For each bar t:
    - Look at the next `future_bars` K bars
    - Calculate max upside move from current close: up_move = max(high) - close[t]
    - Calculate max downside move from current close: down_move = close[t] - min(low)
    - Compare them with a ratio threshold:
      - If up_move > down_move * ratio → LONG
      - If down_move > up_move * ratio → SHORT
      - Else → FLAT

    This generates much denser labels (40-50% LONG+SHORT) compared to ATR-based labels.

    Outcome mapping:
    - up_move significantly larger → label LONG at t
    - down_move significantly larger → label SHORT at t
    - Otherwise balanced → FLAT
    """
    df = df.copy()
    n = len(df)
    future_bars = label_cfg.future_bars
    ratio = label_cfg.ratio

    labels = np.full(n, TradeLabel.FLAT, dtype=int)

    for i in range(n - future_bars):
        entry = df["close"].iloc[i]

        # Look ahead at future bars
        future_slice = df.iloc[i + 1 : i + 1 + future_bars]
        future_high = future_slice["high"].max()
        future_low = future_slice["low"].min()

        # Calculate up and down moves
        up_move = future_high - entry
        down_move = entry - future_low

        # Avoid division by zero
        if down_move == 0 and up_move > 0:
            outcome = TradeLabel.LONG
        elif up_move == 0 and down_move > 0:
            outcome = TradeLabel.SHORT
        elif down_move == 0 and up_move == 0:
            outcome = TradeLabel.FLAT
        else:
            # Apply ratio threshold
            if up_move > down_move * ratio:
                outcome = TradeLabel.LONG
            elif down_move > up_move * ratio:
                outcome = TradeLabel.SHORT
            else:
                outcome = TradeLabel.FLAT

        labels[i] = int(outcome)

        if debug and i < 10:
            print(
                f"  Bar {i}: entry={entry:.2f}, "
                f"future_high={future_high:.2f}, future_low={future_low:.2f}, "
                f"up_move={up_move:.2f}, down_move={down_move:.2f}, "
                f"ratio_check={up_move / (down_move + 1e-8):.2f}, "
                f"label={TradeLabel(outcome).name}"
            )

    return pd.Series(labels, index=df.index, name="label")


# ==============================
# Dataset Assembly
# ==============================


def build_feature_dataframe(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """End-to-end feature and label construction for a single symbol/timeframe."""
    print("  Computing momentum and volatility features...")
    df_feat = add_momentum_and_volatility_features(df)

    print("  Building custom composite indicators...")
    df_feat = add_custom_composite_indicators(df_feat)

    # Drop early rows with NaNs from rolling calculations
    df_feat = df_feat.dropna().copy()
    print(f"  After dropping NaNs: {len(df_feat)} bars")

    # Build labels using Relative Strength
    print("  Constructing Relative Strength labels...")
    labels = build_relative_strength_labels(df_feat, label_cfg=label_cfg, debug=True)

    # Align and drop rows without labels
    df_feat = df_feat.loc[labels.index]

    # Use previous bar features to predict next bar decision
    # Shift labels backward by 1: label at t corresponds to decision based on bar t-1
    labels_shifted = labels.shift(-1).dropna()
    df_feat = df_feat.loc[labels_shifted.index]

    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_close",
        "mom_return",
        "vol_return",
        "atr",
        "trend_strength",
        "momentum_vol_score",
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
    print("=" * 60)
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
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"           Pred FLAT  Pred LONG  Pred SHORT")
    print(f"Actual FLAT   {cm[0,0]:6d}    {cm[0,1]:6d}     {cm[0,2]:6d}")
    print(f"Actual LONG   {cm[1,0]:6d}    {cm[1,1]:6d}     {cm[1,2]:6d}")
    print(f"Actual SHORT  {cm[2,0]:6d}    {cm[2,1]:6d}     {cm[2,2]:6d}")

    print("\nFeature Importance (Top 10):")
    print("=" * 60)
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

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

    label_cfg = LabelConfig(future_bars=10, ratio=1.2)

    print("\nBuilding features and labels...")
    X, y = build_feature_dataframe(df, label_cfg)
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
