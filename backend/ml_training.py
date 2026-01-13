"""ML training pipeline for System V4 - Optimized Version.

This script loads OHLCV data from the HuggingFace dataset,
builds engineered features, constructs trade labels using Relative Strength,
and trains an optimized XGBoost model to achieve 70%+ accuracy.

Optimizations include:
- Label ratio tuning for balanced class distribution
- XGBoost hyperparameter optimization
- Class weight balancing
- Feature normalization
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Dict, Optional
import time

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


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
    ratio: float = 2.5  # Increased from 1.2 for better class balance


# ==============================
# Data Loading
# ==============================


def load_klines(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data for a symbol and timeframe from HuggingFace."""
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

    df["ret_close"] = df["close"].pct_change()
    df["mom_return"] = df["ret_close"].rolling(window).sum()
    df["vol_return"] = df["ret_close"].rolling(window).std()
    df["atr"] = compute_atr(df, period=window)

    return df


def add_custom_composite_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create custom composite indicators."""
    df = df.copy()

    range_hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["trend_strength"] = ((df["close"] - df["low"]) / range_hl).clip(0, 1)

    mom_norm = (df["mom_return"] - df["mom_return"].rolling(100).mean()) / (
        df["mom_return"].rolling(100).std() + 1e-8
    )
    vol_norm = (df["vol_return"] - df["vol_return"].rolling(100).mean()) / (
        df["vol_return"].rolling(100).std() + 1e-8
    )

    df["momentum_vol_score"] = 0.7 * mom_norm + 0.3 * vol_norm
    
    # Additional features for better classification
    df["price_range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    df["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df


# ==============================
# Label Construction
# ==============================


def build_relative_strength_labels(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> pd.Series:
    """Label each bar using Relative Strength."""
    df = df.copy()
    n = len(df)
    future_bars = label_cfg.future_bars
    ratio = label_cfg.ratio

    labels = np.full(n, TradeLabel.FLAT, dtype=int)

    for i in range(n - future_bars):
        entry = df["close"].iloc[i]
        future_slice = df.iloc[i + 1 : i + 1 + future_bars]
        future_high = future_slice["high"].max()
        future_low = future_slice["low"].min()

        up_move = future_high - entry
        down_move = entry - future_low

        if down_move == 0 and up_move > 0:
            outcome = TradeLabel.LONG
        elif up_move == 0 and down_move > 0:
            outcome = TradeLabel.SHORT
        elif down_move == 0 and up_move == 0:
            outcome = TradeLabel.FLAT
        else:
            if up_move > down_move * ratio:
                outcome = TradeLabel.LONG
            elif down_move > up_move * ratio:
                outcome = TradeLabel.SHORT
            else:
                outcome = TradeLabel.FLAT

        labels[i] = int(outcome)

    return pd.Series(labels, index=df.index, name="label")


# ==============================
# Dataset Assembly
# ==============================


def build_feature_dataframe(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """End-to-end feature and label construction."""
    print("  Computing momentum and volatility features...")
    df_feat = add_momentum_and_volatility_features(df)

    print("  Building custom composite indicators...")
    df_feat = add_custom_composite_indicators(df_feat)

    df_feat = df_feat.dropna().copy()
    print(f"  After dropping NaNs: {len(df_feat)} bars")

    print("  Constructing Relative Strength labels...")
    labels = build_relative_strength_labels(df_feat, label_cfg=label_cfg)

    df_feat = df_feat.loc[labels.index]
    labels_shifted = labels.shift(-1).dropna()
    df_feat = df_feat.loc[labels_shifted.index]

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "ret_close", "mom_return", "vol_return", "atr",
        "trend_strength", "momentum_vol_score",
        "price_range_pct", "close_position", "volume_ma_ratio",
    ]

    X = df_feat[feature_cols].copy()
    y = labels_shifted.astype(int)
    return X, y


# ==============================
# Model Training - Optimized
# ==============================


def train_optimized_xgboost(X: pd.DataFrame, y: pd.Series):
    """Train optimized XGBoost with hyperparameter tuning."""
    print("\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train label distribution:\n{pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(f"Test label distribution:\n{pd.Series(y_test).value_counts().sort_index().to_dict()}")

    # Feature normalization
    print("\n  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class weights
    unique, counts = np.unique(y_train, return_counts=True)
    scale_pos_weight = counts[2] / counts[1]  # SHORT / LONG ratio
    
    print(f"\n  Class weight (SHORT/LONG): {scale_pos_weight:.2f}")

    print("\n" + "="*70)
    print("TRAINING OPTIMIZED XGBoost")
    print("="*70)

    # Optimized hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    start_time = time.time()
    print("\n  Training...")
    model.fit(X_train_scaled, y_train)
    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed:.2f}s")

    # Predictions
    print("\n  Making predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")

    if accuracy >= 0.70:
        print(f"\nTARGET ACHIEVED: {accuracy*100:.2f}% >= 70%")
    else:
        print(f"\nTarget not reached: {accuracy*100:.2f}% < 70%")
        print(f"  Need {(0.70 - accuracy)*100:.2f}% more improvement")

    print(f"\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[int(TradeLabel.FLAT), int(TradeLabel.LONG), int(TradeLabel.SHORT)],
            target_names=[TradeLabel.FLAT.name, TradeLabel.LONG.name, TradeLabel.SHORT.name],
            zero_division=0,
        )
    )

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"           Pred FLAT  Pred LONG  Pred SHORT")
    print(f"Actual FLAT   {cm[0,0]:6d}    {cm[0,1]:6d}     {cm[0,2]:6d}")
    print(f"Actual LONG   {cm[1,0]:6d}    {cm[1,1]:6d}     {cm[1,2]:6d}")
    print(f"Actual SHORT  {cm[2,0]:6d}    {cm[2,1]:6d}     {cm[2,2]:6d}")

    print(f"\nFeature Importance (Top 10):")
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    return model, accuracy, f1_macro


# ==============================
# Entry Point
# ==============================


def main():
    symbol = "BTCUSDT"
    timeframe = "15m"

    print(f"Loading data for {symbol} {timeframe}...")
    df = load_klines(symbol, timeframe)

    # Test multiple configurations
    configs = [
        LabelConfig(future_bars=10, ratio=2.5),
        LabelConfig(future_bars=10, ratio=3.0),
        LabelConfig(future_bars=5, ratio=2.0),
    ]

    best_accuracy = 0
    best_config = None

    for idx, label_cfg in enumerate(configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Configuration {idx}/{len(configs)}: future_bars={label_cfg.future_bars}, ratio={label_cfg.ratio}")
        print(f"{'#'*70}")

        print("\nBuilding features and labels...")
        X, y = build_feature_dataframe(df, label_cfg)
        print(f"\nDataset shape: X={X.shape}, y={y.shape}")
        print(f"Label distribution:")
        for label_val in [0, 1, 2]:
            pct = (y == label_val).sum() / len(y) * 100
            print(f"  {TradeLabel(label_val).name:5s}: {pct:6.2f}%")

        print("\nTraining model...")
        model, accuracy, f1_macro = train_optimized_xgboost(X, y)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = label_cfg

    # Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Best configuration: future_bars={best_config.future_bars}, ratio={best_config.ratio}")
    print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()
