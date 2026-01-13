"""ML training pipeline for System V4 - New Label Strategy.

This script loads OHLCV data from the HuggingFace dataset,
builds engineered features with technical indicators,
constructs trade labels using improved logic,
and trains an XGBoost model to achieve 70%+ accuracy.

Improvements:
- Technical indicators: RSI, MACD, Bollinger Bands, ATR
- Improved label logic: trend-following + momentum
- Aggressive hyperparameter tuning
- Feature selection and normalization
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple
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
    SHORT = 0
    FLAT = 1
    LONG = 2


@dataclass
class LabelConfig:
    """Configuration for label construction."""
    future_bars: int = 10
    momentum_threshold: float = 0.005  # 0.5% momentum threshold
    volatility_threshold: float = 0.02  # 2% volatility threshold


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
# Technical Indicators
# ==============================


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD (Moving Average Convergence Divergence)."""
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
    """Compute Bollinger Bands."""
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


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


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators."""
    df = df.copy()
    
    # Basic features
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["price_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    
    # Momentum indicators
    df["rsi_14"] = compute_rsi(df, 14)
    df["rsi_7"] = compute_rsi(df, 7)
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["momentum_20"] = df["close"] - df["close"].shift(20)
    
    # MACD
    macd, signal, hist = compute_macd(df)
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    
    # Bollinger Bands
    upper, middle, lower = compute_bollinger_bands(df)
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    df["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-8)
    df["bb_width"] = (upper - lower) / middle
    
    # ATR
    df["atr"] = compute_atr(df, 14)
    df["atr_normalized"] = df["atr"] / df["close"]
    
    # Volume
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
    
    # Moving averages
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()
    
    # Trend
    df["trend_10"] = (df["sma_10"] - df["sma_20"]) / df["close"]
    df["trend_20"] = (df["sma_20"] - df["sma_50"]) / df["close"]
    
    # Volatility
    df["volatility_10"] = df["returns"].rolling(10).std()
    df["volatility_20"] = df["returns"].rolling(20).std()
    
    return df


# ==============================
# Label Construction - New Strategy
# ==============================


def build_improved_labels(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> pd.Series:
    """Label each bar using multi-factor logic.
    
    LONG: Strong uptrend + positive momentum
    SHORT: Strong downtrend + negative momentum
    FLAT: Sideways/uncertain
    """
    df = df.copy()
    n = len(df)
    future_bars = label_cfg.future_bars
    mom_thresh = label_cfg.momentum_threshold
    vol_thresh = label_cfg.volatility_threshold
    
    labels = np.full(n, TradeLabel.FLAT, dtype=int)
    
    for i in range(n - future_bars):
        entry = df["close"].iloc[i]
        future_slice = df.iloc[i + 1 : i + 1 + future_bars]
        
        future_high = future_slice["high"].max()
        future_low = future_slice["low"].min()
        future_close = future_slice["close"].iloc[-1]
        
        up_move = (future_high - entry) / entry
        down_move = (entry - future_low) / entry
        close_move = (future_close - entry) / entry
        
        # Current indicators
        current_rsi = df["rsi_14"].iloc[i]
        current_macd_hist = df["macd_hist"].iloc[i]
        current_trend = df["trend_10"].iloc[i]
        current_bb_pos = df["bb_position"].iloc[i]
        
        # Logic
        # Strong uptrend
        if (up_move > 0.02 and down_move < 0.01 and close_move > 0.005 and 
            current_rsi > 50 and current_macd_hist > 0 and current_trend > 0):
            labels[i] = TradeLabel.LONG
        # Strong downtrend
        elif (down_move > 0.02 and up_move < 0.01 and close_move < -0.005 and 
              current_rsi < 50 and current_macd_hist < 0 and current_trend < 0):
            labels[i] = TradeLabel.SHORT
        # Everything else
        else:
            labels[i] = TradeLabel.FLAT
    
    return pd.Series(labels, index=df.index, name="label")


# ==============================
# Dataset Assembly
# ==============================


def build_feature_dataframe(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """End-to-end feature and label construction."""
    print("  Adding technical indicators...")
    df_feat = add_technical_indicators(df)
    
    df_feat = df_feat.dropna().copy()
    print(f"  After dropping NaNs: {len(df_feat)} bars")
    
    print("  Constructing improved labels...")
    labels = build_improved_labels(df_feat, label_cfg=label_cfg)
    
    df_feat = df_feat.loc[labels.index]
    labels_shifted = labels.shift(-1).dropna()
    df_feat = df_feat.loc[labels_shifted.index]
    
    # Select best features
    feature_cols = [
        "rsi_14", "rsi_7", "momentum_10", "momentum_20",
        "macd", "macd_signal", "macd_hist",
        "bb_position", "bb_width",
        "atr_normalized",
        "volume_ratio",
        "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
        "trend_10", "trend_20",
        "volatility_10", "volatility_20",
        "returns", "log_returns",
        "price_range", "close_position",
    ]
    
    X = df_feat[feature_cols].copy()
    y = labels_shifted.astype(int)
    return X, y


# ==============================
# Model Training
# ==============================


def train_xgboost(X: pd.DataFrame, y: pd.Series):
    """Train XGBoost with aggressive tuning."""
    print("\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train label distribution:")
    for label_val in [0, 1, 2]:
        pct = (y_train == label_val).sum() / len(y_train) * 100
        print(f"  {TradeLabel(label_val).name:5s}: {pct:6.2f}%")
    
    # Feature normalization
    print("\n  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*70)
    print("TRAINING XGBoost (Aggressive)")
    print("="*70)
    
    # Aggressive hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0,
        min_child_weight=1,
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
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    if accuracy >= 0.70:
        print(f"\n✓ TARGET ACHIEVED: {accuracy*100:.2f}% >= 70%")
    else:
        print(f"\n✗ Target not reached: {accuracy*100:.2f}% < 70%")
    
    print(f"\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[int(TradeLabel.SHORT), int(TradeLabel.FLAT), int(TradeLabel.LONG)],
            target_names=[TradeLabel.SHORT.name, TradeLabel.FLAT.name, TradeLabel.LONG.name],
            zero_division=0,
        )
    )
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"            Pred SHORT  Pred FLAT  Pred LONG")
    print(f"Actual SHORT   {cm[0,0]:6d}    {cm[0,1]:6d}    {cm[0,2]:6d}")
    print(f"Actual FLAT    {cm[1,0]:6d}    {cm[1,1]:6d}    {cm[1,2]:6d}")
    print(f"Actual LONG    {cm[2,0]:6d}    {cm[2,1]:6d}    {cm[2,2]:6d}")
    
    print(f"\nFeature Importance (Top 15):")
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    for idx, row in feature_importance.head(15).iterrows():
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
    
    print("\nBuilding features and labels...")
    label_cfg = LabelConfig(future_bars=10, momentum_threshold=0.005, volatility_threshold=0.02)
    X, y = build_feature_dataframe(df, label_cfg)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Label distribution:")
    for label_val in [0, 1, 2]:
        pct = (y == label_val).sum() / len(y) * 100
        print(f"  {TradeLabel(label_val).name:5s}: {pct:6.2f}%")
    
    print("\nTraining model...")
    model, accuracy, f1_macro = train_xgboost(X, y)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
