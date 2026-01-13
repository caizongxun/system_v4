"""ML training pipeline for System V4 - Optimized Labels & Features.

This script loads OHLCV data from the HuggingFace dataset,
builds direction-focused features,
constructs trade labels using ATR-relative approach for universal applicability,
and trains an XGBoost model.

Key improvements:
- ATR-relative labels: Works across all coin types and timeframes
- Direction-focused features: Momentum, position, K-line structure
- Reduced noise from pure volatility indicators
- Feature importance ranking for interpretability
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
    """Configuration for ATR-relative label construction."""
    future_bars: int = 10
    atr_multiplier: float = 0.5  # ATR × 0.5 for target threshold


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
# Technical Indicators - Direction Focused
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


def add_direction_focused_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction-focused features, minimize noise from pure volatility."""
    df = df.copy()
    
    # ==================== Momentum Features (Direction Signal) ====================
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    
    # Past momentum - absolute price change (not %, universal across coins)
    df["past_momentum_3"] = df["close"] - df["close"].shift(3)
    df["past_momentum_5"] = df["close"] - df["close"].shift(5)
    df["past_momentum_10"] = df["close"] - df["close"].shift(10)
    df["past_momentum_20"] = df["close"] - df["close"].shift(20)
    
    # Normalize past momentum by recent price range
    recent_range = df["high"].rolling(10).max() - df["low"].rolling(10).min()
    df["past_mom_3_norm"] = df["past_momentum_3"] / (recent_range + 1e-8)
    df["past_mom_5_norm"] = df["past_momentum_5"] / (recent_range + 1e-8)
    df["past_mom_10_norm"] = df["past_momentum_10"] / (recent_range + 1e-8)
    
    # ==================== Trend Features ====================
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()
    
    # Trend slopes (direction indicator)
    df["trend_slope_10"] = (df["close"] - df["close"].shift(10)) / 10  # Average slope over 10 bars
    df["trend_slope_20"] = (df["close"] - df["close"].shift(20)) / 20
    df["sma_10_20_cross"] = df["sma_10"] - df["sma_20"]  # Crossover signal
    df["sma_20_50_cross"] = df["sma_20"] - df["sma_50"]
    
    # ==================== Momentum Indicators ====================
    df["rsi_14"] = compute_rsi(df, 14)
    df["rsi_7"] = compute_rsi(df, 7)
    
    # MACD (strong direction signal)
    macd, signal, hist = compute_macd(df)
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    
    # ==================== Position Relative to Recent Range ====================
    high_20 = df["high"].rolling(20).max()
    low_20 = df["low"].rolling(20).min()
    df["position_in_range_20"] = (df["close"] - low_20) / (high_20 - low_20 + 1e-8)  # 0=at low, 1=at high
    
    high_50 = df["high"].rolling(50).max()
    low_50 = df["low"].rolling(50).min()
    df["position_in_range_50"] = (df["close"] - low_50) / (high_50 - low_50 + 1e-8)
    
    # ==================== K-line Structure (Wick patterns) ====================
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    
    # Wick ratios (clipped to avoid extremes)
    df["upper_wick_ratio"] = df["upper_wick"] / (df["body"] + 1e-8)
    df["lower_wick_ratio"] = df["lower_wick"] / (df["body"] + 1e-8)
    df["upper_wick_ratio"] = df["upper_wick_ratio"].clip(-2, 2)
    df["lower_wick_ratio"] = df["lower_wick_ratio"].clip(-2, 2)
    
    # ==================== Volume-Price Relationship ====================
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
    df["volume_ratio"] = df["volume_ratio"].clip(0.1, 3)  # Clip extremes
    
    # Volume and direction relationship
    df["up_volume"] = np.where(df["close"] > df["open"], df["volume"], 0)
    df["down_volume"] = np.where(df["close"] <= df["open"], df["volume"], 0)
    df["up_down_ratio"] = df["up_volume"].rolling(5).sum() / (df["down_volume"].rolling(5).sum() + 1e-8)
    df["up_down_ratio"] = df["up_down_ratio"].clip(0.1, 3)
    
    # ==================== ATR for label generation (not as feature) ====================
    df["atr"] = compute_atr(df, 14)
    
    return df


# ==============================
# Label Construction - ATR Relative
# ==============================


def build_atr_relative_labels(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> pd.Series:
    """Label each bar using ATR-relative thresholds.
    
    This approach is universal across all coin types and timeframes.
    - LONG:  future_close > entry + (ATR * multiplier)
    - SHORT: future_close < entry - (ATR * multiplier)
    - FLAT:  everything else
    """
    df = df.copy()
    n = len(df)
    future_bars = label_cfg.future_bars
    atr_mult = label_cfg.atr_multiplier
    
    labels = np.full(n, TradeLabel.FLAT, dtype=int)
    
    for i in range(n - future_bars):
        entry = df["close"].iloc[i]
        atr_val = df["atr"].iloc[i]
        
        if pd.isna(atr_val) or atr_val == 0:
            labels[i] = TradeLabel.FLAT
            continue
        
        # Target thresholds
        long_target = entry + (atr_val * atr_mult)
        short_target = entry - (atr_val * atr_mult)
        
        # Check future bars
        future_slice = df.iloc[i + 1 : i + 1 + future_bars]
        future_high = future_slice["high"].max()
        future_low = future_slice["low"].min()
        
        # Label based on which target is hit first
        if future_high >= long_target and future_low <= short_target:
            # Both hit - take which comes first
            # For now, use the magnitude
            up_dist = future_high - entry
            down_dist = entry - future_low
            labels[i] = TradeLabel.LONG if up_dist > down_dist else TradeLabel.SHORT
        elif future_high >= long_target:
            labels[i] = TradeLabel.LONG
        elif future_low <= short_target:
            labels[i] = TradeLabel.SHORT
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
    print("  Adding direction-focused indicators...")
    df_feat = add_direction_focused_features(df)
    
    df_feat = df_feat.dropna().copy()
    print(f"  After dropping NaNs: {len(df_feat)} bars")
    
    print("  Constructing ATR-relative labels...")
    labels = build_atr_relative_labels(df_feat, label_cfg=label_cfg)
    
    # Align features and labels
    df_feat = df_feat.loc[labels.index].copy()
    labels = labels.loc[df_feat.index]
    
    # Select direction-focused features only
    feature_cols = [
        # Momentum
        "returns", "log_returns",
        "past_momentum_3", "past_momentum_5", "past_momentum_10", "past_momentum_20",
        "past_mom_3_norm", "past_mom_5_norm", "past_mom_10_norm",
        # Trend
        "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26",
        "trend_slope_10", "trend_slope_20",
        "sma_10_20_cross", "sma_20_50_cross",
        # Momentum indicators
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_hist",
        # Position
        "position_in_range_20", "position_in_range_50",
        # K-line structure
        "upper_wick_ratio", "lower_wick_ratio", "body",
        # Volume
        "volume_ratio", "up_down_ratio",
    ]
    
    X = df_feat[feature_cols].copy()
    y = labels.astype(int)
    return X, y


# ==============================
# Model Training
# ==============================


def train_xgboost(X: pd.DataFrame, y: pd.Series):
    """Train XGBoost with optimized hyperparameters."""
    print("\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train label distribution:")
    for label_val in [0, 1, 2]:
        count = (y_train == label_val).sum()
        pct = count / len(y_train) * 100
        print(f"  {TradeLabel(label_val).name:5s}: {count:7d} ({pct:6.2f}%)")
    
    print(f"\nTest label distribution:")
    for label_val in [0, 1, 2]:
        count = (y_test == label_val).sum()
        pct = count / len(y_test) * 100
        print(f"  {TradeLabel(label_val).name:5s}: {count:7d} ({pct:6.2f}%)")
    
    # Feature normalization
    print("\n  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*70)
    print("TRAINING XGBoost")
    print("="*70)
    
    # Optimized hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.5,
        min_child_weight=2,
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
        print(f"  Gap to 70%: {(0.70 - accuracy)*100:.2f}%")
    
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
    
    print(f"\nFeature Importance (Top 20):")
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    return model, accuracy, f1_macro, f1_weighted


# ==============================
# Entry Point
# ==============================


def main():
    symbol = "BTCUSDT"
    timeframe = "15m"
    
    print(f"Loading data for {symbol} {timeframe}...")
    df = load_klines(symbol, timeframe)
    
    print("\nBuilding features and labels...")
    label_cfg = LabelConfig(future_bars=10, atr_multiplier=0.5)
    X, y = build_feature_dataframe(df, label_cfg)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Label distribution (ATR-relative, universal):")
    for label_val in [0, 1, 2]:
        count = (y == label_val).sum()
        pct = count / len(y) * 100
        print(f"  {TradeLabel(label_val).name:5s}: {count:7d} ({pct:6.2f}%)")
    
    print("\nTraining model...")
    model, accuracy, f1_macro, f1_weighted = train_xgboost(X, y)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
