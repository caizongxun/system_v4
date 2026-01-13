"""ML training pipeline for System V4 - Enhanced with Sample Weights & Multi-Timeframe.

Key improvements:
1. Improved label generation logic (less strict thresholds)
2. Class rebalancing via threshold adjustment
3. Sample weight for multi-class imbalance handling
4. SMA_200 for long-term trend context
5. Multi-timeframe (1h) features for higher-level confirmation
6. ATR-relative labels universal across coins

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
from sklearn.utils.class_weight import compute_sample_weight

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
    atr_multiplier: float = 0.5      # 用於定義進場目標 (ATR × 倍數)
    neutral_atr_mult: float = 0.25   # FLAT 區間：±0.25 × ATR


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


def add_direction_focused_features(df: pd.DataFrame, include_sma200: bool = True) -> pd.DataFrame:
    """Add direction-focused features."""
    df = df.copy()
    
    # ==================== Momentum Features ====================
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    
    df["past_momentum_3"] = df["close"] - df["close"].shift(3)
    df["past_momentum_5"] = df["close"] - df["close"].shift(5)
    df["past_momentum_10"] = df["close"] - df["close"].shift(10)
    df["past_momentum_20"] = df["close"] - df["close"].shift(20)
    
    recent_range = df["high"].rolling(10).max() - df["low"].rolling(10).min()
    df["past_mom_3_norm"] = df["past_momentum_3"] / (recent_range + 1e-8)
    df["past_mom_5_norm"] = df["past_momentum_5"] / (recent_range + 1e-8)
    df["past_mom_10_norm"] = df["past_momentum_10"] / (recent_range + 1e-8)
    
    # ==================== Trend Features ====================
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    if include_sma200:
        df["sma_200"] = df["close"].rolling(200).mean()  # LONG-TERM TREND
    
    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()
    
    df["trend_slope_10"] = (df["close"] - df["close"].shift(10)) / 10
    df["trend_slope_20"] = (df["close"] - df["close"].shift(20)) / 20
    df["sma_10_20_cross"] = df["sma_10"] - df["sma_20"]
    df["sma_20_50_cross"] = df["sma_20"] - df["sma_50"]
    if include_sma200:
        df["sma_50_200_cross"] = df["sma_50"] - df["sma_200"]  # Long-term crossover
    
    # ==================== Momentum Indicators ====================
    df["rsi_14"] = compute_rsi(df, 14)
    df["rsi_7"] = compute_rsi(df, 7)
    
    macd, signal, hist = compute_macd(df)
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    
    # ==================== Position Relative to Recent Range ====================
    high_20 = df["high"].rolling(20).max()
    low_20 = df["low"].rolling(20).min()
    df["position_in_range_20"] = (df["close"] - low_20) / (high_20 - low_20 + 1e-8)
    
    high_50 = df["high"].rolling(50).max()
    low_50 = df["low"].rolling(50).min()
    df["position_in_range_50"] = (df["close"] - low_50) / (high_50 - low_50 + 1e-8)
    
    # ==================== K-line Structure ====================
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    
    df["upper_wick_ratio"] = df["upper_wick"] / (df["body"] + 1e-8)
    df["lower_wick_ratio"] = df["lower_wick"] / (df["body"] + 1e-8)
    df["upper_wick_ratio"] = df["upper_wick_ratio"].clip(-2, 2)
    df["lower_wick_ratio"] = df["lower_wick_ratio"].clip(-2, 2)
    
    # ==================== Volume-Price Relationship ====================
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
    df["volume_ratio"] = df["volume_ratio"].clip(0.1, 3)
    
    df["up_volume"] = np.where(df["close"] > df["open"], df["volume"], 0)
    df["down_volume"] = np.where(df["close"] <= df["open"], df["volume"], 0)
    df["up_down_ratio"] = df["up_volume"].rolling(5).sum() / (df["down_volume"].rolling(5).sum() + 1e-8)
    df["up_down_ratio"] = df["up_down_ratio"].clip(0.1, 3)
    
    # ==================== ATR for label generation ====================
    df["atr"] = compute_atr(df, 14)
    
    return df


def add_higher_timeframe_features(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """Add 1h features as context for 15m predictions."""
    df_15m = df_15m.copy()
    
    # 1h 資料也需要先計算特徵
    print("  Computing 1h features...")
    df_1h_feat = add_direction_focused_features(df_1h, include_sma200=True)
    
    # Resample 1h data to align with 15m index (forward fill 因為 1h bar 涵蓋整個小時)
    df_1h_resampled = df_1h_feat.reindex(df_15m.index, method='ffill')
    
    # Add 1h trend context
    df_15m["sma_20_1h"] = df_1h_resampled["sma_20"]
    df_15m["sma_50_1h"] = df_1h_resampled["sma_50"]
    df_15m["sma_200_1h"] = df_1h_resampled["sma_200"]
    df_15m["rsi_14_1h"] = df_1h_resampled["rsi_14"]
    df_15m["macd_hist_1h"] = df_1h_resampled["macd_hist"]
    df_15m["trend_slope_20_1h"] = df_1h_resampled["trend_slope_20"]
    
    return df_15m


# ==============================
# Label Construction - Improved Logic
# ==============================


def build_improved_labels(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> pd.Series:
    """Improved label construction with better class distribution.
    
    邏輯：
    - 如果未來 future_bars 內同時觸及上下目標 → 比較誰先到 (direction)
    - 如果只觸及上目標 → LONG
    - 如果只觸及下目標 → SHORT
    - 如果都沒觸及 → FLAT (而不是之前的忽略)
    - 如果移動距離在 ±neutral 區間內 → FLAT
    """
    df = df.copy()
    n = len(df)
    future_bars = label_cfg.future_bars
    atr_mult = label_cfg.atr_multiplier
    neutral_mult = label_cfg.neutral_atr_mult
    
    labels = np.full(n, TradeLabel.FLAT, dtype=int)
    
    for i in range(n - future_bars):
        entry = df["close"].iloc[i]
        atr_val = df["atr"].iloc[i]
        
        if pd.isna(atr_val) or atr_val < 1e-6:
            labels[i] = TradeLabel.FLAT
            continue
        
        # 定義目標
        long_target = entry + (atr_val * atr_mult)
        short_target = entry - (atr_val * atr_mult)
        neutral_up = entry + (atr_val * neutral_mult)
        neutral_down = entry - (atr_val * neutral_mult)
        
        # 未來 future_bars 內的價格範圍
        future_slice = df.iloc[i + 1 : i + 1 + future_bars]
        future_high = future_slice["high"].max()
        future_low = future_slice["low"].min()
        
        # 檢查觸及情況
        touches_long = future_high >= long_target
        touches_short = future_low <= short_target
        
        if touches_long and touches_short:
            # 同時觸及 → 誰先到
            up_dist = future_high - entry
            down_dist = entry - future_low
            labels[i] = TradeLabel.LONG if up_dist > down_dist else TradeLabel.SHORT
        elif touches_long:
            labels[i] = TradeLabel.LONG
        elif touches_short:
            labels[i] = TradeLabel.SHORT
        else:
            # 都沒觸及 → FLAT (改善！)
            # 但進一步細分：如果在中立區間內 → 保留 FLAT，否則也是 FLAT
            labels[i] = TradeLabel.FLAT
    
    return pd.Series(labels, index=df.index, name="label")


# ==============================
# Dataset Assembly
# ==============================


def build_feature_dataframe(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    label_cfg: LabelConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """End-to-end feature and label construction."""
    print("  Adding 15m direction-focused indicators...")
    df_feat = add_direction_focused_features(df_15m, include_sma200=True)
    
    print("  Adding 1h context features...")
    df_feat = add_higher_timeframe_features(df_feat, df_1h)
    
    df_feat = df_feat.dropna().copy()
    print(f"  After dropping NaNs: {len(df_feat)} bars")
    
    print("  Constructing improved labels...")
    labels = build_improved_labels(df_feat, label_cfg=label_cfg)
    
    df_feat = df_feat.loc[labels.index].copy()
    labels = labels.loc[df_feat.index]
    
    # Select direction-focused features
    feature_cols = [
        # 15m Momentum
        "returns", "log_returns",
        "past_momentum_3", "past_momentum_5", "past_momentum_10", "past_momentum_20",
        "past_mom_3_norm", "past_mom_5_norm", "past_mom_10_norm",
        # 15m Trend
        "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "trend_slope_10", "trend_slope_20",
        "sma_10_20_cross", "sma_20_50_cross", "sma_50_200_cross",
        # 15m Momentum Indicators
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_hist",
        # 15m Position & Structure
        "position_in_range_20", "position_in_range_50",
        "upper_wick_ratio", "lower_wick_ratio",
        # 15m Volume
        "volume_ratio", "up_down_ratio",
        # 1h Context (Higher Timeframe)
        "sma_20_1h", "sma_50_1h", "sma_200_1h",
        "rsi_14_1h",
        "macd_hist_1h",
        "trend_slope_20_1h",
    ]
    
    X = df_feat[feature_cols].copy()
    y = labels.astype(int)
    return X, y


# ==============================
# Model Training with Sample Weights
# ==============================


def train_xgboost(X: pd.DataFrame, y: pd.Series):
    """Train XGBoost with sample weights for multi-class imbalance."""
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
        pct = count / len(y_train) * 100 if len(y_train) > 0 else 0
        print(f"  {TradeLabel(label_val).name:5s}: {count:7d} ({pct:6.2f}%)")
    
    print(f"\nTest label distribution:")
    for label_val in [0, 1, 2]:
        count = (y_test == label_val).sum()
        pct = count / len(y_test) * 100 if len(y_test) > 0 else 0
        print(f"  {TradeLabel(label_val).name:5s}: {count:7d} ({pct:6.2f}%)")
    
    # Feature normalization
    print("\n  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate sample weights for multi-class imbalance
    print("\n  Computing sample weights for multi-class handling...")
    sample_weights_train = compute_sample_weight('balanced', y_train)
    print(f"  Sample weight range: [{sample_weights_train.min():.3f}, {sample_weights_train.max():.3f}]")
    
    print("\n" + "="*70)
    print("TRAINING XGBoost (with sample weights + SMA_200 + 1h context)")
    print("="*70)
    
    # Optimized hyperparameters with scale_pos_weight handling
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
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed:.2f}s")
    
    # Predictions
    print("\n  Making predictions...")
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
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
    for i, label_name in enumerate([TradeLabel.SHORT.name, TradeLabel.FLAT.name, TradeLabel.LONG.name]):
        print(f"Actual {label_name:5s}   {cm[i,0]:6d}    {cm[i,1]:6d}    {cm[i,2]:6d}")
    
    print(f"\nFeature Importance (Top 20):")
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    return model, accuracy, f1_macro, f1_weighted


# ==============================
# Entry Point
# ==============================


def main():
    symbol = "BTCUSDT"
    
    print(f"Loading data for {symbol}...")
    print("\n  Loading 15m data...")
    df_15m = load_klines(symbol, "15m")
    
    print("\n  Loading 1h data (for context)...")
    df_1h = load_klines(symbol, "1h")
    
    print("\nBuilding features and labels...")
    label_cfg = LabelConfig(
        future_bars=10,
        atr_multiplier=0.5,
        neutral_atr_mult=0.25,
    )
    X, y = build_feature_dataframe(df_15m, df_1h, label_cfg)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Label distribution (improved):")
    for label_val in [0, 1, 2]:
        count = (y == label_val).sum()
        pct = count / len(y) * 100 if len(y) > 0 else 0
        print(f"  {TradeLabel(label_val).name:5s}: {count:7d} ({pct:6.2f}%)")
    
    print("\nTraining model...")
    model, accuracy, f1_macro, f1_weighted = train_xgboost(X, y)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
