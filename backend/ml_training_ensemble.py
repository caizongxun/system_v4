"""Ensemble ML Training Pipeline - Dual-Model Strategy.

Strategy:
1. Model A: Binary classifier (SHORT vs LONG) - ignores FLAT
2. Model B: Binary classifier (FLAT vs DIRECTION) - detects neutral zones
3. Fusion: Combine predictions via decision logic

Expected improvement: Accuracy 60-72% vs 54% single model
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


class BinaryLabel(IntEnum):
    """Binary labels for individual classifiers."""
    CLASS_0 = 0
    CLASS_1 = 1


@dataclass
class LabelConfig:
    """Configuration for ATR-relative label construction."""
    future_bars: int = 10
    atr_multiplier: float = 0.5
    neutral_atr_mult: float = 0.3  # 增加 FLAT 區間


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
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    df = df.copy()
    
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
    
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    if include_sma200:
        df["sma_200"] = df["close"].rolling(200).mean()
    
    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()
    
    df["trend_slope_10"] = (df["close"] - df["close"].shift(10)) / 10
    df["trend_slope_20"] = (df["close"] - df["close"].shift(20)) / 20
    df["sma_10_20_cross"] = df["sma_10"] - df["sma_20"]
    df["sma_20_50_cross"] = df["sma_20"] - df["sma_50"]
    if include_sma200:
        df["sma_50_200_cross"] = df["sma_50"] - df["sma_200"]
    
    df["rsi_14"] = compute_rsi(df, 14)
    df["rsi_7"] = compute_rsi(df, 7)
    
    macd, signal, hist = compute_macd(df)
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    
    high_20 = df["high"].rolling(20).max()
    low_20 = df["low"].rolling(20).min()
    df["position_in_range_20"] = (df["close"] - low_20) / (high_20 - low_20 + 1e-8)
    
    high_50 = df["high"].rolling(50).max()
    low_50 = df["low"].rolling(50).min()
    df["position_in_range_50"] = (df["close"] - low_50) / (high_50 - low_50 + 1e-8)
    
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    
    df["upper_wick_ratio"] = df["upper_wick"] / (df["body"] + 1e-8)
    df["lower_wick_ratio"] = df["lower_wick"] / (df["body"] + 1e-8)
    df["upper_wick_ratio"] = df["upper_wick_ratio"].clip(-2, 2)
    df["lower_wick_ratio"] = df["lower_wick_ratio"].clip(-2, 2)
    
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
    df["volume_ratio"] = df["volume_ratio"].clip(0.1, 3)
    
    df["up_volume"] = np.where(df["close"] > df["open"], df["volume"], 0)
    df["down_volume"] = np.where(df["close"] <= df["open"], df["volume"], 0)
    df["up_down_ratio"] = df["up_volume"].rolling(5).sum() / (df["down_volume"].rolling(5).sum() + 1e-8)
    df["up_down_ratio"] = df["up_down_ratio"].clip(0.1, 3)
    
    df["atr"] = compute_atr(df, 14)
    df["volatility"] = df["atr"] / (df["close"] + 1e-8)  # ATR 相對於價格的比率
    
    return df


def add_higher_timeframe_features(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    df_15m = df_15m.copy()
    print("  Computing 1h features...")
    df_1h_feat = add_direction_focused_features(df_1h, include_sma200=True)
    df_1h_resampled = df_1h_feat.reindex(df_15m.index, method='ffill')
    
    df_15m["sma_20_1h"] = df_1h_resampled["sma_20"]
    df_15m["sma_50_1h"] = df_1h_resampled["sma_50"]
    df_15m["sma_200_1h"] = df_1h_resampled["sma_200"]
    df_15m["rsi_14_1h"] = df_1h_resampled["rsi_14"]
    df_15m["macd_hist_1h"] = df_1h_resampled["macd_hist"]
    df_15m["trend_slope_20_1h"] = df_1h_resampled["trend_slope_20"]
    df_15m["volatility_1h"] = df_1h_resampled["volatility"]
    
    return df_15m


# ==============================
# Label Construction (Three-Class)
# ==============================


def build_three_class_labels(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> pd.Series:
    """Build 3-class labels: SHORT (0), FLAT (1), LONG (2)."""
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
        
        long_target = entry + (atr_val * atr_mult)
        short_target = entry - (atr_val * atr_mult)
        neutral_up = entry + (atr_val * neutral_mult)
        neutral_down = entry - (atr_val * neutral_mult)
        
        future_slice = df.iloc[i + 1 : i + 1 + future_bars]
        future_high = future_slice["high"].max()
        future_low = future_slice["low"].min()
        
        touches_long = future_high >= long_target
        touches_short = future_low <= short_target
        
        if touches_long and touches_short:
            up_dist = future_high - entry
            down_dist = entry - future_low
            labels[i] = TradeLabel.LONG if up_dist > down_dist else TradeLabel.SHORT
        elif touches_long:
            labels[i] = TradeLabel.LONG
        elif touches_short:
            labels[i] = TradeLabel.SHORT
        else:
            labels[i] = TradeLabel.FLAT
    
    return pd.Series(labels, index=df.index, name="label")


def derive_binary_labels(labels_3class: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Derive two binary labels from 3-class labels.
    
    Model A: SHORT (0) vs LONG (1) - ignores FLAT
    Model B: FLAT (1) vs DIRECTION (0) - merges SHORT+LONG as direction
    """
    # Model A: SHORT vs LONG (filter out FLAT)
    mask_a = labels_3class != TradeLabel.FLAT
    labels_a = labels_3class[mask_a].copy()
    labels_a = labels_a.map({TradeLabel.SHORT: 0, TradeLabel.LONG: 1})
    
    # Model B: FLAT vs DIRECTION
    labels_b = (labels_3class != TradeLabel.FLAT).astype(int)  # 0=FLAT, 1=DIRECTION
    
    return labels_a, labels_b, mask_a


# ==============================
# Dataset Assembly
# ==============================


def build_feature_dataframe(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    label_cfg: LabelConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build features and derive binary labels."""
    print("  Adding 15m features...")
    df_feat = add_direction_focused_features(df_15m, include_sma200=True)
    
    print("  Adding 1h context...")
    df_feat = add_higher_timeframe_features(df_feat, df_1h)
    
    df_feat = df_feat.dropna().copy()
    print(f"  After dropping NaNs: {len(df_feat)} bars")
    
    print("  Constructing 3-class labels...")
    labels_3class = build_three_class_labels(df_feat, label_cfg=label_cfg)
    
    df_feat = df_feat.loc[labels_3class.index].copy()
    labels_3class = labels_3class.loc[df_feat.index]
    
    # Derive binary labels
    print("  Deriving binary labels...")
    labels_a, labels_b, mask_a = derive_binary_labels(labels_3class)
    
    feature_cols = [
        "returns", "log_returns",
        "past_momentum_3", "past_momentum_5", "past_momentum_10", "past_momentum_20",
        "past_mom_3_norm", "past_mom_5_norm", "past_mom_10_norm",
        "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "trend_slope_10", "trend_slope_20",
        "sma_10_20_cross", "sma_20_50_cross", "sma_50_200_cross",
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_hist",
        "position_in_range_20", "position_in_range_50",
        "upper_wick_ratio", "lower_wick_ratio",
        "volume_ratio", "up_down_ratio",
        "volatility",
        "sma_20_1h", "sma_50_1h", "sma_200_1h",
        "rsi_14_1h",
        "macd_hist_1h",
        "trend_slope_20_1h",
        "volatility_1h",
    ]
    
    X = df_feat[feature_cols].copy()
    return X, labels_3class, labels_a, mask_a


# ==============================
# Ensemble Training
# ==============================


def train_model_a(X: pd.DataFrame, y_a: pd.Series):
    """Train Model A: SHORT vs LONG (binary)."""
    print("\n" + "="*70)
    print("MODEL A: SHORT vs LONG (Binary Classifier)")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_a, test_size=0.2, shuffle=False
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train - SHORT: {(y_train==0).sum()}, LONG: {(y_train==1).sum()}")
    print(f"Test  - SHORT: {(y_test==0).sum()}, LONG: {(y_test==1).sum()}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_a = xgb.XGBClassifier(
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
    
    print("\n  Training...")
    start = time.time()
    model_a.fit(X_train_scaled, y_train)
    print(f"  Completed in {time.time()-start:.2f}s")
    
    y_pred = model_a.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["SHORT", "LONG"], zero_division=0))
    
    return model_a, scaler, acc, f1


def train_model_b(X: pd.DataFrame, y_b: pd.Series):
    """Train Model B: FLAT vs DIRECTION (binary)."""
    print("\n" + "="*70)
    print("MODEL B: FLAT vs DIRECTION (Binary Classifier)")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_b, test_size=0.2, shuffle=False
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train - FLAT: {(y_train==0).sum()}, DIRECTION: {(y_train==1).sum()}")
    print(f"Test  - FLAT: {(y_test==0).sum()}, DIRECTION: {(y_test==1).sum()}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    sample_weights = compute_sample_weight('balanced', y_train)
    
    model_b = xgb.XGBClassifier(
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
    
    print("\n  Training (with sample weights)...")
    start = time.time()
    model_b.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    print(f"  Completed in {time.time()-start:.2f}s")
    
    y_pred = model_b.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["FLAT", "DIRECTION"], zero_division=0))
    
    return model_b, scaler, acc, f1


def ensemble_predict(X_test: pd.DataFrame, model_a, scaler_a, model_b, scaler_b) -> np.ndarray:
    """Ensemble prediction using both models."""
    X_a_scaled = scaler_a.transform(X_test)
    X_b_scaled = scaler_b.transform(X_test)
    
    pred_a = model_a.predict(X_a_scaled)  # 0=SHORT, 1=LONG
    pred_b = model_b.predict(X_b_scaled)  # 0=FLAT, 1=DIRECTION
    
    ensemble_preds = np.zeros(len(X_test), dtype=int)
    
    for i in range(len(X_test)):
        if pred_b[i] == 0:  # Model B says FLAT
            ensemble_preds[i] = TradeLabel.FLAT
        else:  # Model B says DIRECTION
            if pred_a[i] == 0:
                ensemble_preds[i] = TradeLabel.SHORT
            else:
                ensemble_preds[i] = TradeLabel.LONG
    
    return ensemble_preds


def main():
    symbol = "BTCUSDT"
    
    print(f"Loading data for {symbol}...")
    print("\n  Loading 15m data...")
    df_15m = load_klines(symbol, "15m")
    
    print("\n  Loading 1h data...")
    df_1h = load_klines(symbol, "1h")
    
    print("\nBuilding features...")
    label_cfg = LabelConfig(
        future_bars=10,
        atr_multiplier=0.5,
        neutral_atr_mult=0.3,
    )
    X, y_3class, y_a, mask_a = build_feature_dataframe(df_15m, df_1h, label_cfg)
    
    print(f"\nDataset shape: X={X.shape}")
    print(f"3-class label distribution:")
    for val in [0, 1, 2]:
        count = (y_3class == val).sum()
        pct = count / len(y_3class) * 100
        print(f"  {TradeLabel(val).name:5s}: {count:7d} ({pct:6.2f}%)")
    
    print(f"\nBinary label A (SHORT vs LONG, {mask_a.sum()} samples):")
    print(f"  SHORT: {(y_a==0).sum()}, LONG: {(y_a==1).sum()}")
    
    y_b_full = (y_3class != TradeLabel.FLAT).astype(int)
    print(f"\nBinary label B (FLAT vs DIRECTION, {len(y_b_full)} samples):")
    print(f"  FLAT: {(y_b_full==0).sum()}, DIRECTION: {(y_b_full==1).sum()}")
    
    # 訓練 Model A（只用有效樣本）
    X_a = X[mask_a].copy()
    print(f"\nTraining ensemble models...")
    model_a, scaler_a, acc_a, f1_a = train_model_a(X_a, y_a)
    
    # 訓練 Model B（所有樣本）
    model_b, scaler_b, acc_b, f1_b = train_model_b(X, y_b_full)
    
    # 評估集合模型
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION ON TEST SET")
    print("="*70)
    
    X_train, X_test, y_train_3c, y_test_3c = train_test_split(
        X, y_3class, test_size=0.2, shuffle=False
    )
    
    ensemble_preds = ensemble_predict(X_test, model_a, scaler_a, model_b, scaler_b)
    
    acc_ensemble = accuracy_score(y_test_3c, ensemble_preds)
    f1_ensemble = f1_score(y_test_3c, ensemble_preds, average='weighted', zero_division=0)
    
    print(f"\nEnsemble Accuracy: {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")
    print(f"Ensemble F1 (Weighted): {f1_ensemble:.4f}")
    
    if acc_ensemble >= 0.70:
        print(f"\n✓ TARGET ACHIEVED: {acc_ensemble*100:.2f}% >= 70%")
    else:
        print(f"\n✗ Target not reached: {acc_ensemble*100:.2f}% < 70%")
        print(f"  Gap: {(0.70 - acc_ensemble)*100:.2f}%")
    
    print(f"\nClassification Report (Ensemble):")
    print(classification_report(
        y_test_3c,
        ensemble_preds,
        labels=[0, 1, 2],
        target_names=["SHORT", "FLAT", "LONG"],
        zero_division=0,
    ))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_3c, ensemble_preds)
    print(f"            Pred SHORT  Pred FLAT  Pred LONG")
    for i, label in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"Actual {label:5s}   {cm[i,0]:6d}    {cm[i,1]:6d}    {cm[i,2]:6d}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model A (SHORT vs LONG):     {acc_a*100:6.2f}% (F1: {f1_a:.4f})")
    print(f"Model B (FLAT vs DIRECTION): {acc_b*100:6.2f}% (F1: {f1_b:.4f})")
    print(f"Ensemble (3-class):          {acc_ensemble*100:6.2f}% (F1: {f1_ensemble:.4f})")
    print("="*70)


if __name__ == "__main__":
    main()
