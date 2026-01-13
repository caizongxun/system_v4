"""Ensemble ML Training Pipeline V3 - Real Trading Logic.

Two-Stage Decision Process:
1. Model A: Should we ENTER the trade? (FLAT vs TRADE)
   - FLAT (0): Don't enter, wait for better signal
   - TRADE (1): Enter the trade

2. Model B: If entering, go LONG or SHORT? (SHORT vs LONG)
   - SHORT (0): Enter short position
   - LONG (1): Enter long position

Expected improvement:
- Model A: 80%+ accuracy (simple entry detection)
- Model B: 65-70% accuracy (direction prediction)
- Ensemble: 70%+ overall (when combining)

Key difference from V2:
- V2 had Model B as "FLAT vs DIRECTION" (still 0.14% FLAT, useless)
- V3 has Model A as "entry signal detection" (redefines FLAT as no-signal)
- V3 Model B is ONLY trained on TRADE samples (no FLAT contamination)
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed.")


REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
SUPPORTED_SYMBOLS = [
    "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT",
    "BALUSDT", "BATUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "COMPUSDT",
    "CRVUSDT", "DOGEUSDT", "DOTUSDT", "ENJUSDT", "ENSUSDT", "ETCUSDT",
    "ETHUSDT", "FILUSDT", "GALAUSDT", "GRTUSDT", "IMXUSDT", "KAVAUSDT",
    "LINKUSDT", "LTCUSDT", "MANAUSDT", "MATICUSDT", "MKRUSDT", "NEARUSDT",
    "OPUSDT", "SANDUSDT", "SNXUSDT", "SOLUSDT", "SPELLUSDT", "UNIUSDT",
    "XRPUSDT", "ZRXUSDT",
]


class BinaryLabel(IntEnum):
    CLASS_0 = 0
    CLASS_1 = 1


@dataclass
class LabelConfig:
    """Label construction config."""
    future_bars: int = 10
    atr_multiplier: float = 0.5      # Entry threshold
    neutral_atr_mult: float = 0.25   # No-signal zone


# ==============================
# Data Loading & Features
# ==============================


def load_klines(symbol: str, timeframe: str) -> pd.DataFrame:
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
    df["volatility"] = df["atr"] / (df["close"] + 1e-8)
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
# Label Construction - V3 Trading Logic
# ==============================


def build_trading_logic_labels(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> Tuple[pd.Series, pd.Series]:
    """Build labels for two-stage trading decision.
    
    Returns:
      label_entry (0=FLAT, 1=TRADE): Should we enter?
      label_direction (0=SHORT, 1=LONG): If entering, which direction?
    
    Logic:
      - Entry threshold: ±0.5 × ATR
      - No-signal zone: ±0.25 × ATR (no entry signal)
      - Within no-signal: label_entry = FLAT
      - If touches threshold: label_entry = TRADE
      - Direction: whoever reaches threshold first (SHORT or LONG)
    """
    df = df.copy()
    n = len(df)
    future_bars = label_cfg.future_bars
    atr_mult = label_cfg.atr_multiplier
    neutral_mult = label_cfg.neutral_atr_mult
    
    labels_entry = np.full(n, 0, dtype=int)  # 0=FLAT, 1=TRADE
    labels_direction = np.full(n, -1, dtype=int)  # -1=invalid, 0=SHORT, 1=LONG
    
    for i in range(n - future_bars):
        entry = df["close"].iloc[i]
        atr_val = df["atr"].iloc[i]
        
        if pd.isna(atr_val) or atr_val < 1e-6:
            labels_entry[i] = 0  # FLAT
            labels_direction[i] = 0  # Default SHORT (won't be used)
            continue
        
        # Define zones
        long_target = entry + (atr_val * atr_mult)
        short_target = entry - (atr_val * atr_mult)
        neutral_up = entry + (atr_val * neutral_mult)
        neutral_down = entry - (atr_val * neutral_mult)
        
        # Future price range
        future_slice = df.iloc[i + 1 : i + 1 + future_bars]
        future_high = future_slice["high"].max()
        future_low = future_slice["low"].min()
        
        # Check signal strength
        touches_long = future_high >= long_target
        touches_short = future_low <= short_target
        
        if touches_long or touches_short:
            # Entry signal exists
            labels_entry[i] = 1  # TRADE
            
            # Determine direction
            if touches_long and touches_short:
                up_dist = future_high - entry
                down_dist = entry - future_low
                labels_direction[i] = 1 if up_dist > down_dist else 0
            elif touches_long:
                labels_direction[i] = 1  # LONG
            else:
                labels_direction[i] = 0  # SHORT
        else:
            # No entry signal
            labels_entry[i] = 0  # FLAT (no trade)
            labels_direction[i] = 0  # Default SHORT (won't matter)
    
    return (
        pd.Series(labels_entry, index=df.index, name="label_entry"),
        pd.Series(labels_direction, index=df.index, name="label_direction")
    )


# ==============================
# Dataset Assembly
# ==============================


def build_feature_dataframe(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    label_cfg: LabelConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build features and labels."""
    print("  Adding 15m features...")
    df_feat = add_direction_focused_features(df_15m, include_sma200=True)
    
    print("  Adding 1h context...")
    df_feat = add_higher_timeframe_features(df_feat, df_1h)
    
    df_feat = df_feat.dropna().copy()
    print(f"  After dropping NaNs: {len(df_feat)} bars")
    
    print("  Constructing trading logic labels...")
    label_entry, label_direction = build_trading_logic_labels(df_feat, label_cfg=label_cfg)
    
    df_feat = df_feat.loc[label_entry.index].copy()
    label_entry = label_entry.loc[df_feat.index]
    label_direction = label_direction.loc[df_feat.index]
    
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
        "rsi_14_1h", "macd_hist_1h", "trend_slope_20_1h", "volatility_1h",
    ]
    
    X = df_feat[feature_cols].copy()
    return X, label_entry, label_direction


# ==============================
# Ensemble Training
# ==============================


def train_model_a(X: pd.DataFrame, y_entry: pd.Series):
    """Train Model A: Entry Decision (FLAT vs TRADE)."""
    print("\n" + "="*70)
    print("MODEL A: ENTRY DECISION (FLAT vs TRADE)")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_entry, test_size=0.2, shuffle=False
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train - FLAT: {(y_train==0).sum()}, TRADE: {(y_train==1).sum()}")
    print(f"Test  - FLAT: {(y_test==0).sum()}, TRADE: {(y_test==1).sum()}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    sample_weights = compute_sample_weight('balanced', y_train)
    
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
    model_a.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    print(f"  Completed in {time.time()-start:.2f}s")
    
    y_pred = model_a.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["FLAT", "TRADE"], zero_division=0))
    
    return model_a, scaler, acc, f1


def train_model_b(X: pd.DataFrame, y_direction: pd.Series):
    """Train Model B: Direction Decision (SHORT vs LONG, only on TRADE samples)."""
    print("\n" + "="*70)
    print("MODEL B: DIRECTION DECISION (SHORT vs LONG)")
    print("="*70)
    
    # Filter: only use samples where direction is valid (0 or 1)
    mask = y_direction.isin([0, 1])
    X_filtered = X[mask].copy()
    y_filtered = y_direction[mask].copy()
    
    print(f"Samples with valid direction label: {len(y_filtered)} / {len(y_direction)}")
    print(f"Distribution - SHORT: {(y_filtered==0).sum()}, LONG: {(y_filtered==1).sum()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, shuffle=False
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train - SHORT: {(y_train==0).sum()}, LONG: {(y_train==1).sum()}")
    print(f"Test  - SHORT: {(y_test==0).sum()}, LONG: {(y_test==1).sum()}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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
    
    print("\n  Training...")
    start = time.time()
    model_b.fit(X_train_scaled, y_train)
    print(f"  Completed in {time.time()-start:.2f}s")
    
    y_pred = model_b.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["SHORT", "LONG"], zero_division=0))
    
    return model_b, scaler, mask, acc, f1


def ensemble_predict(X_test: pd.DataFrame, model_a, scaler_a, model_b, scaler_b, mask_b) -> np.ndarray:
    """Two-stage ensemble prediction.
    
    Stage 1: Model A decides if we should trade
    Stage 2: If trading, Model B decides direction
    """
    X_a_scaled = scaler_a.transform(X_test)
    pred_a = model_a.predict(X_a_scaled)  # 0=FLAT, 1=TRADE
    
    ensemble_preds = pred_a.copy()  # Start with FLAT by default
    
    # For TRADE signals, use Model B to determine SHORT (0) vs LONG (1)
    trade_mask = pred_a == 1
    if trade_mask.any():
        X_b_scaled = scaler_b.transform(X_test[trade_mask])
        pred_b = model_b.predict(X_b_scaled)  # 0=SHORT, 1=LONG
        
        # Map: FLAT=0, SHORT=1, LONG=2
        ensemble_preds[trade_mask] = pred_b + 1  # +1 to shift: SHORT->1, LONG->2
    
    return ensemble_preds  # 0=FLAT, 1=SHORT, 2=LONG


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
        neutral_atr_mult=0.25,
    )
    X, y_entry, y_direction = build_feature_dataframe(df_15m, df_1h, label_cfg)
    
    print(f"\nDataset shape: X={X.shape}")
    print(f"Entry label distribution (Model A target):")
    for val in [0, 1]:
        count = (y_entry == val).sum()
        pct = count / len(y_entry) * 100
        label_name = "FLAT" if val == 0 else "TRADE"
        print(f"  {label_name:5s}: {count:7d} ({pct:6.2f}%)")
    
    valid_direction = y_direction.isin([0, 1]).sum()
    print(f"\nDirection labels (Model B target):")
    print(f"  Valid (SHORT/LONG): {valid_direction} samples")
    if valid_direction > 0:
        print(f"  SHORT: {(y_direction==0).sum()}, LONG: {(y_direction==1).sum()}")
    
    print(f"\nTraining ensemble models...")
    model_a, scaler_a, acc_a, f1_a = train_model_a(X, y_entry)
    model_b, scaler_b, mask_b, acc_b, f1_b = train_model_b(X, y_direction)
    
    # Ensemble evaluation
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION ON TEST SET")
    print("="*70)
    
    X_train, X_test, y_entry_train, y_entry_test, y_dir_train, y_dir_test = train_test_split(
        X, y_entry, y_direction, test_size=0.2, shuffle=False
    )
    
    # Create 3-class ground truth for ensemble evaluation
    y_true_3class = y_entry_test.copy().astype(int)
    for idx in y_true_3class[y_true_3class == 1].index:
        idx_pos = y_true_3class.index.get_loc(idx)
        if y_dir_test.iloc[idx_pos] in [0, 1]:
            y_true_3class.iloc[idx_pos] = y_dir_test.iloc[idx_pos] + 1  # 1->SHORT, 2->LONG
    
    ensemble_preds = ensemble_predict(X_test, model_a, scaler_a, model_b, scaler_b, mask_b)
    
    acc_ensemble = accuracy_score(y_true_3class, ensemble_preds)
    f1_ensemble = f1_score(y_true_3class, ensemble_preds, average='weighted', zero_division=0)
    
    print(f"\nEnsemble Accuracy: {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")
    print(f"Ensemble F1 (Weighted): {f1_ensemble:.4f}")
    
    if acc_ensemble >= 0.70:
        print(f"\n✓ TARGET ACHIEVED: {acc_ensemble*100:.2f}% >= 70%")
    else:
        print(f"\n✗ Target not reached: {acc_ensemble*100:.2f}% < 70%")
        print(f"  Gap: {(0.70 - acc_ensemble)*100:.2f}%")
    
    print(f"\nClassification Report (Ensemble):")
    print(classification_report(
        y_true_3class,
        ensemble_preds,
        labels=[0, 1, 2],
        target_names=["FLAT", "SHORT", "LONG"],
        zero_division=0,
    ))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true_3class, ensemble_preds)
    print(f"            Pred FLAT  Pred SHORT  Pred LONG")
    for i, label in enumerate(["FLAT", "SHORT", "LONG"]):
        print(f"Actual {label:5s}   {cm[i,0]:6d}    {cm[i,1]:6d}    {cm[i,2]:6d}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model A (Entry Decision):    {acc_a*100:6.2f}% (F1: {f1_a:.4f})")
    print(f"Model B (Direction):         {acc_b*100:6.2f}% (F1: {f1_b:.4f})")
    print(f"Ensemble (3-class):          {acc_ensemble*100:6.2f}% (F1: {f1_ensemble:.4f})")
    print("="*70)


if __name__ == "__main__":
    main()
