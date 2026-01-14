"""Optimized LSTM Sequence Prediction - Lightweight Version

Optimizations:
1. Reduce model size: 128->64->32 (was overkill)
2. Extract hand-crafted features instead of raw OHLCV
3. Use single LSTM layer + Dense (not stacked)
4. Reduce lookback from 100 to 50 candles (still captures ~8 hours of data)
5. Add technical indicators as features (momentum, volatility)
6. Use smaller batch size for faster iterations

Expected:
- Training time: 2-5 min per epoch (vs 20+ min before)
- Accuracy: Should reach 60-65% by epoch 5-10
- Memory: 50% reduction

Rationale:
- Raw OHLCV sequences are redundant (5 correlated features)
- Technical indicators capture essence of price action
- Smaller window still has enough pattern info
- LSTM is good at temporal patterns, not raw pixel-like data
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path

from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not installed")


REPO_ID = "zongowo111/v2-crypto-ohlcv-data"


def load_klines(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data from HuggingFace dataset."""
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


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators to reduce feature dimensionality.
    
    Instead of 5 raw OHLCV features, extract:
    1. Returns (price change)
    2. RSI (momentum)
    3. Volatility (ATR)
    4. Volume change
    5. MA ratio (trend)
    
    This reduces noise and gives LSTM meaningful patterns to learn.
    """
    df_feat = df.copy()
    
    # 1. Returns (normalized price change)
    df_feat['returns'] = df_feat['close'].pct_change() * 100
    
    # 2. RSI (Relative Strength Index) - momentum indicator
    delta = df_feat['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_feat['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. ATR (Average True Range) - volatility
    high_low = df_feat['high'] - df_feat['low']
    high_close = np.abs(df_feat['high'] - df_feat['close'].shift())
    low_close = np.abs(df_feat['low'] - df_feat['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_feat['atr'] = true_range.rolling(window=14).mean()
    
    # 4. Volume MA ratio (volume trend)
    df_feat['volume_ma'] = df_feat['volume'].rolling(window=20).mean()
    df_feat['volume_ratio'] = df_feat['volume'] / (df_feat['volume_ma'] + 1e-8)
    
    # 5. Price above/below SMA (trend)
    df_feat['sma20'] = df_feat['close'].rolling(window=20).mean()
    df_feat['sma50'] = df_feat['close'].rolling(window=50).mean()
    df_feat['price_sma_ratio'] = df_feat['close'] / (df_feat['sma20'] + 1e-8)
    
    # 6. MACD (trend following)
    exp1 = df_feat['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_feat['close'].ewm(span=26, adjust=False).mean()
    df_feat['macd'] = exp1 - exp2
    df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
    
    # Select features and drop NaN
    feature_cols = [
        'returns', 'rsi', 'atr', 'volume_ratio', 
        'price_sma_ratio', 'macd', 'macd_signal'
    ]
    
    df_feat = df_feat[feature_cols].dropna()
    
    print(f"  Features calculated: {feature_cols}")
    print(f"  Valid rows: {len(df_feat)}")
    
    return df_feat


def create_lstm_sequences_optimized(
    features: pd.DataFrame,
    prices: pd.Series,
    look_back: int = 50,
    look_forward: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences with technical features instead of raw OHLCV.
    
    Reduced lookback from 100 to 50 because:
    - 50 candles at 15m = 750 min = 12.5 hours (still 1.5 trading days)
    - Fewer timesteps = faster training
    - Technical indicators already compress information
    """
    print(f"Creating sequences (look_back={look_back}, look_forward={look_forward})...")
    
    data = features.values
    prices_data = prices.values
    
    # Normalize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = []
    y = []
    
    for i in range(len(data_scaled) - look_back - look_forward + 1):
        # Input: past look_back candles with technical features
        seq_in = data_scaled[i : i + look_back]
        X.append(seq_in)
        
        # Target: price goes UP or DOWN
        current_price = prices_data[i + look_back - 1]
        future_price = prices_data[i + look_back + look_forward - 1]
        
        label = 1 if future_price > current_price else 0
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Label distribution: UP={np.sum(y)}, DOWN={len(y)-np.sum(y)}")
    
    return X, y, scaler


def build_lstm_model_lightweight(
    look_back: int = 50,
    n_features: int = 7,
) -> Sequential:
    """Build lightweight LSTM model.
    
    Changes:
    - Single LSTM layer (64 units, not 128+64)
    - Fewer parameters for faster training
    - Dropout 0.15 (less aggressive)
    - Larger Dense layer before output (captures non-linear patterns)
    """
    model = Sequential([
        # Single LSTM layer is enough for feature sequences
        LSTM(
            64,  # Reduced from 128
            input_shape=(look_back, n_features),
            return_sequences=False,
            name="lstm_1"
        ),
        Dropout(0.15),  # Lighter dropout
        
        # Dense layers for classification
        Dense(32, activation="relu", name="dense_1"),
        Dropout(0.15),
        
        Dense(16, activation="relu", name="dense_2"),
        Dropout(0.1),
        
        # Output layer
        Dense(1, activation="sigmoid", name="output"),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.005),  # Slightly higher LR for faster convergence
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    print("\nModel Summary:")
    model.summary()
    
    return model


def train_lstm_optimized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
) -> Tuple:
    """Train with optimized settings.
    
    Optimization tips:
    - Larger batch size (64 instead of 32) for faster gradient updates
    - Fewer epochs (30 instead of 50, with early stopping)
    - More aggressive early stopping (patience=5 instead of 10)
    """
    print("\nBuilding model...")
    model = build_lstm_model_lightweight(
        look_back=X_train.shape[1],
        n_features=X_train.shape[2],
    )
    
    print(f"\nTraining with batch_size={batch_size}...")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,  # More aggressive stopping
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    
    return model, history


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate model performance."""
    print("\n" + "="*70)
    print("OPTIMIZED LSTM MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score: {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["DOWN", "UP"],
        zero_division=0,
    ))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"            Pred DOWN  Pred UP")
    print(f"Actual DOWN   {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"Actual UP     {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC: {roc_auc:.4f}")
    except:
        pass
    
    print("\n" + "="*70)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def main():
    if not HAS_TF:
        print("TensorFlow is required")
        return
    
    # Configuration
    symbol = "BTCUSDT"
    timeframe = "15m"
    look_back = 50  # Reduced from 100
    look_forward = 10
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    print("OPTIMIZED LSTM Sequence Prediction Model")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Look back: {look_back} candles (~{look_back*15} minutes)")
    print(f"Look forward: {look_forward} candles")
    print(f"Train/Val/Test split: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")
    print(f"\nOptimizations:")
    print(f"  - Lookback reduced: 100 -> 50 candles")
    print(f"  - Features extracted: Raw OHLCV -> 7 technical indicators")
    print(f"  - Model size reduced: (128+64) -> 64 units")
    print(f"  - Batch size increased: 32 -> 64")
    print("="*70)
    
    # Load data
    print(f"\nLoading data for {symbol}...")
    df = load_klines(symbol, timeframe)
    
    # Calculate technical features
    print(f"\nCalculating technical features...")
    features = calculate_technical_features(df)
    
    # Create sequences
    X, y, scaler = create_lstm_sequences_optimized(
        features=features,
        prices=df['close'],
        look_back=look_back,
        look_forward=look_forward,
    )
    
    # Split data
    print(f"\nSplitting data...")
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    
    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Train model
    model, history = train_lstm_optimized(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=30,
        batch_size=64,  # Increased from 32
    )
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # Train set evaluation
    print(f"\nTrain Set Performance:")
    y_train_pred = (model.predict(X_train, verbose=0) >= 0.5).astype(int).flatten()
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Comparison
    print(f"\n" + "="*70)
    print("COMPARISON WITH BASELINE")
    print("="*70)
    baseline_acc = 0.558
    improvement = (results["accuracy"] - baseline_acc) * 100
    print(f"Baseline (XGBoost method):     {baseline_acc*100:.2f}%")
    print(f"Optimized LSTM method:         {results['accuracy']*100:.2f}%")
    print(f"Improvement:                   {improvement:+.2f}%")
    print("="*70)
    
    # Save model
    model_path = Path("models/lstm_optimized_model.h5")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
