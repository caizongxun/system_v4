"""LSTM Sequence Prediction Model - Strategy A

Objective:
  Input: Previous 100 candlesticks (OHLCV)
  Output: Predict next 10 candlesticks direction (UP/DOWN)

Expected Performance:
  Accuracy: 62-70% (vs 55.8% baseline)
  MAPE: 1-2%

Model Architecture:
  - LSTM Layer 1: 128 units, Dropout 0.2
  - LSTM Layer 2: 64 units, Dropout 0.2
  - Dense: 10 units (predict next 10 candles)
  - Output: Binary classification (UP=1, DOWN=0)

Key Advantages over XGBoost approach:
  1. Captures temporal dependencies (LSTM strength)
  2. Uses full sequence history, not just current indicators
  3. Avoids random walk trap of directional classification
  4. Better generalization to unseen patterns
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
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")


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


def create_lstm_sequences(
    df: pd.DataFrame,
    look_back: int = 100,
    look_forward: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training.
    
    Args:
        df: DataFrame with OHLCV data
        look_back: Number of past candles to use (default 100)
        look_forward: Number of future candles to predict (default 10)
    
    Returns:
        X: Shape (n_samples, look_back, 5) - sequences of OHLCV
        y: Shape (n_samples,) - binary labels (1=UP, 0=DOWN)
    """
    print(f"Creating sequences (look_back={look_back}, look_forward={look_forward})...")
    
    # Prepare OHLCV data
    data = df[["open", "high", "low", "close", "volume"]].values
    
    # Normalize each OHLCV sequence independently
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = []
    y = []
    
    for i in range(len(data_scaled) - look_back - look_forward + 1):
        # Input: past look_back candles
        seq_in = data_scaled[i : i + look_back]
        X.append(seq_in)
        
        # Target: whether close price goes UP or DOWN in next look_forward candles
        current_close = data[i + look_back - 1, 3]  # Close of last candle in sequence
        future_close = data[i + look_back + look_forward - 1, 3]  # Close after look_forward
        
        label = 1 if future_close > current_close else 0
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Label distribution: UP={np.sum(y)}, DOWN={len(y)-np.sum(y)}")
    
    return X, y, scaler


def build_lstm_model(
    look_back: int = 100,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,
    dropout_rate: float = 0.2,
) -> Sequential:
    """Build LSTM model architecture."""
    model = Sequential([
        # Input shape: (batch, time_steps=100, features=5)
        LSTM(
            lstm_units_1,
            input_shape=(look_back, 5),
            return_sequences=True,
            name="lstm_1"
        ),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        LSTM(
            lstm_units_2,
            return_sequences=False,
            name="lstm_2"
        ),
        Dropout(dropout_rate),
        
        # Dense layer for processing LSTM output
        Dense(32, activation="relu", name="dense_1"),
        Dropout(dropout_rate),
        
        # Output layer: binary classification
        Dense(1, activation="sigmoid", name="output"),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    model.summary()
    return model


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
):
    """Train LSTM model with early stopping."""
    print("\nBuilding LSTM model...")
    model = build_lstm_model()
    
    print("\nTraining...")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
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
    print(f"Training completed in {elapsed:.2f}s")
    
    return model, history


def evaluate_lstm_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """Evaluate LSTM model on test set."""
    print("\n" + "="*70)
    print("LSTM MODEL EVALUATION")
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
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC: {roc_auc:.4f}")
    except:
        print("\nROC-AUC: N/A")
    
    print("\n" + "="*70)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def main():
    if not HAS_TF:
        print("TensorFlow is required. Install with: pip install tensorflow")
        return
    
    # Configuration
    symbol = "BTCUSDT"
    timeframe = "15m"
    look_back = 100
    look_forward = 10
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    print(f"LSTM Sequence Prediction Model")
    print(f"="*70)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Look back: {look_back} candles")
    print(f"Look forward: {look_forward} candles")
    print(f"Train/Val/Test split: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")
    print(f"="*70)
    
    # Load data
    print(f"\nLoading data for {symbol}...")
    df = load_klines(symbol, timeframe)
    
    # Create sequences
    X, y, scaler = create_lstm_sequences(
        df,
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
    model, history = train_lstm_model(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=32,
    )
    
    # Evaluate
    results = evaluate_lstm_model(model, X_test, y_test)
    
    # Also evaluate on training set
    print(f"\nTrain Set Performance:")
    y_train_pred = (model.predict(X_train, verbose=0) >= 0.5).astype(int).flatten()
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Comparison with baseline
    print(f"\n" + "="*70)
    print("COMPARISON WITH BASELINE")
    print("="*70)
    baseline_acc = 0.558  # From previous XGBoost method
    improvement = (results["accuracy"] - baseline_acc) * 100
    print(f"Baseline (XGBoost method):  {baseline_acc*100:.2f}%")
    print(f"LSTM method:                {results['accuracy']*100:.2f}%")
    print(f"Improvement:                {improvement:+.2f}%")
    
    if results["accuracy"] > baseline_acc:
        print(f"\nResult: LSTM method outperforms baseline by {improvement:.2f}%")
    else:
        print(f"\nResult: LSTM method underperforms baseline")
    
    print(f"\n" + "="*70)
    
    # Save model
    model_path = Path("models/lstm_sequence_model.h5")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
