"""Compare LSTM performance with different lookback windows.

Tests:
1. 100 candles lookback - Standard
2. 200 candles lookback - Extended history
3. Both predicting next 10 candles

Will show:
- Accuracy comparison
- Training time
- Overfitting tendency
- Optimal window recommendation
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

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
    
    print(f"Downloading {path_in_repo}...")
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
    print(f"Loaded {len(df)} bars")
    return df


def create_lstm_sequences(
    df: pd.DataFrame,
    look_back: int,
    look_forward: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training."""
    data = df[["open", "high", "low", "close", "volume"]].values
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = []
    y = []
    
    for i in range(len(data_scaled) - look_back - look_forward + 1):
        seq_in = data_scaled[i : i + look_back]
        X.append(seq_in)
        
        current_close = data[i + look_back - 1, 3]
        future_close = data[i + look_back + look_forward - 1, 3]
        
        label = 1 if future_close > current_close else 0
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def build_lstm_model(look_back: int) -> Sequential:
    """Build LSTM model for given lookback window."""
    model = Sequential([
        LSTM(
            128,
            input_shape=(look_back, 5),
            return_sequences=True,
            name="lstm_1"
        ),
        Dropout(0.2),
        LSTM(64, return_sequences=False, name="lstm_2"),
        Dropout(0.2),
        Dense(32, activation="relu", name="dense_1"),
        Dropout(0.2),
        Dense(1, activation="sigmoid", name="output"),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    return model


def train_and_evaluate(
    symbol: str,
    timeframe: str,
    look_back: int,
    look_forward: int = 10,
    epochs: int = 40,
    batch_size: int = 32,
) -> Dict:
    """Train LSTM with given parameters and return metrics."""
    print(f"\n{'='*70}")
    print(f"LSTM with lookback={look_back}")
    print(f"{'='*70}")
    
    # Load and prepare data
    print(f"\nLoading data...")
    df = load_klines(symbol, timeframe)
    
    print(f"Creating sequences (look_back={look_back})...")
    X, y = create_lstm_sequences(df, look_back=look_back, look_forward=look_forward)
    
    print(f"Sequences created: {len(X)} samples")
    print(f"Label distribution: UP={np.sum(y)}, DOWN={len(y)-np.sum(y)}")
    
    # Split data
    train_ratio = 0.7
    val_ratio = 0.15
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]
    
    # Build model
    print(f"\nBuilding model...")
    model = build_lstm_model(look_back=look_back)
    
    # Train
    print(f"Training (epochs={epochs}, batch_size={batch_size})...")
    start = time.time()
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
    ]
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    
    train_time = time.time() - start
    
    # Evaluate
    print(f"Evaluating...")
    y_train_pred = (model.predict(X_train, verbose=0) >= 0.5).astype(int).flatten()
    y_val_pred = (model.predict(X_val, verbose=0) >= 0.5).astype(int).flatten()
    y_test_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="binary")
    
    # Calculate overfitting metric
    overfit_gap = train_acc - test_acc
    
    results = {
        "look_back": look_back,
        "n_samples": len(X),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "train_time": train_time,
        "overfit_gap": overfit_gap,
        "epochs_trained": len(history.history["loss"]),
    }
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Val Accuracy:   {val_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    print(f"  Test F1:        {test_f1:.4f}")
    print(f"  Overfit Gap:    {overfit_gap*100:.2f}%")
    print(f"  Training Time:  {train_time:.2f}s")
    print(f"  Epochs:         {results['epochs_trained']}/{epochs}")
    
    return results


def main():
    if not HAS_TF:
        print("TensorFlow is required")
        return
    
    symbol = "BTCUSDT"
    timeframe = "15m"
    look_forward = 10
    
    print("LSTM Lookback Window Comparison")
    print(f"Symbol: {symbol}, Timeframe: {timeframe}, Predict: next {look_forward} candles")
    
    # Test different lookback windows
    lookback_values = [100, 200]
    all_results = []
    
    for look_back in lookback_values:
        try:
            result = train_and_evaluate(
                symbol=symbol,
                timeframe=timeframe,
                look_back=look_back,
                look_forward=look_forward,
                epochs=40,
                batch_size=32,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error with look_back={look_back}: {e}")
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Metric':<20} {'100 Candles':<20} {'200 Candles':<20}")
    print("-" * 60)
    
    for result in all_results:
        look_back = result["look_back"]
        test_acc = result["test_acc"]
        test_f1 = result["test_f1"]
        train_time = result["train_time"]
        overfit_gap = result["overfit_gap"]
        
        col = f"{look_back} Candles"
    
    print(f"\nDetailed Results:\n")
    for result in all_results:
        print(f"Lookback: {result['look_back']} candles")
        print(f"  Test Accuracy:  {result['test_acc']*100:>6.2f}%")
        print(f"  Test F1:        {result['test_f1']:>6.4f}")
        print(f"  Train Time:     {result['train_time']:>6.2f}s")
        print(f"  Overfit Gap:    {result['overfit_gap']*100:>6.2f}%")
        print()
    
    # Find best
    best_result = max(all_results, key=lambda x: x["test_acc"])
    print(f"\nBest performing: {best_result['look_back']} candles")
    print(f"  Accuracy: {best_result['test_acc']*100:.2f}%")
    print(f"  Improvement vs baseline (55.8%): {(best_result['test_acc'] - 0.558)*100:+.2f}%")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
