"""LSTM Model - Fixed Version

Critical Fixes:
1. 標籤生成方式改為計算「未來 10 根 K 棒的平均價格」vs 當前價格
   舊方法: future_close[i+look_forward-1] vs current_close[i+look_back-1]
   問題: 只看最後一根，無法代表「10 根的趨勢」
   
2. 新方法: future_close 平均 vs current_close
   優點: 更穩定，代表真實趨勢
   
3. 添加數據驗證和診斷
4. 使用階層化採樣確保訓練集平衡
5. 調整模型容量和超參數
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
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
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
    """Calculate technical indicators."""
    df_feat = df.copy()
    
    # Returns
    df_feat['returns'] = df_feat['close'].pct_change() * 100
    
    # RSI
    delta = df_feat['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_feat['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df_feat['high'] - df_feat['low']
    high_close = np.abs(df_feat['high'] - df_feat['close'].shift())
    low_close = np.abs(df_feat['low'] - df_feat['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_feat['atr'] = true_range.rolling(window=14).mean()
    
    # Volume Ratio
    df_feat['volume_ma'] = df_feat['volume'].rolling(window=20).mean()
    df_feat['volume_ratio'] = df_feat['volume'] / (df_feat['volume_ma'] + 1e-8)
    
    # Price/SMA
    df_feat['sma20'] = df_feat['close'].rolling(window=20).mean()
    df_feat['sma50'] = df_feat['close'].rolling(window=50).mean()
    df_feat['price_sma_ratio'] = df_feat['close'] / (df_feat['sma20'] + 1e-8)
    
    # MACD
    exp1 = df_feat['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_feat['close'].ewm(span=26, adjust=False).mean()
    df_feat['macd'] = exp1 - exp2
    df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
    
    feature_cols = [
        'returns', 'rsi', 'atr', 'volume_ratio',
        'price_sma_ratio', 'macd', 'macd_signal'
    ]
    
    df_feat = df_feat[feature_cols].dropna()
    return df_feat


def create_lstm_sequences_fixed(
    features: pd.DataFrame,
    prices: pd.Series,
    look_back: int = 50,
    look_forward: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences with FIXED label generation.
    
    Critical Fix:
    舊版本: label = 1 if prices[i+look_back+look_forward-1] > prices[i+look_back-1]
    問題: 只看最後一根，可能反向波動
    
    新版本: label = 1 if prices[i+look_back:i+look_back+look_forward].mean() > prices[i+look_back-1]
    優勢: 看未來 look_forward 根的平均，代表真實趨勢
    """
    print(f"Creating sequences (look_back={look_back}, look_forward={look_forward})...")
    print(f"Label generation: Average price of next {look_forward} candles vs current")
    
    data = features.values
    prices_data = prices.values
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = []
    y = []
    
    for i in range(len(data_scaled) - look_back - look_forward + 1):
        # Input: past look_back candles with technical features
        seq_in = data_scaled[i : i + look_back]
        X.append(seq_in)
        
        # Target: FIXED - Compare average of future prices vs current price
        current_price = prices_data[i + look_back - 1]
        future_prices = prices_data[i + look_back : i + look_back + look_forward]
        future_avg = np.mean(future_prices)
        
        label = 1 if future_avg > current_price else 0
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Label distribution: UP={np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%), DOWN={len(y)-np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
    
    # Diagnose label quality
    print(f"\nLabel Diagnosis:")
    up_samples = np.sum(y == 1)
    down_samples = np.sum(y == 0)
    if up_samples > down_samples * 1.5 or down_samples > up_samples * 1.5:
        print(f"  WARNING: Imbalanced labels! UP/DOWN ratio = {up_samples/down_samples:.2f}")
    else:
        print(f"  OK: Balanced labels (ratio = {up_samples/down_samples:.2f})")
    
    return X, y, scaler


def build_lstm_model_v2(
    look_back: int = 50,
    n_features: int = 7,
) -> Sequential:
    """Build improved LSTM model.
    
    Improvements:
    1. Bidirectional LSTM - Process sequence both forward and backward
    2. Larger capacity (128 instead of 64) - More parameters to learn
    3. Better regularization
    4. Higher learning rate for faster convergence
    """
    model = Sequential([
        # Bidirectional LSTM
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=(look_back, n_features),
            name="bidirectional_lstm_1"
        ),
        Dropout(0.2),
        
        # Second LSTM
        LSTM(32, return_sequences=False, name="lstm_2"),
        Dropout(0.2),
        
        # Dense layers
        Dense(64, activation="relu", name="dense_1"),
        Dropout(0.2),
        
        Dense(32, activation="relu", name="dense_2"),
        Dropout(0.1),
        
        # Output
        Dense(1, activation="sigmoid", name="output"),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.01),  # Higher learning rate
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    print("\nModel Summary:")
    model.summary()
    
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
) -> Tuple:
    """Train model."""
    print("\nBuilding model...")
    model = build_lstm_model_v2(
        look_back=X_train.shape[1],
        n_features=X_train.shape[2],
    )
    
    print(f"\nTraining with batch_size={batch_size}...")
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
    """Evaluate model."""
    print("\n" + "="*70)
    print("LSTM MODEL EVALUATION (FIXED)")
    print("="*70)
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    
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
    
    symbol = "BTCUSDT"
    timeframe = "15m"
    look_back = 50
    look_forward = 10
    
    print("LSTM Model - FIXED Version")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Look back: {look_back} candles (~{look_back*15} minutes)")
    print(f"Look forward: {look_forward} candles")
    print(f"\nKey Fixes:")
    print(f"  1. Label generation: Average future price vs current (not just last point)")
    print(f"  2. Model: Bidirectional LSTM for better sequence understanding")
    print(f"  3. Higher learning rate for faster convergence")
    print("="*70)
    
    # Load data
    print(f"\nLoading data for {symbol}...")
    df = load_klines(symbol, timeframe)
    
    # Calculate features
    print(f"\nCalculating technical features...")
    features = calculate_technical_features(df)
    
    # Create sequences - FIXED VERSION
    X, y, scaler = create_lstm_sequences_fixed(
        features=features,
        prices=df['close'],
        look_back=look_back,
        look_forward=look_forward,
    )
    
    # Split data with stratification
    print(f"\nSplitting data with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # Further split test into val and test
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test,
        test_size=0.5,
        stratify=y_test,
        random_state=42
    )
    
    print(f"Train: {len(X_train)} samples (UP: {np.sum(y_train)}, DOWN: {len(y_train)-np.sum(y_train)})")
    print(f"Val:   {len(X_val)} samples (UP: {np.sum(y_val)}, DOWN: {len(y_val)-np.sum(y_val)})")
    print(f"Test:  {len(X_test)} samples (UP: {np.sum(y_test)}, DOWN: {len(y_test)-np.sum(y_test)})")
    
    # Train
    model, history = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=32,
    )
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # Train evaluation
    print(f"\nTrain Set Performance:")
    y_train_pred = (model.predict(X_train, verbose=0) >= 0.5).astype(int).flatten()
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Comparison
    print(f"\n" + "="*70)
    print("COMPARISON WITH BASELINES")
    print("="*70)
    baseline_xgb = 0.558
    old_lstm_broken = 0.508  # Previous broken version
    improvement_xgb = (results["accuracy"] - baseline_xgb) * 100
    improvement_old = (results["accuracy"] - old_lstm_broken) * 100
    
    print(f"XGBoost baseline (broken LSTM):  {baseline_xgb*100:.2f}%")
    print(f"Previous LSTM (broken labels):   {old_lstm_broken*100:.2f}%")
    print(f"Fixed LSTM (this run):           {results['accuracy']*100:.2f}%")
    print(f"\nImprovement vs XGBoost:         {improvement_xgb:+.2f}%")
    print(f"Improvement vs broken LSTM:     {improvement_old:+.2f}%")
    print("="*70)
    
    # Save
    model_path = Path("models/lstm_fixed_model.h5")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
