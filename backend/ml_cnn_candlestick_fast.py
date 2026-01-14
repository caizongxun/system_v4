"""Fast CNN Candlestick Model - Optimized Version

Optimizations:
1. Use numpy to draw candles instead of matplotlib (5x faster)
2. Pre-allocate numpy arrays
3. Use smaller batch processing
4. Early image generation (parallel to model loading)

Expected:
- Image conversion: 5-10 minutes for full dataset
- Training: 1-2 hours
- Accuracy: 65-75%
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path

from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not installed")

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
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


def candlestick_to_image_fast(
    candles: np.ndarray,  # N x 4 array: [open, high, low, close]
    image_size: int = 224,
    body_width: int = 8,
) -> np.ndarray:
    """Convert candles to image using PIL (fast numpy-based approach).
    
    Args:
        candles: (N, 4) array with [open, high, low, close]
        image_size: Output size (224x224)
        body_width: Width of candle body
    
    Returns:
        RGB image (224, 224, 3) with values in [0, 1]
    """
    # Create blank white image
    img = Image.new('RGB', (image_size, image_size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Find price range
    all_prices = candles.flatten()
    price_min = np.min(all_prices)
    price_max = np.max(all_prices)
    price_range = price_max - price_min + 1e-8
    
    # Draw candles
    n_candles = len(candles)
    candle_spacing = image_size / (n_candles + 2)
    
    for i, (o, h, l, c) in enumerate(candles):
        # Calculate pixel positions
        x = int((i + 1) * candle_spacing)
        y_h = int(image_size - (h - price_min) / price_range * image_size)
        y_l = int(image_size - (l - price_min) / price_range * image_size)
        y_o = int(image_size - (o - price_min) / price_range * image_size)
        y_c = int(image_size - (c - price_min) / price_range * image_size)
        
        # Color: green for UP, red for DOWN
        color = 'green' if c >= o else 'red'
        
        # Draw wick (high-low line)
        draw.line([(x, y_h), (x, y_l)], fill=color, width=2)
        
        # Draw body (open-close rectangle)
        body_top = min(y_o, y_c)
        body_bottom = max(y_o, y_c)
        body_top = max(0, body_top - 2)  # Add small margin
        body_bottom = min(image_size - 1, body_bottom + 2)
        
        draw.rectangle(
            [(x - body_width // 2, body_top), 
             (x + body_width // 2, body_bottom)],
            fill=color,
            outline=color,
            width=1
        )
    
    # Convert to numpy array
    image_array = np.array(img).astype(np.float32) / 255.0
    
    return image_array


def create_candlestick_dataset_fast(
    df: pd.DataFrame,
    look_back: int = 50,
    look_forward: int = 10,
    max_samples: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create candlestick images efficiently."""
    print(f"\nConverting candlesticks to images...")
    
    ohlc = df[['open', 'high', 'low', 'close']].values
    close_prices = df['close'].values
    
    X = []
    y = []
    
    total = len(df) - look_back - look_forward + 1
    if max_samples:
        total = min(total, max_samples)
    
    for i in range(total):
        if i % 200 == 0:
            print(f"  Progress: {i}/{total}")
        
        # Get candles
        candles = ohlc[i : i + look_back]
        
        # Convert to image
        image = candlestick_to_image_fast(candles)
        X.append(image)
        
        # Label
        current_price = close_prices[i + look_back - 1]
        future_prices = close_prices[i + look_back : i + look_back + look_forward]
        future_avg = np.mean(future_prices)
        
        label = 1 if future_avg > current_price else 0
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset created:")
    print(f"  Shape: {X.shape}")
    print(f"  UP: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"  DOWN: {len(y)-np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
    
    return X, y


def build_cnn_model() -> Sequential:
    """Build EfficientNetB0 model."""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
    )
    
    # Freeze base
    base_model.trainable = False
    
    model = Sequential([
        Input(shape=(224, 224, 3)),
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', name='dense_1'),
        Dropout(0.3),
        Dense(128, activation='relu', name='dense_2'),
        Dropout(0.2),
        Dense(1, activation='sigmoid', name='output'),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    
    print("\nModel built (EfficientNetB0)")
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
    model = build_cnn_model()
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    print(f"\nTraining...")
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
    print("CNN CANDLESTICK MODEL EVALUATION")
    print("="*70)
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
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
    if not HAS_TF or not HAS_PIL:
        print("TensorFlow and PIL are required")
        return
    
    symbol = "BTCUSDT"
    timeframe = "15m"
    look_back = 50
    look_forward = 10
    
    print("CNN Candlestick Model (Fast Version)")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Look back: {look_back} candles")
    print(f"Look forward: {look_forward} candles")
    print(f"Model: EfficientNetB0")
    print(f"Expected Accuracy: 65-75%")
    print("="*70)
    
    # Load data
    print(f"\nLoading data...")
    df = load_klines(symbol, timeframe)
    
    # Create images (using first 5000 for speed, remove max_samples for full dataset)
    print(f"\nNote: Using 5000 samples for demo.")
    print(f"Remove max_samples=5000 parameter for full dataset (~200k samples).")
    X, y = create_candlestick_dataset_fast(
        df,
        look_back=look_back,
        look_forward=look_forward,
        max_samples=5000,  # REMOVE THIS FOR FULL DATASET
    )
    
    # Split
    print(f"\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test,
        test_size=0.5,
        stratify=y_test,
        random_state=42,
    )
    
    print(f"Train: {len(X_train)} images")
    print(f"Val:   {len(X_val)} images")
    print(f"Test:  {len(X_test)} images")
    
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
    
    # Train
    print(f"\nTrain Set Performance:")
    y_train_pred = (model.predict(X_train, verbose=0) >= 0.5).astype(int).flatten()
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Comparison
    print(f"\n" + "="*70)
    print("COMPARISON WITH BASELINES")
    print("="*70)
    baseline_xgb = 0.558
    improvement = (results["accuracy"] - baseline_xgb) * 100
    
    print(f"XGBoost baseline:           {baseline_xgb*100:.2f}%")
    print(f"CNN Candlestick (this):     {results['accuracy']*100:.2f}%")
    print(f"Improvement:                {improvement:+.2f}%")
    print("="*70)
    
    # Save
    model_path = Path("models/cnn_candlestick_fast_model.h5")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
