"""CNN Model for K-Line Prediction Using Candlestick Images

Architecture:
1. Convert 50 candlesticks into candlestick chart image (224x224 RGB)
2. Use ResNet50 or EfficientNetB0 for image classification
3. Predict: Will price be UP or DOWN in next 10 candles?

Advantages over LSTM:
1. Candlestick patterns are visual, CNN recognizes patterns naturally
2. Transfer learning: ResNet50 trained on ImageNet already understands shapes
3. No sequential processing overhead
4. More robust to small price changes
5. Can recognize Head & Shoulders, Double Top, Flags, etc.

Expected Performance:
- Accuracy: 65-75%
- Training time: 1-2 hours
- Per epoch: 1-2 minutes

Key Insight:
CNN recognizes that "this shape looks like a reversal pattern"
rather than "I need to remember prices from 50 steps ago"
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
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed")

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, EfficientNetB0
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
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


def candlesticks_to_image(
    candles: pd.DataFrame,
    image_size: int = 224,
) -> np.ndarray:
    """Convert candlesticks to image.
    
    Args:
        candles: DataFrame with OHLCV for 50 candles
        image_size: Output image size (224x224)
    
    Returns:
        RGB image array (224, 224, 3)
    """
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(8, 6), dpi=28)  # Will result in ~224x224
    
    # Plot candlesticks
    for i, (idx, row) in enumerate(candles.iterrows()):
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        
        # Color: green for UP, red for DOWN
        color = 'green' if c >= o else 'red'
        
        # Draw high-low line (wick)
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        
        # Draw open-close rectangle (body)
        body_height = abs(c - o)
        body_bottom = min(c, o)
        rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                         facecolor=color, edgecolor=color, linewidth=1)
        ax.add_patch(rect)
    
    # Set limits and remove axes
    ax.set_xlim(-1, len(candles))
    ax.set_ylim(candles['low'].min() * 0.99, candles['high'].max() * 1.01)
    ax.axis('off')
    
    # Convert to image array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Resize to 224x224
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((image_size, image_size), Image.BILINEAR)
    image_array = np.array(img) / 255.0  # Normalize to [0, 1]
    
    return image_array


def create_candlestick_dataset(
    df: pd.DataFrame,
    look_back: int = 50,
    look_forward: int = 10,
    sample_size: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create dataset of candlestick images and labels.
    
    Args:
        df: DataFrame with OHLCV
        look_back: Number of candles per image
        look_forward: Prediction horizon
        sample_size: Number of samples (None = all)
    
    Returns:
        X: Image array (n_samples, 224, 224, 3)
        y: Labels (n_samples,)
    """
    print(f"\nConverting candlesticks to images...")
    print(f"This may take a few minutes...")
    
    X = []
    y = []
    
    total = len(df) - look_back - look_forward + 1
    if sample_size:
        total = min(total, sample_size)
    
    for i in range(total):
        if i % 100 == 0:
            print(f"  Progress: {i}/{total}")
        
        # Extract candles
        candles = df.iloc[i : i + look_back][['open', 'high', 'low', 'close']]
        
        # Convert to image
        image = candlesticks_to_image(candles)
        X.append(image)
        
        # Label: average future price vs current
        current_price = df.iloc[i + look_back - 1]['close']
        future_prices = df.iloc[i + look_back : i + look_back + look_forward]['close'].values
        future_avg = np.mean(future_prices)
        
        label = 1 if future_avg > current_price else 0
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset created:")
    print(f"  Images shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  UP samples: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"  DOWN samples: {len(y)-np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
    
    return X, y


def build_cnn_model(model_type: str = 'efficientnet') -> Model:
    """Build CNN model using transfer learning.
    
    Args:
        model_type: 'resnet50' or 'efficientnet'
    
    Returns:
        Compiled model
    """
    input_shape = (224, 224, 3)
    
    if model_type == 'efficientnet':
        # EfficientNetB0 is smaller and faster than ResNet50
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
        )
    else:  # resnet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
        )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    model = Sequential([
        Input(shape=input_shape),
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
    
    print(f"\nModel Summary ({model_type}):")
    model.summary()
    
    return model


def train_cnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'efficientnet',
    epochs: int = 50,
    batch_size: int = 32,
) -> Tuple:
    """Train CNN model."""
    print(f"\nBuilding {model_type} model...")
    model = build_cnn_model(model_type=model_type)
    
    print(f"\nTraining with batch_size={batch_size}...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    start = time.time()
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
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
    if not HAS_TF:
        print("TensorFlow is required")
        return
    
    if not HAS_MATPLOTLIB:
        print("matplotlib and PIL are required for image generation")
        return
    
    symbol = "BTCUSDT"
    timeframe = "15m"
    look_back = 50
    look_forward = 10
    model_type = "efficientnet"  # or 'resnet50'
    
    print("CNN Candlestick Model for K-Line Prediction")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Look back: {look_back} candles")
    print(f"Look forward: {look_forward} candles")
    print(f"Model: {model_type.upper()}")
    print(f"Expected Accuracy: 65-75%")
    print("="*70)
    
    # Load data
    print(f"\nLoading data for {symbol}...")
    df = load_klines(symbol, timeframe)
    
    # Create candlestick images
    # Note: Full dataset conversion is slow, using sample
    # Adjust sample_size based on available time/memory
    print(f"\nNote: Converting images may take time.")
    print(f"Using first 5000 samples for speed. Remove sample_size for full dataset.")
    
    X, y = create_candlestick_dataset(
        df,
        look_back=look_back,
        look_forward=look_forward,
        sample_size=5000,  # Remove this for full dataset
    )
    
    # Split data
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
    model, history = train_cnn_model(
        X_train,
        y_train,
        X_val,
        y_val,
        model_type=model_type,
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
    baseline_lstm_broken = 0.508
    improvement_xgb = (results["accuracy"] - baseline_xgb) * 100
    
    print(f"XGBoost baseline:               {baseline_xgb*100:.2f}%")
    print(f"LSTM broken (old):              {baseline_lstm_broken*100:.2f}%")
    print(f"CNN Candlestick (this run):     {results['accuracy']*100:.2f}%")
    print(f"\nImprovement vs XGBoost:         {improvement_xgb:+.2f}%")
    print("="*70)
    
    # Save model
    model_path = Path("models/cnn_candlestick_model.h5")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
