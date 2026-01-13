"""Bollinger Bands Indicator Implementation."""

from typing import Dict, Any
import pandas as pd
import numpy as np
from .base import BaseIndicator, IndicatorSignal


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands Indicator.
    
    Consists of three lines:
    - Upper Band = SMA + (std_dev * Standard Deviation)
    - Middle Band = SMA (Simple Moving Average)
    - Lower Band = SMA - (std_dev * Standard Deviation)
    
    Used to identify overbought/oversold conditions and volatility.
    
    Parameters:
        period: Number of periods for SMA (default: 20)
        std_dev: Number of standard deviations (default: 2)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            params = {}
        super().__init__("Bollinger Bands", params)
    
    def get_required_params(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "std_dev": 2
        }
    
    def get_required_columns(self) -> list:
        return ["close", "high", "low"]
    
    def _get_min_data_length(self) -> int:
        return self.params["period"]
    
    def calculate(self, klines: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands values.
        
        Returns DataFrame with columns:
            - bb_middle: Middle band (SMA)
            - bb_upper: Upper band
            - bb_lower: Lower band
            - bb_width: Band width
            - bb_position: Position within bands (0-1)
        """
        self.validate_inputs(klines)
        
        df = klines.copy()
        period = self.params["period"]
        std_dev_mult = self.params["std_dev"]
        
        # Calculate middle band (SMA)
        df["bb_middle"] = df["close"].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = df["close"].rolling(window=period).std()
        
        # Calculate bands
        df["bb_upper"] = df["bb_middle"] + (std * std_dev_mult)
        df["bb_lower"] = df["bb_middle"] - (std * std_dev_mult)
        
        # Calculate band width
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        
        # Calculate position within bands (0 = lower, 1 = upper)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_position"] = df["bb_position"].clip(0, 1)  # Clamp between 0 and 1
        
        self.values = df[["bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position"]]
        return df
    
    def get_signals(self, klines: pd.DataFrame) -> list:
        """
        Generate Bollinger Bands trading signals.
        
        Signals:
        1. Price touches lower band -> OVERSOLD / BUY
        2. Price touches upper band -> OVERBOUGHT / SELL
        3. Band squeeze (width < threshold) -> Volatility compression
        4. Band expansion -> Volatility expansion
        """
        if self.values is None:
            self.calculate(klines)
        
        signals = []
        df = klines.copy()
        df["bb_upper"] = self.values["bb_upper"]
        df["bb_lower"] = self.values["bb_lower"]
        df["bb_middle"] = self.values["bb_middle"]
        df["bb_width"] = self.values["bb_width"]
        df["bb_position"] = self.values["bb_position"]
        
        # Calculate average band width (for squeeze detection)
        avg_band_width = df["bb_width"].rolling(window=20).mean()
        squeeze_threshold = 0.8  # 80% of average width
        
        for idx in range(1, len(df)):
            current_price = df["close"].iloc[idx]
            prev_price = df["close"].iloc[idx - 1]
            upper = df["bb_upper"].iloc[idx]
            lower = df["bb_lower"].iloc[idx]
            middle = df["bb_middle"].iloc[idx]
            width = df["bb_width"].iloc[idx]
            position = df["bb_position"].iloc[idx]
            
            # Price touch lower band
            if current_price <= lower and prev_price > lower:
                signals.append(IndicatorSignal(
                    name="Bollinger Bands",
                    timestamp=df.index[idx],
                    signal_type="BUY",
                    value=current_price,
                    confidence=0.65,
                    metadata={
                        "position": float(position),
                        "touch": "lower",
                        "distance": float(current_price - lower)
                    }
                ))
            
            # Price touch upper band
            elif current_price >= upper and prev_price < upper:
                signals.append(IndicatorSignal(
                    name="Bollinger Bands",
                    timestamp=df.index[idx],
                    signal_type="SELL",
                    value=current_price,
                    confidence=0.65,
                    metadata={
                        "position": float(position),
                        "touch": "upper",
                        "distance": float(upper - current_price)
                    }
                ))
            
            # Band squeeze detection
            if idx > 1 and not pd.isna(avg_band_width.iloc[idx]):
                prev_width = df["bb_width"].iloc[idx - 1]
                if width < squeeze_threshold * avg_band_width.iloc[idx]:
                    signals.append(IndicatorSignal(
                        name="Bollinger Bands",
                        timestamp=df.index[idx],
                        signal_type="HOLD",
                        value=width,
                        confidence=0.5,
                        metadata={
                            "event": "band_squeeze",
                            "width_ratio": float(width / avg_band_width.iloc[idx])
                        }
                    ))
        
        return signals
    
    def get_current_values(self, klines: pd.DataFrame) -> Dict[str, float]:
        """
        Get the latest Bollinger Bands values.
        """
        if self.values is None:
            self.calculate(klines)
        
        latest_close = float(klines["close"].iloc[-1])
        latest_upper = float(self.values["bb_upper"].iloc[-1])
        latest_lower = float(self.values["bb_lower"].iloc[-1])
        latest_middle = float(self.values["bb_middle"].iloc[-1])
        latest_width = float(self.values["bb_width"].iloc[-1])
        latest_position = float(self.values["bb_position"].iloc[-1])
        
        return {
            "upper": latest_upper,
            "middle": latest_middle,
            "lower": latest_lower,
            "width": latest_width,
            "position": latest_position,
            "distance_to_upper": latest_upper - latest_close,
            "distance_to_lower": latest_close - latest_lower
        }
