"""RSI (Relative Strength Index) Indicator Implementation."""

from typing import Dict, Any
import pandas as pd
import numpy as np
from .base import BaseIndicator, IndicatorSignal


class RSI(BaseIndicator):
    """
    RSI (Relative Strength Index) Indicator.
    
    Measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    
    Parameters:
        period: Number of periods (default: 14)
        overbought: Overbought threshold (default: 70)
        oversold: Oversold threshold (default: 30)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            params = {}
        super().__init__("RSI", params)
    
    def get_required_params(self) -> Dict[str, Any]:
        return {
            "period": 14,
            "overbought": 70,
            "oversold": 30
        }
    
    def get_required_columns(self) -> list:
        return ["close"]
    
    def _get_min_data_length(self) -> int:
        return self.params["period"] + 1
    
    def calculate(self, klines: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI values.
        
        Returns DataFrame with column:
            - rsi: RSI values (0-100)
        """
        self.validate_inputs(klines)
        
        df = klines.copy()
        period = self.params["period"]
        
        # Calculate price changes
        delta = df["close"].diff()
        
        # Separate gains and losses
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        
        # Calculate average gains and losses using SMA
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Fill NaN values for the first period
        df["rsi"].fillna(50, inplace=True)
        
        self.values = df[["rsi"]]
        return df
    
    def get_signals(self, klines: pd.DataFrame) -> list:
        """
        Generate RSI trading signals.
        
        Signals:
        1. RSI > overbought -> OVERBOUGHT (potential sell)
        2. RSI < oversold -> OVERSOLD (potential buy)
        3. RSI crosses below overbought -> SELL signal
        4. RSI crosses above oversold -> BUY signal
        """
        if self.values is None:
            self.calculate(klines)
        
        signals = []
        df = klines.copy()
        df["rsi"] = self.values["rsi"]
        
        overbought = self.params["overbought"]
        oversold = self.params["oversold"]
        
        # RSI previous value
        rsi_prev = df["rsi"].shift(1)
        
        # Crossover conditions
        overbought_cross = (df["rsi"] < overbought) & (rsi_prev >= overbought)
        oversold_cross = (df["rsi"] > oversold) & (rsi_prev <= oversold)
        
        # Overbought condition
        is_overbought = df["rsi"] > overbought
        # Oversold condition
        is_oversold = df["rsi"] < oversold
        
        for idx in range(1, len(df)):
            if overbought_cross.iloc[idx]:
                signals.append(IndicatorSignal(
                    name="RSI",
                    timestamp=df.index[idx],
                    signal_type="SELL",
                    value=df["rsi"].iloc[idx],
                    confidence=0.6,
                    metadata={
                        "rsi": float(df["rsi"].iloc[idx]),
                        "condition": "Overbought Crossover"
                    }
                ))
            elif oversold_cross.iloc[idx]:
                signals.append(IndicatorSignal(
                    name="RSI",
                    timestamp=df.index[idx],
                    signal_type="BUY",
                    value=df["rsi"].iloc[idx],
                    confidence=0.6,
                    metadata={
                        "rsi": float(df["rsi"].iloc[idx]),
                        "condition": "Oversold Crossover"
                    }
                ))
            elif is_overbought.iloc[idx]:
                # Maintain overbought signal
                signals.append(IndicatorSignal(
                    name="RSI",
                    timestamp=df.index[idx],
                    signal_type="OVERBOUGHT",
                    value=df["rsi"].iloc[idx],
                    confidence=0.5,
                    metadata={
                        "rsi": float(df["rsi"].iloc[idx]),
                        "condition": "Overbought"
                    }
                ))
            elif is_oversold.iloc[idx]:
                # Maintain oversold signal
                signals.append(IndicatorSignal(
                    name="RSI",
                    timestamp=df.index[idx],
                    signal_type="OVERSOLD",
                    value=df["rsi"].iloc[idx],
                    confidence=0.5,
                    metadata={
                        "rsi": float(df["rsi"].iloc[idx]),
                        "condition": "Oversold"
                    }
                ))
        
        return signals
    
    def get_current_values(self, klines: pd.DataFrame) -> Dict[str, float]:
        """
        Get the latest RSI value and status.
        """
        if self.values is None:
            self.calculate(klines)
        
        rsi_value = float(self.values["rsi"].iloc[-1])
        overbought = self.params["overbought"]
        oversold = self.params["oversold"]
        
        status = "NEUTRAL"
        if rsi_value > overbought:
            status = "OVERBOUGHT"
        elif rsi_value < oversold:
            status = "OVERSOLD"
        
        return {
            "rsi": rsi_value,
            "status": status,
            "overbought_threshold": overbought,
            "oversold_threshold": oversold
        }
