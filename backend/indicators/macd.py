"""MACD (Moving Average Convergence Divergence) Indicator Implementation."""

from typing import Dict, Any
import pandas as pd
import numpy as np
from .base import BaseIndicator, IndicatorSignal


class MACD(BaseIndicator):
    """
    MACD (Moving Average Convergence Divergence) Indicator.
    
    Measures the difference between two exponential moving averages:
    - MACD Line = 12-period EMA - 26-period EMA
    - Signal Line = 9-period EMA of MACD Line
    - Histogram = MACD Line - Signal Line
    
    Parameters:
        fast_period: Period for fast EMA (default: 12)
        slow_period: Period for slow EMA (default: 26)
        signal_period: Period for signal line EMA (default: 9)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            params = {}
        super().__init__("MACD", params)
    
    def get_required_params(self) -> Dict[str, Any]:
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }
    
    def get_required_columns(self) -> list:
        return ["close"]
    
    def _get_min_data_length(self) -> int:
        return self.params["slow_period"] + self.params["signal_period"]
    
    def calculate(self, klines: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD values.
        
        Returns DataFrame with columns:
            - macd: MACD line
            - macd_signal: Signal line
            - macd_histogram: Histogram (MACD - Signal)
        """
        self.validate_inputs(klines)
        
        df = klines.copy()
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        signal_period = self.params["signal_period"]
        
        # Calculate exponential moving averages
        ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()
        
        # MACD Line
        df["macd"] = ema_fast - ema_slow
        
        # Signal Line (EMA of MACD)
        df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        self.values = df[["macd", "macd_signal", "macd_histogram"]]
        return df
    
    def get_signals(self, klines: pd.DataFrame) -> list:
        """
        Generate MACD trading signals.
        
        Signals:
        1. MACD crosses above signal line -> BUY
        2. MACD crosses below signal line -> SELL
        3. Histogram increases -> Momentum increasing
        4. Histogram decreases -> Momentum decreasing
        """
        if self.values is None:
            self.calculate(klines)
        
        signals = []
        df = klines.copy()
        df["macd"] = self.values["macd"]
        df["macd_signal"] = self.values["macd_signal"]
        df["macd_histogram"] = self.values["macd_histogram"]
        
        # Find crossover points
        macd_prev = df["macd"].shift(1)
        signal_prev = df["macd_signal"].shift(1)
        
        # MACD crosses above signal line
        bullish_cross = (df["macd"] > df["macd_signal"]) & (macd_prev <= signal_prev)
        # MACD crosses below signal line
        bearish_cross = (df["macd"] < df["macd_signal"]) & (macd_prev >= signal_prev)
        
        for idx in range(1, len(df)):
            if bullish_cross.iloc[idx]:
                signals.append(IndicatorSignal(
                    name="MACD",
                    timestamp=df.index[idx],
                    signal_type="BUY",
                    value=df["macd"].iloc[idx],
                    confidence=0.7,
                    metadata={
                        "macd": float(df["macd"].iloc[idx]),
                        "signal": float(df["macd_signal"].iloc[idx]),
                        "histogram": float(df["macd_histogram"].iloc[idx])
                    }
                ))
            elif bearish_cross.iloc[idx]:
                signals.append(IndicatorSignal(
                    name="MACD",
                    timestamp=df.index[idx],
                    signal_type="SELL",
                    value=df["macd"].iloc[idx],
                    confidence=0.7,
                    metadata={
                        "macd": float(df["macd"].iloc[idx]),
                        "signal": float(df["macd_signal"].iloc[idx]),
                        "histogram": float(df["macd_histogram"].iloc[idx])
                    }
                ))
        
        return signals
    
    def get_current_values(self, klines: pd.DataFrame) -> Dict[str, float]:
        """
        Get the latest MACD values.
        """
        if self.values is None:
            self.calculate(klines)
        
        return {
            "macd": float(self.values["macd"].iloc[-1]),
            "signal": float(self.values["macd_signal"].iloc[-1]),
            "histogram": float(self.values["macd_histogram"].iloc[-1])
        }
