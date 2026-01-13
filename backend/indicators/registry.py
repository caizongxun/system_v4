"""Indicator Registry - Manages all available indicators and their selection."""

from typing import Dict, Any, Type, List
from .base import BaseIndicator
from .macd import MACD
from .rsi import RSI
from .bollinger_bands import BollingerBands


class IndicatorRegistry:
    """
    Central registry for all available technical indicators.
    
    Allows dynamic selection and configuration of indicators.
    Supports:
    - Registering new indicators
    - Creating indicator instances by name
    - Validating configurations
    - Listing available indicators
    """
    
    _indicators: Dict[str, Type[BaseIndicator]] = {}
    
    @classmethod
    def register(cls, name: str, indicator_class: Type[BaseIndicator]):
        """
        Register a new indicator.
        
        Args:
            name: Display name of the indicator
            indicator_class: The indicator class (must inherit from BaseIndicator)
        """
        if not issubclass(indicator_class, BaseIndicator):
            raise TypeError(f"{indicator_class} must inherit from BaseIndicator")
        cls._indicators[name] = indicator_class
    
    @classmethod
    def get(cls, name: str, params: Dict[str, Any] = None) -> BaseIndicator:
        """
        Get an indicator instance.
        
        Args:
            name: Name of the indicator
            params: Parameter dictionary
            
        Returns:
            Instance of the requested indicator
            
        Raises:
            ValueError: If indicator not found
        """
        if name not in cls._indicators:
            available = list(cls._indicators.keys())
            raise ValueError(
                f"Indicator '{name}' not found. Available: {available}"
            )
        
        indicator_class = cls._indicators[name]
        return indicator_class(params or {})
    
    @classmethod
    def list_indicators(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available indicators with their default parameters.
        
        Returns:
            Dictionary with indicator info
        """
        result = {}
        for name, indicator_class in cls._indicators.items():
            instance = indicator_class()
            result[name] = {
                "name": name,
                "params": instance.get_required_params(),
                "description": indicator_class.__doc__
            }
        return result
    
    @classmethod
    def get_params(cls, name: str) -> Dict[str, Any]:
        """
        Get default parameters for an indicator.
        
        Args:
            name: Name of the indicator
            
        Returns:
            Dictionary of parameters and defaults
        """
        if name not in cls._indicators:
            raise ValueError(f"Indicator '{name}' not found")
        
        instance = cls._indicators[name]()
        return instance.get_required_params()
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an indicator is registered.
        """
        return name in cls._indicators


# Initialize default indicators
def initialize_registry():
    """Register all default indicators."""
    IndicatorRegistry.register("MACD", MACD)
    IndicatorRegistry.register("RSI", RSI)
    IndicatorRegistry.register("Bollinger Bands", BollingerBands)


# Auto-initialize on import
initialize_registry()


class IndicatorPipeline:
    """
    Pipeline for calculating multiple indicators efficiently.
    
    Usage:
        pipeline = IndicatorPipeline()
        pipeline.add("MACD")
        pipeline.add("RSI", {"period": 21})
        results = pipeline.calculate(klines_df)
    """
    
    def __init__(self):
        self.indicators: List[tuple] = []  # [(name, params), ...]
    
    def add(self, indicator_name: str, params: Dict[str, Any] = None) -> 'IndicatorPipeline':
        """
        Add an indicator to the pipeline.
        
        Args:
            indicator_name: Name of the indicator to add
            params: Custom parameters (optional)
            
        Returns:
            Self for method chaining
        """
        if not IndicatorRegistry.is_registered(indicator_name):
            raise ValueError(f"Indicator '{indicator_name}' not registered")
        
        self.indicators.append((indicator_name, params or {}))
        return self
    
    def calculate(self, klines):
        """
        Calculate all indicators in the pipeline.
        
        Args:
            klines: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping indicator names to DataFrames with calculated values
        """
        results = {}
        
        for indicator_name, params in self.indicators:
            try:
                indicator = IndicatorRegistry.get(indicator_name, params)
                indicator.calculate(klines)
                results[indicator_name] = {
                    "instance": indicator,
                    "values": indicator.values,
                    "latest_signals": indicator.get_latest_signal(klines)
                }
            except Exception as e:
                results[indicator_name] = {
                    "error": str(e)
                }
        
        return results
    
    def get_all_signals(self, klines):
        """
        Get all signals from all indicators.
        
        Args:
            klines: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping indicator names to list of signals
        """
        signals = {}
        
        for indicator_name, params in self.indicators:
            try:
                indicator = IndicatorRegistry.get(indicator_name, params)
                indicator.calculate(klines)
                signals[indicator_name] = indicator.get_signals(klines)
            except Exception as e:
                signals[indicator_name] = []
        
        return signals
    
    def clear(self):
        """Clear all indicators from the pipeline."""
        self.indicators.clear()
        return self
    
    def __repr__(self) -> str:
        indicator_names = [name for name, _ in self.indicators]
        return f"IndicatorPipeline({indicator_names})"
