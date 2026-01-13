"""Configuration management for System V4."""

from typing import Dict, Any
from pydantic import BaseSettings
import os


class Settings(BaseSettings):
    """Global application settings."""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "System V4 - Quantitative Trading Dashboard"
    API_VERSION: str = "0.1.0"
    
    # CORS Configuration
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    # Data Configuration
    DATA_CACHE_DIR: str = os.getenv("DATA_CACHE_DIR", "./data/cache")
    MAX_KLINES_PER_REQUEST: int = 5000  # Max candles to return per request
    
    # Indicator Configuration
    INDICATOR_CONFIG: Dict[str, Any] = {
        "MACD": {
            "enabled": True,
            "default_params": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            }
        },
        "RSI": {
            "enabled": True,
            "default_params": {
                "period": 14,
                "overbought": 70,
                "oversold": 30
            }
        },
        "Bollinger Bands": {
            "enabled": True,
            "default_params": {
                "period": 20,
                "std_dev": 2
            }
        }
    }
    
    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL: int = 30  # Seconds
    WS_MAX_CONNECTIONS: int = 1000
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


class IndicatorConfiguration:
    """
    User-configurable indicator settings.
    Can be modified dynamically or loaded from file.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default indicator configuration."""
        return {
            "indicators": {
                "MACD": {
                    "enabled": True,
                    "params": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9
                    }
                },
                "RSI": {
                    "enabled": True,
                    "params": {
                        "period": 14,
                        "overbought": 70,
                        "oversold": 30
                    }
                },
                "Bollinger Bands": {
                    "enabled": True,
                    "params": {
                        "period": 20,
                        "std_dev": 2
                    }
                }
            }
        }
    
    def get_enabled_indicators(self) -> list:
        """Get list of enabled indicators."""
        enabled = []
        for name, config in self.config.get("indicators", {}).items():
            if config.get("enabled", False):
                enabled.append(name)
        return enabled
    
    def get_indicator_params(self, indicator_name: str) -> Dict[str, Any]:
        """Get parameters for a specific indicator."""
        indicator_config = self.config.get("indicators", {}).get(indicator_name, {})
        return indicator_config.get("params", {})
    
    def set_indicator_enabled(self, indicator_name: str, enabled: bool):
        """Enable or disable an indicator."""
        if indicator_name in self.config.get("indicators", {}):
            self.config["indicators"][indicator_name]["enabled"] = enabled
    
    def set_indicator_params(self, indicator_name: str, params: Dict[str, Any]):
        """Set parameters for an indicator."""
        if indicator_name in self.config.get("indicators", {}):
            self.config["indicators"][indicator_name]["params"].update(params)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self.config.copy()
