"""Data loader for HuggingFace K-line dataset."""

from typing import Optional, Tuple
import pandas as pd
from huggingface_hub import hf_hub_download


class KLineDataLoader:
    """
    Load K-line (OHLCV) data from HuggingFace dataset.
    
    Dataset: zongowo111/v2-crypto-ohlcv-data
    Structure: klines/{SYMBOL}/{BASE}_{TIMEFRAME}.parquet
    """
    
    REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    
    # Supported cryptocurrencies (38 total)
    SUPPORTED_SYMBOLS = [
        "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT",
        "BALUSDT", "BATUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "COMPUSDT",
        "CRVUSDT", "DOGEUSDT", "DOTUSDT", "ENJUSDT", "ENSUSDT", "ETCUSDT",
        "ETHUSDT", "FILUSDT", "GALAUSDT", "GRTUSDT", "IMXUSDT", "KAVAUSDT",
        "LINKUSDT", "LTCUSDT", "MANAUSDT", "MATICUSDT", "MKRUSDT", "NEARUSDT",
        "OPUSDT", "SANDUSDT", "SNXUSDT", "SOLUSDT", "SPELLUSDT", "UNIUSDT",
        "XRPUSDT", "ZRXUSDT"
    ]
    
    # Supported timeframes
    SUPPORTED_TIMEFRAMES = ["15m", "1h", "1d"]
    
    # K-line data columns
    REQUIRED_COLUMNS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]
    
    @classmethod
    def load_klines(
        cls,
        symbol: str,
        timeframe: str,
        cache_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load K-line data from HuggingFace.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            timeframe: Time period ('15m', '1h', '1d')
            cache_dir: Optional cache directory for downloaded files
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If symbol or timeframe not supported
        """
        # Validate inputs
        if symbol not in cls.SUPPORTED_SYMBOLS:
            raise ValueError(
                f"Symbol '{symbol}' not supported. "
                f"Available: {cls.SUPPORTED_SYMBOLS}"
            )
        
        if timeframe not in cls.SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Timeframe '{timeframe}' not supported. "
                f"Available: {cls.SUPPORTED_TIMEFRAMES}"
            )
        
        # Build file path
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        try:
            # Download from HuggingFace
            local_path = hf_hub_download(
                repo_id=cls.REPO_ID,
                filename=path_in_repo,
                repo_type="dataset",
                cache_dir=cache_dir
            )
            
            # Load parquet file
            df = pd.read_parquet(local_path)
            
            # Convert timestamp columns to datetime
            if "open_time" in df.columns:
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            if "close_time" in df.columns:
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            
            # Set open_time as index
            if "open_time" in df.columns:
                df.set_index("open_time", inplace=True)
            
            # Sort by index (should already be sorted)
            df.sort_index(inplace=True)
            
            # Ensure columns are numeric
            numeric_columns = ["open", "high", "low", "close", "volume",
                             "quote_asset_volume", "taker_buy_base_asset_volume",
                             "taker_buy_quote_asset_volume"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Drop rows with NaN values in critical columns
            critical_cols = ["open", "high", "low", "close", "volume"]
            df.dropna(subset=critical_cols, inplace=True)
            
            return df
            
        except FileNotFoundError:
            raise ValueError(
                f"Data not found for {symbol} {timeframe}. "
                f"Please ensure the data exists in the HuggingFace repository."
            )
        except Exception as e:
            raise RuntimeError(f"Error loading K-line data: {str(e)}")
    
    @classmethod
    def load_multiple(
        cls,
        symbols: list,
        timeframe: str,
        cache_dir: Optional[str] = None
    ) -> dict:
        """
        Load K-line data for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            timeframe: Time period
            cache_dir: Optional cache directory
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        errors = {}
        
        for symbol in symbols:
            try:
                results[symbol] = cls.load_klines(symbol, timeframe, cache_dir)
            except Exception as e:
                errors[symbol] = str(e)
        
        if errors:
            print(f"Errors loading data: {errors}")
        
        return results
    
    @classmethod
    def get_supported_symbols(cls) -> list:
        """Get list of supported symbols."""
        return cls.SUPPORTED_SYMBOLS.copy()
    
    @classmethod
    def get_supported_timeframes(cls) -> list:
        """Get list of supported timeframes."""
        return cls.SUPPORTED_TIMEFRAMES.copy()
    
    @classmethod
    def validate_data(
        cls,
        df: pd.DataFrame,
        symbol: str = None
    ) -> Tuple[bool, list]:
        """
        Validate K-line data integrity.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol name for error messages
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check required columns
        missing_cols = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for NaN values
        critical_cols = ["open", "high", "low", "close", "volume"]
        for col in critical_cols:
            if col in df.columns and df[col].isnull().any():
                issues.append(f"NaN values found in {col}")
        
        # Check data integrity
        if "high" in df.columns and "low" in df.columns:
            invalid_high_low = (df["high"] < df["low"]).any()
            if invalid_high_low:
                issues.append("Found high < low (invalid data)")
        
        # Check if sorted
        if not df.index.is_monotonic_increasing:
            issues.append("Data is not sorted by timestamp")
        
        # Check for duplicates
        if df.index.duplicated().any():
            issues.append(f"Found {df.index.duplicated().sum()} duplicate timestamps")
        
        return len(issues) == 0, issues
