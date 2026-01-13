"""FastAPI backend service for System V4 Dashboard.

Provides REST API for K-line data and technical indicators,
plus WebSocket support for real-time updates.

Can be run as background service or standalone.
"""

from fastapi import FastAPI, WebSocket, Query, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import logging
from typing import Dict, List, Optional, Any
import asyncio
import threading
import time
import numpy as np
import pandas as pd

from config import settings, IndicatorConfiguration
from data_loader import KLineDataLoader
from indicators.registry import IndicatorRegistry, IndicatorPipeline

# Setup logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Real-time quantitative trading dashboard with modular indicators"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global indicator configuration
user_indicator_config = IndicatorConfiguration()


# ============== Helper Functions ==============

def clean_json_value(value: Any) -> Any:
    """
    Convert numpy/pandas values to JSON-safe format.
    Handles NaN, Inf, and other special values.
    """
    # Handle pandas Timestamp
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    
    # Handle pandas NA
    if pd.isna(value):
        return None
    
    # Handle numpy inf
    try:
        if np.isinf(value):
            return None
    except (TypeError, ValueError):
        pass
    
    # Handle numpy types
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (int, float, str, bool, type(None))):
        return value
    else:
        return str(value)


def clean_dict_for_json(data: Any) -> Any:
    """
    Recursively clean dictionary for JSON serialization.
    Handles nested dictionaries, lists, and pandas objects.
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # Convert pandas Timestamp keys to strings
            if isinstance(k, pd.Timestamp):
                k = k.isoformat()
            result[str(k)] = clean_dict_for_json(v)
        return result
    elif isinstance(data, (list, tuple)):
        return [clean_dict_for_json(item) for item in data]
    elif isinstance(data, pd.Series):
        return clean_dict_for_json(data.to_dict())
    elif isinstance(data, pd.DataFrame):
        return clean_dict_for_json(data.to_dict(orient='index'))
    else:
        return clean_json_value(data)


# ============== REST API Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "System V4 Backend"}


@app.get("/api/symbols")
async def get_symbols():
    """Get list of supported symbols."""
    return {
        "symbols": KLineDataLoader.get_supported_symbols(),
        "count": len(KLineDataLoader.get_supported_symbols())
    }


@app.get("/api/timeframes")
async def get_timeframes():
    """Get list of supported timeframes."""
    return {
        "timeframes": KLineDataLoader.get_supported_timeframes()
    }


@app.get("/api/indicators")
async def get_indicators():
    """Get list of available indicators with their parameters."""
    return {
        "indicators": IndicatorRegistry.list_indicators(),
        "enabled": user_indicator_config.get_enabled_indicators()
    }


@app.get("/api/klines")
async def get_klines(
    symbol: str = Query(..., description="Trading pair (e.g., BTCUSDT)"),
    timeframe: str = Query(..., description="Timeframe (15m, 1h, 1d)"),
    limit: Optional[int] = Query(500, description="Number of candles to return")
):
    """
    Get K-line data with calculated indicators.
    
    Returns OHLCV data plus all enabled indicator values.
    """
    try:
        # Validate inputs
        if symbol not in KLineDataLoader.get_supported_symbols():
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not supported")
        
        if timeframe not in KLineDataLoader.get_supported_timeframes():
            raise HTTPException(status_code=400, detail=f"Timeframe {timeframe} not supported")
        
        # Load K-line data
        klines = KLineDataLoader.load_klines(symbol, timeframe)
        
        # Limit results
        if limit and limit < len(klines):
            klines = klines.tail(limit)
        
        # Validate data
        is_valid, issues = KLineDataLoader.validate_data(klines, symbol)
        if not is_valid:
            logger.warning(f"Data validation issues for {symbol}: {issues}")
        
        # Calculate indicators
        pipeline = IndicatorPipeline()
        for indicator_name in user_indicator_config.get_enabled_indicators():
            params = user_indicator_config.get_indicator_params(indicator_name)
            pipeline.add(indicator_name, params)
        
        indicator_results = pipeline.calculate(klines)
        
        # Extract K-line data and clean for JSON
        klines_subset = klines[['open', 'high', 'low', 'close', 'volume']].tail(limit)
        klines_dict = {}
        for timestamp, row in klines_subset.iterrows():
            timestamp_str = timestamp.isoformat() if isinstance(timestamp, pd.Timestamp) else str(timestamp)
            klines_dict[timestamp_str] = row.to_dict()
        klines_dict = clean_dict_for_json(klines_dict)
        
        # Build response
        response_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_points": len(klines),
            "timestamp_start": klines.index[0].isoformat() if len(klines) > 0 else None,
            "timestamp_end": klines.index[-1].isoformat() if len(klines) > 0 else None,
            "klines": klines_dict,
            "indicators": {}
        }
        
        # Add indicator values
        for indicator_name, result in indicator_results.items():
            if "error" in result:
                response_data["indicators"][indicator_name] = {"error": result["error"]}
            else:
                indicator_values = result["values"].tail(limit)
                indicator_dict = {}
                for timestamp, row in indicator_values.iterrows():
                    timestamp_str = timestamp.isoformat() if isinstance(timestamp, pd.Timestamp) else str(timestamp)
                    indicator_dict[timestamp_str] = row.to_dict()
                indicator_dict = clean_dict_for_json(indicator_dict)
                response_data["indicators"][indicator_name] = indicator_dict
        
        return response_data
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching klines: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/indicators/configure")
async def configure_indicators(config: Dict):
    """
    Update indicator configuration.
    
    Example:
    {
        "MACD": {"enabled": true, "params": {"fast_period": 12}},
        "RSI": {"enabled": false}
    }
    """
    try:
        for indicator_name, indicator_config in config.items():
            if "enabled" in indicator_config:
                user_indicator_config.set_indicator_enabled(
                    indicator_name,
                    indicator_config["enabled"]
                )
            
            if "params" in indicator_config:
                user_indicator_config.set_indicator_params(
                    indicator_name,
                    indicator_config["params"]
                )
        
        return {
            "status": "success",
            "message": "Indicator configuration updated",
            "current_config": user_indicator_config.to_dict()
        }
    except Exception as e:
        logger.error(f"Error updating indicator config: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/signals")
async def get_signals(
    symbol: str = Query(..., description="Trading pair"),
    timeframe: str = Query(..., description="Timeframe")
):
    """
    Get trading signals from all enabled indicators.
    """
    try:
        klines = KLineDataLoader.load_klines(symbol, timeframe)
        
        # Calculate signals
        pipeline = IndicatorPipeline()
        for indicator_name in user_indicator_config.get_enabled_indicators():
            params = user_indicator_config.get_indicator_params(indicator_name)
            pipeline.add(indicator_name, params)
        
        all_signals = pipeline.get_all_signals(klines)
        
        # Convert signals to JSON-serializable format
        response_signals = {}
        for indicator_name, signals in all_signals.items():
            response_signals[indicator_name] = [
                {
                    "timestamp": sig.timestamp.isoformat(),
                    "signal_type": sig.signal_type,
                    "value": clean_json_value(sig.value),
                    "confidence": clean_json_value(sig.confidence),
                    "metadata": sig.metadata
                }
                for sig in signals
            ]
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signals": response_signals
        }
        
    except Exception as e:
        logger.error(f"Error fetching signals: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============== WebSocket Endpoints ==============

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, symbol: str, timeframe: str):
        await websocket.accept()
        key = f"{symbol}:{timeframe}"
        if key not in self.active_connections:
            self.active_connections[key] = []
        self.active_connections[key].append(websocket)
    
    def disconnect(self, websocket: WebSocket, symbol: str, timeframe: str):
        key = f"{symbol}:{timeframe}"
        if key in self.active_connections:
            self.active_connections[key].remove(websocket)
    
    async def broadcast(self, key: str, message: dict):
        if key in self.active_connections:
            for connection in self.active_connections[key]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting message: {str(e)}")


manager = ConnectionManager()


@app.websocket("/ws/chart")
async def websocket_chart(
    websocket: WebSocket,
    symbol: str = Query(...),
    timeframe: str = Query(...)
):
    """
    WebSocket endpoint for real-time chart updates.
    
    Connect with: ws://localhost:8000/ws/chart?symbol=BTCUSDT&timeframe=1h
    """
    await manager.connect(websocket, symbol, timeframe)
    key = f"{symbol}:{timeframe}"
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            command = json.loads(data)
            
            if command.get("action") == "ping":
                # Respond to ping
                await websocket.send_json({"action": "pong", "timestamp": time.time()})
            
            elif command.get("action") == "update":
                # Client requested chart update
                try:
                    klines = KLineDataLoader.load_klines(symbol, timeframe)
                    latest = klines.tail(1)
                    
                    latest_data = {}
                    for timestamp, row in latest[['open', 'high', 'low', 'close', 'volume']].iterrows():
                        timestamp_str = timestamp.isoformat() if isinstance(timestamp, pd.Timestamp) else str(timestamp)
                        latest_data[timestamp_str] = row.to_dict()
                    latest_data = clean_dict_for_json(latest_data)
                    
                    response = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "latest_candle": latest_data
                    }
                    
                    await websocket.send_json(response)
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
            
            elif command.get("action") == "configure":
                # Update indicator configuration
                if "indicators" in command:
                    await configure_indicators(command["indicators"])
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol, timeframe)
        logger.info(f"WebSocket disconnected for {symbol}:{timeframe}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket, symbol, timeframe)


def run_backend(host: str = settings.API_HOST, port: int = settings.API_PORT):
    """
    Run the backend service.
    Can be called from frontend application to start backend in background thread.
    """
    import uvicorn
    
    logger.info(f"Starting System V4 Backend on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=settings.LOG_LEVEL.lower()
    )


def start_backend_thread(host: str = settings.API_HOST, port: int = settings.API_PORT) -> threading.Thread:
    """
    Start backend service in a background thread.
    Returns the thread object.
    """
    backend_thread = threading.Thread(
        target=run_backend,
        args=(host, port),
        daemon=True
    )
    backend_thread.start()
    logger.info("Backend thread started")
    
    # Wait for backend to be ready
    time.sleep(2)
    return backend_thread


if __name__ == "__main__":
    run_backend()
