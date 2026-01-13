"""FastAPI backend service for System V4 Dashboard.

Provides REST API for K-line data and technical indicators,
plus WebSocket support for real-time updates.
"""

from fastapi import FastAPI, WebSocket, Query, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import logging
from typing import Dict, List, Optional
import asyncio

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
        
        # Build response
        response_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_points": len(klines),
            "timestamp_start": klines.index[0].isoformat() if len(klines) > 0 else None,
            "timestamp_end": klines.index[-1].isoformat() if len(klines) > 0 else None,
            "klines": klines[['open', 'high', 'low', 'close', 'volume']].tail(limit).to_dict(orient='index'),
            "indicators": {}
        }
        
        # Add indicator values
        for indicator_name, result in indicator_results.items():
            if "error" in result:
                response_data["indicators"][indicator_name] = {"error": result["error"]}
            else:
                indicator_values = result["values"].tail(limit)
                response_data["indicators"][indicator_name] = indicator_values.to_dict(orient='index')
        
        return response_data
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching klines: {str(e)}")
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
                    "value": sig.value,
                    "confidence": sig.confidence,
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
                await websocket.send_json({"action": "pong", "timestamp": asyncio.get_event_loop().time()})
            
            elif command.get("action") == "update":
                # Client requested chart update
                try:
                    klines = KLineDataLoader.load_klines(symbol, timeframe)
                    latest = klines.tail(1)
                    
                    response = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "latest_candle": latest[['open', 'high', 'low', 'close', 'volume']].to_dict(orient='index')
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


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
