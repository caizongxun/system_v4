# System V4 - Modular Quantitative Trading Dashboard

A professional-grade, real-time cryptocurrency trading dashboard with modular technical indicators, dynamic parameter configuration, and WebSocket support for live data streaming.

## Features

### Core Capabilities

- **Modular Indicator System**: MACD, RSI, Bollinger Bands with extensible architecture for adding custom indicators
- **Real-time Data Integration**: Seamless integration with HuggingFace dataset containing 38 major cryptocurrency pairs
- **Dynamic Configuration**: Enable/disable indicators and adjust parameters on-the-fly without service restart
- **REST API**: Comprehensive endpoints for K-line data, indicators, and signal generation
- **WebSocket Support**: Real-time chart updates and live data streaming
- **Data Validation**: Robust input/output validation and error handling
- **Trading Signal Generation**: Automated signal generation from multiple indicators

### Supported Assets

38 cryptocurrency trading pairs including:
BTC, ETH, SOL, MATIC, LINK, AAVE, UNI, AVAX, XRP, DOGE, and 28 others

### Timeframes

- 15 minutes (15m)
- 1 hour (1h)
- 1 day (1d)

## System Architecture

```
Frontend (React)          Backend (FastAPI)             Data Source (HuggingFace)
     |                          |                              |
     +------ REST API ---------> |                              |
     |      WebSocket            +------ Data Loader ---------> |
     |                           |                              |
     <------ Live Updates <------+                              |
                                 |                              |
                      [Indicator Pipeline]                      |
                   (MACD, RSI, Bollinger Bands)                 |
```

## Technology Stack

### Backend
- **Framework**: FastAPI 0.104.1
- **Server**: Uvicorn (ASGI)
- **Data Processing**: Pandas 2.1.3, NumPy 1.26.2
- **Validation**: Pydantic 2.5.0
- **Data Integration**: huggingface-hub 0.19.4
- **Python**: 3.9+

### Data Source
- **Repository**: zongowo111/v2-crypto-ohlcv-data (HuggingFace)
- **Format**: Parquet files with OHLCV data
- **Coverage**: 38 trading pairs across 3 timeframes

## Quick Start

### Installation

```bash
git clone https://github.com/caizongxun/system_v4.git
cd system_v4/backend
pip install -r requirements.txt
```

### Running the Backend

```bash
python main.py
```

The service will start at `http://localhost:8000`

### API Documentation

Once running, access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### System Information

```
GET /                      Root endpoint
GET /api/health            Health check
```

### Asset and Indicator Management

```
GET /api/symbols           List all 38 supported trading pairs
GET /api/timeframes        List supported timeframes (15m, 1h, 1d)
GET /api/indicators        List available indicators with default parameters
```

### Data and Indicators

```
GET /api/klines?symbol=BTCUSDT&timeframe=1h&limit=100
    Returns K-line data with calculated indicator values
    
POST /api/indicators/configure
    Dynamically configure indicator settings
    
GET /api/signals?symbol=BTCUSDT&timeframe=1h
    Returns trading signals from all active indicators
```

### WebSocket

```
ws://localhost:8000/ws/chart?symbol=BTCUSDT&timeframe=1h
    Real-time chart updates and data streaming
```

## Technical Indicators

### MACD (Moving Average Convergence Divergence)

**Parameters**:
- fast_period: 12 (default)
- slow_period: 26 (default)
- signal_period: 9 (default)

**Output**:
- MACD line
- Signal line
- Histogram

**Signals**: BUY (crossover above), SELL (crossover below)

### RSI (Relative Strength Index)

**Parameters**:
- period: 14 (default)
- overbought: 70 (default)
- oversold: 30 (default)

**Output**: RSI value (0-100)

**Signals**: OVERBOUGHT, OVERSOLD, BUY (oversold crossover), SELL (overbought crossover)

### Bollinger Bands

**Parameters**:
- period: 20 (default)
- std_dev: 2 (default)

**Output**:
- Upper band
- Middle band (SMA)
- Lower band
- Band width
- Position within bands

**Signals**: BUY (lower band touch), SELL (upper band touch), BAND_SQUEEZE

## Python Usage Examples

### Direct Indicator Usage

```python
from backend.data_loader import KLineDataLoader
from backend.indicators import MACD, RSI

# Load K-line data
klines = KLineDataLoader.load_klines('BTCUSDT', '1h')

# Create indicators
macd = MACD({'fast_period': 12, 'slow_period': 26, 'signal_period': 9})
rsi = RSI({'period': 14})

# Calculate indicators
klines_with_macd = macd.calculate(klines)
klines_with_rsi = rsi.calculate(klines)

# Get signals
macd_signals = macd.get_signals(klines_with_macd)
rsi_signals = rsi.get_signals(klines_with_rsi)

for signal in macd_signals[-5:]:  # Last 5 signals
    print(f"{signal.timestamp}: {signal.signal_type} (confidence: {signal.confidence})")
```

### Using IndicatorPipeline

```python
from backend.indicators import IndicatorPipeline
from backend.data_loader import KLineDataLoader

# Load data
klines = KLineDataLoader.load_klines('BTCUSDT', '1h')

# Create pipeline
pipeline = IndicatorPipeline()
pipeline.add('MACD')
pipeline.add('RSI', {'period': 21})
pipeline.add('Bollinger Bands')

# Calculate all indicators
results = pipeline.calculate(klines)

# Get all signals
all_signals = pipeline.get_all_signals(klines)

for indicator_name, signals in all_signals.items():
    print(f"{indicator_name}: {len(signals)} signals generated")
```

### Dynamic Indicator Configuration

```python
from backend.config import IndicatorConfiguration

config = IndicatorConfiguration()

# Enable/disable indicators
config.set_indicator_enabled('MACD', True)
config.set_indicator_enabled('RSI', False)

# Modify parameters
config.set_indicator_params('MACD', {
    'fast_period': 10,
    'slow_period': 20,
    'signal_period': 5
})

# Get enabled indicators
enabled = config.get_enabled_indicators()
print(enabled)  # ['MACD', 'Bollinger Bands']
```

## REST API Examples

### Get Supported Symbols

```bash
curl http://localhost:8000/api/symbols
```

Response:
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", ...],
  "count": 38
}
```

### Get K-line Data with Indicators

```bash
curl "http://localhost:8000/api/klines?symbol=BTCUSDT&timeframe=1h&limit=100"
```

### Configure Indicators

```bash
curl -X POST http://localhost:8000/api/indicators/configure \
  -H "Content-Type: application/json" \
  -d '{
    "MACD": {"enabled": true, "params": {"fast_period": 12}},
    "RSI": {"enabled": true, "params": {"period": 14}},
    "Bollinger Bands": {"enabled": false}
  }'
```

### Get Trading Signals

```bash
curl "http://localhost:8000/api/signals?symbol=BTCUSDT&timeframe=1h"
```

## WebSocket Connection Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chart?symbol=BTCUSDT&timeframe=1h');

ws.onopen = () => {
  // Request chart update
  ws.send(JSON.stringify({ action: 'update' }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Latest candle:', data.latest_candle);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

## Project Structure

```
system_v4/
├── README.md
├── requirements.txt
└── backend/
    ├── __init__.py
    ├── main.py                 # FastAPI service entry point
    ├── config.py               # Configuration management
    ├── data_loader.py          # HuggingFace data integration
    └── indicators/
        ├── __init__.py
        ├── base.py             # Base indicator class
        ├── macd.py             # MACD implementation
        ├── rsi.py              # RSI implementation
        ├── bollinger_bands.py   # Bollinger Bands implementation
        └── registry.py         # Indicator registry and pipeline
```

## Code Quality

- Full input/output validation
- Comprehensive error handling
- Modular and extensible architecture
- Type hints throughout codebase
- Docstrings for all major functions
- RESTful API design

## Future Enhancements

- Frontend: React-based TradingView-style interface
- Additional indicators: EMA, STOCH, ADX, ICHIMOKU
- Backtesting engine integration
- Machine learning model support
- Real-time trading execution
- Strategy optimization tools
- Database support for historical data
- Docker containerization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
