# System V4 快速開始指南

## 系統架構

```
前端 (React)          後端 (FastAPI)              數據源 (HuggingFace)
   ↓                      ↓                            ↓
React Components  ←→  REST API + WebSocket  ←→  K-line Data
  (Chart, Panel)      Indicator Calculation      (OHLCV)
                                                  
                      技術指標模塊
                    (MACD, RSI, BB...)
```

## 後端安裝與運行

### 1. 安裝依賴

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置環境

複製 `.env.example` 創建 `.env`（可選）：

```bash
cp .env.example .env
```

### 3. 運行後端服務

```bash
python main.py
```

後端將在 `http://localhost:8000` 啟動。

### 4. 查看 API 文檔

打開瀏覽器訪問：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 核心 API 端點

### 查詢支援的交易對

```bash
curl http://localhost:8000/api/symbols
```

應答：
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", ...],
  "count": 38
}
```

### 查詢支援的時間框架

```bash
curl http://localhost:8000/api/timeframes
```

### 查詢可用指標

```bash
curl http://localhost:8000/api/indicators
```

應答：
```json
{
  "indicators": {
    "MACD": {
      "name": "MACD",
      "params": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    },
    "RSI": {...},
    "Bollinger Bands": {...}
  },
  "enabled": ["MACD", "RSI", "Bollinger Bands"]
}
```

### 獲取 K 線數據 + 指標

```bash
curl "http://localhost:8000/api/klines?symbol=BTCUSDT&timeframe=1h&limit=100"
```

應答：
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "data_points": 100,
  "klines": {
    "2024-01-01T00:00:00+00:00": {
      "open": 42000.0,
      "high": 42500.0,
      "low": 41500.0,
      "close": 42200.0,
      "volume": 150.5
    }
  },
  "indicators": {
    "MACD": {...},
    "RSI": {...},
    "Bollinger Bands": {...}
  }
}
```

### 配置指標

```bash
curl -X POST http://localhost:8000/api/indicators/configure \
  -H "Content-Type: application/json" \
  -d '{
    "MACD": {"enabled": true, "params": {"fast_period": 12}},
    "RSI": {"enabled": true, "params": {"period": 14}},
    "Bollinger Bands": {"enabled": false}
  }'
```

### 獲取交易信號

```bash
curl "http://localhost:8000/api/signals?symbol=BTCUSDT&timeframe=1h"
```

應答：
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "signals": {
    "MACD": [
      {
        "timestamp": "2024-01-01T10:00:00+00:00",
        "signal_type": "BUY",
        "value": 125.5,
        "confidence": 0.7,
        "metadata": {...}
      }
    ],
    "RSI": [...],
    "Bollinger Bands": [...]
  }
}
```

## WebSocket 連接

### 實時圖表更新

```javascript
const ws = new WebSocket(
  'ws://localhost:8000/ws/chart?symbol=BTCUSDT&timeframe=1h'
);

ws.onopen = () => {
  // 請求更新
  ws.send(JSON.stringify({
    action: 'update'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Latest candle:', data.latest_candle);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

## 指標系統

### 1. MACD (Moving Average Convergence Divergence)

**參數**：
- `fast_period`: 12（預設）
- `slow_period`: 26（預設）
- `signal_period`: 9（預設）

**輸出**：
- `macd`: MACD 線
- `macd_signal`: 信號線
- `macd_histogram`: 柱狀圖

**信號**：
- MACD 穿越信號線上方 → BUY
- MACD 穿越信號線下方 → SELL

### 2. RSI (Relative Strength Index)

**參數**：
- `period`: 14（預設）
- `overbought`: 70（預設）
- `oversold`: 30（預設）

**輸出**：
- `rsi`: 0-100 之間的值

**信號**：
- RSI > 70 → OVERBOUGHT（超買）
- RSI < 30 → OVERSOLD（超賣）
- RSI 穿越 70 下方 → SELL
- RSI 穿越 30 上方 → BUY

### 3. Bollinger Bands（布林通道）

**參數**：
- `period`: 20（預設）
- `std_dev`: 2（預設）

**輸出**：
- `bb_upper`: 上軌
- `bb_middle`: 中軌（SMA）
- `bb_lower`: 下軌
- `bb_width`: 通道寬度
- `bb_position`: 價格在通道內的位置 (0-1)

**信號**：
- 價格觸及下軌 → BUY（超賣）
- 價格觸及上軌 → SELL（超買）
- 通道收窄 → BAND_SQUEEZE（波動率壓低）

## Python 使用示例

### 直接使用指標

```python
from backend.data_loader import KLineDataLoader
from backend.indicators import MACD, RSI, BollingerBands

# 加載 K 線數據
klines = KLineDataLoader.load_klines('BTCUSDT', '1h')

# 創建指標實例
macd = MACD({'fast_period': 12, 'slow_period': 26, 'signal_period': 9})
rsi = RSI({'period': 14})
bb = BollingerBands({'period': 20, 'std_dev': 2})

# 計算指標
klines_with_macd = macd.calculate(klines)
klines_with_rsi = rsi.calculate(klines)
klines_with_bb = bb.calculate(klines)

# 獲取信號
macd_signals = macd.get_signals(klines_with_macd)
rsi_signals = rsi.get_signals(klines_with_rsi)

for signal in macd_signals[-5:]:  # 最後 5 個信號
    print(f"{signal.timestamp}: {signal.signal_type} (confidence: {signal.confidence})")
```

### 使用 IndicatorPipeline

```python
from backend.indicators import IndicatorPipeline

# 創建管道
pipeline = IndicatorPipeline()
pipeline.add('MACD')
pipeline.add('RSI', {'period': 21})
pipeline.add('Bollinger Bands')

# 計算所有指標
results = pipeline.calculate(klines)

# 獲取所有信號
all_signals = pipeline.get_all_signals(klines)

for indicator_name, signals in all_signals.items():
    print(f"{indicator_name}: {len(signals)} signals")
```

## 動態配置指標

```python
from backend.config import IndicatorConfiguration

# 創建配置
config = IndicatorConfiguration()

# 啟用/禁用指標
config.set_indicator_enabled('MACD', True)
config.set_indicator_enabled('RSI', False)

# 修改參數
config.set_indicator_params('MACD', {
    'fast_period': 10,
    'slow_period': 20,
    'signal_period': 5
})

# 獲取啟用的指標
enabled = config.get_enabled_indicators()
print(enabled)  # ['MACD', 'Bollinger Bands']
```

## 故障排除

### 1. 找不到 HuggingFace 數據

確保已安裝 `huggingface-hub`：
```bash
pip install huggingface-hub
```

### 2. 導入錯誤

確保 Python 路徑正確：
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"
```

### 3. 數據驗證失敗

檢查 K 線數據完整性：
```python
from backend.data_loader import KLineDataLoader

klines = KLineDataLoader.load_klines('BTCUSDT', '1h')
is_valid, issues = KLineDataLoader.validate_data(klines)
print(f"Valid: {is_valid}")
if not is_valid:
    print(f"Issues: {issues}")
```

## 下一步

- 實現前端 React 界面
- 添加更多技術指標（EMA、STOCH、ADX 等）
- 集成回測引擎
- 開發策略優化功能
- 機器學習模型集成

## 文檔

詳細文檔請查看 `docs/` 目錄：
- `docs/indicator_guide.md` - 指標使用指南
- `docs/api_reference.md` - API 完整參考
- `docs/deployment.md` - 部署指南
