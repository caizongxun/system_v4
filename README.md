# System V4 - 模塊化量化交易看盤系統

一個類似 TradingView 的實時看盤系統，支援模塊化技術指標、動態參數調整、信號生成。

## 系統特性

✅ **模塊化指標設計**
- MACD、RSI、布林通道、SMA、EMA 等基礎指標
- 自定義參數實時調整
- 指標動態開啟/關閉

✅ **TradingView 風格界面**
- K 線蠟燭圖
- 多時間框架支援（15m、1h、1d）
- 38 個交易對選擇

✅ **可擴展架構**
- 標準化指標基類
- 便捷的指標註冊機制
- 模型訓練接口預留

✅ **實時數據推送**
- WebSocket 連接
- 動態圖表更新
- 信號即時生成

## 快速開始

### 前置要求
- Python 3.9+
- Node.js 16+
- HuggingFace 數據集訪問權限

### 後端安裝

```bash
cd backend
pip install -r requirements.txt
python main.py
```

後端運行在 `http://localhost:8000`

### 前端安裝

```bash
cd frontend
npm install
npm start
```

前端運行在 `http://localhost:3000`

## 項目結構

```
system_v4/
├── backend/
│   ├── indicators/          # 技術指標模塊（MACD、RSI、Bollinger Bands等）
│   ├── strategy/            # 策略生成層
│   ├── data_loader.py       # 數據加載器
│   ├── config.py            # 配置文件
│   ├── main.py              # FastAPI 服務入口
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/      # React 組件
│   │   ├── pages/           # 頁面
│   │   ├── hooks/           # React Hooks
│   │   └── App.jsx
│   └── package.json
└── docs/
    ├── indicator_guide.md
    └── api_reference.md
```

## 核心模塊

### 1. 指標系統（indicators/）

所有指標繼承 `BaseIndicator`，實現標準化接口：

```python
class BaseIndicator:
    def calculate(self, klines: pd.DataFrame) -> pd.DataFrame:
        """計算指標"""
        pass
    
    def get_signals(self, values: pd.Series) -> dict:
        """生成交易信號"""
        pass
```

### 2. 支援指標

| 指標 | 類名 | 參數 | 用途 |
|------|------|------|------|
| MACD | MACD | fast(12), slow(26), signal(9) | 動量/趨勢 |
| RSI | RSI | period(14), overbought(70), oversold(30) | 超買超賣 |
| 布林通道 | BollingerBands | period(20), std_dev(2) | 波動率/支撐阻力 |
| SMA | SMA | period | 趨勢方向 |
| EMA | EMA | period, alpha | 快速趨勢 |

### 3. 配置示例

```python
config = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "indicators": {
        "macd": {
            "enabled": True,
            "params": {"fast": 12, "slow": 26, "signal": 9}
        },
        "rsi": {
            "enabled": True,
            "params": {"period": 14}
        },
        "bollinger_bands": {
            "enabled": True,
            "params": {"period": 20, "std_dev": 2}
        }
    }
}
```

## API 端點

### 獲取 K 線數據
```
GET /api/klines?symbol=BTCUSDT&timeframe=1h
```

### 計算指標
```
POST /api/indicators/calculate
Body: {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "indicator_config": {...}
}
```

### WebSocket
```
ws://localhost:8000/ws/chart?symbol=BTCUSDT&timeframe=1h
```

## 開發路線圖

- [x] 項目架構設計
- [ ] 後端核心指標實現（MACD、RSI、Bollinger Bands）
- [ ] FastAPI 服務框架
- [ ] 前端 React 組件
- [ ] 圖表集成
- [ ] WebSocket 實時推送
- [ ] 單元測試
- [ ] Docker 容器化
- [ ] 策略回測引擎集成
- [ ] 機器學習模型接口

## 技術棧

**後端**
- FastAPI
- Pandas、NumPy
- TA-Lib（或自實現指標）
- Pydantic
- WebSocket

**前端**
- React 18
- Lightweight Charts / Chart.js
- Axios
- CSS 3

## 數據來源

- **HuggingFace Dataset**: `zongowo111/v2-crypto-ohlcv-data`
- **38 個交易對**：BTC、ETH、SOL、MATIC 等主流幣種
- **時間框架**：15m、1h、1d

## 貢獻指南

1. Fork 此項目
2. 創建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 許可證

MIT License

## 聯繫方式

如有問題或建議，請開啟 Issue。
