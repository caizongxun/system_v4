# System V4 - Modular Cryptocurrency Trading Dashboard

一個類似 TradingView 的看盤系統，具備完全模組化的技術指標架構，支持動態啟用/禁用指標，為未來的 AI 模型訓練預留擴展空間。

## 核心特性

- **即時 K 線圖表**：支援 15m、1h、1d 多時間框架
- **模組化指標系統**：MACD、RSI、布林通道等基礎指標，支持動態載入
- **實時數據更新**：從 HuggingFace 資料集讀取加密貨幣數據
- **指標組態管理**：JSON 配置文件動態控制指標開啟/關閉
- **數據視覺化**：Plotly 互動式圖表
- **Web 前端**：Streamlit 互動式應用界面

## 項目結構

```
system_v4/
├── README.md
├── requirements.txt
├── config/
│   ├── indicators_config.json          # 指標配置文件
│   └── trading_pairs_config.json       # 交易對配置
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py              # 從 HF 讀取 K 線
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── base.py                     # 基類定義
│   │   ├── macd.py                     # MACD 指標
│   │   ├── rsi.py                      # RSI 指標
│   │   └── bollinger_bands.py          # 布林通道
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── chart.py                    # 圖表繪製
│   └── engine/
│       ├── __init__.py
│       └── indicator_engine.py         # 指標引擎
├── app/
│   └── dashboard.py                    # Streamlit 應用
└── notebooks/
    └── exploration.ipynb               # 數據探索筆記本
```

## 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 運行應用

```bash
streamlit run app/dashboard.py
```

## 指標配置

在 `config/indicators_config.json` 中配置要啟用的指標：

```json
{
  "enabled_indicators": [
    "macd",
    "rsi",
    "bollinger_bands"
  ],
  "indicator_settings": {
    "macd": {
      "fast": 12,
      "slow": 26,
      "signal": 9
    },
    "rsi": {
      "period": 14
    },
    "bollinger_bands": {
      "period": 20,
      "std_dev": 2
    }
  }
}
```

## 技術棧

- **後端**：Python 3.9+
- **數據處理**：Pandas, NumPy
- **技術分析**：自實現 + TA-Lib（可選）
- **可視化**：Plotly, Streamlit
- **數據源**：HuggingFace Datasets

## 架構設計

### 模組化指標系統

每個指標繼承 `BaseIndicator` 基類，實現統一的計算接口：

```python
class BaseIndicator:
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame
    def get_config(self) -> dict
    def set_config(self, config: dict)
```

這個設計允許：
- 動態載入/卸載指標
- 統一的配置管理
- 易於擴展新指標
- 為 AI 模型提供統一的輸入

### 數據流程

```
HuggingFace 資料集
    ↓
DataLoader (讀取 K 線)
    ↓
IndicatorEngine (計算所有啟用的指標)
    ↓
Chart (繪製圖表)
    ↓
Web 界面 (Streamlit)
```

## 後續擴展方向

1. **新增指標**：只需在 `src/indicators/` 中新增模組，在配置中啟用
2. **AI 模型訓練**：指標輸出可直接作為模型特徵
3. **回測系統**：利用指標信號進行策略回測
4. **實時交易**：集成交易所 API，自動執行交易信號

## 開發手冊

如何新增一個指標：

1. 在 `src/indicators/` 下創建新文件
2. 繼承 `BaseIndicator`
3. 實現 `calculate()` 方法
4. 在 `indicators_config.json` 中添加配置項
5. 重啟應用，指標自動載入

## License

MIT
