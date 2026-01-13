"""System V4 Desktop Application Frontend.

PyQt5-based desktop application for real-time cryptocurrency trading dashboard.
Automatically starts the backend service in a background thread.
"""

import sys
import json
import requests
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QCheckBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QGridLayout, QGroupBox, QMessageBox,
    QProgressBar, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from main import start_backend_thread


class BackendThread(QThread):
    """Thread to start backend service."""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def run(self):
        try:
            start_backend_thread()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class DataFetchThread(QThread):
    """Thread to fetch data without blocking UI."""
    data_fetched = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, symbol, timeframe, limit=200):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def run(self):
        try:
            url = f"http://localhost:8000/api/klines"
            params = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "limit": self.limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.data_fetched.emit(data)
        except Exception as e:
            self.error.emit(f"Failed to fetch data: {str(e)}")


class InteractiveChartWidget(FigureCanvas):
    """Interactive matplotlib chart widget for PyQt5 with zoom/pan/crosshair."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='white')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Add toolbar for navigation
        self.toolbar = NavigationToolbar2QT(self, parent)
        
        # Interactive features
        self.press = None
        self.xpress = None
        self.ypress = None
        self.cur_xlim = None
        self.cur_ylim = None
        
        # Crosshair
        self.crosshair_on = False
        self.crosshair = None
        self.vline = None
        self.hline = None
        
        # Data for interaction
        self.x_data = None
        self.y_data = None
        
        # Connect events
        self.mpl_connect('press_event', self.on_press)
        self.mpl_connect('release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('figure_leave_event', self.on_leave)
    
    def on_press(self, event):
        """Mouse button pressed."""
        if event.inaxes != self.axes:
            return
        
        self.cur_xlim = self.axes.get_xlim()
        self.cur_ylim = self.axes.get_ylim()
        self.press = [event.xdata, event.ydata]
        self.xpress = event.xdata
        self.ypress = event.ydata
    
    def on_release(self, event):
        """Mouse button released."""
        self.press = None
        self.xpress = None
        self.ypress = None
    
    def on_motion(self, event):
        """Mouse motion event - for panning and crosshair."""
        if event.inaxes != self.axes:
            return
        
        # Update crosshair
        self._update_crosshair(event.xdata, event.ydata)
        
        # Pan (middle mouse button or right click)
        if self.press is not None:
            dx = event.xdata - self.press[0]
            dy = event.ydata - self.press[1]
            
            if event.button == 2:  # Middle mouse button
                new_xlim = [self.cur_xlim[0] - dx, self.cur_xlim[1] - dx]
                new_ylim = [self.cur_ylim[0] - dy, self.cur_ylim[1] - dy]
                
                self.axes.set_xlim(new_xlim)
                self.axes.set_ylim(new_ylim)
                self.draw_idle()
    
    def on_scroll(self, event):
        """Mouse scroll event - for zooming."""
        if event.inaxes != self.axes:
            return
        
        # Get current axis limits
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()
        
        # Get event location
        xdata = event.xdata
        ydata = event.ydata
        
        # Set zoom factor
        if event.button == 'up':
            # Zoom in
            scale_factor = 0.8
        elif event.button == 'down':
            # Zoom out
            scale_factor = 1.2
        else:
            return
        
        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]
        
        self.axes.set_xlim(new_xlim)
        self.axes.set_ylim(new_ylim)
        self.draw_idle()
    
    def on_leave(self, event):
        """Mouse leaves the plot."""
        self._remove_crosshair()
    
    def _update_crosshair(self, xdata, ydata):
        """Update crosshair position."""
        if not self.crosshair_on:
            return
        
        # Remove old crosshair
        if self.vline is not None:
            self.vline.remove()
        if self.hline is not None:
            self.hline.remove()
        
        # Draw new crosshair
        self.vline = self.axes.axvline(color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        self.hline = self.axes.axhline(color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        
        self.vline.set_xdata(xdata)
        self.hline.set_ydata(ydata)
        
        self.draw_idle()
    
    def _remove_crosshair(self):
        """Remove crosshair."""
        if self.vline is not None:
            self.vline.remove()
            self.vline = None
        if self.hline is not None:
            self.hline.remove()
            self.hline = None
        self.draw_idle()
    
    def enable_crosshair(self, enabled=True):
        """Enable or disable crosshair."""
        self.crosshair_on = enabled
        if not enabled:
            self._remove_crosshair()
    
    def reset_view(self):
        """Reset to initial view."""
        self.axes.autoscale()
        self.draw_idle()
    
    def plot_candlestick(self, klines_data):
        """Plot candlestick chart with TradingView-like styling."""
        self.axes.clear()
        
        if not klines_data:
            self.axes.text(0.5, 0.5, 'No data available', 
                          ha='center', va='center', transform=self.axes.transAxes)
            self.draw()
            return
        
        df = pd.DataFrame(klines_data).T
        df.index = pd.to_datetime(df.index)
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Store data for interaction
        self.x_data = np.arange(len(df))
        self.y_data = df['close'].values
        
        # Plot candlesticks
        width = 0.6
        width2 = 0.05
        
        for idx, (timestamp, row) in enumerate(df.iterrows()):
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Color: green for up, red for down
            color = 'green' if close_price >= open_price else 'red'
            
            # Wick (high-low)
            self.axes.plot([idx, idx], [low_price, high_price], color=color, linewidth=1)
            
            # Body (open-close)
            if close_price >= open_price:
                self.axes.bar(idx, close_price - open_price, width=width2, 
                            bottom=open_price, color=color, edgecolor=color)
            else:
                self.axes.bar(idx, open_price - close_price, width=width2, 
                            bottom=close_price, color=color, edgecolor=color)
        
        self.axes.set_title('Price Chart - Scroll to Zoom, Middle Click to Pan', fontsize=12, fontweight='bold')
        self.axes.set_xlabel('Time', fontsize=10)
        self.axes.set_ylabel('Price', fontsize=10)
        self.axes.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()
        self.draw()
    
    def plot_indicator(self, indicator_data, indicator_name):
        """Plot technical indicator."""
        if not indicator_data:
            return
        
        df = pd.DataFrame(indicator_data).T
        df.index = pd.to_datetime(df.index)
        
        if indicator_name == 'MACD':
            self.axes.clear()
            df['macd'] = pd.to_numeric(df.get('macd', 0))
            df['macd_signal'] = pd.to_numeric(df.get('macd_signal', 0))
            df['macd_histogram'] = pd.to_numeric(df.get('macd_histogram', 0))
            
            self.axes.plot(df.index, df['macd'], label='MACD', color='blue')
            self.axes.plot(df.index, df['macd_signal'], label='Signal', color='red')
            self.axes.bar(df.index, df['macd_histogram'], label='Histogram', alpha=0.3, color='gray')
            self.axes.set_title('MACD - Scroll to Zoom', fontsize=12, fontweight='bold')
            self.axes.legend()
            self.axes.grid(True, alpha=0.3)
        
        elif indicator_name == 'RSI':
            self.axes.clear()
            df['rsi'] = pd.to_numeric(df.get('rsi', 50))
            
            self.axes.plot(df.index, df['rsi'], label='RSI', color='purple')
            self.axes.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            self.axes.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            self.axes.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
            self.axes.set_title('RSI - Scroll to Zoom', fontsize=12, fontweight='bold')
            self.axes.set_ylim([0, 100])
            self.axes.legend()
            self.axes.grid(True, alpha=0.3)
        
        elif indicator_name == 'Bollinger Bands':
            self.axes.clear()
            df['bb_upper'] = pd.to_numeric(df.get('bb_upper', 0))
            df['bb_middle'] = pd.to_numeric(df.get('bb_middle', 0))
            df['bb_lower'] = pd.to_numeric(df.get('bb_lower', 0))
            
            self.axes.plot(df.index, df['bb_upper'], label='Upper', color='red', linestyle='--')
            self.axes.plot(df.index, df['bb_middle'], label='Middle (SMA)', color='blue')
            self.axes.plot(df.index, df['bb_lower'], label='Lower', color='green', linestyle='--')
            self.axes.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='blue')
            self.axes.set_title('Bollinger Bands - Scroll to Zoom', fontsize=12, fontweight='bold')
            self.axes.legend()
            self.axes.grid(True, alpha=0.3)
        
        self.fig.autofmt_xdate()
        self.draw()


class System_V4_App(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('System V4 - Quantitative Trading Dashboard')
        self.setGeometry(100, 100, 1400, 900)
        
        self.api_base = "http://localhost:8000"
        self.current_data = {}
        self.fetch_thread = None
        
        # Initialize UI
        self.init_ui()
        
        # Start backend
        self.start_backend()
        
        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh)
        self.refresh_timer.start(10000)  # Refresh every 10 seconds
    
    def init_ui(self):
        """Initialize user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Symbol selector
        control_layout.addWidget(QLabel('Symbol:'))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems([])
        control_layout.addWidget(self.symbol_combo)
        
        # Timeframe selector
        control_layout.addWidget(QLabel('Timeframe:'))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['15m', '1h', '1d'])
        control_layout.addWidget(self.timeframe_combo)
        
        # Limit
        control_layout.addWidget(QLabel('Candles:'))
        self.limit_spinbox = QSpinBox()
        self.limit_spinbox.setRange(10, 500)
        self.limit_spinbox.setValue(100)
        control_layout.addWidget(self.limit_spinbox)
        
        # Refresh button
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.clicked.connect(self.fetch_data)
        control_layout.addWidget(self.refresh_btn)
        
        # Reset view button
        self.reset_btn = QPushButton('Reset View')
        self.reset_btn.clicked.connect(self.reset_chart_view)
        control_layout.addWidget(self.reset_btn)
        
        # Spacer
        control_layout.addStretch()
        
        # Status
        self.status_label = QLabel('Loading...')
        control_layout.addWidget(self.status_label)
        
        main_layout.addLayout(control_layout)
        
        # Indicator panel
        indicator_layout = QHBoxLayout()
        
        self.macd_check = QCheckBox('MACD')
        self.macd_check.setChecked(True)
        self.macd_check.stateChanged.connect(self.on_indicator_changed)
        indicator_layout.addWidget(self.macd_check)
        
        self.rsi_check = QCheckBox('RSI')
        self.rsi_check.setChecked(True)
        self.rsi_check.stateChanged.connect(self.on_indicator_changed)
        indicator_layout.addWidget(self.rsi_check)
        
        self.bb_check = QCheckBox('Bollinger Bands')
        self.bb_check.setChecked(True)
        self.bb_check.stateChanged.connect(self.on_indicator_changed)
        indicator_layout.addWidget(self.bb_check)
        
        # Crosshair checkbox
        self.crosshair_check = QCheckBox('Enable Crosshair')
        self.crosshair_check.setChecked(False)
        self.crosshair_check.stateChanged.connect(self.on_crosshair_changed)
        indicator_layout.addWidget(self.crosshair_check)
        
        indicator_layout.addStretch()
        main_layout.addLayout(indicator_layout)
        
        # Tab widget for charts
        self.tab_widget = QTabWidget()
        
        # Price chart
        self.price_chart = InteractiveChartWidget()
        self.tab_widget.addTab(self.price_chart, 'Price Chart')
        
        # Indicator charts
        self.macd_chart = InteractiveChartWidget()
        self.tab_widget.addTab(self.macd_chart, 'MACD')
        
        self.rsi_chart = InteractiveChartWidget()
        self.tab_widget.addTab(self.rsi_chart, 'RSI')
        
        self.bb_chart = InteractiveChartWidget()
        self.tab_widget.addTab(self.bb_chart, 'Bollinger Bands')
        
        main_layout.addWidget(self.tab_widget)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(5)
        self.data_table.setHorizontalHeaderLabels(['Time', 'Open', 'High', 'Low', 'Close'])
        self.data_table.setMaximumHeight(150)
        main_layout.addWidget(self.data_table)
        
        central_widget.setLayout(main_layout)
    
    def start_backend(self):
        """Start backend service in background thread."""
        self.backend_thread = BackendThread()
        self.backend_thread.finished.connect(self.on_backend_started)
        self.backend_thread.error.connect(self.on_backend_error)
        self.backend_thread.start()
    
    def on_backend_started(self):
        """Called when backend starts successfully."""
        self.status_label.setText('Backend: Running')
        self.load_symbols()
    
    def on_backend_error(self, error_msg):
        """Called when backend fails to start."""
        self.status_label.setText(f'Backend: Error - {error_msg}')
        QMessageBox.critical(self, 'Backend Error', error_msg)
    
    def load_symbols(self):
        """Load available symbols from API."""
        try:
            response = requests.get(f"{self.api_base}/api/symbols", timeout=5)
            if response.status_code == 200:
                data = response.json()
                symbols = data['symbols']
                self.symbol_combo.addItems(symbols)
                self.symbol_combo.setCurrentIndex(10)  # Default to BTCUSDT
                self.fetch_data()
        except Exception as e:
            self.status_label.setText(f'Error loading symbols: {str(e)}')
    
    def fetch_data(self):
        """Fetch K-line data and indicators."""
        if self.fetch_thread and self.fetch_thread.isRunning():
            return
        
        symbol = self.symbol_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        limit = self.limit_spinbox.value()
        
        self.status_label.setText(f'Fetching {symbol} {timeframe}...')
        self.refresh_btn.setEnabled(False)
        
        self.fetch_thread = DataFetchThread(symbol, timeframe, limit)
        self.fetch_thread.data_fetched.connect(self.on_data_fetched)
        self.fetch_thread.error.connect(self.on_fetch_error)
        self.fetch_thread.start()
    
    def on_data_fetched(self, data):
        """Called when data is successfully fetched."""
        self.current_data = data
        self.update_charts()
        self.update_table()
        
        symbol = data.get('symbol', 'N/A')
        timeframe = data.get('timeframe', 'N/A')
        points = data.get('data_points', 0)
        self.status_label.setText(f'{symbol} {timeframe} - {points} candles loaded')
        self.refresh_btn.setEnabled(True)
    
    def on_fetch_error(self, error_msg):
        """Called when data fetch fails."""
        self.status_label.setText(f'Error: {error_msg}')
        self.refresh_btn.setEnabled(True)
        QMessageBox.warning(self, 'Data Error', error_msg)
    
    def update_charts(self):
        """Update all charts."""
        if not self.current_data or 'klines' not in self.current_data:
            return
        
        klines = self.current_data['klines']
        indicators = self.current_data.get('indicators', {})
        
        # Price chart
        self.price_chart.plot_candlestick(klines)
        
        # Indicator charts
        if self.macd_check.isChecked() and 'MACD' in indicators:
            self.macd_chart.plot_indicator(indicators['MACD'], 'MACD')
        
        if self.rsi_check.isChecked() and 'RSI' in indicators:
            self.rsi_chart.plot_indicator(indicators['RSI'], 'RSI')
        
        if self.bb_check.isChecked() and 'Bollinger Bands' in indicators:
            self.bb_chart.plot_indicator(indicators['Bollinger Bands'], 'Bollinger Bands')
    
    def update_table(self):
        """Update data table."""
        if not self.current_data or 'klines' not in self.current_data:
            return
        
        klines = self.current_data['klines']
        klines_list = list(klines.items())[-10:]  # Show last 10
        
        self.data_table.setRowCount(len(klines_list))
        
        for row, (timestamp, data) in enumerate(klines_list):
            self.data_table.setItem(row, 0, QTableWidgetItem(str(timestamp)))
            self.data_table.setItem(row, 1, QTableWidgetItem(f"{data['open']:.2f}"))
            self.data_table.setItem(row, 2, QTableWidgetItem(f"{data['high']:.2f}"))
            self.data_table.setItem(row, 3, QTableWidgetItem(f"{data['low']:.2f}"))
            self.data_table.setItem(row, 4, QTableWidgetItem(f"{data['close']:.2f}"))
    
    def on_indicator_changed(self):
        """Called when indicator checkboxes change."""
        self.update_charts()
    
    def on_crosshair_changed(self):
        """Called when crosshair checkbox changes."""
        enabled = self.crosshair_check.isChecked()
        self.price_chart.enable_crosshair(enabled)
        self.macd_chart.enable_crosshair(enabled)
        self.rsi_chart.enable_crosshair(enabled)
        self.bb_chart.enable_crosshair(enabled)
    
    def reset_chart_view(self):
        """Reset all chart views to initial state."""
        self.price_chart.reset_view()
        self.macd_chart.reset_view()
        self.rsi_chart.reset_view()
        self.bb_chart.reset_view()
    
    def auto_refresh(self):
        """Auto refresh data."""
        if self.current_data:
            self.fetch_data()
    
    def closeEvent(self, event):
        """Handle application close."""
        self.refresh_timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = System_V4_App()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
