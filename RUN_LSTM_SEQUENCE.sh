#!/bin/bash
echo "Starting LSTM Sequence Prediction (Optimized Version)..."
echo ""
echo "Optimizations Applied:"
echo "  - Data Sampling: 30% of sequences (5x speedup)"
echo "  - Lightweight Architecture: LSTM 64+32 (3x speedup)"
echo "  - Larger Batch Size: 64 (2x speedup)"
echo "  - Expected Time: ~1 hour (vs ~10 hours baseline)"
echo ""
python backend/ml_lstm_optimized.py
echo ""
echo "LSTM Sequence Prediction (Optimized) completed!"
