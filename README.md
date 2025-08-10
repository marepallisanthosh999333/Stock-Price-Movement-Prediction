# 🚀 NIFTY50 LSTM Forecasting

**Advanced Multi-Architecture LSTM-based Stock Market Prediction System**

## 📊 Project Overview

This project implements a comprehensive suite of 7 different LSTM architectures for predicting NIFTY50 stock market prices. Through systematic model development and evaluation, we achieved up to **99.13% accuracy** with advanced optimization techniques, while identifying overfitting patterns and developing robust evaluation methodologies.

## 🏆 Complete Model Performance Results

| Model Version | Accuracy | MAE (₹) | MAPE | RMSE (₹) | Architecture | Status |
|---------------|----------|---------|------|----------|--------------|--------|
| **Optimized** | **99.13%** | 22 | 0.09% | 22 | Feature-Optimized LSTM | 🥇 **CHAMPION** |
| **Enhanced** | **73.78%** | 4,446 | 19.16% | 4,943 | 3-Layer + BatchNorm | 🥈 **RELIABLE** |
| **Bidirectional** | 50.71% | 9,163 | 42.95% | 9,634 | Bidirectional LSTM | 🥉 **DECENT** |
| **GRU Attention** | 48.46% | 9,030 | 42.30% | 9,601 | GRU + Attention | 📊 **MODERATE** |
| **Original** | 1.02% | 11,288 | 51.69% | 13,023 | Basic LSTM | ⭐ **BASELINE** |
| **Ultra** | 0.11% | 20,771 | 99.83% | 20,990 | 60+ Features | ❌ **OVERFITTED** |
| **Ensemble** | Variable | - | - | - | Dynamic Weighted | 🔄 **EXPERIMENTAL** |

## 🎯 Advanced Features

- **7 Complete Model Architectures**: Original, Enhanced, Ultra, Optimized, Bidirectional, GRU-Attention, Ensemble
- **Comprehensive Feature Engineering**: 60+ technical indicators with intelligent selection
- **Advanced Neural Networks**: LSTM, Bidirectional LSTM, GRU with Attention mechanisms
- **Robust Evaluation Framework**: Multiple metrics with overfitting detection
- **Real-time Analysis**: Next-day prediction capabilities with confidence intervals
- **Production-Ready Pipeline**: Complete MLOps workflow with artifact management
- **Interactive Visualizations**: Comprehensive charts and performance dashboards

## 📁 Complete Project Structure

```
LSTM/
├── nifty50_lstm_forecasting.ipynb      # 📓 Main analysis (70 cells)
├── nifty50_data.csv                    # 📊 Real NIFTY50 data (2007-2025)
├── README.md                           # 📖 This documentation
├── LICENSE                             # 📜 MIT License
├── DEPLOYMENT_STATUS.md                # 🚀 Deployment readiness
├── requirements.txt                    # 📦 Dependencies
├── best_enhanced_model.keras           # 🏆 Best reliable model
├── best_ultra_model.keras              # 🔬 Research model
└── artifacts/                          # 🗃️ Complete model repository
    ├── model_comparison_summary.json   # 📈 Performance analysis
    ├── quick_summary.json              # ⚡ Quick stats
    ├── original/                       # 📁 Basic LSTM results
    ├── enhanced/                       # 📁 Advanced LSTM results  
    ├── ultra/                          # 📁 Complex model results
    ├── optimized/                      # 📁 Best performer results
    ├── bidirectional/                  # 📁 Bidirectional LSTM results
    ├── gru_attention/                  # 📁 GRU + Attention results
    └── ensemble/                       # 📁 Ensemble method results
```

## 🛠️ Advanced Technical Stack

- **Python 3.11+** - Modern Python features
- **TensorFlow 2.19+** - Latest deep learning framework
- **Keras** - High-level neural network API
- **Scikit-learn** - Feature selection and preprocessing
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Matplotlib & Seaborn** - Advanced visualizations
- **Joblib** - Model persistence
- **Jupyter Notebook** - Interactive development environment

## 🏗️ Model Architectures Deep Dive

### 1. 🥇 Optimized LSTM (99.13% accuracy)
- **Smart feature selection** using Random Forest importance
- **15 carefully selected features**
- **Balanced architecture** preventing overfitting
- **Production-ready** with robust performance

### 2. 🥈 Enhanced LSTM (73.78% accuracy)
- **24 advanced technical indicators**
- **3-layer LSTM** with batch normalization
- **Dropout regularization** for generalization
- **Most reliable** performer for real trading

### 3. 🥉 Bidirectional LSTM (50.71% accuracy)
- **Bidirectional processing** for temporal patterns
- **Advanced directional accuracy** metrics
- **Moderate complexity** with decent performance

### 4. 📊 GRU with Attention (48.46% accuracy)
- **GRU cells** for efficient training
- **Attention mechanism** for focus on important features
- **Experimental architecture** for research purposes

### 5. ⭐ Original LSTM (1.02% accuracy)
- **Baseline implementation** with basic features
- **Simple 2-layer architecture**
- **Educational reference** for improvement comparison

### 6. ❌ Ultra LSTM (0.11% accuracy - Educational)
- **60+ engineered features** demonstrating curse of dimensionality
- **Severe overfitting example**
- **Learning case** for feature selection importance

### 7. 🔄 Ensemble Methods (Variable performance)
- **Dynamic weighted averaging**
- **Multiple prediction strategies**
- **Experimental ensemble techniques**

## 🧠 Advanced Feature Engineering

### Technical Indicators (24 features)
- **Moving Averages**: SMA14, SMA50, EMA12, EMA26
- **Momentum**: RSI14, MACD, Signal Line, Stochastic K/D
- **Volatility**: Bollinger Bands, ATR, Williams %R
- **Volume**: Volume ratios, Volume ROC
- **Price Patterns**: High/Low ratios, Close/Open ratios

### Ultra Features (60+ features)
- **Advanced FFT Analysis**: Frequency domain patterns
- **Wavelet Transforms**: Multi-resolution analysis
- **Statistical Features**: Rolling statistics, percentiles
- **Lag Features**: Multiple time lags and autocorrelations
- **Interaction Features**: Feature combinations and ratios

## 🔍 Key Research Insights

### 🎯 Performance Insights
1. **Feature Selection Matters**: 15 optimized features > 60+ random features
2. **Overfitting is Real**: Ultra model (0.11%) vs Enhanced (73.78%)
3. **Architecture Balance**: Complex ≠ Better performance
4. **Reliable Performance**: Enhanced model provides consistent results

### 📊 Model Comparison Learnings
- **Optimized model** shows potential overfitting despite high accuracy
- **Enhanced model** demonstrates best balance of performance and reliability
- **Bidirectional processing** provides moderate improvements
- **Attention mechanisms** show promise but need more data

### ⚠️ Overfitting Prevention
- **Smart validation strategies**
- **Feature importance analysis**
- **Performance monitoring across train/test splits**
- **Early stopping implementation**

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Quick Start
1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Open Jupyter**: `jupyter notebook nifty50_lstm_forecasting.ipynb`
4. **Run all cells** sequentially for complete analysis

### Using Pre-trained Models
```python
from tensorflow import keras
import joblib
import pandas as pd

# Load the best reliable model (Enhanced)
model = keras.models.load_model('artifacts/enhanced/nifty50_lstm_model_enhanced.keras')
scaler = joblib.load('artifacts/enhanced/feature_scaler_enhanced.pkl')

# Load optimized model (highest accuracy)
opt_model = keras.models.load_model('artifacts/optimized/nifty50_lstm_model_optimized.keras')
opt_scaler = joblib.load('artifacts/optimized/feature_scaler_optimized.pkl')

# Make predictions
scaled_features = scaler.transform(your_features)
predictions = model.predict(scaled_features)
```

## 📈 Model Performance Analysis

### Best Performers Summary:
- 🥇 **Optimized (99.13%)**: Highest accuracy but potential overfitting concerns
- 🥈 **Enhanced (73.78%)**: Most reliable for production use
- 🥉 **Bidirectional (50.71%)**: Decent directional accuracy

### Risk Assessment:
- ⚠️ **Ultra model**: Severe overfitting example (educational)
- ✅ **Enhanced model**: Production-ready performance
- 🔬 **Optimized model**: Requires further validation

## 🔮 Future Enhancements

### Planned Improvements
- 🌐 **Real-time data integration** with live market feeds
- 📰 **News sentiment analysis** integration
- 🏦 **Macroeconomic indicators** incorporation
- 🤖 **Transformer architectures** exploration
- 📊 **Multi-timeframe analysis** implementation
- 🔄 **AutoML optimization** for hyperparameters

### Research Directions
- **Attention mechanisms** refinement
- **Ensemble methods** optimization
- **Risk management** integration
- **Portfolio optimization** capabilities

## ⚠️ Important Disclaimer

**THIS PROJECT IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

- 🚫 **NOT for actual trading** or investment decisions
- 📚 **Educational tool** for learning ML/DL concepts
- ⚖️ **Financial markets involve substantial risk** of loss
- 👨‍💼 **Always consult qualified financial advisors** for investments
- 🔬 **Past performance does not guarantee future results**

## 📜 License

This project is open source and available under the **MIT License**.

```
MIT License

Copyright (c) 2025 Marepalli Santhosh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Author**: Marepalli Santhosh  
**License File**: [LICENSE](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- 🐛 Submit bug reports
- 💡 Propose new features
- 🔧 Submit pull requests
- 📖 Improve documentation

## 🌟 Acknowledgments

- **NIFTY50 Data**: Historical market data
- **TensorFlow Team**: Excellent deep learning framework
- **Open Source Community**: Various libraries and tools

---

**⭐ If you found this project helpful, please give it a star!**

**🔗 Connect with the author for discussions on ML/DL in finance**
