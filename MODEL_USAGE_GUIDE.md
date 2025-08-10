# 🚀 Using Pre-trained LSTM Models - Complete User Guide

**Comprehensive Guide for Implementing Pre-trained NIFTY50 LSTM Models on Your Own Datasets**

---

## 📋 Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Model Selection](#model-selection)
3. [Data Preparation](#data-preparation)
4. [Feature Engineering](#feature-engineering)
5. [Model Loading & Implementation](#model-loading--implementation)
6. [Prediction Pipeline](#prediction-pipeline)
7. [Performance Evaluation](#performance-evaluation)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)
10. [Best Practices](#best-practices)

---

## 🚀 Quick Start Guide

### 📦 Prerequisites

```bash
# Install required packages
pip install tensorflow>=2.19.0
pip install pandas>=1.5.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install ta>=0.10.2  # For technical indicators
pip install joblib>=1.3.0
```

### ⚡ 5-Minute Setup

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Load pre-trained model (choose one)
model = tf.keras.models.load_model('artifacts/enhanced/nifty50_lstm_model_enhanced.keras')
scaler = joblib.load('artifacts/enhanced/feature_scaler_enhanced.pkl')

print("✅ Model loaded successfully!")
print(f"📊 Model expects {model.input_shape[1]} features")
```

---

## 🎯 Model Selection

### 📊 Available Pre-trained Models

| Model | Accuracy | Use Case | Recommended For |
|-------|----------|----------|-----------------|
| **Enhanced** | 73.78% | Production use | ✅ **RECOMMENDED** - Most reliable |
| **Optimized** | 99.13% | High accuracy | ⚠️ **CAUTION** - Potential overfitting |
| **Bidirectional** | 50.71% | Experimental | 🔬 Research purposes |
| **Original** | 1.02% | Learning | 📚 Educational baseline |

### 🏆 Recommended Model: Enhanced LSTM

```python
# Load the most reliable model
model_path = 'artifacts/enhanced/nifty50_lstm_model_enhanced.keras'
scaler_path = 'artifacts/enhanced/feature_scaler_enhanced.pkl'

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# Model specifications
print("📋 Enhanced Model Specifications:")
print(f"   • Input Features: 24")
print(f"   • Sequence Length: 60 days")
print(f"   • Architecture: 3-layer LSTM")
print(f"   • Accuracy: 73.78%")
print(f"   • Status: Production-ready ✅")
```

---

## 📊 Data Preparation

### 📈 Required Data Format

Your dataset must contain these essential columns:

```python
# Required columns for stock data
required_columns = [
    'Date',      # Trading date (YYYY-MM-DD format)
    'Open',      # Opening price
    'High',      # Highest price of the day
    'Low',       # Lowest price of the day
    'Close',     # Closing price (main prediction target)
    'Volume'     # Trading volume
]

# Example data structure
sample_data = pd.DataFrame({
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'Open': [20000.0, 20100.0, 20050.0],
    'High': [20150.0, 20200.0, 20100.0],
    'Low': [19950.0, 20000.0, 19980.0],
    'Close': [20080.0, 20120.0, 20090.0],
    'Volume': [1000000, 1200000, 950000]
})
```

### 🔄 Data Loading & Validation

```python
def load_and_validate_data(file_path):
    """
    Load and validate your dataset
    
    Parameters:
    file_path (str): Path to your CSV file
    
    Returns:
    pd.DataFrame: Validated dataset
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")
    
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Basic validation
    print(f"✅ Data loaded successfully!")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"💰 Price range: ₹{df['Close'].min():.2f} to ₹{df['Close'].max():.2f}")
    
    return df

# Usage example
your_data = load_and_validate_data('your_stock_data.csv')
```

### 🧹 Data Cleaning

```python
def clean_data(df):
    """
    Clean and prepare data for processing
    
    Parameters:
    df (pd.DataFrame): Raw dataset
    
    Returns:
    pd.DataFrame: Cleaned dataset
    """
    # Remove duplicates
    df = df.drop_duplicates(subset=['Date'])
    
    # Handle missing values
    df = df.dropna()
    
    # Validate price data
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        df = df[df[col] > 0]  # Remove zero/negative prices
    
    # Validate high >= low, etc.
    df = df[df['High'] >= df['Low']]
    df = df[df['High'] >= df['Open']]
    df = df[df['High'] >= df['Close']]
    df = df[df['Low'] <= df['Open']]
    df = df[df['Low'] <= df['Close']]
    
    print(f"🧹 Data cleaned! Final shape: {df.shape}")
    return df

# Clean your data
clean_data = clean_data(your_data)
```

---

## 🛠️ Feature Engineering

### 📊 Essential Technical Indicators

The pre-trained models expect 24 specific features. Here's how to create them:

```python
import ta

def create_features(df):
    """
    Create all 24 features required by the Enhanced LSTM model
    
    Parameters:
    df (pd.DataFrame): Stock data with OHLCV columns
    
    Returns:
    pd.DataFrame: Dataset with engineered features
    """
    data = df.copy()
    
    # Core price features
    data['Close_Lag1'] = data['Close'].shift(1)
    
    # Moving averages
    data['SMA5'] = data['Close'].rolling(window=5).mean()
    data['SMA10'] = data['Close'].rolling(window=10).mean()
    data['SMA14'] = data['Close'].rolling(window=14).mean()
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['EMA50'] = data['Close'].ewm(span=50).mean()
    
    # Technical indicators using 'ta' library
    data['RSI14'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['RSI21'] = ta.momentum.RSIIndicator(data['Close'], window=21).rsi()
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    
    # Other indicators
    data['Williams_R'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['ROC_5'] = ta.momentum.ROCIndicator(data['Close'], window=5).roc()
    data['ROC_10'] = ta.momentum.ROCIndicator(data['Close'], window=10).roc()
    data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
    data['PPO'] = ta.momentum.PercentagePriceOscillator(data['Close']).ppo()
    data['UO'] = ta.momentum.UltimateOscillator(data['High'], data['Low'], data['Close']).ultimate_oscillator()
    data['TRIX'] = ta.trend.TRIXIndicator(data['Close']).trix()
    
    print(f"🔧 Features created! Dataset shape: {data.shape}")
    return data

# Create features for your data
featured_data = create_features(clean_data)
```

### 🎯 Feature Selection

```python
def get_model_features():
    """
    Get the exact 24 features used by the Enhanced LSTM model
    
    Returns:
    list: List of feature names
    """
    features = [
        'Close', 'Open', 'High', 'Low', 'Close_Lag1',
        'SMA5', 'SMA10', 'SMA14', 'SMA20', 'SMA50',
        'EMA12', 'EMA26', 'EMA50', 'RSI14', 'RSI21',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI',
        'ROC_5', 'ROC_10'
    ]
    return features

def prepare_model_data(df):
    """
    Prepare data with exact features needed by the model
    
    Parameters:
    df (pd.DataFrame): Dataset with all features
    
    Returns:
    pd.DataFrame: Model-ready dataset
    """
    features = get_model_features()
    
    # Select required features
    model_data = df[features].copy()
    
    # Remove rows with NaN values (due to technical indicators)
    model_data = model_data.dropna()
    
    print(f"📋 Model data prepared!")
    print(f"   • Features: {len(features)}")
    print(f"   • Rows: {len(model_data)}")
    print(f"   • Missing values: {model_data.isnull().sum().sum()}")
    
    return model_data

# Prepare data for the model
model_ready_data = prepare_model_data(featured_data)
```

---

## 🤖 Model Loading & Implementation

### 📚 Complete Model Loading

```python
class LSTMPredictor:
    """
    Complete LSTM prediction class for easy model usage
    """
    
    def __init__(self, model_type='enhanced'):
        """
        Initialize the predictor
        
        Parameters:
        model_type (str): 'enhanced', 'optimized', 'bidirectional', or 'original'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.sequence_length = 60
        self.features = self.get_model_features()
        
        # Load model and scaler
        self.load_model()
    
    def get_model_features(self):
        """Get features for the specific model type"""
        if self.model_type == 'enhanced':
            return [
                'Close', 'Open', 'High', 'Low', 'Close_Lag1',
                'SMA5', 'SMA10', 'SMA14', 'SMA20', 'SMA50',
                'EMA12', 'EMA26', 'EMA50', 'RSI14', 'RSI21',
                'MACD', 'MACD_Signal', 'MACD_Histogram',
                'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI',
                'ROC_5', 'ROC_10'
            ]
        elif self.model_type == 'optimized':
            return [
                'Close_Lag1', 'SMA14', 'Close', 'EMA12', 'RSI14',
                'SMA50', 'MACD', 'EMA26', 'Returns_Lag1', 'MACD_Signal',
                'BB_Position', 'Stoch_K', 'ATR', 'Volume_Ratio', 'Volume_ROC'
            ]
        else:
            return ['Close', 'Open', 'High', 'Low', 'Volume']
    
    def load_model(self):
        """Load the pre-trained model and scaler"""
        try:
            model_path = f'artifacts/{self.model_type}/nifty50_lstm_model_{self.model_type}.keras'
            scaler_path = f'artifacts/{self.model_type}/feature_scaler_{self.model_type}.pkl'
            
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            
            print(f"✅ {self.model_type.title()} model loaded successfully!")
            print(f"📊 Expected features: {len(self.features)}")
            print(f"🔧 Input shape: {self.model.input_shape}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def create_sequences(self, data):
        """
        Create sequences for LSTM input
        
        Parameters:
        data (np.array): Scaled feature data
        
        Returns:
        np.array: Sequences for LSTM
        """
        X = []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
        return np.array(X)
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Parameters:
        data (pd.DataFrame): Dataset with required features
        
        Returns:
        dict: Predictions and metadata
        """
        try:
            # Validate input data
            missing_features = [f for f in self.features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Select and scale features
            feature_data = data[self.features].values
            scaled_data = self.scaler.transform(feature_data)
            
            # Create sequences
            X = self.create_sequences(scaled_data)
            
            if len(X) == 0:
                raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1} rows.")
            
            # Make predictions
            predictions = self.model.predict(X, verbose=0)
            
            # Get prediction dates
            prediction_dates = data.index[self.sequence_length:].tolist()
            
            return {
                'predictions': predictions.flatten(),
                'dates': prediction_dates,
                'actual_prices': data['Close'].iloc[self.sequence_length:].values,
                'model_type': self.model_type,
                'total_predictions': len(predictions)
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise

# Initialize predictor
predictor = LSTMPredictor(model_type='enhanced')
```

---

## 📈 Prediction Pipeline

### 🚀 Complete Prediction Workflow

```python
def complete_prediction_pipeline(data_file, model_type='enhanced'):
    """
    Complete end-to-end prediction pipeline
    
    Parameters:
    data_file (str): Path to your CSV file
    model_type (str): Type of model to use
    
    Returns:
    dict: Complete prediction results
    """
    print("🚀 Starting prediction pipeline...")
    
    # Step 1: Load and clean data
    print("\n📊 Step 1: Loading data...")
    data = load_and_validate_data(data_file)
    clean_data = clean_data(data)
    
    # Step 2: Feature engineering
    print("\n🔧 Step 2: Creating features...")
    featured_data = create_features(clean_data)
    model_data = prepare_model_data(featured_data)
    
    # Step 3: Make predictions
    print(f"\n🤖 Step 3: Making predictions with {model_type} model...")
    predictor = LSTMPredictor(model_type=model_type)
    results = predictor.predict(model_data)
    
    # Step 4: Calculate performance metrics
    print("\n📊 Step 4: Calculating performance...")
    actual = results['actual_prices']
    predicted = results['predictions']
    
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # Calculate directional accuracy
    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    performance = {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse,
        'Directional_Accuracy': directional_accuracy
    }
    
    print(f"\n✅ Pipeline completed!")
    print(f"📈 Performance Summary:")
    print(f"   • MAE: ₹{mae:.2f}")
    print(f"   • MAPE: {mape:.2f}%")
    print(f"   • RMSE: ₹{rmse:.2f}")
    print(f"   • Directional Accuracy: {directional_accuracy:.2f}%")
    
    return {
        'predictions': results,
        'performance': performance,
        'data': model_data
    }

# Run complete pipeline
results = complete_prediction_pipeline('your_stock_data.csv', model_type='enhanced')
```

### 📊 Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(results, days_to_show=100):
    """
    Plot actual vs predicted prices
    
    Parameters:
    results (dict): Results from prediction pipeline
    days_to_show (int): Number of recent days to display
    """
    predictions = results['predictions']
    performance = results['performance']
    
    actual = predictions['actual_prices'][-days_to_show:]
    predicted = predictions['predictions'][-days_to_show:]
    dates = predictions['dates'][-days_to_show:]
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates, actual, label='Actual Price', color='blue', linewidth=2)
    plt.plot(dates, predicted, label='Predicted Price', color='red', linewidth=2, alpha=0.8)
    plt.title(f'Stock Price Prediction - {predictions["model_type"].title()} Model')
    plt.xlabel('Date')
    plt.ylabel('Price (₹)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.subplot(2, 1, 2)
    errors = actual - predicted
    plt.plot(dates, errors, color='green', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Prediction Errors')
    plt.xlabel('Date')
    plt.ylabel('Error (₹)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Add performance text
    textstr = f'MAE: ₹{performance["MAE"]:.2f}\nMAPE: {performance["MAPE"]:.2f}%\nDirectional Accuracy: {performance["Directional_Accuracy"]:.2f}%'
    plt.figtext(0.02, 0.02, textstr, fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.show()

# Visualize results
plot_predictions(results, days_to_show=100)
```

---

## 📊 Performance Evaluation

### 🎯 Comprehensive Evaluation

```python
def evaluate_model_performance(results):
    """
    Comprehensive model evaluation
    
    Parameters:
    results (dict): Results from prediction pipeline
    
    Returns:
    dict: Detailed evaluation metrics
    """
    predictions = results['predictions']
    actual = predictions['actual_prices']
    predicted = predictions['predictions']
    
    # Basic metrics
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Directional accuracy
    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    # Profit analysis (hypothetical)
    returns_actual = np.diff(actual) / actual[:-1]
    returns_predicted = np.sign(np.diff(predicted))
    
    # Calculate hypothetical trading returns
    trading_returns = returns_actual * returns_predicted
    total_return = np.sum(trading_returns) * 100
    win_rate = np.mean(trading_returns > 0) * 100
    
    evaluation = {
        'Basic Metrics': {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2
        },
        'Trading Metrics': {
            'Directional Accuracy': directional_accuracy,
            'Total Return (%)': total_return,
            'Win Rate (%)': win_rate
        },
        'Data Quality': {
            'Total Predictions': len(actual),
            'Price Range': f"₹{np.min(actual):.2f} - ₹{np.max(actual):.2f}",
            'Volatility': np.std(actual) / np.mean(actual) * 100
        }
    }
    
    # Print detailed report
    print("📊 DETAILED PERFORMANCE EVALUATION")
    print("=" * 50)
    
    for category, metrics in evaluation.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'Return' in metric or 'Accuracy' in metric or 'Rate' in metric:
                    print(f"  • {metric}: {value:.2f}%")
                elif '₹' in str(value):
                    print(f"  • {metric}: {value}")
                else:
                    print(f"  • {metric}: {value:.4f}")
            else:
                print(f"  • {metric}: {value}")
    
    return evaluation

# Evaluate model performance
evaluation = evaluate_model_performance(results)
```

---

## 🔧 Troubleshooting

### ❌ Common Issues & Solutions

#### 1. **Model Loading Errors**

```python
# Problem: Model file not found
# Solution: Check file paths
import os

def check_model_files(model_type='enhanced'):
    """Check if model files exist"""
    model_path = f'artifacts/{model_type}/nifty50_lstm_model_{model_type}.keras'
    scaler_path = f'artifacts/{model_type}/feature_scaler_{model_type}.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure you've downloaded the artifacts folder")
        return False
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file not found: {scaler_path}")
        print("💡 Make sure you've downloaded the complete artifacts folder")
        return False
    
    print(f"✅ All files found for {model_type} model")
    return True

check_model_files('enhanced')
```

#### 2. **Feature Engineering Issues**

```python
# Problem: Missing or incorrect features
# Solution: Validate feature creation

def validate_features(df, model_type='enhanced'):
    """Validate that all required features are present"""
    predictor = LSTMPredictor(model_type=model_type)
    required_features = predictor.features
    
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        print("\n💡 Solutions:")
        for feature in missing_features:
            if 'SMA' in feature:
                print(f"   • {feature}: Simple Moving Average - use df['Close'].rolling(window=X).mean()")
            elif 'EMA' in feature:
                print(f"   • {feature}: Exponential Moving Average - use df['Close'].ewm(span=X).mean()")
            elif 'RSI' in feature:
                print(f"   • {feature}: RSI indicator - use ta.momentum.RSIIndicator()")
            elif 'MACD' in feature:
                print(f"   • {feature}: MACD indicator - use ta.trend.MACD()")
        return False
    
    print(f"✅ All required features present for {model_type} model")
    return True

# Validate your features
validate_features(featured_data, 'enhanced')
```

#### 3. **Data Quality Issues**

```python
# Problem: Poor prediction performance
# Solution: Check data quality

def diagnose_data_quality(df):
    """Diagnose potential data quality issues"""
    issues = []
    
    # Check for sufficient data
    if len(df) < 100:
        issues.append(f"⚠️ Limited data: {len(df)} rows (recommended: 500+)")
    
    # Check for missing values
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    if missing_pct > 5:
        issues.append(f"⚠️ High missing values: {missing_pct:.1f}%")
    
    # Check price volatility
    price_volatility = df['Close'].std() / df['Close'].mean() * 100
    if price_volatility > 50:
        issues.append(f"⚠️ High volatility: {price_volatility:.1f}%")
    elif price_volatility < 5:
        issues.append(f"⚠️ Low volatility: {price_volatility:.1f}% (may be less predictable)")
    
    # Check for outliers
    q1, q3 = df['Close'].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((df['Close'] < (q1 - 1.5 * iqr)) | (df['Close'] > (q3 + 1.5 * iqr))).sum()
    outlier_pct = outliers / len(df) * 100
    if outlier_pct > 5:
        issues.append(f"⚠️ High outliers: {outlier_pct:.1f}%")
    
    if issues:
        print("🔍 DATA QUALITY ISSUES DETECTED:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Consider data cleaning or using more data")
    else:
        print("✅ Data quality looks good!")
    
    return len(issues) == 0

# Diagnose your data
diagnose_data_quality(clean_data)
```

---

## 🚀 Advanced Usage

### 🔄 Real-time Prediction

```python
def real_time_predictor(data_source, model_type='enhanced', update_interval=60):
    """
    Real-time prediction system
    
    Parameters:
    data_source (callable): Function that returns latest data
    model_type (str): Model to use
    update_interval (int): Update interval in seconds
    """
    import time
    
    predictor = LSTMPredictor(model_type=model_type)
    
    print(f"🔄 Starting real-time predictions with {model_type} model...")
    print(f"⏱️ Update interval: {update_interval} seconds")
    
    while True:
        try:
            # Get latest data
            latest_data = data_source()
            
            if len(latest_data) >= predictor.sequence_length + 50:  # Minimum data needed
                # Make prediction
                results = predictor.predict(latest_data)
                latest_prediction = results['predictions'][-1]
                current_price = latest_data['Close'].iloc[-1]
                
                print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"💰 Current: ₹{current_price:.2f}")
                print(f"🔮 Predicted: ₹{latest_prediction:.2f}")
                print(f"📊 Change: {((latest_prediction - current_price) / current_price * 100):+.2f}%")
                print("-" * 50)
            
            time.sleep(update_interval)
            
        except KeyboardInterrupt:
            print("\n⏹️ Real-time prediction stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(update_interval)

# Example usage (you would need to implement your data_source function)
# real_time_predictor(your_data_source_function, 'enhanced', 60)
```

### 📊 Model Comparison

```python
def compare_all_models(data):
    """
    Compare predictions from all available models
    
    Parameters:
    data (pd.DataFrame): Your prepared dataset
    
    Returns:
    dict: Comparison results
    """
    models = ['enhanced', 'optimized', 'bidirectional', 'original']
    results = {}
    
    print("🔄 Comparing all models...")
    
    for model_type in models:
        try:
            print(f"\n📊 Testing {model_type} model...")
            predictor = LSTMPredictor(model_type=model_type)
            
            # Make predictions
            pred_results = predictor.predict(data)
            
            # Calculate metrics
            actual = pred_results['actual_prices']
            predicted = pred_results['predictions']
            
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Directional accuracy
            actual_direction = np.sign(np.diff(actual))
            predicted_direction = np.sign(np.diff(predicted))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            results[model_type] = {
                'MAE': mae,
                'MAPE': mape,
                'Directional_Accuracy': directional_accuracy,
                'Predictions': len(predicted)
            }
            
            print(f"✅ {model_type}: {directional_accuracy:.1f}% accuracy")
            
        except Exception as e:
            print(f"❌ {model_type} failed: {e}")
            results[model_type] = None
    
    # Print comparison table
    print("\n📊 MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<15} {'Accuracy':<12} {'MAE':<10} {'MAPE':<8}")
    print("-" * 60)
    
    for model, metrics in results.items():
        if metrics:
            print(f"{model:<15} {metrics['Directional_Accuracy']:<8.1f}%    ₹{metrics['MAE']:<8.0f} {metrics['MAPE']:<6.1f}%")
        else:
            print(f"{model:<15} {'Failed':<12}")
    
    return results

# Compare all models
comparison = compare_all_models(model_ready_data)
```

---

## 💡 Best Practices

### ✅ Do's

1. **Data Quality First**
   ```python
   # Always validate your data
   assert len(data) >= 100, "Need at least 100 data points"
   assert data['Close'].min() > 0, "Invalid price data"
   assert data['Date'].is_monotonic_increasing, "Data must be chronologically sorted"
   ```

2. **Feature Engineering**
   ```python
   # Create features in the correct order
   # Always handle missing values after technical indicators
   data = create_features(data)
   data = data.dropna()  # Remove NaN from technical indicators
   ```

3. **Model Selection**
   ```python
   # Start with Enhanced model for most use cases
   # Only use Optimized model with caution (potential overfitting)
   predictor = LSTMPredictor(model_type='enhanced')
   ```

4. **Validation**
   ```python
   # Always validate predictions make sense
   assert np.all(predictions > 0), "Predictions must be positive"
   assert np.all(np.isfinite(predictions)), "Predictions must be finite"
   ```

### ❌ Don'ts

1. **Don't Skip Data Validation**
   ```python
   # ❌ Wrong
   predictions = model.predict(raw_data)
   
   # ✅ Correct
   cleaned_data = validate_and_clean(raw_data)
   predictions = model.predict(cleaned_data)
   ```

2. **Don't Ignore Feature Requirements**
   ```python
   # ❌ Wrong - using different features
   my_features = ['Close', 'Volume', 'Custom_Indicator']
   
   # ✅ Correct - using exact model features
   required_features = predictor.get_model_features()
   ```

3. **Don't Use Overfitted Models Blindly**
   ```python
   # ❌ Be cautious with 99%+ accuracy models
   # ✅ Prefer robust, validated models
   ```

### 🎯 Performance Tips

1. **Optimize for Your Use Case**
   ```python
   # For real trading: Use Enhanced model (73.78% accuracy)
   # For research: Try Optimized model with validation
   # For learning: Start with Original model
   ```

2. **Monitor Performance**
   ```python
   def monitor_performance(predictions, actual):
       recent_accuracy = calculate_recent_accuracy(predictions[-30:], actual[-30:])
       if recent_accuracy < 50:
           print("⚠️ Model performance degrading - consider retraining")
   ```

3. **Regular Updates**
   ```python
   # Update your data regularly
   # Retrain models periodically
   # Monitor for concept drift
   ```

---

## 📞 Support & Resources

### 🆘 Getting Help

1. **Check troubleshooting section** for common issues
2. **Validate your data** using provided functions
3. **Start with Enhanced model** for best results
4. **Review error messages** carefully

### 📚 Additional Resources

- **Technical Indicators**: [TA-Lib Documentation](https://ta-lib.org/)
- **TensorFlow**: [Official Documentation](https://tensorflow.org/)
- **Time Series**: [Best Practices Guide](https://www.tensorflow.org/tutorials/structured_data/time_series)

### 🐛 Reporting Issues

If you encounter problems:

1. **Provide complete error messages**
2. **Include data sample** (anonymized)
3. **Specify model type** you're using
4. **Include Python/package versions**

---

## 🎯 Example Complete Implementation

Here's a complete example putting everything together:

```python
"""
Complete example: Using Enhanced LSTM model on your own stock data
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import ta
import matplotlib.pyplot as plt
from datetime import datetime

def complete_example():
    """Complete implementation example"""
    
    print("🚀 COMPLETE LSTM PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Step 1: Load your data (replace with your file)
    print("\n📊 Step 1: Loading data...")
    # data = pd.read_csv('your_stock_data.csv')  # Your data file
    # For this example, we'll create sample data
    
    # Step 2: Load pre-trained model
    print("\n🤖 Step 2: Loading Enhanced LSTM model...")
    predictor = LSTMPredictor(model_type='enhanced')
    
    # Step 3: Create sample data (replace with your actual data loading)
    print("\n🔧 Step 3: Preparing sample data...")
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic stock data
    initial_price = 20000
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': [p * 0.999 for p in prices],
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Step 4: Feature engineering
    print("\n🔧 Step 4: Creating features...")
    featured_data = create_features(sample_data)
    model_data = prepare_model_data(featured_data)
    
    # Step 5: Make predictions
    print("\n🔮 Step 5: Making predictions...")
    results = predictor.predict(model_data)
    
    # Step 6: Evaluate and visualize
    print("\n📊 Step 6: Evaluating results...")
    evaluation = evaluate_model_performance({'predictions': results})
    
    # Step 7: Plot results
    print("\n📈 Step 7: Creating visualizations...")
    plot_predictions({'predictions': results, 'performance': evaluation['Trading Metrics']}, days_to_show=60)
    
    print("\n✅ Complete example finished successfully!")
    return results, evaluation

# Run the complete example
if __name__ == "__main__":
    results, evaluation = complete_example()
```

---

**🎉 Congratulations! You now have everything you need to use the pre-trained LSTM models on your own datasets.**

**Remember**: 
- 📊 **Start with clean, validated data**
- 🔧 **Create features correctly** 
- 🤖 **Use Enhanced model** for production
- 📈 **Validate predictions** thoroughly
- ⚠️ **This is for educational purposes** - not financial advice

**⭐ Happy predicting!**
