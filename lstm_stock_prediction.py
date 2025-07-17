import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Data processing libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Optional: for data download (uncomment if needed)
import yfinance as yf

# =============================================================================
# HYPERPARAMETERS - Easy to modify for experimentation
# =============================================================================
TIME_STEPS = 60          # Number of time steps to look back
LSTM_UNITS_1 = 100       # Units in first LSTM layer
LSTM_UNITS_2 = 50        # Units in second LSTM layer
DROPOUT_RATE = 0.2       # Dropout rate for regularization
EPOCHS = 100             # Maximum training epochs
BATCH_SIZE = 32          # Batch size for training
VALIDATION_SPLIT = 0.1   # Validation split during training
PATIENCE = 10            # Early stopping patience
TRAIN_SPLIT = 0.8        # Train-test split ratio

# =============================================================================
# DATA LOADING AND INITIAL SETUP
# =============================================================================
def print_user_guidance():
    print("""
Welcome to the LSTM Stock Price Prediction Script!
==================================================
This script predicts stock prices using an LSTM neural network.

You can:
- Use your own CSV file: Pass the path as an argument (python lstm_stock_prediction.py your_data.csv)
- Use yfinance (uncomment the relevant lines in the code and provide a stock symbol)
- Use sample generated data (default)

Main Steps:
1. Data Loading
2. Feature Engineering
3. Data Preprocessing
4. Model Building & Training
5. Evaluation & Visualization
6. Save Model

For best results, provide a CSV with columns: Date, Open, High, Low, Close, Volume
""")

def load_stock_data(symbol=None, csv_path=None, start_date='2020-01-01', end_date='2024-01-01'):
    """
    Load stock data from either yfinance or CSV file
    
    Parameters:
    symbol (str): Stock symbol (e.g., 'AAPL') for yfinance
    csv_path (str): Path to CSV file with stock data
    start_date (str): Start date for data download
    end_date (str): End date for data download
    
    Returns:
    pandas.DataFrame: Stock data with required columns
    """
    if csv_path:
        print(f"Loading data from CSV: {csv_path}")
        try:
            data = pd.read_csv(csv_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date')
            print("First 5 rows of loaded data:")
            print(data.head())
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
    else:
        # Use Tata Motors stock data from yfinance by default
        print("Downloading Tata Motors stock data from yfinance (symbol: TATAMOTORS.NS)...")
        try:
            data = yf.download('TATAMOTORS.NS', start=start_date, end=end_date)
            if data is None or data.empty:
                print("No data was downloaded for Tata Motors (TATAMOTORS.NS). Please check your internet connection or the symbol.")
                sys.exit(1)
            data = data.reset_index()
            # Flatten columns if multi-indexed (from yfinance)
            if isinstance(data.columns, pd.MultiIndex):
                # For each column, join non-empty levels with '_'
                data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
                # If columns are like 'Close_TATAMOTORS.NS', rename to 'Close', etc.
                rename_map = {c: c.split('_')[0] for c in data.columns if '_' in c}
                data = data.rename(columns=rename_map)
            # Ensure only required columns are present
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"Missing columns after flattening: {missing_cols}")
                sys.exit(1)
            print("First 5 rows of Tata Motors data:")
            print(data.head())
        except Exception as e:
            print(f"Error downloading Tata Motors data: {e}")
            sys.exit(1)
    return data

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def add_technical_indicators(data):
    """
    Add technical indicators to the dataset
    
    Parameters:
    data (pandas.DataFrame): Stock data
    
    Returns:
    pandas.DataFrame: Data with additional technical indicators
    """
    print("Adding technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, etc.)...")
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Price-based features
    df['Price_change'] = df['Close'].pct_change()
    df['High_Low_ratio'] = df['High'] / df['Low']
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    print("First 5 rows after adding technical indicators:")
    print(df.head())
    return df

def add_external_features_placeholder(data):
    """
    Placeholder for adding external features like sentiment scores or macro indicators
    
    Parameters:
    data (pandas.DataFrame): Stock data
    
    Returns:
    pandas.DataFrame: Data with placeholder for external features
    """
    df = data.copy()
    
    # Placeholder for sentiment scores (replace with actual sentiment data)
    # df['Sentiment_Score'] = sentiment_data['Score']
    
    # Placeholder for macroeconomic indicators
    # df['Interest_Rate'] = macro_data['Interest_Rate']
    # df['VIX'] = macro_data['VIX']
    
    return df

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
def create_sequences(data, time_steps, target_col='Close'):
    """
    Create sequences for LSTM training
    
    Parameters:
    data (numpy.array): Preprocessed data
    time_steps (int): Number of time steps to look back
    target_col (str): Name of target column
    
    Returns:
    tuple: (X, y) sequences
    """
    X, y = [], []
    
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, -1])  # Assuming target is the last column
    
    return np.array(X), np.array(y)

def preprocess_data(data, time_steps=TIME_STEPS, train_split=TRAIN_SPLIT):
    """
    Complete data preprocessing pipeline
    
    Parameters:
    data (pandas.DataFrame): Raw stock data
    time_steps (int): Number of time steps for sequences
    train_split (float): Ratio for train-test split
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, scaler_X, scaler_y)
    """
    print("Preprocessing: Adding technical indicators and external features...")
    data = add_technical_indicators(data)
    data = add_external_features_placeholder(data)
    print("Dropping rows with NaN values...")
    data = data.dropna()
    print(f"Data shape after dropping NaNs: {data.shape}")
    
    # Select features for training
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_30', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_signal', 'MACD_histogram', 'RSI',
        'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
        'Price_change', 'High_Low_ratio', 'Volume_ratio', 'Volatility'
    ]
    
    # Ensure all feature columns exist
    available_features = [col for col in feature_columns if col in data.columns]
    features = data[available_features].values
    target = data['Close'].values.reshape(-1, 1)
    
    # Scale features and target separately
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Split data chronologically
    split_index = int(len(features) * train_split)
    
    # Fit scalers on training data only
    X_train_scaled = scaler_X.fit_transform(features[:split_index])
    y_train_scaled = scaler_y.fit_transform(target[:split_index])
    
    # Transform test data
    X_test_scaled = scaler_X.transform(features[split_index:])
    y_test_scaled = scaler_y.transform(target[split_index:])
    
    # Combine features and target for sequence creation
    train_data = np.hstack([X_train_scaled, y_train_scaled])
    test_data = np.hstack([X_test_scaled, y_test_scaled])
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, time_steps)
    X_test, y_test = create_sequences(test_data, time_steps)
    
    print(f"Feature columns used: {available_features}")
    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

# =============================================================================
# LSTM MODEL ARCHITECTURE
# =============================================================================
def build_lstm_model(input_shape, lstm_units_1=LSTM_UNITS_1, lstm_units_2=LSTM_UNITS_2, dropout_rate=DROPOUT_RATE):
    """
    Build LSTM model architecture
    
    Parameters:
    input_shape (tuple): Shape of input data
    lstm_units_1 (int): Units in first LSTM layer
    lstm_units_2 (int): Units in second LSTM layer
    dropout_rate (float): Dropout rate
    
    Returns:
    tensorflow.keras.Model: Compiled LSTM model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(units=lstm_units_1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        LSTM(units=lstm_units_2, return_sequences=False),
        Dropout(dropout_rate),
        
        # Dense output layer
        Dense(units=1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def build_advanced_lstm_model(input_shape, lstm_units_1=LSTM_UNITS_1, lstm_units_2=LSTM_UNITS_2, dropout_rate=DROPOUT_RATE):
    """
    Advanced LSTM model with additional layers (optional)
    
    Parameters:
    input_shape (tuple): Shape of input data
    lstm_units_1 (int): Units in first LSTM layer
    lstm_units_2 (int): Units in second LSTM layer
    dropout_rate (float): Dropout rate
    
    Returns:
    tensorflow.keras.Model: Compiled advanced LSTM model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(units=lstm_units_1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        LSTM(units=lstm_units_2, return_sequences=True),
        Dropout(dropout_rate),
        
        # Third LSTM layer
        LSTM(units=lstm_units_2//2, return_sequences=False),
        Dropout(dropout_rate),
        
        # Dense layers
        Dense(units=25, activation='relu'),
        Dropout(dropout_rate),
        Dense(units=1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_model(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                validation_split=VALIDATION_SPLIT, patience=PATIENCE):
    """
    Train the LSTM model
    
    Parameters:
    model: Compiled Keras model
    X_train, y_train: Training data
    epochs (int): Maximum number of epochs
    batch_size (int): Batch size
    validation_split (float): Validation split ratio
    patience (int): Early stopping patience
    
    Returns:
    tuple: (trained_model, history)
    """
    print(f"Training model for up to {epochs} epochs (early stopping patience: {patience})...")
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

# =============================================================================
# PREDICTION AND EVALUATION
# =============================================================================
def make_predictions(model, X_train, X_test, scaler_y):
    """
    Make predictions and inverse transform
    
    Parameters:
    model: Trained model
    X_train, X_test: Input data
    scaler_y: Scaler for target variable
    
    Returns:
    tuple: (train_predictions, test_predictions)
    """
    # Make predictions
    train_pred_scaled = model.predict(X_train)
    test_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions
    train_predictions = scaler_y.inverse_transform(train_pred_scaled)
    test_predictions = scaler_y.inverse_transform(test_pred_scaled)
    
    return train_predictions, test_predictions

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    
    Parameters:
    y_true: True values
    y_pred: Predicted values
    
    Returns:
    dict: Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print("\nMetric Explanations:")
    print("MAE  (Mean Absolute Error): Average absolute difference between actual and predicted values.")
    print("MSE  (Mean Squared Error): Average squared difference between actual and predicted values.")
    print("RMSE (Root Mean Squared Error): Square root of MSE, interpretable in original units.")
    print("MAPE (Mean Absolute Percentage Error): Average absolute percent error.")
    print("R2   (R-squared): Proportion of variance explained by the model (1 is perfect).")
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def calculate_direction_accuracy(y_true, y_pred):
    """
    Calculate Movement Direction Accuracy (MDA)
    
    Parameters:
    y_true: True values
    y_pred: Predicted values
    
    Returns:
    float: Direction accuracy percentage
    """
    # Calculate actual and predicted directions
    actual_direction = np.diff(y_true.flatten()) > 0
    predicted_direction = np.diff(y_pred.flatten()) > 0
    
    # Calculate accuracy
    direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    return direction_accuracy

def visualize_predictions(y_true, y_pred, title="Stock Price Prediction"):
    """
    Visualize actual vs predicted prices
    
    Parameters:
    y_true: True values
    y_pred: Predicted values
    title: Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Price', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted Price', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_training_history(history):
    """
    Plot training history
    
    Parameters:
    history: Training history object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """
    Main execution function
    """
    print_user_guidance()
    print("LSTM Stock Price Prediction Model")
    print("=" * 50)
    # Parse command-line arguments for CSV path
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            sys.exit(1)
    # Load data
    print("Loading stock data...")
    data = load_stock_data(csv_path=csv_path)  # You can specify csv_path or symbol here
    print(f"Loaded {len(data)} rows of data")
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(data)
    # Build model
    print("Building LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    print(model.summary())
    # Train model
    print("Training model...")
    model, history = train_model(model, X_train, y_train)
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    # Make predictions
    print("Making predictions...")
    train_predictions, test_predictions = make_predictions(model, X_train, X_test, scaler_y)
    # Inverse transform actual values for evaluation
    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    # Calculate metrics
    print("\nTraining Set Metrics:")
    train_metrics = calculate_metrics(y_train_actual, train_predictions)
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nTest Set Metrics:")
    test_metrics = calculate_metrics(y_test_actual, test_predictions)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    # Calculate direction accuracy
    train_direction_acc = calculate_direction_accuracy(y_train_actual, train_predictions)
    test_direction_acc = calculate_direction_accuracy(y_test_actual, test_predictions)
    print(f"\nMovement Direction Accuracy:")
    print(f"Training Set: {train_direction_acc:.2f}%")
    print(f"Test Set: {test_direction_acc:.2f}%")
    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(y_test_actual, test_predictions, "Test Set: Actual vs Predicted Prices")
    # Save model
    model.save('lstm_stock_prediction_model.h5')
    print("\nModel saved as 'lstm_stock_prediction_model.h5'")
    return model, scaler_X, scaler_y

# =============================================================================
# PREDICTION FUNCTION FOR NEW DATA
# =============================================================================
def predict_next_day(model, recent_data, scaler_X, scaler_y, time_steps=TIME_STEPS):
    """
    Predict next day's stock price
    
    Parameters:
    model: Trained model
    recent_data: Recent stock data (last time_steps days)
    scaler_X: Feature scaler
    scaler_y: Target scaler
    time_steps: Number of time steps
    
    Returns:
    float: Predicted next day price
    """
    # Preprocess recent data
    recent_data_processed = add_technical_indicators(recent_data)
    recent_data_processed = recent_data_processed.dropna()
    
    # Select same features used in training
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_30', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_signal', 'MACD_histogram', 'RSI',
        'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
        'Price_change', 'High_Low_ratio', 'Volume_ratio', 'Volatility'
    ]
    
    available_features = [col for col in feature_columns if col in recent_data_processed.columns]
    features = recent_data_processed[available_features].values
    
    # Scale features
    features_scaled = scaler_X.transform(features)
    
    # Get last time_steps for prediction
    if len(features_scaled) >= time_steps:
        X_pred = features_scaled[-time_steps:].reshape(1, time_steps, -1)
        
        # Make prediction
        pred_scaled = model.predict(X_pred)
        pred_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
        
        return pred_price
    else:
        raise ValueError(f"Need at least {time_steps} days of data for prediction")

if __name__ == "__main__":
    # Run main execution
    model, scaler_X, scaler_y = main()
    
    # Example of predicting next day (uncomment to use)
    # next_day_prediction = predict_next_day(model, data, scaler_X, scaler_y)
    # print(f"Next day predicted price: ${next_day_prediction:.2f}")