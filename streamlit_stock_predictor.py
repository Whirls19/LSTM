import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import sys
import os
from datetime import datetime, timedelta

# Data processing libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Data download
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="LSTM Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
TIME_STEPS = 60
LSTM_UNITS_1 = 100
LSTM_UNITS_2 = 50
DROPOUT_RATE = 0.2
EPOCHS = 50  # Reduced for faster training in Streamlit
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
PATIENCE = 10
TRAIN_SPLIT = 0.8

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_data
def load_stock_data(symbol, start_date, end_date):
    """Load stock data with caching for better performance"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data is None or data.empty:
            return None, f"No data found for {symbol}"
        
        data = data.reset_index()
        # Handle multi-index columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
            rename_map = {c: c.split('_')[0] for c in data.columns if '_' in c}
            data = data.rename(columns=rename_map)
        
        return data, None
    except Exception as e:
        return None, f"Error downloading data: {str(e)}"

def add_technical_indicators(data):
    """Add technical indicators to the dataset"""
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
    
    return df

def create_sequences(data, time_steps, target_col='Close'):
    """Create sequences for LSTM training"""
    X, y = [], []
    
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, -1])  # Assuming target is the last column
    
    return np.array(X), np.array(y)

def preprocess_data(data, time_steps=TIME_STEPS, train_split=TRAIN_SPLIT):
    """Complete data preprocessing pipeline"""
    data = add_technical_indicators(data)
    data = data.dropna()
    
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
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, available_features

def build_lstm_model(input_shape, lstm_units_1=LSTM_UNITS_1, lstm_units_2=LSTM_UNITS_2, dropout_rate=DROPOUT_RATE):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(units=lstm_units_1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=lstm_units_2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=1, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

@st.cache_resource
def train_model_cached(X_train, y_train, input_shape):
    """Train model with caching"""
    model = build_lstm_model(input_shape)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model, history

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def calculate_direction_accuracy(y_true, y_pred):
    """Calculate Movement Direction Accuracy"""
    actual_direction = np.diff(y_true.flatten()) > 0
    predicted_direction = np.diff(y_pred.flatten()) > 0
    direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    return direction_accuracy

def create_prediction_plot(data, y_test_actual, test_predictions, symbol):
    """Create interactive prediction plot using Plotly"""
    # Create date range for test data
    test_dates = data['Date'].iloc[-len(y_test_actual):].values
    
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test_actual.flatten(),
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>Actual:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_predictions.flatten(),
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_technical_indicators_plot(data):
    """Create technical indicators plot"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Moving Averages', 'MACD', 'RSI'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_10'], name='SMA 10', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_30'], name='SMA 30', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD_signal'], name='Signal', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Bar(x=data['Date'], y=data['MACD_histogram'], name='Histogram', marker_color='gray'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(height=600, showlegend=True, template='plotly_white')
    
    return fig

def predict_next_days(model, recent_data, scaler_X, scaler_y, days_ahead=5):
    """Predict next few days"""
    # Start with the most recent processed data
    recent_data_processed = add_technical_indicators(recent_data)
    recent_data_processed = recent_data_processed.dropna()
    
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_30', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_signal', 'MACD_histogram', 'RSI',
        'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
        'Price_change', 'High_Low_ratio', 'Volume_ratio', 'Volatility'
    ]
    
    available_features = [col for col in feature_columns if col in recent_data_processed.columns]
    features = recent_data_processed[available_features].values
    target = recent_data_processed['Close'].values.reshape(-1, 1)
    
    if len(features) < TIME_STEPS:
        return None, "Insufficient data for prediction"
    
    # We'll keep a DataFrame for updating technicals
    df_future = recent_data_processed.copy()
    predictions = []
    
    for _ in range(days_ahead):
        # Scale features and target for the last TIME_STEPS rows
        features = df_future[available_features].values
        target = df_future['Close'].values.reshape(-1, 1)
        features_scaled = scaler_X.transform(features)
        target_scaled = scaler_y.transform(target)
        # Concatenate features and target as in training
        data_scaled = np.hstack([features_scaled, target_scaled])
        current_sequence = data_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, -1)
        # Predict next close price
        pred_scaled = model.predict(current_sequence, verbose=0)
        pred_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
        predictions.append(pred_price)
        # Create new row for next day by copying last row and updating only 'Close'
        new_row = df_future.iloc[-1].copy()
        new_row['Close'] = pred_price
        # For all other features, just copy the last value (do not recompute technicals)
        df_future = pd.concat([df_future, pd.DataFrame([new_row])], ignore_index=True)
        # Do NOT recompute technical indicators or dropna
    
    return predictions, None

# =============================================================================
# STREAMLIT APP
# =============================================================================
def main():
    st.markdown('<h1 class="main-header">📈 LSTM Stock Price Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for user inputs
    st.sidebar.header("🔧 Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT, TATAMOTORS.NS)"
    ).upper()
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*2),
            max_value=datetime.now() - timedelta(days=1)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    epochs = st.sidebar.slider("Epochs", 10, 100, EPOCHS)
    lstm_units_1 = st.sidebar.slider("LSTM Units 1", 50, 200, LSTM_UNITS_1)
    lstm_units_2 = st.sidebar.slider("LSTM Units 2", 25, 100, LSTM_UNITS_2)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, DROPOUT_RATE, 0.1)

    with st.sidebar.expander("❓ Parameter Explanations"):
        st.markdown("""
        - **Epochs**: Number of times the model sees the entire training data. More epochs can improve learning, but too many may cause overfitting.
        - **LSTM Units 1**: Number of memory cells (neurons) in the first LSTM layer. More units = more learning capacity.
        - **LSTM Units 2**: Number of memory cells in the second LSTM layer. Controls model depth and complexity.
        - **Dropout Rate**: Fraction of neurons randomly dropped during training to prevent overfitting (e.g., 0.2 = 20% dropped).
        - **Time Steps**: Number of previous days the model looks at to make a prediction (fixed at 60 in this app).
        - **Batch Size**: Number of samples processed before updating model weights (fixed at 32 in this app).
        - **Validation Split**: Fraction of training data used for validation (fixed at 0.1 = 10%).
        - **Patience**: Number of epochs to wait for improvement before stopping training early (fixed at 10).
        - **Train Split**: Fraction of data used for training (fixed at 0.8 = 80%).
        - **LSTM (Long Short-Term Memory)**: A type of neural network layer designed to learn from sequences and remember information over time. Ideal for time series like stock prices.
        """)
    
    # Load data button
    if st.sidebar.button("🚀 Load Data & Train Model", type="primary"):
        if not symbol:
            st.error("Please enter a stock symbol")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load data
            status_text.text("Loading stock data...")
            progress_bar.progress(10)
            
            data, error = load_stock_data(symbol, start_date, end_date)
            if error:
                st.error(error)
                return
            
            if len(data) < 100:
                st.error("Insufficient data for training. Please select a longer date range.")
                return
            
            # Preprocess data
            status_text.text("Preprocessing data...")
            progress_bar.progress(30)
            
            X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names = preprocess_data(data)
            
            # Train model
            status_text.text("Training LSTM model...")
            progress_bar.progress(50)
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            model, history = train_model_cached(X_train, y_train, input_shape)
            
            # Make predictions
            status_text.text("Making predictions...")
            progress_bar.progress(80)
            
            train_predictions = model.predict(X_train, verbose=0)
            test_predictions = model.predict(X_test, verbose=0)
            
            # Inverse transform
            train_predictions = scaler_y.inverse_transform(train_predictions)
            test_predictions = scaler_y.inverse_transform(test_predictions)
            y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
            y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            
            # Store results in session state
            st.session_state.data = data
            st.session_state.model = model
            st.session_state.scaler_X = scaler_X
            st.session_state.scaler_y = scaler_y
            st.session_state.history = history
            st.session_state.train_predictions = train_predictions
            st.session_state.test_predictions = test_predictions
            st.session_state.y_train_actual = y_train_actual
            st.session_state.y_test_actual = y_test_actual
            st.session_state.feature_names = feature_names
            st.session_state.symbol = symbol
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return
    
    # Display results if available
    if 'data' in st.session_state:
        data = st.session_state.data
        model = st.session_state.model
        scaler_X = st.session_state.scaler_X
        scaler_y = st.session_state.scaler_y
        history = st.session_state.history
        train_predictions = st.session_state.train_predictions
        test_predictions = st.session_state.test_predictions
        y_train_actual = st.session_state.y_train_actual
        y_test_actual = st.session_state.y_test_actual
        feature_names = st.session_state.feature_names
        symbol = st.session_state.symbol
        
        # Main content
        st.success(f"✅ Model trained successfully for {symbol}!")
        
        # Stock info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            st.metric("Daily Change", f"${price_change:.2f}")
        with col3:
            total_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            st.metric("Total Return", f"{total_return:.2f}%")
        with col4:
            volatility = data['Close'].pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "📈 Technical Indicators", "📋 Metrics", "🔮 Future Predictions"])
        
        with tab1:
            st.subheader("Stock Price Predictions")
            
            # Prediction plot
            fig = create_prediction_plot(data, y_test_actual, test_predictions, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Training history
            col1, col2 = st.columns(2)
            with col1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                fig_loss.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Scatter(y=history.history['mae'], name='Training MAE'))
                fig_mae.add_trace(go.Scatter(y=history.history['val_mae'], name='Validation MAE'))
                fig_mae.update_layout(title='Model MAE', xaxis_title='Epoch', yaxis_title='MAE')
                st.plotly_chart(fig_mae, use_container_width=True)
        
        with tab2:
            st.subheader("Technical Indicators")
            tech_data = add_technical_indicators(data)
            fig_tech = create_technical_indicators_plot(tech_data)
            st.plotly_chart(fig_tech, use_container_width=True)
        
        with tab3:
            st.subheader("Model Performance Metrics")
            
            # Calculate metrics
            train_metrics = calculate_metrics(y_train_actual, train_predictions)
            test_metrics = calculate_metrics(y_test_actual, test_predictions)
            train_direction_acc = calculate_direction_accuracy(y_train_actual, train_predictions)
            test_direction_acc = calculate_direction_accuracy(y_test_actual, test_predictions)
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Training Set Metrics")
                for metric, value in train_metrics.items():
                    st.metric(metric, f"{value:.4f}")
                st.metric("Direction Accuracy", f"{train_direction_acc:.2f}%")
            
            with col2:
                st.markdown("### Test Set Metrics")
                for metric, value in test_metrics.items():
                    st.metric(metric, f"{value:.4f}")
                st.metric("Direction Accuracy", f"{test_direction_acc:.2f}%")
            
            # Metric explanations
            with st.expander("📖 Metric Explanations"):
                st.markdown("""
                - **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values
                - **MSE (Mean Squared Error)**: Average squared difference between actual and predicted values
                - **RMSE (Root Mean Squared Error)**: Square root of MSE, interpretable in original units
                - **MAPE (Mean Absolute Percentage Error)**: Average absolute percent error
                - **R² (R-squared)**: Proportion of variance explained by the model (1 is perfect)
                - **Direction Accuracy**: Percentage of times the model correctly predicts price movement direction
                """)
        
        with tab4:
            st.subheader("Future Price Predictions")
            
            # Predict next few days
            days_ahead = st.slider("Days to predict ahead", 1, 10, 5)
            
            if st.button("🔮 Predict Future Prices"):
                predictions, error = predict_next_days(model, data, scaler_X, scaler_y, days_ahead)
                
                if error:
                    st.error(error)
                else:
                    # Create future dates
                    last_date = data['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
                    
                    # Display predictions
                    st.markdown("### Predicted Prices")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create prediction plot
                        fig_future = go.Figure()
                        
                        # Add historical data
                        fig_future.add_trace(go.Scatter(
                            x=data['Date'].iloc[-30:],  # Last 30 days
                            y=data['Close'].iloc[-30:],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Add predictions
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions,
                            mode='lines+markers',
                            name='Predictions',
                            line=dict(color='red', dash='dash'),
                            marker=dict(size=8)
                        ))
                        
                        fig_future.update_layout(
                            title=f'{symbol} Future Price Predictions',
                            xaxis_title='Date',
                            yaxis_title='Stock Price ($)',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_future, use_container_width=True)
                    
                    with col2:
                        st.markdown("### Predictions Table")
                        pred_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted Price': [f"${p:.2f}" for p in predictions]
                        })
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Prediction summary
                        current_price = data['Close'].iloc[-1]
                        avg_prediction = np.mean(predictions)
                        price_change_pred = ((avg_prediction - current_price) / current_price) * 100
                        
                        st.markdown("### Summary")
                        st.metric("Average Predicted Price", f"${avg_prediction:.2f}")
                        st.metric("Predicted Change", f"{price_change_pred:.2f}%")
        
        # Download section
        st.sidebar.markdown("---")
        st.sidebar.subheader("💾 Download Results")
        
        if st.sidebar.button("📥 Download Predictions CSV"):
            # Create download dataframe
            test_dates = data['Date'].iloc[-len(y_test_actual):].values
            download_df = pd.DataFrame({
                'Date': test_dates,
                'Actual_Price': y_test_actual.flatten(),
                'Predicted_Price': test_predictions.flatten()
            })
            
            csv = download_df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{symbol}_predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 