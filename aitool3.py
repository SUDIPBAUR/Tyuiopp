import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta import add_all_ta_features
import os

# Binance setup with error handling
def setup_exchange():
    api_key = (zSytUpX7UqgSZH2Ig9DrTpEQTR77wBqTwldiLIrwoJWpXUU1ksUAywdxUWVMtCsm)
    api_secret = os.getenv(orEZ15ayLiAfBjwYuEDLlTsNwrp4hIec8GSyWvOqmEbImpzsjJU9PmKbpxRqijzu)
    
    if not api_key or not api_secret:
        st.error("Binance API keys not found in environment variables!")
        st.stop()
    
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'adjustForTimeDifference': True,
        }
    })
    
    # Test connectivity
    try:
        exchange.fetch_status()
    except Exception as e:
        st.error(f"Binance connection failed: {str(e)}")
        st.error("Possible reasons: 1) Invalid API keys 2) Network issues 3) Binance maintenance")
        st.stop()
    
    return exchange

exchange = setup_exchange()

# Fetch historical data with retries
def fetch_data():
    max_retries = 3
    symbol = 'BTC/USDT'
    timeframe = '1h'
    
    for _ in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except ccxt.NetworkError as e:
            st.warning(f"Network error: {str(e)} - retrying...")
            exchange.sleep(5000)
        except ccxt.ExchangeError as e:
            st.error(f"Exchange error: {str(e)}")
            st.stop()
    
    st.error("Failed to fetch data after multiple attempts")
    st.stop()

# Feature engineering with TA
def add_indicators(df):
    try:
        df = add_all_ta_features(
            df,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume"
        )
        return df.dropna()
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        st.stop()

# ... [rest of your original code remains the same] ...

# Modified train_and_predict with error handling
def train_and_predict():
    try:
        df = fetch_data()
        df = add_indicators(df)
        X, y = prepare_data(df)
        
        if len(X) == 0 or len(y) == 0:
            st.error("Not enough data to train model")
            return pd.DataFrame(), 'HOLD', 0.5
            
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X[:-10], y[:-10], epochs=5, batch_size=32, verbose=0)
        
        if len(X) < 1:
            st.error("No recent data for prediction")
            return df, 'HOLD', 0.5
            
        pred = model.predict(X[-1:])[0][0]
        signal = 'BUY' if pred > 0.6 else 'SELL' if pred < 0.4 else 'HOLD'
        confidence = pred
        timestamp = datetime.datetime.now()
        price = df['close'].iloc[-1]

        insert_signal(timestamp, signal, confidence, price)
        return df, signal, confidence
        
    except Exception as e:
        st.error(f"Critical error in prediction pipeline: {str(e)}")
        return pd.DataFrame(), 'HOLD', 0.5

# ... [rest of your Streamlit UI code] ...
