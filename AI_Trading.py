# Streamlit for GUI
import streamlit as st

# Data manipulation and analysis
import pandas as pd
import numpy as np

# For handling dates and time
import datetime as dt

# For fetching data
import yfinance as yf
import requests

# Hidden Markov Model
from hmmlearn.hmm import GaussianHMM

# Technical indicators
import ta

# For option Greeks calculation
from scipy.stats import norm
import math

# For plotting
import matplotlib.pyplot as plt

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

# For saving and loading models
import joblib

# For logging
import logging
import os

# Ticker symbol
ticker = 'SPY'

# Time period for historical data
start_date = '2020-01-01'
end_date = dt.date.today().strftime('%Y-%m-%d')

# API Keys
ALPHA_VANTAGE_API_KEY = 'DH9M6JVZBDFG55ZP'  # Replace with your Alpha Vantage API key
TRADIER_API_KEY = '6YdbuXPeFWnfDM8mf5vmNmMGtObL'  # Replace with your Tradier API key

# Risk-free interest rate (e.g., 1% as 0.01)
risk_free_rate = 0.01

# Trading capital
total_capital = 500

# Risk management parameters
max_risk_per_trade = 0.02  # 2% of total capital
desired_delta = 0.4  # Target delta for option selection

# HMM parameters
n_components = 3  # Number of hidden states

# Set up Tradier API endpoint and headers
TRADIER_API_URL = 'https://sandbox.tradier.com/v1/markets/options/chains'
tradier_headers = {
    'Authorization': f'Bearer {TRADIER_API_KEY}',
    'Accept': 'application/json'
}

# Continuous learning configurations
model_save_path = 'hmm_model.pkl'  # Path to save the HMM model
retrain_interval = 5  # Retrain the model every 5 days
rolling_window_size = 252  # Use the last 252 days of data for retraining

# Backtesting configurations
test_size = 0.2  # 20% of the data for testing in backtesting

# Logging configuration
logging.basicConfig(filename='trading_tool.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# Fetch historical data using yfinance
def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.dropna()
    return data


# Prepare data for HMM
def prepare_hmm_data(data):
    X = data['Returns'].values.reshape(-1, 1)
    return X


# Train HMM
def train_hmm(X, n_components):
    model = GaussianHMM(n_components=n_components, covariance_type='full', n_iter=1000)
    model.fit(X)
    return model


# Get market state predictions
def get_market_states(model, X):
    hidden_states = model.predict(X)
    return hidden_states


# Map hidden states to market conditions
def map_market_states(data):
    state_means = data.groupby('State')['Returns'].mean()
    state_mapping = state_means.sort_values().index.tolist()
    num_states = len(state_mapping)

    # Generate condition labels dynamically
    conditions = []
    for i in range(num_states):
        # Assign labels based on position
        if i < num_states // 2:
            conditions.append('Bearish')
        elif i == num_states // 2 and num_states % 2 != 0:
            conditions.append('Neutral')
        else:
            conditions.append('Bullish')

    # Map the sorted states to conditions
    state_dict = {state_mapping[i]: conditions[i] for i in range(num_states)}
    data['Market_State'] = data['State'].map(state_dict)
    return data


# Save and load model functions
def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    if os.path.exists(path):
        model = joblib.load(path)
        return model
    else:
        return None


# Update model function with continuous learning
def update_hmm_model(data, model_path, n_components, retrain_interval, rolling_window_size):
    # Load existing model if available
    model = load_model(model_path)
    # Check if retraining is needed
    if model is None or (len(data) % retrain_interval == 0):
        # Use the most recent data for training
        train_data = data.tail(rolling_window_size)
        X = prepare_hmm_data(train_data)
        model = train_hmm(X, n_components)
        # Save the updated model
        save_model(model, model_path)
        logging.info(f"Model retrained with {n_components} hidden states.")
    else:
        logging.info("Using existing model without retraining.")
    return model


# Calculate technical indicators
def calculate_technical_indicators(data):
    # Moving Averages
    data['MA10'] = data['Adj Close'].rolling(window=10).mean()
    data['MA50'] = data['Adj Close'].rolling(window=50).mean()

    # Relative Strength Index (RSI)
    data['RSI'] = ta.momentum.RSIIndicator(data['Adj Close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(data['Adj Close'])
    data['MACD'] = macd.macd()
    data['Signal_Line'] = macd.macd_signal()

    data = data.dropna()
    return data


# Function to fetch option chain with Greeks from Tradier API
def fetch_option_chain(ticker, expiration_date, option_type):
    params = {
        'symbol': ticker,
        'expiration': expiration_date,
        'greeks': 'true'
    }
    response = requests.get(TRADIER_API_URL, params=params, headers=tradier_headers)
    if response.status_code == 200:
        options_data = response.json()
        if 'options' in options_data and options_data['options'] is not None:
            options_list = options_data['options']['option']
            df = pd.DataFrame(options_list)
            df = df[df['option_type'] == option_type]
            df['strike'] = df['strike'].astype(float)
            df['bid'] = df['bid'].astype(float)
            df['ask'] = df['ask'].astype(float)
            df['delta'] = df['greeks'].apply(lambda x: x['delta'] if x is not None else None)
            df['last'] = df['last'].astype(float)
            df = df.dropna(subset=['delta'])
            return df
        else:
            st.warning("No options data available.")
            return pd.DataFrame()
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return pd.DataFrame()


# Get the nearest expiration date
def get_nearest_expiration():
    ticker_info = yf.Ticker(ticker)
    expirations = ticker_info.options
    if expirations:
        return expirations[0]  # First expiration date
    else:
        st.warning("No expiration dates available.")
        return None


# Define signal generation functions
def check_ma_crossover(data):
    if data['MA10'].iloc[-2] < data['MA50'].iloc[-2] and data['MA10'].iloc[-1] > data['MA50'].iloc[-1]:
        return True
    else:
        return False


def check_rsi(data):
    if data['RSI'].iloc[-1] < 70:
        return True
    else:
        return False


def check_macd_crossover(data):
    if data['MACD'].iloc[-2] < data['Signal_Line'].iloc[-2] and data['MACD'].iloc[-1] > data['Signal_Line'].iloc[-1]:
        return True
    else:
        return False


# Define signal generation functions
def generate_signal(data, current_state):
    if (current_state == 'Bullish' and
            check_ma_crossover(data) and
            check_rsi(data) and
            check_macd_crossover(data)):
        return 'Buy Call Option'
    elif (current_state == 'Bearish' and
          not check_ma_crossover(data) and
          not check_rsi(data) and
          not check_macd_crossover(data)):
        return 'Buy Put Option'
    else:
        return 'No Signal'


# Select option based on signal
def select_option(data, signal):
    current_price = data['Adj Close'].iloc[-1]
    if signal == 'Buy Call Option' and not calls.empty:
        # Filter calls with delta close to desired_delta
        calls['delta_diff'] = abs(calls['delta'] - desired_delta)
        selected_option = calls.loc[calls['delta_diff'].idxmin()]
        option_type = 'Call'
    elif signal == 'Buy Put Option' and not puts.empty:
        # Filter puts with delta close to -desired_delta
        puts['delta_diff'] = abs(puts['delta'] + desired_delta)
        selected_option = puts.loc[puts['delta_diff'].idxmin()]
        option_type = 'Put'
    else:
        selected_option = None
        option_type = None
    return selected_option, option_type


# Backtesting function
def backtest_strategy(data, model, n_components):
    # Initialize variables
    signals = []
    positions = []
    portfolio_value = total_capital
    cash = total_capital
    holdings = 0
    entry_price = 0
    portfolio_df = pd.DataFrame(index=data.index)
    portfolio_df['Portfolio_Value'] = np.nan

    for i in range(n_components, len(data)):
        # Slice data up to current point
        train_data = data.iloc[:i]
        # Update model
        X = prepare_hmm_data(train_data)
        model = train_hmm(X, n_components)
        # Predict market state
        train_data['State'] = get_market_states(model, X)
        train_data = map_market_states(train_data)
        current_state = train_data['Market_State'].iloc[-1]
        # Generate signal
        signal = generate_signal(train_data)
        signals.append(signal)

        # Simulate trading logic
        current_price = train_data['Adj Close'].iloc[-1]
        if signal == 'Buy Call Option' and holdings == 0:
            # Buy
            holdings = cash / current_price
            cash = 0
            entry_price = current_price
        elif signal == 'Buy Put Option' and holdings > 0:
            # Sell
            cash = holdings * current_price
            holdings = 0
            portfolio_value = cash
        else:
            # Hold position
            if holdings > 0:
                portfolio_value = holdings * current_price
            else:
                portfolio_value = cash
        positions.append(portfolio_value)
        portfolio_df['Portfolio_Value'].iloc[i] = portfolio_value

    # Evaluate performance
    portfolio_df['Portfolio_Value'] = portfolio_df['Portfolio_Value'].fillna(method='ffill')
    portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change().fillna(0)
    total_return = (portfolio_df['Portfolio_Value'].iloc[-1] - total_capital) / total_capital
    st.write(f"Total Return from Backtesting: {total_return * 100:.2f}%")
    return portfolio_df, signals


def main():
    st.title("Options Trading Signal Tool")

    st.sidebar.header("Configuration")

    # Allow user to input or select configurations
    global total_capital
    total_capital = st.sidebar.number_input("Total Capital ($)", value=500)
    global max_risk_per_trade
    max_risk_per_trade = st.sidebar.slider("Max Risk per Trade (%)", min_value=1, max_value=100, value=2) / 100
    global desired_delta
    desired_delta = st.sidebar.slider("Desired Delta", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
    global n_components
    n_components = st.sidebar.slider("Number of Hidden States (HMM)", min_value=2, max_value=5, value=3)

    st.header("Market Data and Analysis")

    # Fetch historical data
    data = fetch_historical_data(ticker, start_date, end_date)
    data['Returns'] = data['Adj Close'].pct_change().fillna(0)
    data = calculate_technical_indicators(data)

    # Display latest data
    st.subheader("Latest Market Data")
    st.write(data.tail())

    # Prepare data for HMM
    X = prepare_hmm_data(data)

    # Update HMM model with continuous learning
    hmm_model = update_hmm_model(data, model_save_path, n_components, retrain_interval, rolling_window_size)

    # Predict market states
    data['State'] = get_market_states(hmm_model, X)
    data = map_market_states(data)
    current_state = data['Market_State'].iloc[-1]
    st.write(f"**Current Market State**: {current_state}")

    # Generate trading signal
    signal = generate_signal(data, current_state)
    st.write(f"**Trading Signal**: {signal}")

    # Fetch option chain data
    expiration_date = get_nearest_expiration()
    if expiration_date:
        calls = fetch_option_chain(ticker, expiration_date, 'call')
        puts = fetch_option_chain(ticker, expiration_date, 'put')
    else:
        calls = pd.DataFrame()
        puts = pd.DataFrame()

    # Select option based on signal
    selected_option, option_type = select_option(data, signal)

    # Compile recommendation
    if selected_option is not None:
        option_price = (selected_option['bid'] + selected_option['ask']) / 2
        contract_name = selected_option['symbol']
        strike_price = selected_option['strike']
        expiration_date = selected_option['expiration_date']

        # Risk management
        risk_amount = total_capital * max_risk_per_trade
        num_contracts = int(risk_amount / (option_price * 100))
        if num_contracts < 1:
            num_contracts = 1  # Minimum of one contract

        # Stop-loss and take-profit levels
        entry_price = option_price
        stop_loss_price = entry_price * 0.5  # 50% loss
        take_profit_price = entry_price * 2  # 100% gain

        recommendation = {
            'Action': f'Buy {option_type} Option',
            'Ticker': ticker,
            'Current_Price': data['Adj Close'].iloc[-1],
            'Strike_Price': strike_price,
            'Expiration_Date': expiration_date,
            'Option_Price': option_price,
            'Contract_Name': contract_name,
            'Number_of_Contracts': num_contracts,
            'Stop_Loss_Price': stop_loss_price,
            'Take_Profit_Price': take_profit_price
        }
        st.subheader("Trading Recommendation")
        st.write(recommendation)
    else:
        st.write("No suitable option found based on the current analysis.")

    # Backtesting
    st.header("Backtesting Results")
    if st.button("Run Backtest"):
        portfolio_df, signals = backtest_strategy(data, hmm_model, n_components)
        st.line_chart(portfolio_df['Portfolio_Value'])
        st.write(portfolio_df)

    # Additional analysis can be added here


if __name__ == '__main__':
    main()
