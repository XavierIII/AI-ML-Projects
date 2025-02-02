import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy.stats import norm
import pandas as pd


# Function to process uploaded chart images and detect patterns
def process_uploaded_image(image_path):
    """
    Process uploaded chart image for pattern detection.
    """
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Detect lines (e.g., support, resistance) using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw detected lines

    return image


# ATR Calculation Function
def calculate_atr(data, n=14):
    """
    Calculate ATR for a given DataFrame containing High, Low, and Close prices.
    """
    data['TR'] = data[['High', 'Low', 'Close']].apply(
        lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['Close']), abs(row['Low'] - row['Close'])),
        axis=1
    )
    data['ATR'] = data['TR'].rolling(window=n).mean()
    return data


# Credit Spread Metrics Calculation
def calculate_credit_spread_with_metrics(sell_strike, buy_strike, credit_received, delta_short, current_price, volatility, time_to_expiration, atr, direction):
    """
    Calculate max profit, max risk, reward/risk ratio, POP, POT, and POE for a credit spread.
    """
    max_profit = credit_received
    max_risk = (sell_strike - buy_strike) - credit_received
    reward_risk_ratio = max_profit / max_risk
    pop = 1 - delta_short  # Probability of Profit
    pot = 2 * delta_short  # Probability of Touch

    # POE Calculation using Black-Scholes approximation
    d1 = (np.log(current_price / sell_strike) + (0.5 * volatility ** 2) * time_to_expiration) / (volatility * np.sqrt(time_to_expiration))
    poe = 1 - norm.cdf(d1)  # CDF for the probability of expiring ITM

    # Ensure strikes align with ATR-based thresholds
    if direction == "bullish" and sell_strike >= current_price - atr:
        return None  # Invalid for bullish spread
    if direction == "bearish" and sell_strike <= current_price + atr:
        return None  # Invalid for bearish spread

    return max_profit, max_risk, reward_risk_ratio, pop, pot, poe


# Optimize Spreads Function
def optimize_spreads(current_price, atr, volatility, time_to_expiration, direction, delta_options, credit_received_options):
    suggestions = []
    for sell_strike, credit_received, delta_short in zip(delta_options.keys(), credit_received_options, delta_options.values()):
        buy_strike = sell_strike - 5 if direction == "bullish" else sell_strike + 5  # Fixed spread width
        metrics = calculate_credit_spread_with_metrics(
            sell_strike, buy_strike, credit_received, delta_short,
            current_price, volatility, time_to_expiration, atr, direction
        )
        if metrics:
            suggestions.append((sell_strike, buy_strike, *metrics))

    # Sort suggestions by POP, POE, and reward/risk ratio
    suggestions = sorted(suggestions, key=lambda x: (x[4], x[5], x[3]), reverse=True)  # Sort by POP, POE, reward/risk
    return suggestions[:3]  # Return top 3 spreads


# Streamlit App
st.title("Integrated Options Trading Tool with Image Upload")

# Section 1: Image Upload and Pattern Detection
st.header("1. Upload Premarket, Order Book, or Early Morning Market Chart")
uploaded_file = st.file_uploader("Upload a Chart Image (e.g., PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Save the uploaded file temporarily
    image_path = "uploaded_chart.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the image for pattern detection
    processed_image = process_uploaded_image(image_path)
    st.image(processed_image, caption="Pattern Detection Results", use_column_width=True)
else:
    st.write("Upload a chart to detect patterns.")

# Section 2: ATR Calculation
st.header("2. ATR Calculation")
if st.checkbox("Simulate ATR Calculation"):
    # Example price data
    data = pd.DataFrame({
        'High': [305, 310, 315],
        'Low': [295, 300, 305],
        'Close': [300, 307, 310]
    })
    atr_data = calculate_atr(data)
    st.write(atr_data[['High', 'Low', 'Close', 'ATR']])

# Section 3: Credit Spread Optimization
st.header("3. Credit Spread Optimization")

# Input Parameters
current_price = st.number_input("Current Price:", min_value=0.0, value=300.0, step=0.1)
atr = st.number_input("ATR (Average True Range):", min_value=0.0, value=5.0, step=0.1)
volatility = st.number_input("Implied Volatility (%):", min_value=0.0, value=20.0, step=0.1) / 100  # Convert to decimal
time_to_expiration = st.number_input("Time to Expiration (Days):", min_value=1, value=30, step=1) / 365  # Convert to years
direction = st.selectbox("Direction:", ["Bullish", "Bearish"])

# Simulate Delta and Credit Options
delta_options = {295: 0.30, 290: 0.25, 285: 0.20} if direction == "bullish" else {305: 0.30, 310: 0.25, 315: 0.20}
credit_received_options = [1.50, 1.25, 1.00]

# Generate and display recommendations
suggestions = optimize_spreads(current_price, atr, volatility, time_to_expiration, direction, delta_options, credit_received_options)

st.header("Top 3 Optimized Credit Spreads")
if suggestions:
    for i, suggestion in enumerate(suggestions, 1):
        sell_strike, buy_strike, max_profit, max_risk, reward_risk_ratio, pop, pot, poe = suggestion
        st.write(f"### Spread {i}")
        st.write(f"**Sell Strike:** {sell_strike}, **Buy Strike:** {buy_strike}")
        st.write(f"Max Profit: ${max_profit:.2f}, Max Risk: ${max_risk:.2f}, Reward/Risk: {reward_risk_ratio:.2f}")
        st.write(f"POP (Probability of Profit): {pop:.2%}")
        st.write(f"POT (Probability of Touch): {pot:.2%}")
        st.write(f"POE (Probability of Expiry): {poe:.2%}")
        st.write("---")
else:
    st.write("No spreads meet the criteria. Adjust parameters for better results.")
