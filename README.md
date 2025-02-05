**Trading and Market Analysis Suite**

📌 **Overview**

Welcome to the Trading and Market Analysis Suite, a collection of advanced tools designed for data-driven traders and investors. This suite integrates machine learning, technical analysis, and real-time financial data to empower decision-making in stock and options trading.

📁 **Projects**

🛠 **Project 1: Options Trading & Pattern Detection**

A Streamlit-powered interactive tool for identifying trade setups using image processing and quantitative analysis.

🔹 **Key Features**

✅ Chart Pattern Recognition: Uses OpenCV to detect key support/resistance levels.✅ ATR-Based Risk Analysis: Computes Average True Range (ATR) for trade validation.✅ Credit Spread Optimization: Recommends high probability trades using advanced probability metrics (POP, POT, POE).✅ User-Friendly UI: Built with Streamlit for seamless interaction.

🏗 **Tech Stack**

🔹 Python, OpenCV, NumPy, SciPy, Pandas, Streamlit

🚀 **Installation & Usage**

pip install streamlit opencv-python numpy pandas scipy Pillow
streamlit run app.py

Upload a chart, compute ATR, and optimize credit spreads based on real-time inputs.

📊 **Project 2: Enhanced Stock Screener with AI-driven Insights**

A powerful stock screener that combines quantitative analysis and AI-driven qualitative evaluation.

🔹 **Key Features**

✅ Fundamental Screening: P/E, P/B, dividend yield, ROE, market cap, debt-to-equity, and more.✅ Sentiment Analysis: NLP-based sentiment scoring from financial news.✅ Insider Trading & Analyst Ratings: Tracks buying/selling activities & professional recommendations.✅ ESG Analysis: Measures environmental, social, and governance impact.✅ Interactive Data Visualization: Dynamic stock comparisons using Plotly.

🏗 **Tech Stack**

🔹 Python, Yahoo Finance API, Plotly, NLTK, Pandas, AI-based caching

🚀 **Installation & Usage**

pip install streamlit pandas yfinance plotly nltk aiohttp aiocache requests
streamlit run stock_screener.py

Select filtering criteria, analyze stock performance, and visualize trends in real-time.

🤖 **Project 3: AI-Powered Options Trading Signal Tool (HMM-based)**

A cutting-edge Hidden Markov Model (HMM)-powered trading system for market state prediction and trade execution.

🔹 **Key Features**

✅ Market State Prediction: Gaussian HMM classifies market trends (Bullish, Bearish, Neutral).✅ Advanced Technical Indicators: RSI, MACD, Moving Averages, ATR.✅ Automated Options Selection: Filters options based on Greek parameters and risk constraints.✅ Backtesting Engine: Evaluates strategy performance across historical data.✅ Dynamic Portfolio Management: Adapts risk allocation based on market conditions.

🏗 **Tech Stack**

🔹 Python, Hidden Markov Models (HMM), Yahoo Finance API, Joblib, Matplotlib, TA-Lib

🚀 **Installation & Usage**

pip install streamlit yfinance requests hmmlearn ta matplotlib joblib
streamlit run trading_signal.py

Monitor real-time market signals, receive optimal trade suggestions, and backtest strategies with AI-driven insights.

🔒 **License**

This project is released under the MIT License. Feel free to modify and use it for personal or commercial purposes.

👨‍💻 **Author & Contributions**

Developed by Alazare Bati. Contributions, suggestions, and enhancements are always welcome! Feel free to open an issue or submit a pull request.

🔗 Contact: alazar.2800@gmail.com
