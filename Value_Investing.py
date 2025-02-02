import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import logging
import asyncio
import aiohttp
import requests
from aiocache import cached, SimpleMemoryCache
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import time

# Download NLTK data
nltk.download('vader_lexicon')

# Configure logging to output to a file
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Your API Keys (Replace 'YOUR_API_KEY' with your actual keys)
FMP_API_KEY = 'caooV649AAQctlWIoIsikKIpfqe3mWxc'  # Get from https://financialmodelingprep.com/
NEWS_API_KEY = 'd04ec4b73b7e4ba186764fb26965ab1c'  # Get from https://newsapi.org/
FINNHUB_API_KEY = 'crob0kpr01qtbpsrscpgcrob0kpr01qtbpsrscq0'  # Get from https://finnhub.io/

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


# Function to load tickers based on selected index
def load_index_tickers(index_name):
    if index_name == 'S&P 500':
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
            return df['Symbol'].tolist()
        except Exception as e:
            st.error("Failed to load S&P 500 tickers from Wikipedia.")
            logging.error(f"Error loading S&P 500 tickers: {e}")
            return []
    elif index_name == 'NASDAQ 100':
        try:
            url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
            tables = pd.read_html(url)
            df = tables[3]  # Adjust the index if necessary
            df['Ticker'] = df['Ticker'].str.replace('.', '-', regex=False)
            return df['Ticker'].tolist()
        except Exception as e:
            st.error("Failed to load NASDAQ 100 tickers from Wikipedia.")
            logging.error(f"Error loading NASDAQ 100 tickers: {e}")
            return []
    else:
        return []

# Function to calculate 5-year EPS growth
@cached(ttl=86400, cache=SimpleMemoryCache)
async def calculate_eps_growth(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings
        if earnings is None or earnings.empty or len(earnings) < 5:
            return None
        eps_5y_ago = earnings.iloc[-5]['Earnings']
        eps_current = earnings.iloc[-1]['Earnings']
        if eps_5y_ago == 0:
            return None
        growth = ((eps_current - eps_5y_ago) / abs(eps_5y_ago)) * 100
        return growth
    except Exception as e:
        logging.error(f"Error calculating EPS growth for {ticker}: {e}")
        return None


# Function to get news sentiment
@cached(ttl=3600, cache=SimpleMemoryCache)
async def get_news_sentiment(ticker):
    try:
        url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&language=en&pageSize=5'
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response_json = await response.json()
                articles = response_json.get('articles', [])
                sentiments = []
                for article in articles:
                    content = ' '.join(filter(None, [article.get('title'), article.get('description')]))
                    if content:
                        sentiment = sia.polarity_scores(content)
                        sentiments.append(sentiment['compound'])
                if sentiments:
                    average_sentiment = sum(sentiments) / len(sentiments)
                    return average_sentiment
                else:
                    return None
    except Exception as e:
        logging.error(f"Error fetching news sentiment for {ticker}: {e}")
        return None


# Function to get analyst rating
@cached(ttl=86400, cache=SimpleMemoryCache)
async def get_analyst_rating(ticker):
    try:
        url = f'https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{ticker}?limit=10&apikey={FMP_API_KEY}'
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                if data:
                    ratings = [item['rating'] for item in data]
                    rating_score = sum(
                        2 if r == 'Strong Buy' else
                        1 if r == 'Buy' else
                        0 if r == 'Hold' else
                        -1 if r == 'Underperform' else
                        -2 if r == 'Sell' else 0
                        for r in ratings
                    )
                    return rating_score / len(ratings) if ratings else None
                else:
                    return None
    except Exception as e:
        logging.error(f"Error fetching analyst rating for {ticker}: {e}")
        return None


# Function to get insider activity
@cached(ttl=86400, cache=SimpleMemoryCache)
async def get_insider_activity(ticker):
    try:
        url = f'https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={FINNHUB_API_KEY}'
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                transactions = data.get('data', [])
                if transactions:
                    buys = sum(t['change'] for t in transactions if t['change'] > 0)
                    sells = sum(abs(t['change']) for t in transactions if t['change'] < 0)
                    total = buys + sells
                    if total > 0:
                        insider_score = (buys - sells) / total
                        return insider_score
                    else:
                        return None
                else:
                    return None
    except Exception as e:
        logging.error(f"Error fetching insider activity for {ticker}: {e}")
        return None


# Function to get ESG score
@cached(ttl=86400, cache=SimpleMemoryCache)
async def get_esg_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        esg_score = stock.sustainability
        if esg_score is not None and not esg_score.empty:
            total_esg = esg_score.loc['totalEsg']['Value']
            return total_esg
        else:
            return None
    except Exception as e:
        logging.error(f"Error fetching ESG score for {ticker}: {e}")
        return None


# Function to fetch stock data asynchronously
@st.cache_data
def get_stock_data(tickers):
    data = []

    async def fetch_data(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            pe = info.get('trailingPE', float('nan'))
            pb = info.get('priceToBook', float('nan'))
            dividend = info.get('dividendYield', float('nan'))
            debt_to_equity = info.get('debtToEquity', float('nan'))
            roe = info.get('returnOnEquity', float('nan'))
            current_ratio = info.get('currentRatio', float('nan'))
            market_cap = info.get('marketCap', float('nan'))

            # Calculate 5-year EPS growth
            eps_growth = await calculate_eps_growth(ticker)

            # New metrics
            ps_ratio = info.get('priceToSalesTrailing12Months', float('nan'))
            gross_margin = info.get('grossMargins', float('nan'))
            free_cash_flow = info.get('freeCashflow', float('nan'))
            enterprise_value = info.get('enterpriseValue', float('nan'))

            fcf_yield = None
            if enterprise_value and free_cash_flow and enterprise_value != 0:
                fcf_yield = (free_cash_flow / enterprise_value) * 100

            # Qualitative metrics
            news_sentiment = await get_news_sentiment(ticker)
            analyst_rating = await get_analyst_rating(ticker)
            insider_activity = await get_insider_activity(ticker)
            esg_score = await get_esg_score(ticker)

            # Basic error handling for missing sector info
            sector = info.get('sector', 'N/A')

            return {
                'Ticker': ticker,
                'P/E Ratio': pe,
                'P/B Ratio': pb,
                'Dividend Yield (%)': None if pd.isna(dividend) else dividend * 100,
                'Debt-to-Equity Ratio': debt_to_equity,
                'EPS Growth (5Y %)': eps_growth,
                'ROE (%)': None if pd.isna(roe) else roe * 100,
                'Current Ratio': current_ratio,
                'Market Cap ($B)': None if pd.isna(market_cap) else market_cap / 1e9,  # Convert to billions
                'P/S Ratio': ps_ratio,
                'Gross Margin (%)': None if pd.isna(gross_margin) else gross_margin * 100,
                'FCF Yield (%)': fcf_yield,
                'News Sentiment': news_sentiment,
                'Analyst Rating': analyst_rating,
                'Insider Activity': insider_activity,
                'ESG Score': esg_score,
                'Sector': sector,
            }
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
            return None

    async def fetch_all_data(tickers):
        tasks = [fetch_data(ticker) for ticker in tickers]
        return await asyncio.gather(*tasks)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(fetch_all_data(tickers))

    # Filter out None results
    data = [result for result in results if result is not None]
    df = pd.DataFrame(data)
    return df


# Main app logic
def main():
    st.set_page_config(layout="wide")
    st.title("Enhanced Stock Screener with Qualitative Analysis")

    st.markdown("""
    This stock screener integrates both quantitative and qualitative analysis to help you identify high-quality investment opportunities.
    """)

    st.sidebar.header('Set Screening Criteria')

    # Index selection
    index_name = st.sidebar.selectbox('Select Stock Index', ['S&P 500', 'NASDAQ 100'])
    tickers = load_index_tickers(index_name)

    # Define screening criteria
    pe_ratio = st.sidebar.slider('Maximum P/E Ratio', 0.0, 100.0, 25.0,
                                 help="Price-to-Earnings ratio indicates how much investors are willing to pay per dollar of earnings.")
    pb_ratio = st.sidebar.slider('Maximum P/B Ratio', 0.0, 20.0, 3.0,
                                 help="Price-to-Book ratio compares a firm's market capitalization to its book value.")
    dividend_yield = st.sidebar.slider('Minimum Dividend Yield (%)', 0.0, 10.0, 2.0,
                                       help="Dividend Yield is the ratio of a company's annual dividend compared to its share price.")
    de_ratio = st.sidebar.slider('Maximum Debt-to-Equity Ratio', 0.0, 3.0, 1.0,
                                 help="Debt-to-Equity ratio measures a company's financial leverage.")
    eps_growth = st.sidebar.slider('Minimum EPS Growth (5Y, %)', -100.0, 100.0, 5.0,
                                   help="Earnings Per Share growth over the past 5 years.")
    roe = st.sidebar.slider('Minimum Return on Equity (%)', -100.0, 100.0, 15.0,
                            help="Return on Equity indicates how efficiently a company is using shareholders' equity.")
    current_ratio = st.sidebar.slider('Minimum Current Ratio', 0.0, 10.0, 1.5,
                                      help="Current Ratio measures a company's ability to pay short-term obligations.")
    market_cap = st.sidebar.slider('Minimum Market Cap ($B)', 0.0, 2000.0, 10.0,
                                   help="Market Capitalization is the total market value of a company's outstanding shares.")
    ps_ratio = st.sidebar.slider('Maximum P/S Ratio', 0.0, 50.0, 3.0,
                                 help="Price-to-Sales ratio compares a company's stock price to its revenues.")
    gross_margin = st.sidebar.slider('Minimum Gross Margin (%)', -100.0, 100.0, 30.0,
                                     help="Gross Margin indicates the percentage of revenue that exceeds the cost of goods sold.")
    fcf_yield = st.sidebar.slider('Minimum Free Cash Flow Yield (%)', -100.0, 100.0, 5.0,
                                  help="Free Cash Flow Yield measures a company's total free cash flow relative to its enterprise value.")
    news_sentiment_threshold = st.sidebar.slider('Minimum News Sentiment', -1.0, 1.0, 0.0,
                                                 help="Average sentiment of recent news articles about the company.")
    analyst_rating_threshold = st.sidebar.slider('Minimum Analyst Rating', -2.0, 2.0, 0.0,
                                                 help="Average analyst recommendation score.")
    insider_activity_threshold = st.sidebar.slider('Minimum Insider Activity', -1.0, 1.0, 0.0,
                                                   help="Net insider trading activity; positive values indicate buying.")
    esg_score_threshold = st.sidebar.slider('Maximum ESG Score', 0.0, 50.0, 50.0,
                                            help="Environmental, Social, and Governance score; lower is better.")

    # Sector selection
    all_sectors = ['All Sectors', 'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
                   'Communication Services', 'Industrials', 'Consumer Defensive', 'Energy', 'Utilities', 'Real Estate',
                   'Materials']
    selected_sectors = st.sidebar.multiselect('Select Sectors', all_sectors, default=['All Sectors'],
                                              help="Filter stocks by sector.")

    with st.spinner('Fetching stock data...'):
        start_time = time.time()
        stock_data = get_stock_data(tickers)
        end_time = time.time()
        st.write(f"Data fetched in {end_time - start_time:.2f} seconds.")

    if stock_data.empty:
        st.error("Failed to fetch data. Please check your API keys and internet connection.")
        logging.error("Stock data is empty after fetching.")
        return

    # Verify data fetching
    st.write(f"Total stocks fetched: {stock_data.shape[0]}")

    # Handle missing data and prevent KeyErrors
    filtered_data = stock_data.copy()

    # Apply filters with inclusion of NaN values
    if 'P/E Ratio' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['P/E Ratio'] <= pe_ratio) | (filtered_data['P/E Ratio'].isna())]

    if 'P/B Ratio' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['P/B Ratio'] <= pb_ratio) | (filtered_data['P/B Ratio'].isna())]

    if 'Dividend Yield (%)' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['Dividend Yield (%)'] >= dividend_yield) | (filtered_data['Dividend Yield (%)'].isna())]

    if 'Debt-to-Equity Ratio' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['Debt-to-Equity Ratio'] <= de_ratio) | (filtered_data['Debt-to-Equity Ratio'].isna())]

    if 'EPS Growth (5Y %)' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['EPS Growth (5Y %)'] >= eps_growth) | (filtered_data['EPS Growth (5Y %)'].isna())]

    if 'ROE (%)' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['ROE (%)'] >= roe) | (filtered_data['ROE (%)'].isna())]

    if 'Current Ratio' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['Current Ratio'] >= current_ratio) | (filtered_data['Current Ratio'].isna())]

    if 'Market Cap ($B)' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['Market Cap ($B)'] >= market_cap) | (filtered_data['Market Cap ($B)'].isna())]

    if 'P/S Ratio' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['P/S Ratio'] <= ps_ratio) | (filtered_data['P/S Ratio'].isna())]

    if 'Gross Margin (%)' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['Gross Margin (%)'] >= gross_margin) | (filtered_data['Gross Margin (%)'].isna())]

    if 'FCF Yield (%)' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['FCF Yield (%)'] >= fcf_yield) | (filtered_data['FCF Yield (%)'].isna())]

    # Qualitative filters
    if 'News Sentiment' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['News Sentiment'] >= news_sentiment_threshold) | (filtered_data['News Sentiment'].isna())]

    if 'Analyst Rating' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['Analyst Rating'] >= analyst_rating_threshold) | (filtered_data['Analyst Rating'].isna())]

    if 'Insider Activity' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['Insider Activity'] >= insider_activity_threshold) | (
            filtered_data['Insider Activity'].isna())]

    if 'ESG Score' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['ESG Score'] <= esg_score_threshold) | (filtered_data['ESG Score'].isna())]

    # Sector filter
    if 'Sector' in filtered_data.columns and selected_sectors != ['All Sectors']:
        filtered_data = filtered_data[filtered_data['Sector'].isin(selected_sectors)]

    st.header('Filtered Stocks')

    if filtered_data.empty:
        st.write("No stocks meet the screening criteria.")
        logging.info("No stocks meet the screening criteria.")
    else:
        st.write(f"Number of stocks that meet the criteria: {filtered_data.shape[0]}")

        # Custom Score
        def calculate_custom_score(row):
            score = (
                    (row['ROE (%)'] or 0) * 0.3 +
                    (row['EPS Growth (5Y %)'] or 0) * 0.2 +
                    (row['Analyst Rating'] or 0) * 0.2 +
                    (row['News Sentiment'] or 0) * 0.1 -
                    (row['Debt-to-Equity Ratio'] or 0) * 0.1 -
                    (row['P/E Ratio'] or 0) * 0.1
            )
            return score

        filtered_data['Custom Score'] = filtered_data.apply(calculate_custom_score, axis=1)
        filtered_data = filtered_data.sort_values(by='Custom Score', ascending=False)

        # Pagination for the dataframe
        rows_per_page = 10
        total_rows = filtered_data.shape[0]
        max_pages = (total_rows - 1) // rows_per_page + 1

        # Ensure max_pages is at least 1
        if max_pages < 1:
            max_pages = 1

        page_num = st.number_input('Page Number', min_value=1, max_value=max_pages, value=1, step=1)

        start_idx = (page_num - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        st.dataframe(filtered_data.reset_index(drop=True).iloc[start_idx:end_idx])

        # Export results as CSV
        csv = filtered_data.to_csv(index=False).encode()
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='filtered_stocks.csv',
            mime='text/csv',
        )

        st.header('Data Visualization')

        # Select metrics to plot
        metrics_to_plot = st.multiselect(
            'Select Metrics to Plot',
            ['P/E Ratio', 'P/B Ratio', 'Dividend Yield (%)', 'Debt-to-Equity Ratio', 'EPS Growth (5Y %)',
             'ROE (%)', 'Current Ratio', 'Market Cap ($B)', 'P/S Ratio', 'Gross Margin (%)', 'FCF Yield (%)',
             'News Sentiment', 'Analyst Rating', 'Insider Activity', 'ESG Score', 'Custom Score']
        )

        for metric in metrics_to_plot:
            if metric in filtered_data.columns:
                fig = px.histogram(filtered_data, x=metric, nbins=50, title=f'Distribution of {metric}')
                st.plotly_chart(fig)

        # Comparative Analysis
        st.header('Comparative Analysis')
        selected_stocks = st.multiselect('Select Stocks to Compare', filtered_data['Ticker'].tolist())
        if selected_stocks:
            comparison_data = filtered_data[filtered_data['Ticker'].isin(selected_stocks)]
            st.dataframe(comparison_data)

            # Historical Price Charts
            for ticker in selected_stocks:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                fig = px.line(hist, x=hist.index, y='Close', title=f'{ticker} Closing Prices')
                st.plotly_chart(fig)

    # Option to show error log
    if st.checkbox('Show Error Log'):
        try:
            with open('app.log', 'r') as log_file:
                st.text(log_file.read())
        except FileNotFoundError:
            st.write("No errors logged.")


if __name__ == '__main__':
    main()
