import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from textblob import TextBlob
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
from scipy.optimize import minimize
import yfinance as yf

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
API_KEY = "REDACTED_COINGECKO_KEY"  # Replace with your CoinGecko API key
RISK_FREE_RATE = 0.02  # Risk-free rate for Sharpe ratio
INITIAL_BALANCE = 1000  # Initial portfolio balance


# BLOCK 1: COINGECKO API WRAPPER
class CoinGeckoAPI:
    """
    Wrapper for CoinGecko API.
    """

    @staticmethod
    def get_valid_crypto_ids():
        """
        Fetch the list of valid cryptocurrency IDs from CoinGecko.
        Returns:
            list: List of valid cryptocurrency IDs.
        """
        url = f"{COINGECKO_API_URL}/coins/list"
        headers = {"accept": "application/json", "x-cg-demo-api-key": API_KEY}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            coins = response.json()
            return [coin['id'] for coin in coins]
        except Exception as e:
            logging.error(f"Error fetching valid crypto IDs: {e}")
            return []

    @staticmethod
    def fetch_crypto_data(ticker, start_date, end_date):
        """
        Fetch historical cryptocurrency data from CoinGecko.

        Parameters:
            ticker (str): Cryptocurrency ticker (CoinGecko ID).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame with historical price data.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Ensure date range is within 365 days
        if (end_dt - start_dt).days > 365:
            logging.warning("Date range exceeds 365 days. Truncating to the last 365 days.")
            start_dt = end_dt - timedelta(days=365)

        url = f"{COINGECKO_API_URL}/coins/{ticker}/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": int(start_dt.timestamp()),
            "to": int(end_dt.timestamp()),
        }
        headers = {"accept": "application/json", "x-cg-demo-api-key": API_KEY}

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if 'prices' not in data or not data['prices']:
                logging.warning(f"No price data found for {ticker}. Response: {data}")
                return pd.DataFrame()

            prices = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)
            return prices
        except Exception as e:
            logging.error(f"Error fetching crypto data for {ticker}: {e}")
            return pd.DataFrame()


# BLOCK 2: YAHOO FINANCE WRAPPER
class YahooFinanceAPI:
    """
    Wrapper for Yahoo Finance API for stock data.
    """

    @staticmethod
    def fetch_stock_data(tickers, start_date, end_date):
        """
        Fetch stock data using yfinance.

        Parameters:
            tickers (list): List of stock tickers.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            dict: Dictionary of DataFrames with stock data.
        """
        valid_data = {}
        for ticker in tickers:
            try:
                logging.info(f"Fetching stock data for {ticker}...")
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    logging.warning(f"No data returned for {ticker}. Skipping...")
                    continue
                valid_data[ticker] = data
            except Exception as e:
                logging.error(f"Failed to fetch stock data for {ticker}: {e}")
        return valid_data


# BLOCK 3: SENTIMENT ANALYSIS
def fetch_tweets_sentiment(tickers):
    """
    Simulate fetching tweets and calculating sentiment using TextBlob.

    Parameters:
        tickers (list): List of tickers (crypto and stocks).

    Returns:
        dict: Sentiment scores for each ticker.
    """
    sentiments = {}
    for ticker in tickers:
        try:
            tweets = [f"Example tweet about {ticker}"]  # Simulate fetching tweets
            sentiment_score = np.mean([TextBlob(tweet).sentiment.polarity for tweet in tweets])
            sentiments[ticker] = sentiment_score
        except Exception as e:
            logging.error(f"Error fetching sentiment for {ticker}: {e}")
            sentiments[ticker] = 0  # Default neutral sentiment
    return sentiments


# BLOCK 4: MACHINE LEARNING MODULE
def train_deep_learning_model(data):
    """
    Train a deep learning model on stock/crypto price data.

    Parameters:
        data (pd.DataFrame): DataFrame with historical price data.

    Returns:
        tuple: Trained model and scaler.
    """
    try:
        data = data['price'].values.reshape(-1, 1)  # Use 'price' column
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare training data
        X_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build the model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        return model, scaler
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None, None


# BLOCK 5: DASH GUI
def create_dashboard():
    """
    Create a Dash GUI for the trading bot.
    """
    app = Dash(__name__)

    # Layout
    app.layout = html.Div([
        html.H1("Trading Bot Dashboard", style={"textAlign": "center"}),

        # Input Section
        html.Div([
            html.Label("Cryptocurrencies"),
            dcc.Input(id="crypto-input", type="text", placeholder="Enter crypto tickers (comma-separated)", style={"width": "100%"}),
            html.Label("Stocks"),
            dcc.Input(id="stock-input", type="text", placeholder="Enter stock tickers (comma-separated)", style={"width": "100%"}),
            html.Button("Run Bot", id="run-button", n_clicks=0),
        ], style={"padding": "20px"}),

        # Output Section
        html.Div(id="output-section"),
    ])

    # Callbacks
    @app.callback(
        Output("output-section", "children"),
        Input("run-button", "n_clicks"),
        [Input("crypto-input", "value"), Input("stock-input", "value")]
    )
    def update_output(n_clicks, crypto_input, stock_input):
        if n_clicks == 0:
            return "Enter tickers and click 'Run Bot' to start."
        
        crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
        stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

        # Simulate fetching data and running the bot
        crypto_data = {ticker: CoinGeckoAPI.fetch_crypto_data(ticker, "2023-01-01", "2023-12-31") for ticker in crypto_tickers}
        stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, "2023-01-01", "2023-12-31")
        sentiment_scores = fetch_tweets_sentiment(crypto_tickers + stock_tickers)

        return f"Crypto Data: {crypto_data.keys()}, Stock Data: {stock_data.keys()}, Sentiments: {sentiment_scores}"

    return app


# Main Execution
if __name__ == "__main__":
    app = create_dashboard()
    app.run_server(debug=True, use_reloader=False)