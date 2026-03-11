# Imports
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from scipy.optimize import minimize
import yfinance as yf
import webbrowser
import time
from textblob import TextBlob  # For sentiment analysis
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
API_KEY = "REDACTED_COINGECKO_KEY"  # Replace with your CoinGecko API key
RAPIDAPI_KEY = "REDACTED_RAPIDAPI_KEY"  # Replace with your RapidAPI key for Twitter
ALPACA_API_KEY = "REDACTED_ALPACA_KEY_2"  # Replace with your Alpaca API key
ALPACA_SECRET_KEY = "REDACTED_ALPACA_SECRET_2"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading endpoint
RISK_FREE_RATE = 0.07365  # Risk-free rate for Sharpe ratio
INITIAL_BALANCE = 1000  # Initial portfolio balance

# Initialize Alpaca API client
from alpaca_trade_api.rest import REST
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

class CoinGeckoAPI:
    """
    Wrapper for CoinGecko API.
    """

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
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            # Ensure date range is within 365 days
            if (end_dt - start_dt).days > 365:
                logging.warning("Date range exceeds 365 days. Truncating to the last 365 days.")
                start_dt = end_dt - timedelta(days=365)

            url = f"{COINGECKO_API_URL}/coins/{ticker}/market_chart/range"
            
            headers = {
            "accept": "application/json/",
            "?x-cg-pro-api-key": API_KEY  # Include the API key in headers
            }
            params = {
                "vs_currency": "usd",
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp()),
            }
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            if 'prices' not in data or not data['prices']:
                logging.warning(f"No price data found for {ticker}. Response: {data}")
                return pd.DataFrame()

            prices = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            return prices.set_index('timestamp')
        except Exception as e:
            logging.error(f"Error fetching crypto data for {ticker}: {e}")
            return pd.DataFrame()
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
                if not data.empty:
                    valid_data[ticker] = data
            except Exception as e:
                logging.error(f"Failed to fetch stock data for {ticker}: {e}")
        return valid_data

class TwitterSentimentAnalysis:
    """
    Perform sentiment analysis using Twitter API via RapidAPI.
    """

    @staticmethod
    def fetch_sentiment(ticker):
        """
        Fetch tweet sentiment for a given ticker using RapidAPI's Twitter API.

        Parameters:
            ticker (str): Stock or crypto ticker.

        Returns:
            float: Average sentiment score (-1 to 1).
        """
        url = "https://twitter-api45.p.rapidapi.com/search.php"
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,  # Replace with your RapidAPI key
            "x-rapidapi-host": "twitter-api45.p.rapidapi.com"
        }
        querystring = {"query": ticker}

        # Retry strategy for handling rate limits or temporary errors
        retries = 0
        max_retries = 5
        backoff = 2  # Initial backoff in seconds

        while retries < max_retries:
            try:
                # Make the API request
                response = requests.get(url, headers=headers, params=querystring)

                # Handle 403 Forbidden (likely an API key or subscription issue)
                if response.status_code == 403:
                    logging.error(f"403 Forbidden for {ticker}. Check your API key or subscription.")
                    return 0

                # Handle 429 Too Many Requests (rate limit exceeded)
                if response.status_code == 429:
                    retries += 1
                    delay = backoff ** retries  # Exponential backoff
                    logging.warning(f"429 Too Many Requests for {ticker}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue  # Retry the request

                # Raise an exception for other HTTP errors
                response.raise_for_status()

                # Parse the response JSON
                tweets = response.json().get('timeline')

                if not tweets:
                    logging.warning(f"No tweets found for {ticker}.")
                    return 0

                # Perform sentiment analysis using TextBlob
                sentiments = [TextBlob(tweets[i]['text']).sentiment.polarity for i in range(0,len(tweets)-1)]

                # Add delay between requests to avoid rate-limiting
                time.sleep(2)  # Add a delay of 2 seconds between requests
                return np.mean(sentiments)

            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching sentiment for {ticker}: {e}")
                retries += 1
                delay = backoff ** retries  # Exponential backoff
                logging.warning(f"Retrying in {delay} seconds...")
                time.sleep(delay)

        # If max retries exceeded
        logging.error(f"Failed to fetch sentiment for {ticker} after {max_retries} retries.")
        return 0
def calculate_technical_indicators(data):
    """
    Calculate technical indicators (SMA, EMA) for price data.

    Parameters:
        data (pd.DataFrame): DataFrame with price data.

    Returns:
        dict: Technical indicator scores.
    """
    try:
        if data.empty or 'price' not in data.columns:
            logging.warning("Data is empty or missing 'price' column.")
            return {"SMA": 0, "EMA": 0}

        data['SMA'] = data['price'].rolling(window=20).mean()
        data['EMA'] = data['price'].ewm(span=20).mean()

        return {
            "SMA": data['SMA'].iloc[-1] if not data['SMA'].isna().all() else 0,
            "EMA": data['EMA'].iloc[-1] if not data['EMA'].isna().all() else 0,
        }
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return {"SMA": 0, "EMA": 0}
def train_lstm_model(data):
    """
    Train an LSTM model on price data.

    Parameters:
        data (pd.DataFrame): DataFrame with price data.

    Returns:
        tuple: Trained model and scaler.
    """
    try:
        if data.empty or 'price' not in data.columns or len(data) < 60:
            logging.warning("Insufficient data for training.")
            return None, None

        data = data['price'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare training data
        X_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build the LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

        return model, scaler
    except Exception as e:
        logging.error(f"Error training LSTM model: {e}")
        return None, None
    
def create_dashboard():
    """
    Create a Dash GUI for AmbiVest with cryptocurrency and stock analysis.
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

    # Layout
    app.layout = dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.H1(
                        "AmbiVest: Intelligent Investment Analysis",
                        style={
                            "textAlign": "center",
                            "color": "#FFD700",
                            "padding": "20px",
                            "fontWeight": "bold",
                            "fontFamily": "Arial, sans-serif",
                        },
                    )
                ),
                style={"backgroundColor": "#1a1a1a", "borderBottom": "2px solid #FFD700"},
            ),

            # Input Section
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Inputs", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    [
                                        html.Label("Cryptocurrencies", style={"color": "white"}),
                                        dcc.Input(
                                            id="crypto-input",
                                            type="text",
                                            placeholder="Enter crypto tickers (e.g., bitcoin, ethereum)",
                                            style={"width": "100%", "marginBottom": "10px"},
                                        ),
                                        html.Label("Stocks", style={"color": "white"}),
                                        dcc.Input(
                                            id="stock-input",
                                            type="text",
                                            placeholder="Enter stock tickers (e.g., AAPL, MSFT)",
                                            style={"width": "100%", "marginBottom": "10px"},
                                        ),
                                        html.Label("Start Date", style={"color": "white"}),
                                        dcc.Input(
                                            id="start-date",
                                            type="text",
                                            placeholder="YYYY-MM-DD",
                                            style={"width": "100%", "marginBottom": "10px"},
                                        ),
                                        html.Label("End Date", style={"color": "white"}),
                                        dcc.Input(
                                            id="end-date",
                                            type="text",
                                            placeholder="YYYY-MM-DD",
                                            style={"width": "100%", "marginBottom": "10px"},
                                        ),
                                        dbc.Button(
                                            "Run Analysis",
                                            id="run-button",
                                            color="warning",
                                            style={"width": "100%"},
                                        ),
                                    ]
                                ),
                            ],
                            style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                        ),
                        width=4,
                    ),

                    # Portfolio Metrics Section
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Portfolio Metrics", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    [
                                        html.P(id="portfolio-equity", style={"color": "white"}),
                                        html.P(id="portfolio-buying-power", style={"color": "white"}),
                                        html.P(id="portfolio-pnl", style={"color": "white"}),
                                    ]
                                ),
                            ],
                            style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                        ),
                        width=8,
                    ),
                ],
                style={"marginTop": "20px"},
            ),

            # Graphs Section
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Cryptocurrency Performance", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    dcc.Graph(id="crypto-performance-trend"),
                                ),
                            ],
                            style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Stock Performance", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    dcc.Graph(id="stock-performance-trend"),
                                ),
                            ],
                            style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                        ),
                        width=6,
                    ),
                ],
                style={"marginTop": "20px"},
            ),

            # Strategy Comparison
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Strategy Comparison", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="strategy-comparison"),
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Sentiment Analysis and Trades
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Sentiment Analysis", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    html.Div(id="sentiment-analysis", style={"color": "white"}),
                                ),
                            ],
                            style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Trades Executed", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    html.Div(id="trades-executed", style={"color": "white"}),
                                ),
                            ],
                            style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                        ),
                        width=6,
                    ),
                ],
                style={"marginTop": "20px"},
            ),
        ],
        fluid=True,
        style={"backgroundColor": "#121212", "padding": "20px"},
    )



    @app.callback(
        [
            Output("crypto-performance-trend", "figure"),
            Output("stock-performance-trend", "figure"),
            Output("strategy-comparison", "figure"),
            Output("portfolio-equity", "children"),
            Output("portfolio-buying-power", "children"),
            Output("portfolio-pnl", "children"),
            Output("sentiment-analysis", "children"),
            Output("trades-executed", "children"),
        ],
        Input("run-button", "n_clicks"),
        [
            Input("crypto-input", "value"),
            Input("stock-input", "value"),
            Input("start-date", "value"),
            Input("end-date", "value"),
        ],
    )
    def update_dashboard(n_clicks, crypto_input, stock_input, start_date, end_date):
        if n_clicks is None:
            return {}, {}, {}, "No data", "No data", "No data", "No sentiment analysis", "No trades executed."

        # Validate inputs
        crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
        stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

        # Default date range
        start_date = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        # Fetch data
        crypto_data = {ticker: CoinGeckoAPI.fetch_crypto_data(ticker, start_date, end_date) for ticker in crypto_tickers}
        stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, start_date, end_date)

        # Sentiment analysis
        sentiments = {ticker: TwitterSentimentAnalysis.fetch_sentiment(ticker) for ticker in crypto_tickers + stock_tickers}

        # Placeholder graphs and data
        crypto_fig = {"data": [], "layout": {"title": "No Data"}}
        stock_fig = {"data": [], "layout": {"title": "No Data"}}
        strategy_fig = {"data": [], "layout": {"title": "No Data"}}

        # Portfolio metrics
        equity = "Portfolio Equity: $100,000"
        buying_power = "Buying Power: $200,000"
        pnl = "PnL: $0"

        sentiment_text = "\n".join([f"{ticker}: Sentiment Score = {score:.2f}" for ticker, score in sentiments.items()])
        trades_summary = "No trades executed."

        return crypto_fig, stock_fig, strategy_fig, equity, buying_power, pnl, sentiment_text, trades_summary
    return app

if __name__ == "__main__":
    port = 8050
    app = create_dashboard()
    webbrowser.open_new(f"http://127.0.0.1:{port}")
    app.run_server(debug=True, use_reloader=False, port=port)