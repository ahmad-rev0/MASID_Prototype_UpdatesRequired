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
ALPACA_API_KEY = "REDACTED_ALPACA_KEY_5"  # Replace with your Alpaca API key
ALPACA_SECRET_KEY = "REDACTED_ALPACA_SECRET_5"
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

            # Initialize an empty DataFrame to store results
            all_prices = pd.DataFrame()

            # Split the date range into chunks of 90 days (API granularity limit for hourly data)
            while start_dt < end_dt:
                chunk_end_dt = min(start_dt + timedelta(days=90), end_dt)  # 90-day chunks
                url = f"{COINGECKO_API_URL}/coins/{ticker}/market_chart/range"
                headers = {
                    "accept": "application/json",
                    "x-cg-pro-api-key": API_KEY  # Include the API key in headers
                }
                params = {
                    "vs_currency": "usd",
                    "from": int(start_dt.timestamp()),
                    "to": int(chunk_end_dt.timestamp()),
                }

                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                if 'prices' not in data or not data['prices']:
                    logging.warning(f"No price data found for {ticker} in range {start_dt} to {chunk_end_dt}.")
                else:
                    # Convert the prices into a DataFrame
                    chunk_prices = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
                    chunk_prices['timestamp'] = pd.to_datetime(chunk_prices['timestamp'], unit='ms')
                    chunk_prices.set_index('timestamp', inplace=True)

                    # Append the chunk to the main DataFrame
                    all_prices = pd.concat([all_prices, chunk_prices])

                # Move to the next date range chunk
                start_dt = chunk_end_dt

            # Return the complete DataFrame
            if all_prices.empty:
                logging.warning(f"No data retrieved for {ticker} in the given date range.")
            return all_prices

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
        Fetch historical stock data for the specified tickers.

        Parameters:
            tickers (list): List of stock tickers.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.

        Returns:
            dict: Dictionary of DataFrames keyed by ticker symbol.
        """
        stock_data = {}

        for ticker in tickers:
            try:
                # Fetch historical data using Yahoo Finance
                df = yf.download(ticker, start=start_date, end=end_date)

                if df.empty:
                    logging.warning(f"No data found for {ticker} in the specified date range.")
                    continue

                # Ensure DateTime index and keep only necessary columns
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                stock_data[ticker] = df[['Close']].sort_index()

            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {e}")

        return stock_data

def fetch_latest_price(ticker):
    """
    Fetch the latest price of a stock or crypto asset from Alpaca.

    Parameters:
        ticker (str): The ticker symbol for the asset.

    Returns:
        float: The latest price of the asset.
    """
    try:
        url = f"{ALPACA_BASE_URL}/v2/stocks/{ticker}/quotes/latest"
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("quote", {}).get("ap", 0)  # Ask price as the latest price
    except Exception as e:
        logging.error(f"Error fetching latest price for {ticker}: {e}")
        return 0

def place_order(ticker, quantity, side="buy", order_type="market", time_in_force="gtc"):
    """
    Place an order via Alpaca.

    Parameters:
        ticker (str): Asset ticker symbol.
        quantity (int): Number of shares/units to buy or sell.
        side (str): "buy" or "sell".
        order_type (str): Order type (default is "market").
        time_in_force (str): Time in force (default is "gtc").

    Returns:
        dict: Response from Alpaca API.
    """
    try:
        url = f"{ALPACA_BASE_URL}/v2/orders"
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        order_data = {
            "symbol": ticker,
            "qty": quantity,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        response = requests.post(url, json=order_data, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error placing order for {ticker}: {e}")
        return {}
    
def fetch_account_balance():
    """
    Fetch the Alpaca account balance.

    Returns:
        float: Account buying power.
    """
    try:
        url = f"{ALPACA_BASE_URL}/v2/account"
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return float(data.get("buying_power", 0))
    except Exception as e:
        logging.error(f"Error fetching account balance: {e}")
        return 0

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
            float: Average sentiment score (-1 to 1). Returns 0 if no valid sentiment is found.
        """
        url = "https://twitter-api45.p.rapidapi.com/search.php"
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,  # Replace with your RapidAPI key
            "x-rapidapi-host": "twitter-api45.p.rapidapi.com"
        }
        querystring = {"query": ticker}

        retries = 0
        max_retries = 5
        backoff = 2  # Initial backoff in seconds

        while retries < max_retries:
            try:
                # Make the API request
                response = requests.get(url, headers=headers, params=querystring)

                # Handle 403 Forbidden
                if response.status_code == 403:
                    logging.error(f"403 Forbidden for {ticker}. Check your API key or subscription.")
                    return 0

                # Handle 429 Too Many Requests
                if response.status_code == 429:
                    retries += 1
                    delay = backoff ** retries  # Exponential backoff
                    logging.warning(f"429 Too Many Requests for {ticker}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue  # Retry the request

                # Raise an exception for other HTTP errors
                response.raise_for_status()

                # Parse the response JSON
                data = response.json()
                tweets = data.get('timeline')

                # Debugging: Log the raw API response
                logging.debug(f"Raw API response for {ticker}: {data}")

                if not tweets:
                    logging.warning(f"No tweets found for {ticker}.")
                    return 0

                # Perform sentiment analysis
                sentiments = []
                for tweet in tweets:
                    text = tweet.get('text')
                    if text and isinstance(text, str):  # Ensure text is a valid string
                        try:
                            polarity = TextBlob(text).sentiment.polarity
                            sentiments.append(polarity)
                        except Exception as e:
                            logging.warning(f"Failed to process tweet text for {ticker}: {e}")

                if not sentiments:
                    logging.warning(f"No valid tweets found for {ticker}.")
                    return 0

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
    Calculate technical indicators (SMA, EMA, RSI) for price data.

    Parameters:
        data (pd.DataFrame): DataFrame with price data.

    Returns:
        dict: Normalized technical indicator scores.
    """
    try:
        if data.empty or "price" not in data.columns:
            logging.warning("Data is empty or missing 'price' column.")
            return {"SMA": 0, "EMA": 0, "RSI": 0}

        # Calculate SMA and EMA
        data["SMA"] = data["price"].rolling(window=20).mean()
        data["EMA"] = data["price"].ewm(span=20).mean()

        # Calculate RSI
        delta = data["price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data["RSI"] = 100 - (100 / (1 + rs))

        # Normalize indicators to [0, 1]
        indicators = {
            "SMA": (data["SMA"].iloc[-1] - data["price"].min()) / (data["price"].max() - data["price"].min()),
            "EMA": (data["EMA"].iloc[-1] - data["price"].min()) / (data["price"].max() - data["price"].min()),
            "RSI": data["RSI"].iloc[-1] / 100,
        }

        return indicators
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return {"SMA": 0, "EMA": 0, "RSI": 0}

def train_lstm_model(data):
    """
    Train an LSTM model on price data and return a prediction score.

    Parameters:
        data (pd.DataFrame): DataFrame with price data.

    Returns:
        float: Prediction score (0 to 1).
    """
    try:
        if data.empty or "price" not in data.columns or len(data) < 60:
            logging.warning("Insufficient data for training.")
            return 0

        # Prepare data
        data = data["price"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i - 60:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Train LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        # Predict the next price
        last_60_days = scaled_data[-60:]
        X_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
        predicted_price = model.predict(X_test)[0, 0]

        # Convert prediction to probability (0-1)
        return max(0, min(1, predicted_price))
    except Exception as e:
        logging.error(f"Error training LSTM model: {e}")
        return 0
    
def calculate_investment_surety(sentiment, technical_scores, ml_score):
    """
    Calculate the investment surety metric using optimized weights.

    Parameters:
        sentiment (float): Sentiment score (0 to 1).
        technical_scores (dict): Technical indicator scores (SMA, EMA, RSI).
        ml_score (float): ML prediction score (0 to 1).

    Returns:
        float: Investment surety metric (0 to 1).
    """
    def objective(weights):
        # Weights: [w1, w2, w3] for sentiment, technicals, ML
        w1, w2, w3 = weights
        return -(
            w1 * sentiment +
            w2 * (technical_scores["SMA"] + technical_scores["EMA"] + technical_scores["RSI"]) / 3 +
            w3 * ml_score
        )

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1}
    bounds = [(0, 1), (0, 1), (0, 1)]  # Weights between 0 and 1

    # Optimize weights
    result = minimize(objective, x0=[1/3, 1/3, 1/3], bounds=bounds, constraints=constraints)
    optimal_weights = result.x

    # Calculate metric with optimal weights
    return -objective(optimal_weights)
    
def generate_crypto_graph(crypto_data):
    """
    Generate cryptocurrency graph using Plotly.

    Parameters:
        crypto_data (dict): Dictionary of DataFrames with crypto data.

    Returns:
        dict: Plotly figure for cryptocurrency performance.
    """
    try:
        data = []
        for ticker, df in crypto_data.items():
            if not df.empty:
                data.append(go.Scatter(
                    x=df.index,
                    y=df['price'],
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
        return {
            "data": data,
            "layout": go.Layout(
                title="Cryptocurrency Performance",
                xaxis={"title": "Date"},
                yaxis={"title": "Price (USD)"},
                template="plotly_dark",
                plot_bgcolor="#1a1a1a",
                paper_bgcolor="#1a1a1a",
            ),
        }
    except Exception as e:
        logging.error(f"Error generating crypto graph: {e}")
        return {"data": [], "layout": {"title": "No Data"}}


def generate_stock_graph(stock_data):
    """
    Generate stock graph using Plotly.

    Parameters:
        stock_data (dict): Dictionary of DataFrames with stock data.

    Returns:
        dict: Plotly figure for stock performance.
    """
    try:
        # Combine all stock data into a single DataFrame
        combined_data = pd.DataFrame()

        for ticker, df in stock_data.items():
            if not df.empty:
                combined_data[ticker] = df['Close']

        # Align all data to a common index (dates)
        combined_data = combined_data.sort_index().fillna(method='ffill').dropna()

        # Create Plotly traces
        data = [
            go.Scatter(
                x=combined_data.index,
                y=combined_data[ticker],
                mode='lines',
                name=ticker,
                line=dict(width=2)
            )
            for ticker in combined_data.columns
        ]

        return {
            "data": data,
            "layout": go.Layout(
                title="Stock Performance",
                xaxis={"title": "Date", "type": "date", "tickformat": "%b %d, %Y"},
                yaxis={"title": "Price (USD)"},
                template="plotly_dark",
                plot_bgcolor="#1a1a1a",
                paper_bgcolor="#1a1a1a",
            ),
        }
    except Exception as e:
        logging.error(f"Error generating stock graph: {e}")
        return {"data": [], "layout": {"title": "No Data"}}

def generate_strategy_comparison_graph(crypto_data, stock_data, trades, investment_surety_scores):
    """
    Generate strategy comparison graph.

    Parameters:
        crypto_data (dict): Cryptocurrency historical data.
        stock_data (dict): Stock historical data.
        trades (dict): Dictionary of executed trades.
        investment_surety_scores (dict): Investment surety scores for each asset.

    Returns:
        dict: Plotly figure comparing strategies.
    """
    try:
        # Combine all prices into a single DataFrame
        all_prices = pd.DataFrame()
        dates = None

        for ticker, df in {**crypto_data, **stock_data}.items():
            if not df.empty:
                # Use 'price' for crypto and 'Close' for stocks
                all_prices[ticker] = df['price'] if 'price' in df.columns else df['Close']
                dates = df.index if dates is None else dates.union(df.index)

        # Align all data to a common date range
        all_prices = all_prices.reindex(dates).sort_index().fillna(method='ffill')
        all_prices = all_prices.dropna(axis=1)

        if all_prices.empty:
            logging.error("No data available for strategy comparison.")
            return {"data": [], "layout": {"title": "No Data"}}

        # Initialize strategy values
        app_strategy_values = []
        buy_and_hold_value = [INITIAL_BALANCE]
        risk_parity_value = [INITIAL_BALANCE]

        # Filter invalid investment surety scores
        valid_surety_scores = {
            ticker: score for ticker, score in investment_surety_scores.items() if score is not None
        }

        # App's Strategy (Investment Surety Metric)
        for i in range(len(all_prices)):
            metric = sum(
                valid_surety_scores[ticker] * all_prices.iloc[i][ticker]
                for ticker in valid_surety_scores.keys()
                if ticker in all_prices.columns
            )
            app_strategy_values.append(metric)

        # Buy and Hold Strategy
        buy_and_hold_weights = {ticker: 1 / len(all_prices.columns) for ticker in all_prices.columns}
        for i in range(1, len(all_prices)):
            buy_and_hold_value.append(
                sum(buy_and_hold_weights[ticker] * all_prices.iloc[i][ticker] * INITIAL_BALANCE for ticker in all_prices.columns)
            )

        # Risk Parity Strategy
        volatility = all_prices.pct_change().std()
        risk_parity_weights = (1 / (volatility + 1e-8)).fillna(0)
        risk_parity_weights /= risk_parity_weights.sum()
        for i in range(1, len(all_prices)):
            risk_parity_value.append(
                sum(risk_parity_weights[ticker] * all_prices.iloc[i][ticker] * INITIAL_BALANCE for ticker in all_prices.columns)
            )

        # Generate the graph
        data = [
            go.Scatter(x=all_prices.index, y=app_strategy_values, mode='lines', name="App's Strategy", line=dict(width=2)),
            go.Scatter(x=all_prices.index, y=buy_and_hold_value, mode='lines', name="Buy and Hold", line=dict(width=2)),
            go.Scatter(x=all_prices.index, y=risk_parity_value, mode='lines', name="Risk Parity", line=dict(width=2)),
        ]

        return {
            "data": data,
            "layout": go.Layout(
                title="Strategy Comparison",
                xaxis={"title": "Date", "type": "date", "tickformat": "%b %d, %Y"},
                yaxis={"title": "Portfolio Value (USD)"},
                template="plotly_dark",
                plot_bgcolor="#1a1a1a",
                paper_bgcolor="#1a1a1a",
            ),
        }

    except Exception as e:
        logging.error(f"Error generating strategy comparison graph: {e}")
        return {"data": [], "layout": {"title": "No Data"}}

def execute_trades(optimized_weights, balance=INITIAL_BALANCE):
    """
    Execute trades based on optimized portfolio weights using Alpaca.

    Parameters:
        optimized_weights (dict): Optimized weights for each asset.
        balance (float): Initial balance for trading.

    Returns:
        dict: Summary of executed trades.
    """
    try:
        trades_summary = {}
        for ticker, weight in optimized_weights.items():
            allocation = balance * weight
            latest_price = alpaca.get_last_trade(ticker).price  # Fetch the most recent price
            quantity = allocation // latest_price  # Calculate quantity to buy
            if quantity > 0:
                # Simulate trade execution (replace with Alpaca's actual trade API in production)
                trades_summary[ticker] = {
                    "quantity": quantity,
                    "price": latest_price,
                    "total": quantity * latest_price,
                }
        return trades_summary
    except Exception as e:
        logging.error(f"Error executing trades: {e}")
        return {}

def fetch_closing_prices(data):
    """
    Get the most recent closing prices from historical data.

    Parameters:
        data (dict): Dictionary of DataFrames with historical data.

    Returns:
        dict: Closing prices for each ticker.
    """
    try:
        closing_prices = {}
        for ticker, df in data.items():
            if not df.empty:
                closing_prices[ticker] = df['Close'].iloc[-1] if 'Close' in df.columns else df['price'].iloc[-1]
        return closing_prices
    except Exception as e:
        logging.error(f"Error fetching closing prices: {e}")
        return {}
    
def get_last_trade_result():
    """
    Fetch the last trade result from Alpaca using the Trading API.
    """
    try:
        # API endpoint and headers for Alpaca
        orders_url = f"{ALPACA_BASE_URL}/v2/orders"
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }

        # Make the API request to fetch all orders
        response = requests.get(orders_url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses

        # Debug the response
        print(f"Raw response: {response.json()}")  # Add this line for debugging

        # Parse the response
        orders = response.json()
        if not orders:
            logging.info("No trading activity found.")
            return "No trades executed."

        # Get the most recent order
        last_order = orders[0]

        # Extract relevant details
        trade_result = f"""
        Symbol: {last_order['symbol']}
        Quantity: {last_order['qty']}
        Filled Quantity: {last_order['filled_qty']}
        Price: {last_order.get('filled_avg_price', 'N/A')}
        Status: {last_order['status']}
        Side: {last_order['side']}
        Submitted At: {last_order['submitted_at']}
        Filled At: {last_order.get('filled_at', 'Not filled yet')}
        """

        return trade_result.strip()

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching last trade result: {e}")
        return "Error fetching last trade result."
    
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
                    
                    # Add Investment Surety Metric Section
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Investment Surety Metric", style={"color": "#FFD700", "fontWeight": "bold"}),
                                        dbc.CardBody(
                                            [
                                                html.Div(id="investment-surety", style={"color": "white", "fontSize": "18px"}),
                                            ],
                                        ),
                                    ],
                                    style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                                ),
                                width=12,
                            )
                        ],
                        style={"marginTop": "20px"},
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

            # Strategy Comparison Graph
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Strategy Comparison", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="strategy-comparison"),  # Add the missing graph here
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
            Output("investment-surety", "children"),
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

        try:
            # **1. Validate Inputs**
            crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
            stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

            # Default date range
            start_date = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")

            # **2. Fetch Data**
            crypto_data = {ticker: CoinGeckoAPI.fetch_crypto_data(ticker, start_date, end_date) for ticker in crypto_tickers}
            stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, start_date, end_date)

            # **3. Sentiment Analysis**
            sentiments = {ticker: TwitterSentimentAnalysis.fetch_sentiment(ticker) for ticker in crypto_tickers + stock_tickers}

            # **4. Calculate Technical Indicators**
            technical_scores = {}
            for ticker, df in {**crypto_data, **stock_data}.items():
                if not df.empty:
                    technical_scores[ticker] = calculate_technical_indicators(df)

            # **5. Train ML Model and Compute ML Scores**
            ml_scores = {}
            for ticker, df in {**crypto_data, **stock_data}.items():
                if not df.empty:
                    ml_scores[ticker] = train_lstm_model(df)

            # **6. Calculate Investment Surety Metric**
            investment_surety_scores = {}
            for ticker in {**crypto_data, **stock_data}.keys():
                sentiment_score = sentiments.get(ticker, 0)
                technical_score = technical_scores.get(ticker, {"SMA": 0, "EMA": 0, "RSI": 0})
                ml_score = ml_scores.get(ticker, 0)

                investment_surety_scores[ticker] = calculate_investment_surety(
                    sentiment_score,
                    technical_score,
                    ml_score
                )

            # **7. Simulate Trades**
            optimized_weights = {**{ticker: 0.5 / len(crypto_tickers) for ticker in crypto_tickers},
                                **{ticker: 0.5 / len(stock_tickers) for ticker in stock_tickers}}
            trades = execute_trades(optimized_weights)

            # **8. Generate Graphs**
            crypto_fig = generate_crypto_graph(crypto_data)
            stock_fig = generate_stock_graph(stock_data)
            strategy_fig = generate_strategy_comparison_graph(
                crypto_data, stock_data, trades, investment_surety_scores
            )

            # **9. Portfolio Metrics**
            total_trades_value = sum([trade["total"] for trade in trades.values()])
            equity = f"Portfolio Equity: ${INITIAL_BALANCE:.2f}"
            pnl = f"PnL: ${total_trades_value - INITIAL_BALANCE:.2f}"
            buying_power = f"Buying Power: ${INITIAL_BALANCE - total_trades_value:.2f}"

            # **10. Investment Surety Metric Display**
            investment_surety_text = "\n".join(
                [f"{ticker}: {score:.2f}" for ticker, score in investment_surety_scores.items()]
            )

            # **11. Fetch Last Trade Result**
            last_trade_result = get_last_trade_result()

            # **12. Return Outputs**
            return (
                crypto_fig,
                stock_fig,
                strategy_fig,
                equity,
                buying_power,
                pnl,
                investment_surety_text,
                last_trade_result,
            )

        except Exception as e:
            logging.error(f"Error in update_dashboard: {e}")
            return {}, {}, {}, "Error", "Error", "Error", "Error", "Error"
    return app

if __name__ == "__main__":
    port = 8050
    app = create_dashboard()
    webbrowser.open_new(f"http://127.0.0.1:{port}")
    app.run_server(debug=True, use_reloader=False, port=port)