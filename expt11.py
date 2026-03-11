# Imports
import requests
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
import os

# Create a folder for debugging DataFrames
DEBUG_FOLDER = "debug_dataframes"
if not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)
    
def save_dataframe_to_csv(df, filename):
    """
    Save a DataFrame to a CSV file in the debug_dataframes folder.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the CSV file (without path).
    """
    try:
        filepath = os.path.join(DEBUG_FOLDER, filename)
        df.to_csv(filepath, index=True)
        logging.info(f"Saved DataFrame to {filepath}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to {filename}: {e}")

# Polygon API for trading and market data
from polygon import RESTClient  # Install with pip install polygon-api-client
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
RAPIDAPI_KEY = "REDACTED_RAPIDAPI_KEY"  # Replace with your RapidAPI key for Twitter
ALPHA_VANTAGE_API_KEY = "REDACTED_ALPHA_VANTAGE_KEY"  # Replace with your Alpha Vantage API key
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
RISK_FREE_RATE = 0.07365  # Risk-free rate for Sharpe ratio
INITIAL_BALANCE = 1000  # Initial portfolio balance


def fetch_with_throttle(url, params):
        """
        Wrapper for API requests with throttling to handle rate limits.
        """
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Too Many Requests
                logging.warning("Rate limit hit. Waiting for 12 seconds...")
                time.sleep(12)  # Wait before retrying
                return requests.get(url, params=params)
            else:
                response.raise_for_status()
        except Exception as e:
            logging.error(f"Error during API request: {e}")
            return None

class AlphaVantageAPI:
    """
    Wrapper for Alpha Vantage API for trading and market data.
    """
    


    @staticmethod
    def fetch_latest_price(ticker, asset_type="stock"):
        """
        Fetch the latest price for a stock or cryptocurrency using Alpha Vantage API.

        Parameters:
            ticker (str): The ticker symbol for the asset.
            asset_type (str): Type of asset - "stock" or "crypto".

        Returns:
            float: The latest price of the asset, or None if invalid.
        """
        try:
            if asset_type == "stock":
                # Fetch latest stock price
                url = f"{ALPHA_VANTAGE_BASE_URL}"
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": ticker,
                    "apikey": ALPHA_VANTAGE_API_KEY,
                }
            elif asset_type == "crypto":
                # Fetch latest cryptocurrency price
                url = f"{ALPHA_VANTAGE_BASE_URL}"
                params = {
                    "function": "CURRENCY_EXCHANGE_RATE",
                    "from_currency": ticker,
                    "to_currency": "USD",
                    "apikey": ALPHA_VANTAGE_API_KEY,
                }
            else:
                logging.error(f"Invalid asset type: {asset_type}")
                return None

            # Make the API request
            response = fetch_with_throttle(url, params=params)
            response.raise_for_status()  # Raise an error for bad HTTP responses
            data = response.json()

            # Parse the price data
            if asset_type == "stock":
                if "Global Quote" in data and "05. price" in data["Global Quote"]:
                    return float(data["Global Quote"]["05. price"])
                else:
                    logging.warning(f"Invalid or incomplete data for stock {ticker}: {data}")
                    return None
            elif asset_type == "crypto":
                if "Realtime Currency Exchange Rate" in data and "5. Exchange Rate" in data["Realtime Currency Exchange Rate"]:
                    return float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
                else:
                    logging.warning(f"Invalid or incomplete data for crypto {ticker}: {data}")
                    return None

        except Exception as e:
            logging.error(f"Error fetching latest price for {ticker}: {e}")
            return None


    @staticmethod
    def fetch_historical_data(ticker, start_date, end_date, asset_type="stock"):
        """
        Fetch historical data for a stock or cryptocurrency using Alpha Vantage API.

        Parameters:
            ticker (str): The ticker symbol for the asset.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            asset_type (str): Type of asset - "stock" or "crypto".

        Returns:
            pd.DataFrame: DataFrame with historical price data, or an empty DataFrame if invalid.
        """
        try:
            # Decide the function based on the asset type
            if asset_type == "stock":
                function = "TIME_SERIES_DAILY_ADJUSTED"
            elif asset_type == "crypto":
                function = "DIGITAL_CURRENCY_DAILY"
            else:
                logging.error(f"Invalid asset type: {asset_type}")
                return pd.DataFrame()

            # Fetch historical data
            url = f"{ALPHA_VANTAGE_BASE_URL}"
            params = {
                "function": function,
                "symbol": ticker,
                "market": "USD",
                "apikey": ALPHA_VANTAGE_API_KEY,
            }
            response = fetch_with_throttle(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse the historical data
            if asset_type == "stock" and "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
            elif asset_type == "crypto" and "Time Series (Digital Currency Daily)" in data:
                time_series = data["Time Series (Digital Currency Daily)"]
            else:
                logging.warning(f"Invalid or incomplete historical data for {ticker}: {data}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)  # Convert index to datetime
            df = df.rename(columns={"4. close": "price"})  # Rename price column
            df = df.sort_index()  # Ensure data is sorted

            # Filter by date range
            df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

            if df.empty or "price" not in df.columns:
                logging.warning(f"No valid historical data found for {ticker} between {start_date} and {end_date}.")
            return df[["price"]]

        except Exception as e:
            logging.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()


class SimulatedTrading:
    """
    Simulate trades using Alpha Vantage data.
    """

    @staticmethod
    def execute_trade(ticker, quantity, side="buy", asset_type="stock"):
        """
        Simulate a trade (buy/sell) for a given ticker.

        Parameters:
            ticker (str): Asset ticker symbol.
            quantity (float): Number of units to buy or sell.
            side (str): "buy" or "sell".
            asset_type (str): Type of asset - "stock" or "crypto".

        Returns:
            dict: Simulated trade result, or an empty dict if invalid.
        """
        try:
            latest_price = AlphaVantageAPI.fetch_latest_price(ticker, asset_type=asset_type)
            if latest_price is None or latest_price <= 0:
                raise ValueError("Invalid price fetched for trade simulation.")

            total_cost = quantity * latest_price
            return {
                "ticker": ticker,
                "quantity": quantity,
                "price": latest_price,
                "total_cost": total_cost,
                "side": side,
            }
        except Exception as e:
            logging.error(f"Error executing trade for {ticker}: {e}")
            return {}

        
class CoinGeckoAPI:
    """
    Wrapper for CoinGecko API for cryptocurrency data.
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
            pd.DataFrame: DataFrame with historical price, market cap, and volume data.
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            # Ensure date range does not exceed 365 days (API limitation)
            if (end_dt - start_dt).days > 365:
                logging.warning("Date range exceeds 365 days. Adjusting to the last 365 days.")
                start_dt = end_dt - timedelta(days=365)

            # Build API request
            url = f"{COINGECKO_API_URL}/coins/{ticker}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp()),
            }

            # Fetch the data
            response = requests.get(url, params=params)
            time.sleep(1)
            response.raise_for_status()
            data = response.json()

            # Check if required keys are present
            if not all(key in data for key in ["prices", "market_caps", "total_volumes"]):
                logging.warning(f"Missing data in response for {ticker}. Response: {data}")
                return pd.DataFrame()

            # Convert each key into a DataFrame
            prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
            total_volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])

            # Merge the DataFrames on the timestamp
            combined_df = prices.merge(market_caps, on="timestamp").merge(total_volumes, on="timestamp")

            # Convert timestamp from milliseconds to datetime
            combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], unit="ms")

            # Set timestamp as the index
            return combined_df.set_index("timestamp")

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
        Fetch historical stock data for the specified tickers using Yahoo Finance.

        Parameters:
            tickers (list): List of stock tickers.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            dict: Dictionary of DataFrames keyed by ticker symbol, containing 'Adj Close' prices.
        """
        stock_data = {}
        for ticker in tickers:
            try:
                # Download historical data for the ticker
                df = yf.download(ticker, start=start_date, end=end_date)

                if df.empty:
                    logging.warning(f"No data found for {ticker}.")
                    continue

                # Use 'Adj Close' for adjusted prices. If unavailable, fallback to 'Close'.
                if "Adj Close" in df.columns:
                    stock_data[ticker] = df[["Adj Close"]].rename(columns={"Adj Close": "Close"}).sort_index()
                else:
                    logging.warning(f"'Adj Close' not available for {ticker}. Using 'Close' instead.")
                    stock_data[ticker] = df[["Close"]].sort_index()

            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {e}")
                
            time.sleep(1)

        # Return valid stock data only
        return {ticker: df for ticker, df in stock_data.items() if not df.empty}
    
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
            float: Average sentiment score (-1 to 1). Returns 0 if no valid sentiment.
        """
        url = "https://twitter-api45.p.rapidapi.com/search.php"
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": "twitter-api45.p.rapidapi.com",
        }
        querystring = {"query": ticker}

        retries = 0
        max_retries = 5
        backoff = 2  # Initial backoff in seconds

        while retries < max_retries:
            time.sleep(1)
            try:
                response = requests.get(url, headers=headers, params=querystring)
                if response.status_code in [403, 429]:
                    retries += 1
                    delay = backoff ** retries
                    logging.warning(f"Rate limit hit for {ticker}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                data = response.json()

                # Debug: Check raw API response
                logging.debug(f"Twitter API Response for {ticker}: {data}")

                tweets = data.get("timeline", [])
                if not tweets:
                    logging.warning(f"No tweets found for {ticker}.")
                    return 0

                # Extract sentiment polarities for valid tweets
                sentiments = [
                    TextBlob(tweet.get("text", "")).sentiment.polarity
                    for tweet in tweets
                    if "text" in tweet and isinstance(tweet["text"], str)
                ]

                if not sentiments:
                    logging.warning(f"No valid tweets found for {ticker}.")
                    return 0

                return np.mean(sentiments)

            except Exception as e:
                logging.error(f"Error fetching sentiment for {ticker}: {e}")
                retries += 1
                time.sleep(2)

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
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)  # Avoid division by zero
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
            logging.warning("Insufficient data for training LSTM model.")
            return 0

        # Prepare data for LSTM training
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

        # Convert prediction to a probability (0-1)
        return max(0, min(1, predicted_price))

    except Exception as e:
        logging.error(f"Error training LSTM model: {e}")
        return 0
    
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
                    y=df["price"],
                    mode="lines",
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
        data = []
        for ticker, df in stock_data.items():
            if not df.empty:
                data.append(go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    mode="lines",
                    name=ticker,
                    line=dict(width=2)
                ))

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
    
def generate_strategy_comparison_graph(crypto_data, stock_data, investment_surety_scores):
    """
    Generate strategy comparison graph.

    Parameters:
        crypto_data (dict): Cryptocurrency historical data.
        stock_data (dict): Stock historical data.
        investment_surety_scores (dict): Investment surety scores for each asset.

    Returns:
        dict: Plotly figure comparing strategies.
    """
    try:
        # Combine all prices into a single DataFrame
        all_prices = pd.DataFrame()

        for ticker, df in {**crypto_data, **stock_data}.items():
            if not df.empty:
                all_prices[ticker] = df["price"] if "price" in df.columns else df["Close"]

        # Ensure all rows have valid data and align on a common date range
        all_prices = all_prices.fillna(method="ffill").dropna()

        if all_prices.empty:
            logging.error("No valid price data available for strategy comparison.")
            return {"data": [], "layout": {"title": "No Data"}}

        # Initialize strategy values
        app_strategy_values = []  # App's strategy (based on investment surety metric)
        buy_and_hold_values = [INITIAL_BALANCE]  # Buy & Hold strategy
        risk_parity_values = [INITIAL_BALANCE]  # Risk Parity strategy

        # Calculate weights for Buy & Hold strategy
        buy_and_hold_weights = {ticker: 1 / len(all_prices.columns) for ticker in all_prices.columns}

        # Calculate weights for Risk Parity strategy (inverse of volatility)
        volatility = all_prices.pct_change().std()
        risk_parity_weights = (1 / (volatility + 1e-8)).fillna(0)  # Avoid division by zero
        risk_parity_weights /= risk_parity_weights.sum()

        # Iterate through each day to calculate strategy values
        for i in range(len(all_prices)):
            # App's Strategy
            app_metric = sum(
                investment_surety_scores.get(ticker, 0) * all_prices.iloc[i][ticker]
                for ticker in all_prices.columns
            )
            app_strategy_values.append(app_metric)

            # Buy & Hold Strategy
            if i > 0:
                buy_and_hold_value = sum(
                    buy_and_hold_weights[ticker] * all_prices.iloc[i][ticker] * INITIAL_BALANCE
                    for ticker in all_prices.columns
                )
                buy_and_hold_values.append(buy_and_hold_value)

            # Risk Parity Strategy
            if i > 0:
                risk_parity_value = sum(
                    risk_parity_weights[ticker] * all_prices.iloc[i][ticker] * INITIAL_BALANCE
                    for ticker in all_prices.columns
                )
                risk_parity_values.append(risk_parity_value)

        # Generate Plotly Graph
        data = [
            go.Scatter(
                x=all_prices.index,
                y=app_strategy_values,
                mode="lines",
                name="App's Strategy",
                line=dict(width=2, color="blue"),
            ),
            go.Scatter(
                x=all_prices.index,
                y=buy_and_hold_values,
                mode="lines",
                name="Buy and Hold",
                line=dict(width=2, color="orange"),
            ),
            go.Scatter(
                x=all_prices.index,
                y=risk_parity_values,
                mode="lines",
                name="Risk Parity",
                line=dict(width=2, color="green"),
            ),
        ]

        layout = go.Layout(
            title="Strategy Comparison",
            xaxis={"title": "Date", "type": "date", "tickformat": "%b %d, %Y"},
            yaxis={"title": "Portfolio Value (USD)"},
            template="plotly_dark",
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
        )

        return {"data": data, "layout": layout}

    except Exception as e:
        logging.error(f"Error generating strategy comparison graph: {e}")
        return {"data": [], "layout": {"title": "No Data"}}
def create_dashboard():
    """
    Create a Dash GUI for AmbiVest with cryptocurrency and stock analysis.
    """
    # Initialize the app
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

                    # Portfolio Metrics Section (Including Investment Surety and Trades)
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Portfolio Metrics", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    [
                                        html.P(id="portfolio-equity", style={"color": "white"}),
                                        html.P(id="portfolio-buying-power", style={"color": "white"}),
                                        html.P(id="portfolio-pnl", style={"color": "white"}),
                                        html.Div(
                                            id="investment-surety",
                                            style={"color": "white", "marginTop": "10px"},
                                        ),
                                        html.Div(  # Add the missing trades-executed element
                                            id="trades-executed",
                                            style={"color": "white", "marginTop": "10px"},
                                        ),
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
                            dbc.CardBody(dcc.Graph(id="strategy-comparison")),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
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
        """
        Update the dashboard with fetched data, calculations, and trades.

        Parameters:
            n_clicks (int): Number of times the Run Analysis button is clicked.
            crypto_input (str): Comma-separated cryptocurrency tickers.
            stock_input (str): Comma-separated stock tickers.
            start_date (str): Start date for analysis (YYYY-MM-DD).
            end_date (str): End date for analysis (YYYY-MM-DD).

        Returns:
            tuple: Graphs, portfolio metrics, investment surety, and trade summaries.
        """
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
            portfolio_balance = INITIAL_BALANCE
            trades = {}
            for ticker, score in investment_surety_scores.items():
                weight = score / sum(investment_surety_scores.values())
                allocation = portfolio_balance * weight

                # Fetch the latest price
                latest_price = AlphaVantageAPI.fetch_latest_price(ticker, asset_type="stock")

                # Validate the latest price
                if latest_price is None or latest_price <= 0:
                    logging.warning(f"Skipping trade for {ticker}: Invalid latest price ({latest_price}).")
                    continue  # Skip this ticker if the price is invalid

                # Calculate quantity and simulate trade
                quantity = allocation // latest_price  # Buy as many whole units as possible
                if quantity > 0:
                    trade = SimulatedTrading.execute_trade(ticker, quantity, side="buy", asset_type="stock")
                    trades[ticker] = trade
                else:
                    logging.warning(f"Skipping trade for {ticker}: Allocation too small for trade.")

            # Calculate portfolio metrics
            total_trades_value = sum(trade["total_cost"] for trade in trades.values() if "total_cost" in trade)
            portfolio_equity = portfolio_balance
            buying_power = portfolio_balance - total_trades_value
            pnl = total_trades_value - INITIAL_BALANCE

            # **8. Generate Graphs**
            crypto_fig = generate_crypto_graph(crypto_data)
            stock_fig = generate_stock_graph(stock_data)
            strategy_fig = generate_strategy_comparison_graph(crypto_data, stock_data, investment_surety_scores)

            # **9. Format Outputs**
            equity_text = f"Portfolio Equity: ${portfolio_equity:.2f}"
            buying_power_text = f"Buying Power: ${buying_power:.2f}"
            pnl_text = f"PnL: ${pnl:.2f}"
            investment_surety_text = "\n".join(
                [f"{ticker}: {score:.2f}" for ticker, score in investment_surety_scores.items()]
            )
            trades_text = "\n".join(
                [f"{trade['ticker']}: {trade['quantity']} units at ${trade['price']:.2f} ({trade['side']})"
                for trade in trades.values()]
            )

            # **10. Return Outputs**
            return (
                crypto_fig,
                stock_fig,
                strategy_fig,
                equity_text,
                buying_power_text,
                pnl_text,
                investment_surety_text,
                trades_text,
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