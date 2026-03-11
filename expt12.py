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
import os

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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

# Constants
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
RAPIDAPI_KEY = "REDACTED_RAPIDAPI_KEY"  # Replace with your RapidAPI key for Twitter
RISK_FREE_RATE = 0.07365  # Risk-free rate for Sharpe ratio
INITIAL_BALANCE = 1000  # Initial portfolio balance
DEVELOPER_KEY = "econ3086@hkbu"  # Developer key for advanced metrics access

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
            dict: Dictionary of DataFrames keyed by ticker symbol, each with a "price" column.
        """
        stock_data = {}
        for ticker in tickers:
            try:
                # Download historical data for the ticker
                df = yf.download(ticker, start=start_date, end=end_date)

                if df.empty:
                    logging.warning(f"No data found for {ticker}.")
                    continue

                # Rename columns
                if "Adj Close" in df.columns:
                    df = df.rename(columns={"Adj Close": "price"})
                elif "Close" in df.columns:
                    df = df.rename(columns={"Close": "price"})
                else:
                    logging.warning(f"No 'Close' or 'Adj Close' column found for {ticker}.")
                    continue

                # Save the DataFrame for debugging
                save_dataframe_to_csv(df, f"{ticker}_stock_data.csv")

                stock_data[ticker] = df[["price"]]

            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {e}")

        return stock_data

    @staticmethod
    def fetch_latest_price(ticker):
        """
        Fetch the latest price for a stock using Yahoo Finance.

        Parameters:
            ticker (str): Stock ticker.

        Returns:
            float: The latest price of the stock, or None if invalid.
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
            else:
                logging.warning(f"No latest price data found for {ticker}.")
                return None

        except Exception as e:
            logging.error(f"Error fetching latest price for {ticker}: {e}")
            return None
        
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
            pd.DataFrame: DataFrame with "price" and "timestamp" columns, or an empty DataFrame if invalid.
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
            if not response or response.status_code != 200:
                logging.warning(f"Failed to fetch crypto data for {ticker}.")
                return pd.DataFrame()

            data = response.json()

            # Parse data
            if "prices" not in data or not data["prices"]:
                logging.warning(f"No price data found for {ticker}. Response: {data}")
                return pd.DataFrame()

            # Create DataFrame
            prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")  # Convert timestamp to datetime
            prices.set_index("timestamp", inplace=True)

            # Filter by date range
            prices = prices.loc[(prices.index >= start_date) & (prices.index <= end_date)]

            # Save the DataFrame for debugging
            save_dataframe_to_csv(prices, f"{ticker}_crypto_data.csv")

            return prices

        except Exception as e:
            logging.error(f"Error fetching crypto data for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_latest_price(ticker):
        """
        Fetch the latest price for a cryptocurrency using CoinGecko.

        Parameters:
            ticker (str): Cryptocurrency ticker (CoinGecko ID).

        Returns:
            float: The latest price of the cryptocurrency, or None if invalid.
        """
        try:
            url = f"{COINGECKO_API_URL}/simple/price"
            params = {
                "ids": ticker,
                "vs_currencies": "usd",
            }

            # Fetch the data
            response = requests.get(url, params=params)
            if not response or response.status_code != 200:
                logging.warning(f"Failed to fetch latest price for {ticker}.")
                return None

            data = response.json()
            if ticker in data and "usd" in data[ticker]:
                return float(data[ticker]["usd"])
            else:
                logging.warning(f"Invalid or incomplete data for crypto {ticker}: {data}")
                return None

        except Exception as e:
            logging.error(f"Error fetching latest price for {ticker}: {e}")
            return None
        
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
    Calculate technical indicators (SMA, EMA, RSI, DMAC) for price data.

    Parameters:
        data (pd.DataFrame): DataFrame with price data.

    Returns:
        dict: Normalized technical indicator scores.
    """
    try:
        if data.empty or "price" not in data.columns:
            logging.warning("Data is empty or missing 'price' column.")
            return {"SMA": 0, "EMA": 0, "RSI": 0, "DMAC": 0}

        # Calculate SMA and EMA
        data["SMA"] = data["price"].rolling(window=20).mean()
        data["EMA"] = data["price"].ewm(span=20).mean()

        # Calculate RSI
        delta = data["price"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)  # Avoid division by zero
        data["RSI"] = 100 - (100 / (1 + rs))

        # Calculate DMAC (Double Moving Average Crossover)
        data["SMA_short"] = data["price"].rolling(window=10).mean()
        data["SMA_long"] = data["price"].rolling(window=30).mean()
        data["DMAC"] = (data["SMA_short"] - data["SMA_long"]) / data["price"]

        # Normalize indicators to [0, 1]
        indicators = {
            "SMA": (data["SMA"].iloc[-1] - data["price"].min()) / (data["price"].max() - data["price"].min()),
            "EMA": (data["EMA"].iloc[-1] - data["price"].min()) / (data["price"].max() - data["price"].min()),
            "RSI": data["RSI"].iloc[-1] / 100,
            "DMAC": (data["DMAC"].iloc[-1] + 1) / 2,  # Shift DMAC to [0, 1]
        }

        return indicators
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return {"SMA": 0, "EMA": 0, "RSI": 0, "DMAC": 0}
    
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
    
def calculate_sharpe_ratio(data):
    """
    Calculate the Sharpe Ratio for price data.

    Parameters:
        data (pd.DataFrame): DataFrame with price data.

    Returns:
        float: Sharpe Ratio (risk-adjusted return).
    """
    try:
        if data.empty or "price" not in data.columns:
            logging.warning("Data is empty or missing 'price' column.")
            return 0

        # Calculate daily returns
        returns = data["price"].pct_change().dropna()

        # Calculate Sharpe Ratio
        mean_return = returns.mean()
        std_dev = returns.std()
        sharpe_ratio = (mean_return - RISK_FREE_RATE / 252) / (std_dev + 1e-8)  # Avoid division by zero

        # Normalize Sharpe Ratio to [0, 1]
        return max(0, min(1, (sharpe_ratio + 1) / 2))  # Shift to [0, 1]
    except Exception as e:
        logging.error(f"Error calculating Sharpe Ratio: {e}")
        return 0
    
def calculate_investment_surety(sentiment, technical_scores, ml_score, sharpe_ratio):
    """
    Calculate the Investment Surety Index (ISI) using optimized weights.

    Parameters:
        sentiment (float): Sentiment score (0 to 1).
        technical_scores (dict): Technical indicator scores (SMA, EMA, RSI, DMAC).
        ml_score (float): Machine Learning prediction score (0 to 1).
        sharpe_ratio (float): Sharpe Ratio score (0 to 1).

    Returns:
        float: Investment Surety Index (ISI) (0 to 1).
        dict: Optimized weights for the ISI components.
    """
    try:
        # Combine all metrics into a single list
        metrics = [
            sentiment,
            technical_scores.get("SMA", 0),
            technical_scores.get("EMA", 0),
            technical_scores.get("RSI", 0),
            technical_scores.get("DMAC", 0),
            ml_score,
            sharpe_ratio,
        ]

        # Define the objective function to maximize the weighted sum
        def objective(weights):
            return -sum(w * m for w, m in zip(weights, metrics))  # Negative for maximization

        # Constraints: weights must sum to 1
        constraints = {"type": "eq", "fun": lambda w: sum(w) - 1}

        # Bounds: weights must lie between 0 and 1
        bounds = [(0, 1)] * len(metrics)

        # Initial weights (equal distribution)
        initial_weights = [1 / len(metrics)] * len(metrics)

        # Perform optimization
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

        # Extract optimized weights
        optimized_weights = result.x

        # Calculate the ISI using the optimized weights
        isi = -objective(optimized_weights)  # Negate the value to get the positive ISI

        return isi, dict(zip(["sentiment", "SMA", "EMA", "RSI", "DMAC", "ML", "Sharpe"], optimized_weights))

    except Exception as e:
        logging.error(f"Error calculating Investment Surety Index: {e}")
        return 0, {}
    
class SimulatedTrading:
    """
    Simulate trades using the Investment Surety Index (ISI).
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
            # Fetch the latest price
            if asset_type == "crypto":
                latest_price = CoinGeckoAPI.fetch_latest_price(ticker)
            else:
                latest_price = YahooFinanceAPI.fetch_latest_price(ticker)

            # Validate the latest price
            if latest_price is None or latest_price <= 0:
                raise ValueError(f"Invalid price fetched for {ticker}: {latest_price}")

            # Calculate total trade cost
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
        
def update_portfolio(trades, initial_balance):
    """
    Update portfolio metrics based on executed trades.

    Parameters:
        trades (dict): Dictionary of executed trades.
        initial_balance (float): Initial portfolio balance.

    Returns:
        dict: Updated portfolio metrics (equity, buying power, PnL).
    """
    try:
        total_trades_value = sum(trade["total_cost"] for trade in trades.values() if "total_cost" in trade)
        portfolio_equity = initial_balance
        buying_power = initial_balance - total_trades_value
        pnl = total_trades_value - initial_balance

        return {
            "equity": portfolio_equity,
            "buying_power": buying_power,
            "pnl": pnl,
        }
    except Exception as e:
        logging.error(f"Error updating portfolio metrics: {e}")
        return {"equity": 0, "buying_power": 0, "pnl": 0}
    
def make_trading_decision(ticker, isi, threshold, max_allocation=1000, asset_type="stock"):
    """
    Make a trading decision based on the ISI.

    Parameters:
        ticker (str): Asset ticker symbol.
        isi (float): Investment Surety Index score.
        threshold (float): Threshold for buy/sell decisions.
        max_allocation (float): Maximum allocation for the trade.
        asset_type (str): Type of asset - "stock" or "crypto".

    Returns:
        dict: Trade details or None if no trade is made.
    """
    try:
        if isi >= threshold:
            # Buy decision
            latest_price = CoinGeckoAPI.fetch_latest_price(ticker) if asset_type == "crypto" else YahooFinanceAPI.fetch_latest_price(ticker)
            if not latest_price or latest_price <= 0:
                logging.warning(f"Skipping trade for {ticker}: Invalid price ({latest_price}).")
                return None

            quantity = max_allocation // latest_price
            if quantity > 0:
                return SimulatedTrading.execute_trade(ticker, quantity, side="buy", asset_type=asset_type)
            else:
                logging.warning(f"Skipping trade for {ticker}: Allocation too small for trade.")
                return None
        else:
            # Hold or sell decision (for simplicity, we hold when ISI < threshold)
            logging.info(f"Holding {ticker}: ISI ({isi}) below threshold ({threshold}).")
            return None
    except Exception as e:
        logging.error(f"Error making trading decision for {ticker}: {e}")
        return None

def simulate_trading(crypto_tickers, stock_tickers, start_date, end_date, threshold, initial_balance):
    """
    Simulate trading for a portfolio of cryptocurrencies and stocks based on the ISI.

    Parameters:
        crypto_tickers (list): List of cryptocurrency tickers (CoinGecko IDs).
        stock_tickers (list): List of stock tickers (Yahoo Finance tickers).
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        threshold (float): ISI threshold for buy/sell decisions.
        initial_balance (float): Initial portfolio balance.

    Returns:
        dict: Results of the simulation, including portfolio metrics and executed trades.
    """
    try:
        # Fetch historical data
        crypto_data = {ticker: CoinGeckoAPI.fetch_crypto_data(ticker, start_date, end_date) for ticker in crypto_tickers}
        stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, start_date, end_date)

        # Initialize portfolio and trades
        portfolio_balance = initial_balance
        trades = {}
        portfolio_metrics = {"equity": initial_balance, "buying_power": initial_balance, "pnl": 0}

        for ticker in crypto_tickers + stock_tickers:
            # Determine asset type
            asset_type = "crypto" if ticker in crypto_tickers else "stock"
            data = crypto_data.get(ticker, pd.DataFrame()) if asset_type == "crypto" else stock_data.get(ticker, pd.DataFrame())

            if data.empty:
                logging.warning(f"No valid data found for {ticker}. Skipping.")
                continue

            # Calculate metrics
            sentiment = TwitterSentimentAnalysis.fetch_sentiment(ticker)
            technical_scores = calculate_technical_indicators(data)
            ml_score = train_lstm_model(data)
            sharpe_ratio = calculate_sharpe_ratio(data)

            # Calculate ISI
            isi, weights = calculate_investment_surety(sentiment, technical_scores, ml_score, sharpe_ratio)

            # Make a trading decision
            trade = make_trading_decision(ticker, isi, threshold, max_allocation=portfolio_balance, asset_type=asset_type)
            if trade:
                trades[ticker] = trade
                portfolio_balance -= trade["total_cost"]  # Deduct trade cost from portfolio balance

        # Update portfolio metrics
        portfolio_metrics = update_portfolio(trades, initial_balance)

        # Return results
        return {
            "metrics": portfolio_metrics,
            "trades": trades,
            "crypto_data": crypto_data,
            "stock_data": stock_data,
        }

    except Exception as e:
        logging.error(f"Error during trading simulation: {e}")
        return {"metrics": {}, "trades": {}, "crypto_data": {}, "stock_data": {}}

def create_dashboard():
    """
    Create a Dash GUI for AmbiVest with cryptocurrency and stock analysis.
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

    # Layout
    app.layout = dbc.Container(
        [
            # Header
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

            # Developer Key Section
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Developer Access", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                [
                                    html.Label("Enter Developer Key", style={"color": "white"}),
                                    dcc.Input(
                                        id="developer-key",
                                        type="password",
                                        placeholder="Enter developer key here",
                                        style={"width": "100%", "marginBottom": "10px"},
                                    ),
                                    dbc.Button(
                                        "Submit Key",
                                        id="submit-key",
                                        color="warning",
                                        style={"width": "100%"},
                                    ),
                                    html.Div(id="access-message", style={"color": "white", "marginTop": "10px"}),
                                ]
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    )
                ),
                style={"marginTop": "20px"},
            ),

            # ISI Threshold Slider Section
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Adjust ISI Threshold", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                [
                                    html.Label(
                                        "Investment Surety Index (ISI) Threshold",
                                        style={"color": "white"},
                                    ),
                                    dcc.Slider(
                                        id="isi-threshold-slider",
                                        min=0,
                                        max=1,
                                        step=0.01,
                                        value=0.7,  # Default threshold
                                        marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11)},
                                    ),
                                    html.Div(
                                        id="threshold-value",
                                        style={"color": "white", "marginTop": "10px"},
                                    ),
                                ]
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    )
                ),
                style={"marginTop": "20px"},
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
                                        html.Div(id="investment-surety", style={"color": "white", "marginTop": "10px"}),
                                        html.Div(id="trades-executed", style={"color": "white", "marginTop": "10px"}),
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

            # Detailed Metrics Section (Developer Access)
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Detailed Metrics (Developer Access)", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="investment-surety-breakdown"),
                                            html.Div(id="sentiment-analysis-values", style={"color": "white", "marginTop": "10px"}),
                                        ],
                                        id="developer-metrics",
                                        style={"display": "none"},  # Hidden by default
                                    ),
                                ]
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    )
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
            Output("trades-executed", "children"),
        ],
        [
            Input("run-button", "n_clicks"),
            Input("isi-threshold-slider", "value"),
        ],
        [
            State("crypto-input", "value"),
            State("stock-input", "value"),
            State("start-date", "value"),
            State("end-date", "value"),
        ],
    )
    def update_dashboard(n_clicks, threshold, crypto_input, stock_input, start_date, end_date):
        """
        Update the dashboard with trading results when the "Run Analysis" button is clicked.

        Parameters:
            n_clicks (int): Number of times the Run Analysis button is clicked.
            threshold (float): ISI threshold for buy/sell decisions.
            crypto_input (str): Comma-separated cryptocurrency tickers.
            stock_input (str): Comma-separated stock tickers.
            start_date (str): Start date for analysis (YYYY-MM-DD).
            end_date (str): End date for analysis (YYYY-MM-DD).

        Returns:
            tuple: Updated graphs and portfolio metrics.
        """
        if n_clicks is None:
            return {}, {}, {}, "No data", "No data", "No data", "No trades executed."

        try:
            # Parse inputs
            crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
            stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

            # Default date range
            start_date = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")

            # Simulate trading
            results = simulate_trading(crypto_tickers, stock_tickers, start_date, end_date, threshold, INITIAL_BALANCE)

            # Extract results
            portfolio_metrics = results["metrics"]
            trades = results["trades"]
            crypto_data = results["crypto_data"]
            stock_data = results["stock_data"]

            # Generate graphs
            crypto_fig = generate_crypto_graph(crypto_data)
            stock_fig = generate_stock_graph(stock_data)
            strategy_fig = generate_strategy_comparison_graph(crypto_data, stock_data, {})

            # Format portfolio metrics
            equity_text = f"Portfolio Equity: ${portfolio_metrics['equity']:.2f}"
            buying_power_text = f"Buying Power: ${portfolio_metrics['buying_power']:.2f}"
            pnl_text = f"PnL: ${portfolio_metrics['pnl']:.2f}"

            # Format trade summaries
            trades_text = "\n".join(
                [
                    f"{trade['ticker']}: {trade['quantity']} units at ${trade['price']:.2f} ({trade['side']})"
                    for trade in trades.values()
                ]
            )

            return crypto_fig, stock_fig, strategy_fig, equity_text, buying_power_text, pnl_text, trades_text

        except Exception as e:
            logging.error(f"Error updating dashboard: {e}")
            return {}, {}, {}, "Error", "Error", "Error", "Error"
        
    @app.callback(
        Output("threshold-value", "children"),
        Input("isi-threshold-slider", "value"),
    )
    def update_threshold_display(threshold):
        """
        Update the displayed ISI threshold value when the slider is adjusted.

        Parameters:
            threshold (float): ISI threshold value.

        Returns:
            str: Displayed threshold value.
        """
        return f"Current Threshold: {threshold:.2f}"

    @app.callback(
        [Output("developer-metrics", "style"), Output("access-message", "children")],
        [Input("submit-key", "n_clicks")],
        [State("developer-key", "value")],
    )
    def validate_developer_key(n_clicks, entered_key):
        """
        Validate the entered developer key and toggle the visibility of advanced metrics.

        Parameters:
            n_clicks (int): Number of times the "Submit Key" button is clicked.
            entered_key (str): The entered developer key.

        Returns:
            tuple: Style for the advanced metrics section and access message.
        """
        if n_clicks is None:
            return {"display": "none"}, ""
        
        # Validate the developer key
        if entered_key == DEVELOPER_KEY:
            return {"display": "block"}, "Access granted. Advanced metrics are now visible."
        else:
            return {"display": "none"}, "Invalid key. Please try again."
        
    @app.callback(
        Output("investment-surety-breakdown", "figure"),
        [Input("run-button", "n_clicks")],
        [
            State("crypto-input", "value"),
            State("stock-input", "value"),
            State("start-date", "value"),
            State("end-date", "value"),
            State("isi-threshold-slider", "value"),
        ],
    )
    def update_isi_breakdown(n_clicks, crypto_input, stock_input, start_date, end_date, threshold):
        """
        Update the ISI breakdown pie chart based on the trading simulation.

        Parameters:
            n_clicks (int): Number of times the Run Analysis button is clicked.
            crypto_input (str): Comma-separated cryptocurrency tickers.
            stock_input (str): Comma-separated stock tickers.
            start_date (str): Start date for analysis (YYYY-MM-DD).
            end_date (str): End date for analysis (YYYY-MM-DD).
            threshold (float): ISI threshold for buy/sell decisions.

        Returns:
            dict: Updated pie chart figure for ISI breakdown.
        """
        if n_clicks is None:
            return {"data": [], "layout": {"title": "No Data"}}

        try:
            # Parse inputs
            crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
            stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

            # Default date range
            start_date = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")

            # Fetch data for the first ticker as an example
            ticker = crypto_tickers[0] if crypto_tickers else (stock_tickers[0] if stock_tickers else None)
            if not ticker:
                logging.warning("No tickers provided.")
                return {"data": [], "layout": {"title": "No Data"}}

            # Determine asset type and fetch data
            asset_type = "crypto" if ticker in crypto_tickers else "stock"
            data = (
                CoinGeckoAPI.fetch_crypto_data(ticker, start_date, end_date)
                if asset_type == "crypto"
                else YahooFinanceAPI.fetch_stock_data([ticker], start_date, end_date).get(ticker, pd.DataFrame())
            )

            if data.empty:
                logging.warning(f"No data available for {ticker}.")
                return {"data": [], "layout": {"title": "No Data"}}

            # Calculate metrics
            sentiment = TwitterSentimentAnalysis.fetch_sentiment(ticker)
            technical_scores = calculate_technical_indicators(data)
            ml_score = train_lstm_model(data)
            sharpe_ratio = calculate_sharpe_ratio(data)

            # Calculate ISI and weights
            _, weights = calculate_investment_surety(sentiment, technical_scores, ml_score, sharpe_ratio)

            # Generate pie chart
            return generate_isi_breakdown_pie_chart(weights)

        except Exception as e:
            logging.error(f"Error updating ISI breakdown: {e}")
            return {"data": [], "layout": {"title": "No Data"}}
    return app

if __name__ == "__main__":
    app = create_dashboard()
    app.run_server(debug=True, use_reloader=False, port=8050)