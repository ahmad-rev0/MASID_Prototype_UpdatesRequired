# BLOCK 1: SETUP AND CONFIGURATION
import subprocess
bat_file_path = "pip_installs.bat"
subprocess.run([bat_file_path], shell=True)
# Import necessary libraries
import requests
import pandas as pd
import numpy as np
import logging
import yfinance as yf
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import re
import matplotlib.pyplot as plt


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants for API Configuration
RAPIDAPI_KEY = "REDACTED_RAPIDAPI_KEY"  # Replace with your RapidAPI key
RAPIDAPI_HOST = "twitter154.p.rapidapi.com"  # RapidAPI Host for Twitter
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
RISK_FREE_RATE = 0.02  # Risk-free rate for Sharpe ratio calculation
INITIAL_BALANCE = 1000  # Default initial balance for backtesting
LOOKBACK_WINDOW = 14  # Lookback window for indicators
SENTIMENT_THRESHOLD = 0.2  # Threshold for sentiment-based decisions
LEARNING_RATE = 0.01  # Learning rate for weight adjustments
DEVELOPER_KEY = "econ3086@hkbu"  # Developer key for debugging mode

# Global asset selection
CRYPTO_SYMBOLS = ["bitcoin", "ethereum"]
STOCK_SYMBOLS = ["AAPL", "MSFT"]


# BLOCK 2: API WRAPPERS FOR DATA FETCHING

# CoinGecko API Wrapper for Cryptocurrencies
class CoinGeckoAPI:
    """
    Wrapper for CoinGecko API to fetch price data for cryptocurrencies.
    """

# Headers for CoinGecko API requests (with demo API key)
HEADERS = {
    "accept": "application/json",
    "x-cg-demo-api-key": "REDACTED_COINGECKO_KEY"
}

class CoinGeckoAPI:
    """
    Wrapper for CoinGecko API to fetch price data and historical data for cryptocurrencies.
    """
    
    @staticmethod
    def get_price(symbol, currency="usd"):
        """
        Fetches the current price of a cryptocurrency in the specified currency.
        
        Args:
            symbol (str): The CoinGecko ID of the cryptocurrency (e.g., bitcoin, ethereum).
            currency (str): The fiat or cryptocurrency to compare against (default is 'usd').

        Returns:
            float or None: The current price of the cryptocurrency or None if an error occurs.
        """
        url = f"{COINGECKO_API_URL}/simple/price"
        params = {"ids": symbol, "vs_currencies": currency}

        try:
            # Make the GET request
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            # Extract and return the price
            return data.get(symbol, {}).get(currency, None)
        except requests.exceptions.RequestException as e:
            # Log error if request fails
            logging.error(f"HTTP error fetching price for {symbol}: {e}")
        except Exception as e:
            # Log any other error
            logging.error(f"Unexpected error fetching price for {symbol}: {e}")
        return None

    @staticmethod
    def get_historical_data(symbol, currency="usd", days=30):
        """
        Fetches historical price data for a cryptocurrency for a given number of days.

        Args:
            symbol (str): The CoinGecko ID of the cryptocurrency (e.g., bitcoin, ethereum).
            currency (str): The fiat or cryptocurrency to compare against (default is 'usd').
            days (int): The number of days to fetch historical data for (default is 30).

        Returns:
            pd.DataFrame: A DataFrame containing timestamps and prices or an empty DataFrame if an error occurs.
        """
        url = f"{COINGECKO_API_URL}/coins/{symbol}/market_chart"
        params = {
            "vs_currency": currency,
            "days": days,
            "interval": "daily"  # Fetch daily prices
        }

        try:
            # Make the GET request
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            # Extract price data
            prices = data.get("prices", [])
            if not prices:
                logging.warning(f"No historical prices found for {symbol}")
                return pd.DataFrame()  # Return empty DataFrame if no data is available
            # Convert to DataFrame
            historical_data = pd.DataFrame(prices, columns=["timestamp", "price"])
            historical_data["timestamp"] = pd.to_datetime(historical_data["timestamp"], unit="ms")
            return historical_data
        except requests.exceptions.RequestException as e:
            # Log error if request fails
            logging.error(f"HTTP error fetching historical data for {symbol}: {e}")
        except Exception as e:
            # Log any other error
            logging.error(f"Unexpected error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()


# Yahoo Finance Wrapper for Stock Data
class YahooFinanceAPI:
    """
    Wrapper for Yahoo Finance API for fetching stock data.
    """
    @staticmethod
    def get_historical_data(symbol, start_date, end_date):
        """
        Fetch historical stock price data using yfinance.

        Parameters:
            symbol (str): Stock symbol (e.g., 'AAPL').
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame with historical stock data.
        """
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            stock_data = stock_data.reset_index()
            stock_data = stock_data.rename(columns={"Close": "price", "Date": "timestamp"})
            return stock_data[["timestamp", "price"]]
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
        
# BLOCK 3: TECHNICAL INDICATORS MODULE

def calculate_technical_indicators(data, lookback_window=14):
    """
    Calculate technical indicators: SMA, EMA, RSI, and Bollinger Bands.

    Parameters:
        data (pd.DataFrame): DataFrame with price data.
        lookback_window (int): Lookback window for calculations.

    Returns:
        pd.DataFrame: DataFrame with technical indicators added.
    """
    data = data.copy()

    # Simple Moving Average (SMA)
    data["SMA"] = data["price"].rolling(window=lookback_window).mean()

    # Exponential Moving Average (EMA)
    data["EMA"] = data["price"].ewm(span=lookback_window, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = data["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=lookback_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=lookback_window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data["BB_Upper"] = data["SMA"] + 2 * data["price"].rolling(window=lookback_window).std()
    data["BB_Lower"] = data["SMA"] - 2 * data["price"].rolling(window=lookback_window).std()

    return data
# BLOCK 4: TWITTER API WRAPPER

class RapidAPITwitter:
    """
    Wrapper for the Twitter API via RapidAPI to fetch tweets for sentiment analysis.
    """

    def __init__(self, api_key, host="twitter-api45.p.rapidapi.com"):
        """
        Initialize the Twitter API wrapper.

        Parameters:
            api_key (str): Your RapidAPI key.
            host (str): RapidAPI host for the Twitter API.
        """
        self.api_key = api_key
        self.host = host

    def fetch_tweets(self, query, count=10):
        """
        Fetch tweets based on a query.

        Parameters:
            query (str): Query string (e.g., hashtag or keyword).
            count (int): Number of tweets to fetch.

        Returns:
            list: List of fetched tweets (text).
        """
        url = f"https://{self.host}/search.php"
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }
        params = {
            "q": query,
            "count": count
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return [tweet["text"] for tweet in data.get("statuses", [])]  # Extract tweet texts
        except Exception as e:
            logging.error(f"Error fetching tweets for query '{query}': {e}")
            return []
# BLOCK 5: MACHINE LEARNING MODULE (UPDATED WITH LSTM MODEL AND SENTIMENT ANALYSIS)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
import numpy as np
import re
import logging

class MLModel:
    """
    A simple ML model for price prediction using Linear Regression.
    """
    def __init__(self):
        self.model = LinearRegression()

    def train(self, data):
        """
        Train the model on historical price data.

        Parameters:
            data (pd.DataFrame): DataFrame with historical price data. Must include "price" column.

        Returns:
            self: Trained MLModel instance.
        """
        # Validate input data
        if "price" not in data.columns:
            raise ValueError("Input data must contain a 'price' column.")

        # Shift the target column to predict the next day's price
        data["target"] = data["price"].shift(-1)
        data = data.dropna()

        # Features (X) and target (y)
        X = data[["price"]]
        y = data["target"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model on the test set
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Trained ML model with Mean Squared Error (MSE): {mse:.4f}")

        return self

    def predict(self, price):
        """
        Predict the next price based on the current price.

        Parameters:
            price (float): Current price.

        Returns:
            float: Predicted next price.
        """
        # Validate input
        if not isinstance(price, (float, int)):
            raise ValueError("Input price must be a float or integer.")
        return self.model.predict([[price]])[0]


class LSTMModel:
    """
    A deep learning LSTM model for price prediction.
    """
    def __init__(self, lookback=30):
        """
        Initialize the LSTM model.

        Parameters:
            lookback (int): Number of previous time steps used for prediction.
        """
        self.model = None
        self.lookback = lookback
        self.scaler = MinMaxScaler()

    def prepare_data(self, data):
        """
        Prepare data for LSTM training.

        Parameters:
            data (pd.DataFrame): Historical price data with a 'price' column.

        Returns:
            np.array, np.array: Prepared training features (X) and labels (y).
        """
        # Validate input data
        if "price" not in data.columns:
            raise ValueError("Input data must contain a 'price' column.")

        # Scale the price data to the range [0, 1]
        scaled_data = self.scaler.fit_transform(data["price"].values.reshape(-1, 1))
        X, y = [], []

        # Create sequences of `lookback` length for training
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i, 0])  # Features: last `lookback` prices
            y.append(scaled_data[i, 0])  # Label: next price

        return np.array(X), np.array(y)

    def build_model(self):
        """
        Build the LSTM model architecture.
        """
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback, 1)),  # First LSTM layer
            Dropout(0.2),  # Regularization
            LSTM(50, return_sequences=False),  # Second LSTM layer
            Dropout(0.2),
            Dense(25),  # Dense layer with 25 neurons
            Dense(1)  # Output layer (single price prediction)
        ])
        self.model.compile(optimizer="adam", loss="mean_squared_error")  # Compile with MSE loss

    def train(self, data):
        """
        Train the LSTM model on historical price data.

        Parameters:
            data (pd.DataFrame): Historical price data with a 'price' column.

        Returns:
            self: Trained LSTMModel instance.
        """
        # Prepare the training data
        X, y = self.prepare_data(data)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input format

        # Build and train the model
        self.build_model()
        self.model.fit(X, y, batch_size=1, epochs=10, verbose=1)  # Small batch size for time-series

        return self

    def predict(self, data):
        """
        Predict the next price using the trained LSTM model.

        Parameters:
            data (pd.DataFrame): Historical price data with a 'price' column.

        Returns:
            float: Predicted next price.
        """
        # Validate input data
        if "price" not in data.columns:
            raise ValueError("Input data must contain a 'price' column.")

        # Scale the price data
        scaled_data = self.scaler.transform(data["price"].values.reshape(-1, 1))

        # Use the last `lookback` prices for prediction
        last_sequence = scaled_data[-self.lookback:].reshape(1, self.lookback, 1)

        # Predict and inverse scale the result
        scaled_prediction = self.model.predict(last_sequence)
        return self.scaler.inverse_transform(scaled_prediction)[0, 0]


def fetch_and_analyze_sentiment(twitter_api, query, count=10):
    """
    Fetch tweets and analyze their sentiment.

    Parameters:
        twitter_api (RapidAPITwitter): Instance of the RapidAPITwitter class.
        query (str): Query string (e.g., cryptocurrency or stock symbol).
        count (int): Number of tweets to fetch.

    Returns:
        float: Average sentiment polarity of the fetched tweets.
    """
    # Fetch tweets using the Twitter API
    tweets = twitter_api.fetch_tweets(query, count)
    if not tweets:
        logging.warning(f"No tweets fetched for query '{query}'. Returning neutral sentiment.")
        return 0.0  # Return neutral sentiment if no tweets are fetched

    # Analyze sentiment for each tweet
    sentiments = []
    for tweet in tweets:
        # Clean the tweet text
        text = re.sub(r"http\S+", "", tweet)  # Remove URLs
        text = re.sub(r"[^a-zA-Z ]", "", text).lower()  # Remove special characters
        sentiment = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment)

    # Compute and return the average sentiment polarity
    average_sentiment = np.mean(sentiments)
    logging.info(f"Average sentiment for query '{query}': {average_sentiment:.4f}")
    return average_sentiment
    
# BLOCK 6: INVESTMENT SURETY METRIC (ISM) AND OPTIMIZATION MODULE

class InvestmentSuretyMetric:
    """
    Computes the Investment Surety Metric (ISM) and optimizes its weights.
    """

    @staticmethod
    def calculate_ism(weights, sentiment_score, technical_score, prediction_score, sharpe_ratio, max_dd):
        """
        Calculate the Investment Surety Metric (ISM).

        Parameters:
            weights (dict): Weights for each ISM component.
            sentiment_score (float): Sentiment score (-1 to 1).
            technical_score (float): Technical indicator score (0 to 1).
            prediction_score (float): ML prediction confidence (0 to 1).
            sharpe_ratio (float): Portfolio Sharpe ratio (>0 is favorable).
            max_dd (float): Maximum drawdown (negative, 0 represents no drawdown).

        Returns:
            float: Normalized ISM score (0 to 1).
        """
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate ISM as a weighted sum
        ism_score = (
            normalized_weights["sentiment"] * sentiment_score +
            normalized_weights["technical"] * technical_score +
            normalized_weights["ml"] * prediction_score +
            normalized_weights["sharpe"] * sharpe_ratio -
            normalized_weights["max_dd"] * abs(max_dd)  # Penalize Max DD
        )

        # Normalize ISM to a range of 0 to 1
        return max(0, min(1, (ism_score + 1) / 2))

    @staticmethod
    def optimize_weights(data, initial_weights):
        """
        Optimize ISM weights using nonlinear optimization to maximize returns.

        Parameters:
            data (dict): Dictionary containing ISM component values (sentiment, technical, etc.).
            initial_weights (dict): Initial weights for ISM components.

        Returns:
            dict: Optimized weights for ISM components.
        """
        def objective(weights):
            # Unpack weights
            w_sentiment, w_technical, w_ml, w_sharpe, w_max_dd = weights
            weights_dict = {
                "sentiment": w_sentiment,
                "technical": w_technical,
                "ml": w_ml,
                "sharpe": w_sharpe,
                "max_dd": w_max_dd
            }
            # Recalculate ISM
            ism = InvestmentSuretyMetric.calculate_ism(
                weights_dict,
                data["sentiment_score"],
                data["technical_score"],
                data["prediction_score"],
                data["sharpe_ratio"],
                data["max_dd"]
            )
            # Objective: maximize ISM
            return -ism  # Scipy minimizes, so we negate ISM

        # Initial weights
        initial_weights_list = [
            initial_weights["sentiment"],
            initial_weights["technical"],
            initial_weights["ml"],
            initial_weights["sharpe"],
            initial_weights["max_dd"]
        ]

        # Constraints: Weights must sum to 1
        constraints = {"type": "eq", "fun": lambda weights: sum(weights) - 1}

        # Bounds: Weights must be between 0 and 1
        bounds = [(0, 1) for _ in range(5)]

        # Optimize weights
        result = minimize(objective, initial_weights_list, bounds=bounds, constraints=constraints)

        # Extract optimized weights
        optimized_weights = result.x
        return {
            "sentiment": optimized_weights[0],
            "technical": optimized_weights[1],
            "ml": optimized_weights[2],
            "sharpe": optimized_weights[3],
            "max_dd": optimized_weights[4]
        }
        
# BLOCK 7: PORTFOLIO OPTIMIZATION MODULE

def optimize_portfolio(returns):
    """
    Optimize portfolio allocation to maximize the Sharpe ratio using Modern Portfolio Theory.

    Parameters:
        returns (pd.DataFrame): DataFrame of historical daily returns for portfolio assets.

    Returns:
        dict: Optimal weights for each asset in the portfolio.
        float: Maximum Sharpe ratio achieved.
    """
    # Number of assets
    n_assets = returns.shape[1]

    # Mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Objective function: Negative Sharpe ratio (to maximize Sharpe ratio)
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
        return -sharpe_ratio

    # Constraints: Weights must sum to 1
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    # Bounds: Each weight must be between 0 and 1
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess: Equal allocation
    initial_weights = np.ones(n_assets) / n_assets

    # Optimize portfolio
    result = minimize(negative_sharpe, initial_weights, bounds=bounds, constraints=constraints)

    # Extract optimal weights and maximum Sharpe ratio
    optimal_weights = result.x
    max_sharpe_ratio = -result.fun

    return dict(zip(returns.columns, optimal_weights)), max_sharpe_ratio

# BLOCK 8: TRADING BOT MODULE

class TradingBot:
    """
    A trading bot that makes decisions based on ISM and executes trades for crypto and stocks.
    """

    def __init__(self, initial_balance=1000, risk_tolerance=0.05, allocation={"crypto": 0.5, "stocks": 0.5}):
        """
        Initialize the trading bot.

        Parameters:
            initial_balance (float): Starting balance for the portfolio.
            risk_tolerance (float): Maximum allowable drawdown (as a proportion of balance).
            allocation (dict): Allocation of funds between crypto and stocks.
        """
        self.balance = initial_balance
        self.crypto_balance = initial_balance * allocation["crypto"]
        self.stock_balance = initial_balance * allocation["stocks"]
        self.risk_tolerance = risk_tolerance
        self.crypto_holdings = {}  # Track cryptocurrency holdings
        self.stock_holdings = {}  # Track stock holdings
        self.trade_history = []  # Log of all trades executed

    def execute_crypto_trade(self, symbol, price, action, amount=None):
        """
        Execute a trade for cryptocurrencies.

        Parameters:
            symbol (str): Cryptocurrency symbol (e.g., 'bitcoin').
            price (float): Current price of the cryptocurrency.
            action (str): Trade action ('buy', 'sell', or 'hold').
            amount (float): Amount to trade (optional, defaults to max possible based on balance).

        Returns:
            dict: Details of the executed trade.
        """
        trade = {"symbol": symbol, "price": price, "action": action, "timestamp": pd.Timestamp.now()}

        if action == "buy":
            # Calculate amount to buy
            amount_to_buy = amount if amount else self.crypto_balance / price
            if self.crypto_balance >= amount_to_buy * price:
                self.crypto_balance -= amount_to_buy * price
                self.crypto_holdings[symbol] = self.crypto_holdings.get(symbol, 0) + amount_to_buy
                trade["amount"] = amount_to_buy
                logging.info(f"Bought {amount_to_buy:.4f} {symbol} at {price:.2f}")
            else:
                trade["amount"] = 0
                logging.warning(f"Insufficient balance to buy {symbol}")
        elif action == "sell":
            # Calculate amount to sell
            amount_to_sell = amount if amount else self.crypto_holdings.get(symbol, 0)
            if self.crypto_holdings.get(symbol, 0) >= amount_to_sell:
                self.crypto_balance += amount_to_sell * price
                self.crypto_holdings[symbol] -= amount_to_sell
                trade["amount"] = amount_to_sell
                logging.info(f"Sold {amount_to_sell:.4f} {symbol} at {price:.2f}")
            else:
                trade["amount"] = 0
                logging.warning(f"Insufficient holdings to sell {symbol}")
        else:
            trade["amount"] = 0  # Hold action
            logging.info(f"Held {symbol} at {price:.2f}")

        self.trade_history.append(trade)
        return trade

    def execute_stock_trade(self, symbol, price, action, amount=None, alpaca_api=None):
        """
        Execute a trade for stocks using Alpaca's paper trading API.

        Parameters:
            symbol (str): Stock symbol (e.g., 'AAPL').
            price (float): Current price of the stock.
            action (str): Trade action ('buy', 'sell', or 'hold').
            amount (float): Number of shares to trade (optional, defaults to max possible based on balance).
            alpaca_api (AlpacaAPI): Instance of the Alpaca API wrapper.

        Returns:
            dict: Details of the executed trade.
        """
        trade = {"symbol": symbol, "price": price, "action": action, "timestamp": pd.Timestamp.now()}

        if action == "buy":
            # Calculate amount to buy
            amount_to_buy = amount if amount else int(self.stock_balance / price)
            if self.stock_balance >= amount_to_buy * price:
                self.stock_balance -= amount_to_buy * price
                self.stock_holdings[symbol] = self.stock_holdings.get(symbol, 0) + amount_to_buy
                trade["amount"] = amount_to_buy

                # Alpaca API call for paper trading
                if alpaca_api:
                    alpaca_api.place_order(symbol, qty=amount_to_buy, side="buy")
                logging.info(f"Bought {amount_to_buy} shares of {symbol} at {price:.2f}")
            else:
                trade["amount"] = 0
                logging.warning(f"Insufficient balance to buy {symbol}")
        elif action == "sell":
            # Calculate amount to sell
            amount_to_sell = amount if amount else self.stock_holdings.get(symbol, 0)
            if self.stock_holdings.get(symbol, 0) >= amount_to_sell:
                self.stock_balance += amount_to_sell * price
                self.stock_holdings[symbol] -= amount_to_sell
                trade["amount"] = amount_to_sell

                # Alpaca API call for paper trading
                if alpaca_api:
                    alpaca_api.place_order(symbol, qty=amount_to_sell, side="sell")
                logging.info(f"Sold {amount_to_sell} shares of {symbol} at {price:.2f}")
            else:
                trade["amount"] = 0
                logging.warning(f"Insufficient holdings to sell {symbol}")
        else:
            trade["amount"] = 0  # Hold action
            logging.info(f"Held {symbol} at {price:.2f}")

        self.trade_history.append(trade)
        return trade

    def evaluate_trade_signal(self, ism, threshold=0.5):
        """
        Evaluate the ISM and decide on a trade action.

        Parameters:
            ism (float): Investment Surety Metric value.
            threshold (float): Threshold for buy/sell decisions.

        Returns:
            str: 'buy', 'sell', or 'hold'.
        """
        if ism > threshold:
            return "buy"
        elif ism < -threshold:
            return "sell"
        else:
            return "hold"
        
# BLOCK 9: DASH GUI MODULE

def create_dashboard(trading_bot, results, ism_weights, optimized_weights, developer_key=None):
    """
    Create an interactive Dash dashboard for AmbiVest.

    Parameters:
        trading_bot (TradingBot): Instance of the TradingBot class for displaying trade history.
        results (dict): Results dictionary containing portfolio values, ISM scores, and metrics.
        ism_weights (dict): Initial ISM weights for user interaction.
        optimized_weights (dict): Optimized ISM weights from the optimization module.
        developer_key (str): Optional developer key to enable debugging mode.

    Returns:
        Dash: The Dash app instance.
    """
    app = Dash(__name__)

    # Extract portfolio values and metrics
    portfolio_values = results["portfolio_values"]
    trade_history = trading_bot.trade_history
    sentiment_score = results["sentiment_score"]
    technical_score = results["technical_score"]
    prediction_score = results["prediction_score"]
    sharpe_ratio = results["sharpe_ratio"]
    max_dd = results["max_dd"]
    ism = results["ism"]

    # Layout for the dashboard
    app.layout = html.Div([
        html.H1("AmbiVest: Crypto and Stock Trading Dashboard", style={"textAlign": "center"}),

        # User Asset Selection Section
        html.Div([
            html.H2("Select Assets"),
            html.Label("Cryptocurrencies"),
            dcc.Dropdown(
                id="crypto-selection",
                options=[{"label": symbol.capitalize(), "value": symbol} for symbol in CRYPTO_SYMBOLS],
                value=CRYPTO_SYMBOLS,
                multi=True
            ),
            html.Label("Stocks"),
            dcc.Dropdown(
                id="stock-selection",
                options=[{"label": symbol, "value": symbol} for symbol in STOCK_SYMBOLS],
                value=STOCK_SYMBOLS,
                multi=True
            ),
        ], style={"padding": "20px"}),

        # Portfolio Performance Section
        html.Div([
            html.H2("Portfolio Performance"),
            dcc.Graph(
                id="portfolio-performance",
                figure={
                    "data": [
                        go.Scatter(
                            x=portfolio_values["timestamp"],
                            y=portfolio_values["Portfolio Value"],
                            mode="lines",
                            name="ISM Strategy Portfolio",
                            line=dict(color="blue"),
                        ),
                        go.Scatter(
                            x=portfolio_values["timestamp"],
                            y=portfolio_values["Buy and Hold Value"],
                            mode="lines",
                            name="Buy and Hold Portfolio",
                            line=dict(color="orange", dash="dash"),
                        ),
                    ],
                    "layout": go.Layout(
                        title="Portfolio Performance Over Time",
                        xaxis={"title": "Date"},
                        yaxis={"title": "Portfolio Value ($)"},
                        hovermode="closest",
                    ),
                },
            )
        ]),

        # ISM Breakdown Section
        html.Div([
            html.H2("ISM Breakdown"),
            dcc.Graph(
                id="ism-breakdown",
                figure={
                    "data": [
                        go.Pie(
                            labels=["Sentiment", "Technical", "ML Prediction", "Sharpe Ratio", "Max Drawdown"],
                            values=[
                                sentiment_score * ism_weights["sentiment"],
                                technical_score * ism_weights["technical"],
                                prediction_score * ism_weights["ml"],
                                sharpe_ratio * ism_weights["sharpe"],
                                max_dd * ism_weights["max_dd"]
                            ],
                            hole=0.4,
                        )
                    ],
                    "layout": go.Layout(
                        title="Investment Surety Metric (ISM) Breakdown",
                    ),
                },
            )
        ]),

        # Interactive Weight Optimization Section
        html.Div([
            html.H2("Optimize ISM Weights"),
            html.Label("Sentiment Weight"),
            dcc.Slider(
                id="sentiment-weight",
                min=0,
                max=1,
                step=0.01,
                value=ism_weights["sentiment"],
                marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11)},
            ),
            html.Label("Technical Weight"),
            dcc.Slider(
                id="technical-weight",
                min=0,
                max=1,
                step=0.01,
                value=ism_weights["technical"],
                marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11)},
            ),
            html.Label("ML Prediction Weight"),
            dcc.Slider(
                id="ml-weight",
                min=0,
                max=1,
                step=0.01,
                value=ism_weights["ml"],
                marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11)},
            ),
            html.Label("Sharpe Ratio Weight"),
            dcc.Slider(
                id="sharpe-weight",
                min=0,
                max=1,
                step=0.01,
                value=ism_weights["sharpe"],
                marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11)},
            ),
            html.Label("Max Drawdown Weight"),
            dcc.Slider(
                id="max-dd-weight",
                min=0,
                max=1,
                step=0.01,
                value=ism_weights["max_dd"],
                marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11)},
            ),
            html.Button("Optimize Weights", id="optimize-button", n_clicks=0),
            html.Div(id="optimized-weights-output", style={"paddingTop": "20px"}),
        ]),

        # Trade History Section
        html.Div([
            html.H2("Trade History"),
            html.Table(
                # Create table header
                [html.Tr([html.Th(col) for col in ["Symbol", "Action", "Amount", "Price", "Timestamp"]])] +
                # Create table rows
                [html.Tr([html.Td(trade[col]) for col in ["symbol", "action", "amount", "price", "timestamp"]])
                 for trade in trade_history],
                style={"width": "100%", "border": "1px solid black", "padding": "10px"},
            )
        ]),

        # Debugging Mode Section (Visible only with Developer Key)
        html.Div(
            id="debug-output",
            style={"display": "none"} if developer_key != DEVELOPER_KEY else {"display": "block"},
            children=[
                html.H2("Developer Debugging Mode"),
                html.Pre(f"Backtesting Results:\n{results}", style={"whiteSpace": "pre-wrap", "border": "1px solid grey"})
            ]
        )
    ])

    # Callbacks for Weight Optimization
    @app.callback(
        Output("optimized-weights-output", "children"),
        Input("optimize-button", "n_clicks"),
        [
            Input("sentiment-weight", "value"),
            Input("technical-weight", "value"),
            Input("ml-weight", "value"),
            Input("sharpe-weight", "value"),
            Input("max-dd-weight", "value"),
        ]
    )
    def optimize_weights_callback(n_clicks, sentiment, technical, ml, sharpe, max_dd):
        if n_clicks > 0:
            # Perform optimization
            initial_weights = {
                "sentiment": sentiment,
                "technical": technical,
                "ml": ml,
                "sharpe": sharpe,
                "max_dd": max_dd,
            }
            optimized = InvestmentSuretyMetric.optimize_weights(results, initial_weights)
            return f"Optimized Weights: {optimized}"
        return "Click 'Optimize Weights' to find the best weights."

    return app

# BLOCK 10: EXECUTION PIPELINE WITH LSTM AND FALLBACK INTEGRATION

def execute_ambivest_pipeline_with_refinement(crypto_symbols, stock_symbols, start_date, end_date, developer_key=None):
    """
    Execute the full AmbiVest trading pipeline with comparative analysis and fallback support.

    Parameters:
        crypto_symbols (list): List of cryptocurrencies to include (e.g., ['bitcoin', 'ethereum']).
        stock_symbols (list): List of stocks to include (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date for fetching historical stock data (YYYY-MM-DD).
        end_date (str): End date for fetching historical stock data (YYYY-MM-DD).
        developer_key (str): Optional developer key to enable debugging mode.

    Returns:
        Dash: The Dash app instance for visualization.
    """
    # Initialize API clients
    coin_gecko = CoinGeckoAPI()
    yahoo_finance = YahooFinanceAPI()
    twitter_api = RapidAPITwitter(api_key=RAPIDAPI_KEY)

    # Initialize Trading Bot
    trading_bot = TradingBot(initial_balance=INITIAL_BALANCE)

    # Initialize ISM weights and trading threshold
    ism_weights = {"sentiment": 0.25, "technical": 0.25, "ml": 0.25, "sharpe": 0.15, "max_dd": 0.1}
    trading_threshold = SENTIMENT_THRESHOLD

    # Historical data storage
    historical_data = {}

    # Step 1: Fetch Historical Data
    logging.info("Fetching historical data for crypto and stocks...")
    crypto_data = {}
    stock_data = {}

    for symbol in crypto_symbols:
        crypto_data[symbol] = coin_gecko.get_historical_data(symbol, days=30)
        historical_data[symbol] = crypto_data[symbol]

    for symbol in stock_symbols:
        stock_data[symbol] = yahoo_finance.get_historical_data(symbol, start_date, end_date)
        historical_data[symbol] = stock_data[symbol]

    # Step 2: Perform Sentiment Analysis
    logging.info("Performing sentiment analysis...")
    sentiment_scores = {}
    for symbol in crypto_symbols + stock_symbols:
        sentiment_scores[symbol] = fetch_and_analyze_sentiment(twitter_api, query=symbol, count=10)

    # Step 3: Calculate Technical Indicators
    logging.info("Calculating technical indicators...")
    for symbol, data in crypto_data.items():
        crypto_data[symbol] = calculate_technical_indicators(data, lookback_window=LOOKBACK_WINDOW)
    for symbol, data in stock_data.items():
        stock_data[symbol] = calculate_technical_indicators(data, lookback_window=LOOKBACK_WINDOW)

    # Step 4: Train ML Models and Predict Prices
    logging.info("Training LSTM model and predicting prices...")
    lstm_model = LSTMModel(lookback=30)  # Initialize the LSTM model
    prediction_scores = {}  # Store predictions for each asset

    for symbol, data in {**crypto_data, **stock_data}.items():
        try:
            # Attempt to train and predict with LSTM
            lstm_model.train(data)
            prediction_scores[symbol] = lstm_model.predict(data)
        except Exception as e:
            logging.warning(f"LSTM model failed for {symbol}. Falling back to Linear Regression. Error: {e}")
            # Fallback to Linear Regression if LSTM fails
            ml_model = MLModel()
            ml_model.train(data)
            prediction_scores[symbol] = ml_model.predict(data["price"].iloc[-1])

    # Step 5: Calculate Sharpe Ratio and Max Drawdown
    logging.info("Calculating Sharpe ratio and Max Drawdown...")
    sharpe_ratios = {}
    max_drawdowns = {}

    for symbol, data in {**crypto_data, **stock_data}.items():
        # Calculate daily returns
        data["Daily Returns"] = data["price"].pct_change()
        sharpe_ratios[symbol] = (
            (data["Daily Returns"].mean() - RISK_FREE_RATE) / data["Daily Returns"].std()
        ) * np.sqrt(252)  # Assuming 252 trading days

        # Calculate Max Drawdown
        cumulative_returns = (1 + data["Daily Returns"]).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdowns[symbol] = drawdown.min()

    # Step 6: Calculate ISM for Each Asset
    logging.info("Calculating ISM for each asset...")
    isms = {}
    for symbol in crypto_symbols + stock_symbols:
        isms[symbol] = InvestmentSuretyMetric.calculate_ism(
            ism_weights,
            sentiment_scores[symbol],
            crypto_data[symbol]["SMA"].iloc[-1] / crypto_data[symbol]["price"].iloc[-1] if symbol in crypto_data else stock_data[symbol]["SMA"].iloc[-1] / stock_data[symbol]["price"].iloc[-1],
            prediction_scores[symbol],
            sharpe_ratios[symbol],
            max_drawdowns[symbol]
        )

    # Step 7: Execute Trades Using Trading Bot
    logging.info("Executing trades based on ISM...")
    for symbol in crypto_symbols + stock_symbols:
        action = trading_bot.evaluate_trade_signal(isms[symbol], threshold=trading_threshold)
        price = crypto_data[symbol]["price"].iloc[-1] if symbol in crypto_data else stock_data[symbol]["price"].iloc[-1]

        if symbol in crypto_data:
            trading_bot.execute_crypto_trade(symbol, price, action)
        else:
            trading_bot.execute_stock_trade(symbol, price, action)

    # Step 8: Compare AmbiVest to Benchmark Strategies
    logging.info("Performing comparative analysis...")
    ambivest_results = {
        "portfolio_values": pd.DataFrame({
            "timestamp": pd.date_range(start=start_date, end=end_date, periods=30),
            "Portfolio Value": [trading_bot.balance] * 30,  # Simulated data
            "Buy and Hold Value": [INITIAL_BALANCE] * 30  # Simulated data
        }),
        "sharpe_ratio": np.mean(list(sharpe_ratios.values())),
        "max_dd": np.mean(list(max_drawdowns.values()))
    }
    benchmark_results = compare_strategies(ambivest_results, historical_data, crypto_symbols, stock_symbols)

    # Step 9: Refine Strategy Based on Comparisons
    logging.info("Refining strategy based on benchmark comparisons...")
    ism_weights, trading_threshold = refine_ambivest_strategy(benchmark_results, ism_weights, trading_threshold)

    # Step 10: Launch Dash GUI
    logging.info("Launching Dash GUI...")
    app = create_dashboard(trading_bot, ambivest_results, ism_weights, {}, developer_key)
    app.run_server(debug=True, use_reloader=False)
    
# BLOCK 11: TESTING, REFINEMENT, AND COMPARATIVE ANALYSIS

def compare_strategies(ambivest_results, historical_data, crypto_symbols, stock_symbols):
    """
    Compare AmbiVest's performance against existing strategies.

    Parameters:
        ambivest_results (dict): Results dictionary containing AmbiVest portfolio metrics.
        historical_data (dict): Dictionary of historical price data for crypto and stocks.
        crypto_symbols (list): List of cryptocurrencies used in the analysis.
        stock_symbols (list): List of stocks used in the analysis.

    Returns:
        dict: Performance metrics for AmbiVest and benchmark strategies.
    """
    # Extract AmbiVest metrics
    ambivest_portfolio_values = ambivest_results["portfolio_values"]
    ambivest_cumulative_return = (ambivest_portfolio_values["Portfolio Value"].iloc[-1] - INITIAL_BALANCE) / INITIAL_BALANCE
    ambivest_sharpe_ratio = ambivest_results["sharpe_ratio"]
    ambivest_max_dd = ambivest_results["max_dd"]

    # Initialize benchmark results
    benchmark_results = {
        "AmbiVest": {
            "Cumulative Return (%)": ambivest_cumulative_return * 100,
            "Sharpe Ratio": ambivest_sharpe_ratio,
            "Max Drawdown (%)": ambivest_max_dd * 100,
        },
        "Buy and Hold": {},
        "Risk Parity": {},
        "MPT": {},
    }

    # 1. Buy-and-Hold Strategy
    logging.info("Simulating Buy-and-Hold strategy...")
    buy_and_hold_values = []
    for symbol in crypto_symbols + stock_symbols:
        data = historical_data[symbol]
        initial_price = data["price"].iloc[0]
        buy_and_hold_values.append((INITIAL_BALANCE / len(historical_data)) / initial_price * data["price"].iloc[-1])

    buy_and_hold_portfolio_value = sum(buy_and_hold_values)
    buy_and_hold_cumulative_return = (buy_and_hold_portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE
    buy_and_hold_daily_returns = pd.concat([data["price"].pct_change() for data in historical_data.values()], axis=1).mean(axis=1)
    buy_and_hold_sharpe_ratio = (
        (buy_and_hold_daily_returns.mean() - RISK_FREE_RATE) / buy_and_hold_daily_returns.std()
    ) * np.sqrt(252)
    benchmark_results["Buy and Hold"] = {
        "Cumulative Return (%)": buy_and_hold_cumulative_return * 100,
        "Sharpe Ratio": buy_and_hold_sharpe_ratio,
        "Max Drawdown (%)": 0,  # Placeholder: Max Drawdown for Buy-and-Hold can be added
    }

    # 2. Risk Parity Strategy
    logging.info("Simulating Risk Parity strategy...")
    # Risk Parity allocates capital to equalize risk contributions
    risk_parity_weights = 1 / pd.concat([data["price"].pct_change().std() for data in historical_data.values()], axis=0)
    risk_parity_weights /= risk_parity_weights.sum()
    risk_parity_values = [
        risk_parity_weights[i] * (INITIAL_BALANCE / len(historical_data)) / data["price"].iloc[0] * data["price"].iloc[-1]
        for i, data in enumerate(historical_data.values())
    ]
    risk_parity_portfolio_value = sum(risk_parity_values)
    risk_parity_cumulative_return = (risk_parity_portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE
    benchmark_results["Risk Parity"] = {
        "Cumulative Return (%)": risk_parity_cumulative_return * 100,
        "Sharpe Ratio": 0,  # Placeholder: Sharpe Ratio for Risk Parity can be added
        "Max Drawdown (%)": 0,  # Placeholder: Max Drawdown for Risk Parity can be added
    }

    # 3. Modern Portfolio Theory (MPT)
    logging.info("Simulating MPT strategy...")
    historical_returns = pd.concat([data["price"].pct_change() for data in historical_data.values()], axis=1)
    mpt_weights, mpt_sharpe_ratio = optimize_portfolio(historical_returns)
    mpt_values = [
        mpt_weights[i] * (INITIAL_BALANCE / len(historical_data)) / data["price"].iloc[0] * data["price"].iloc[-1]
        for i, data in enumerate(historical_data.values())
    ]
    mpt_portfolio_value = sum(mpt_values)
    mpt_cumulative_return = (mpt_portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE
    benchmark_results["MPT"] = {
        "Cumulative Return (%)": mpt_cumulative_return * 100,
        "Sharpe Ratio": mpt_sharpe_ratio,
        "Max Drawdown (%)": 0,  # Placeholder: Max Drawdown for MPT can be added
    }

    return benchmark_results

def refine_ambivest_strategy(benchmark_results, ism_weights, trading_threshold):
    """
    Dynamically refine AmbiVest strategy based on benchmark comparisons.

    Parameters:
        benchmark_results (dict): Performance metrics for AmbiVest and benchmark strategies.
        ism_weights (dict): Current ISM weights.
        trading_threshold (float): Current threshold for ISM-driven trades.

    Returns:
        dict: Refined ISM weights.
        float: Refined trading threshold.
    """
    # Extract performance metrics for comparison
    ambivest_return = benchmark_results["AmbiVest"]["Cumulative Return (%)"]
    buy_and_hold_return = benchmark_results["Buy and Hold"]["Cumulative Return (%)"]
    mpt_return = benchmark_results["MPT"]["Cumulative Return (%)"]

    # Refine ISM weights based on performance gaps
    if ambivest_return < buy_and_hold_return:
        logging.info("AmbiVest underperformed Buy-and-Hold. Adjusting ISM weights...")
        ism_weights["sentiment"] += LEARNING_RATE
        ism_weights["technical"] -= LEARNING_RATE / 2
    if ambivest_return < mpt_return:
        logging.info("AmbiVest underperformed MPT. Adjusting ISM weights...")
        ism_weights["sharpe"] += LEARNING_RATE
        ism_weights["max_dd"] -= LEARNING_RATE / 2

    # Normalize weights to ensure they sum to 1
    total_weight = sum(ism_weights.values())
    ism_weights = {k: v / total_weight for k, v in ism_weights.items()}

    # Refine trading threshold based on performance
    if ambivest_return < buy_and_hold_return:
        trading_threshold += LEARNING_RATE * 0.1  # Increase threshold to reduce false positives
    elif ambivest_return > mpt_return:
        trading_threshold -= LEARNING_RATE * 0.1  # Decrease threshold to capture more opportunities

    return ism_weights, trading_threshold

# BLOCK 12: FULL EXECUTION PIPELINE WITH COMPARATIVE ANALYSIS AND REFINEMENT

def execute_ambivest_pipeline_with_refinement(crypto_symbols, stock_symbols, start_date, end_date, developer_key=None):
    """
    Execute the full AmbiVest trading pipeline with comparative analysis and refinement.

    Parameters:
        crypto_symbols (list): List of cryptocurrencies to include (e.g., ['bitcoin', 'ethereum']).
        stock_symbols (list): List of stocks to include (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date for fetching historical stock data (YYYY-MM-DD).
        end_date (str): End date for fetching historical stock data (YYYY-MM-DD).
        developer_key (str): Optional developer key to enable debugging mode.

    Returns:
        Dash: The Dash app instance for visualization.
    """
    # Initialize API clients
    coin_gecko = CoinGeckoAPI()
    yahoo_finance = YahooFinanceAPI()
    twitter_api = RapidAPITwitter(api_key=RAPIDAPI_KEY, host=RAPIDAPI_HOST)

    # Initialize Trading Bot
    trading_bot = TradingBot(initial_balance=INITIAL_BALANCE)

    # Initialize ISM weights and trading threshold
    ism_weights = {"sentiment": 0.25, "technical": 0.25, "ml": 0.25, "sharpe": 0.15, "max_dd": 0.1}
    trading_threshold = SENTIMENT_THRESHOLD

    # Historical data storage
    historical_data = {}

    # Step 1: Fetch Historical Data
    logging.info("Fetching historical data for crypto and stocks...")
    crypto_data = {}
    stock_data = {}

    for symbol in crypto_symbols:
        crypto_data[symbol] = coin_gecko.get_historical_data(symbol, days=30)
        historical_data[symbol] = crypto_data[symbol]

    for symbol in stock_symbols:
        stock_data[symbol] = yahoo_finance.get_historical_data(symbol, start_date, end_date)
        historical_data[symbol] = stock_data[symbol]

    # Step 2: Perform Sentiment Analysis
    logging.info("Performing sentiment analysis...")
    sentiment_scores = {}
    for symbol in crypto_symbols + stock_symbols:
        sentiment_scores[symbol] = fetch_and_analyze_sentiment(twitter_api, symbol)

    # Step 3: Calculate Technical Indicators
    logging.info("Calculating technical indicators...")
    for symbol, data in crypto_data.items():
        crypto_data[symbol] = calculate_technical_indicators(data, lookback_window=LOOKBACK_WINDOW)
    for symbol, data in stock_data.items():
        stock_data[symbol] = calculate_technical_indicators(data, lookback_window=LOOKBACK_WINDOW)

    # Step 4: Train ML Model and Predict Prices
    logging.info("Training ML models and predicting prices...")
    ml_model = MLModel()
    prediction_scores = {}
    for symbol, data in {**crypto_data, **stock_data}.items():
        ml_model.train(data)
        prediction_scores[symbol] = ml_model.predict(data["price"].iloc[-1])

    # Step 5: Calculate Sharpe Ratio and Max Drawdown
    logging.info("Calculating Sharpe ratio and Max Drawdown...")
    sharpe_ratios = {}
    max_drawdowns = {}

    for symbol, data in {**crypto_data, **stock_data}.items():
        # Calculate daily returns
        data["Daily Returns"] = data["price"].pct_change()
        sharpe_ratios[symbol] = (
            (data["Daily Returns"].mean() - RISK_FREE_RATE) / data["Daily Returns"].std()
        ) * np.sqrt(252)  # Assuming 252 trading days

        # Calculate Max Drawdown
        cumulative_returns = (1 + data["Daily Returns"]).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdowns[symbol] = drawdown.min()

    # Step 6: Calculate ISM for Each Asset
    logging.info("Calculating ISM for each asset...")
    isms = {}
    for symbol in crypto_symbols + stock_symbols:
        isms[symbol] = InvestmentSuretyMetric.calculate_ism(
            ism_weights,
            sentiment_scores[symbol],
            crypto_data[symbol]["SMA"].iloc[-1] / crypto_data[symbol]["price"].iloc[-1] if symbol in crypto_data else stock_data[symbol]["SMA"].iloc[-1] / stock_data[symbol]["price"].iloc[-1],
            prediction_scores[symbol],
            sharpe_ratios[symbol],
            max_drawdowns[symbol]
        )

    # Step 7: Execute Trades Using Trading Bot
    logging.info("Executing trades based on ISM...")
    for symbol in crypto_symbols + stock_symbols:
        action = trading_bot.evaluate_trade_signal(isms[symbol], threshold=trading_threshold)
        price = crypto_data[symbol]["price"].iloc[-1] if symbol in crypto_data else stock_data[symbol]["price"].iloc[-1]

        if symbol in crypto_data:
            trading_bot.execute_crypto_trade(symbol, price, action)
        else:
            trading_bot.execute_stock_trade(symbol, price, action)

    # Step 8: Compare AmbiVest to Benchmark Strategies
    logging.info("Performing comparative analysis...")
    ambivest_results = {
        "portfolio_values": pd.DataFrame({
            "timestamp": pd.date_range(start=start_date, end=end_date, periods=30),
            "Portfolio Value": [trading_bot.balance] * 30,  # Simulated data
            "Buy and Hold Value": [INITIAL_BALANCE] * 30  # Simulated data
        }),
        "sharpe_ratio": np.mean(list(sharpe_ratios.values())),
        "max_dd": np.mean(list(max_drawdowns.values()))
    }
    benchmark_results = compare_strategies(ambivest_results, historical_data, crypto_symbols, stock_symbols)

    # Step 9: Refine Strategy Based on Comparisons
    logging.info("Refining strategy based on benchmark comparisons...")
    ism_weights, trading_threshold = refine_ambivest_strategy(benchmark_results, ism_weights, trading_threshold)

    # Step 10: Launch Dash GUI
    logging.info("Launching Dash GUI...")
    app = create_dashboard(trading_bot, ambivest_results, ism_weights, {}, developer_key)
    app.run_server(debug=True, use_reloader=False)
    
# BLOCK 13: TESTING THE PIPELINE (UPDATED FOR TWITTER AND FALLBACK VALIDATION)

def run_tests():
    """
    Run a comprehensive test suite for AmbiVest.

    Returns:
        dict: Results of all test cases.
    """
    test_results = {}

    # Test 1: Core Pipeline Functionality
    try:
        logging.info("Testing core pipeline functionality...")
        core_test_result = test_ambivest_pipeline()
        test_results["Core Pipeline"] = core_test_result
    except Exception as e:
        logging.error(f"Core pipeline test failed: {e}")
        test_results["Core Pipeline"] = {"status": "failed", "error": str(e)}

    # Test 2: Twitter API Wrapper
    try:
        logging.info("Testing Twitter API wrapper...")
        twitter_api = RapidAPITwitter(api_key=RAPIDAPI_KEY)
        tweets = twitter_api.fetch_tweets(query="bitcoin", count=5)
        assert isinstance(tweets, list), "Twitter API did not return a list."
        assert len(tweets) > 0, "Twitter API returned an empty list."
        test_results["Twitter API"] = {
            "status": "success",
            "fetched_tweets": tweets
        }
    except Exception as e:
        logging.error(f"Twitter API test failed: {e}")
        test_results["Twitter API"] = {"status": "failed", "error": str(e)}

    # Test 3: Sentiment Analysis Function
    try:
        logging.info("Testing sentiment analysis function...")
        twitter_api = RapidAPITwitter(api_key=RAPIDAPI_KEY)
        sentiment_score = fetch_and_analyze_sentiment(twitter_api, query="bitcoin", count=5)
        assert isinstance(sentiment_score, float), "Sentiment analysis did not return a float."
        test_results["Sentiment Analysis"] = {
            "status": "success",
            "sentiment_score": sentiment_score
        }
    except Exception as e:
        logging.error(f"Sentiment analysis test failed: {e}")
        test_results["Sentiment Analysis"] = {"status": "failed", "error": str(e)}

    # Test 4: Fallback to Linear Regression
    try:
        logging.info("Testing fallback to Linear Regression...")
        lstm_model = LSTMModel(lookback=30)

        # Simulate insufficient data for LSTM
        insufficient_data = pd.DataFrame({"price": np.linspace(100, 200, 10)})

        # Attempt to train the LSTM model
        try:
            lstm_model.train(insufficient_data)
            prediction = lstm_model.predict(insufficient_data)
            test_results["Fallback"] = {
                "status": "failed",
                "error": "LSTM did not fail on insufficient data when expected."
            }
        except Exception as lstm_error:
            # Fallback to Linear Regression if LSTM fails
            logging.info("LSTM failed as expected. Testing Linear Regression fallback...")
            ml_model = MLModel()
            ml_model.train(insufficient_data)
            prediction = ml_model.predict(insufficient_data["price"].iloc[-1])
            test_results["Fallback"] = {
                "status": "success",
                "prediction": prediction,
                "details": f"Linear Regression fallback succeeded. Predicted next price: {prediction:.2f}"
            }
    except Exception as e:
        logging.error(f"Fallback test failed: {e}")
        test_results["Fallback"] = {"status": "failed", "error": str(e)}

    # Test 5: Comparative Analysis
    try:
        logging.info("Testing comparative analysis...")
        dummy_crypto_data = {"bitcoin": pd.DataFrame({"price": np.linspace(100, 150, 100)})}
        dummy_stock_data = {"AAPL": pd.DataFrame({"price": np.linspace(200, 250, 100)})}
        historical_data = {**dummy_crypto_data, **dummy_stock_data}

        # Simulate AmbiVest results
        ambivest_results = {
            "portfolio_values": pd.DataFrame({
                "Portfolio Value": [1000, 1100, 1200],
                "Buy and Hold Value": [1000, 1050, 1100]
            }),
            "sharpe_ratio": 1.5,
            "max_dd": -0.2
        }

        # Perform comparative analysis
        benchmark_results = compare_strategies(
            ambivest_results, historical_data, ["bitcoin"], ["AAPL"]
        )
        test_results["Comparative Analysis"] = {
            "status": "success",
            "results": benchmark_results
        }
    except Exception as e:
        logging.error(f"Comparative analysis test failed: {e}")
        test_results["Comparative Analysis"] = {"status": "failed", "error": str(e)}

    # Test 6: Dynamic Refinement
    try:
        logging.info("Testing dynamic refinement...")
        benchmark_results = {
            "AmbiVest": {"Cumulative Return (%)": 5, "Sharpe Ratio": 1.2, "Max Drawdown (%)": -15},
            "Buy and Hold": {"Cumulative Return (%)": 7, "Sharpe Ratio": 1.1, "Max Drawdown (%)": -10},
            "MPT": {"Cumulative Return (%)": 10, "Sharpe Ratio": 1.5, "Max Drawdown (%)": -5}
        }

        ism_weights = {"sentiment": 0.25, "technical": 0.25, "ml": 0.25, "sharpe": 0.15, "max_dd": 0.1}
        trading_threshold = 0.5

        refined_weights, refined_threshold = refine_ambivest_strategy(
            benchmark_results, ism_weights, trading_threshold
        )

        test_results["Dynamic Refinement"] = {
            "status": "success",
            "refined_weights": refined_weights,
            "refined_threshold": refined_threshold
        }
    except Exception as e:
        logging.error(f"Dynamic refinement test failed: {e}")
        test_results["Dynamic Refinement"] = {"status": "failed", "error": str(e)}

    # Return all test results
    logging.info("All tests completed.")
    return test_results


# Run the test suite
if __name__ == "__main__":
    results = run_tests()
    for test, result in results.items():
        print(f"{test}: {result}")

# BLOCK 14: TESTING THE PIPELINE CORE FUNCTIONALITY

def test_ambivest_pipeline():
    """
    Test the core functionality of the AmbiVest pipeline.

    Returns:
        dict: Results of the core pipeline test.
    """
    try:
        logging.info("Starting core pipeline test...")

        # Define test parameters
        crypto_symbols = ["bitcoin"]
        stock_symbols = ["AAPL"]
        start_date = "2023-01-01"
        end_date = "2023-12-31"

        # Initialize Twitter API wrapper
        twitter_api = RapidAPITwitter(api_key=RAPIDAPI_KEY)

        # Fetch historical data for Bitcoin
        coin_gecko = CoinGeckoAPI()
        btc_data = coin_gecko.get_historical_data("bitcoin", days=30)
        assert not btc_data.empty, "Failed to fetch Bitcoin data."

        # Fetch historical stock data for AAPL
        yahoo_finance = YahooFinanceAPI()
        aapl_data = yahoo_finance.get_historical_data("AAPL", start_date, end_date)
        assert not aapl_data.empty, "Failed to fetch AAPL data."

        # Perform sentiment analysis
        sentiment_score = fetch_and_analyze_sentiment(twitter_api, query="bitcoin", count=10)
        assert isinstance(sentiment_score, float), "Sentiment analysis failed."

        # Calculate technical indicators
        btc_indicators = calculate_technical_indicators(btc_data)
        assert "SMA" in btc_indicators.columns, "Technical indicator calculation failed."

        # Train and predict with ML model
        ml_model = MLModel()
        ml_model.train(btc_indicators)
        prediction = ml_model.predict(btc_indicators["price"].iloc[-1])
        assert isinstance(prediction, float), "ML model prediction failed."

        logging.info("Core pipeline test passed successfully.")
        return {"status": "success"}
    except AssertionError as e:
        logging.error(f"Core pipeline test failed: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        logging.error(f"Unexpected error during pipeline test: {e}")
        return {"status": "failed", "error": str(e)}