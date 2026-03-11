# Imports
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
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from scipy.optimize import minimize
import yfinance as yf
import webbrowser

# Logging setup
LOG_FILE_NAME = "app_error_log.txt"  # Name of the log file

logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler(LOG_FILE_NAME),  # Write logs to a file
        logging.StreamHandler(),  # Print logs to the console
    ],
)

# Constants
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
API_KEY = "REDACTED_COINGECKO_KEY"  # Replace with your CoinGecko API key
RISK_FREE_RATE = 0.07365  # Risk-free rate for Sharpe ratio
INITIAL_BALANCE = 1000  # Initial portfolio balance


# Alpaca API credentials
ALPACA_API_KEY = "REDACTED_ALPACA_KEY_1"
ALPACA_SECRET_KEY = "REDACTED_ALPACA_SECRET_1"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading endpoint

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
            params = {
                "vs_currency": "usd",
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp()),
            }
            headers = {"accept": "application/json", "x-cg-demo-api-key": API_KEY}

            response = requests.get(url, params=params, headers=headers)
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
            # Simulate tweets with more variability
            tweets = [
                f"Positive news about {ticker}",
                f"Negative sentiment about {ticker}",
                f"Neutral discussion about {ticker}",
                f"{ticker} is doing great!",
                f"Concerns about {ticker}'s performance"
            ]
            sentiment_score = np.mean([TextBlob(tweet).sentiment.polarity for tweet in tweets])
            sentiments[ticker] = sentiment_score
        except Exception as e:
            logging.error(f"Error fetching sentiment for {ticker}: {e}")
            sentiments[ticker] = 0  # Default neutral sentiment
    return sentiments

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for stock/crypto data.

    Parameters:
        data (pd.DataFrame): DataFrame with historical price data.

    Returns:
        dict: Calculated indicators (SMA, EMA).
    """
    try:
        if data.empty or 'price' not in data.columns:
            logging.warning("Data is empty or missing 'price' column for technical indicators.")
            return {"SMA": 0, "EMA": 0}

        data['SMA'] = data['price'].rolling(window=20).mean()
        data['EMA'] = data['price'].ewm(span=20, adjust=False).mean()

        sma = data['SMA'].iloc[-1] if not data['SMA'].isna().all() else 0
        ema = data['EMA'].iloc[-1] if not data['EMA'].isna().all() else 0

        # Return the last valid SMA and EMA values
        return {"SMA": sma, "EMA": ema}
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return {"SMA": 0, "EMA": 0}
    
def train_deep_learning_model(data):
    """
    Train a deep learning model on stock/crypto price data.

    Parameters:
        data (pd.DataFrame): DataFrame with historical price data.

    Returns:
        tuple: Trained model and scaler.
    """
    try:
        if data.empty or 'price' not in data.columns or len(data) < 60:
            logging.warning("Insufficient data for training LSTM model.")
            return None, None

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
    
def calculate_investment_surety(sentiment_score, technical_score, ml_prediction, sharpe_ratio, weights):
    """
    Calculate the investment surety score using a weighted sum of various metrics.

    Parameters:
        sentiment_score (float): Sentiment analysis score (-1 to 1).
        technical_score (float): Technical indicator score (e.g., SMA or EMA).
        ml_prediction (float): ML model's predicted price or confidence score.
        sharpe_ratio (float): Sharpe ratio for the asset.
        weights (list): List of weights for each component in ISM.

    Returns:
        float: Investment Surety Metric (ISM) score.
    """
    return (
        weights[0] * sentiment_score +
        weights[1] * technical_score +
        weights[2] * ml_prediction +
        weights[3] * sharpe_ratio
    )

def optimize_portfolio(returns):
    """
    Optimize portfolio allocation to maximize the Sharpe ratio using Modern Portfolio Theory.

    Parameters:
        returns (pd.DataFrame): DataFrame of historical daily returns for portfolio assets.

    Returns:
        dict: Optimal weights for each asset in the portfolio.
        float: Maximum Sharpe ratio achieved.
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Objective function: Negative Sharpe ratio (to maximize Sharpe ratio)
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
        return -sharpe_ratio  # Use negative Sharpe ratio for minimization

    # Constraints: Weights must sum to 1
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    # Bounds: Each weight must be between 0 and 1
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess: Equal allocation
    initial_weights = np.ones(n_assets) / n_assets

    # Optimize portfolio
    result = minimize(negative_sharpe, initial_weights, bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    max_sharpe_ratio = -result.fun  # Convert back to positive Sharpe ratio
    return dict(zip(returns.columns, optimal_weights)), max_sharpe_ratio

def optimize_weights(data, returns):
    """
    Optimize the weights for the investment surety metric to maximize returns.

    Parameters:
        data (pd.DataFrame): DataFrame containing ISM components (sentiment, technical, etc.).
        returns (list): List of actual returns for the assets.

    Returns:
        list: Optimized weights for ISM.
    """
    def objective_function(weights):
        weights = weights / np.sum(weights)  # Normalize weights
        investment_scores = data.apply(
            lambda row: calculate_investment_surety(
                row['sentiment'], row['technical'], row['ml'], row['sharpe'], weights
            ), axis=1
        )
        # Calculate the negative correlation between investment scores and actual returns
        return -np.corrcoef(investment_scores, returns)[0, 1]

    initial_weights = [0.25, 0.25, 0.25, 0.25]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * 4

    result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    if result.success:
        return result.x / np.sum(result.x)  # Normalize the optimized weights
    else:
        logging.error("Weight optimization failed!")
        return initial_weights
def execute_alpaca_orders(optimized_weights, tickers, portfolio_balance):
    """
    Execute buy orders on Alpaca based on optimized portfolio weights.

    Parameters:
        optimized_weights (dict): Dictionary of weights for each asset.
        tickers (list): List of asset tickers.
        portfolio_balance (float): Total portfolio balance.

    Returns:
        dict: A dictionary containing order details for each ticker.
    """
    try:
        orders = {}
        for ticker, weight in optimized_weights.items():
            # Calculate the amount to invest in this ticker
            allocation = portfolio_balance * weight

            # Get the latest price for the ticker
            try:
                if ticker in tickers:
                    barset = alpaca.get_bars(ticker, "day", limit=1).df
                    if barset.empty:
                        logging.warning(f"No price data available for {ticker}. Skipping.")
                        continue

                    latest_price = barset['close'].iloc[-1]

                    # Calculate the number of shares to buy
                    quantity = int(allocation // latest_price)

                    if quantity > 0:
                        # Place a market order
                        order = alpaca.submit_order(
                            symbol=ticker,
                            qty=quantity,
                            side="buy",
                            type="market",
                            time_in_force="gtc"
                        )
                        orders[ticker] = {
                            "quantity": quantity,
                            "price": latest_price,
                            "allocation": allocation
                        }
                        logging.info(f"Order placed for {ticker}: {quantity} shares at ${latest_price:.2f}")
                    else:
                        logging.warning(f"Insufficient funds to place order for {ticker}. Skipping.")
            except Exception as e:
                logging.error(f"Error placing order for {ticker}: {e}")
                continue

        return orders
    except Exception as e:
        logging.error(f"Error executing Alpaca orders: {e}")
        return {}
    
def get_portfolio_performance():
    """
    Fetch the current portfolio performance from Alpaca.

    Returns:
        dict: A dictionary with portfolio performance metrics.
    """
    try:
        # Fetch account details
        account = alpaca.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        pnl = float(account.equity) - float(account.last_equity)

        performance = {
            "equity": equity,
            "buying_power": buying_power,
            "pnl": pnl
        }

        logging.info(f"Portfolio Performance: {performance}")
        return performance
    except Exception as e:
        logging.error(f"Error fetching portfolio performance: {e}")
        return {}

def buy_and_hold(returns, initial_balance=INITIAL_BALANCE):
    """
    Simulate a Buy-and-Hold strategy.

    Parameters:
        returns (pd.DataFrame): Daily returns for the portfolio assets.
        initial_balance (float): Initial portfolio balance.

    Returns:
        float: Final portfolio value.
    """
    try:
        n_assets = len(returns.columns)  # Number of assets
        allocation = initial_balance / n_assets  # Equal allocation to all assets
        final_values = []

        for col in returns.columns:
            cumulative_return = (1 + returns[col]).prod()
            final_values.append(cumulative_return * allocation)

        return sum(final_values)
    except Exception as e:
        logging.error(f"Error in Buy-and-Hold strategy: {e}")
        return 0
    
def risk_parity(returns, initial_balance=INITIAL_BALANCE):
    """
    Simulate a Risk Parity strategy.

    Parameters:
        returns (pd.DataFrame): Daily returns for the portfolio assets (each column represents an asset).
        initial_balance (float): Initial portfolio balance.

    Returns:
        float: Final portfolio value after risk parity strategy.
    """
    try:
        volatilities = returns.std()
        weights = 1 / volatilities
        weights /= weights.sum()

        cumulative_returns = (1 + returns).prod()
        portfolio_value = (weights * cumulative_returns * initial_balance).sum()
        return portfolio_value
    except Exception as e:
        logging.error(f"Error in Risk Parity calculation: {e}")
        return 0

def compare_strategies(returns):
    """
    Compare the performance of AmbiVest, Buy-and-Hold, and Risk Parity strategies.

    Parameters:
        returns (pd.DataFrame): DataFrame of daily returns for portfolio assets.

    Returns:
        dict: Performance metrics for each strategy.
    """
    try:
        # AmbiVest: Use optimized portfolio allocation
        ambivest_weights, max_sharpe_ratio = optimize_portfolio(returns)
        weighted_returns = returns.dot(list(ambivest_weights.values()))
        ambivest_portfolio_value = (1 + weighted_returns).cumprod()[-1] * INITIAL_BALANCE

        # Buy-and-Hold Strategy
        buy_hold_value = buy_and_hold(returns)

        # Risk Parity Strategy
        risk_parity_value = risk_parity(returns)

        return {
            "AmbiVest": {
                "Portfolio Value": ambivest_portfolio_value,
                "Sharpe Ratio": max_sharpe_ratio,
            },
            "Buy and Hold": {"Portfolio Value": buy_hold_value},
            "Risk Parity": {"Portfolio Value": risk_parity_value},
        }
    except Exception as e:
        logging.error(f"Error comparing strategies: {e}")
        return {}
    
def execute_trading_pipeline(crypto_tickers, stock_tickers, start_date, end_date):
    """
    Execute the full trading pipeline.

    Parameters:
        crypto_tickers (list): List of cryptocurrency tickers.
        stock_tickers (list): List of stock tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        dict: Results of the pipeline execution.
    """
    try:
        # Step 1: Fetch cryptocurrency data
        logging.info("Fetching cryptocurrency data...")
        crypto_data = {
            ticker: CoinGeckoAPI.fetch_crypto_data(ticker, start_date, end_date)
            for ticker in crypto_tickers
        }

        # Step 2: Fetch stock data
        logging.info("Fetching stock data...")
        stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, start_date, end_date)

        # Step 3: Combine daily returns
        combined_returns = pd.DataFrame()
        crypto_returns = pd.DataFrame()
        stock_returns = pd.DataFrame()

        for ticker, data in crypto_data.items():
            if not data.empty:
                daily_returns = data['price'].pct_change().dropna()
                combined_returns[ticker] = daily_returns
                crypto_returns[ticker] = daily_returns
            else:
                logging.warning(f"No data for cryptocurrency ticker: {ticker}")

        for ticker, data in stock_data.items():
            if not data.empty:
                daily_returns = data['Close'].pct_change().dropna()
                combined_returns[ticker] = daily_returns
                stock_returns[ticker] = daily_returns
            else:
                logging.warning(f"No data for stock ticker: {ticker}")

        # Ensure combined returns are not empty
        if combined_returns.empty:
            logging.error("Combined returns are empty. No valid tickers provided.")
            return None

        # Step 4: Perform sentiment analysis
        sentiment_scores = fetch_tweets_sentiment(crypto_tickers + stock_tickers)

        # Step 5: Train ML models and calculate ISM metrics
        all_data = []
        returns = []
        for ticker in combined_returns.columns:
            price_data = crypto_data.get(ticker, pd.DataFrame()) if ticker in crypto_data else stock_data.get(ticker)

            model, scaler = None, None
            if price_data is not None and not price_data.empty:
                model, scaler = train_deep_learning_model(price_data)

            technicals = calculate_technical_indicators(price_data) if price_data is not None else {}
            ml_prediction = 0
            if model and scaler and not price_data.empty:
                last_60_days = price_data['price'].values[-60:].reshape(-1, 1)
                last_60_days_scaled = scaler.transform(last_60_days)
                X_test = np.array([last_60_days_scaled])
                ml_prediction = model.predict(X_test)[0][0]

            sharpe_ratio = (
                (combined_returns[ticker].mean() - RISK_FREE_RATE) / combined_returns[ticker].std()
            ) * np.sqrt(252) if not combined_returns[ticker].empty else 0

            all_data.append({
                "ticker": ticker,
                "sentiment": sentiment_scores.get(ticker, 0),
                "technical": technicals.get("SMA", 0),
                "ml": ml_prediction,
                "sharpe": sharpe_ratio,
            })
            returns.append(combined_returns[ticker].sum())

        # Step 6: Optimize ISM weights
        data_df = pd.DataFrame(all_data)
        if data_df.empty:
            logging.error("ISM DataFrame is empty. No valid metrics calculated.")
            return None

        optimized_weights = optimize_weights(data_df, returns)

        # Calculate ISM scores
        data_df["investment_surety"] = data_df.apply(
            lambda row: calculate_investment_surety(
                row["sentiment"], row["technical"], row["ml"], row["sharpe"], optimized_weights
            ),
            axis=1,
        )

        # Step 7: Compare strategies
        strategy_results = compare_strategies(combined_returns)
                # Alpaca trading
        portfolio_balance = float(alpaca.get_account().equity)  # Fetch current equity
        orders = execute_alpaca_orders(optimized_weights, combined_returns.columns, portfolio_balance)

        # Track portfolio performance
        performance = get_portfolio_performance()

        return {
            "ISM Scores": data_df.to_dict("records"),
            "Strategy Results": strategy_results,
            "Optimized Weights": optimized_weights,
            "crypto_returns": crypto_returns,
            "stock_returns": stock_returns,
            "orders": orders,  # Include trade execution details
            "performance": performance  # Include portfolio performance metrics
        }
    except Exception as e:
        logging.error(f"Error in trading pipeline: {e}")
        return None

from flask import Flask, has_request_context, request

def create_dashboard():
    """
    Create a modern Dash GUI for the trading bot with a Binance-inspired design.
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

    # Layout
    app.layout = dbc.Container(
        [
            # Header Section
            dbc.Row(
                dbc.Col(
                    html.H1(
                        "AmbiVest: Intelligent Portfolio Strategies",
                        style={
                            "textAlign": "center",
                            "color": "#FFD700",
                            "padding": "20px",
                            "fontWeight": "bold",
                            "fontFamily": "Arial, sans-serif"
                        },
                    )
                ),
                style={"backgroundColor": "#1a1a1a", "padding": "10px", "borderBottom": "2px solid #FFD700"},
            ),

            # Input Section
            dbc.Row(
                [
                    # Input Panel for Cryptos, Stocks, and Developer Key
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Input Section", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    [
                                        html.Label(
                                            "Cryptocurrencies",
                                            style={"color": "#FFFFFF", "fontWeight": "bold"},
                                        ),
                                        dcc.Input(
                                            id="crypto-input",
                                            type="text",
                                            placeholder="Enter crypto tickers (e.g., bitcoin, ethereum)",
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "marginBottom": "10px",
                                                "borderRadius": "5px",
                                            },
                                        ),
                                        html.Label(
                                            "Stocks",
                                            style={"color": "#FFFFFF", "fontWeight": "bold"},
                                        ),
                                        dcc.Input(
                                            id="stock-input",
                                            type="text",
                                            placeholder="Enter stock tickers (e.g., AAPL, MSFT)",
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "marginBottom": "10px",
                                                "borderRadius": "5px",
                                            },
                                        ),
                                        html.Label(
                                            "Developer Key (Optional)",
                                            style={"color": "#FFFFFF", "fontWeight": "bold"},
                                        ),
                                        dcc.Input(
                                            id="developer-key",
                                            type="password",
                                            placeholder="Enter developer key for detailed metrics",
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "marginBottom": "10px",
                                                "borderRadius": "5px",
                                            },
                                        ),
                                        dbc.Button(
                                            "Run Analysis",
                                            id="run-button",
                                            color="warning",
                                            className="w-100",
                                            style={"marginTop": "10px"},
                                        ),
                                    ]
                                ),
                            ],
                            style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                        ),
                        width=4,
                    ),

                    # Date Range Inputs
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Date Range", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    [
                                        html.Label(
                                            "Start Date",
                                            style={"color": "#FFFFFF", "fontWeight": "bold"},
                                        ),
                                        dcc.Input(
                                            id="start-date",
                                            type="text",
                                            placeholder="YYYY-MM-DD",
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "marginBottom": "10px",
                                                "borderRadius": "5px",
                                            },
                                        ),
                                        html.Label(
                                            "End Date",
                                            style={"color": "#FFFFFF", "fontWeight": "bold"},
                                        ),
                                        dcc.Input(
                                            id="end-date",
                                            type="text",
                                            placeholder="YYYY-MM-DD",
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "marginBottom": "10px",
                                                "borderRadius": "5px",
                                            },
                                        ),
                                        html.P(
                                            "Note: CoinGecko's demo version supports only the last 365 days.",
                                            style={"color": "#FF0000", "fontWeight": "bold"},
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

            # Placeholder for Cryptocurrency Performance Trend
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Cryptocurrency Performance Trend", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="crypto-performance-trend")  # Placeholder for crypto trend
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Placeholder for Stock Performance Trend
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Stock Performance Trend", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="stock-performance-trend")  # Placeholder for stock trend
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Placeholder for Portfolio Performance
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Portfolio Performance", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="portfolio-performance")  # Placeholder for portfolio performance
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Placeholder for Strategy Comparison
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Strategy Comparison", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="strategy-comparison")  # Placeholder for strategy comparison
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Portfolio Metrics Section
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Portfolio Metrics", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                [
                                    html.P(id="portfolio-equity", style={"color": "#FFFFFF", "fontSize": "16px"}),  # Placeholder for equity
                                    html.P(id="portfolio-buying-power", style={"color": "#FFFFFF", "fontSize": "16px"}),  # Placeholder for buying power
                                    html.P(id="portfolio-pnl", style={"color": "#FFFFFF", "fontSize": "16px"}),  # Placeholder for PnL
                                ]
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),
            # Optimized Weights
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Optimized Weights", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                html.Div(id="optimized-weights", style={"color": "#FFFFFF", "fontSize": "16px"})  # Placeholder for weights
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Developer Metrics
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Developer Metrics", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                html.Pre(
                                    id="developer-metrics",
                                    style={
                                        "color": "#FFFFFF",
                                        "backgroundColor": "#2a2a2a",
                                        "padding": "15px",
                                        "borderRadius": "5px",
                                        "overflowX": "auto",
                                    },
                                )
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Historical Data
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Historical Data", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                html.Div(id="historical-data", style={"color": "#FFFFFF", "whiteSpace": "pre-wrap"})
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Trades Executed
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Trades Executed", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                html.Div(id="trades-executed", style={"color": "#FFFFFF", "whiteSpace": "pre-wrap"})
                            ),
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

    # Callbacks
    @app.callback(
        [
            Output("crypto-performance-trend", "figure"),
            Output("stock-performance-trend", "figure"),
            Output("portfolio-performance", "figure"),
            Output("strategy-comparison", "figure"),
            Output("optimized-weights", "children"),
            Output("developer-metrics", "children"),
            Output("portfolio-equity", "children"),
            Output("portfolio-buying-power", "children"),
            Output("portfolio-pnl", "children"),
            Output("historical-data", "children"),
            Output("trades-executed", "children"),
        ],
        Input("run-button", "n_clicks"),
        [
            Input("crypto-input", "value"),
            Input("stock-input", "value"),
            Input("start-date", "value"),
            Input("end-date", "value"),
            Input("developer-key", "value"),
        ],
    )
    def update_dashboard(n_clicks, crypto_input, stock_input, start_date, end_date, developer_key):
        try:
            # Ensure the callback is only triggered after the button is clicked
            if n_clicks == 0:
                return {}, {}, {}, {}, "Enter tickers and click 'Run Analysis' to start.", "", "", "", "", "No historical data available.", "No trades executed."

            # Parse user inputs
            crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
            stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

            # Default date range to last 365 days if not provided
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")
            start_date = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            # Log user input for debugging
            logging.info(f"Running analysis for crypto: {crypto_tickers}, stocks: {stock_tickers}, start_date: {start_date}, end_date: {end_date}")

            # Execute the trading pipeline
            results = execute_trading_pipeline(crypto_tickers, stock_tickers, start_date, end_date)
            if not results:
                logging.error("Failed to execute the trading pipeline.")
                return {}, {}, {}, {}, "Error: Failed to execute the trading pipeline.", "", "", "", "", "No historical data available.", "No trades executed."

            # Extract performance metrics from the pipeline results
            performance = results.get("performance", {})
            equity = f"Portfolio Equity: ${performance.get('equity', 0):.2f}"
            buying_power = f"Buying Power: ${performance.get('buying_power', 0):.2f}"
            pnl = f"PnL (Change): ${performance.get('pnl', 0):.2f}"

            # Generate Cryptocurrency Performance Trendline
            crypto_fig = {
                "data": [
                    go.Scatter(
                        x=results["crypto_returns"].index,
                        y=(1 + results["crypto_returns"].sum(axis=1)).cumprod(),
                        mode="lines",
                        name="Crypto Portfolio",
                        line=dict(color="#FFD700"),
                    )
                ],
                "layout": go.Layout(
                    title="Cryptocurrency Performance Over Time",
                    xaxis={"title": "Date"},
                    yaxis={"title": "Portfolio Value"},
                    plot_bgcolor="#121212",
                    paper_bgcolor="#121212",
                    font={"color": "#FFD700"},
                ),
            }

            # Generate Stock Performance Trendline
            stock_fig = {
                "data": [
                    go.Scatter(
                        x=results["stock_returns"].index,
                        y=(1 + results["stock_returns"].sum(axis=1)).cumprod(),
                        mode="lines",
                        name="Stock Portfolio",
                        line=dict(color="#00CC96"),
                    )
                ],
                "layout": go.Layout(
                    title="Stock Performance Over Time",
                    xaxis={"title": "Date"},
                    yaxis={"title": "Portfolio Value"},
                    plot_bgcolor="#121212",
                    paper_bgcolor="#121212",
                    font={"color": "#FFD700"},
                ),
            }

            # Portfolio Performance Graph (Placeholder)
            portfolio_fig = {
                "data": [],  # Add portfolio performance data here
                "layout": go.Layout(
                    title="Portfolio Performance",
                    xaxis={"title": "Date"},
                    yaxis={"title": "Portfolio Value"},
                    plot_bgcolor="#121212",
                    paper_bgcolor="#121212",
                    font={"color": "#FFD700"},
                ),
            }

            # Strategy Comparison Graph (Placeholder)
            strategy_fig = {
                "data": [],  # Add strategy comparison data here
                "layout": go.Layout(
                    title="Strategy Comparison",
                    xaxis={"title": "Strategy"},
                    yaxis={"title": "Portfolio Value"},
                    plot_bgcolor="#121212",
                    paper_bgcolor="#121212",
                    font={"color": "#FFD700"},
                ),
            }

            # Optimized Weights
            optimized_weights = results.get("Optimized Weights", {})
            optimized_weights_text = f"Optimized Weights: {optimized_weights}"

            # Developer Mode Metrics
            is_developer_mode = developer_key == "econ3086"
            developer_metrics = ""
            if is_developer_mode:
                developer_metrics = (
                    f"ISM Scores:\n{results['ISM Scores']}\n\n"
                    f"Strategy Results:\n{results['Strategy Results']}\n\n"
                    f"Optimized Weights:\n{optimized_weights}\n"
                )

            # Historical Data
            historical_data = "Historical Data:\n\n"
            for ticker, data in results.get("crypto_returns", {}).items():
                if not data.empty:
                    latest_price = data["price"].iloc[-1]
                    historical_data += f"Cryptocurrency: {ticker}\nLatest Price: ${latest_price:.2f}\n\n"
                else:
                    historical_data += f"Cryptocurrency: {ticker} - No data available\n\n"

            for ticker, data in results.get("stock_returns", {}).items():
                if not data.empty:
                    latest_price = data["Close"].iloc[-1]
                    historical_data += f"Stock: {ticker}\nLatest Close Price: ${latest_price:.2f}\n\n"
                else:
                    historical_data += f"Stock: {ticker} - No data available\n\n"

            # Trades Executed
            trades_executed = results.get("orders", {})
            trades_summary = "Trades Executed:\n\n"
            if trades_executed:
                for ticker, trade in trades_executed.items():
                    trades_summary += (
                        f"Ticker: {ticker}\n"
                        f"Quantity: {trade['quantity']}\n"
                        f"Price: ${trade['price']:.2f}\n"
                        f"Allocation: ${trade['allocation']:.2f}\n\n"
                    )
            else:
                trades_summary += "No trades executed.\n"

            # Return all outputs for the callback
            return (
                crypto_fig,
                stock_fig,
                portfolio_fig,
                strategy_fig,
                optimized_weights_text,
                developer_metrics,
                equity,
                buying_power,
                pnl,
                historical_data,
                trades_summary,
            )

        except Exception as e:
            # Log the error and return placeholders
            logging.error(f"Error in update_dashboard callback: {str(e)}")
            return {}, {}, {}, {}, "An error occurred. Please check the logs.", "", "", "", "", "Error in historical data.", "Error in trades executed."
    return app
if __name__ == "__main__":
    port = 8050
    app = create_dashboard()
    webbrowser.open_new(f"http://127.0.0.1:{port}")
    app.run_server(debug=True, use_reloader=False, port=port)