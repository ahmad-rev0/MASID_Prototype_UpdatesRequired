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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
API_KEY = "REDACTED_COINGECKO_KEY"  # Replace with your CoinGecko API key
RISK_FREE_RATE = 0.02  # Risk-free rate for Sharpe ratio
INITIAL_BALANCE = 1000  # Initial portfolio balance

# COINGECKO API WRAPPER
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
                return pd.DataFrame()

            prices = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            return prices.set_index('timestamp')
        except Exception as e:
            logging.error(f"Error fetching crypto data for {ticker}: {e}")
            return pd.DataFrame()

# YAHOO FINANCE WRAPPER
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
                data = yf.download(ticker, start=start_date, end=end_date)
                if not data.empty:
                    valid_data[ticker] = data
            except Exception as e:
                logging.error(f"Failed to fetch stock data for {ticker}: {e}")
        return valid_data
# SENTIMENT ANALYSIS
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
            # Simulate tweets with variability for demonstration
            tweets = [f"Positive news about {ticker}", f"Negative sentiment about {ticker}"]
            sentiment_score = np.mean([TextBlob(tweet).sentiment.polarity for tweet in tweets])
            sentiments[ticker] = sentiment_score
        except Exception as e:
            logging.error(f"Error fetching sentiment for {ticker}: {e}")
            sentiments[ticker] = 0  # Default neutral sentiment
    return sentiments
# MACHINE LEARNING MODULE: LSTM MODEL
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
            logging.warning(f"Insufficient data for training LSTM model.")
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
    # TECHNICAL INDICATORS
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
        return {
            'SMA': data['SMA'].iloc[-1],
            'EMA': data['EMA'].iloc[-1],
        }
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return {"SMA": 0, "EMA": 0}
# INVESTMENT SURETY METRIC (ISM)
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
# PORTFOLIO OPTIMIZATION
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
        return -sharpe_ratio

    # Constraints: Weights must sum to 1
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    # Bounds: Each weight must be between 0 and 1
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess: Equal allocation
    initial_weights = np.ones(n_assets) / n_assets

    # Optimize portfolio
    result = minimize(negative_sharpe, initial_weights, bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    max_sharpe_ratio = -result.fun
    return dict(zip(returns.columns, optimal_weights)), max_sharpe_ratio
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
    
def optimize_weights(data, returns):
    """
    Optimize the weights for the investment surety metric to maximize returns.
    """
    def objective_function(weights):
        weights = weights / np.sum(weights)
        investment_scores = data.apply(
            lambda row: calculate_investment_surety(
                row['sentiment'], row['technical'], row['ml'], row['sharpe'], weights
            ), axis=1
        )
        return -np.corrcoef(investment_scores, returns)[0, 1]

    initial_weights = [0.25, 0.25, 0.25, 0.25]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * 4

    result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    if result.success:
        return result.x / np.sum(result.x)
    else:
        logging.error("Weight optimization failed!")
        return initial_weights

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
            # Calculate cumulative return for each asset
            cumulative_return = (1 + returns[col]).prod()
            final_values.append(cumulative_return * allocation)

        # Sum the final values of all assets to get the total portfolio value
        return sum(final_values)
    except Exception as e:
        logging.error(f"Error in Buy-and-Hold strategy: {e}")
        return 0

def compare_strategies(returns):
    """
    Compare the performance of AmbiVest, Buy-and-Hold, and Risk Parity strategies.
    """
    try:
        ambivest_weights, max_sharpe_ratio = optimize_portfolio(returns)
        weighted_returns = returns.dot(list(ambivest_weights.values()))
        ambivest_portfolio_value = (1 + weighted_returns).cumprod()[-1] * INITIAL_BALANCE

        buy_hold_value = buy_and_hold(returns)
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
        for ticker, data in crypto_data.items():
            if not data.empty:
                combined_returns[ticker] = data['price'].pct_change().dropna()

        for ticker, data in stock_data.items():
            if ticker in stock_data and not data.empty:
                combined_returns[ticker] = data['Close'].pct_change().dropna()

        # Step 4: Perform sentiment analysis
        logging.info("Performing sentiment analysis...")
        sentiment_scores = fetch_tweets_sentiment(crypto_tickers + stock_tickers)

        # Step 5: Train LSTM models and calculate ISM metrics
        logging.info("Training LSTM models and calculating ISM metrics...")
        all_data = []
        returns = []

        for ticker in combined_returns.columns:
            # Fetch historical price data
            price_data = crypto_data.get(ticker, pd.DataFrame()) if ticker in crypto_data else stock_data.get(ticker)

            # Train LSTM model if data is available
            model, scaler = None, None
            if price_data is not None and not price_data.empty:
                model, scaler = train_deep_learning_model(price_data)

            # Calculate technical indicators
            technicals = calculate_technical_indicators(price_data) if price_data is not None else {}

            # Predict future value using LSTM model
            ml_prediction = 0
            if model and scaler and not price_data.empty:
                last_60_days = price_data['price'].values[-60:].reshape(-1, 1)
                last_60_days_scaled = scaler.transform(last_60_days)
                X_test = np.array([last_60_days_scaled])
                ml_prediction = model.predict(X_test)[0][0]

            # Compute Sharpe ratio
            sharpe_ratio = (
                (combined_returns[ticker].mean() - RISK_FREE_RATE) / combined_returns[ticker].std()
            ) * np.sqrt(252)  # Assuming 252 trading days
            if sharpe_ratio < -5:
                logging.warning(f"Extreme negative Sharpe ratio for {ticker}: {sharpe_ratio}")

            # Collect ISM components
            all_data.append({
                "ticker": ticker,
                "sentiment": sentiment_scores.get(ticker, 0),
                "technical": technicals.get("SMA", 0),
                "ml": ml_prediction,
                "sharpe": sharpe_ratio,
            })
            returns.append(combined_returns[ticker].sum())  # Example cumulative return calculation

        # Step 6: Optimize ISM weights
        logging.info("Optimizing ISM weights...")
        data_df = pd.DataFrame(all_data)
        optimized_weights = optimize_weights(data_df, returns)

        # Calculate ISM scores
        data_df["investment_surety"] = data_df.apply(
            lambda row: calculate_investment_surety(
                row["sentiment"], row["technical"], row["ml"], row["sharpe"], optimized_weights
            ),
            axis=1,
        )

        # Step 7: Compare strategies
        logging.info("Comparing strategies...")
        strategy_results = compare_strategies(combined_returns)

        # Final results
        logging.info("Trading pipeline executed successfully!")
        return {
            "ISM Scores": data_df.to_dict("records"),
            "Strategy Results": strategy_results,
            "Optimized Weights": optimized_weights,
        }
    except Exception as e:
        logging.error(f"Error in trading pipeline: {e}")
        return None
# DASH APP WITH MODERN DESIGN AND DEVELOPER MODE
def create_dashboard():
    """
    Create a Dash GUI for the trading bot with a modern design and developer mode.
    """
    # Initialize Dash app with Bootstrap theme
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    # Layout
    app.layout = dbc.Container(
        [
            # Header
            dbc.Row(
                dbc.Col(
                    html.H1(
                        "AmbiVest Dashboard",
                        style={
                            "textAlign": "center",
                            "color": "#FFD700",
                            "padding": "20px",
                            "fontWeight": "bold",
                        },
                    )
                ),
                style={"backgroundColor": "#1a1a1a", "padding": "10px"},
            ),

            # Input Section (Cryptos, Stocks, and Developer Key)
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
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
                                    color="primary",
                                    className="w-100",
                                    style={"marginBottom": "10px"},
                                ),
                            ],
                            body=True,
                            style={"backgroundColor": "#2a2a2a", "padding": "15px"},
                        ),
                        width=4,
                    ),

                    # Optimized Weights Section
                    dbc.Col(
                        dbc.Card(
                            [
                                html.H5(
                                    "Optimized Weights",
                                    className="card-title",
                                    style={"color": "#FFD700"},
                                ),
                                html.P(
                                    "Weights will display here after analysis.",
                                    id="optimized-weights",
                                    style={"color": "#FFFFFF", "fontSize": "14px"},
                                ),
                            ],
                            body=True,
                            style={"backgroundColor": "#2a2a2a", "padding": "15px"},
                        ),
                        width=8,
                    ),
                ],
                style={"marginTop": "20px"},
            ),

            # Portfolio Performance Graph
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            html.H5(
                                "Portfolio Performance",
                                className="card-title",
                                style={"color": "#FFD700"},
                            ),
                            dcc.Graph(id="portfolio-performance"),
                        ],
                        body=True,
                        style={"backgroundColor": "#2a2a2a", "padding": "15px"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Strategy Comparison Graph
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            html.H5(
                                "Strategy Comparison",
                                className="card-title",
                                style={"color": "#FFD700"},
                            ),
                            dcc.Graph(id="strategy-comparison"),
                        ],
                        body=True,
                        style={"backgroundColor": "#2a2a2a", "padding": "15px"},
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),

            # Developer Mode Metrics Section
            dbc.Row(
                dbc.Col(
                    dbc.Collapse(
                        dbc.Card(
                            [
                                html.H5(
                                    "Developer Metrics",
                                    className="card-title",
                                    style={"color": "#FFD700"},
                                ),
                                html.Pre(
                                    id="developer-metrics",
                                    style={
                                        "color": "#FFFFFF",
                                        "backgroundColor": "#2a2a2a",
                                        "padding": "15px",
                                        "borderRadius": "5px",
                                        "overflowX": "auto",
                                    },
                                ),
                            ],
                            body=True,
                            style={"backgroundColor": "#2a2a2a", "padding": "15px"},
                        ),
                        id="developer-mode",
                        is_open=False,
                    ),
                    width=12,
                ),
                style={"marginTop": "20px"},
            ),
        ],
        fluid=True,
        style={"backgroundColor": "#121212", "padding": "20px"},
    )

    # CALLBACKS FOR UPDATING DASHBOARD
    @app.callback(
        [
            Output("portfolio-performance", "figure"),
            Output("strategy-comparison", "figure"),
            Output("optimized-weights", "children"),
            Output("developer-mode", "is_open"),
            Output("developer-metrics", "children"),
        ],
        Input("run-button", "n_clicks"),
        [
            Input("crypto-input", "value"),
            Input("stock-input", "value"),
            Input("developer-key", "value"),
        ],
    )
    def update_dashboard(n_clicks, crypto_input, stock_input, developer_key):
        if n_clicks == 0:
            return {}, {}, "Enter tickers and click 'Run Analysis' to start.", False, ""

        crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
        stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

        # Execute the pipeline
        results = execute_trading_pipeline(crypto_tickers, stock_tickers, "2023-01-01", "2023-12-31")
        if not results:
            return {}, {}, "Error: Failed to execute the trading pipeline.", False, ""

        # Portfolio Performance Graph
        portfolio_fig = {
            "data": [
                go.Scatter(
                    x=[ticker['investment_surety'] for ticker in results["ISM Scores"]],
                    y=[ticker['sharpe'] for ticker in results["ISM Scores"]],
                    mode="markers",
                    name="Assets",
                    marker=dict(color="#FFD700"),
                )
            ],
            "layout": go.Layout(
                title="ISM Scores vs Sharpe Ratios",
                xaxis={"title": "ISM Score"},
                yaxis={"title": "Sharpe Ratio"},
                plot_bgcolor="#121212",  # Set graph background
                paper_bgcolor="#121212",  # Set surrounding background
                font={"color": "#FFD700"},  # Set font color to gold
            ),
        }

        # Strategy Comparison Graph
        strategy_fig = {
            "data": [
                go.Bar(
                    x=list(results["Strategy Results"].keys()),
                    y=[results["Strategy Results"][strategy]["Portfolio Value"] for strategy in results["Strategy Results"]],
                    name="Portfolio Value",
                    marker_color="#FFD700",  # Golden bars
                )
            ],
            "layout": go.Layout(
                title="Strategy Comparison",
                xaxis={"title": "Strategy"},
                yaxis={"title": "Portfolio Value"},
                plot_bgcolor="#121212",  # Set graph background
                paper_bgcolor="#121212",  # Set surrounding background
                font={"color": "#FFD700"},  # Set font color to gold
            ),
        }

        # Check if Developer Key is correct
        is_developer_mode = developer_key == "econ3086"

        # Developer Metrics
        developer_metrics = ""
        if is_developer_mode:
            developer_metrics = (
                f"ISM Scores:\n{results['ISM Scores']}\n\n"
                f"Strategy Results:\n{results['Strategy Results']}\n\n"
                f"Optimized Weights:\n{results['Optimized Weights']}\n"
            )

        return portfolio_fig, strategy_fig, f"Optimized Weights: {results['Optimized Weights']}", is_developer_mode, developer_metrics

    return app


# RUN THE APP
if __name__ == "__main__":
    port = 8050
    webbrowser.open_new(f"http://127.0.0.1:{port}")
    app = create_dashboard()
    app.run_server(debug=True, use_reloader=False, port=port)