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


# COINGECKO API WRAPPER
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
            return {coin['id']: coin['symbol'].upper() for coin in coins}  # ID to Symbol mapping
        except Exception as e:
            logging.error(f"Error fetching valid crypto IDs: {e}")
            return {}

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
                logging.info(f"Fetching stock data for {ticker}...")
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    logging.warning(f"No data returned for {ticker}. Skipping...")
                    continue
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
            # Simulate tweets; replace with actual API calls for real-world usage
            tweets = [f"Example tweet about {ticker}"] * 5
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
        data['SMA'] = data['price'].rolling(window=20).mean()
        data['EMA'] = data['price'].ewm(span=20, adjust=False).mean()
        return {
            'SMA': data['SMA'].iloc[-1],
            'EMA': data['EMA'].iloc[-1],
        }
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return {}
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


# WEIGHT OPTIMIZATION
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
        # Normalize weights to ensure they sum to 1
        weights = weights / np.sum(weights)
        investment_scores = data.apply(
            lambda row: calculate_investment_surety(
                row['sentiment'], row['technical'], row['ml'], row['sharpe'], weights
            ), axis=1
        )
        # Calculate the negative correlation between investment scores and actual returns (maximize returns)
        return -np.corrcoef(investment_scores, returns)[0, 1]

    # Initial weights (equal distribution)
    initial_weights = [0.25, 0.25, 0.25, 0.25]
    # Constraints: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    # Bounds: weights must be between 0 and 1
    bounds = [(0, 1)] * 4

    # Perform optimization
    result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x / np.sum(result.x)  # Return normalized weights
    else:
        logging.error("Weight optimization failed!")
        return initial_weights  # Return equal weights as fallback
    
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

def buy_and_hold(returns, initial_balance=INITIAL_BALANCE):
    """
    Simulate a Buy-and-Hold strategy.

    Parameters:
        returns (pd.DataFrame): Daily returns for the portfolio assets.
        initial_balance (float): Initial portfolio balance.

    Returns:
        float: Final portfolio value.
    """
    n_assets = len(returns.columns)
    allocation = initial_balance / n_assets  # Equal allocation
    final_values = []

    for col in returns.columns:
        cumulative_return = (1 + returns[col]).prod()
        final_values.append(cumulative_return * allocation)

    return sum(final_values)
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
        # Compute the volatility (standard deviation of returns) for each asset
        volatilities = returns.std()

        # Compute risk parity weights (inverse of volatility) and normalize them
        weights = 1 / volatilities
        weights /= weights.sum()

        # Calculate cumulative return for each asset
        cumulative_returns = (1 + returns).prod()  # Cumulative product of (1 + daily returns)

        # Compute the portfolio value based on risk parity weights
        portfolio_value = (weights * cumulative_returns * initial_balance).sum()

        logging.info(f"Risk Parity Weights: {weights.to_dict()}")
        return portfolio_value
    except Exception as e:
        logging.error(f"Error in Risk Parity calculation: {e}")
        return 0  # Return 0 if an error occurs

def compare_strategies(returns):
    """
    Compare the performance of AmbiVest, Buy-and-Hold, and Risk Parity strategies.

    Parameters:
        returns (pd.DataFrame): DataFrame of daily returns for portfolio assets.

    Returns:
        dict: Performance metrics for each strategy.
    """
    # AmbiVest: Use optimized portfolio allocation
    ambivest_weights, max_sharpe_ratio = optimize_portfolio(returns)
    weighted_returns = returns.dot(list(ambivest_weights.values()))
    ambivest_portfolio_value = (1 + weighted_returns).cumprod()[-1] * INITIAL_BALANCE

    # Buy-and-Hold Strategy
    buy_hold_value = buy_and_hold(returns)

    # Risk Parity Strategy
    risk_parity_value = risk_parity(returns)

    return {
        "AmbiVest": {"Portfolio Value": ambivest_portfolio_value, "Sharpe Ratio": max_sharpe_ratio},
        "Buy and Hold": {"Portfolio Value": buy_hold_value},
        "Risk Parity": {"Portfolio Value": risk_parity_value},
    }
    
# DASH GUI
def create_dashboard():
    """
    Create a Dash GUI for the trading bot.
    """
    app = Dash(__name__)

    # Layout
    app.layout = html.Div([
        html.H1("AmbiVest: Trading Strategies Comparison", style={"textAlign": "center"}),

        # Input Section
        html.Div([
            html.Label("Cryptocurrencies"),
            dcc.Input(id="crypto-input", type="text", placeholder="Enter crypto tickers (comma-separated)", style={"width": "100%"}),
            html.Label("Stocks"),
            dcc.Input(id="stock-input", type="text", placeholder="Enter stock tickers (comma-separated)", style={"width": "100%"}),
            html.Button("Run Analysis", id="run-button", n_clicks=0),
        ], style={"padding": "20px"}),

        # Portfolio Performance Graph
        html.Div([
            html.H2("Portfolio Performance"),
            dcc.Graph(id="portfolio-performance")
        ]),

        # Strategy Comparison
        html.Div([
            html.H2("Strategy Comparison"),
            dcc.Graph(id="strategy-comparison")
        ]),
    ])

    # Callbacks
    @app.callback(
        [Output("portfolio-performance", "figure"),
         Output("strategy-comparison", "figure")],
        Input("run-button", "n_clicks"),
        [Input("crypto-input", "value"), Input("stock-input", "value")]
    )
    def update_dashboard(n_clicks, crypto_input, stock_input):
        if n_clicks == 0:
            return {}, {}

        crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
        stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

        # Fetch data
        crypto_data = {ticker: CoinGeckoAPI.fetch_crypto_data(ticker, "2023-01-01", "2023-12-31") for ticker in crypto_tickers}
        stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, "2023-01-01", "2023-12-31")

        # Combine returns
        combined_returns = pd.DataFrame()
        for ticker, data in crypto_data.items():
            if not data.empty:
                combined_returns[ticker] = data['price'].pct_change().dropna()

        for ticker, data in stock_data.items():
            if not data.empty:
                combined_returns[ticker] = data['Close'].pct_change().dropna()

        # Compare strategies
        results = compare_strategies(combined_returns)

        # Portfolio Performance Figure
        portfolio_fig = {
            "data": [
                go.Scatter(
                    x=combined_returns.index,
                    y=(1 + combined_returns.mean(axis=1)).cumprod(),
                    mode="lines",
                    name="Portfolio Performance"
                )
            ],
            "layout": go.Layout(
                title="Portfolio Performance Over Time",
                xaxis={"title": "Date"},
                yaxis={"title": "Portfolio Value"}
            )
        }

        # Strategy Comparison Figure
        strategy_fig = {
            "data": [
                go.Bar(
                    x=list(results.keys()),
                    y=[results[strategy]["Portfolio Value"] for strategy in results],
                    name="Portfolio Value"
                )
            ],
            "layout": go.Layout(
                title="Strategy Comparison",
                xaxis={"title": "Strategy"},
                yaxis={"title": "Portfolio Value"}
            )
        }

        return portfolio_fig, strategy_fig

    return app


# Run the app
if __name__ == "__main__":
    app = create_dashboard()
    app.run_server(debug=True, use_reloader=False)
    
# TESTING FUNCTIONS
def test_crypto_data_fetching():
    """
    Test the functionality of fetching cryptocurrency data from CoinGecko.
    """
    try:
        logging.info("Testing cryptocurrency data fetching...")
        crypto_data = CoinGeckoAPI.fetch_crypto_data("bitcoin", "2023-01-01", "2023-12-31")
        assert not crypto_data.empty, "Failed to fetch valid cryptocurrency data."
        logging.info("Crypto data fetching test passed!")
        return "Crypto Data Fetching Test: PASSED"
    except Exception as e:
        logging.error(f"Crypto data fetching test failed: {e}")
        return f"Crypto Data Fetching Test: FAILED ({e})"


def test_stock_data_fetching():
    """
    Test the functionality of fetching stock data from Yahoo Finance.
    """
    try:
        logging.info("Testing stock data fetching...")
        stock_data = YahooFinanceAPI.fetch_stock_data(["AAPL"], "2023-01-01", "2023-12-31")
        assert "AAPL" in stock_data and not stock_data["AAPL"].empty, "Failed to fetch valid stock data."
        logging.info("Stock data fetching test passed!")
        return "Stock Data Fetching Test: PASSED"
    except Exception as e:
        logging.error(f"Stock data fetching test failed: {e}")
        return f"Stock Data Fetching Test: FAILED ({e})"


def test_sentiment_analysis():
    """
    Test the functionality of sentiment analysis.
    """
    try:
        logging.info("Testing sentiment analysis...")
        sentiment_scores = fetch_tweets_sentiment(["bitcoin", "AAPL"])
        assert isinstance(sentiment_scores, dict), "Sentiment analysis did not return a dictionary."
        assert all(isinstance(score, float) for score in sentiment_scores.values()), "Invalid sentiment scores."
        logging.info("Sentiment analysis test passed!")
        return "Sentiment Analysis Test: PASSED"
    except Exception as e:
        logging.error(f"Sentiment analysis test failed: {e}")
        return f"Sentiment Analysis Test: FAILED ({e})"


def test_portfolio_optimization():
    """
    Test the functionality of portfolio optimization.
    """
    try:
        logging.info("Testing portfolio optimization...")
        dummy_returns = pd.DataFrame({
            "bitcoin": np.random.normal(0.001, 0.02, 252),
            "AAPL": np.random.normal(0.001, 0.015, 252)
        })
        weights, sharpe_ratio = optimize_portfolio(dummy_returns)
        assert isinstance(weights, dict), "Portfolio optimization did not return a dictionary of weights."
        assert sharpe_ratio > 0, "Sharpe ratio calculation failed."
        logging.info("Portfolio optimization test passed!")
        return "Portfolio Optimization Test: PASSED"
    except Exception as e:
        logging.error(f"Portfolio optimization test failed: {e}")
        return f"Portfolio Optimization Test: FAILED ({e})"


def run_tests():
    """
    Run all test cases for the trading bot.
    """
    test_results = [
        test_crypto_data_fetching(),
        test_stock_data_fetching(),
        test_sentiment_analysis(),
        test_portfolio_optimization()
    ]

    for result in test_results:
        print(result)
    return test_results

# FINAL EXECUTION LOGIC
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
        # Fetch data
        logging.info("Fetching cryptocurrency data...")
        crypto_data = {
            ticker: CoinGeckoAPI.fetch_crypto_data(ticker, start_date, end_date)
            for ticker in crypto_tickers
        }
        logging.info("Fetching stock data...")
        stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, start_date, end_date)

        # Combine returns
        combined_returns = pd.DataFrame()
        for ticker, data in crypto_data.items():
            if not data.empty:
                combined_returns[ticker] = data['price'].pct_change().dropna()

        for ticker, data in stock_data.items():
            if not data.empty:
                combined_returns[ticker] = data['Close'].pct_change().dropna()

        # Perform sentiment analysis
        logging.info("Performing sentiment analysis...")
        sentiment_scores = fetch_tweets_sentiment(crypto_tickers + stock_tickers)

        # Train ML models and calculate ISM
        logging.info("Training ML models and calculating ISM...")
        all_data = []
        returns = []
        for ticker in combined_returns.columns:
            model, scaler = train_deep_learning_model(crypto_data.get(ticker, pd.DataFrame()))
            technicals = calculate_technical_indicators(crypto_data.get(ticker, pd.DataFrame()))
            ml_prediction = 0
            if model:
                last_60_days = crypto_data[ticker]['price'].values[-60:].reshape(-1, 1)
                last_60_days_scaled = scaler.transform(last_60_days)
                X_test = np.array([last_60_days_scaled])
                ml_prediction = model.predict(X_test)[0][0]

            sharpe_ratio = (
                (combined_returns[ticker].mean() - RISK_FREE_RATE) / combined_returns[ticker].std()
            ) * np.sqrt(252)  # Assuming 252 trading days

            all_data.append({
                'sentiment': sentiment_scores.get(ticker, 0),
                'technical': technicals.get('SMA', 0),
                'ml': ml_prediction,
                'sharpe': sharpe_ratio,
            })
            returns.append(combined_returns[ticker].sum())  # Example return calculation

        data_df = pd.DataFrame(all_data)

        # Optimize ISM weights
        logging.info("Optimizing ISM weights...")
        optimized_weights = optimize_weights(data_df, returns)

        # Calculate ISM scores
        data_df['investment_surety'] = data_df.apply(
            lambda row: calculate_investment_surety(
                row['sentiment'], row['technical'], row['ml'], row['sharpe'], optimized_weights
            ),
            axis=1
        )

        # Compare strategies
        logging.info("Comparing strategies...")
        strategy_results = compare_strategies(combined_returns)

        logging.info("Trading pipeline executed successfully!")
        return {
            "ISM Scores": data_df.to_dict('records'),
            "Strategy Results": strategy_results,
            "Optimized Weights": optimized_weights,
        }
    except Exception as e:
        logging.error(f"Error in trading pipeline: {e}")
        return None
    
@app.callback(
    [Output("portfolio-performance", "figure"),
     Output("strategy-comparison", "figure"),
     Output("output-section", "children")],
    Input("run-button", "n_clicks"),
    [Input("crypto-input", "value"), Input("stock-input", "value")]
)
def update_dashboard(n_clicks, crypto_input, stock_input):
    if n_clicks == 0:
        return {}, {}, "Enter tickers and click 'Run Analysis' to start."

    crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
    stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

    # Execute the pipeline
    results = execute_trading_pipeline(crypto_tickers, stock_tickers, "2023-01-01", "2023-12-31")
    if not results:
        return {}, {}, "Error: Failed to execute the trading pipeline."

    # Portfolio Performance Graph
    portfolio_fig = {
        "data": [
            go.Scatter(
                x=[ticker['investment_surety'] for ticker in results["ISM Scores"]],
                y=[ticker['sharpe'] for ticker in results["ISM Scores"]],
                mode="markers",
                name="Assets",
            )
        ],
        "layout": go.Layout(
            title="ISM Scores vs Sharpe Ratios",
            xaxis={"title": "ISM Score"},
            yaxis={"title": "Sharpe Ratio"}
        )
    }

    # Strategy Comparison Graph
    strategy_fig = {
        "data": [
            go.Bar(
                x=list(results["Strategy Results"].keys()),
                y=[results["Strategy Results"][strategy]["Portfolio Value"] for strategy in results["Strategy Results"]],
                name="Portfolio Value"
            )
        ],
        "layout": go.Layout(
            title="Strategy Comparison",
            xaxis={"title": "Strategy"},
            yaxis={"title": "Portfolio Value"}
        )
    }

    # Output section with optimized weights
    output_text = f"Optimized Weights: {results['Optimized Weights']}"
    return portfolio_fig, strategy_fig, output_text