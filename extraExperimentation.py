# BLOCK 1: SETUP AND CONFIGURATION

# Import necessary libraries
import requests
import pandas as pd
import numpy as np
import logging
from textblob import TextBlob
import re
import matplotlib as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants for Configurations and API Keys
# Replace with your own RapidAPI key
RAPIDAPI_KEY = "your_rapidapi_key"
RAPIDAPI_HOST = "twitter154.p.rapidapi.com"  # Example for the "Twitter API v2" or "Twitter Data API"

COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# Set up global constants
INITIAL_BALANCE = 1000  # Default initial balance for backtesting
RISK_FREE_RATE = 0.0  # Assumed risk-free rate for Sharpe ratio calculation
LEARNING_RATE = 0.01  # Learning rate for adaptive signal adjustment

# Global Configuration for Logging and Data Handling
logging.info("Setting up configuration and dependencies...")

# RapidAPI Twitter API Wrapper
class RapidAPITwitter:
    """
    Wrapper for fetching Twitter data using RapidAPI.
    """

    def __init__(self, api_key, host):
        """
        Initialize with RapidAPI key and host.

        Parameters:
            api_key (str): Your RapidAPI key.
            host (str): Host for the Twitter API on RapidAPI.
        """
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": host
        }
        self.base_url = f"https://{host}"

    def search_tweets(self, query, count=100):
        """
        Fetch tweets based on a search query.

        Parameters:
            query (str): Search query (e.g., "#bitcoin").
            count (int): Number of tweets to fetch.

        Returns:
            list: List of tweets or an empty list if an error occurs.
        """
        url = f"{self.base_url}/search"
        params = {"query": query, "count": count}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()  # Raise an error for bad HTTP responses
            data = response.json()
            if "results" in data:
                tweets = [tweet["text"] for tweet in data["results"]]
                logging.info(f"Fetched {len(tweets)} tweets for query: {query}")
                return tweets
            else:
                logging.warning(f"No results found for query: {query}")
                return []
        except Exception as e:
            logging.error(f"Failed to fetch tweets from RapidAPI: {e}")
            return []

# CoinGecko API Wrapper
class CoinGeckoAPI:
    """
    A simple wrapper for CoinGecko's API.
    """

    @staticmethod
    def get_price(symbol, currency="usd"):
        """
        Fetch the current price of a cryptocurrency.

        Parameters:
            symbol (str): The symbol of the cryptocurrency (e.g., 'bitcoin', 'ethereum').
            currency (str): The target currency to convert to (default is 'usd').

        Returns:
            float: The current price of the cryptocurrency in the target currency.
        """
        url = f"{COINGECKO_API_URL}/simple/price"
        params = {
            "ids": symbol,
            "vs_currencies": currency
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            price = data.get(symbol, {}).get(currency, None)
            if price is not None:
                logging.info(f"Fetched current price for {symbol}: {price} {currency}")
                return price
            else:
                logging.warning(f"Price data not available for {symbol} in {currency}.")
                return None
        except Exception as e:
            logging.error(f"Error fetching price for {symbol} from CoinGecko: {e}")
            return None

    @staticmethod
    def get_historical_data(symbol, currency="usd", days=30):
        """
        Fetch historical price data for a cryptocurrency.

        Parameters:
            symbol (str): The symbol of the cryptocurrency (e.g., 'bitcoin', 'ethereum').
            currency (str): The target currency to convert to (default is 'usd').
            days (int): Number of past days to fetch data for (default is 30).

        Returns:
            pd.DataFrame: A DataFrame containing historical price data (date, price).
        """
        url = f"{COINGECKO_API_URL}/coins/{symbol}/market_chart"
        params = {
            "vs_currency": currency,
            "days": days,
            "interval": "daily"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse the historical data
            prices = data.get("prices", [])
            historical_data = pd.DataFrame(prices, columns=["timestamp", "price"])
            historical_data["timestamp"] = pd.to_datetime(historical_data["timestamp"], unit="ms")
            logging.info(f"Fetched {len(historical_data)} days of historical data for {symbol}.")
            return historical_data
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol} from CoinGecko: {e}")
            return pd.DataFrame()

# Initialize the RapidAPI Twitter client
twitter_api = RapidAPITwitter(api_key=RAPIDAPI_KEY, host=RAPIDAPI_HOST)

# Example Initialization and Usage for Testing
if __name__ == "__main__":
    # Example: Fetch tweets for Bitcoin
    query = "#bitcoin"
    fetched_tweets = twitter_api.search_tweets(query=query, count=100)
    if fetched_tweets:
        print(f"Fetched {len(fetched_tweets)} tweets for query '{query}':")
        for tweet in fetched_tweets[:5]:  # Display the first 5 tweets
            print(f"- {tweet}")
    else:
        print("No tweets fetched.")

    # Example: Fetch current price for Bitcoin
    current_price = CoinGeckoAPI.get_price(symbol="bitcoin", currency="usd")
    print(f"Current Bitcoin price: ${current_price}")

    # Example: Fetch historical price data for Bitcoin
    historical_data = CoinGeckoAPI.get_historical_data(symbol="bitcoin", currency="usd", days=30)
    print(historical_data.head())
# BLOCK 2: DATA FETCHING AND SENTIMENT ANALYSIS

# 1. Fetch Live Cryptocurrency Prices from CoinGecko
def fetch_live_crypto_prices(symbols, currency="usd"):
    # """
    # Fetch live cryptocurrency prices from CoinGecko.

    # Parameters:
    #     symbols (list): List of cryptocurrency symbols (e.g., ['bitcoin', 'ethereum']).
    #     currency (str): The target currency to fetch prices in (e.g., 'usd').

    # Returns:
    #     dict: Current prices for each symbol.
    # """
    prices = {}
    try:
        for symbol in symbols:
            price = CoinGeckoAPI.get_price(symbol, currency)
            if price is not None:
                prices[symbol] = price
            else:
                prices[symbol] = None
        logging.info(f"Fetched live prices for {len(symbols)} symbols.")
    except Exception as e:
        logging.error(f"Error fetching live prices: {e}")
    return prices


# 2. Fetch Historical Price Data from CoinGecko
def fetch_historical_data(symbol, currency="usd", days=30):
    # """
    # Fetch historical price data for a cryptocurrency from CoinGecko.

    # Parameters:
    #     symbol (str): The cryptocurrency symbol (e.g., 'bitcoin', 'ethereum').
    #     currency (str): The target currency for the prices (e.g., 'usd').
    #     days (int): Number of past days to fetch data for.

    # Returns:
    #     pd.DataFrame: Historical price data with columns ['timestamp', 'price'].
    # """
    try:
        historical_data = CoinGeckoAPI.get_historical_data(symbol, currency, days)
        if not historical_data.empty:
            logging.info(f"Fetched historical data for {symbol} over the last {days} days.")
            return historical_data
        else:
            logging.warning(f"No historical data found for {symbol}.")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()


# 3. Sentiment Analysis Using RapidAPI for Tweets
def fetch_and_analyze_tweets(twitter_api, query, count=100):
    # """
    # Fetch recent tweets based on a query and analyze their sentiment.

    # Parameters:
    #     twitter_api (RapidAPITwitter): Initialized RapidAPI Twitter client.
    #     query (str): The search query (e.g., '#bitcoin').
    #     count (int): Number of tweets to fetch.

    # Returns:
    #     float: Average sentiment score for the query.
    # """
    try:
        tweets = twitter_api.search_tweets(query=query, count=count)
        sentiment_scores = []
        for tweet in tweets:
            # Clean the tweet text
            text = re.sub(r"http\S+", "", tweet)  # Remove URLs
            text = re.sub(r"[^a-zA-Z ]", "", text)  # Remove special characters
            text = text.lower()

            # Analyze sentiment
            analysis = TextBlob(text)
            sentiment_scores.append(analysis.sentiment.polarity)

        # Calculate average sentiment score
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            logging.info(f"Fetched and analyzed {len(sentiment_scores)} tweets for query '{query}'.")
            return avg_sentiment
        else:
            logging.warning(f"No tweets found for query '{query}'.")
            return 0.0
    except Exception as e:
        logging.error(f"Error fetching or analyzing tweets: {e}")
        return 0.0
# BLOCK 3: TECHNICAL ANALYSIS AND SIGNAL GENERATION

# 1. Technical Indicators

def calculate_moving_averages(data, window=14):
    """
    Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA).

    Parameters:
        data (pd.Series): Asset's closing price data.
        window (int): Window size for the moving averages.

    Returns:
        pd.DataFrame: DataFrame with columns ['SMA', 'EMA'].
    """
    data = data.copy()
    data["SMA"] = data["close"].rolling(window=window).mean()
    data["EMA"] = data["close"].ewm(span=window, adjust=False).mean()
    return data


def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI).

    Parameters:
        data (pd.Series): Asset's closing price data.
        window (int): Lookback period for RSI.

    Returns:
        pd.Series: RSI values.
    """
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data["RSI"] = rsi
    return data


# 2. Generate Buy/Sell Signals

def generate_trading_signals(data, sentiment_score, threshold=0.5):
    """
    Generate buy/sell signals based on technical indicators and sentiment analysis.

    Parameters:
        data (pd.DataFrame): DataFrame with columns ['SMA', 'EMA', 'RSI', 'close'].
        sentiment_score (float): Average sentiment score (-1 to 1).
        threshold (float): Sentiment threshold for buy/sell signals.

    Returns:
        pd.DataFrame: DataFrame with an additional 'Signal' column.
    """
    data = data.copy()
    data["Signal"] = 0  # Default to no signal

    # Generate signals based on SMA and EMA crossover
    data.loc[data["SMA"] > data["EMA"], "Signal"] = 1  # Buy signal
    data.loc[data["SMA"] < data["EMA"], "Signal"] = -1  # Sell signal

    # Adjust signals based on RSI (overbought/oversold)
    data.loc[data["RSI"] > 70, "Signal"] = -1  # Overbought, sell
    data.loc[data["RSI"] < 30, "Signal"] = 1   # Oversold, buy

    # Incorporate sentiment analysis
    if sentiment_score > threshold:
        data["Signal"] = data["Signal"].apply(lambda x: x if x == 1 else 0)
    elif sentiment_score < -threshold:
        data["Signal"] = data["Signal"].apply(lambda x: x if x == -1 else 0)

    return data
# MODULE: InvestmentSuretyMetric.py
class InvestmentSuretyMetric:
    """
    Computes the Investment Surety Metric (ISM) based on the weighted sum of sub-metrics.
    """

    @staticmethod
    def calculate_ism(weights, sentiment_score, technical_confidence, prediction_confidence):
        """
        Calculate the Investment Surety Metric (ISM).

        Parameters:
            weights (dict): Weights for sentiment, technical indicators, and predictions
                            (e.g., {'sentiment': 0.33, 'technical': 0.33, 'predictions': 0.33}).
            sentiment_score (float): Normalized sentiment score (-1 to 1).
            technical_confidence (float): Confidence score from technical indicators (0 to 1).
            prediction_confidence (float): Confidence score from predictions (0 to 1).

        Returns:
            float: Normalized investment surety score (0 to 1).
        """
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate weighted sum of sub-metrics
        ism = (
            normalized_weights["sentiment"] * sentiment_score +
            normalized_weights["technical"] * technical_confidence +
            normalized_weights["predictions"] * prediction_confidence
        )

        # Normalize ISM to a range of 0 to 1
        ism_normalized = max(0, min(1, (ism + 1) / 2))  # Shift to 0-1 range if sentiment is -1 to 1

        return ism_normalized

    @staticmethod
    def calculate_technical_confidence(data):
        # """
        # Calculate technical confidence based on signal alignment (e.g., SMA/EMA crossover, RSI levels).

        # Parameters:
        #     data (pd.DataFrame): Dataframe containing technical indicators and signals.

        # Returns:
        #     float: Technical confidence score (0 to 1).
        # """
        try:
            # Example: Confidence if SMA > EMA and RSI is in a healthy range
            sma_above_ema = data["SMA"] > data["EMA"]
            rsi_healthy = (data["RSI"] > 30) & (data["RSI"] < 70)

            # Calculate confidence as the proportion of "confident" periods
            confidence = (sma_above_ema & rsi_healthy).mean()
            return confidence
        except Exception as e:
            logging.error(f"Error calculating technical confidence: {e}")
            return 0.0

        
# BLOCK 4: BACKTESTING MODULE

def backtest_strategy(data, initial_balance=1000):
    """
    Backtest the trading strategy using generated signals.

    Parameters:
        data (pd.DataFrame): DataFrame with columns ['timestamp', 'close', 'Signal'].
        initial_balance (float): Starting balance for backtesting.

    Returns:
        pd.DataFrame: DataFrame with portfolio value and performance metrics.
        dict: Dictionary with performance metrics (cumulative return, Sharpe ratio, etc.).
    """
    data = data.copy()
    balance = initial_balance
    position = 0  # Tracks the number of units held
    portfolio_values = []  # Portfolio value over time

    # Simulate trading based on signals
    for i in range(len(data)):
        signal = data.iloc[i]["Signal"]
        close_price = data.iloc[i]["close"]

        # Buy signal: Invest all balance
        if signal == 1 and balance > 0:
            position = balance / close_price  # Buy crypto
            balance = 0  # All funds invested

        # Sell signal: Liquidate position
        elif signal == -1 and position > 0:
            balance = position * close_price  # Sell crypto
            position = 0  # No holdings

        # Calculate portfolio value
        portfolio_value = balance + (position * close_price)
        portfolio_values.append(portfolio_value)

    # Add portfolio value to DataFrame
    data["Portfolio Value"] = portfolio_values

    # Calculate performance metrics
    data["Daily Returns"] = data["Portfolio Value"].pct_change()
    cumulative_return = (data["Portfolio Value"].iloc[-1] - initial_balance) / initial_balance
    sharpe_ratio = data["Daily Returns"].mean() / data["Daily Returns"].std() * np.sqrt(252)  # Assuming 252 trading days
    volatility = data["Daily Returns"].std() * np.sqrt(252)

    performance_metrics = {
        "Cumulative Return (%)": cumulative_return * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Volatility (%)": volatility * 100,
    }

    return data, performance_metrics


def compare_to_baseline(data):
    """
    Compare the strategy's performance to a buy-and-hold baseline.

    Parameters:
        data (pd.DataFrame): DataFrame with columns ['timestamp', 'close', 'Portfolio Value'].

    Returns:
        pd.DataFrame: DataFrame with baseline values added.
        dict: Performance metrics for the baseline strategy.
    """
    data = data.copy()

    # Baseline buy-and-hold strategy: Invest initial balance at the start and hold
    initial_price = data["close"].iloc[0]
    data["Buy and Hold Value"] = (initial_balance / initial_price) * data["close"]

    # Calculate buy-and-hold performance metrics
    cumulative_return = (data["Buy and Hold Value"].iloc[-1] - initial_balance) / initial_balance
    daily_returns = data["Buy and Hold Value"].pct_change()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    volatility = daily_returns.std() * np.sqrt(252)

    baseline_metrics = {
        "Cumulative Return (%)": cumulative_return * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Volatility (%)": volatility * 100,
    }

    return data, baseline_metrics

# BLOCK 5: DYNAMIC WEIGHT ADJUSTMENT AND REINFORCEMENT LEARNING

# 1. Dynamic Weight Adjustment
def adjust_weights(recent_performance, initial_weights, learning_rate=0.01):
    """
    Dynamically adjust the weights of sentiment, technical indicators, and predictions
    based on recent strategy performance.

    Parameters:
        recent_performance (dict): Dictionary containing recent performance metrics 
                                   (e.g., Sharpe ratio, cumulative return).
        initial_weights (dict): Current weights for each component 
                                (e.g., {'sentiment': 0.33, 'technical': 0.33, 'predictions': 0.33}).
        learning_rate (float): Learning rate for weight adjustments.

    Returns:
        dict: Updated weights.
    """
    # Extract performance metrics
    sharpe_ratio = recent_performance.get("Sharpe Ratio", 0)
    cumulative_return = recent_performance.get("Cumulative Return (%)", 0)

    # Adjust weights based on cumulative return and Sharpe ratio
    updated_weights = initial_weights.copy()
    for key in initial_weights:
        # Reward positive performance and penalize negative performance
        adjustment = learning_rate * sharpe_ratio * (cumulative_return / 100)
        updated_weights[key] += adjustment

    # Normalize weights to ensure they sum to 1
    total_weight = sum(updated_weights.values())
    updated_weights = {k: v / total_weight for k, v in updated_weights.items()}

    logging.info(f"Updated weights: {updated_weights}")
    return updated_weights


# 2. Reinforcement Learning Framework
class RLAgent:
    """
    A basic reinforcement learning agent to optimize trading signals.
    """

    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.95):
        """
        Initialize the RL agent.

        Parameters:
            state_size (int): Number of input features in the state.
            action_size (int): Number of possible actions (e.g., -1, 0, 1).
            learning_rate (float): Learning rate for Q-value updates.
            gamma (float): Discount factor for future rewards.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = {}  # Q-table to store state-action values

    def get_action(self, state):
        """
        Choose an action based on the current state.

        Parameters:
            state (tuple): Current state (e.g., sentiment score, SMA/EMA crossover, RSI).

        Returns:
            int: Chosen action (-1 = sell, 0 = hold, 1 = buy).
        """
        if state not in self.q_table:
            # Initialize Q-values for unseen states
            self.q_table[state] = np.zeros(self.action_size)
        # Choose action with the highest Q-value (greedy policy)
        return np.argmax(self.q_table[state]) - 1  # Convert index to action (-1, 0, 1)

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value using the Q-learning formula.

        Parameters:
            state (tuple): Current state.
            action (int): Action taken.
            reward (float): Reward received for the action.
            next_state (tuple): Next state after taking the action.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)

        # Convert action (-1, 0, 1) to index (0, 1, 2)
        action_idx = action + 1

        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])  # Best future action
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.learning_rate * td_error


# 3. Reward Function
def calculate_reward(portfolio_value, baseline_value):
    """
    Calculate the reward for the RL agent based on portfolio performance.

    Parameters:
        portfolio_value (float): Current portfolio value.
        baseline_value (float): Current buy-and-hold portfolio value.

    Returns:
        float: Reward value.
    """
    return portfolio_value - baseline_value

# BLOCK 6: END-TO-END STRATEGY EXECUTION PIPELINE

def execute_strategy(symbol, initial_balance=1000, lookback_window=14, sentiment_threshold=0.2):
    """
    Execute the full trading strategy pipeline.

    Parameters:
        symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
        initial_balance (float): Starting balance for backtesting.
        lookback_window (int): Lookback window for technical indicators.
        sentiment_threshold (float): Threshold for sentiment-based signal adjustment.

    Returns:
        dict: Results including trading signals, portfolio values, and performance metrics.
    """
    # Initialize weights for sentiment, technical analysis, and predictions
    weights = {"sentiment": 0.33, "technical": 0.33, "predictions": 0.33}

    # Fetch historical price data
    logging.info("Fetching historical price data...")
    price_data = fetch_historical_data(symbol, limit=100)

    # Calculate technical indicators
    logging.info("Calculating technical indicators...")
    price_data = calculate_moving_averages(price_data, window=lookback_window)
    price_data = calculate_rsi(price_data, window=lookback_window)

    # Fetch sentiment score
    logging.info("Performing sentiment analysis...")
    twitter_api = authenticate_twitter()
    sentiment_score = fetch_and_analyze_tweets(twitter_api, query=f"#{symbol.lower()}", count=100)

    # Generate trading signals
    logging.info("Generating trading signals...")
    price_data = generate_trading_signals(price_data, sentiment_score, threshold=sentiment_threshold)

    # Backtest the strategy
    logging.info("Backtesting the strategy...")
    backtest_results, strategy_metrics = backtest_strategy(price_data, initial_balance=initial_balance)

    # Compare to buy-and-hold baseline
    logging.info("Comparing to buy-and-hold baseline...")
    comparison_results, baseline_metrics = compare_to_baseline(backtest_results)

    # Dynamically adjust weights based on strategy performance
    logging.info("Adjusting weights dynamically...")
    weights = adjust_weights(strategy_metrics, weights)

    # Output results
    results = {
        "signals": price_data[["timestamp", "close", "Signal"]].tail(),
        "portfolio_values": comparison_results[["timestamp", "Portfolio Value", "Buy and Hold Value"]],
        "strategy_metrics": strategy_metrics,
        "baseline_metrics": baseline_metrics,
        "adjusted_weights": weights,
    }

    return results


# Visualization Function
def visualize_results(results):
    """
    Visualize portfolio performance and trading signals.

    Parameters:
        results (dict): Results dictionary from the execute_strategy function.
    """
    portfolio_values = results["portfolio_values"]

    # Plot portfolio value comparison
    plt.figure(figsize=(12, 6))
    plt.plot(
        portfolio_values["timestamp"],
        portfolio_values["Portfolio Value"],
        label="Strategy Portfolio Value",
        color="blue",
    )
    plt.plot(
        portfolio_values["timestamp"],
        portfolio_values["Buy and Hold Value"],
        label="Buy and Hold Value",
        color="orange",
        linestyle="--",
    )
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid()
    plt.show()

    # Print strategy and baseline metrics
    print("\nStrategy Performance Metrics:")
    for metric, value in results["strategy_metrics"].items():
        print(f"{metric}: {value:.2f}")

    print("\nBaseline (Buy-and-Hold) Performance Metrics:")
    for metric, value in results["baseline_metrics"].items():
        print(f"{metric}: {value:.2f}")

    # Display trading signals
    print("\nRecent Trading Signals:")
    print(results["signals"])
    
# BLOCK 7: ERROR HANDLING, OPTIMIZATION, AND FINE-TUNING

# 1. Error Handling for Data Fetching
def safe_fetch_historical_data(symbol, interval="1d", limit=100):
    """
    Safely fetch historical price data with error handling.

    Parameters:
        symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
        interval (str): Time interval (e.g., '1d', '1h').
        limit (int): Number of data points to fetch.

    Returns:
        pd.DataFrame: Historical price data or empty DataFrame on failure.
    """
    try:
        data = fetch_historical_data(symbol, interval=interval, limit=limit)
        if data.empty:
            raise ValueError(f"No data returned for symbol {symbol}.")
        return data
    except Exception as e:
        logging.error(f"Failed to fetch historical data for {symbol}: {e}")
        return pd.DataFrame()


def safe_fetch_and_analyze_tweets(twitter_api, query, count=100):
    """
    Safely fetch and analyze tweets with error handling.

    Parameters:
        twitter_api (Tweepy API Object): Authenticated Tweepy API object.
        query (str): The search query (e.g., '#bitcoin').
        count (int): Number of tweets to fetch.

    Returns:
        float: Average sentiment score or 0.0 on failure.
    """
    try:
        sentiment_score = fetch_and_analyze_tweets(twitter_api, query, count=count)
        return sentiment_score
    except Exception as e:
        logging.error(f"Failed to fetch or analyze tweets for query '{query}': {e}")
        return 0.0


# 2. Optimizing Batch Processing for Sentiment Analysis
def batch_sentiment_analysis(tweets):
    """
    Perform sentiment analysis in batch for improved performance.

    Parameters:
        tweets (list): List of tweet texts.

    Returns:
        float: Average sentiment score for the batch.
    """
    try:
        cleaned_tweets = [
            re.sub(r"http\S+", "", tweet) for tweet in tweets  # Remove URLs
        ]
        cleaned_tweets = [
            re.sub(r"[^a-zA-Z ]", "", tweet).lower() for tweet in cleaned_tweets
        ]  # Remove special characters and lowercase

        # Perform batch sentiment analysis
        sentiment_scores = [TextBlob(tweet).sentiment.polarity for tweet in cleaned_tweets]
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            return avg_sentiment
        else:
            return 0.0
    except Exception as e:
        logging.error(f"Error during batch sentiment analysis: {e}")
        return 0.0


# 3. Fine-Tuning Parameters
def fine_tune_parameters(initial_weights, performance_metrics, learning_rate=0.01):
    """
    Fine-tune hyperparameters and weights dynamically.

    Parameters:
        initial_weights (dict): Current weights for components (e.g., 'sentiment', 'technical').
        performance_metrics (dict): Dictionary of performance metrics (e.g., Sharpe ratio, returns).
        learning_rate (float): Learning rate for adjustments.

    Returns:
        dict: Tuned weights for each component.
    """
    try:
        # Adjust weights based on performance
        tuned_weights = adjust_weights(performance_metrics, initial_weights, learning_rate)

        # Additional fine-tuning logic (e.g., clipping weights, adding constraints)
        for key in tuned_weights:
            tuned_weights[key] = max(0, min(1, tuned_weights[key]))  # Ensure weights are between 0 and 1

        logging.info(f"Fine-tuned weights: {tuned_weights}")
        return tuned_weights
    except Exception as e:
        logging.error(f"Error during parameter fine-tuning: {e}")
        return initial_weights  # Fallback to initial weights


# 4. Integrated Error Handling and Optimization in the Pipeline
def execute_robust_strategy(symbol, initial_balance=1000, lookback_window=14, sentiment_threshold=0.2):
    """
    Execute the full trading strategy pipeline with error handling and optimization.

    Parameters:
        symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
        initial_balance (float): Starting balance for backtesting.
        lookback_window (int): Lookback window for technical indicators.
        sentiment_threshold (float): Threshold for sentiment-based signal adjustment.

    Returns:
        dict: Results including trading signals, portfolio values, and performance metrics.
    """
    # Initialize weights for sentiment, technical analysis, and predictions
    weights = {"sentiment": 0.33, "technical": 0.33, "predictions": 0.33}

    # Safe fetching of historical price data
    logging.info("Fetching historical price data...")
    price_data = safe_fetch_historical_data(symbol, limit=100)
    if price_data.empty:
        logging.error("No historical data available. Exiting strategy execution.")
        return {}

    # Calculate technical indicators
    logging.info("Calculating technical indicators...")
    try:
        price_data = calculate_moving_averages(price_data, window=lookback_window)
        price_data = calculate_rsi(price_data, window=lookback_window)
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return {}

    # Fetch sentiment score safely
    logging.info("Performing sentiment analysis...")
    twitter_api = authenticate_twitter()
    sentiment_score = safe_fetch_and_analyze_tweets(twitter_api, query=f"#{symbol.lower()}", count=100)

    # Generate trading signals
    logging.info("Generating trading signals...")
    try:
        price_data = generate_trading_signals(price_data, sentiment_score, threshold=sentiment_threshold)
    except Exception as e:
        logging.error(f"Error generating trading signals: {e}")
        return {}

    # Backtest the strategy
    logging.info("Backtesting the strategy...")
    try:
        backtest_results, strategy_metrics = backtest_strategy(price_data, initial_balance=initial_balance)
    except Exception as e:
        logging.error(f"Error during backtesting: {e}")
        return {}

    # Compare to buy-and-hold baseline
    logging.info("Comparing to buy-and-hold baseline...")
    try:
        comparison_results, baseline_metrics = compare_to_baseline(backtest_results)
    except Exception as e:
        logging.error(f"Error comparing to baseline: {e}")
        return {}

    # Dynamically adjust weights based on strategy performance
    logging.info("Adjusting weights dynamically...")
    weights = fine_tune_parameters(weights, strategy_metrics)

    # Output results
    results = {
        "signals": price_data[["timestamp", "close", "Signal"]].tail(),
        "portfolio_values": comparison_results[["timestamp", "Portfolio Value", "Buy and Hold Value"]],
        "strategy_metrics": strategy_metrics,
        "baseline_metrics": baseline_metrics,
        "adjusted_weights": weights,
    }

    return results

# MODULE: DataHandler.py
class DataHandler:
    """
    Handles fetching and processing of historical price data.
    """

    @staticmethod
    def fetch_data(symbol, interval="1d", limit=100):
        """
        Fetch historical price data.

        Parameters:
            symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
            interval (str): Time interval (e.g., '1d', '1h').
            limit (int): Number of data points.

        Returns:
            pd.DataFrame: Historical price data or empty DataFrame on failure.
        """
        try:
            data = fetch_historical_data(symbol, interval=interval, limit=limit)
            if data.empty:
                raise ValueError(f"No data returned for symbol {symbol}.")
            return data
        except Exception as e:
            logging.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def calculate_indicators(data, lookback_window=14):
        """
        Calculate technical indicators (SMA, EMA, RSI).

        Parameters:
            data (pd.DataFrame): Price data.
            lookback_window (int): Lookback window for indicators.

        Returns:
            pd.DataFrame: Price data with indicators added.
        """
        try:
            data = calculate_moving_averages(data, window=lookback_window)
            data = calculate_rsi(data, window=lookback_window)
            return data
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            return pd.DataFrame()
        
# MODULE: SentimentHandler.py
class SentimentHandler:
    """
    Handles sentiment analysis using social media data.
    """

    def __init__(self, twitter_api):
        """
        Initialize with an authenticated Twitter API object.

        Parameters:
            twitter_api: Authenticated Tweepy API object.
        """
        self.twitter_api = twitter_api

    def fetch_sentiment(self, query, count=100):
        """
        Fetch sentiment score from tweets.

        Parameters:
            query (str): Search query (e.g., '#bitcoin').
            count (int): Number of tweets to fetch.

        Returns:
            float: Average sentiment score or 0.0 on failure.
        """
        try:
            return fetch_and_analyze_tweets(self.twitter_api, query, count=count)
        except Exception as e:
            logging.error(f"Failed to fetch sentiment for query {query}: {e}")
            return 0.0

    @staticmethod
    def batch_analyze_sentiment(tweets):
        """
        Perform sentiment analysis in batch.

        Parameters:
            tweets (list): List of tweet texts.

        Returns:
            float: Average sentiment score for the batch.
        """
        return batch_sentiment_analysis(tweets)

# MODULE: StrategyExecutor.py
class StrategyExecutor:
    """
    Executes the trading strategy pipeline.
    """

    def __init__(self, initial_balance=1000, lookback_window=14, sentiment_threshold=0.2):
        """
        Initialize the strategy executor with configurable parameters.

        Parameters:
            initial_balance (float): Starting balance for backtesting.
            lookback_window (int): Lookback window for technical indicators.
            sentiment_threshold (float): Threshold for sentiment-based signal adjustment.
        """
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.sentiment_threshold = sentiment_threshold
        self.weights = {"sentiment": 0.33, "technical": 0.33, "predictions": 0.33}

    def execute(self, symbol, twitter_api):
        """
        Execute the full trading strategy pipeline.

        Parameters:
            symbol (str): Cryptocurrency symbol.
            twitter_api: Authenticated Twitter API object.

        Returns:
            dict: Results including signals, portfolio values, and metrics.
        """
        try:
            # Fetch and process data
            price_data = DataHandler.fetch_data(symbol, limit=100)
            if price_data.empty:
                raise ValueError("No price data available.")

            price_data = DataHandler.calculate_indicators(price_data, self.lookback_window)

            # Perform sentiment analysis
            sentiment_handler = SentimentHandler(twitter_api)
            sentiment_score = sentiment_handler.fetch_sentiment(f"#{symbol.lower()}", count=100)

            # Generate signals
            price_data = generate_trading_signals(price_data, sentiment_score, self.sentiment_threshold)

            # Backtest the strategy
            backtest_results, strategy_metrics = backtest_strategy(price_data, self.initial_balance)

            # Compare to buy-and-hold baseline
            comparison_results, baseline_metrics = compare_to_baseline(backtest_results)

            # Adjust weights dynamically
            self.weights = fine_tune_parameters(self.weights, strategy_metrics)

            # Return results
            return {
                "signals": price_data[["timestamp", "close", "Signal"]].tail(),
                "portfolio_values": comparison_results[["timestamp", "Portfolio Value", "Buy and Hold Value"]],
                "strategy_metrics": strategy_metrics,
                "baseline_metrics": baseline_metrics,
                "adjusted_weights": self.weights,
            }
        except Exception as e:
            logging.error(f"Error executing strategy: {e}")
            return {}
# MODULE: PersistenceHandler.py
import pickle
import json

class PersistenceHandler:
    """
    Handles saving and loading of models and results.
    """

    @staticmethod
    def save_results(results, filename="results.json"):
        """
        Save results to a JSON file.

        Parameters:
            results (dict): Results dictionary to save.
            filename (str): File name to save the results.
        """
        try:
            with open(filename, "w") as file:
                json.dump(results, file, indent=4)
            logging.info(f"Results saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")

    @staticmethod
    def load_results(filename="results.json"):
        """
        Load results from a JSON file.

        Parameters:
            filename (str): File name to load the results from.

        Returns:
            dict: Loaded results dictionary.
        """
        try:
            with open(filename, "r") as file:
                results = json.load(file)
            logging.info(f"Results loaded from {filename}")
            return results
        except Exception as e:
            logging.error(f"Failed to load results: {e}")
            return {}

    @staticmethod
    def save_model(model, filename="model.pkl"):
        """
        Save a model to a pickle file.

        Parameters:
            model: Model object to save.
            filename (str): File name to save the model.
        """
        try:
            with open(filename, "wb") as file:
                pickle.dump(model, file)
            logging.info(f"Model saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")

    @staticmethod
    def load_model(filename="model.pkl"):
        """
        Load a model from a pickle file.

        Parameters:
            filename (str): File name to load the model from.

        Returns:
            object: Loaded model object.
        """
        try:
            with open(filename, "rb") as file:
                model = pickle.load(file)
            logging.info(f"Model loaded from {filename}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return None
        
# Update the StrategyExecutor class
class StrategyExecutor:
    """
    Executes the trading strategy pipeline.
    """

    def __init__(self, initial_balance=1000, lookback_window=14, sentiment_threshold=0.2):
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.sentiment_threshold = sentiment_threshold
        self.weights = {"sentiment": 0.33, "technical": 0.33, "predictions": 0.33}

    def execute(self, symbol, twitter_api):
        """
        Execute the full trading strategy pipeline.

        Parameters:
            symbol (str): Cryptocurrency symbol.
            twitter_api: Authenticated Twitter API object.

        Returns:
            dict: Results including signals, portfolio values, and metrics.
        """
        try:
            # Fetch and process data
            price_data = DataHandler.fetch_data(symbol, limit=100)
            price_data = DataHandler.calculate_indicators(price_data, self.lookback_window)

            # Perform sentiment analysis
            sentiment_handler = SentimentHandler(twitter_api)
            sentiment_score = sentiment_handler.fetch_sentiment(f"#{symbol.lower()}", count=100)

            # Calculate technical confidence
            technical_confidence = InvestmentSuretyMetric.calculate_technical_confidence(price_data)

            # Generate prediction confidence (placeholder: can be based on ML model's output)
            prediction_confidence = 0.8  # Example fixed value for now

            # Calculate ISM
            ism = InvestmentSuretyMetric.calculate_ism(
                self.weights,
                sentiment_score,
                technical_confidence,
                prediction_confidence,
            )

            logging.info(f"Investment Surety Metric (ISM): {ism:.2f}")

            # Generate trading signals
            price_data = generate_trading_signals(price_data, sentiment_score, self.sentiment_threshold)

            # Backtest the strategy
            backtest_results, strategy_metrics = backtest_strategy(price_data, self.initial_balance)

            # Compare to buy-and-hold baseline
            comparison_results, baseline_metrics = compare_to_baseline(backtest_results)

            # Adjust weights dynamically
            self.weights = fine_tune_parameters(self.weights, strategy_metrics)

            # Return results
            return {
                "signals": price_data[["timestamp", "close", "Signal"]].tail(),
                "portfolio_values": comparison_results[["timestamp", "Portfolio Value", "Buy and Hold Value"]],
                "strategy_metrics": strategy_metrics,
                "baseline_metrics": baseline_metrics,
                "adjusted_weights": self.weights,
                "ism": ism,  # Include ISM in the results
            }
        except Exception as e:
            logging.error(f"Error executing strategy: {e}")
            return {}