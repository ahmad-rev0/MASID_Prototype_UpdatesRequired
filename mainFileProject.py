import subprocess
bat_file_path = "pip_installs.bat"
subprocess.run([bat_file_path], shell=True)

from datetime import datetime, timedelta
import logging
import time
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import requests
import tkinter as tk
from tkinter import scrolledtext, messagebox
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from textblob import TextBlob
from scipy.optimize import minimize  # Add this for weight optimization

# Logging setup
logging.basicConfig(
    filename="trading_bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Helper functions
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data using yfinance."""
    valid_data = {}
    tickers = [ticker.strip() for ticker in tickers]  # Remove accidental whitespace
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

# Define your API key here
API_KEY = "REDACTED_COINGECKO_KEY"  # Replace with your actual API key

def fetch_crypto_data(tickers, start_date, end_date):
    """
    Fetch cryptocurrency data from the CoinGecko API with robust error handling.
    """
    crypto_data = {}
    valid_tickers = get_valid_crypto_ids()
    

    base_url = "https://api.coingecko.com/api/v3/coins/{ticker}/market_chart/range?x_cg_demo_api_key=REDACTED_COINGECKO_KEY"

    headers = {"accept": "application/json"}

    headers = {
        "accept": "application/json/",
        "?x-cg-pro-api-key": API_KEY  # Include the API key in headers
    }

    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Ensure date range is within 365 days
    if (end_dt - start_dt).days > 365:
        logging.warning("Date range exceeds 365 days. Truncating to the last 365 days.")
        start_dt = end_dt - timedelta(days=365)

    for ticker in tickers:
        if ticker not in valid_tickers:
            logging.warning(f"Invalid ticker: {ticker}. Skipping...")
            continue

        try:
            logging.info(f"Fetching crypto data for {ticker} from {start_date} to {end_date}...")
            url = base_url.format(ticker=ticker)
            params = {
                "vs_currency": "usd",
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp())
            }

            # Make the request with retries
            response = make_request_with_retries(url, params, headers)
            if response is None or response.status_code != 200:
                logging.warning(f"Failed to fetch data for {ticker}. Status code: {response.status_code if response else 'N/A'}")
                continue

            data = response.json()
            if 'prices' not in data or not data['prices']:
                logging.warning(f"No price data found for {ticker}. Response: {data}")
                continue

            # Convert prices to DataFrame
            prices = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)
            crypto_data[ticker] = prices

        except Exception as e:
            logging.error(f"Error fetching crypto data for {ticker}: {e}")
    
    return crypto_data


def make_request_with_retries(url, params, headers, max_retries=5):
    """
    Make an API request with retries and robust error handling.
    """
    wait_time = 1  # Start with 1 second
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 429:  # Rate limit exceeded
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 32)  # Exponential backoff
            else:
                return response
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 32)  # Exponential backoff
    logging.error(f"Failed after {max_retries} retries.")
    return None


def get_valid_crypto_ids():
    """
    Fetch the list of valid cryptocurrency IDs from CoinGecko.
    """


    url = "https://api.coingecko.com/api/v3/coins/list"

    headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": "REDACTED_COINGECKO_KEY"
    }


    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            coins = response.json()
            return [coin['id'] for coin in coins]
        else:
            logging.error(f"Failed to fetch valid crypto IDs. Status code: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Error fetching valid crypto IDs: {e}")
        return []



def fetch_tweets_sentiment(accounts, tickers):
    """Fetch tweets and calculate sentiment using TextBlob."""
    sentiments = {}
    for account in accounts:
        try:
            logging.info(f"Fetching tweets for {account}...")
            # Simulate fetching tweets (replace this with actual Twitter API calls if available)
            tweets = [f"Example tweet about {ticker}" for ticker in tickers]
            sentiment_score = np.mean([TextBlob(tweet).sentiment.polarity for tweet in tweets])
            sentiments[account] = sentiment_score
        except Exception as e:
            logging.error(f"Error fetching tweets for {account}: {e}")
    return sentiments

def train_deep_learning_model(data):
    """Train a deep learning model on stock/crypto data."""
    try:
        data = data['price'].values.reshape(-1, 1)  # Use 'price' column for crypto or 'Close' for stocks
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

def calculate_technical_indicators(data):
    """Calculate technical indicators for stock/crypto data."""
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

def calculate_investment_surety(sentiment_score, technical_score, ml_prediction, sharpe_ratio, weights):
    """
    Calculate the investment surety score using a weighted sum of various metrics.
    """
    return (weights[0] * sentiment_score + 
            weights[1] * technical_score + 
            weights[2] * ml_prediction + 
            weights[3] * sharpe_ratio)

def optimize_weights(data, returns):
    """
    Optimize the weights for the investment surety metric to maximize returns.
    """
    def objective_function(weights):
        # Normalize weights to ensure they sum to 1
        weights = weights / np.sum(weights)
        investment_scores = data.apply(
            lambda row: calculate_investment_surety(row['sentiment'], row['technical'], row['ml'], row['sharpe'], weights), axis=1
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

def run_trading_bot():
    """Main function to run the trading bot."""
    try:
        stock_tickers = ["AAPL", "MSFT", "GOOG"]
        crypto_tickers = ["bitcoin", "ethereum", "dogecoin"]
        twitter_accounts = ["elonmusk", "cryptonews"]
        
        start_date = "2024-01-01"
        end_date = "2024-03-29"
        
        stock_data = fetch_stock_data(stock_tickers, start_date, end_date)
        crypto_data = fetch_crypto_data(crypto_tickers, start_date, end_date)
        sentiment_scores = fetch_tweets_sentiment(twitter_accounts, stock_tickers + crypto_tickers)

        # Prepare data for optimization
        all_data = []
        returns = []

        for ticker in crypto_tickers:
            model, scaler = train_deep_learning_model(crypto_data[ticker])
            technicals = calculate_technical_indicators(crypto_data[ticker])
            ml_prediction = 0
            if model:
                last_60_days = crypto_data[ticker]['price'].values[-60:].reshape(-1, 1)
                last_60_days_scaled = scaler.transform(last_60_days)
                X_test = np.array([last_60_days_scaled])
                ml_prediction = model.predict(X_test)[0][0]
            
            # Example Sharpe ratio (replace with actual calculation if needed)
            sharpe_ratio = 1.2  

            # Collect data for optimization
            all_data.append({
                'sentiment': sentiment_scores.get(ticker, 0),
                'technical': technicals.get('SMA', 0),
                'ml': ml_prediction,
                'sharpe': sharpe_ratio
            })
            returns.append(crypto_data[ticker]['price'].pct_change().sum())  # Example return calculation

        data_df = pd.DataFrame(all_data)

        # Optimize weights
        optimized_weights = optimize_weights(data_df, returns)

        # Calculate final investment surety scores
        data_df['investment_surety'] = data_df.apply(
            lambda row: calculate_investment_surety(row['sentiment'], row['technical'], row['ml'], row['sharpe'], optimized_weights), axis=1
        )

        logging.info("Trading bot completed successfully!")
        return data_df.to_dict('records')
    except Exception as e:
        logging.error(f"Error running trading bot: {e}")
        return None

# GUI Setup
def setup_gui():
    """Setup the graphical user interface for the trading bot."""
    def run_bot():
        try:
            results = run_trading_bot()
            if results:
                output_box.delete(1.0, tk.END)
                output_box.insert(tk.END, "Trading Bot Results:\n")
                for result in results:
                    output_box.insert(
                        tk.END,
                        f"Sentiment: {result['sentiment']:.2f}\n"
                        f"Technical Score: {result['technical']:.2f}\n"
                        f"ML Prediction: {result['ml']:.2f}\n"
                        f"Sharpe Ratio: {result['sharpe']:.2f}\n"
                        f"Investment Surety: {result['investment_surety']:.2f}\n\n"
                    )
            else:
                messagebox.showerror("Error", "Failed to run the trading bot. Check logs for details.")
        except Exception as e:
            logging.error(f"Error running the bot: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    root = tk.Tk()
    root.title("Trading Bot")
    root.geometry("800x600")
    tk.Label(root, text="Trading Bot", font=("Helvetica", 16)).pack(pady=10)
    tk.Button(root, text="Run Trading Bot", font=("Helvetica", 12), command=run_bot).pack(pady=10)
    output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=30, font=("Helvetica", 10))
    output_box.pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    setup_gui()