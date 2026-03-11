# Imports
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from scipy.optimize import minimize
import yfinance as yf
import webbrowser
import os
import textblob as TextBlob

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
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Alpaca paper trading endpoint
ALPACA_API_KEY = "your_alpaca_api_key"  # Replace with your Alpaca API Key
ALPACA_SECRET_KEY = "your_alpaca_secret_key"  # Replace with your Alpaca Secret Key
RAPIDAPI_KEY = "REDACTED_RAPIDAPI_KEY"  # Replace with your RapidAPI key
RISK_FREE_RATE = 0.07365  # Risk-free rate for Sharpe ratio
INITIAL_BALANCE = 1000  # Initial portfolio balance
DEVELOPER_KEY = "econ3086@hkbu"  # Developer key for advanced metrics access


class PaperTradingAPI:
    """
    Wrapper for Alpaca API for paper trading.
    """

    @staticmethod
    def place_trade(ticker, quantity, side="buy"):
        """
        Place a paper trade using Alpaca API.

        Parameters:
            ticker (str): Asset ticker symbol (e.g., "AAPL" for stocks, "BTC/USD" for crypto).
            quantity (int): Number of shares/units to buy or sell.
            side (str): "buy" or "sell".

        Returns:
            dict: Order details or error message.
        """
        try:
            url = f"{ALPACA_BASE_URL}/v2/orders"
            headers = {
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            }
            order = {
                "symbol": ticker,
                "qty": quantity,
                "side": side,
                "type": "market",
                "time_in_force": "gtc",  # Good till canceled
            }

            response = requests.post(url, json=order, headers=headers)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logging.error(f"Error placing trade for {ticker}: {e}")
            return {"error": str(e)}

    @staticmethod
    def get_account_status():
        """
        Fetch Alpaca account details (e.g., equity, buying power).

        Returns:
            dict: Account details or error message.
        """
        try:
            url = f"{ALPACA_BASE_URL}/v2/account"
            headers = {
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logging.error(f"Error fetching account status: {e}")
            return {"error": str(e)}

    @staticmethod
    def get_portfolio_positions():
        """
        Fetch all current portfolio positions.

        Returns:
            list: List of portfolio positions or error message.
        """
        try:
            url = f"{ALPACA_BASE_URL}/v2/positions"
            headers = {
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logging.error(f"Error fetching portfolio positions: {e}")
            return {"error": str(e)}
        
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
                df = yf.download(ticker, start=start_date, end=end_date)

                if df.empty:
                    logging.warning(f"No data found for {ticker}.")
                    continue

                if "Adj Close" in df.columns:
                    df = df.rename(columns={"Adj Close": "price"})
                elif "Close" in df.columns:
                    df = df.rename(columns={"Close": "price"})
                else:
                    logging.warning(f"No 'Close' or 'Adj Close' column found for {ticker}.")
                    continue

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

        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()

            tweets = data.get("timeline", [])
            sentiments = [
                TextBlob(tweet.get("text", "")).sentiment.polarity
                for tweet in tweets
                if "text" in tweet and isinstance(tweet["text"], str)
            ]

            return np.mean(sentiments) if sentiments else 0

        except Exception as e:
            logging.error(f"Error fetching sentiment for {ticker}: {e}")
            return 0


def calculate_technical_indicators(data):
    """
    Calculate technical indicators for a given price DataFrame.

    Parameters:
        data (pd.DataFrame): Historical price data.

    Returns:
        dict: Technical indicator scores (SMA, EMA, RSI, DMAC).
    """
    try:
        if data.empty or "price" not in data.columns:
            logging.warning("Data is empty or missing 'price' column.")
            return {"SMA": 0, "EMA": 0, "RSI": 0, "DMAC": 0}

        # SMA, EMA
        data["SMA"] = data["price"].rolling(window=20).mean()
        data["EMA"] = data["price"].ewm(span=20).mean()

        # RSI
        delta = data["price"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data["RSI"] = 100 - (100 / (1 + rs))

        # DMAC
        data["SMA_short"] = data["price"].rolling(window=10).mean()
        data["SMA_long"] = data["price"].rolling(window=30).mean()
        data["DMAC"] = ((data["SMA_short"] - data["SMA_long"]) / data["price"]).fillna(0)

        # Normalize scores
        indicators = {
            "SMA": (data["SMA"].iloc[-1] - data["price"].min()) / (data["price"].max() - data["price"].min()),
            "EMA": (data["EMA"].iloc[-1] - data["price"].min()) / (data["price"].max() - data["price"].min()),
            "RSI": data["RSI"].iloc[-1] / 100,
            "DMAC": (data["DMAC"].iloc[-1] + 1) / 2,
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
    Simulate trades using the Alpaca API for paper trading.
    """

    @staticmethod
    def execute_trade(ticker, quantity, side="buy"):
        """
        Execute a trade using Alpaca's paper trading API.

        Parameters:
            ticker (str): Asset ticker symbol (e.g., "AAPL" for stocks or "BTC/USD" for crypto).
            quantity (int): Number of shares/units to buy or sell.
            side (str): "buy" or "sell".

        Returns:
            dict: Order details or error message.
        """
        try:
            trade = PaperTradingAPI.place_trade(ticker, quantity, side)
            if "error" in trade:
                logging.error(f"Trade failed for {ticker}: {trade['error']}")
                return {}
            return trade
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
        # Calculate total trade value
        total_trades_value = sum(trade["total_cost"] for trade in trades.values() if "total_cost" in trade)

        # Calculate portfolio equity and PnL
        portfolio_equity = initial_balance - total_trades_value
        realized_pnl = sum(
            (trade["quantity"] * trade["price"] - trade["total_cost"])
            for trade in trades.values()
            if "price" in trade and "quantity" in trade
        )

        # Remaining buying power
        buying_power = initial_balance - total_trades_value

        return {
            "equity": portfolio_equity,
            "buying_power": max(0, buying_power),
            "pnl": realized_pnl,
        }
    except Exception as e:
        logging.error(f"Error updating portfolio metrics: {e}")
        return {"equity": 0, "buying_power": 0, "pnl": 0}
    

    
def generate_crypto_graph(crypto_data):
    """
    Generate cryptocurrency performance graph using Plotly.

    Parameters:
        crypto_data (dict): Dictionary of DataFrames with crypto data.

    Returns:
        dict: Plotly figure for cryptocurrency performance.
    """
    try:
        data = []
        for ticker, df in crypto_data.items():
            if not df.empty:
                # Ensure the data has a continuous index
                df = df.resample("D").ffill()  # Fill missing dates
                data.append(
                    go.Scatter(
                        x=df.index,
                        y=df["price"],
                        mode="lines",
                        name=ticker,
                        line=dict(width=2),
                    )
                )

        if not data:
            logging.warning("No cryptocurrency data available to plot.")
            return {"data": [], "layout": {"title": "No Data"}}

        return {
            "data": data,
            "layout": go.Layout(
                title="Cryptocurrency Performance",
                xaxis={"title": "Date", "type": "date"},
                yaxis={"title": "Price (USD)"},
                template="plotly_dark",
                plot_bgcolor="#1a1a1a",
                paper_bgcolor="#1a1a1a",
            ),
        }
    except Exception as e:
        logging.error(f"Error generating cryptocurrency graph: {e}")
        return {"data": [], "layout": {"title": "Error"}}


def generate_stock_graph(stock_data):
    """
    Generate stock performance graph using Plotly.

    Parameters:
        stock_data (dict): Dictionary of DataFrames with stock data.

    Returns:
        dict: Plotly figure for stock performance.
    """
    try:
        data = []
        for ticker, df in stock_data.items():
            if not df.empty:
                # Ensure proper date alignment
                df = df.resample("D").ffill()  # Fill missing dates
                df["price"] = df["price"].round(2)  # Format price to 2 decimal places

                data.append(
                    go.Scatter(
                        x=df.index,
                        y=df["price"],
                        mode="lines",
                        name=ticker,
                        line=dict(width=2),
                    )
                )

        if not data:
            logging.warning("No stock data available to plot.")
            return {"data": [], "layout": {"title": "No Data"}}

        layout = go.Layout(
            title="Stock Performance",
            xaxis={
                "title": "Date",
                "type": "date",
                "tickformat": "%b %d, %Y",  # Format dates (e.g., Jan 01, 2024)
                "tickangle": -45,  # Rotate labels for clarity
            },
            yaxis={
                "title": "Price (USD)",
                "tickprefix": "$",  # Add dollar sign to Y-axis values
            },
            template="plotly_dark",
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
        )

        return {"data": data, "layout": layout}
    except Exception as e:
        logging.error(f"Error generating stock graph: {e}")
        return {"data": [], "layout": {"title": "Error"}}


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
                all_prices[ticker] = df["price"]

        # Ensure all rows have valid data and align on a common date range
        all_prices = all_prices.fillna(method="ffill").dropna()

        if all_prices.empty:
            logging.warning("No valid price data available for strategy comparison.")
            return {"data": [], "layout": {"title": "No Data"}}

        # Initialize strategy values
        app_strategy_values = [INITIAL_BALANCE]  # App's strategy (based on ISI)
        buy_and_hold_values = [INITIAL_BALANCE]  # Buy & Hold strategy
        risk_parity_values = [INITIAL_BALANCE]  # Risk Parity strategy

        # Calculate weights for Buy & Hold strategy
        buy_and_hold_weights = {ticker: 1 / len(all_prices.columns) for ticker in all_prices.columns}

        # Iterate through each date to calculate strategy values
        for i in range(1, len(all_prices)):
            # ISI-Based Strategy
            app_metric = sum(
                investment_surety_scores.get(ticker, 0) * all_prices.iloc[i][ticker]
                for ticker in all_prices.columns
            )
            app_strategy_values.append(app_metric)

            # Buy & Hold Strategy
            buy_and_hold_value = sum(
                buy_and_hold_weights[ticker] * all_prices.iloc[i][ticker] for ticker in all_prices.columns
            )
            buy_and_hold_values.append(buy_and_hold_value)

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
        return {"data": [], "layout": {"title": "Error"}}
    
def generate_isi_breakdown_pie_chart(weights):
    """
    Generate a pie chart for the Investment Surety Index (ISI) breakdown.

    Parameters:
        weights (dict): Optimized weights for ISI components.

    Returns:
        dict: Plotly figure for ISI breakdown.
    """
    try:
        # Filter out components with negligible contributions
        significant_weights = {k: v for k, v in weights.items() if v > 1e-2}  # Threshold for visibility

        if not significant_weights:
            logging.warning("No significant components found for ISI breakdown.")
            return {"data": [], "layout": {"title": "No significant components"}}

        # Generate the pie chart
        figure = {
            "data": [
                go.Pie(
                    labels=list(significant_weights.keys()),
                    values=list(significant_weights.values()),
                    textinfo="label+percent",
                    hole=0.4,
                )
            ],
            "layout": go.Layout(
                title="Investment Surety Index (ISI) Breakdown",
                template="plotly_dark",
                plot_bgcolor="#1a1a1a",
                paper_bgcolor="#1a1a1a",
            ),
        }
        return figure
    except Exception as e:
        logging.error(f"Error generating ISI breakdown pie chart: {e}")
        return {"data": [], "layout": {"title": "Error"}}

def simulate_trading(crypto_tickers, stock_tickers, start_date, end_date, threshold, initial_balance):
    """
    Simulate trading for a portfolio of cryptocurrencies and stocks based on the ISI.

    Parameters:
        crypto_tickers (list): List of cryptocurrency tickers (e.g., BTC/USD).
        stock_tickers (list): List of stock tickers (e.g., AAPL).
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        threshold (float): ISI threshold for buy/sell decisions.
        initial_balance (float): Initial portfolio balance.

    Returns:
        dict: Results of the simulation, including portfolio metrics and executed trades.
    """
    try:
        # Fetch historical data
        crypto_data = {
            ticker: CoinMarketCapAPI.fetch_crypto_data(ticker, start_date, end_date) for ticker in crypto_tickers
        }
        stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, start_date, end_date)

        # Fetch initial account details from Alpaca
        portfolio_metrics = PaperTradingAPI.get_account_status()
        if "error" in portfolio_metrics:
            logging.error("Error fetching account status from Alpaca.")
            return {"metrics": {}, "trades": {}, "crypto_data": {}, "stock_data": {}}

        # Initialize portfolio balance and trades dictionary
        buying_power = float(portfolio_metrics.get("buying_power", initial_balance))
        trades = {}

        # Process each ticker for trading decisions
        for ticker in crypto_tickers + stock_tickers:
            asset_type = "crypto" if ticker in crypto_tickers else "stock"

            # Get historical data for the ticker
            data = (
                crypto_data.get(ticker, pd.DataFrame())
                if asset_type == "crypto"
                else stock_data.get(ticker, pd.DataFrame())
            )

            if data.empty:
                logging.warning(f"No valid data found for {ticker}. Skipping.")
                continue

            # Calculate key metrics
            sentiment = TwitterSentimentAnalysis.fetch_sentiment(ticker)
            technical_scores = calculate_technical_indicators(data)
            ml_score = train_lstm_model(data)
            sharpe_ratio = calculate_sharpe_ratio(data)

            # Calculate ISI
            isi, _ = calculate_investment_surety(sentiment, technical_scores, ml_score, sharpe_ratio)

            # Make a trading decision based on ISI
            if isi >= threshold:
                # Determine the number of shares/units to buy based on buying power
                current_price = float(data["price"].iloc[-1])  # Use the latest price
                if current_price > 0:
                    trade_quantity = int(buying_power // current_price)
                    if trade_quantity > 0:
                        trade = SimulatedTrading.execute_trade(ticker, trade_quantity, side="buy")
                        if trade:
                            trades[ticker] = trade
                            buying_power -= trade["total_cost"]

        # Update portfolio metrics
        portfolio_metrics = PaperTradingAPI.get_account_status()

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
    # Initialize Dash app
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

    # App layout
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
                                dbc.CardHeader(
                                    "Cryptocurrency Performance",
                                    style={"color": "#FFD700", "fontWeight": "bold"},
                                ),
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
                                dbc.CardHeader(
                                    "Stock Performance",
                                    style={"color": "#FFD700", "fontWeight": "bold"},
                                ),
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
                            dbc.CardHeader(
                                "Strategy Comparison",
                                style={"color": "#FFD700", "fontWeight": "bold"},
                            ),
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
                            dbc.CardHeader(
                                "Detailed Metrics (Developer Access)",
                                style={"color": "#FFD700", "fontWeight": "bold"},
                            ),
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="investment-surety-breakdown"),
                                            html.Div(
                                                id="sentiment-analysis-values",
                                                style={"color": "white", "marginTop": "10px"},
                                            ),
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

    # Callbacks
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

            # Fetch data for the first ticker for breakdown as an example
            ticker = crypto_tickers[0] if crypto_tickers else (stock_tickers[0] if stock_tickers else None)
            if not ticker:
                logging.warning("No tickers provided.")
                return {"data": [], "layout": {"title": "No Data"}}

            asset_type = "crypto" if ticker in crypto_tickers else "stock"
            data = (
                CoinMarketCapAPI.fetch_crypto_data(ticker, start_date, end_date)
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
            return {"data": [], "layout": {"title": "Error"}}
        
    @app.callback(
        [
            Output("portfolio-equity", "children"),
            Output("portfolio-buying-power", "children"),
            Output("portfolio-pnl", "children"),
        ],
        [Input("run-button", "n_clicks")],
    )
    def update_portfolio_metrics(n_clicks):
        """
        Update portfolio metrics based on Alpaca account status.

        Parameters:
            n_clicks (int): Number of times the Run Analysis button is clicked.

        Returns:
            tuple: Updated equity, buying power, and PnL strings.
        """
        if n_clicks is None:
            return "Portfolio Equity: $0.00", "Buying Power: $0.00", "PnL: $0.00"

        try:
            account_status = PaperTradingAPI.get_account_status()
            if "error" in account_status:
                return "Error fetching equity", "Error fetching buying power", "Error fetching PnL"

            equity = float(account_status.get("equity", 0))
            buying_power = float(account_status.get("buying_power", 0))
            pnl = float(account_status.get("unrealized_pl", 0))

            return (
                f"Portfolio Equity: ${equity:.2f}",
                f"Buying Power: ${buying_power:.2f}",
                f"PnL: ${pnl:.2f}",
            )
        except Exception as e:
            logging.error(f"Error updating portfolio metrics: {e}")
            return "Error", "Error", "Error"
    return app



if __name__ == "__main__":
    app = create_dashboard()
    port = 8050
    webbrowser.open_new(f"http://127.0.0.1:{port}")
    app.run_server(debug=True, use_reloader=False, port=port)