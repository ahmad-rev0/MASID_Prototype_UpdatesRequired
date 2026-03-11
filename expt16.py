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
import textblob

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
COINMARKETCAP_API_URL = "https://pro-api.coinmarketcap.com/v1"
COINMARKETCAP_API_KEY = "REDACTED_COINGECKO_KEY"  # Replace with your API key
RAPIDAPI_KEY = "REDACTED_RAPIDAPI_KEY"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = "REDACTED_ALPACA_KEY_5"  # Replace with your Alpaca API key
ALPACA_SECRET_KEY = "REDACTED_ALPACA_SECRET_5"

RISK_FREE_RATE = 0.07365  # Risk-free rate for Sharpe ratio
INITIAL_BALANCE = 1000  # Initial portfolio balance

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

                stock_data[ticker] = df[["price"]]

            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {e}")

        return stock_data
    
class CoinMarketCapAPI:
    """
    Wrapper for CoinMarketCap API for cryptocurrency data.
    """

    @staticmethod
    def fetch_crypto_data(symbol, start_date, end_date):
        """
        Fetch historical cryptocurrency data from CoinMarketCap.

        Parameters:
            symbol (str): Cryptocurrency symbol (e.g., BTC, ETH).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame with historical price data.
        """
        try:
            # Convert dates to UNIX timestamps
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

            url = f"{COINMARKETCAP_API_URL}/cryptocurrency/quotes/historical"
            headers = {
                "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY,
                "Accept": "application/json",
            }
            params = {
                "symbol": symbol,
                "time_start": start_timestamp,
                "time_end": end_timestamp,
                "interval": "daily",  # Fetch daily data
            }

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            if "data" not in data or "quotes" not in data["data"]:
                logging.warning(f"No data found for {symbol}. Response: {data}")
                return pd.DataFrame()

            prices = [
                {"timestamp": quote["time_open"], "price": quote["quote"]["USD"]["close"]}
                for quote in data["data"]["quotes"]
            ]
            prices_df = pd.DataFrame(prices)
            prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"])
            prices_df.set_index("timestamp", inplace=True)

            return prices_df

        except Exception as e:
            logging.error(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()
        
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
                textblob(tweet.get("text", "")).sentiment.polarity
                for tweet in tweets
                if "text" in tweet and isinstance(tweet["text"], str)
            ]

            return np.mean(sentiments) if sentiments else 0

        except Exception as e:
            logging.error(f"Error fetching sentiment for {ticker}: {e}")
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

        # Initialize trades and portfolio buying power
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
    
def update_portfolio_metrics():
    """
    Fetch and update portfolio metrics from Alpaca.

    Returns:
        dict: Updated portfolio metrics, including equity, buying power, and PnL.
    """
    try:
        account_status = PaperTradingAPI.get_account_status()
        if "error" in account_status:
            logging.error("Error fetching portfolio metrics: Alpaca API issue.")
            return {"equity": 0, "buying_power": 0, "pnl": 0}

        equity = float(account_status.get("equity", 0))
        buying_power = float(account_status.get("buying_power", 0))
        pnl = float(account_status.get("unrealized_pl", 0))

        return {
            "equity": equity,
            "buying_power": buying_power,
            "pnl": pnl,
        }
    except Exception as e:
        logging.error(f"Error updating portfolio metrics: {e}")
        return {"equity": 0, "buying_power": 0, "pnl": 0}

def generate_isi_breakdown_pie_chart(weights):
    """
    Generate a pie chart for the ISI breakdown.

    Parameters:
        weights (dict): Optimized weights for ISI components.

    Returns:
        dict: Plotly figure for ISI breakdown.
    """
    try:
        # Filter out components with negligible contributions
        significant_weights = {k: v for k, v in weights.items() if v > 0.01}  # Threshold for visibility

        if not significant_weights:
            logging.warning("No significant components found for ISI breakdown.")
            return {"data": [], "layout": {"title": "No significant components"}}

        # Generate pie chart
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
    
def generate_performance_graph(data, title):
    """
    Generate a performance graph for cryptocurrency or stock data.

    Parameters:
        data (dict): Dictionary of DataFrames with historical price data.
        title (str): Title of the graph.

    Returns:
        dict: Plotly figure for performance trend.
    """
    try:
        traces = []
        for ticker, df in data.items():
            if not df.empty:
                traces.append(
                    go.Scatter(
                        x=df.index,
                        y=df["price"],
                        mode="lines",
                        name=ticker,
                        line=dict(width=2),
                    )
                )

        if not traces:
            logging.warning("No data available to plot.")
            return {"data": [], "layout": {"title": "No Data"}}

        layout = go.Layout(
            title=title,
            xaxis={"title": "Date", "type": "date"},
            yaxis={"title": "Price (USD)"},
            template="plotly_dark",
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
        )

        return {"data": traces, "layout": layout}
    except Exception as e:
        logging.error(f"Error generating performance graph: {e}")
        return {"data": [], "layout": {"title": "Error"}}
    
def create_dashboard():
    """
    Create a Dash GUI for AmbiVest with cryptocurrency and stock analysis.
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

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
                        },
                    )
                )
            ),

            # ISI Threshold Slider
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Label("ISI Threshold:", style={"color": "white"}),
                                dcc.Slider(
                                    id="isi-threshold-slider",
                                    min=0,
                                    max=1,
                                    step=0.01,
                                    value=0.7,
                                    marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11)},
                                ),
                                html.Div(id="threshold-value", style={"color": "white", "marginTop": "10px"}),
                            ]
                        )
                    )
                )
            ),

            # Input Section
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Label("Cryptocurrency Tickers:", style={"color": "white"}),
                                    dcc.Input(
                                        id="crypto-input",
                                        type="text",
                                        placeholder="e.g., bitcoin, ethereum",
                                        style={"width": "100%", "marginBottom": "10px"},
                                    ),
                                    html.Label("Stock Tickers:", style={"color": "white"}),
                                    dcc.Input(
                                        id="stock-input",
                                        type="text",
                                        placeholder="e.g., AAPL, MSFT",
                                        style={"width": "100%", "marginBottom": "10px"},
                                    ),
                                    html.Label("Start Date:", style={"color": "white"}),
                                    dcc.Input(
                                        id="start-date",
                                        type="text",
                                        placeholder="YYYY-MM-DD",
                                        style={"width": "100%", "marginBottom": "10px"},
                                    ),
                                    html.Label("End Date:", style={"color": "white"}),
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
                            )
                        )
                    )
                ]
            ),

            # Portfolio Metrics Section
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Portfolio Metrics", style={"color": "#FFD700"}),
                                html.P(id="portfolio-equity", style={"color": "white"}),
                                html.P(id="portfolio-buying-power", style={"color": "white"}),
                                html.P(id="portfolio-pnl", style={"color": "white"}),
                            ]
                        )
                    )
                )
            ),

            # Graphs Section
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="crypto-performance-trend"), width=6),
                    dbc.Col(dcc.Graph(id="stock-performance-trend"), width=6),
                ]
            ),

            # ISI Breakdown
            dbc.Row(dcc.Graph(id="isi-breakdown")),
        ],
        fluid=True,
        style={"backgroundColor": "#121212", "padding": "20px"},
    )
    
    @app.callback(
        Output("threshold-value", "children"),
        Input("isi-threshold-slider", "value"),
    )
    def update_threshold_display(threshold):
        """
        Update the displayed ISI threshold value when the slider is adjusted.

        Parameters:
            threshold (float): Current ISI threshold slider value.

        Returns:
            str: Displayed threshold value.
        """
        return f"Current ISI Threshold: {threshold:.2f}"

    @app.callback(
        [
            Output("crypto-performance-trend", "figure"),
            Output("stock-performance-trend", "figure"),
            Output("isi-breakdown", "figure"),
            Output("portfolio-equity", "children"),
            Output("portfolio-buying-power", "children"),
            Output("portfolio-pnl", "children"),
        ],
        [Input("run-button", "n_clicks")],
        [
            State("crypto-input", "value"),
            State("stock-input", "value"),
            State("start-date", "value"),
            State("end-date", "value"),
            State("isi-threshold-slider", "value"),
        ],
    )
    def run_simulation(
        n_clicks, crypto_input, stock_input, start_date, end_date, threshold
    ):
        """
        Run the trading simulation and update the dashboard.

        Parameters:
            n_clicks (int): Number of clicks on the "Run Analysis" button.
            crypto_input (str): Comma-separated cryptocurrency tickers.
            stock_input (str): Comma-separated stock tickers.
            start_date (str): Start date for analysis (YYYY-MM-DD).
            end_date (str): End date for analysis (YYYY-MM-DD).
            threshold (float): ISI threshold for buy/sell decisions.

        Returns:
            tuple: Updated graphs, portfolio metrics, and trade summaries.
        """
        if n_clicks is None:
            # If no clicks, return empty figures and default values
            return (
                {"data": [], "layout": {"title": "No Data"}},
                {"data": [], "layout": {"title": "No Data"}},
                {"data": [], "layout": {"title": "No Data"}},
                "Portfolio Equity: $0.00",
                "Buying Power: $0.00",
                "PnL: $0.00",
            )

        try:
            # Parse inputs
            crypto_tickers = [ticker.strip() for ticker in crypto_input.split(",")] if crypto_input else []
            stock_tickers = [ticker.strip() for ticker in stock_input.split(",")] if stock_input else []

            # Default date range
            start_date = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")

            # Run the trading simulation
            results = simulate_trading(crypto_tickers, stock_tickers, start_date, end_date, threshold, INITIAL_BALANCE)

            # Extract results
            portfolio_metrics = results["metrics"]
            trades = results["trades"]
            crypto_data = results["crypto_data"]
            stock_data = results["stock_data"]

            # Generate graphs
            crypto_fig = generate_performance_graph(crypto_data, "Cryptocurrency Performance")
            stock_fig = generate_performance_graph(stock_data, "Stock Performance")

            # Generate ISI breakdown chart
            if trades:
                first_trade = list(trades.values())[0]  # Example trade for ISI breakdown
                isi_weights = first_trade.get("isi_weights", {})
                isi_fig = generate_isi_breakdown_pie_chart(isi_weights)
            else:
                isi_fig = {"data": [], "layout": {"title": "No ISI Breakdown"}}

            # Format portfolio metrics
            equity_text = f"Portfolio Equity: ${float(portfolio_metrics.get('equity', 0)):.2f}"
            buying_power_text = f"Buying Power: ${float(portfolio_metrics.get('buying_power', 0)):.2f}"
            pnl_text = f"PnL: ${float(portfolio_metrics.get('pnl', 0)):.2f}"

            return (
                crypto_fig,
                stock_fig,
                isi_fig,
                equity_text,
                buying_power_text,
                pnl_text,
            )

        except Exception as e:
            logging.error(f"Error running simulation: {e}")
            return (
                {"data": [], "layout": {"title": "Error"}},
                {"data": [], "layout": {"title": "Error"}},
                {"data": [], "layout": {"title": "Error"}},
                "Portfolio Equity: $0.00",
                "Buying Power: $0.00",
                "PnL: $0.00",
            )
    return app
if __name__ == "__main__":
    app = create_dashboard()
    port = 8050  # Default Dash port
    webbrowser.open_new(f"http://127.0.0.1:{port}")
    app.run_server(debug=True, use_reloader=False, port=port)