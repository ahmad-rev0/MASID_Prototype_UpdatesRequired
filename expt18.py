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
from textblob import TextBlob
import os
import tensorflow as tf
import yfinance as yf

# Alpaca API Configuration
ALPACA_API_KEY = "REDACTED_ALPACA_KEY_3"
ALPACA_SECRET_KEY = "REDACTED_ALPACA_SECRET_3"
BASE_URL = "https://paper-api.alpaca.markets"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

# Constants
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
INITIAL_BALANCE = 1000  # Starting portfolio balance
MINIMUM_TRADE_AMOUNT = 1  # Minimum trade allocation amount
DEBUG_FOLDER = "debug_dataframes"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Ensure Debug Folder Exists
if not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

# Save DataFrame to CSV for Debugging
def save_dataframe_to_csv(df, filename):
    filepath = os.path.join(DEBUG_FOLDER, filename)
    try:
        df.to_csv(filepath, index=True)
        logging.info(f"Saved DataFrame to {filepath}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to {filename}: {e}")
        
def execute_trade(ticker, quantity, side="buy"):
    """
    Execute a trade using Alpaca's paper trading API.
    """
    try:
        url = f"{BASE_URL}/v2/orders"
        data = {
            "symbol": ticker,
            "qty": quantity,
            "side": side,
            "type": "market",
            "time_in_force": "gtc",
        }
        response = requests.post(url, headers=HEADERS, json=data)

        if response.status_code in [200, 201]:
            logging.info(f"Successfully executed {side} order for {ticker}, quantity: {quantity}.")
            return response.json()
        else:
            logging.error(f"Error placing {side} order for {ticker}: {response.text}")
            return {}
    except Exception as e:
        logging.error(f"Error executing trade for {ticker}: {e}")
        return {}


def get_all_orders():
    """
    Fetch all orders from the Alpaca API.
    """
    try:
        url = f"{BASE_URL}/v2/orders"
        response = requests.get(url, headers=HEADERS)

        if response.status_code == 200:
            logging.info("Successfully fetched all orders.")
            return response.json()
        else:
            logging.error(f"Error fetching orders: {response.text}")
            return []
    except Exception as e:
        logging.error(f"Error fetching orders: {e}")
        return []


def get_portfolio_history():
    """
    Fetch the account portfolio history from the Alpaca API.
    """
    try:
        url = f"{BASE_URL}/v2/account/portfolio/history?intraday_reporting=market_hours&pnl_reset=per_day"
        response = requests.get(url, headers=HEADERS)

        if response.status_code == 200:
            logging.info("Successfully fetched portfolio history.")
            return response.json()
        else:
            logging.error(f"Error fetching portfolio history: {response.text}")
            return {}
    except Exception as e:
        logging.error(f"Error fetching portfolio history: {e}")
        return {}
    
class YahooFinanceAPI:
    @staticmethod
    def fetch_stock_data(tickers, start_date, end_date):
        """
        Fetch historical stock data from Yahoo Finance.
        """
        stock_data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date)
                if "Adj Close" in df.columns:
                    df = df.rename(columns={"Adj Close": "price"})
                save_dataframe_to_csv(df, f"{ticker}_stock_data.csv")
                stock_data[ticker] = df[["price"]]
            except Exception as e:
                logging.error(f"Error fetching stock data for {ticker}: {e}")
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
    @staticmethod
    def fetch_crypto_data(ticker, start_date, end_date):
        """
        Fetch historical cryptocurrency data from CoinGecko.
        """
        try:
            # Ensure the date range is within the last 365 days
            start_ts = int(max(datetime.strptime(start_date, "%Y-%m-%d").timestamp(), (datetime.now() - timedelta(days=365)).timestamp()))
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

            url = f"{COINGECKO_API_URL}/coins/{ticker}/market_chart/range"
            params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}
            response = requests.get(url, params=params)

            if not response.ok:
                logging.warning(f"Failed to fetch crypto data for {ticker}. Response: {response.text}")
                return pd.DataFrame()

            data = response.json()
            if "prices" not in data or not data["prices"]:
                logging.warning(f"No valid price data for crypto: {ticker}")
                return pd.DataFrame()

            prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
            prices.set_index("timestamp", inplace=True)
            save_dataframe_to_csv(prices, f"{ticker}_crypto_data.csv")
            return prices
        except Exception as e:
            logging.error(f"Error fetching crypto data for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_latest_price(ticker):
        """
        Fetch the latest price for a cryptocurrency using CoinGecko.
        """
        try:
            url = f"{COINGECKO_API_URL}/simple/price"
            params = {"ids": ticker, "vs_currencies": "usd"}
            response = requests.get(url, params=params)

            if not response.ok:
                logging.warning(f"Failed to fetch latest price for {ticker}. Response: {response.text}")
                return None

            data = response.json()
            if ticker in data and "usd" in data[ticker]:
                return float(data[ticker]["usd"])
            else:
                logging.warning(f"No valid price data for crypto: {ticker}")
                return None
        except Exception as e:
            logging.error(f"Error fetching latest price for {ticker}: {e}")
            return None
        
def generate_graph(data, title, yaxis_label):
    """
    Generate a Plotly graph for crypto or stock data.
    """
    try:
        if not data or all(df.empty for df in data.values()):
            logging.warning(f"No data available to plot for {title}.")
            return {
                "data": [],
                "layout": {
                    "title": {"text": f"{title} - No Data", "x": 0.5},
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": yaxis_label},
                    "template": "plotly_dark",
                },
            }

        graph_data = []
        for ticker, df in data.items():
            if not df.empty:
                graph_data.append(
                    go.Scatter(
                        x=df.index,
                        y=df["price"],
                        mode="lines",
                        name=ticker,
                        line=dict(width=2),
                    )
                )

        layout = go.Layout(
            title=title,
            xaxis={"title": "Date", "type": "date"},
            yaxis={"title": yaxis_label},
            template="plotly_dark",
        )
        return {"data": graph_data, "layout": layout}
    except Exception as e:
        logging.error(f"Error generating graph for {title}: {e}")
        return {"data": [], "layout": {"title": "Error"}}
def generate_strategy_comparison_graph(crypto_data, stock_data, trades):
    """
    Generate a graph comparing App's Strategy, Buy and Hold, and Risk Parity strategies.

    Parameters:
        crypto_data (dict): Dictionary of historical crypto data.
        stock_data (dict): Dictionary of historical stock data.
        trades (dict): Trades executed by the app's strategy.

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
            logging.error("No valid price data available for strategy comparison.")
            return {"data": [], "layout": {"title": "No Data"}}

        # Initialize strategy values
        initial_balance = INITIAL_BALANCE
        app_strategy_values = [initial_balance]  # App's Strategy
        buy_and_hold_values = [initial_balance]  # Buy & Hold Strategy
        risk_parity_values = [initial_balance]  # Risk Parity Strategy

        # Calculate weights for Buy & Hold strategy
        buy_and_hold_weights = {ticker: 1 / len(all_prices.columns) for ticker in all_prices.columns}

        # Calculate weights for Risk Parity strategy (inverse of volatility)
        volatility = all_prices.pct_change().std()
        risk_parity_weights = (1 / volatility).fillna(0)
        risk_parity_weights /= risk_parity_weights.sum()

        # Iterate through each day to calculate strategy values
        for i in range(1, len(all_prices)):
            # App's Strategy
            if trades:
                app_value = sum(
                    trade["quantity"] * all_prices.iloc[i][trade["ticker"]]
                    for trade in trades.values()
                )
                app_strategy_values.append(app_value)

            # Buy & Hold Strategy
            buy_and_hold_value = sum(
                buy_and_hold_weights[ticker] * all_prices.iloc[i][ticker] * initial_balance
                for ticker in all_prices.columns
            )
            buy_and_hold_values.append(buy_and_hold_value)

            # Risk Parity Strategy
            risk_parity_value = sum(
                risk_parity_weights[ticker] * all_prices.iloc[i][ticker] * initial_balance
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
            logging.warning("Insufficient data for LSTM model.")
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

        # Define and train the LSTM model
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
    
def calculate_technical_indicators(data):
    """
    Calculate technical indicators: SMA, EMA, and RSI.

    Parameters:
        data (pd.DataFrame): DataFrame with price data.

    Returns:
        dict: Dictionary of technical scores.
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
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)  # Add small value to avoid division by zero
        data["RSI"] = 100 - (100 / (1 + rs))

        indicators = {
            "SMA": data["SMA"].iloc[-1] if not data["SMA"].isna().all() else 0,
            "EMA": data["EMA"].iloc[-1] if not data["EMA"].isna().all() else 0,
            "RSI": data["RSI"].iloc[-1] / 100 if not data["RSI"].isna().all() else 0,
        }
        return indicators
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return {"SMA": 0, "EMA": 0, "RSI": 0}
    
def calculate_investment_surety(sentiment, technical_scores, ml_score):
    """
    Calculate the Investment Surety Index (ISI) using the provided metrics.

    Parameters:
        sentiment (float): Sentiment score (0 to 1).
        technical_scores (dict): Technical indicator scores (SMA, EMA, RSI).
        ml_score (float): Machine Learning prediction score (0 to 1).

    Returns:
        float: Investment Surety Index (ISI) (0 to 1).
    """
    try:
        # Combine all metrics into a weighted average
        metrics = [
            sentiment,
            technical_scores.get("SMA", 0),
            technical_scores.get("EMA", 0),
            technical_scores.get("RSI", 0),
            ml_score,
        ]

        # Normalize the ISI to fall between 0 and 1
        isi = np.mean(metrics)
        return max(0, min(1, isi))
    except Exception as e:
        logging.error(f"Error calculating Investment Surety Index: {e}")
        return 0
    
class TwitterSentimentAnalysis:
    """
    Perform sentiment analysis using a placeholder method (e.g., TextBlob).
    """

    @staticmethod
    def fetch_sentiment(ticker):
        """
        Fetch sentiment for a given ticker using dummy sentiment analysis.

        Parameters:
            ticker (str): Stock or cryptocurrency ticker.

        Returns:
            float: Sentiment score (-1 to 1).
        """
        try:
            # Placeholder sentiment analysis (you can integrate real APIs)
            tweets = [f"{ticker} is doing great!", f"{ticker} is terrible!"]
            sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
            return np.mean(sentiments) if sentiments else 0
        except Exception as e:
            logging.error(f"Error fetching sentiment for {ticker}: {e}")
            return 0
        
def simulate_trading(crypto_tickers, stock_tickers, start_date, end_date, threshold):
    """
    Simulate trading by fetching data, analyzing metrics, and executing trades via Alpaca.

    Parameters:
        crypto_tickers (list): List of cryptocurrency tickers (CoinGecko IDs).
        stock_tickers (list): List of stock tickers (Yahoo Finance tickers).
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        threshold (float): ISI threshold for buy/sell decisions.

    Returns:
        dict: Results of the simulation, including trades and portfolio metrics.
    """
    try:
        # Fetch historical data
        crypto_data = {ticker: CoinGeckoAPI.fetch_crypto_data(ticker, start_date, end_date) for ticker in crypto_tickers}
        stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, start_date, end_date)

        trades = {}
        for ticker in crypto_tickers + stock_tickers:
            asset_type = "crypto" if ticker in crypto_tickers else "stock"
            data = crypto_data.get(ticker, pd.DataFrame()) if asset_type == "crypto" else stock_data.get(ticker, pd.DataFrame())

            if data.empty:
                logging.warning(f"No valid data for {ticker}. Skipping.")
                continue

            # Calculate metrics
            sentiment = TwitterSentimentAnalysis.fetch_sentiment(ticker)
            technical_scores = calculate_technical_indicators(data)
            ml_score = train_lstm_model(data)

            # Calculate ISI
            isi = calculate_investment_surety(sentiment, technical_scores, ml_score)

            # Make a trading decision
            if isi >= threshold:
                latest_price = CoinGeckoAPI.fetch_latest_price(ticker) if asset_type == "crypto" else YahooFinanceAPI.fetch_latest_price(ticker)
                if latest_price is None or latest_price <= 0:
                    logging.warning(f"Skipping trade for {ticker}: Invalid price.")
                    continue

                quantity = max(MINIMUM_TRADE_AMOUNT, int(INITIAL_BALANCE // latest_price))
                if quantity > 0:
                    trade = execute_trade(ticker, quantity, side="buy")
                    if trade:
                        trades[ticker] = trade

        # Fetch portfolio metrics
        orders = get_all_orders()
        portfolio_history = get_portfolio_history()

        return {"trades": trades, "orders": orders, "portfolio_history": portfolio_history, "crypto_data": crypto_data, "stock_data": stock_data}
    except Exception as e:
        logging.error(f"Error during trading simulation: {e}")
        return {"trades": {}, "orders": [], "portfolio_history": {}, "crypto_data": {}, "stock_data": {}}
    
def create_dashboard():
    """
    Create a Dash GUI for AmbiVest with cryptocurrency and stock analysis.
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

    # App layout
    app.layout = dbc.Container(
        [
            # Header
            dbc.Row(
                dbc.Col(
                    html.H1(
                        "AmbiVest: Intelligent Investment Analysis",
                        style={"textAlign": "center", "color": "#FFD700", "padding": "20px"},
                    )
                ),
                style={"backgroundColor": "#1a1a1a", "borderBottom": "2px solid #FFD700"},
            ),

            # Threshold Slider
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Adjust ISI Threshold", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                [
                                    dcc.Slider(
                                        id="isi-threshold-slider",
                                        min=0,
                                        max=1,
                                        step=0.01,
                                        value=0.7,  # Default threshold
                                        marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11)},
                                    ),
                                    html.Div(id="threshold-value", style={"color": "white", "marginTop": "10px"}),
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
                                        dcc.Input(
                                            id="crypto-input",
                                            type="text",
                                            placeholder="Enter crypto tickers (e.g., bitcoin, ethereum)",
                                            style={"width": "100%", "marginBottom": "10px"},
                                        ),
                                        dcc.Input(
                                            id="stock-input",
                                            type="text",
                                            placeholder="Enter stock tickers (e.g., AAPL, MSFT)",
                                            style={"width": "100%", "marginBottom": "10px"},
                                        ),
                                        dcc.Input(
                                            id="start-date",
                                            type="text",
                                            placeholder="YYYY-MM-DD",
                                            style={"width": "100%", "marginBottom": "10px"},
                                        ),
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
                                dbc.CardBody(dcc.Graph(id="crypto-performance-trend")),
                            ],
                            style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Stock Performance", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(dcc.Graph(id="stock-performance-trend")),
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
            Output("trades-executed", "children"),
        ],
        [Input("run-button", "n_clicks"), Input("isi-threshold-slider", "value")],
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

            start_date = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")

            # Simulate trading
            results = simulate_trading(crypto_tickers, stock_tickers, start_date, end_date, threshold)

            trades = results["trades"]
            crypto_data = results["crypto_data"]
            stock_data = results["stock_data"]

            # Generate graphs
            crypto_fig = generate_graph(crypto_data, "Crypto Performance", "Price (USD)")
            stock_fig = generate_graph(stock_data, "Stock Performance", "Price (USD)")
            strategy_fig = generate_strategy_comparison_graph(crypto_data, stock_data, trades)

            # Format portfolio metrics
            equity_text = f"Portfolio Equity: ${INITIAL_BALANCE:.2f}"
            buying_power_text = f"Buying Power: ${INITIAL_BALANCE:.2f}"  # Placeholder
            pnl_text = f"PnL: ${0:.2f}"  # Placeholder

            trades_text = "\n".join([f"{trade['symbol']}: {trade['qty']} units ({trade['side']})" for trade in trades.values()])

            return crypto_fig, stock_fig, strategy_fig, equity_text, buying_power_text, pnl_text, trades_text
        except Exception as e:
            logging.error(f"Error updating dashboard: {e}")
            return {}, {}, {}, "Error", "Error", "Error", "Error"

    return app

if __name__ == "__main__":
    app = create_dashboard()

    # Define the port and URL
    port = 8050
    url = f"http://127.0.0.1:{port}"

    # Open the app in the default web browser
    import webbrowser
    webbrowser.open_new(url)

    # Run the Dash server
    app.run_server(debug=True, use_reloader=False, port=port)