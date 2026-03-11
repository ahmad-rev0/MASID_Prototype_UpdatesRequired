# Imports
import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from textblob import TextBlob
import yfinance as yf
import webbrowser

# Alpaca API Configuration
ALPACA_API_KEY = "YOUR_ALPACA_API_KEY"
ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

class CoinGeckoAPI:
    @staticmethod
    def fetch_crypto_data(ticker, start_date, end_date):
        """
        Fetch historical cryptocurrency data from CoinGecko.
        """
        try:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
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
    
def calculate_technical_indicators(data):
    """
    Calculate technical indicators: SMA, EMA, and RSI.

    Parameters:
        data (pd.DataFrame): DataFrame with price data.

    Returns:
        dict: Dictionary of technical indicator scores.
    """
    try:
        if data.empty or "price" not in data.columns:
            logging.warning("Data is empty or missing 'price' column.")
            return {"SMA": 0, "EMA": 0, "RSI": 0}

        # Simple Moving Average (SMA)
        data["SMA"] = data["price"].rolling(window=20).mean()

        # Exponential Moving Average (EMA)
        data["EMA"] = data["price"].ewm(span=20).mean()

        # Relative Strength Index (RSI)
        delta = data["price"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)  # Avoid division by zero
        data["RSI"] = 100 - (100 / (1 + rs))

        # Normalize indicators
        indicators = {
            "SMA": (data["SMA"].iloc[-1] - data["price"].min()) / (data["price"].max() - data["price"].min()),
            "EMA": (data["EMA"].iloc[-1] - data["price"].min()) / (data["price"].max() - data["price"].min()),
            "RSI": data["RSI"].iloc[-1] / 100,  # Normalize RSI to range [0, 1]
        }
        return indicators
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return {"SMA": 0, "EMA": 0, "RSI": 0}
    
class SentimentAnalysis:
    @staticmethod
    def fetch_sentiment(ticker):
        """
        Simulated sentiment analysis for a ticker.

        Parameters:
            ticker (str): The ticker symbol.

        Returns:
            float: Sentiment score between -1 and 1.
        """
        try:
            # Simulated sentiment score (replace with actual API calls if needed)
            sentiment_score = np.random.uniform(-1, 1)  # Random sentiment for demonstration
            return sentiment_score
        except Exception as e:
            logging.error(f"Error fetching sentiment for {ticker}: {e}")
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
def execute_trade(ticker, quantity, side="buy"):
    """
    Execute a trade using Alpaca's paper trading API.

    Parameters:
        ticker (str): Ticker symbol of the asset to trade.
        quantity (int): Quantity of the asset to trade.
        side (str): Trade operation - "buy" or "sell" (default: "buy").

    Returns:
        dict: Response from the Alpaca API on success or an empty dictionary on failure.
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
    Fetch all executed orders from the Alpaca API.

    Returns:
        list: List of executed orders fetched from Alpaca API.
    """
    try:
        url = f"{BASE_URL}/v2/orders"
        response = requests.get(url, headers=HEADERS)

        if response.status_code == 200:
            logging.info("Successfully fetched all executed orders.")
            return response.json()  # Return the list of orders
        else:
            logging.error(f"Error fetching orders: {response.text}")
            return []
    except Exception as e:
        logging.error(f"Error fetching orders: {e}")
        return []   
def simulate_trading(coingecko_tickers, stock_tickers, start_date, end_date, threshold):
    """
    Simulate trading by fetching data, analyzing metrics, and executing trades.

    Parameters:
        coingecko_tickers (list): List of cryptocurrency tickers (e.g., bitcoin, ethereum).
        stock_tickers (list): List of stock tickers (e.g., AAPL, MSFT).
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
        threshold (float): ISI threshold for buy/sell decisions.

    Returns:
        dict: Results of simulation with metrics, trades, and data.
    """
    try:
        # Fetch historical data
        crypto_data = {ticker: CoinGeckoAPI.fetch_crypto_data(ticker, start_date, end_date) for ticker in coingecko_tickers}
        stock_data = YahooFinanceAPI.fetch_stock_data(stock_tickers, start_date, end_date)

        trades = {}
        for ticker, data in {**crypto_data, **stock_data}.items():
            if data.empty:
                logging.warning(f"No valid data for {ticker}. Skipping.")
                continue

            # Calculate sentiment
            sentiment = SentimentAnalysis.fetch_sentiment(ticker)

            # Calculate technical indicators
            technical_scores = calculate_technical_indicators(data)

            # Get LSTM prediction
            ml_score = train_lstm_model(data)

            # Combine metrics into the Investment Surety Index (ISI)
            isi = np.mean([
                sentiment,
                technical_scores.get("SMA", 0),
                technical_scores.get("EMA", 0),
                technical_scores.get("RSI", 0),
                ml_score,
            ])

            # Trading decision
            if isi >= threshold:
                latest_price = CoinGeckoAPI.fetch_latest_price(ticker) if ticker in crypto_data else YahooFinanceAPI.fetch_latest_price(ticker)
                if latest_price is None or latest_price <= 0:
                    logging.warning(f"Skipping trade for {ticker}: Invalid price.")
                    continue

                quantity = max(MINIMUM_TRADE_AMOUNT, int(INITIAL_BALANCE // latest_price))
                if quantity > 0:
                    trade = execute_trade(ticker, quantity, side="buy")
                    if trade:
                        trades[ticker] = trade

        return {"crypto_data": crypto_data, "stock_data": stock_data, "trades": trades}
    except Exception as e:
        logging.error(f"Error during trading simulation: {e}")
        return {"crypto_data": {}, "stock_data": {}, "trades": {}}
    
def generate_graph(data, title, yaxis_label):
    """
    Generate a Plotly graph for crypto or stock data.

    Parameters:
        data (dict): Dictionary of DataFrames with price data.
        title (str): Title of the graph.
        yaxis_label (str): Label for the Y-axis.

    Returns:
        dict: Plotly figure.
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
        return {"data": [], "layout": {"title": "Error"}}\
            
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
                        "AmbiVest: Intelligent Investment Dashboard",
                        style={"textAlign": "center", "color": "#FFD700", "padding": "20px", "fontWeight": "bold"},
                    )
                ),
                style={"backgroundColor": "#1a1a1a", "borderBottom": "2px solid #FFD700"},
            ),

            # ISI Threshold Slider
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Adjust ISI Threshold", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                [
                                    html.Label("ISI Threshold", style={"color": "white"}),
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
                            ),
                        ],
                        style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                    )
                ),
                style={"marginTop": "20px"},
            ),

            # Input Section and Trade Execution
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Label("Cryptocurrencies (CoinGecko)", style={"color": "white"}),
                                    dcc.Input(
                                        id="crypto-coingecko-input",
                                        type="text",
                                        placeholder="Enter tickers (e.g., bitcoin, ethereum)",
                                        style={"width": "100%", "marginBottom": "10px"},
                                    ),
                                    html.Label("Stocks (Yahoo Finance)", style={"color": "white"}),
                                    dcc.Input(
                                        id="stock-input",
                                        type="text",
                                        placeholder="Enter stock tickers (e.g., AAPL, MSFT)",
                                        style={"width": "100%", "marginBottom": "10px"},
                                    ),
                                    html.Label("Trade Quantity", style={"color": "white"}),
                                    dcc.Input(
                                        id="quantity-input",
                                        type="number",
                                        placeholder="Enter quantity",
                                        style={"width": "100%", "marginBottom": "10px"},
                                    ),
                                    dbc.Button(
                                        "Execute Trade",
                                        id="trade-button",
                                        color="warning",
                                        style={"width": "100%", "marginTop": "10px"},
                                    ),
                                ]
                            )
                        ),
                        width=4,
                    ),

                    # Executed Trades Section
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Executed Trades", style={"color": "#FFD700", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    html.Div(id="executed-trades", style={"color": "white", "marginTop": "10px"}),
                                ),
                            ],
                            style={"backgroundColor": "#2a2a2a", "border": "1px solid #FFD700"},
                        ),
                        width=8,
                    ),
                ],
                style={"marginTop": "20px"},
            ),

            # Graph Section
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Performance Graphs", style={"color": "#FFD700", "fontWeight": "bold"}),
                            dbc.CardBody(
                                [
                                    dcc.Graph(id="crypto-performance-trend"),
                                    dcc.Graph(id="stock-performance-trend"),
                                ]
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

    @app.callback(
        Output("executed-trades", "children"),
        [
            Input("trade-button", "n_clicks"),
        ],
        [
            State("crypto-coingecko-input", "value"),
            State("stock-input", "value"),
            State("quantity-input", "value"),
        ],
    )
    def update_executed_trades(n_clicks, crypto_input, stock_input, quantity):
        """
        Execute trades and display all executed trades.

        Parameters:
            n_clicks (int): Number of clicks on the Execute Trade button.
            crypto_input (str): Cryptocurrencies to trade.
            stock_input (str): Stocks to trade.
            quantity (int): Quantity to trade.

        Returns:
            str: HTML content displaying executed trades.
        """
        if n_clicks is None or quantity is None:
            return "No trades executed yet."

        try:
            tickers = []
            if crypto_input:
                tickers.extend(crypto_input.split(","))
            if stock_input:
                tickers.extend(stock_input.split(","))
            tickers = [ticker.strip() for ticker in tickers]

            # Execute trades for each ticker
            for ticker in tickers:
                execute_trade(ticker, quantity)

            # Fetch all executed trades
            orders = get_all_orders()

            # Format orders for display
            if not orders:
                return "No trades executed yet."

            trade_details = []
            for order in orders:
                trade_details.append(
                    html.Div(
                        [
                            html.P(f"Ticker: {order['symbol']}"),
                            html.P(f"Quantity: {order['qty']}"),
                            html.P(f"Side: {order['side']}"),
                            html.P(f"Status: {order['status']}"),
                            html.Hr(style={"borderColor": "#FFD700"}),
                        ],
                        style={"marginBottom": "10px"},
                    )
                )

            return trade_details
        except Exception as e:
            logging.error(f"Error executing or displaying trades: {e}")
            return "An error occurred. Check logs for details."
    return app

if __name__ == "__main__":
    app = create_dashboard()
    app.run_server(debug=True)