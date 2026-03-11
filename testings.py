import yfinance as yf
import requests
import pandas as pd

# Function to get stock data using yfinance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to get cryptocurrency data from CoinGecko API
def get_crypto_data(crypto_id, vs_currency, days):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        # Extract prices into a DataFrame
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")  # Convert timestamp to datetime
        return prices
    else:
        print(f"Failed to fetch data from CoinGecko. Status Code: {response.status_code}")
        return pd.DataFrame()

# Main script
if __name__ == "__main__":
    # Get stock data
    ticker = "AAPL"  # Example: Apple Inc.
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    stock_data = get_stock_data(ticker, start_date, end_date)
    stock_csv_filename = f"{ticker}_stock_data.csv"
    stock_data.to_csv(stock_csv_filename)
    print(f"Stock data saved to {stock_csv_filename}")

    # Get crypto data
    crypto_id = "bitcoin"  # Example: Bitcoin
    vs_currency = "usd"
    days = "365"  # Past year's data
    crypto_data = get_crypto_data(crypto_id, vs_currency, days)
    crypto_csv_filename = f"{crypto_id}_crypto_data.csv"
    if not crypto_data.empty:
        crypto_data.to_csv(crypto_csv_filename, index=False)
        print(f"Crypto data saved to {crypto_csv_filename}")