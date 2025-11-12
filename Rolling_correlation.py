import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def calculate_correlation_signals(data: pd.DataFrame, asset1: str, asset2: str, window: int=50, wide_window: int=100, std_factor: float=3.0):

    # Validate column names
    for col in [asset1,asset2]:
        if col not in data.columns:
            raise ValueError(f"Column {col} not found in DataFrame.")
    
    df = data.copy()

    # Compute rolling correlation using the short window
    df["rolling_corr"] = df[asset1].rolling(window=window, min_periods=window).corr(df[asset2])

    # Compute wide-window stats (average and std dev of correlation)
    df["avg_corr"] = df["rolling_corr"].rolling(window=wide_window, min_periods=10).mean()
    df["std_corr"] = df["rolling_corr"].rolling(window=wide_window, min_periods=10).std()

    # Compute upper and lower thresholds using std_factor
    df["upper threshold"] = df["avg_corr"] + (std_factor * df["std_corr"])
    df["lower threshold"] = df["avg_corr"] - (std_factor * df["std_corr"])

    # Initialize signals to 0
    df["signal"] = 0

    # If correlation is above the upper threshold, set signal to -1 (expecting revert downward)
    df.loc[df["rolling_corr"] > df["upper threshold"], "signal"] = -1

    # If correlation is below the lower threshold, set signal to 2 (expecting revert upwards)
    df.loc[df["rolling_corr"] < df["lower threshold"], "signal"] = 2

    # Return only the relevant columns
    return df[["rolling_corr", "avg_corr", "upper threshold", "lower threshold", "signal"]]

def plot_correlation_signals(df: pd.DataFrame, start_idx: int=None, end_idx: int=None):

    # Filter data based on the given index range
    if start_idx is not None and end_idx is not None:
        df = df.iloc[start_idx:end_idx]

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["rolling_corr"], label="Rolling Correlation", color="blue")
    plt.plot(df.index, df["avg_corr"], label="Average Correlation", linestyle="--", color="green")
    plt.fill_between(df.index, df["lower threshold"], df["upper threshold"], color="gray", alpha=0.2, label="Threshold Range")

    # Highlight buy/sell signals
    buy_signals = df[df["signal"] == 2]
    sell_signals = df[df["signal"] == -1]
    plt.scatter(buy_signals.index, buy_signals["rolling_corr"], color="red", marker="^", label="Lower Signal", alpha=0.7)
    plt.scatter(sell_signals.index, sell_signals["rolling_corr"], color="black", marker="v", label="Upper Signal", alpha=0.7)

    plt.title("Rolling Correlation and Trading Signals")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_asset_prices(data: pd.DataFrame, start_idx: int=None, end_idx: int=None):
    
    # Filter data based on given index range
    if start_idx is not None and end_idx is not None:
        data = data.iloc[start_idx:end_idx]

    plt.figure(figsize=(12,6))
    plt.plot(data.index, data.iloc[:,0], label=data.columns[0], color="blue")
    plt.plot(data.index, data.iloc[:,1], label=data.columns[1], color="green")
    plt.title("Asset Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_combined_data(price_data: pd.DataFrame, correlation_data: pd.DataFrame, start_idx: int=None, end_idx: int=None):

    if start_idx is not None and end_idx is not None:
        price_data = price_data.iloc[start_idx:end_idx]
        correlation_data = correlation_data.iloc[start_idx:end_idx]
    
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Plot asset prices
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color="black")
    ax1.plot(price_data.index, price_data.iloc[:,0], label=price_data.columns[0], color="blue")
    ax1.plot(price_data.index, price_data.iloc[:,1], label=price_data.columns[1], color="green")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # Create a second y-axis for the correlation data
    ax2 = ax1.twinx()
    ax2.set_ylabel("Correlation", color="Purple")
    ax2.plot(correlation_data.index, correlation_data["rolling_corr"], label="Rolling Correlation", color="Purple")
    ax2.plot(correlation_data.index, correlation_data["avg_corr"], label="Average Correlation", linestyle="--", color="orange")
    ax2.fill_between(correlation_data.index, correlation_data["lower threshold"], correlation_data["upper threshold"], color="gray", alpha=0.2, label="Threshold Range")
    ax2.tick_params(axis="y", labelcolor="purple")

    # Highlight buy/sell signals
    buy_signals = correlation_data[correlation_data["signal"] == 2]
    sell_signals = correlation_data[correlation_data["signal"] == -1]
    plt.scatter(buy_signals.index, buy_signals["rolling_corr"], color="red", marker="^", label="Lower Signal", alpha=0.7)
    plt.scatter(sell_signals.index, sell_signals["rolling_corr"], color="black", marker="v", label="Upper Signal", alpha=0.7)

    ax2.legend(loc="upper right")

    plt.title("Asset Prices and Rolling Correlation")
    plt.legend()
    plt.show()

def fetch_stock_data(ticker, max_period="max"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=max_period, interval="1d")
    return data["Close"]

# Fetch the maximum duration of daily closing prices
asset1_ticker = "JPM"
asset2_ticker = "BAC"

asset1_prices = fetch_stock_data(asset1_ticker)
asset2_prices = fetch_stock_data(asset2_ticker)

# Align both datasets to the same date range
data = pd.concat([asset1_prices, asset2_prices], axis=1, join="inner")
data.columns = [asset1_ticker, asset2_ticker]

# Calculate correlation signals
result = calculate_correlation_signals(data, asset1_ticker, asset2_ticker, window=50, wide_window=100, std_factor=1)

# Plot correlation signals
plot_correlation_signals(result, start_idx=200, end_idx=400)

# Plot asset prices
plot_asset_prices(data, start_idx=200, end_idx=400)

# Combine both plots
plot_combined_data(data, result, start_idx=200, end_idx=400)