import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---- Parameters ----
Tickers = ["SBUX", "NVDA", "LMT", "UNH"]
Bench = "^GSPC"
Start = "2015-01-01"
End = "2025-10-30"
Trading_days = 252
Risk_free = 0.04
SMA_short = 5
SMA_long = 15
Initial_capital = 1000000

# ---- 1) Download Data ----
print("Downloading data...")
all_tickers = Tickers + [Bench]
data = yf.download(all_tickers, start=Start, end=End, auto_adjust=True)["Close"]
data = data.dropna(how="all", axis=1)
if Bench not in data.columns:
    raise ValueError(f"Benchmark {Bench} data could not be fetched.")
data = data.dropna()

# ---- 2) Compute Returns ----
log_returns = np.log(data/data.shift(1)).dropna()
asset_returns = log_returns[Tickers]
bench_returns = log_returns[Bench]

# ---- 3) Portfolio Optimization ----
mean_daily = asset_returns.mean()
expected_annual_return = mean_daily * Trading_days
covariance_annual = asset_returns.cov() * Trading_days

mu = expected_annual_return.values
Sigma = covariance_annual.values
n = len(Tickers)

def portfolio_stats(weights):
    portfolio_returns = weights.dot(mu)
    portfolio_volatility = np.sqrt(weights.dot(Sigma).dot(weights))
    sharpe = (portfolio_returns - Risk_free) / portfolio_volatility if portfolio_volatility != 0 else 0
    return portfolio_returns, portfolio_volatility, sharpe

def neg_sharpe(weights):
    return -portfolio_stats(weights)[2]

constraints = {"type":"eq", "fun":lambda w: np.sum(w) - 1} 
bound = tuple((0,1) for _ in range(n))
init_guess = np.repeat(1/n, n)

optimal = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bound, constraints=constraints)
optimal_weight = pd.Series(optimal.x, index=Tickers)
return_optimal, volatility_optimal, sharpe_optimal = portfolio_stats(optimal_weight.values)

print("\nOptimized Portfolio (Max Sharpe):")
print(optimal_weight)
print(f"Expected return: {return_optimal:.2%}, Volatility: {volatility_optimal:.2%}, Sharpe: {sharpe_optimal: .2f}")

# ---- 4) Efficient Frontier ----
target_returns = np.linspace(mu.min(), mu.max(), 50)
ef_vols = []

for tr in target_returns:
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w)-1},
        {"type": "eq", "fun": lambda w, tr=tr: w.dot(mu)-tr}
    )
    res = minimize(lambda w: w.dot(Sigma).dot(w), init_guess, method="SLSQP", bounds=bound, constraints=cons)
    ef_vols.append(np.sqrt(res.fun) if res.success else np.nan)

# ---- 5) SMA Strategy for each stock ----
result = {}
for ticker in Tickers:
    df = data[ticker].to_frame(name= "Close")
    df["SMA_Short"] = df["Close"].rolling(SMA_short).mean()
    df["SMA_Long"] = df["Close"].rolling(SMA_long).mean()
    df["Signal"] = 0
    mask = df["SMA_Short"] > df["SMA_Long"]
    df.loc[mask, "Signal"] = 1
    df.loc[~mask, "Signal"] = -1
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Daily_Return"] * df["Position"]
    df["Portfolio_Value"] = Initial_capital * (1 + df["Strategy_Return"]).cumprod()
    result[ticker] = df

    # Plot SMA Crossover
    plt.figure(figsize=(10,5))
    plt.plot(df["Close"], label=f"{ticker} Price", alpha=0.7)
    plt.plot(df["SMA_Short"], label=f"SMA {SMA_short}", linestyle="--")
    plt.plot(df["SMA_Long"], label=f"SMA {SMA_long}", linestyle="--")
    plt.title(f"{ticker} SMA Strategy")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---- 6) Combine all SMA Strategies ----
combined = pd.DataFrame({t: result[t]["Strategy_Return"] for t in Tickers})
combined["Portfolio_Return"] = combined.mean(axis=1)
combined["Portfolio_Value"] = Initial_capital * (1 + combined["Portfolio_Return"]).cumprod()

# ---- 7) Buy and Hold Portfolio ---
buy_hold = data[Tickers].pct_change().dropna().mean(axis=1)
buy_hold_value = Initial_capital * (1 + buy_hold).cumprod()

# ---- 8) Plotting Results ----
# Portfolio Comparision
plt.figure(figsize=(12,6))
for t in Tickers:
    plt.plot(result[t]["Portfolio_Value"], label=f"{t} SMA Portfolio")
plt.plot(combined["Portfolio_Value"], color="black", linewidth=2, linestyle="--", label="Combined SMA Portfolio")
plt.plot(buy_hold_value, color="orange", linewidth=2, linestyle="--", label="Buy and Hold Portfolio")
plt.title("Portfolio Value Comparison")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.show()

# Efficient Frontier
plt.figure(figsize=(8,5))
plt.plot(ef_vols, target_returns,label="Efficient Frontier", color="blue")
plt.scatter(volatility_optimal, return_optimal, c="red", label="Max Sharpe Portfolio", s=100)
plt.xlabel("Volatility (Annual)")
plt.ylabel("Expected Return (Annual)")
plt.title("Efficient Frontier with Max-Sharpe Portfolio")
plt.legend()
plt.grid(True)
plt.show()

# ---- 9) Performance Metrics ----
sharpe_sma = combined["Portfolio_Return"].mean() / combined["Portfolio_Return"].std() * np.sqrt(Trading_days)
total_return_sma = combined["Portfolio_Value"].iloc[-1] / Initial_capital - 1
max_drawdown = (combined["Portfolio_Value"] / combined["Portfolio_Value"].cummax() - 1)

print("\nCombined SMA Portfolio Performace:")
print(f"Total Return: {total_return_sma:.2%}")
print(f"Sharpe Ratio: {sharpe_sma:.2f}")
print(f"Max Drawdown: {max_drawdown}")