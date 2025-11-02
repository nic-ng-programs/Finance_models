import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# Parameters
T1 = "KO" # Coca cola
T2 = "PEP" # Pepsi
START = "2015-01-01"
END = "2025-10-31"
Roll_window = 60 # Rolling window for mean/std of spread
Entry_Z = 2.0
Exit_Z = 0.5
Initial_Capital = 100000 # for P&L Scaling
Transaction_cost_per_trade = 0.0 # assume flat costs per trade

# 1) Download prices
data = yf.download([T1, T2], start= START, end=END)
if isinstance(data.columns, pd.MultiIndex):
    data = data["Close"]
else: 
    data = pd.DataFrame(data["Close"])
data = data.dropna()
prices_a = data[T1]
prices_b = data[T2]

# 2) Correlation check
corr = prices_a.pct_change().dropna().corr(prices_b.pct_change().dropna())
print(f"Return correlation between {T1} and {T2}: {corr:.3f}")

# 3) Cointegration test (Engler-Granger)
logA = np.log(prices_a)
logB = np.log(prices_b)
score, pvalue, _ = coint(logA, logB)
print(f"Cointegration test p-value: {pvalue:.4f}")

if pvalue >= 0.05:
    print("No strong evidence of cointegration (p >= 0.05). Proceed with caution")
else:
    print("Series appears to be cointegrated (p < 0.05). Mean reversion plausible.")

# 4) Estimate hedge ratio via OLS
X = sm.add_constant(logB)
res = sm.OLS(logA, X).fit()
beta = res.params[1]
alpha = res.params[0]
print(f"OLS hedge ratio (beta): {beta:.4f}")

# 5) Compute spread and z-score
spread = logA - (alpha + beta * logB)
rolling_mean = spread.rolling(window=Roll_window).mean()
rolling_std = spread.rolling(window=Roll_window).std()
zscore = (spread - rolling_mean) / rolling_std

# 6) Generating trading positions
positions = pd.DataFrame(index=spread.index, columns=[f"{T1}_pos", f"{T2}_pos"])
positions.iloc[:,:] = 0.0
in_trade = False

for i in range(len(spread)):
    z = zscore.iloc[i]
    if not np.isfinite(z):
        continue
    if not in_trade:
        if z > Entry_Z:
            positions.iloc[i] = [-1.0, beta] #short spread
            in_trade = True
        elif z < -Entry_Z:
            positions.iloc[i] = [1.0, -beta] # long spread
            in_trade = True
    else:
        positions.iloc[i] = positions.iloc[i-1]
        if abs(z) < Exit_Z:
            positions.iloc[i] = [0.0, 0.0]
            in_trade = False

positions = positions.ffill().fillna(0.0)

# 7) Compute P&L
dollar_exp_a = positions[f"{T1}_pos"] * prices_a
dollar_exp_b = positions[f"{T2}_pos"] * prices_b
abs_exp = (dollar_exp_a.abs() + dollar_exp_b.abs()).replace(0, np.nan)
scaling = (Initial_Capital * 0.5) / abs_exp
scaling = scaling.fillna(0.0)

notional_a = dollar_exp_a * scaling
notional_b = dollar_exp_b * scaling

returns_a = prices_a.pct_change().fillna(0.0)
returns_b = prices_b.pct_change().fillna(0.0)
pnl = notional_a.shift(1) * returns_a + notional_b.shift(1) * returns_b
cum_returns = (1 + pnl / Initial_Capital).cumprod() - 1

# 8) Performance summary
total_return = cum_returns.iloc[-1]
sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252)
print(f"Total return: {total_return:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")

# 9) Plot results
fig, axs = plt.subplots(3,1, figsize=(10,10), sharex=True)
axs[0].plot(spread, label= "Spread (A-βB)")
axs[0].plot(rolling_mean, "--", label="Rolling mean")
axs[0].legend; axs[0].set_title("Spread")

axs[1].plot(zscore, label="Z-score")
axs[1].axhline(Entry_Z, color="k", linestyle="--")
axs[1].axhline(-Entry_Z, color="k", linestyle="--")
axs[1].axhline(Exit_Z, color="grey", linestyle=":")
axs[1].axhline(-Exit_Z, color="grey", linestyle=":")

axs[2].plot(cum_returns, label="Cumulative returns")
axs[2].legend(); axs[2].set_title("Strategy returns")

plt.tight_layout()
plt.show()