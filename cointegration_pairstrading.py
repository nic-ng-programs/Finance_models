import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import scipy.optimize as spop
import matplotlib.pyplot as plt

# specifying parameter
stocks = ["JPM", "C"]
start = "2024-01-01"
end = "2025-10-31"
fee = 0.001
window = 252
t_threshold = -2.5

# retrieving data
data = pd.DataFrame()
returns = pd.DataFrame()
for stock in stocks:
    prices = yf.download(stock, start, end)
    data[stock] = prices["Close"]
    returns[stock] = np.append(data[stock][1:].reset_index(drop=True) / data[stock][:-1].reset_index(drop=True) - 1, 0)
    
# initialising arrays
gross_returns = np.array([])
net_returns = np.array([])
t_statistics = np.array([])
stock_1 = stocks[0]
stock_2 = stocks[1]

# moving through the sample
for t in range(window, len(data)):
    # defining the unit root function: stock2 = a + b*stock1
    def unit_root(b):
        a = np.average(data[stock_2][t-window:t] - b*data[stock_1][t-window:t]) 
        fair_value = a + b*data[stock_1][t-window:t]
        diff = np.array(fair_value - data[stock_2][t-window:t])
        diff_diff = diff[1:] - diff[:-1] 
        reg = sm.OLS(diff_diff, diff[:-1])
        result = reg.fit()
        return result.params[0]/result.bse[0]
    
    #optimising the cointegration equation parameters
    result_1 = spop.minimize(unit_root, data[stock_2][t]/data[stock_1][t], method="Nelder-Mead")
    t_optimal = result_1.fun
    b_optimal = float(result_1.x)
    a_optimal = np.average(data[stock_2][t-window:t] - b_optimal*data[stock_1][t-window:t])     
    fair_value = a_optimal + b_optimal*data[stock_1][t]

    # simulating trading
    if t == window:
        old_signal = 0
    if t_optimal > t_threshold:
        signal = 0
        gross_return = 0
    else:
        signal = np.sign(fair_value - data[stock_2][t])
        gross_return = signal*returns[stock_2][t] - signal*returns[stock_1][t]
    fees = fee*abs(signal - old_signal)
    net_return = gross_return - fees
    gross_returns = np.append(gross_returns, gross_return)
    net_returns = np.append(net_returns, net_return)
    t_statistics = np.append(t_statistics, t_optimal)
    
    #interface: reporting daily positions and realised returns
    print("day " +str(data.index[t]))
    print("")
    if signal == 0:
        print("No trading")
    elif signal == 1:
        print("Long position on " +stock_2+" and short position on "+stock_1)
    else:
        print("Long position on "+stock_1+" and short position on "+stock_2)
    print("Gross daily return: "+str(round(gross_return*100,2))+"%")    
    print("Net daily return: "+str(round(net_return*100,2))+"%")
    print("Cumulative net return: "+str(round(np.prod(net_return)*100-100,2))+"%")
    print("")
    old_signal = signal

# plotting equity curves
plt.plot(np.append(1,np.cumprod(1+gross_returns)))
plt.plot(np.append(1,np.cumprod(1+net_returns)))
plt.show()