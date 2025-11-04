# Relative strength indicator strategy: 
# Buying when prices are oversold and selling when prices are overbought
# Essentially a mean reversion strategy
# When RSI drops below 30, it is oversold so buy
# When RSI exceeds 70, it is overbought so sell
# Exit strateguy: Average True Range (ATR) indicator
# ATR measures the averahe range for each period and is therefore a good volatility indicator
# After entering a trade, we will set take profit and stop loss to 2 ATR each

import yfinance as yf
import pandas as pd
import plotly.express as px

# Parameters
Symbol = "BAC" 
period = "1000d" # 1000 daily bars
interval = "1d"

# Download data
df = yf.download(Symbol, period=period, interval=interval)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()
df.rename(columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close"}, inplace=True)

fig = px.line(df, x="time", y="close", title=f"{Symbol} Daily Close Prices")
fig.show()

# setting the RSI period
rsi_period = 14

# to calculate RSI, first calculate the exponential weighted average gain and loss during the period
df["gain"] = (df["close"] - df["open"]).apply(lambda x: x if x > 0 else 0)
df["loss"] = (df["close"] - df["open"]).apply(lambda x: -x if x < 0 else 0)

# Calculate the Exponential moving average
df["ema_gain"] = df["gain"].ewm(span=rsi_period, min_periods=rsi_period).mean()
df["ema_loss"] = df["loss"].ewm(span=rsi_period, min_periods=rsi_period).mean()

# Relative strength is the ratio between the exponential av gain divided by the exponential avg loss
df["rs"] = df["ema_gain"]/df["ema_loss"]

# RSI is calculated based on the Relative strength using the following formula
df["rsi_14"] = 100 - (100/(df["rs"] +1))

# Display results
print(df[["time", "rsi_14", "rs", "ema_gain", "ema_loss"]])

# plotting the RSI
fig_rsi = px.line(df, x="time", y="rsi_14", title="RSI Indicator")

# RSI levels
overbought_level = 70
oversold_level = 30

# adding overbought and oversold levels to the plot
fig_rsi.add_hline(y=overbought_level)
fig_rsi.add_hline(y=oversold_level)

# showing the RSI Figure
fig_rsi.show()

# Calculate the ATR indicator
atr_period = 14
df["range"] = df["high"] - df["low"] #calculating range of each candle
df["atr_14"] = df["range"].rolling(atr_period).mean() #calculating the average value of ranges
print(df[["time", "atr_14"]])

# Plotting the ATR indicator
fig_atr = px.line(df, x="time", y="atr_14", title="ATR indicator")
fig_atr.show()

# Class Position contain data about trades opened/closed during the backtest
class Position:
    def __init__(self, open_datetime, open_price, order_type, volume, sl, tp):
        self.open_datetime = open_datetime
        self.open_price = open_price
        self.order_type = order_type
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.close_datetime = None
        self.close_price = None
        self.profit = None
        self.status = "open"
    
    def close_position(self, close_datetime, close_price):
        self.close_datetime = close_datetime
        self.close_price = close_price
        self.profit = (self.close_price - self.open_price) * self.volume if self.order_type == "buy" else (self.open_price - self.close_price) * self.volume
        self.status = "closed"
    
    def _asdict(self):
        return {
            "open_datetime": self.open_datetime,
            "open_price": self.open_price,
            "order_type": self.order_type,
            "volume": self.volume,
            "sl": self.sl,
            "tp": self.tp,
            "close_datetime": self.close_datetime,
            "close_price": self.close_price,
            "profit": self.profit,
            "status": self.status
        }
    
# class Strategy defines trading logic and evaluates the backtest based on opened/closed positions
class Strategy:
    def __init__(self, df, starting_balance):
        self.starting_balance = starting_balance
        self.positions = []
        self.data = df
    
    # return backtest result
    def get_positions_df(self):
        df = pd.DataFrame([position._asdict() for position in self.positions])
        df["pnl"] = df["profit"].cumsum() + self.starting_balance
        return df
    
    # add Position class to list
    def add_position(self, position):
        self.positions.append(position)
        return True
    
    #close positions when stop loss or take profit is reached
    def close_tp_sl(self, data):
        for pos in self.positions:
            if pos.status == "open":
                if pos.sl >= data.close and pos.order_type == "buy":
                    pos.close_position(data.time, pos.sl)
                elif pos.sl <= data.close and pos.order_type == "sell":
                    pos.close_position(data.time, pos.sl)
                elif pos.tp <= data.close and pos.order_type == "buy":
                    pos.close_position(data.time, pos.tp)
                elif pos.tp >= data.close and pos.order_type == "sell":
                    pos.close_position(data.time, pos.tp)
    
    # check for open positions
    def has_open_positions(self):
        for pos in self.positions:
            if pos.status == "open":
                return True
        return False
    
    # strategy logic how positions should be opened/closed
    def logic(self, data):
        if not self.has_open_positions(): #if no position is open
            if data["rsi_14"] < 30: # if RSI less than 30, buy
                # Position variables
                open_datetime = data["time"]
                open_price = data["close"]
                order_type = "buy"
                volume = 10000
                sl = open_price - 2 * data["atr_14"]
                tp = open_price + 2 * data["atr_14"]
                self.add_position(Position(open_datetime, open_price, order_type, volume, sl, tp))

            elif data["rsi_14"] > 70: # if RSI more than 70, sell
                #Position variables
                open_datetime = data["time"]
                open_price = data["close"]
                order_type = "sell"
                volume = 10000
                sl = open_price + 2 * data["atr_14"]
                tp = open_price - 2 * data["atr_14"]
                self.add_position(Position(open_datetime, open_price, order_type, volume, sl, tp))

    def run(self): #Logic
        #data represents a moment in time while iterating through a backtest
        for i, data in self.data.iterrows():
            # close positions when stop loss or take profit is reached
            self.close_tp_sl(data)

            # strategy logic
            self.logic(data)

        return self.get_positions_df()
    
#preparing data for backtest
backtest_df = df[14:] # removing NaN values
print(backtest_df)

# creating an instance of Strategy class
rsi_strategy = Strategy(backtest_df, 10000)

# running backtest
backtest_result = rsi_strategy.run()
print(backtest_result)

# analysing closed positions only
backtest_result = backtest_result[backtest_result["status"] == "closed"]

# visualising trades
fig_backtest = px.line(df, x="time", y=["close"], title="RSI Strategy - Trades")

# adding trades to plots
for i, position in backtest_result.iterrows():
    if position.status == "closed":
        fig_backtest.add_shape(type="line",x0=position.open_datetime,
        y0=position.open_price, x1=position.close_datetime, y1=position.close_price,
        line=dict(color="green" if position.profit >= 0 else "red", width=3))

fig_backtest.show()

fig_pnl = px.line(backtest_result, x="close_datetime", y="pnl")
fig_pnl.show()

