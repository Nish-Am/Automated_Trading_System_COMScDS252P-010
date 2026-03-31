import pandas as pd
import matplotlib.pyplot as plt
import boto3

tesla_trading_data = pd.read_csv('data/raw_data/tsla_raw_data.csv')
# print(tesla_trading_data.head())
'''
                        Date        Open        High         Low       Close     Volume  Dividends  Stock Splits
0  2025-03-21 00:00:00-04:00  234.990005  249.520004  234.550003  248.710007  132728700        0.0           0.0
1  2025-03-24 00:00:00-04:00  258.079987  278.640015  256.329987  278.390015  169079900        0.0           0.0
2  2025-03-25 00:00:00-04:00  283.600006  288.200012  271.279999  288.140015  150361500        0.0           0.0
3  2025-03-26 00:00:00-04:00  282.660004  284.899994  266.510010  272.059998  153629800        0.0           0.0
4  2025-03-27 00:00:00-04:00  272.480011  291.850006  271.820007  273.130005  162572100        0.0           0.0
'''
# ------------- EDA ----------------------
tesla_trading_data['Date'] = pd.to_datetime(
    tesla_trading_data['Date'], utc = True
    ).dt.date

tesla_trading_data.set_index('Date', inplace = True)
# print(tesla_trading_data.head())
'''
                  Open        High         Low       Close     Volume  Dividends  Stock Splits
Date
2025-03-21  234.990005  249.520004  234.550003  248.710007  132728700        0.0           0.0
2025-03-24  258.079987  278.640015  256.329987  278.390015  169079900        0.0           0.0
2025-03-25  283.600006  288.200012  271.279999  288.140015  150361500        0.0           0.0
2025-03-26  282.660004  284.899994  266.510010  272.059998  153629800        0.0           0.0
2025-03-27  272.480011  291.850006  271.820007  273.130005  162572100        0.0           0.0
'''

# checking missing values 
print(tesla_trading_data.isna().sum())   # no null data 

# -------------featue engineering ---------------------- 

# calculating moving average
tesla_trading_data['MA_20'] = tesla_trading_data['Close'].rolling(window=20).mean()
tesla_trading_data['MA_50'] = tesla_trading_data['Close'].rolling(window=50).mean()

# Calculate Returns  - daily percentage change in price
tesla_trading_data['Returns'] = tesla_trading_data['Close'].pct_change()

# RSI - Relative Strength Index - A momentum indicator used in trading
# Shows overbought or oversold conditions

delta = tesla_trading_data['Close'].diff() # today’s price minus yesterday’s price
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

rs = gain / loss
tesla_trading_data['RSI_14'] = 100 - (100 / (1 + rs))

# visualization 

plt.figure()
plt.plot(tesla_trading_data['Close'], label='Close Price')
plt.plot(tesla_trading_data['MA_20'], label='MA 20')
plt.plot(tesla_trading_data['MA_50'], label='MA 50')
plt.legend()
# plt.show()

# Volatility = how much the price moves (up and down)
# Risk Indicator
# High volatility → High risk
# Low volatility → Stable
volatility = tesla_trading_data['Returns'].std()

print("Volatility:", volatility)

# create desicion row
def decision(row):
    if row['MA_20'] > row['MA_50'] and row['RSI_14'] < 70:
        return "BUY"
    elif row['MA_20'] < row['MA_50'] or row['RSI_14'] > 70:
        return "SELL"
    else:
        return "HOLD"

tesla_trading_data['Decision'] = tesla_trading_data.apply(decision, axis=1)

# buy sell signals 
tesla_trading_data['Signal'] = 0
tesla_trading_data.loc[tesla_trading_data['Decision'] == 'BUY', 'Signal'] = 1
tesla_trading_data.loc[tesla_trading_data['Decision'] == 'SELL', 'Signal'] = -1

tesla_trading_data = tesla_trading_data.dropna()

print(tesla_trading_data.head())

'''
                  Open        High         Low       Close     Volume  Dividends  Stock Splits       MA_20       MA_50   Returns     RSI_14 Decision  Signal
Date
2025-06-02  343.500000  348.019989  333.329987  342.690002   81873800        0.0           0.0  328.806003  288.542002 -0.010881  60.563128      BUY       1
2025-06-03  346.600006  355.399994  343.040009  344.269989   99324500        0.0           0.0  332.006502  290.453202  0.004611  55.051494      BUY       1
2025-06-04  345.100006  345.600006  327.329987  332.049988   98912100        0.0           0.0  334.841501  291.526401 -0.035495  42.151253      BUY       1
2025-06-05  322.489990  324.549988  273.209991  284.700012  287499800        0.0           0.0  335.265501  291.457601 -0.142599  29.543865      BUY       1
2025-06-06  298.829987  305.500000  291.140015  295.140015  164747700        0.0           0.0  335.781502  291.919201  0.036670  31.133902      BUY       1
'''

# save processed data locally 
tesla_trading_data.to_csv('data/processed_data/tsla_processed_data.csv')

# Upload to AWS - s3
s3 = boto3.client('s3')
s3.upload_file('data/processed_data/tsla_processed_data.csv', 'ac-trading-data', 'processed/tsla_raw_data.csv')    

# ------------ Backtesting ---------------------------

# shift 1 - decisions to execute on the following day
tesla_trading_data['Strategy_Return'] = tesla_trading_data['Returns'] * tesla_trading_data['Signal'].shift(1) 

# addding cumilative returns 
tesla_trading_data['Cumulative_Market'] = (1 + tesla_trading_data['Returns']).cumprod()
tesla_trading_data['Cumulative_Strategy'] = (1 + tesla_trading_data['Strategy_Return']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(tesla_trading_data['Cumulative_Market'], label='Market Return', color='blue')
plt.plot(tesla_trading_data['Cumulative_Strategy'], label='Strategy Return', color='green')
plt.title('Strategy vs Market Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

market_return = tesla_trading_data['Cumulative_Market'].iloc[-1] - 1
strategy_return = tesla_trading_data['Cumulative_Strategy'].iloc[-1] - 1

print("Total Market Return:", round(market_return*100, 2), "%")
print("Total Strategy Return:", round(strategy_return*100, 2), "%")

days = len(tesla_trading_data)
annual_factor = 252 / days  # 252 trading days in a year
annualized_strategy = ((1 + strategy_return) ** annual_factor) - 1
annualized_market = ((1 + market_return) ** annual_factor) - 1

print("Annualized Strategy Return:", round(annualized_strategy*100, 2), "%")
print("Annualized Market Return:", round(annualized_market*100, 2), "%")

strategy_vol = tesla_trading_data['Strategy_Return'].std() * (252**0.5)  # annualized volatility
market_vol = tesla_trading_data['Returns'].std() * (252**0.5)

print("Strategy Volatility:", round(strategy_vol*100, 2), "%")
print("Market Volatility:", round(market_vol*100, 2), "%")

'''
Total Market Return: 6.21 %
Total Strategy Return: -42.62 %
Annualized Strategy Return: -49.99 %
Annualized Market Return: 7.8 %
Strategy Volatility: 47.28 %
Market Volatility: 47.31 %
'''