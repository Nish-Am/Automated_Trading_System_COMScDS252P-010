import pandas as pd
import matplotlib.pyplot as plt
import boto3
import io

# ------------- EDA ----------------------
def market_analysis(tesla_trading_data):    

    tesla_trading_data['Date'] = pd.to_datetime(
    tesla_trading_data['Date'], utc = True  
    ).dt.date

    tesla_trading_data.set_index('Date', inplace = True)

    # checking missing values 
    # print(tesla_trading_data.isna().sum())   # no null data 

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

    def get_market_trend(row):
        if row['MA_20'] > row['MA_50']:
            return 'UP'
        elif row['MA_20'] < row['MA_50']:
            return 'DOWN'
        else:
            return 'SIDEWAYS'

    # create desicion row 
    def decision(row):
        if row['MA_20'] > row['MA_50'] and row['RSI_14'] < 30:
            return 'BUY'
        elif row['MA_20'] < row['MA_50'] and row['RSI_14'] > 70:
            return 'SELL'
        else:
            return 'HOLD'       
    
    tesla_trading_data['Trend'] = tesla_trading_data.apply(get_market_trend, axis=1)
    tesla_trading_data['Decision'] = tesla_trading_data.apply(decision, axis=1)

    # buy sell signals 
    tesla_trading_data['Signal'] = 0
    tesla_trading_data.loc[tesla_trading_data['Decision'] == 'BUY', 'Signal'] = 1
    tesla_trading_data.loc[tesla_trading_data['Decision'] == 'SELL', 'Signal'] = -1

    tesla_trading_data = tesla_trading_data.dropna()

    # save processed data locally 
    tesla_trading_data.to_csv('data/processed_data/tsla_processed_data.csv')

    # Upload to AWS - s3
    s3 = boto3.client('s3')
    # s3.upload_file('data/processed_data/tsla_processed_data.csv', 'ac-trading-data', 'processed/tsla_raw_data.csv')  

    return tesla_trading_data  

# ------------ Backtesting ---------------------------
def backtesting(tesla_trading_data):    

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
    # plt.show()

    market_return = tesla_trading_data['Cumulative_Market'].iloc[-1] - 1
    strategy_return = tesla_trading_data['Cumulative_Strategy'].iloc[-1] - 1

    print('Total Market Return:', round(market_return*100, 2), '%')
    print('Total Strategy Return:', round(strategy_return*100, 2), '%')

    days = len(tesla_trading_data)
    annual_factor = 252 / days  # 252 trading days in a year
    annualized_strategy = ((1 + strategy_return) ** annual_factor) - 1
    annualized_market = ((1 + market_return) ** annual_factor) - 1

    print('Annualized Strategy Return:', round(annualized_strategy*100, 2), '%')
    print('Annualized Market Return:', round(annualized_market*100, 2), '%')

    strategy_vol = tesla_trading_data['Strategy_Return'].std() * (252**0.5)  # annualized volatility
    market_vol = tesla_trading_data['Returns'].std() * (252**0.5)

    # print('Strategy Volatility:', round(strategy_vol*100, 2), '%')
    # print('Market Volatility:', round(market_vol*100, 2), '%')

    '''
    Total Market Return: 6.21 %
    Total Strategy Return: -42.62 %
    Annualized Strategy Return: -49.99 %
    Annualized Market Return: 7.8 %
    Strategy Volatility: 47.28 %
    Market Volatility: 47.31 %
    '''
    return tesla_trading_data

def risk_management(tesla_trading_data):
    
    # Calculate rolling volatility (risk over time)
    tesla_trading_data['Volatility'] = tesla_trading_data['Returns'].rolling(20).std()

    # Stop-loss: if daily return is too negative → exit
    stop_loss_threshold = -0.02  # -2%

    tesla_trading_data.loc[
        tesla_trading_data['Returns'] < stop_loss_threshold,
        'Signal'
    ] = 0
    
    # Define threshold (you can tune later)
    threshold = tesla_trading_data['Volatility'].mean()
    
    # Apply risk filter
    tesla_trading_data.loc[tesla_trading_data['Volatility'] > threshold, 'Signal'] = 0
    
    return tesla_trading_data

def save_output_to_s3(latest_row, bucket, key):
    s3 = boto3.client('s3')
    
    try:
        # Try reading existing file
        obj = s3.get_object(Bucket=bucket, Key=key)
        existing_data = pd.read_csv(io.BytesIO(obj['Body'].read()))
    except:
        # If file doesn't exist, create new
        existing_data = pd.DataFrame(columns=['Date', 'Trend', 'Decision'])

    # Create new row
    new_row = pd.DataFrame([{
        'Date': latest_row.name,
        'Trend': latest_row['Trend'],
        'Decision': latest_row['Decision']
    }])

    # Append
    updated_data = pd.concat([existing_data, new_row], ignore_index=True)

    # Upload back to S3
    csv_buffer = io.StringIO()
    updated_data.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue()
    )

tesla_trading_data = pd.read_csv('data/raw_data/tsla_raw_data.csv')
tesla_trading_data = market_analysis(tesla_trading_data)
tesla_trading_data = risk_management(tesla_trading_data)
tesla_trading_data = backtesting(tesla_trading_data)

latest = tesla_trading_data.iloc[-1]

# save locally 
with open('E:\MSc Data Science\Principals of DS\course work\Individual_Work_02\Automated_Trading_System_COMScDS252P-010\data\outputs\output.txt', 'a') as f:
    f.write(f"{latest.name} | {latest['Trend']} | {latest['Decision']}\n")

# save in S3 
save_output_to_s3(latest_row = latest, bucket = 'ac-trading-data', key = 'outputs/daily_output.csv')

# with open('/home/ubuntu/Automated-Trading-System/daily_output.txt', 'a') as f:
#     f.write(f"{latest.name} | {latest['Trend']} | {latest['Decision']}\n")