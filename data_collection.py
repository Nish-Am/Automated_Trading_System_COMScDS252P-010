import yfinance as yf
import boto3

def fetch_data():
    stock = yf.Ticker('TSLA')
    data = stock.history(period='1y')
    data.to_csv('data/raw_data/tsla_raw_data.csv')
    print('TSLA - raw data saved')

def upload_data():
    s3 = boto3.client('s3')
    s3.upload_file('data/raw_data/tsla_raw_data.csv', 'ac-trading-data', 'raw/tsla_raw_data.csv')

fetch_data()
upload_data()
