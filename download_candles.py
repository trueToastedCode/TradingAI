import yfinance as yf
import datetime

symbol = 'GOOG'

fmt = '%Y-%m-%d'
now = datetime.datetime.now()
data = yf.download(symbol,
                   start=(now - datetime.timedelta(days=7)).strftime(fmt),
                   end=(now - datetime.timedelta(days=1)).strftime(fmt),
                   interval='1m')
data.to_csv(f'data/{symbol}-Train.csv.txt', sep=' ')
