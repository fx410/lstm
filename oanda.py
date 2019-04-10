
import csv 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
import warnings
warnings.filterwarnings('ignore')

import oandapy
import configparser

 #プライベートデータ
config = configparser.ConfigParser()
config.read('./config/config.txt')
account_id = config['oanda']['account_id']
api_key = config['oanda']['api_key']

 #oandaのapiと繋げる
oanda = oandapy.API(environment="practice",
                  access_token=api_key)

 #為替のデータ取得
res = pd.DataFrame(oanda.get_history(instrument="USD_JPY", granularity="D")['candles'])
print(res[-1:])


 #特徴量の整理
df = res[['openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']]
df.columns = ['open', 'close', 'high', 'low', 'volume']

 #最新windowのデータ
df = df[-11:-1]
print(df)

res_now = oanda.get_prices(instruments="USD_JPY")
print(res_now)

ary = df.values[-1]
str = ary.astype('U')
print(str)

with open('record.csv', 'w') as f:
    list = [['date', 'pred_close', 'today_close']]
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(list)

