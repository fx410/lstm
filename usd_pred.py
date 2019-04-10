
 #ライブラリ
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date 
from keras.models import model_from_json
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
 #特徴量の整理
df = res[['openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']]
df.columns = ['open', 'close', 'high', 'low', 'volume']
 #最新windowのデータ
df = df[-11:-1]
print(df)
 #入力データの正規化
lstm_in = df.copy()
for col in lstm_in:
    lstm_in.loc[:, col] = lstm_in[col] / lstm_in[col].iloc[0] - 1
lstm_in = [lstm_in]

 #numpy型に変更
lstm_in = [np.array(tmp) for tmp in lstm_in]
lstm_in = np.array(lstm_in)
 
  #終値予測モデルの読み込み
model_close = model_from_json(open('usd_model.json').read())
 #重みの読み込み
model_close.load_weights('usd_weights.h5')

 #明日の終値の予測
model_close.compile(loss='mae', optimizer='adam')
pred = model_close.predict(lstm_in, batch_size=1)[0][0]
print(pred)
pred_close = round(pred * (df['close'].values[:])[0], 3)
print('\n10_yesterday_close', (df['close'].values[:])[0])
print('\npred_close', pred_close)


  #高値予測モデルの読み込み
model_high = model_from_json(open('usd_high_model.json').read())
 #重みの読み込み
model_high.load_weights('usd_high_weights.h5')

 #明日の高値の予測
model_high.compile(loss='mae', optimizer='adam')
pred = model_high.predict(lstm_in, batch_size=1)[0][0]
print(pred)
pred_high = round(pred * (df['high'].values[:])[0], 3)
print(pred_high)


  #安値予測モデルの読み込み
model_low = model_from_json(open('usd_low_model.json').read())
 #重みの読み込み
model_low.load_weights('usd_low_weights.h5')

 #明日の安値の予測
model_low.compile(loss='mae', optimizer='adam')
pred = model_low.predict(lstm_in, batch_size=1)[0][0]
print(pred)
pred_low = round(pred * (df['low'].values[:])[0], 3)
print(pred_low)

past_close = df.iloc[-1,1]
record = [['date', 'past_close', 'pred_close', 'pred_high', 'pred_low'],[str(date.today()), past_close, pred_close, pred_high, pred_low], [None, None, past_close-pred_close, pred_high-past_close, past_close-pred_low]]
record=pd.DataFrame(record)
print(record)
 #データを記録
 #with open('/users/isshin/project/fx_app/lstm/record.csv','r') as f1:
  #   reader = csv.reader(f1)
   #  list = [e for e in reader]
    # print(list)
    # list.append(record)
    # print(list)
    # with open('/users/isshin/project/fx_app/lstm/record.csv','w') as f2:
     #     writer = csv.writer(f2, lineterminator="\n")
      #     writer = writer.writerow([list])
