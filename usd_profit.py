
 #ライブラリ
import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import seaborn as sns
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
warnings.filterwarnings('ignore')

import oandapy
import pytz
import configparser
 
 #プライベートデータ
config = configparser.ConfigParser()
config.read('./config/config.txt')
account_id = config['oanda']['account_id']
api_key = config['oanda']['api_key']

 #oandaのapiと繋げる
oanda = oandapy.API(environment="practice",
                  access_token=api_key)

 #iso形式から日本時間に変換
def iso_to_jp(iso):
    date = None
    try:
        date = pd.to_datetime(iso)
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            date = pd.to_datetime(iso)
            date = date.astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return date

 #見やすいように文字列に変換
def date_to_str(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')

  #為替のデータ取得
response = oanda.get_history(instrument="USD_JPY", granularity="D", count='5000')
res = pd.DataFrame(response['candles'])
res['time'] = res['time'].apply(lambda x:date_to_str(iso_to_jp(x)))
 
 #特徴量の整理
df = res[['time', 'openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']]
df.columns = ['time', 'open', 'close', 'high', 'low', 'volume']

 #訓練データとテストデータへの分割
split_date = '2015/05/07 06:00:00'
train, test = df[df['time'] < split_date], df[df['time'] >= split_date]
del train['time']
del test['time']
 
 #ウィンドウの数
window_len = 10

 #訓練データのクレイピング(in:10日間の各特徴量の変動幅、out:終値の10日後比)
train_in = []
for i in range(len(train) - window_len):
	  tmp = train[i:(i + window_len)].copy()
	  for col in train:
			  tmp.loc[:,col] = tmp[col] / tmp[col].iloc[0] - 1
	  train_in.append(tmp)
train_out = (train['close'][window_len:].values / train['close'][:-window_len].values)
 
 #テストデータのクレイピング
test_in = []
for i in range(len(test) - window_len):
	  tmp = test[i:(i + window_len)].copy()
	  for col in test:
			  tmp.loc[:,col] = tmp[col] / tmp[col].iloc[0] - 1
	  test_in.append(tmp)
test_out = (test['close'][window_len:].values / test['close'][:-window_len].values)

 #dataframeからnumpy型に変更
train_in = [np.array(train_input) for train_input in train_in]
train_in = np.array(train_in)

test_in = [np.array(test_input) for test_input in test_in]
test_in = np.array(test_in)
  
 #モデルの読み込み
model_close = model_from_json(open('usd_model.json').read())
 #重みの読み込み
model_close.load_weights('usd_weights.h5')

 #明日の終値の予測
model_close.compile(loss='mae', optimizer='adam')
pred = model_close.predict(test_in, batch_size=1)
pred_close = pred.reshape(-1) * (test['close'].values[:-10])

 #損益を可視化
past_close = np.array(test['close'].values[9:-10])
pred_close = pred_close[:-9]
print('\npred', pred_close.shape, 'past',past_close.shape)
print('\npred_close', pred_close, '\npast_close:', past_close)
total_return = np.zeros(len(pred_close)-1)

for i in range(len(pred_close)-1):
     #予測が前日の終値よりも高い場合、買い
    if pred_close[i]>=past_close[i]:
        total_return[i] = total_return[i-1] + (past_close[i+1] - past_close[i]) * 10000
        if (past_close[i+1] - past_close[i]) > 0:
            print('Ask',int((past_close[i+1] - past_close[i]) * 10000))
        else:
            print('Ask     ',int((past_close[i+1] - past_close[i]) * 10000))
     #予測が前日の終値よりも低い場合、売り
    if pred_close[i]<=past_close[i]:
        total_return[i] = total_return[i-1] + (past_close[i] - past_close[i+1]) * 10000
        if (past_close[i+1] - past_close[i]) < 0:
            print('Bid',int((past_close[i] - past_close[i+1]) * 10000))
        else:
            print('Bid     ',int((past_close[i] - past_close[i+1]) * 10000))

   
print(total_return)
plt.plot(total_return)
plt.show()
