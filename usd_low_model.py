
 #ライブラリ
import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import seaborn as sns
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
train_out = (train['low'][window_len:].values / train['low'][:-window_len].values)
 
 #テストデータのクレイピング
test_in = []
for i in range(len(test) - window_len):
	  tmp = test[i:(i + window_len)].copy()
	  for col in test:
			  tmp.loc[:,col] = tmp[col] / tmp[col].iloc[0] - 1
	  test_in.append(tmp)
test_out = (test['low'][window_len:].values / test['low'][:-window_len].values)

 #dataframeからnumpy型に変更
train_in = [np.array(train_input) for train_input in train_in]
train_in = np.array(train_in)

test_in = [np.array(test_input) for test_input in test_in]
test_in = np.array(test_in)

 #モデル構築の定義
def build_model(inputs, output_size, neurons, activ_func="linear", dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(202)
 #モデルの構築
yen_model = build_model(train_in, output_size=1, neurons=20)

 #モデルの学習
yen_history = yen_model.fit(train_in, train_out, epochs=25, batch_size=1, verbose=2, shuffle=True)

model_json = yen_model.to_json()
open('usd_low_model.json', 'w').write(model_json)
yen_model.save_weights('usd_low_weights.h5')

 #モデルによる予測と実測の比較
fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']< split_date]['time'][window_len:].astype(datetime),train['low'][window_len:],label='Actual', color='blue')
ax1.plot(df[df['time']< split_date]['time'][window_len:].astype(datetime),((np.transpose(yen_model.predict(train_in))) * train['low'].values[:-window_len])[0],label='Predicted', color='red')
plt.show()

fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']>= split_date]['time'][window_len:].astype(datetime),test['low'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']>= split_date]['time'][window_len:].astype(datetime),((np.transpose(yen_model.predict(test_in)))*test['low'].values[:-window_len])[0],label='Predicted', color='red')
ax1.grid(True)
plt.show()

