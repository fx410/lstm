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
def iso_jp(iso):
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

end = "2019-03-01T00:00:00.000000Z"
end = datetime.strptime(end, '%Y-%m-%dT%H:%M:%S.%fZ')
start = []
for i in range(300):
	  end = end + timedelta(days=-1)
	  weekday = end.weekday()
	  if weekday<5:
		    start.append(end.isoformat())

data = pd.DataFrame([])
length = 100
for i in range(length):
     res_5m = oanda.get_history(instrument="USD_JPY", granularity="M5", start=start[i+1] + ".000000Z", end=start[i] + ".000000Z")
     tmp = pd.DataFrame(res_5m['candles'])
     data = pd.concat([data, tmp]).reset_index(drop=True)

print(data.shape)

#見やすいように文字列に変換
def date_to_str(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')

 #日付の調整
data['time'] = data['time'].apply(lambda x:date_to_str(iso_jp(x)))
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values(by='time')

 
 #特徴量の整理
df = data[['time', 'openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']]
df.columns = ['time', 'open', 'close', 'high', 'low', 'volume']

print(df.head())

 #訓練データとテストデータへの分割
split_date = '2019-01-18 22:55:00'
train, test = df[df['time'] < split_date], df[df['time'] >= split_date]
del train['time']
del test['time']
 
 #ウィンドウの数
window_len = 12

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
yen_history = yen_model.fit(train_in, train_out, epochs=12, batch_size=1, verbose=2, shuffle=True)

model_json = yen_model.to_json()
open('usd_5m_model.json', 'w').write(model_json)
yen_model.save_weights('usd_5m_weights.h5')

 #モデルによる予測と実測の比較
fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']< split_date]['time'][window_len:].astype(datetime),train['close'][window_len:],label='Actual', color='blue')
ax1.plot(df[df['time']< split_date]['time'][window_len:].astype(datetime),((np.transpose(yen_model.predict(train_in))) * train['close'].values[:-window_len])[0],label='Predicted', color='red')
plt.show()

fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']>= split_date]['time'][window_len:].astype(datetime),test['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']>= split_date]['time'][window_len:].astype(datetime),((np.transpose(yen_model.predict(test_in)))*test['close'].values[:-window_len])[0],label='Predicted', color='red')
ax1.grid(True)
plt.show()

