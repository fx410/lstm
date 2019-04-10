 #ライブラリ
import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
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
 #為替のデータ
res = pd.DataFrame(oanda.get_history(instrument="EUR_JPY", granularity="D")['candles'])
print(res[-1:]) 

 #特徴量の整理
df = res[['openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']]
df.columns = ['open', 'close', 'high', 'low', 'volume']

 #最新windowのデータ
df = df[-10:]

 #入力データの正規化
lstm_in = df.copy()
for col in lstm_in:
    lstm_in.loc[:, col] = lstm_in[col] / lstm_in[col].iloc[0] - 1
lstm_in = [lstm_in]

 #numpy型に変更
lstm_in = [np.array(tmp) for tmp in lstm_in]
lstm_in = np.array(lstm_in)
 
  #モデルの読み込み
model = model_from_json(open('eur_model.json').read())
 #重みの読み込み
model.load_weights('eur_weights.h5')
 #モデルの要約？ 
model.summary();

model.compile(loss='mae', optimizer='adam')
pred = model.predict(lstm_in, batch_size=1)[0][0]
pred_yen = pred * (df['close'].values[:])[0]
print(pred_yen)
