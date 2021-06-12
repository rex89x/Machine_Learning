#套件
!pip install psycopg2

import numpy as np
import pandas as pd
import psycopg2
import keras
import sklearn
import re

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

#Sentiment資料集
from google.colab import drive
drive.mount('/content/drive')

#Google Drive授權
data = pd.read_csv('drive/MyDrive/二技資管一甲/下學期/news_dataset/senti_analy/Sentiment.csv')
data = data[['text','sentiment']]


#正負面詞彙資料集整理
data = data[data.sentiment != "Neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)

for idx,row in data.iterrows():
        row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words = max_fatures, split = ' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)


#載入神經網路模型結構權重
from keras.models import load_model

model = Sequential()
model = load_model("senti.h5")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])


#正負面含意分類主要程式

#連結pgSQL
conn = psycopg2.connect(database = "d7eu9qsublnd6e",
                        user = "lvbstcegsalqac",
                        password = "a1001590e6979b1c0dcc8ac371462f756c1b8c5f8db7cb7d3c1df952323bf011",
                        host = "184.73.198.174",
                        port = "5432"
                        )

#資料庫預備
twt = ['']
cur = conn.cursor()
cur.execute("SELECT news_id, news_content, news_title FROM public.bcd_news")
rows = cur.fetchall()

#存取資料庫
for row in rows:
        num = row[0]
        if len(row[1]) == 0:
                twt = row[2]
                twt = tokenizer.texts_to_sequences(twt)
                twt = pad_sequences(twt, maxlen = 28, dtype = 'int32', value = 0)
                sentiment = model.predict(twt, batch_size = 1, verbose = 2)[0]
                if(np.argmax(sentiment) == 0):
                        print("負面")
                        sql = "UPDATE public.bcd_news SET trend = '1' WHERE news_id = %d;" % num
                        cur.execute(sql)
                        conn.commit()
                elif (np.argmax(sentiment) == 1):
                        print("正面")
                        sql = "UPDATE public.bcd_news SET trend = '0' WHERE news_id = %d;" % num
                        cur.execute(sql)
                        conn.commit()
                continue
        else:
                twt = row[1]
                twt = tokenizer.texts_to_sequences(twt)
                twt = pad_sequences(twt, maxlen = 28, dtype = 'int32', value = 0)
                sentiment = model.predict(twt, batch_size = 1, verbose = 2)[0]
                if(np.argmax(sentiment) == 0):
                        print(num)
                        print("負面")
                        #UPDATE public.bcd_news SET trend = NULL WHERE news_id = 1000000008;
                        sql = "UPDATE public.bcd_news SET trend = '1' WHERE news_id = %d;" % num
                        cur.execute(sql)
                        conn.commit()
                elif (np.argmax(sentiment) == 1):
                        print(num)
                        print("正面")
                        sql = "UPDATE public.bcd_news SET trend = '0' WHERE news_id = %d;" % num
                        cur.execute(sql)
                        conn.commit() 
        
conn.close()