from matplotlib import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr # for finance (stock price)
import datetime as dt
import tensorflow as tf
import yfinance as yf
import fix_yahoo_finance as fyf

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout, LSTM, Dense
from keras.models import Sequential

#specifying real and crypto currency
crypto_currency='BTC'  # 'ETH' 
real_currency='INR'

start=dt.datetime(2010,1,1)
end=dt.datetime.now()

print(end)

data=pdr.DataReader(f'{crypto_currency}-{real_currency}', 'yahoo', start, end)
data= pdr

print(data.head())

scaler=MinMaxScaler(feature_range=(0,1))
scaled_Data=scaler.fit_transform(data['Close'].values.reshape(-1,1))

predict_days=120
x_train, y_train=[], []

for x in range(predict_days, len(scaled_Data)):
    x_train.append(scaled_Data[x-predict_days:x, 0])
    y_train.append(scaled_Data[x, 0])

x_train, y_train=np.array(x_train), np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))


model=Sequential()
model.add(LSTM(units =50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units= 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units= 50))
model.add(Dropout(0.2))
model.add(Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


#Testing model

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()



test_data=pdr.DataReader(f'{crypto_currency}-{real_currency}', 'yahoo', test_start, test_end)
actual_price =test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)


model_input = total_dataset[len(total_dataset) - len(test_data) - predict_days:].values
model_input = model_input.reshape(-1,1)
model_input = scaler.fit_transform(model_input)


x_test = []

for x in range(predict_days, len(model_input)):
    x_test.append(model_input[x-predict_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predict_price = model.predict(x_test)
predict_price = scaler.inverse_transform(predict_price)

plt.plot(actual_price, color='black', label='Actual_Price')
plt.plot(predict_price, color='green', label='Predict_Price')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
