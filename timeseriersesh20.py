import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import itertools
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv('AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
print (df.head)
print (df.tail)
print(df.info())
print (df.shape)
plt.plot(df['#Passengers'])
plt.show()
decompose = seasonal_decompose(df["#Passengers"])
decompose.plot()
plt.show()
result = adfuller(df['#Passengers'])
print (result[1])
if result[1] <= 0.05:
    print('Null hypo is reject')
else:
    print("Null hypo is accepted")
rollingMean = df.rolling(window=12).mean()
rollingStd = df.rolling(window = 12).std()
plt.plot(df, color='blue')
plt.plot(rollingMean, color='red')
plt.plot(rollingStd, color = 'black')
plt.show()
log_df = np.log(df)
print (log_df)
rollingMeanlog = log_df.rolling(window=12).mean()
rollingStdlog = log_df.rolling(window = 12).std()
plt.plot(log_df, color='blue')
plt.plot(rollingMeanlog, color='red')
plt.plot(rollingStdlog, color = 'black')
plt.show()
diff_data = log_df.diff()
diff_data.dropna(inplace = True)
print (diff_data)
rollingMeandiff = diff_data.rolling(window=12).mean()
rollingStddiff = diff_data.rolling(window = 12).std()
plt.plot(diff_data, color='blue')
plt.plot(rollingMeandiff, color='red')
plt.plot(rollingStddiff, color = 'black')
plt.show()
resultd = adfuller(diff_data['#Passengers'])
print (resultd[1])
if resultd[1] <= 0.05:
    print('Null hypo is reject')
else:
    print("Null hypo is accepted")
shift_data = diff_data.shift(-1)
shift_data.dropna(inplace=True)
resultsh = adfuller(shift_data['#Passengers'])
print(resultsh[1])
print (resultsh)
pVal = resultsh[1]
if pVal <= 0.05:
  print('Null hypo is rejcected')
else:
  print('Null hypo acccepted')
train = log_df.iloc[:120, :]
test = log_df.iloc[120 : , : ]
modela = ARIMA(train, order = (1,1,2))
modela = modela.fit()
log_df['Arima Pred'] = modela.predict(start = len(train), end = len(train) + len(test) - 1)
print (log_df)
plt.plot(log_df)
plt.show()
p = range(1, 8)
d = range(1, 2)
q = range(1, 8)
pdq_combination = list(itertools.product(p, d, q))
print (pdq_combination)
print (len(pdq_combination))
rmse = []
order1 = []
for pdq in pdq_combination:
  model = ARIMA(train, order = pdq)
  model_fit = model.fit()
  pred = model_fit.predict(start = len(train), end = len(train) + len(test) - 1)
  error = np.sqrt(mean_squared_error(test, pred))
  order1.append(pdq)
  rmse.append(error)
result = pd.DataFrame(index=order1, data=rmse, columns=['RMSE'])
result.sort_values(by='RMSE',ascending=True)
print (result)
model = ARIMA(train, order = (5, 1, 4))
model = model.fit()
log_df['Arima Pred'] = model.predict(start = len(train), end = len(train) + len(test) - 1)
print (log_df)
plt.plot(log_df)
plt.show()
s_model = SARIMAX(train,order = (5,1,4),seasonal_order=(5,1,4,12))
s_model = s_model.fit()
log_df['Sarima Pred'] = s_model.predict(start = len(train), end = len(train) + len(test) - 1)
print (log_df)
plt.plot(log_df['#Passengers'])
plt.plot(log_df['Sarima_pred'])
plt.show()
plt.plot(log_df)
plt.legend('best')
plt.show()
future = s_model.forecast(steps = 60)
plt.plot(log_df['#Passengers'])
plt.plot(future)
plt.show()
