x = 3
sig = 1/(1+2.71828**-x)
print (sig)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("bank-additional-full.csv", sep=';')
print (df.head())
print (df.shape)
print (df.info())
print (df) 
print (df.isnull().sum().sum())
df.dropna(inplace = True)
print (df.isnull().sum().sum())
dup = df[df.duplicated()]
print (dup)
df.drop_duplicates(inplace = True)
print (df.duplicated().sum())
print (df.shape)
for col in df.columns:
    if df[col].dtype != 'object':
        plt.boxplot(df[col])
        plt.title(col)
        plt.show()
outcols = ['age' ,'duration' ,'campaign', 'cons.conf.idx']
for col in outcols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
for col in df.columns:
  if df[col].dtype != 'object':
    plt.boxplot(df[col])
    plt.title(col)
    plt.show()
    print (df.shape)
  catCol = []
  for col in df.columns:
    if df[col].dtype == 'object':
        catCol.append(col)
  
print (catCol)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lableMap = {}
for col in catCol:
    df[col] = le.fit_transform(df[col])
    lableMap[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print (f'Mapped for {col}: {lableMap[col]}')
print (df)
print (df.head())
x= df.drop(columns= ['y'])
print (x)
y = df['y']
print (y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,random_state=42)
print (x_train)
print (x_test)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print (y_pred)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))
print (classification_report(y_test, y_pred))
def sigmoid(x):
  result = 1/(1+np.exp(-x))
  return result
y_score = model.predict_proba(x_test)[:,1]
print (y_score)
sorInd = np.argmax(y_score)
print (sorInd)
sortLable = y_test.iloc[sorInd]
print (sortLable)
sortScore = y_score[sorInd]
print (sortScore)
x_value = np.linspace(-10,10,100)
y_sigmoid = sigmoid(x_value)
print (y_sigmoid)
plt.plot(x_value, y_sigmoid)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.scatter(sortScore, sortLable)
plt.title('Sigmoid Function')
plt.xlabel('X')
plt.ylabel('Sigmoid(X)')
plt.show()
