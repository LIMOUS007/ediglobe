import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

insurance = pd.read_csv("insurance.csv")
print (insurance.head())
print (insurance.shape)
print (insurance.info())
print (insurance.isnull().sum())
col = list(insurance.columns)
print (col)
for cname in col:
    if (insurance[cname].dtype == 'float64'):
        plt.boxplot(insurance[cname])

num_col = insurance.select_dtypes(include = ["number"])
print (num_col)
print (num_col.corr())
corrMat = num_col.corr()
sns.heatmap(corrMat, annot = True, cmap="Greens")
plt.show()

from sklearn.preprocessing import LabelEncoder
cat_col = insurance.select_dtypes(include = ["object"])
print (cat_col)
le = LabelEncoder()
for col in cat_col:
    insurance[col] = le.fit_transform(insurance[col])

print (insurance)
from sklearn.model_selection import train_test_split
x = insurance.drop('charges' , axis = 1) 
y = insurance['charges']
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8 , random_state = 0)
print (x_train)
print (x_test)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print (y_pred)
from sklearn.metrics import r2_score, root_mean_squared_error
print (root_mean_squared_error(y_test, y_pred))
print (r2_score(y_test, y_pred))
print (model.coef_)
print (model.intercept_)
sns.regplot(x=y_pred, y=y_test, ci=None, scatter_kws={"color":"r", "s":9})
plt.xlabel("Predicted charges")
plt.ylabel("Actual charges")
plt.title("Predicted vs Actual charges")
plt.show()
