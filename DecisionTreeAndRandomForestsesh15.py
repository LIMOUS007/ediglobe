import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import tree
df = pd.read_csv("customer_churn.csv")
print (df.head()) 
print (df.shape)
print (df.info())
print (df.isnull().sum())
print (df)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")
print (df.isnull().sum())
df.dropna(inplace = True)
print (df.info())
df.drop(columns = ["customerID"], inplace = True)
print (df.duplicated().sum())
df.drop_duplicates(inplace = True)
print (df.shape)
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f'Column: {col}')
        print(f'Original---> Endcoded')
        for orig, enc in mapping.items():
            print(f'{orig} ---> {enc}')
        print()
print (df.head())
sns.histplot(data=df,x='tenure',hue='Churn',element='step',bins=30)
plt.show()
sns.kdeplot(data=df,x='MonthlyCharges',hue='Churn',common_norm=False, fill= True)
plt.show()
from sklearn.model_selection import train_test_split
x = df.drop(columns = ["Churn"], axis=1)
y = df["Churn"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print (x_train)
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_train)
y_pred = log_model.predict(x_test)
print (y_pred)
from sklearn.metrics import *
print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))
print (classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
print (classification_report(y_test, y_pred_dt))
print (confusion_matrix(y_test, y_pred_dt))
print (accuracy_score(y_test, y_pred_dt))
rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_train)
print (classification_report(y_train, y_pred_rf))
print (confusion_matrix(y_train, y_pred_rf))
print (accuracy_score(y_train, y_pred_rf))
from sklearn.model_selection import GridSearchCV
paraGrid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=paraGrid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(x_train, y_train)
print (grid_search)
y_pred_grid = grid_search.predict(x_test)
print (classification_report(y_test, y_pred_grid))
print (confusion_matrix(y_test, y_pred_grid))
print (accuracy_score(y_test, y_pred_grid))


