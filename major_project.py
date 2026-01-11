import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('train.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df['Hour'] = df['timestamp'].dt.hour
df['DayOfWeek'] = df['timestamp'].dt.dayofweek
df['wind_speed_cubed'] = df['wind_speed_raw'] ** 3 
df['power_ratio'] = df['active_power_raw'] / (df['wind_speed_raw'] ** 3 + 1e-6)
le = LabelEncoder()
df['turbine_id'] = le.fit_transform(df['turbine_id'])
n = len(df)
x = df[['active_power_calculated_by_converter', 'ambient_temperature', 'generator_speed', 'generator_winding_temp_max', 'grid_power10min_average', 'nc1_inside_temp', 'nacelle_temp', 'reactice_power_calculated_by_converter', 'reactive_power', 'wind_direction_raw', 'wind_speed_turbulence', 'turbine_id', 'Hour', 'DayOfWeek', 'wind_speed_cubed', 'power_ratio']]
y = df['Target']
df = df.reset_index(drop=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

corr = df.corr()
sns.heatmap(corr, cmap='coolwarm')
plt.show()


rf = RandomForestRegressor(random_state=42)
print("Training Random Forest...")
rf.fit(x_train, y_train)
print("Training complete. Now predicting...")
y_pred_rf = rf.predict(x_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest MSE:", mse_rf)
print("Random Forest R²:", r2_rf)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf, param_grid= param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(x_train, y_train)
print (grid_search)
y_pred_grid = grid_search.predict(x_test)
mse = mean_squared_error(y_test, y_pred_grid)
r2 = r2_score(y_test, y_pred_grid)
mae = mean_absolute_error(y_test, y_pred_grid)
print("GridSearchCV RandomForest MAE:", mae)
print("GridSearchCV RandomForest MSE:", mse)
print("GridSearchCV RandomForest R²:", r2)

lr = LinearRegression()
lr.fit(x_train, y_train)    
y_pred_lr = lr.predict(x_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Linear Regression MSE:", mse_lr)
print("Linear Regression R²:", r2_lr)

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.4, label="Linear Regression")
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.4, label="Random Forest")
plt.xlabel("Actual Power Output")
plt.ylabel("Predicted Power Output")
plt.title("Actual vs Predicted: Linear Regression vs Random Forest")
plt.legend()
plt.show()
plt.figure(figsize=(8,4))
sns.histplot(y_test - y_pred_lr, color="blue", label="LR Residuals", kde=True)
sns.histplot(y_test - y_pred_rf, color="orange", label="RF Residuals", kde=True)
plt.title("Residual Distribution Comparison")
plt.xlabel("Residuals (Actual - Predicted)")
plt.legend()
plt.show()
feature_importance = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance - Random Forest")
plt.show()
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:200], label='Actual', color='black')
plt.plot(y_pred_lr[:200], label='LR Predicted', color='blue')
plt.plot(y_pred_rf[:200], label='RF Predicted', color='orange')
plt.title("Actual vs Predicted (First 200 Samples)")
plt.xlabel("Time Step")
plt.ylabel("Power Output")
plt.legend()
plt.show()
