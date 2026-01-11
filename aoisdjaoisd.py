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
print (df.head())
print (df.info())
print (df.columns)