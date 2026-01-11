import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

sns.set()

def load_dfs():
    df1 = pd.read_csv('Book1.csv')
    df2 = pd.read_csv('Book2.csv')
    return {'Book1': df1, 'Book2': df2}

def infer_target(df):
    # prefer common names
    for name in ['target', 'y', 'value', 'Value', 'target_value', 'amount']:
        if name in df.columns:
            return name
    # else choose last numeric column
    numcols = df.select_dtypes(include=[np.number]).columns
    if len(numcols) == 0:
        return None
    return numcols[-1]

def preprocess_for_regression(df, target):
    df = df.copy()
    # drop rows where target is missing
    df = df[df[target].notna()]
    # simple imputation
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    catcols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in numcols:
        df[c] = df[c].fillna(df[c].median())
    for c in catcols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "missing")
    X = df.drop(columns=[target])
    y = df[target].values
    return X, y, numcols, catcols

def build_and_evaluate_regression(X, y, name):
    # basic pipeline: onehot categorical -> RF and Linear
    catcols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numcols = X.select_dtypes(include=[np.number]).columns.tolist()
    preproc = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), catcols)
    ], remainder='passthrough')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Linear Regression
    lr_pipe = Pipeline([('pre', preproc), ('lr', LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    # Random Forest
    rf_pipe = Pipeline([('pre', preproc), ('rf', RandomForestRegressor(n_estimators=100, random_state=42))])
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    def metrics(y_true, y_pred):
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    print(f"\n--- Regression results for {name} (target variable) ---")
    print("LinearRegression:", metrics(y_test, y_pred_lr))
    print("RandomForestRegressor:", metrics(y_test, y_pred_rf))
    # scatter plot predicted vs actual for RF
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred_rf, s=10, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name} - RF Predicted vs Actual')
    plt.tight_layout()
    plt.show()
    return lr_pipe, rf_pipe

def infer_datetime_column(df):
    # returns name of datetime-like column or None
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
    # try common names and parsing
    for name in ['date', 'Date', 'timestamp', 'time', 'Time', 'month']:
        if name in df.columns:
            try:
                _ = pd.to_datetime(df[name])
                return name
            except:
                pass
    # try to parse any string column with parseable dates (first few rows)
    for col in df.select_dtypes(include=['object']).columns:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            if parsed.notna().sum() > 0.6 * len(parsed):  # mostly parseable
                return col
        except:
            pass
    return None

def timeseries_workflow(df, name):
    ts_col = infer_datetime_column(df)
    if ts_col is None:
        print(f"\n{name}: No datetime column detected. Skipping time series steps.")
        return
    df_ts = df.copy()
    df_ts[ts_col] = pd.to_datetime(df_ts[ts_col], errors='coerce')
    df_ts = df_ts.set_index(ts_col).sort_index()
    # find numeric columns to model individually
    numcols = df_ts.select_dtypes(include=[np.number]).columns.tolist()
    if len(numcols) == 0:
        print(f"{name}: no numeric columns for time series.")
        return
    print(f"\n--- Time series for {name} (datetime: {ts_col}) ---")
    for col in numcols:
        series = df_ts[col].dropna()
        if len(series) < 24:
            print(f"{name} - {col}: too few observations ({len(series)}), skipping.")
            continue
        print(f"\n{name} - {col}: length={len(series)}")
        # decomposition
        try:
            dec = seasonal_decompose(series, model='additive', period=max(1, int(len(series)/6)))
            dec.plot()
            plt.suptitle(f'{name} - {col} decomposition', y=1.02)
            plt.show()
        except Exception as e:
            print(f"decompose failed for {col}: {e}")
        # adfuller
        adf_p = adfuller(series.dropna())[1]
        print(f"ADF p-value: {adf_p:.4f} -> {'stationary' if adf_p<=0.05 else 'non-stationary'}")
        # simple ARIMA fit (order chosen small); wrap in try
        try:
            order = (1,1,1)
            model = ARIMA(series, order=order)
            res = model.fit()
            print(res.summary().tables[1])
            # forecast next 12 steps
            n_forecast = min(24, max(12, int(len(series)*0.1)))
            forecast = res.get_forecast(steps=n_forecast)
            fc_mean = forecast.predicted_mean
            fc_ci = forecast.conf_int()
            plt.figure(figsize=(8,4))
            plt.plot(series.index, series.values, label='history')
            plt.plot(fc_mean.index, fc_mean.values, label='forecast', color='orange')
            plt.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], color='orange', alpha=0.2)
            plt.title(f'{name} - {col} ARIMA forecast')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"ARIMA failed for {col}: {e}")

def main():
    dfs = load_dfs()
    for name, df in dfs.items():
        print(f"\n########## Processing {name} ##########")
        print(df.head())
        print(df.info())
        print(df.describe())
        # Regression
        target = infer_target(df)
        if target is None:
            print(f"{name}: No numeric target found for regression. Skipping regression.")
        else:
            print(f"{name}: Inferred target -> {target}")
            X, y, numcols, catcols = preprocess_for_regression(df, target)
            try:
                build_and_evaluate_regression(X, y, name)
            except Exception as e:
                print(f"Regression failed for {name}: {e}")
        # Time series
        try:
            timeseries_workflow(df, name)
        except Exception as e:
            print(f"Time series failed for {name}: {e}")

