# -*- coding: utf-8 -*-
"""
Road Collisions SVC 
Author: James Khonje
Date: 09/12/2024


"""
#%%

# Load necessary ibraries/packages
# ============================================
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# Package for stationarity check
from statsmodels.tsa.stattools import adfuller

# Packages for statistical and ML calculations/models 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error


from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


# GRU Model

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense

# FB Prophet
from prophet import Prophet


import warnings
warnings.filterwarnings('ignore')

#%%

# Import data
data = pd.read_csv("..\\data\\Wales_data_Monthly.csv")

# Convert Date variable to consistent date format
# =============================================================================

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df.set_index('Date', inplace=True)

df.head()

#%%

# Creating data for decomposistion
df_comparisonData = df.copy()

# decomposition using Additive model -The

decomposition = sm.tsa.seasonal_decompose(df, model='additive')

fig = decomposition.plot()

plt.show()

# The plot shows that our raw data is not stationary as indicated by 

#%%

# Autocorrelation checks
# =============================================================================

sm.graphics.tsa.plot_acf(df)
plt.show()  # Use AR = 1

sm.graphics.tsa.plot_pacf(df)
plt.show() # Use MA = 1


#%%

# =============================================================================
# SARIMA (Seasonal Autoregressive Integrated Moving Average) 
# =============================================================================
    
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]


# Stationarity Test - Is our data stationary in levels?
# =============================================================================

# Augmented Dickey-Fuller (ADF) test will be used to check for trend non-stationarity.

# H0: There is unit root (data is not stationary)
# H1: Dataset is stationary

adf_test = adfuller(train['Collisions'].dropna())

print(f'ADF test Result:, {round(adf_test[0],4)}')
print(f'p-value: {round(adf_test[1],4)}')

# As the p-value is greater than 0.05% data is non stationary
# Differentiation : As our data is non stationary in levels we have to differentiate our data to make it stationary

# Function to use first difference if the p-value > 0.05 otherwise proceed

if adf_test[1] > 0.05:  # i.e if the p-value is greater than 0.05
    train_diff = train['Collisions'].diff().dropna()
    model = SARIMAX(train_diff, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
else:
    model = SARIMAX(train['Collisions'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))


# Fit SARIMA model on the differenced data to estimate model parameters
model_fit = model.fit()


forecast_values_diff = model_fit.forecast(steps=len(test))

# Invense differentiation to get actual forecast values

last_value = train['Collisions'].iloc[-1]
forecast_values = forecast_values_diff.cumsum() + last_value


# Actual values for the test period 
actual_values = test['Collisions'].values 

# Check if the length are the same
len(actual_values) == len(forecast_values)

# Calculate error metrics 
mse_sarima = mean_squared_error(actual_values, forecast_values) 
rmse_sarima = np.sqrt(mse_sarima) 
mae_sarima = mean_absolute_error(actual_values, forecast_values) 


# Create a DataFrame to calculate MAPE and sMAPE
Col_svc_df_diff = pd.DataFrame({ 
    'Actual': actual_values, 
    'Forecast': forecast_values
    })

mape_sarima = np.mean(np.abs((Col_svc_df_diff.Actual - Col_svc_df_diff.Forecast) / Col_svc_df_diff.Actual)) * 100

# sMAPE Calculation 

def smape(y_true, y_pred): 
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) 

smape_sarima = smape(Col_svc_df_diff.Actual, Col_svc_df_diff.Forecast) 

#%%

# =============================================================================
# FBProphet
# =============================================================================


# Prepare data for Prophet
df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'Collisions': 'y'})

# Fit the model
model_prophet = Prophet()
model_prophet.fit(df_prophet)

# Forecast
future = model_prophet.make_future_dataframe(periods=12, freq='M')
forecast_prophet = model_prophet.predict(future)
forecast_prophet = forecast_prophet.set_index('ds')['yhat'][-12:]

# Metrics

mse_prophet = mean_squared_error(df_prophet['y'][-12:], forecast_prophet)
rmse_prophet = np.sqrt(mse_prophet)
mae_prophet = mean_absolute_error(df_prophet['y'][-12:], forecast_prophet)
mape_prophet = np.mean(np.abs((df_prophet['y'][-12:] - forecast_prophet) / df_prophet['y'][-12:])) * 100


# Create a DataFrame to calculate MAPE and sMAPE
collision_svc_df_FB = pd.DataFrame({ 
    'Actual': df_prophet['y'][-12:], 
    'Forecast': forecast_prophet.values
    })

mape_prophet = np.mean(np.abs((collision_svc_df_FB.Actual - collision_svc_df_FB.Forecast) / collision_svc_df_FB.Actual)) * 100

smape_Prophet = smape(collision_svc_df_FB.Actual, collision_svc_df_FB.Forecast) 

# FB Prophet forecast

future_prophet = model_prophet.make_future_dataframe(periods=12, freq='MS')  # Forecast for the next 12 months
prophet_forecast = model_prophet.predict(future_prophet)

# Plot the forecast
fig = model_prophet.plot(prophet_forecast)
plt.ylabel('Number of Collisions', fontweight='bold', fontsize = 14)
plt.xlabel("Year", fontweight='bold', fontsize = 14)
plt.grid()
sns.despine(left=False, bottom=False)
plt.legend()
plt.show()


#%%

# =============================================================================
# LSTM (Long short term memory)
# =============================================================================

# Install the following packages first to run LSTM (Important)

# pip install keras
# pip install tensorflow
# -----------------------

from keras.models import Sequential
from keras.layers import LSTM, Dense

# Use MinMax scaling - important step to avoid having some data dominate 
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12
X, y = create_sequences(scaled_data, seq_length)

# Split into train and test
X_train, y_train = X[:-12], y[:-12]
X_test, y_test = X[-12:], y[-12:]

# Build and train the LSTM model
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32)

# Forecast
predictions_lstm = model_lstm.predict(X_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)

# Metrics
mse_lstm = mean_squared_error(y_test, predictions_lstm)
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm = mean_absolute_error(y_test, predictions_lstm)
mape_lstm = np.mean(np.abs((y_test - predictions_lstm) / y_test)) * 100

# sMAPE Calculation

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

msle_lstm = mean_squared_log_error(y_test, predictions_lstm)
smape_lstm = smape(y_test, predictions_lstm)

#%%

# =============================================================================
# XGBoost (Extreme Gradient Boosting)
# =============================================================================

# Create lag features
def create_lag_features(df, lag=12):
    df_lagged = df.copy()
    for i in range(1, lag + 1):
        df_lagged[f'lag_{i}'] = df_lagged['Collisions'].shift(i)
    df_lagged.dropna(inplace=True)
    return df_lagged

df_lagged = create_lag_features(df)

# Prepare the data for modeling
X = df_lagged.drop(columns=['Collisions'])
y = df_lagged['Collisions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize the model
model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=1000)

# Fit the model
model_xgb.fit(X_train, y_train)

# Forecast
y_pred = model_xgb.predict(X_test)

# Calculate the performance metrics
mse_xgb = mean_squared_error(y_test, y_pred)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred)
mape_xgb = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# sMAPE Calculation
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

msle_xgb = mean_squared_log_error(y_test, y_pred)
smape_xgb = smape(y_test, y_pred)


#%%

# =============================================================================
## GRU (Gated Recurrent Unit) Model
# =============================================================================

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12
X, y = create_sequences(scaled_data, seq_length)

# Split into train and test

X_train, y_train = X[:-12], y[:-12]
X_test, y_test = X[-12:], y[-12:]


# Build and train the GRU model
# ------------------------------

model_gru = Sequential([
    GRU(50, return_sequences=True, input_shape=(seq_length, 1)),
    GRU(50, return_sequences=False),
    Dense(1)
])

model_gru.compile(optimizer='adam', loss='mse')
model_gru.fit(X_train, y_train, epochs=20, batch_size=32)

# Forecast
predictions_gru = model_gru.predict(X_test)
predictions_gru = scaler.inverse_transform(predictions_gru)

# Evaluate the model
mse_gru = mean_squared_error(y_test, predictions_gru)
rmse_gru = np.sqrt(mse_gru)
mae_gru = mean_absolute_error(y_test, predictions_gru)
mape_gru = np.mean(np.abs((y_test - predictions_gru) / y_test)) * 100

# sMAPE Calculation
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

msle_gru = mean_squared_log_error(y_test, predictions_gru)
smape_gru = smape(y_test, predictions_gru)

#%%

# =============================================================================
# Results for all Models using common error metrics
# =============================================================================

print("==" * 15)
print(f'SARIMA  \nMSE: {mse_sarima:.4f}, \nRMSE: {rmse_sarima:.4f}, \nMAE: {mae_sarima:.4f}, \nMAPE: {mape_sarima:.4f}%, \nsMAPE: {smape_sarima:.4f}%')
print("==" * 15)
print(f'Prophet  \nMSE: {mse_prophet:.4f}, \nRMSE: {rmse_prophet:.4f}, \nMAE: {mae_prophet:.4f}, \nMAPE: {mape_prophet:.4f}%, \nsMAPE: {smape_Prophet:.4f}%')
print("==" * 15)
print(f'LSTM  \nMSE: {mse_lstm:.4f}, \nRMSE: {rmse_lstm:.4f}, \nMAE: {mae_lstm:.4f}, \nMAPE: {mape_lstm:.4f}%, \nsMAPE: {smape_lstm:.4f}%, \nMSLE: {msle_lstm:.4f}')
print("==" * 15)
print(f'GRU  \nMSE: {mse_gru:.4f}, \nRMSE: {rmse_gru:.4f}, \nMAE: {mae_gru:.4f}, \nMAPE: {mape_gru:.4f}%, \nsMAPE: {smape_gru:.4f}%, \nMSLE: {msle_gru:.4f}')
print("==" * 15)
print(f'XGBoost  \nMSE: {mse_xgb:.4f}, \nRMSE: {rmse_xgb:.4f}, \nMAE: {mae_xgb:.4f}, \nMAPE: {mape_xgb:.4f}%, \nsMAPE: {smape_xgb:.4f}%, \nMSLE: {msle_xgb:.4f}')
print("==" * 15)

#%%

# ===================================== THE END ===============================



