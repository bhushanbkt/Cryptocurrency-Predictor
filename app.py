from flask import Flask, render_template, request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import tensorflow as tf
import pandas_datareader as pdr
import requests
import plotly.graph_objs as go
import warnings

app = Flask(__name__)

# Function to convert USD to INR
def usd_to_inr(usd_amount):
    # API endpoint to get the latest exchange rate from USD to INR
    api_endpoint = "https://api.exchangerate-api.com/v4/latest/USD"
    
    try:
        # Fetch the latest exchange rate data
        response = requests.get(api_endpoint)
        data = response.json()
        
        # Extract the exchange rate
        exchange_rate = data['rates']['INR']
        
        # Convert USD to INR
        inr_amount = usd_amount * exchange_rate
        
        return inr_amount
        
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        return None    

# Load models and fetch data
model_btc = tf.keras.models.load_model('btc_model.h5')
model_bnb = tf.keras.models.load_model('bnb_model.h5')
model_ltc = tf.keras.models.load_model('ltc_model.h5')

crypto_1 = 'BTC'
crypto_2 = 'BNB'
crypto_3 = 'LTC'
against_currency = 'USD'

yf.pdr_override()

data_1 = pdr.get_data_yahoo(f'{crypto_1}-{against_currency}', start='2012-1-1', end=dt.datetime.now())
data_2 = pdr.get_data_yahoo(f'{crypto_2}-{against_currency}', start='2017-8-1', end=dt.datetime.now())
data_3 = pdr.get_data_yahoo(f'{crypto_3}-{against_currency}', start='2020-1-1', end=dt.datetime.now())

if data_1.empty or data_2.empty or data_3.empty:
    raise ValueError("Data retrieval failed. Please check the currency symbols or the availability of data.")

data_1['Time'] = np.arange(len(data_1.index))
data_2['Time'] = np.arange(len(data_2.index))
data_3['Time'] = np.arange(len(data_3.index))

data_1['Lag_1'] = data_1['Close'].shift(1)
data_1['Lag_2'] = data_1['Close'].shift(2)
data_1['Lag_3'] = data_1['Close'].shift(3)
data_1['Lag_4'] = data_1['Close'].shift(4)
data_1['Lag_5'] = data_1['Close'].shift(5)
data_1.dropna(inplace=True)

data_2['Lag_1'] = data_2['Close'].shift(1)
data_2['Lag_2'] = data_2['Close'].shift(2)
data_2['Lag_3'] = data_2['Close'].shift(3)
data_2['Lag_4'] = data_2['Close'].shift(4)
data_2['Lag_5'] = data_2['Close'].shift(5)
data_2.dropna(inplace=True)

data_3['Lag_1'] = data_3['Close'].shift(1)
data_3['Lag_2'] = data_3['Close'].shift(2)
data_3['Lag_3'] = data_3['Close'].shift(3)
data_3['Lag_4'] = data_3['Close'].shift(4)
data_3['Lag_5'] = data_3['Close'].shift(5)
data_3.dropna(inplace=True)

prep_data_1 = data_1[['Close', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Time']]
prep_data_2 = data_2[['Close', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Time']]
prep_data_3 = data_3[['Close', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Time']]

split_index1 = int(0.8 * len(prep_data_1))
x1_test = pd.DataFrame(prep_data_1.iloc[split_index1:, 1:])

split_index2 = int(0.8 * len(prep_data_2))
x2_test = pd.DataFrame(prep_data_2.iloc[split_index2:, 1:])

split_index3 = int(0.8 * len(prep_data_3))
x3_test = pd.DataFrame(prep_data_3.iloc[split_index3:, 1:])

def prediction(model, days, data):
    future_steps = days
    future_features = data.iloc[-1:].values.reshape(1, 1, -1, data.shape[1])
    future_predictions = []
    for _ in range(future_steps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        future_pred = model.predict(future_features)
        future_predictions.append(future_pred[0])
        future_features = np.roll(future_features, shift=-1, axis=2)
        future_features[0, 0, -1] = future_pred
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days + 1, freq='D')[1:]
    future_predictions = np.array(future_predictions).flatten()  # Flatten predictions to 1D array
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name='Future Predictions'))
    fig.update_layout(title='Time Series Forecasting', xaxis_title='Time', yaxis_title='Value')
    return fig

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        selected_crypto = request.form['crypto']
        selected_days = int(request.form['days'])
        selected_model = None
        data = None
        
        if selected_crypto == "Bitcoin":
            selected_model = model_btc
            data = x1_test
        elif selected_crypto == "Binance Coin":
            selected_model = model_bnb
            data = x2_test
        elif selected_crypto == "Litecoin":
            selected_model = model_ltc
            data = x3_test
        
        if selected_model is not None:
            prediction_graph = prediction(selected_model, selected_days, data)
            return render_template('prediction.html', prediction_graph=prediction_graph.to_html(full_html=False, include_plotlyjs='cdn'))
        else:
            return "Model not found"
    else:
        return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
