import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
import warnings
import tensorflow as tf
import requests
import plotly.graph_objs as go

# Function to store user credentials
def store_credentials(username, password):
    with open("credentials.txt", "a") as file:
        file.write(f"{username}:{password}\n")

# Function to authenticate user
def authenticate(username, password):
    with open("credentials.txt", "r") as file:
        for line in file:
            stored_username, stored_password = line.strip().split(":")
            if stored_username == username and stored_password == password:
                return True
    return False

# Function to check if a username exists
def check_username(username):
    with open("credentials.txt", "r") as file:
        for line in file:
            stored_username, _ = line.strip().split(":")
            if stored_username == username:
                return True
    return False

# Function to convert USD to INR
def usd_to_inr(usd_amount):
    api_endpoint = "https://api.exchangerate-api.com/v4/latest/USD"
    try:
        response = requests.get(api_endpoint)
        data = response.json()
        exchange_rate = data['rates']['INR']
        inr_amount = usd_amount * exchange_rate
        return inr_amount
    except Exception as e:
        st.error(f"Error fetching exchange rate: {e}")
        return None    

# Function to make predictions and display plot
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
    future_predictions = np.array(future_predictions).flatten()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name='Future Predictions'))
    fig.update_layout(title='Time Series Forecasting', xaxis_title='Time', yaxis_title='Value')
    st.plotly_chart(fig)
    return future_pred[0]

# Streamlit UI for login
def login():
    st.markdown("<p style='font-family:Serif Fonts'>Welcome to our Cryptocurrency Predictor platform! Whether you're a seasoned investor or just starting out, our tools can help you forecast the future prices of top cryptocurrencies. Sign up or log in to unlock the full potential of our platform and start making informed investment decisions today.</p>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.success(f"You are now logged in as {username}")
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password.")



# Streamlit UI for sign up
def signup():
    st.title("Sign Up")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        if check_username(new_username):
            st.error("Username already exists.")
        else:
            store_credentials(new_username, new_password)
            st.success(f"You have successfully signed up as {new_username}")

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        try:
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
                future_predictions = np.array(future_predictions).flatten()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name='Future Predictions'))
                fig.update_layout(title='Time Series Forecasting', xaxis_title='Time', yaxis_title='Value')
                st.plotly_chart(fig)
                return future_pred[0]

            st.title("Cryptocurrency Predictor")
            st.markdown("<p style='font-family:Serif Fonts; font-size:24px'>Unlock Insights: Predict Future Prices of Leading Cryptocurrencies.</p>", unsafe_allow_html=True)
  
            
            crypto_options = ['Bitcoin', 'Binance Coin', 'Litecoin']
            selected_crypto = st.sidebar.selectbox('Select Cryptocurrency:', crypto_options)
            st.image('sc.jpg',use_column_width=True)
            st.subheader("Select Number of Days From Today")
            selected_days = st.slider("Number of Days", min_value=1, max_value=30, value=7)

            st.write(f"Selected Cryptocurrency: {selected_crypto}")
            st.write(f"Selected Number of Days: {selected_days}")

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
                try:
                    res = prediction(selected_model, selected_days, data)

                    # Convert predicted price from USD to INR
                    predicted_price_usd = res[0]
                    predicted_price_inr = usd_to_inr(predicted_price_usd)
                    if predicted_price_inr is not None:
                        st.write(f"Predicted {selected_crypto} Price (USD): {predicted_price_usd}")
                        st.write(f"Predicted {selected_crypto} Price (INR): {predicted_price_inr}")
                    else:
                        st.error("Failed to fetch exchange rate. Please try again later.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Model not found for the selected cryptocurrency.")
            st.sidebar.image('bitcoin-11225_512.gif')
            if st.sidebar.button("Logout"):
                st.session_state.logged_in = False
                st.success("You have been logged out.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.title("Login or Sign Up")
        menu = ["Login", "SignUp"]
        choice = st.sidebar.selectbox("Menu", menu)
        
        if choice == "Login":
            login()
        elif choice == "SignUp":
            signup()

if __name__ == "__main__":
    main()



