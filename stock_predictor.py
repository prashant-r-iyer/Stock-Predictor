# Import libraries
import yfinance as yf

from copy import deepcopy as dc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import streamlit as st

from plotly import graph_objs as go


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Class for our data
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]


# LSTM class
class LSTM(nn.Module):
    def __init__(self, input, hidden, stacked):
        super().__init__()

        self.hidden = hidden
        self.stacked = stacked

        self.lstm = nn.LSTM(input, hidden, stacked, batch_first=True)

        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.stacked, batch_size, self.hidden).to(device)
        c0 = torch.zeros(self.stacked, batch_size, self.hidden).to(device)

        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])

        return x


# Train/test loop
def train_model(x_train, x_test, y_train, y_test, retrain, previous_loading_text):
    train_set = TimeSeriesDataset(x_train, y_train)

    if not retrain:
        test_set = TimeSeriesDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    if not retrain:
        test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    model = LSTM(1, 4, 1)
    model.to(device)

    lr = 0.001
    epochs = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    previous_loading_text.text("Feeding data into model... Done!")
    loading_text = st.text("Training model... initializing")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = model(x_batch)

            loss = loss_fn(output, y_batch)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_loader)

        if not retrain:
            model.train(False)
            total_loss = 0

            for batch_index, batch in enumerate(test_loader):
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)

                with torch.no_grad():
                    output = model(x_batch)

                    loss = loss_fn(output, y_batch)
                    total_loss += loss.item()

            average_loss = total_loss / len(test_loader)

        loading_text.text(f"Training model... Epoch {epoch}/10")

    return model, loading_text


st.title("Stock Predictor")

st.write("Welcome to Stock Predictor! We will use a Long Short-Term Network (LSTM) to forecast stocks!")

stocks = ("AMZN", "AAPL", "GOOG", "MSFT")
stock_input = st.selectbox("Select stock for prediction", stocks)

stock_button = st.button("Confirm")

if stock_button:
    st.subheader("Raw data")

    loading_text_0 = st.text("Loading raw data...")

    # Load data from Yahoo Finance
    stock_data = yf.Ticker(stock_input).history(period="max")

    stock_data_copy = dc(stock_data)
    stock_data_copy.reset_index(inplace=True)

    st.write(stock_data_copy)

    # Display data
    stock_plot = go.Figure()
    stock_plot.add_trace(go.Scatter(x=stock_data_copy['Date'], y=stock_data_copy['Close'], name='Close value'))
    stock_plot.layout.update(title_text='Stock Data (it\'s interactive!)', xaxis_rangeslider_visible=True)

    st.plotly_chart(stock_plot)

    loading_text_0.text("Loading raw data... Done!")

    loading_text_1 = st.text("Feeding data into model...")

    # Process data
    stock_data['Date'] = stock_data.index
    stock_data = stock_data[['Date', 'Close']]
    stock_data = stock_data.reset_index(drop=True)

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    df = dc(stock_data)
    df.set_index('Date', inplace=True)

    # Create lookback columns
    lookback = 7
    for i in range(lookback):
        df[f'Close(t-{i + 1})'] = df['Close'].shift(i + 1)

    df.dropna(inplace=True)

    data = df.to_numpy()

    # Scale data and prepare it for model
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)

    x = scaled_data[:, 1:]
    y = scaled_data[:, 0]

    x = dc(np.flip(x, axis=1))

    split_index = int(len(x) * 0.95)

    x_train = x[:split_index]
    x_test = x[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    x_train = x_train.reshape((-1, lookback, 1))
    x_test = x_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    model, loading_text_2 = train_model(x_train, x_test, y_train, y_test, False, loading_text_1)

    # Predicted train values
    with torch.no_grad():
        train_predicted = model(x_train.to(device)).to('cpu').numpy()

    temp = np.zeros((x_train.shape[0], lookback + 1))
    temp[:, 0] = train_predicted.flatten()
    temp = scaler.inverse_transform(temp)

    rescaled_train_predictions = dc(temp[:, 0])

    temp = np.zeros((x_train.shape[0], lookback + 1))
    temp[:, 0] = y_train.flatten()
    temp = scaler.inverse_transform(temp)

    rescaled_y_train = dc(temp[:, 0])

    loading_text_2.text("Training model... Done!")
    st.subheader("Training results")

    st.write("This is how our LSTM model performed on the training data (the first 95% of the raw data available). The model should perform almost perfectly since it's seen this data before!")

    plt.plot(rescaled_y_train, label='Actual closing value')
    plt.plot(rescaled_train_predictions, label='Predicted closing value')

    plt.xlabel('Day')
    plt.ylabel('Close value')

    plt.legend()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # Predicted test values
    with torch.no_grad():
        test_predicted = model(x_test.to(device)).cpu().numpy()

    temp = np.zeros((x_test.shape[0], lookback + 1))
    temp[:, 0] = test_predicted.flatten()
    temp = scaler.inverse_transform(temp)

    rescaled_test_predictions = dc(temp[:, 0])

    temp = np.zeros((x_test.shape[0], lookback + 1))
    temp[:, 0] = y_test.flatten()
    temp = scaler.inverse_transform(temp)

    rescaled_y_test = dc(temp[:, 0])

    st.subheader("Testing results")

    st.write("This is how the LSTM model performs on testing data (the most recent 5% of the raw data available) that it's never seen before.")

    plt.plot(rescaled_y_test, label='Actual closing value')
    plt.plot(rescaled_test_predictions, label='Predicted closing value')

    plt.xlabel('Day')
    plt.ylabel('Close value')

    plt.legend()

    st.pyplot()

    # Predict tomorrow's price
    st.subheader("Predicting tomorrow's price")

    st.write("We'll first retrain our model to use all 100% of the data available rather than 95%. We'll then predict tomorrow's closing stock value.")

    loading_text_3 = st.text("Reefeding data into model...")

    x_train = scaled_data[:, 1:]
    y_train = scaled_data[:, 0]

    x_train = dc(np.flip(x, axis=1))
    x_train = x_train.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))

    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()

    # Retrain model
    model, loading_text_4 = train_model(x_train, None, y_train, None, True, loading_text_3)

    loading_text_4.text("Training model... Done!")

    x_test = y[-1 * lookback:]
    x_test = x_test.reshape((-1, lookback, 1))
    x_test = torch.tensor(x_test).float()

    x_test = x_test.to(device)

    with torch.no_grad():
        y_test = model(x_test).cpu().numpy()

    # Display results
    st.write(f"The previous 7 days of stock for {stock_input} have been:")
    st.write(stock_data_copy[['Date', 'Close']].iloc[-7:,:])

    temp = np.zeros((x_test.shape[0], lookback + 1))
    temp[:, 0] = y_test.flatten()
    temp = scaler.inverse_transform(temp)

    rescaled_test_predictions = dc(temp[:, 0])

    st.write("Based on this, the predicted value for tomorrow is %.2f." % rescaled_test_predictions[0])
