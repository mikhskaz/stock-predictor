import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from matplotlib import pyplot as plt

# Define helper functions (same as the training code)
def calculate_bollinger_bands(data, window=10, num_of_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def calculate_rsi(data, window=10):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_roc(data, periods=10):
    roc = ((data - data.shift(periods)) / data.shift(periods)) * 100
    return roc

# Download TSLA data
ticker = 'TSLA'
data = yf.download(ticker, period="1mo", interval="5m")

# Prepare the feature data for TSLA
close = data['Close']
upper, lower = calculate_bollinger_bands(close, window=14, num_of_std=2)
width = upper - lower
rsi = calculate_rsi(close, window=14)
roc = calculate_roc(close, periods=14)
volume = data['Volume']
diff = data['Close'].diff(1)
percent_change_close = data['Close'].pct_change() * 100

# Create DataFrame and normalize using the stats from training
tsla_df = pd.DataFrame({
    'close': close.squeeze(),
    'width': width.squeeze(),
    'rsi': rsi.squeeze(),
    'roc': roc.squeeze(),
    'volume': volume.squeeze(),
    'diff': diff.squeeze(),
    'percent_change_close': percent_change_close.squeeze(),
}, index=data.index)

# Replace infinities and drop NaN values
tsla_df.replace([np.inf, -np.inf], np.nan, inplace=True)
tsla_df.dropna(inplace=True)

# Normalize with the training stats
train_mean = tsla_df.mean()
train_std = tsla_df.std()
tsla_df_normalized = (tsla_df - train_mean) / train_std

# Create sequences for prediction
SEQUENCE_LEN = 24

def create_test_sequences(data, sequence_length=SEQUENCE_LEN):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

test_sequences = create_test_sequences(tsla_df_normalized.values)

def custom_mae_loss(y_true, y_pred):
    y_true_next = tf.cast(y_true[:, 1], tf.float64)
    y_pred_next = tf.cast(y_pred[:, 0], tf.float64)
    abs_error = tf.abs(y_true_next - y_pred_next)

    return tf.reduce_mean(abs_error)


def dir_acc(y_true, y_pred):
    mean, std = tf.cast(y_true[:, 2], tf.float64), tf.cast(y_true[:, 3], tf.float64)

    y_true_prev = (tf.cast(y_true[:, 0], tf.float64) * std) + mean
    y_true_next = (tf.cast(y_true[:, 1], tf.float64) * std) + mean
    y_pred_next = (tf.cast(y_pred[:, 0], tf.float64) * std) + mean

    true_change = y_true_next - y_true_prev
    pred_change = y_pred_next - y_true_prev

    correct_direction = tf.equal(tf.sign(true_change), tf.sign(pred_change))

    return tf.reduce_mean(tf.cast(correct_direction, tf.float64))

# Load the trained model
model = tf.keras.models.load_model(
    "transformer_train_model.keras",
    custom_objects={"custom_mae_loss": custom_mae_loss, "dir_acc": dir_acc}
)

# Make predictions
predictions = model.predict(test_sequences)

# Calculate profit over the last 100 predictions
profit = 0
profit_track = []

# Use the raw data to compare actual closes with predictions
for i in range(-100, 0):  # Last 100 predictions
    pred_close = (predictions[i, 0] * train_std['close']) + train_mean['close']
    prev_close = float(close.iloc[SEQUENCE_LEN + i - 1])
    actual_close = float(close.iloc[SEQUENCE_LEN + i])

    if pred_close > prev_close:  # If predicted close > previous close
        profit += actual_close - prev_close
    profit_track.append(profit)

# Plot the profit growth
plt.figure(figsize=(10, 6))
plt.plot(profit_track, marker='o', label="Profit")
plt.title(f"Profit Growth Over Last 100 Predictions ({ticker})")
plt.xlabel("Prediction Step")
plt.ylabel("Profit ($)")
plt.grid()
plt.legend()
plt.show()
