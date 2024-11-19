# Importing libraries
import yfinance as yf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Set the start and end dates for the data
start_date = '1990-01-01'
end_date = None

# Fetch S&P 500 data from yfinance
ticker = '^GSPC'  # S&P 500 index ticker
data = yf.download(ticker, start=start_date, end=end_date)['Close'].dropna()

# Debugging: Check the fetched data
print(f"Data fetched from yfinance:\n{data.head()}")
print(f"Data shape before conversion: {data.shape}")

# Convert to a 1D numpy array
data = data.to_numpy()  # Convert Pandas Series to NumPy array

# Flatten the data to ensure it's 1D
data = data.flatten()  # Resulting shape: (8545,)

# Debugging: Check data after flattening
print(f"Data shape after flattening: {data.shape}")
print(f"First 10 data points (after flattening): {data[:10]}")

# Apply differencing once to make the data stationary
if len(data) > 1:  # Ensure there's enough data to apply differencing
    data = np.diff(data)  # Resulting shape: (len(data) - 1,)
else:
    raise ValueError("Insufficient data points after fetching to perform differencing.")

# Debugging: Check data after differencing
print(f"Data length after differencing: {len(data)}")
print(f"Data preview after differencing: {data[:10]}")  # Print first 10 values for verification

# Debugging: Check data after conversion
print(f"Data shape after conversion: {data.shape}")
print(f"First 10 data points: {data[:10]}")

# Apply differencing once to make the data stationary
if len(data) > 1:  # Ensure there's enough data to apply differencing
    data = np.diff(data)  # Resulting shape: (len(data) - 1,)
else:
    raise ValueError("Insufficient data points after fetching to perform differencing.")


# Set hyperparameters
num_lags = 100
train_test_split_ratio = 0.90
num_neurons_in_hidden_layers = 500
num_epochs = 50
batch_size = 16

# Proceed with preprocessing
if len(data) <= num_lags:
    raise ValueError(f"The data length ({len(data)}) must be greater than num_lags ({num_lags}). Reduce num_lags.")

# Data preprocessing function
def data_preprocessing(data, num_lags, train_test_split_ratio):
    print(f"Data shape before preprocessing: {data.shape}")
    print(f"Num lags: {num_lags}")

    # Ensure data length is sufficient for num_lags
    if len(data) <= num_lags:
        raise ValueError(f"The data length ({len(data)}) must be greater than num_lags ({num_lags}). Reduce num_lags.")

    x, y = [], []
    for i in range(len(data) - num_lags):
        x.append(data[i:i + num_lags])  # Create sequences of size num_lags
        y.append(data[i + num_lags])  # Target is the next value

    # Debugging: Check if x and y contain data
    if len(x) == 0 or len(y) == 0:
        raise ValueError("The data preprocessing step produced empty x or y arrays.")

    x, y = np.array(x), np.array(y)
    print(f"Shape of x: {x.shape}, Shape of y: {y.shape}")

    # Ensure the shape of x is (samples, num_lags) for Dense layers
    x = x.reshape(x.shape[0], num_lags)

    # Split the data
    split_index = int(train_test_split_ratio * len(x))
    x_train, y_train = x[:split_index], y[:split_index]
    x_test, y_test = x[split_index:], y[split_index:]
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
    return x_train, y_train, x_test, y_test


# Check the size of the data after differencing
print(f"Original data length: {len(data) + 1}")  # +1 because np.diff reduces the length by 1
print(f"Data length after differencing: {len(data)}")

# Call the preprocessing function
x_train, y_train, x_test, y_test = data_preprocessing(data, num_lags, train_test_split_ratio)

# Build the neural network model
model = Sequential()
model.add(Dense(num_neurons_in_hidden_layers, input_dim=num_lags, activation='relu'))
model.add(Dense(num_neurons_in_hidden_layers, activation='relu'))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# Train the model
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

# Make predictions
y_pred = model.predict(x_test).flatten()

# Predict the next 3 days
last_sequence = data[-num_lags:]  # Use the last `num_lags` values as input
future_predictions = []

for _ in range(3):  # Predict for 3 future days
    next_prediction = model.predict(last_sequence.reshape(1, -1)).flatten()[0]
    future_predictions.append(next_prediction)
    # Update the sequence with the new prediction
    last_sequence = np.append(last_sequence[1:], next_prediction)

# Print the predictions
print("Predicted values for the next 3 days (differenced):")
for i, pred in enumerate(future_predictions, start=1):
    print(f"Day {i}: {pred:.2f}")

# Optional: Transform predictions back to actual price if necessary
# Assuming you have the last known actual price
last_actual_price = yf.download(ticker, period="1d")['Close'].values[-1]
actual_future_prices = [last_actual_price]
for pred in future_predictions:
    next_price = actual_future_prices[-1] + pred  # Reverse differencing
    actual_future_prices.append(next_price)


# Plot the predictions vs. true values
plt.figure(figsize=(12, 6))
plt.plot(y_test[-100:], label='True Data', marker='o', alpha=0.7, color='blue')
plt.plot(y_pred[-100:], label='Predicted Data', linestyle='--', marker='x', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title('S&P 500 Prediction vs True Data')
plt.xlabel('Time Steps')
plt.ylabel('Differenced Price')
plt.legend()
plt.grid()
plt.show()

# Calculate hit ratio
same_sign_count = np.sum(np.sign(y_pred) == np.sign(y_test)) / len(y_test) * 100
print(f'Hit Ratio: {same_sign_count:.2f}%')
print("\nPredicted actual prices for the next 3 days:")
for i, price in enumerate(actual_future_prices[1:], start=1):
    print(f"Day {i}: {price:.2f}")

