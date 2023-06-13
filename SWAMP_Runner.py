import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

scaler = MinMaxScaler()
window_length = 20
batch_size = 5
number_features = 3

# Load the original model architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(window_length, number_features), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

# Load the trained model weights
model.load_weights('model.h5')

while True:
    data = pd.read_csv('FL.csv')  # Replace 'new_data.csv' with the filename/path of your new dataset

    data['Date Time'] = pd.to_datetime(data['Date Time'], infer_datetime_format=True)
    data.set_index('Date Time')[['FL', 'RF', 'AP']].plot(subplots=True)

    data_input = data[['FL', 'RF', 'AP']]

    # Apply the same scaling as in the original program
    data_scaled = scaler.fit_transform(data_input)

    features = data_scaled

    # Generate time series samples
    data_gen = TimeseriesGenerator(features, features[:, 0], length=window_length, sampling_rate=1, batch_size=1)

    # Use the loaded model to predict flood levels
    predictions = model.predict(data_gen)

    predicted_data = pd.concat([pd.DataFrame(predictions), pd.DataFrame(features[window_length:, 1:])], axis=1)

    # Reverse-transform the predicted values
    reverse_transform = scaler.inverse_transform(predicted_data)

    final_prediction = data_input[predictions.shape[0] * -1:]
    final_prediction['FL_PREDICTED'] = reverse_transform[:, 0]

    predicted_values = final_prediction["FL_PREDICTED"]

    plt.figure()
    plt.plot(predicted_values.index, predicted_values.values, label='Predicted Flood Level')
    plt.xlabel('Date Time')
    plt.ylabel('FL_PREDICTED')
    plt.legend()
    plt.show()

    print(predicted_values)

    predicted_values.to_csv("predicted_values.csv", index=False)
