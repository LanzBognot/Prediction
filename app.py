import pandas as pd
from flask import Flask, request, jsonify
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Set up Flask application
app = Flask(__name__)

# Load the trained model weights
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(window_length, number_features), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))
model.load_weights('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the request
    data = request.json
    
    # Load and preprocess the data
    data_input = pd.DataFrame(data, columns=['FL', 'RF', 'AP'])
    data_scaled = scaler.transform(data_input)
    features = data_scaled

    # Generate time series samples
    data_gen = TimeseriesGenerator(features, features[:, 0], length=window_length, sampling_rate=1, batch_size=1)

    # Use the loaded model to predict flood levels
    predictions = model.predict(data_gen)

    # Reverse-transform the predicted values
    reverse_transform = scaler.inverse_transform(predictions)

    # Prepare the predicted values for response
    predicted_values = reverse_transform.flatten().tolist()

    # Return the prediction as a JSON response
    return jsonify({'predictions': predicted_values})

if __name__ == '__main__':
    # Load the scaler
    scaler = MinMaxScaler()
    
    # Load the original data
    original_data = pd.read_csv('FL.csv')

    # Preprocess the data with the scaler
    original_input = original_data[['FL', 'RF', 'AP']]
    scaler.fit(original_input)

    # Set other variables
    window_length = 20
    number_features = 3

    app.run(debug=True)
