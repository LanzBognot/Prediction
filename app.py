import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Import the machine learning code
    import pandas as pd
    import matplotlib as mpl
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.sequence import TimeseriesGenerator
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    
    mpl.rcParams['figure.figsize'] = (10, 8)
    mpl.rcParams['axes.grid'] = False

    data = pd.read_csv('swamp.csv')

    data['Date Time'] = pd.to_datetime(data['Date Time'], infer_datetime_format=True)
    data.set_index('Date Time')[['FL', 'RF', 'AP']].plot(subplots=True)

    data_input = data[['FL', 'RF', 'AP']]

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_input)

    features = data_scaled
    target = data_scaled[:, 0]

    TimeseriesGenerator(features, target, length=2, sampling_rate=1, batch_size=1)[0]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123, shuffle=False)

    window_length = 720
    batch_size = 32
    number_features = 3
    train_gen = TimeseriesGenerator(x_train, y_train, length=window_length, sampling_rate=1, batch_size=batch_size)
    test_gen = TimeseriesGenerator(x_test, y_test, length=window_length, sampling_rate=1, batch_size=batch_size)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(window_length, number_features), return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(64, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adagrad(learning_rate=0.01),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(train_gen, epochs=1, validation_data=test_gen, shuffle=False, callbacks=[early_stopping])

    model.evaluate(test_gen, verbose=0)
    model.save('model.h5')

    # Retrieve data from the request
    data = request.json

    # Perform inference
    predictions = model.predict(test_gen)

    # Prepare response
    response = {'predictions': predictions.tolist()}

    # Return the response as JSON
    return render_template('forecast.html')


if __name__ == '__main__':
    app.run(debug=True)
