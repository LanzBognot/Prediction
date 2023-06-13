import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn. preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import csv

mpl.rcParams ['figure.figsize'] = (10, 8)
mpl.rcParams ['axes.grid'] = False

data = pd.read_csv('swamp.csv')

data['Date Time'] = pd.to_datetime(data['Date Time'], infer_datetime_format=True)
data.set_index('Date Time')[['FL', 'RF', 'AP']].plot(subplots=True)

data_input = data[['FL', 'RF', 'AP']]

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_input)

features = data_scaled
target = data_scaled[:,0]

TimeseriesGenerator(features, target, length=2, sampling_rate=1, batch_size=1)[0]

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123, shuffle=False)

window_length = 720
batch_size = 32
number_features = 3
train_gen = TimeseriesGenerator(x_train, y_train, length=window_length, sampling_rate=1, batch_size=batch_size)
test_gen = TimeseriesGenerator(x_test, y_test, length=window_length, sampling_rate=1, batch_size=batch_size)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape = (window_length, number_features), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=2,
                                                mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adagrad(learning_rate=0.01),
            metrics=[tf.metrics.MeanAbsoluteError()])

history = model.fit(train_gen, epochs=20,
                            validation_data=test_gen,
                            shuffle=False,
                            callbacks=[early_stopping])

model.evaluate(test_gen, verbose=0)
model.save('model.h5')

predictions = model.predict(test_gen)

predicted_data = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:,1:][window_length:])], axis=1)

reverse_transform = scaler.inverse_transform(predicted_data)

final_prediction =  data_input[predictions.shape[0]*-1:]
final_prediction['FL_PREDICTED']=reverse_transform[:,0]

predicted_values = final_prediction["FL_PREDICTED"]

print(predicted_values)

predicted_values.to_csv("predicted_values.csv", index=False)