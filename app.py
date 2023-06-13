from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('SWAMP_Runner.py')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Perform prediction using the model
    prediction = model.predict([data['input']])

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
