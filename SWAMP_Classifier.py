while True:

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load the dataset
    dataset = pd.read_csv('MAYtraining.csv')  # Replace 'testing.csv' with your actual dataset file

    # Split the data into features (X) and labels (y)
    X = dataset[['FL_PREDICTED']]
    y = dataset['Danger Level']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the Random Forest classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Read the predicted data point from the CSV file
    latest_data = pd.read_csv('predicted_values.csv').tail(1)
    latest_data = latest_data[['FL_PREDICTED']]  # Remove 'Danger Level' column

    # Predict the danger level for the latest data point
    prediction = classifier.predict(latest_data)
    print(f'Predicted danger level: {prediction[0]}')

    from PIL import Image

    def display_color_output(prediction):
        if prediction == "Yellow (Monitor)":
            image_path = "yellow.jpg"  # Path to the yellow image file
        elif prediction == "Orange (Alerto)":
            image_path = "orange.jpg"  # Path to the orange image file
        elif prediction == "Red (Lumikas)":
            image_path = "red.jpg"  # Path to the red image file
        else:
            print("Invalid color")

        # Load and display the image
        image = Image.open(image_path)
        image.show()

    display_color_output(prediction)