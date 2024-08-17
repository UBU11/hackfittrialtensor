import numpy as np
import tensorflow as tf
import os
from flask import Flask, request, jsonify, render_template
import firebase_admin
from firebase_admin import credentials, firestore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate('safetrix-117b2-firebase-adminsdk-gakry-63a5157e5f.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


# Function to create and train the initial model
def create_initial_model():
    # Generate synthetic data
    np.random.seed(42)
    num_samples = 1000

    # Inputs (x1 to x9)
    rainfall_intensity = np.random.uniform(10, 200, num_samples)
    duration = np.random.uniform(0.5, 10, num_samples)
    frequency = np.random.uniform(1, 30, num_samples)
    runoff_coefficient = np.random.uniform(0.1, 0.9, num_samples)
    catchment_area = np.random.uniform(1, 500, num_samples)
    land_cover = np.random.uniform(0.2, 0.8, num_samples)
    evaporation_rate = np.random.uniform(1, 10, num_samples)
    temperature = np.random.uniform(15, 40, num_samples)
    humidity = np.random.uniform(30, 90, num_samples)

    X = np.stack([rainfall_intensity, duration, frequency, runoff_coefficient,
                  catchment_area, land_cover, evaporation_rate, temperature, humidity], axis=1)

    # Calculate flood probability
    flood_probability = (0.3 * rainfall_intensity + 0.2 * duration +
                         0.3 * runoff_coefficient + 0.1 * catchment_area +
                         0.05 * (100 - humidity) - 0.2 * evaporation_rate)
    flood_probability = (flood_probability - flood_probability.min()) / (
                flood_probability.max() - flood_probability.min())
    y = (flood_probability > 0.5).astype(int)

    # Create and compile the model
    model = Sequential([
        Dense(64, input_dim=X.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

    return model


# Create or load the model
if os.path.exists('flood_prediction_model.h5'):
    model = tf.keras.models.load_model('flood_prediction_model.h5')
else:
    model = create_initial_model()
    model.save('flood_prediction_model.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit-data', methods=['POST'])
def submit_data():
    data = {
        'rainfall_intensity': float(request.form.get('rainfall_intensity')),
        'duration': float(request.form.get('duration')),
        'frequency': float(request.form.get('frequency')),
        'runoff_coefficient': float(request.form.get('runoff_coefficient')),
        'catchment_area': float(request.form.get('catchment_area')),
        'land_cover': float(request.form.get('land_cover')),
        'evaporation_rate': float(request.form.get('evaporation_rate')),
        'temperature': float(request.form.get('temperature')),
        'humidity': float(request.form.get('humidity'))
    }

    db.collection('collected_data').add(data)
    return "Data submitted successfully"


@app.route('/get-data')
def get_data():
    docs = db.collection('collected_data').get()
    data = [doc.to_dict() for doc in docs]
    return jsonify(data)


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    X = np.array([[
        input_data['rainfall_intensity'],
        input_data['duration'],
        input_data['frequency'],
        input_data['runoff_coefficient'],
        input_data['catchment_area'],
        input_data['land_cover'],
        input_data['evaporation_rate'],
        input_data['temperature'],
        input_data['humidity']
    ]])

    prediction = model.predict(X)
    return jsonify({'flood_probability': float(prediction[0][0])})


@app.route('/retrain')
def retrain():
    docs = db.collection('collected_data').get()
    data = [doc.to_dict() for doc in docs]

    X = np.array([[d['rainfall_intensity'], d['duration'], d['frequency'],
                   d['runoff_coefficient'], d['catchment_area'], d['land_cover'],
                   d['evaporation_rate'], d['temperature'], d['humidity']] for d in data])

    flood_probability = (0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.3 * X[:, 3] +
                         0.1 * X[:, 4] + 0.05 * (100 - X[:, 8]) - 0.2 * X[:, 6])
    flood_probability = (flood_probability - flood_probability.min()) / (
                flood_probability.max() - flood_probability.min())
    y = (flood_probability > 0.5).astype(int)

    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save('flood_prediction_model.h5')

    return "Model retrained successfully"


if __name__ == '__main__':
    app.run(debug=True)