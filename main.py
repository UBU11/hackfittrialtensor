import numpy as np
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic data
np.random.seed(42)
num_samples = 1000

# Inputs (x1 to x9)
rainfall_intensity = np.random.uniform(10, 200, num_samples)  # in mm/hr
duration = np.random.uniform(0.5, 10, num_samples)  # in hours
frequency = np.random.uniform(1, 30, num_samples)  # in days
runoff_coefficient = np.random.uniform(0.1, 0.9, num_samples)
catchment_area = np.random.uniform(1, 500, num_samples)  # in km^2
land_cover = np.random.uniform(0.2, 0.8, num_samples)  # vegetation index
evaporation_rate = np.random.uniform(1, 10, num_samples)  # in mm/day
temperature = np.random.uniform(15, 40, num_samples)  # in Celsius
humidity = np.random.uniform(30, 90, num_samples)  # in percentage

# Combine inputs into one matrix
X = np.stack([rainfall_intensity, duration, frequency, runoff_coefficient, 
              catchment_area, land_cover, evaporation_rate, temperature, humidity], axis=1)

# Outputs (y1: 1 for flood, 0 for no flood)
# Here we create a simple relationship to determine flood probability
flood_probability = (0.3 * rainfall_intensity + 0.2 * duration + 
                     0.3 * runoff_coefficient + 0.1 * catchment_area + 
                     0.05 * (100 - humidity) - 0.2 * evaporation_rate)

# Normalize the flood probability
flood_probability = (flood_probability - flood_probability.min()) / (flood_probability.max() - flood_probability.min())

# Convert flood probability to binary outcome
y = (flood_probability > 0.5).astype(int)  # 1 if probability > 0.5, else 0

# Define a simple neural network model
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Model Accuracy: {accuracy:.2f}")

# Make predictions
predictions = model.predict(X[:5])
print(f"Predictions (Probability of Flood):\n{predictions}")
print(f"Actual values:\n{y[:5]}")

# Train the model with initial data
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model to a file
model.save('flood_prediction_model.h5')

# Load the existing model
model = tf.keras.models.load_model('flood_prediction_model.h5')

# New data (assuming new_X and new_y are your new data inputs and outputs)
# Train the model on new data
model.fit(new_X, new_y, epochs=10, batch_size=32, validation_split=0.2)

# Save the updated model
model.save('flood_prediction_model.h5') 


# Generate new synthetic data (just for demonstration)
new_num_samples = 200  # smaller batch of new data

new_rainfall_intensity = np.random.uniform(10, 200, new_num_samples)
new_duration = np.random.uniform(0.5, 10, new_num_samples)
new_frequency = np.random.uniform(1, 30, new_num_samples)
new_runoff_coefficient = np.random.uniform(0.1, 0.9, new_num_samples)
new_catchment_area = np.random.uniform(1, 500, new_num_samples)
new_land_cover = np.random.uniform(0.2, 0.8, new_num_samples)
new_evaporation_rate = np.random.uniform(1, 10, new_num_samples)
new_temperature = np.random.uniform(15, 40, new_num_samples)
new_humidity = np.random.uniform(30, 90, new_num_samples)

new_X = np.stack([new_rainfall_intensity, new_duration, new_frequency, new_runoff_coefficient, 
                  new_catchment_area, new_land_cover, new_evaporation_rate, new_temperature, new_humidity], axis=1)

new_flood_probability = (0.3 * new_rainfall_intensity + 0.2 * new_duration + 
                         0.3 * new_runoff_coefficient + 0.1 * new_catchment_area + 
                         0.05 * (100 - new_humidity) - 0.2 * new_evaporation_rate)

new_flood_probability = (new_flood_probability - new_flood_probability.min()) / (new_flood_probability.max() - new_flood_probability.min())

new_y = (new_flood_probability > 0.5).astype(int)

# Update the model with the new data
model = tf.keras.models.load_model('flood_prediction_model.h5')
model.fit(new_X, new_y, epochs=10, batch_size=32, validation_split=0.2)
model.save('flood_prediction_model.h5')
