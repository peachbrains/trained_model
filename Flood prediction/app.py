import os
import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained machine learning model
MODEL_PATH = os.path.join('ML_Model', 'random_forest_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None

# Feature names as specified in the requirements
FEATURE_NAMES = [
    'total_slope', 
    'total_elevation', 
    'annual_rainfall_mm', 
    'flood_month', 
    'river_basin_width_km'
]

# Initialize scaler (if your model requires feature scaling)
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict_flood():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get input features from the request
        input_data = request.json

        # Validate input
        if not input_data or len(input_data) != len(FEATURE_NAMES):
            return jsonify({
                'error': f'Invalid input. Expected {len(FEATURE_NAMES)} features: {", ".join(FEATURE_NAMES)}'
            }), 400

        # Additional input validations
        # You might want to adjust these based on your specific domain knowledge
        validations = {
            'total_slope': (0, 90),  # slope in degrees
            'total_elevation': (-500, 10000),  # elevation in meters
            'annual_rainfall_mm': (0, 10000),  # annual rainfall in mm
            'flood_month': (1, 12),  # month of the year
            'river_basin_width_km': (0, 500)  # river basin width in km
        }

        # Check each feature against its expected range
        for feature, value in zip(FEATURE_NAMES, input_data):
            min_val, max_val = validations[feature]
            if not (min_val <= value <= max_val):
                return jsonify({
                    'error': f'Invalid value for {feature}. Must be between {min_val} and {max_val}.'
                }), 400

        # Convert input to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Scale the features if required (uncomment if model was trained with scaling)
        # input_array = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_array)
        
        # Convert prediction to flood probability
        flood_prob = prediction[0]
        flood_binary = 1 if flood_prob > 0.5 else 0

        return jsonify({
            'flood_probability': float(flood_prob),
            'flood_prediction': int(flood_binary)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Flood Prediction API',
        'features': FEATURE_NAMES,
        'feature_descriptions': {
            'total_slope': 'Terrain slope in degrees',
            'total_elevation': 'Elevation in meters',
            'annual_rainfall_mm': 'Annual rainfall in millimeters',
            'flood_month': 'Month of potential flood (1-12)',
            'river_basin_width_km': 'Width of river basin in kilometers'
        },
        'usage': 'Send a POST request to /predict with an array of 5 features'
    })

@app.route('/validate-input', methods=['POST'])
def validate_input():
    """
    Additional route to help users validate their input before prediction
    """
    try:
        input_data = request.json

        if not input_data or len(input_data) != len(FEATURE_NAMES):
            return jsonify({
                'error': f'Invalid input. Expected {len(FEATURE_NAMES)} features'
            }), 400

        validation_results = {}
        for feature, value in zip(FEATURE_NAMES, input_data):
            min_val, max_val = {
                'total_slope': (0, 90),
                'total_elevation': (-500, 10000),
                'annual_rainfall_mm': (0, 10000),
                'flood_month': (1, 12),
                'river_basin_width_km': (0, 500)
            }[feature]

            is_valid = min_val <= value <= max_val
            validation_results[feature] = {
                'value': value,
                'is_valid': is_valid,
                'min': min_val,
                'max': max_val
            }

        return jsonify(validation_results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)