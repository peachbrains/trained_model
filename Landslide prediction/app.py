import os
import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained machine learning model
MODEL_PATH = os.path.join('ML_Model', 'logistic_regression_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None

# Feature names as specified in the requirements
FEATURE_NAMES = [
    'Aspect', 'Curvature', 'Earthquake', 'Elevation', 
    'FlowLithology', 'NDVI', 'NDWI', 'Plan', 
    'Precipitation', 'Profile', 'Slope'
]

# Initialize scaler (if your model requires feature scaling)
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict_landslide():
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

        # Validate feature ranges (1 to 5)
        for feature, value in zip(FEATURE_NAMES, input_data):
            if not (1 <= value <= 5):
                return jsonify({
                    'error': f'Invalid value for {feature}. Must be between 1 and 5.'
                }), 400

        # Convert input to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Scale the features if required (uncomment if model was trained with scaling)
        # input_array = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_array)
        
        # Convert prediction to landslide probability
        landslide_prob = prediction[0]
        landslide_binary = 1 if landslide_prob > 0.5 else 0

        return jsonify({
            'landslide_probability': float(landslide_prob),
            'landslide_prediction': int(landslide_binary)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Landslide Prediction API',
        'features': FEATURE_NAMES,
        'usage': 'Send a POST request to /predict with an array of 11 features'
    })

if __name__ == '__main__':
    app.run(debug=True)