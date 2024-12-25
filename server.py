from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Resolve paths dynamically
base_dir = os.path.abspath(os.path.dirname(__file__))
classification_model_path = os.path.join(base_dir, "Instances", "ElecClassification_model.pkl")
regression_model_path = os.path.join(base_dir, "Instances", "ElecRegression_model.pkl")

classification_model = None
regression_model = None

# Load models with error handling
try:
    classification_model = joblib.load(classification_model_path)
    print(f"Classification model loaded successfully from {classification_model_path}")
except FileNotFoundError as e:
    print(f"Classification model not found: {e}")
except Exception as e:
    print(f"Failed to load classification model: {e}")

try:
    regression_model = joblib.load(regression_model_path)
    print(f"Regression model loaded successfully from {regression_model_path}")
except FileNotFoundError as e:
    print(f"Regression model not found: {e}")
except Exception as e:
    print(f"Failed to load regression model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/classification', methods=['POST'])
def predict_classification():
    if classification_model is None:
        return jsonify({"error": "Classification model not loaded"}), 500

    data = request.json
    try:
        features = [data.get(feature) for feature in joblib.load(os.path.join(base_dir, "Instances", "feature_names.pkl"))]
        prediction = classification_model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": f"Failed to predict classification: {str(e)}"}), 500

@app.route('/predict/regression', methods=['POST'])
def predict_regression():
    if regression_model is None:
        return jsonify({"error": "Regression model not loaded"}), 500

    data = request.json
    try:
        features = [data.get(feature) for feature in joblib.load(os.path.join(base_dir, "Instances", "feature_names.pkl"))]
        predictions = regression_model.predict([features])
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": f"Failed to predict regression: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
