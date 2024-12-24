# from flask import Flask, request, render_template, jsonify
# from flask_cors import CORS
# import joblib
# import pandas as pd
# import numpy as np
# import os

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# classification_model = None
# elecRegression_modal = None

# try:
#     classification_model_path= os.path.join("Instances", "ElecClassification_model.pkl")
#     regression_model_path = os.path.join("Instances", "ElecRegression_model.pkl")

#     with open(classification_model_path, 'rb') as f:
#         classification_model = joblib.load(f)
#     print(f"classification model loaded successfully from {classification_model}")

#     with open(regression_model_path, 'rb') as f:
#         regression_model = joblib.load(f)
#     print(f"Regression modal loaded successfully from {regression_model}")

# except Exception as e:
#     print(f"Failed to load model or scaler: {e}")

# # # Load models (adjusted paths)
# # classification_model_path = os.path.join(os.path.dirname(__file__), 'ElecClassification_model.pkl')
# # regression_model_path = os.path.join(os.path.dirname(__file__), 'ElecRegression_model.pkl')

# # classification_model = joblib.load(classification_model_path)
# # regression_model = joblib.load(regression_model_path)

# # Define routes
# @app.route('/')
# def home():
#     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Get input data from form
# #     data = request.form.to_dict()

# #     # Prepare data for prediction
# #     # Convert categorical inputs to dummy variables as in training
# #     input_data = pd.DataFrame([data])

# #     # Ensure numerical conversion
# #     input_data['Year'] = pd.to_numeric(input_data['Year'])
# #     input_data['Inflation'] = pd.to_numeric(input_data['Inflation'])
# #     input_data['Unemployment'] = pd.to_numeric(input_data['Unemployment'])
# #     input_data['Growth Rate'] = pd.to_numeric(input_data['Growth Rate'])

# #     # Handle categorical columns (Region, Incumbent Party)
# #     regions = ['Region_A', 'Region_B', 'Region_C']  # Replace with actual dummy columns
# #     parties = ['Incumbent Party_Other', 'Incumbent Party_X']  # Replace with actual dummy columns
# #     for col in regions + parties:
# #         input_data[col] = 0
# #     input_data[f"Region_{data['Region']}"] = 1
# #     input_data[f"Incumbent Party_{data['Incumbent Party']}"] = 1

# #     # Drop unused columns
# #     input_data = input_data.drop(columns=['Region', 'Incumbent Party'], errors='ignore')

# #     # Predictions
# #     classification_prediction = classification_model.predict(input_data)
# #     regression_prediction = regression_model.predict(input_data)

# #     # Prepare response
# #     response = {
# #         "Winner Prediction": int(classification_prediction[0]),
# #         "NPP Votes Prediction": regression_prediction[0][0],
# #         "NDC Votes Prediction": regression_prediction[0][1],
# #         "Others Votes Prediction": regression_prediction[0][2],
# #     }

# #     return jsonify(response)



# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get JSON data from request
#     data = request.json

#     if not data:
#         print("No input data provided")
#         return jsonify({'error': 'No input data provided'})
    
#     print(f"Received input data: {data}")

#     # Prepare data for prediction
#     input_data = pd.DataFrame([data])
#     print(f"Input data")
#     print(input_data)

#     # Convert numerical columns
#     numerical_columns = ['Year', 'Inflation', 'Unemployment', 'Growth Rate']
#     input_data[numerical_columns] = input_data[numerical_columns].apply(pd.to_numeric)

#     # Handle categorical columns (Region, Incumbent Party)
#     regions = ['Greater Accra', 'Ashanti', 'Western', 'Eastern', 'Northern', 'Central', 
#                'Upper East', 'Upper West', 'Volta', 'Oti', 'Savannah', 'North East', 
#                'Bono', 'Bono East', 'Ahafo', 'Western North']
#     parties = ['NPP', 'NDC', 'Other']
    
#     for col in regions + parties:
#         input_data[f'Region_{col}'] = 0
#         input_data[f'Incumbent Party_{col}'] = 0

#     input_data[f'Region_{data["Region"]}'] = 1
#     input_data[f'Incumbent Party_{data["Incumbent Party"]}'] = 1

#     # Drop unused columns
#     input_data = input_data.drop(columns=['Region', 'Incumbent Party'], errors='ignore')

#     # Predictions
#     classification_prediction = classification_model.predict(input_data)
#     regression_prediction = regression_model.predict(input_data)

#     # Prepare response
#     response = {
#         "Winner Prediction": int(classification_prediction[0]),
#         "NPP Votes Prediction": regression_prediction[0][0],
#         "NDC Votes Prediction": regression_prediction[0][1],
#         "Others Votes Prediction": regression_prediction[0][2],
#     }

#     return jsonify(response)


# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models
try:
    classification_model_path = 'Instances\ElecClassification_model.pkl'  # Adjust the path if needed
    regression_model_path = 'Instances\ElecRegression_model.pkl'

    classification_model = joblib.load(classification_model_path)
    print(f"Classification model loaded successfully from {classification_model_path}")

    regression_model = joblib.load(regression_model_path)
    print(f"Regression model loaded successfully from {regression_model_path}")

except Exception as e:
    print(f"Failed to load models: {e}")
    classification_model = None
    regression_model = None

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Validate input data
#     data = request.json
#     if not data:
#         return jsonify({'error': 'No input data provided'}), 400

#     try:
#         # Prepare input data
#         input_data = pd.DataFrame([data])

#         # Convert numerical columns
#         numerical_columns = ['Year', 'Inflation', 'Unemployment', 'Growth Rate']
#         input_data[numerical_columns] = input_data[numerical_columns].apply(pd.to_numeric)

#         # Handle categorical columns
#         regions = [
#             'Greater Accra', 'Ashanti', 'Western', 'Eastern', 'Northern', 'Central', 
#             'Upper East', 'Upper West', 'Volta','Brong Ahafo'
#         ]
#         parties = ['NPP', 'NDC', 'Other']

#         for col in regions:
#             input_data[f'Region_{col}'] = 0
#         for col in parties:
#             input_data[f'Incumbent Party_{col}'] = 0

#         input_data[f'Region_{data["Region"]}'] = 1
#         input_data[f'Incumbent Party_{data["Incumbent Party"]}'] = 1

#         # Drop unused columns
#         input_data = input_data.drop(columns=['Region', 'Incumbent Party'], errors='ignore')

#         # Predictions
#         classification_prediction = classification_model.predict(input_data)
#         regression_prediction = regression_model.predict(input_data)

#         # Prepare response
#         response = {
#             "Winner Prediction": int(classification_prediction[0]),
#             "NPP Votes Prediction": regression_prediction[0][0],
#             "NDC Votes Prediction": regression_prediction[0][1],
#             "Others Votes Prediction": regression_prediction[0][2],
#         }
#         return jsonify(response)

#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         return jsonify({'error': 'An error occurred during prediction. Please check your input data.'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json

    if not data:
        return jsonify({'error': 'No input data provided'})

    # Load feature names
    feature_names = joblib.load('Instances/feature_names.pkl')

    # Prepare data for prediction
    input_data = pd.DataFrame([data])
    
    # Handle missing or extra columns
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Ensure numerical conversion
    numerical_columns = ['Year', 'Inflation', 'Unemployment', 'Growth Rate']
    input_data[numerical_columns] = input_data[numerical_columns].apply(pd.to_numeric)

    # Predictions
    classification_prediction = classification_model.predict(input_data)
    regression_prediction = regression_model.predict(input_data)

    # Prepare response
    response = {
        "Winner Prediction": int(classification_prediction[0]),
        "NPP Votes Prediction": regression_prediction[0][0],
        "NDC Votes Prediction": regression_prediction[0][1],
        "Others Votes Prediction": regression_prediction[0][2],
    }

    return jsonify(response)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    
