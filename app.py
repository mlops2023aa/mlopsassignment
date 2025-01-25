from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd


# Load the model and encoder
MODEL_PATH = os.path.join("models", "selected_model.pkl")
ENCODER_PATH = os.path.join("models", "encoder.pkl")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model file not found."
                            "Please train the model first.")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("Encoder file not found."
                            "Please save the encoder during training.")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# Create a Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the ML Model API!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided."}), 400

    try:
        # Convert input JSON to DataFrame
        df = pd.DataFrame(data)

        # Apply the encoder to preprocess the input data
        df_encoded = encoder.transform(df)

        # Make predictions
        predictions = model.predict(df_encoded)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5009)
