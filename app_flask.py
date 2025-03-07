from flask import Flask, render_template, request
import joblib
import numpy as np

# Load model and scaler
MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "scaler.joblib"
FEATURES_PATH = "feature_names.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)  # Load feature names dynamically
except Exception as e:
    raise RuntimeError(f"Error loading model or feature names: {str(e)}")

# Initialize Flask application
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Handles form display and prediction."""
    if request.method == "POST":
        try:
            # Get features from the form
            input_values = [float(request.form[feature]) for feature in feature_names]

            # Convert input to NumPy array and scale it
            input_values = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_values)

            # Predict
            prediction = model.predict(input_scaled)
            churn_result = "Churn" if prediction[0] == 1 else "Not Churn"

            return render_template(
                "index.html", feature_names=feature_names, prediction=churn_result
            )

        except Exception as e:
            return {"error": str(e)}

    # Render the form if the method is GET
    return render_template("index.html", feature_names=feature_names, prediction=None)


# Main entry point to run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8088)
