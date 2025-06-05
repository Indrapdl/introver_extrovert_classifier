from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)  # must exist at top-level

# Load models
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("pca.pickle", "rb") as f:
    pca = pickle.load(f)
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputs = [
            float(request.form["time_spent_alone"]),
            int(request.form["stage_fear"]),
            int(request.form["drained_after_socializing"]),
            float(request.form["social_events"]),
            float(request.form["outdoor_frequency"]),
            float(request.form["post_frequency"]),
            float(request.form["friend_circle_size"]),
        ]

        X = np.array(inputs).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        prediction = model.predict(X_pca)

        result = "Model says you are an Extrovert 😄" if prediction[0] == 1 else "Model says you are an Introvert 🤔"
        return render_template("index.html", prediction=result)
    
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
