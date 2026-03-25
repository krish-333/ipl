from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[ 
        data["team"],
        data["opponent"],
        data["powerplay"],
        data["runs"],
        data["wickets"],
        data["pitch"],
        data["toss"]
    ]])

    prediction = model.predict(features)[0][0]

    return jsonify({"score": int(prediction)})

app.run()
