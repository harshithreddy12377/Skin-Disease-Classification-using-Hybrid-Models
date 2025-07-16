import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.applications import EfficientNetB0

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

lightgbm_model = joblib.load("lightgbm_model.pkl") 
xgb_model = joblib.load("xgb_model.pkl")            


effnet = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
effnet.trainable = False

label_map = {
    0: "Nevus (NV)",
    1: "Melanoma (MEL)",
    2: "Benign Keratosis (BKL)",
    3: "Basal Cell Carcinoma (BCC)",
    4: "Actinic Keratosis (AKIEC)",
    5: "Vascular Lesion (VASC)",
    6: "Dermatofibroma (DF)"
}

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def extract_features(image):
    features = effnet.predict(image)
    return features.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded!"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file!"})

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = preprocess_image(file_path)
        features = extract_features(img)

        try:
            xgb_probs = xgb_model.predict(features)
            xgb_probs = np.array(xgb_probs).reshape(-1, len(label_map))
            xgb_class = int(np.argmax(xgb_probs, axis=1)[0])
            xgb_result = label_map.get(xgb_class, "Unknown")

        except Exception as e:
            xgb_result = f"XGBoost error: {str(e)}"
            xgb_probs = np.zeros((1, len(label_map)))

        try:
            light_probs = lightgbm_model.predict(features)
            light_probs = np.array(light_probs).reshape(-1, len(label_map))
            light_class = int(np.argmax(light_probs, axis=1)[0])
            light_result = label_map.get(light_class, "Unknown")

        except Exception as e:
            light_result = f"LightGBM error: {str(e)}"
            light_probs = np.zeros((1, len(label_map)))

        avg_probs = (light_probs + xgb_probs) / 2.0
        voted_class = int(np.argmax(avg_probs, axis=1)[0])
        voted_result = label_map.get(voted_class, "Unknown")

        return jsonify({
            "LightGBM Prediction": light_result,
            "XGBoost Prediction": xgb_result,
            "Voting Prediction": voted_result
        })

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
