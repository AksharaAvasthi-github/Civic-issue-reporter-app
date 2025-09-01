from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import uuid

app = Flask(__name__)

# Load ML model from ml folder
model = tf.keras.models.load_model('../ml/garbage_vs_pothole_cnn.h5')

# Define class labels
classes = ['garbage', 'pothole']

# Where to save uploaded images
UPLOAD_FOLDER = '../data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = classes[1] if prediction > 0.5 else classes[0]
    confidence = round(float(prediction if prediction > 0.5 else 1 - prediction), 2)
    return label, confidence

@app.route('/submit', methods=['POST'])
def submit_issue():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is missing'}), 400

    file = request.files['image']
    desc = request.form.get('description', '')
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    img_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(img_path)

    label, confidence = predict_image(img_path)

    result = {
        'predicted_label': label,
        'confidence': confidence,
        'description': desc,
        'image_path': img_path
    }

    return jsonify(result), 200

@app.route('/')
def home():
    return '✅ Civic ML Flask API is up and running!'

if __name__ == '__main__':
    app.run(debug=True)
