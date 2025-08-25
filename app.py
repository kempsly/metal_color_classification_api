from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests
import logging
from io import BytesIO
from PIL import Image
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV3Large

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 8
CLASS_NAMES = ['black_gold', 'gold', 'platinum', 'rose_gold', 'silver', 'two_tone', 'white_gold', 'yellow_gold']

# Use environment variable for model path
MODEL_PATH = os.getenv('MODEL_PATH', 'model/metal_model.keras')

def build_model():
    """Rebuild the model architecture to match the saved weights"""
    base_model = MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg',
        minimalistic=False
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Load the model
try:
    model = build_model()
    model.load_weights(MODEL_PATH)
    logger.info("Model weights loaded successfully")
except Exception as e:
    logger.error(f"Error loading model weights: {e}")
    raise

def preprocess_image(img):
    """Preprocess image for model prediction"""
    if isinstance(img, str):
        img = image.load_img(img, target_size=(IMG_SIZE, IMG_SIZE))
    elif isinstance(img, Image.Image):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    else:
        img = Image.fromarray(img.astype('uint8')).resize((IMG_SIZE, IMG_SIZE))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    return img_array

def predict_image(img):
    """Make prediction on preprocessed image"""
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_classes = [CLASS_NAMES[i] for i in top3_indices]
    top3_probs = [float(predictions[0][i]) for i in top3_indices]

    results = []
    for cls, prob in zip(top3_classes, top3_probs):
        results.append({
            "class": cls,
            "probability": round(prob, 4)
        })

    return results

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        try:
            img = Image.open(BytesIO(file.read()))
            results = predict_image(img)
            logger.info(f"Prediction successful: {results}")
            return jsonify({'success': True, 'predictions': results})
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return jsonify({'error': str(e)}), 500

    elif 'url' in request.json:
        url = request.json['url']
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            results = predict_image(img)
            logger.info(f"Prediction successful: {results}")
            return jsonify({'success': True, 'predictions': results})
        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            return jsonify({'error': str(e)}), 500

    else:
        logger.warning("No file or URL provided")
        return jsonify({'error': 'No file or URL provided'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
