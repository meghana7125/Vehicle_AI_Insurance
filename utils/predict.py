import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Get project root
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Model location
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "vehicle_damage_model.keras"
)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


def predict_damage(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert to numpy
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # IMPORTANT:
    # Dataset classes:
    # 0 = 00-damage
    # 1 = 01-whole

    if prediction >= 0.5:

        label = "No Damage"
        confidence = prediction * 100

    else:

        label = "Damage Detected"
        confidence = (1 - prediction) * 100

    return {
        "label": label,
        "confidence": round(float(confidence), 2)
    }