import os
import cv2
import numpy as np
import tensorflow as tf


IMG_SIZE = (224, 224)


def load_image(image_path):
    """
    Load and preprocess image for MobileNetV2.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read image.")

    original = image.copy()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image_rgb, IMG_SIZE)

    input_image = resized.astype(np.float32)

    input_image = tf.keras.applications.mobilenet_v2.preprocess_input(
        input_image
    )

    input_image = np.expand_dims(input_image, axis=0)

    return original, input_image


def find_last_conv_layer(model):
    """
    Find the last convolutional layer in the model.
    """

    for layer in reversed(model.layers):

        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D
            )
        ):
            return layer.name

    raise ValueError("No convolutional layer found.")


def generate_gradcam(model, image_path):
    """
    Generate Grad-CAM heatmap.
    """

    original, input_image = load_image(image_path)

    last_conv_layer_name = find_last_conv_layer(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_image
        )

        prediction = predictions[:, 0]

    gradients = tape.gradient(
        prediction,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + 1e-8
    )

    heatmap = heatmap.numpy()

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    return overlay, heatmap


def calculate_severity(heatmap):
    """
    Estimate severity based on activated area.
    """

    threshold = 0.55

    damaged_pixels = np.sum(
        heatmap > threshold
    )

    total_pixels = heatmap.size

    percentage = (
        damaged_pixels /
        total_pixels
    ) * 100

    if percentage < 5:

        severity = "LOW"

    elif percentage < 15:

        severity = "MODERATE"

    elif percentage < 30:

        severity = "HIGH"

    else:

        severity = "SEVERE"

    return severity, round(
        percentage,
        2
    )


def estimate_repair_cost(severity):
    """
    Simple demo repair-cost estimation.
    """

    costs = {

        "LOW": (
            5000,
            15000
        ),

        "MODERATE": (
            15000,
            40000
        ),

        "HIGH": (
            40000,
            80000
        ),

        "SEVERE": (
            80000,
            150000
        )
    }

    return costs.get(
        severity,
        (0, 0)
    )
    