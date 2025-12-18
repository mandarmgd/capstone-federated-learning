# explainability/gradcam_ecg.py

import numpy as np
import tensorflow as tf
import cv2

def compute_gradcam_feature_based(
    cnn_model,
    image_array,
    target_layer_name: str
):
    """
    Grad-CAM ONLY on CNN feature extractor.
    Returns a qualitative summary instead of raw heatmap.
    """

    img = np.expand_dims(image_array / 255.0, axis=0)

    grad_model = tf.keras.models.Model(
        [cnn_model.inputs],
        [cnn_model.get_layer(target_layer_name).output]
    )

    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img)
        loss = tf.reduce_mean(conv_outputs)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    activation_strength = tf.reduce_mean(
        tf.multiply(pooled_grads, conv_outputs)
    ).numpy()

    # Convert numeric result into interpretable signal
    if activation_strength > 0.6:
        strength = "strong"
    elif activation_strength > 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "activation_strength": strength,
        "interpretation": "CNN feature regions showed {} activation".format(strength)
    }
