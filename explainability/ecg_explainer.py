# explainability/ecg_explainer.py

from explainability.gradcam_ecg import compute_gradcam_feature_based
from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
import cv2
import numpy as np

def explain_ecg(image_path: str, ecg_pipeline: dict, ecg_out: dict):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (75, 75))

    cnn = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(75, 75, 3)
    )

    gradcam_info = compute_gradcam_feature_based(
        cnn_model=cnn,
        image_array=image,
        target_layer_name="mixed10"
    )

    return {
        "modality": "ecg",
        "predicted_subclass": ecg_out.get("reason_label"),
        "risk_prob": round(ecg_out["risk_prob"], 4),
        "signals": [
            gradcam_info["interpretation"],
            "Handcrafted texture and morphology features contributed to classification"
        ]
    }
