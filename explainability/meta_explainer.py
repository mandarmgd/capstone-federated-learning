# explainability/meta_explainer.py

import numpy as np

def explain_meta(meta_vector, meta_pipeline: dict):
    model = meta_pipeline["model"]
    importance = model.feature_importances_

    top_indices = np.argsort(importance)[-3:][::-1]

    return {
        "modality": "meta",
        "top_contributing_features": [
            {
                "feature_index": int(i),
                "model_influence": "high"
            }
            for i in top_indices
        ]
    }
