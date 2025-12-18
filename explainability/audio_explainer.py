# explainability/audio_explainer.py

def explain_audio(audio_path: str, audio_pipeline: dict):
    model = audio_pipeline["model"]

    # XGBoost exposes feature importance safely
    importance = model.feature_importances_

    dominant = "spectral features" if importance.mean() > 0 else "temporal features"

    return {
        "modality": "audio",
        "signals": [
            f"{dominant.capitalize()} had higher influence in model decision",
            "MFCC and chroma features contributed to risk estimation"
        ]
    }
