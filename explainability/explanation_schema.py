# explainability/explanation_schema.py

def build_explanation_payload(
    modality: str,
    risk_prob: float,
    risk_label: int,
    class_label: str | None,
    top_features: list[dict],
    visual_summary: str | None = None
):
    return {
        "modality": modality,
        "risk_probability": round(risk_prob, 4),
        "risk_label": "At-risk" if risk_label == 1 else "Normal",
        "predicted_class": class_label,
        "top_contributing_features": top_features,
        "visual_summary": visual_summary,
    }
