# clients/client2_node.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
import joblib

from utils.crypto_utils import CryptoBox
from clients.client2_numerical import Client2FL

SHARED_KEY = b'u61MxLVa1ly2OqZ_TH11PHWSKTCRXmQ2cZndR3XxMsM='
crypto = CryptoBox(SHARED_KEY)

app = FastAPI(title="Metadata Client Node")

current_pipeline = None


class TrainRequest(BaseModel):
    prev_pipeline_token: str | None = None


class TrainResponse(BaseModel):
    pipeline_token: str
    metrics: dict


class PredictRequest(BaseModel):
    # Raw numeric patient features in same order as training CSV columns (without label)
    features: list[float]


class PredictResponse(BaseModel):
    risk_prob: float


@app.get("/health")
def health():
    return {"status": "ok", "client": "meta"}


@app.post("/train_round", response_model=TrainResponse)
def train_round(req: TrainRequest):
    global current_pipeline

    prev_model = None
    if req.prev_pipeline_token is not None:
        prev_pipeline_bytes = base64.b64decode(req.prev_pipeline_token.encode())
        prev_pipeline = crypto.decrypt_obj(prev_pipeline_bytes)
        prev_model = prev_pipeline.get("model", None)

    client = Client2FL("datasets/heart_failure.csv", crypto, prev_model=prev_model)
    pipeline, metrics = client.local_train()

    current_pipeline = pipeline

    token_bytes = crypto.encrypt_obj(pipeline)
    token_b64 = base64.b64encode(token_bytes).decode()

    return TrainResponse(pipeline_token=token_b64, metrics=metrics)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict risk from patient metadata feature vector.
    Assumes same feature order as training.
    """
    global current_pipeline
    if current_pipeline is None:
        current_pipeline = joblib.load("client2_meta_pipeline.pkl")

    model = current_pipeline["model"]
    scaler = current_pipeline.get("scaler", None)
    selected_indices = current_pipeline.get("selected_indices", None)

    X = np.array(req.features, dtype=float).reshape(1, -1)

    # For numerical client, typically selection happens after scaling, but here
    # we assume features are raw, so replicate your training pipeline order:
    if scaler is not None:
        X = scaler.transform(X)
    if selected_indices is not None:
        X = X[:, selected_indices]

    proba = model.predict_proba(X)[0, 1]

    return PredictResponse(risk_prob=float(proba))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
