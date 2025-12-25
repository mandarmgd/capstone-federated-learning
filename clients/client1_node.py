# clients/client1_node.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
import joblib

from utils.crypto_utils import CryptoBox
from clients.client1_image import Client1FL

# Use the SAME key as the server
SHARED_KEY = b'u61MxLVa1ly2OqZ_TH11PHWSKTCRXmQ2cZndR3XxMsM='
crypto = CryptoBox(SHARED_KEY)

app = FastAPI(title="ECG Client Node")

# Pipeline kept in memory
current_pipeline = None


# ---------- Schemas ----------

class TrainRequest(BaseModel):
    # base64-encoded encrypted pipeline from previous round, or null
    prev_pipeline_token: str | None = None


class TrainResponse(BaseModel):
    pipeline_token: str          # base64-encoded encrypted pipeline
    metrics: dict                # validation metrics from local_train


class PredictRequest(BaseModel):
    # Fused ECG feature vector (same representation used at training time)
    features: list[float]


class PredictResponse(BaseModel):
    risk_prob: float             # probability of At-risk (label 1)


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "client": "ecg"}


@app.post("/train_round", response_model=TrainResponse)
def train_round(req: TrainRequest):
    """
    One local training round for ECG.
    Optionally warm-started from previous global pipeline.
    """
    global current_pipeline

    prev_model = None
    if req.prev_pipeline_token is not None:
        # Decrypt previous pipeline and extract model
        prev_pipeline_bytes = base64.b64decode(req.prev_pipeline_token.encode())
        prev_pipeline = crypto.decrypt_obj(prev_pipeline_bytes)
        prev_model = prev_pipeline.get("model", None)

    # Local federated client
    client = Client1FL("datasets/ECG Dataset", crypto, prev_model=prev_model)
    pipeline, metrics = client.local_train()

    current_pipeline = pipeline

    # Encrypt pipeline to send back to server
    token_bytes = crypto.encrypt_obj(pipeline)
    token_b64 = base64.b64encode(token_bytes).decode()

    return TrainResponse(pipeline_token=token_b64, metrics=metrics)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict risk from a single fused ECG feature vector.
    For now we assume features are already in the fused space before selection.
    """
    global current_pipeline
    if current_pipeline is None:
        # Fallback: try to load a saved pipeline
        current_pipeline = joblib.load("client1_ecg_pipeline.pkl")

    model = current_pipeline["model"]
    scaler = current_pipeline.get("scaler", None)
    selected_indices = current_pipeline.get("selected_indices", None)

    X = np.array(req.features, dtype=float).reshape(1, -1)

    if selected_indices is not None:
        X = X[:, selected_indices]

    if scaler is not None:
        X = scaler.transform(X)

    proba = model.predict_proba(X)[0, 1]  # probability of label=1 (At-risk)

    return PredictResponse(risk_prob=float(proba))


if __name__ == "__main__":
    # Run this client node as:  py -m clients.client1_node  OR  py clients\client1_node.py
    uvicorn.run(app, host="0.0.0.0", port=8001)
