# clients/client3_node.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
import joblib

from utils.crypto_utils import CryptoBox
from clients.client3_audio import Client3FL

# SAME key as server + other clients
SHARED_KEY = b'u61MxLVa1ly2OqZ_TH11PHWSKTCRXmQ2cZndR3XxMsM='
crypto = CryptoBox(SHARED_KEY)

app = FastAPI(title="Audio Client Node")

current_pipeline = None  # last trained pipeline stored in memory


class TrainRequest(BaseModel):
    # base64-encoded encrypted pipeline from previous round, or null
    prev_pipeline_token: str | None = None


class TrainResponse(BaseModel):
    pipeline_token: str
    metrics: dict


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    risk_prob: float


@app.get("/health")
def health():
    return {"status": "ok", "client": "audio"}


@app.post("/train_round", response_model=TrainResponse)
def train_round(req: TrainRequest):
    """
    One local training round for audio client.
    If prev_pipeline_token is provided, we decrypt it and warm-start the model.
    """
    global current_pipeline

    prev_pipeline = None
    if req.prev_pipeline_token is not None:
        try:
            prev_bytes = base64.b64decode(req.prev_pipeline_token.encode())
            prev_pipeline = crypto.decrypt_obj(prev_bytes)
            print("Audio node: received previous pipeline for warm-start.")
        except Exception as e:
            print(f"Audio node: could not decode previous pipeline ({e}), training from scratch.")

    client = Client3FL(
        audio_root="datasets/Heartbeats",
        csv_path="datasets/Heartbeats/combined.csv",
        cryptobox=crypto,
        prev_pipeline=prev_pipeline,
    )

    pipeline, metrics = client.local_train()
    current_pipeline = pipeline

    token_bytes = client.send_encrypted_pipeline()
    token_b64 = base64.b64encode(token_bytes).decode()

    return TrainResponse(pipeline_token=token_b64, metrics=metrics)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict risk from a single audio feature vector.
    """
    global current_pipeline
    if current_pipeline is None:
        current_pipeline = joblib.load("client3_audio_pipeline.pkl")

    model = current_pipeline["model"]
    scaler = current_pipeline.get("scaler", None)
    selected_indices = current_pipeline.get("selected_indices", None)

    X = np.array(req.features, dtype=float).reshape(1, -1)

    if scaler is not None:
        X = scaler.transform(X)
    if selected_indices is not None:
        X = X[:, selected_indices]

    proba = model.predict_proba(X)[0, 1]

    return PredictResponse(risk_prob=float(proba))


if __name__ == "__main__":
    # Run from project root:  py -m clients.client3_node
    uvicorn.run(app, host="0.0.0.0", port=8003)
