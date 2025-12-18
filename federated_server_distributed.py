# federated_server_distributed.py

import base64
import json
import joblib
import requests

from utils.crypto_utils import CryptoBox

# Same key as in client nodes
SHARED_KEY = b'u61MxLVa1ly2OqZ_TH11PHWSKTCRXmQ2cZndR3XxMsM='
crypto = CryptoBox(SHARED_KEY)

CLIENT_URLS = {
    "ecg": "http://localhost:8001",
    "meta": "http://localhost:8002",
    "audio": "http://localhost:8003",
}


class DistributedFederatedServer:
    """
    Distributed multi-task federated server.
    Talks to three HTTP clients: ECG, metadata, audio.
    """

    def __init__(self, crypto: CryptoBox):
        self.crypto = crypto
        self.pipelines = {"ecg": None, "meta": None, "audio": None}
        self.metrics = {"ecg": None, "meta": None, "audio": None}

    def run_round(self, use_ecg=True, use_meta=True, use_audio=True):
        print("Server: Starting distributed federated round.")

        # ------ ECG ------
        if use_ecg:
            prev_token_b64 = None
            if self.pipelines["ecg"] is not None:
                prev_bytes = self.crypto.encrypt_obj(self.pipelines["ecg"])
                prev_token_b64 = base64.b64encode(prev_bytes).decode()

            print("Server: Requesting ECG client to train...")
            resp = requests.post(
                f"{CLIENT_URLS['ecg']}/train_round",
                json={"prev_pipeline_token": prev_token_b64},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()

            pipeline_bytes = base64.b64decode(data["pipeline_token"].encode())
            self.pipelines["ecg"] = self.crypto.decrypt_obj(pipeline_bytes)
            self.metrics["ecg"] = data["metrics"]
            print("Server: ECG updated.")

        # ------ Meta ------
        if use_meta:
            prev_token_b64 = None
            if self.pipelines["meta"] is not None:
                prev_bytes = self.crypto.encrypt_obj(self.pipelines["meta"])
                prev_token_b64 = base64.b64encode(prev_bytes).decode()

            print("Server: Requesting META client to train...")
            resp = requests.post(
                f"{CLIENT_URLS['meta']}/train_round",
                json={"prev_pipeline_token": prev_token_b64},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()

            pipeline_bytes = base64.b64decode(data["pipeline_token"].encode())
            self.pipelines["meta"] = self.crypto.decrypt_obj(pipeline_bytes)
            self.metrics["meta"] = data["metrics"]
            print("Server: Meta updated.")

        # ------ Audio ------
        if use_audio:
            prev_token_b64 = None
            if self.pipelines["audio"] is not None:
                prev_bytes = self.crypto.encrypt_obj(self.pipelines["audio"])
                prev_token_b64 = base64.b64encode(prev_bytes).decode()

            print("Server: Requesting AUDIO client to train...")
            resp = requests.post(
                f"{CLIENT_URLS['audio']}/train_round",
                json={"prev_pipeline_token": prev_token_b64},
                timeout=1200,
            )
            resp.raise_for_status()
            data = resp.json()

            pipeline_bytes = base64.b64decode(data["pipeline_token"].encode())
            self.pipelines["audio"] = self.crypto.decrypt_obj(pipeline_bytes)
            self.metrics["audio"] = data["metrics"]
            print("Server: Audio updated.")

    def save_pipelines(self, path="federated_multitask_pipelines_distributed.pkl"):
        joblib.dump(self.pipelines, path)
        print(f"Server: Saved pipelines to {path}")

    def save_metrics(self, path="federated_multitask_metrics_distributed.json"):
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Server: Saved metrics to {path}")


if __name__ == "__main__":
    server = DistributedFederatedServer(crypto)

    ROUNDS = 3
    for r in range(ROUNDS):
        print(f"\n=== Distributed Federated Round {r + 1} ===")
        server.run_round(use_ecg=True, use_meta=True, use_audio=True)

    server.save_pipelines()
    server.save_metrics()
