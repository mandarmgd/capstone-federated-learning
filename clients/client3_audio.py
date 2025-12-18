# clients/client3_audio.py

import os
import pandas as pd
import numpy as np
import librosa
import unicodedata
import string

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from utils.crypto_utils import CryptoBox
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


class Client3:
    """
    Heart sound (PCG) client.

    CSV columns (combined.csv):
        - 'fname'   : relative path like 'set_a/normal__201101070538.wav'
                      or 'set_b/murmur__171_1307971016233_D.wav'
        - 'label'   : one of {'normal', 'murmur', 'extrahls', 'extrastole'}
        - 'sublabel': 0 (normal) or 1 (at-risk)   [kept for reference/metrics]

    Federated Learning + personalization (Option B):

        Single model (4-class):
            y_multiclass = label_idx in {0,1,2,3}
                0 -> normal
                1 -> murmur
                2 -> extrahls
                3 -> extrastole

        Derived binary risk at inference:
            risk = 0 if predicted_label == 'normal'
                   1 otherwise

        'label' (4-way) is available in the pipeline for personalization text.
    """

    def __init__(self, audio_root: str, csv_path: str):
        self.audio_root = audio_root
        self.set_a_dir = os.path.join(audio_root, 'set_a')
        self.set_b_dir = os.path.join(audio_root, 'set_b')
        self.csv_path = csv_path

        df = pd.read_csv(csv_path)

        # Normalize label strings and keep only the four categories
        df['label'] = df['label'].astype(str).str.lower()
        valid_labels = ['normal', 'murmur', 'extrahls', 'extrastole']
        df = df[df['label'].isin(valid_labels)].copy()

        # Ensure sublabel is int 0/1 (used only for reference/metrics)
        df['sublabel'] = df['sublabel'].astype(int)

        # Map label -> integer class for multi-class model
        self.label_to_idx = {
            'normal': 0,
            'murmur': 1,
            'extrahls': 2,
            'extrastole': 3,
        }
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.normal_idx = self.label_to_idx['normal']

        df['label_idx'] = df['label'].map(self.label_to_idx)

        self.df = df
        self.train_df = df[df['label'].notnull()].copy()

        # Features & targets (filled by preprocess())
        self.X_train_raw = None
        self.X_train_scaled = None
        self.y_multiclass = None  # 0..3

        self.scaler = None
        self.selected_indices = None
        self.X_train_selected = None

        print(f"Client3: Train samples: {len(self.train_df)}")

    # ---------------------- File resolving ---------------------- #

    def get_full_path(self, fname):
        """
        Resolve a relative path from CSV into a real path under audio_root.

        Strategy:
        1) Clean up invisible / weird characters and normalize slashes.
        2) Try audio_root + CSV path directly.
        3) Try basename in set_a and set_b.
        4) Walk the whole Heartbeats tree and look for matching basename.
        """

        raw = str(fname)
        # Debug: show exactly what came from CSV
        # Comment this out if it becomes too noisy:
        # print(f"[DEBUG] raw fname from CSV: {repr(raw)}")

        # 1) Normalize Unicode
        fname = unicodedata.normalize("NFC", raw)

        # Strip common invisible junk
        for bad in ["\u200b", "\ufeff", "\r", "\n", "\t"]:
            fname = fname.replace(bad, "")

        # Keep only printable chars
        fname = "".join(ch for ch in fname if ch in string.printable)

        # Trim whitespace + leading ./ or .\
        fname = fname.strip()
        fname = fname.lstrip("./\\")

        # Normalize slashes
        fname = fname.replace("\\", "/")

        # print(f"[DEBUG] normalized fname: {repr(fname)}")

        # Split parts
        parts = [p for p in fname.split("/") if p != ""]
        fname_norm = "/".join(parts)

        # 2) Try direct path: audio_root / <parts from CSV>
        candidate = os.path.join(self.audio_root, *parts)
        if os.path.exists(candidate):
            # print(f"[DEBUG] Found directly: {candidate}")
            return candidate

        # 3) Try just basename in set_a / set_b
        base = os.path.basename(fname_norm)
        cand_a = os.path.join(self.set_a_dir, base)
        cand_b = os.path.join(self.set_b_dir, base)

        if os.path.exists(cand_a):
            # print(f"[DEBUG] Found by basename in set_a: {cand_a}")
            return cand_a
        if os.path.exists(cand_b):
            # print(f"[DEBUG] Found by basename in set_b: {cand_b}")
            return cand_b

        # 4) LAST RESORT: walk entire Heartbeats tree and find basename
        for root, dirs, files in os.walk(self.audio_root):
            if base in files:
                found = os.path.join(root, base)
                print(f"[INFO] Resolved '{fname_norm}' by walking -> {found}")
                return found

        # Still not found – print helpful info
        print(f"[WARN] Audio file not found for fname={repr(fname_norm)} base={repr(base)}")
        print(f"       Tried: {candidate}")
        print(f"       Also:  {cand_a}")
        print(f"              {cand_b}")

        return None

    # --------------------- Feature extraction --------------------- #

    def extract_audio_features(self, file_path):
        """
        Extracts a 27-D feature vector:
            13 MFCCs + 12 chroma + 1 ZCR + 1 spectral centroid
        and sanitizes it to remove NaNs/Infs.
        """
        try:
            y, sr = librosa.load(file_path, sr=None)
            if y is None or len(y) == 0:
                raise ValueError("Empty audio")

            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

            feats = np.hstack([mfccs, chroma, zcr, centroid])  # 13 + 12 + 1 + 1 = 27
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

            return feats

        except Exception as e:
            print(f"Could not extract features from {file_path}: {e}")
            # Keep feature dimension consistent
            return np.zeros(27, dtype=float)

    def extract_features(self, df):
        features = []
        for fname in df['fname']:
            path = self.get_full_path(fname)
            if path:
                feats = self.extract_audio_features(path)
            else:
                feats = np.zeros(27, dtype=float)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(feats)
        return np.array(features, dtype=float)

    # --------------------- Preprocessing --------------------- #

    def preprocess(self):
        """
        - Extract raw 27-D features
        - Builds multi-class target (0..3) from label_idx
        - Scales features (StandardScaler)
        """
        print("Client3: Extracting audio features...")
        self.X_train_raw = self.extract_features(self.train_df)

        # 4-way personalization / master label
        self.y_multiclass = self.train_df['label_idx'].values.astype(int)

        print("Client3: Scaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X_train_raw)

        # Extra safety
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        self.X_train_scaled = X_scaled
        print(f"Client3: X_train shape: {self.X_train_scaled.shape}")

    # --------------------- Feature Selection --------------------- #

    def select_features(self, max_features: int = 15):
        """
        Feature selection using mutual information against the 4-way class label.

        - Use mutual information between each feature and y_multiclass.
        - Pick the top-k (up to max_features) features.
        """
        print("Client3: Selecting features via mutual information...")

        X = np.nan_to_num(self.X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.y_multiclass

        mi = mutual_info_classif(X, y, discrete_features=False)
        sorted_idx = np.argsort(mi)[::-1]

        k = min(max_features, X.shape[1])
        self.selected_indices = sorted_idx[:k]

        print(f"Client3: Selected top {k} features by mutual information.")
        print("Client3: Selected indices:", self.selected_indices.tolist())

        self.X_train_selected = X[:, self.selected_indices]
        print(f"Client3: X_train_selected shape: {self.X_train_selected.shape}")

    # --------------------- Accessor --------------------- #

    def get_selected_features(self):
        """
        Returns:
            X_selected, y_multiclass
        """
        return self.X_train_selected, self.y_multiclass


class Client3FL(Client3):
    """
    Federated Learning wrapper for Client3 (audio).

    Option B: Single multi-class model

        - XGBoost multi-class model: predicts 0..3 -> (normal / murmur / extrahls / extrastole)
        - Binary risk is derived at inference time:
              risk = 0 if predicted_label == 'normal'
                     1 otherwise

    Returned pipeline (encrypted by server):

        {
            'model':           multi_class XGBClassifier,
            'scaler':          StandardScaler,
            'selected_indices': list[int],
            'label_to_idx':    dict,
            'idx_to_label':    dict
        }
    """

    def __init__(self, audio_root: str, csv_path: str,
                 cryptobox: CryptoBox, prev_pipeline=None):
        super().__init__(audio_root, csv_path)
        self.cryptobox = cryptobox
        self.prev_pipeline = prev_pipeline  # may be None or a dict with model + selected_indices
        self.pipeline = None

    def local_train(self, test_size=0.2, random_state=42):
        print("Client3FL: Starting local training (audio)...")

        # 1) Preprocess to get scaled features + y_multiclass
        self.preprocess()

        # 2) Reuse previous selected_indices if available, otherwise run MI selection
        if self.prev_pipeline is not None and self.prev_pipeline.get("selected_indices") is not None:
            print("Client3FL: Reusing selected feature indices from previous round.")
            self.selected_indices = np.array(self.prev_pipeline["selected_indices"])
            X = self.X_train_scaled[:, self.selected_indices]
        else:
            self.select_features()
            X, y_multi = self.get_selected_features()
            self.y_multiclass = y_multi
            # Note: y_multiclass already set in preprocess; this just syncs shapes
            # if any rows were dropped due to missing features.

        y_multi = self.y_multiclass

        # 3) Train/validation split (on 4-way labels)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_multi,
            test_size=test_size,
            stratify=y_multi,
            random_state=random_state,
        )

        # 4) Optional: Balance classes with SMOTE on 4-way labels
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        print("Client3FL: After SMOTE (multi-class):",
              dict(zip(*np.unique(y_train_bal, return_counts=True))))

        # 5) Multi-class XGBoost (single model)
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=4,
            random_state=42,
            n_estimators=80,
        )

        if self.prev_pipeline is not None and self.prev_pipeline.get("model") is not None:
            prev_model = self.prev_pipeline["model"]
            print("Client3FL: Warm-starting multi-class model from previous global audio model.")
            model.fit(X_train_bal, y_train_bal, xgb_model=prev_model)
        else:
            print("Client3FL: Training multi-class audio model from scratch.")
            model.fit(X_train_bal, y_train_bal)

        # 6) Evaluation – multi-class AND derived binary risk

        # Multi-class metrics
        y_pred = model.predict(X_val)
        acc_multi = accuracy_score(y_val, y_pred)
        report_multi = classification_report(
            y_val,
            y_pred,
            target_names=[self.idx_to_label[i] for i in range(4)],
            zero_division=0
        )
        cm_multi = confusion_matrix(y_val, y_pred)

        print("\n=== Client3FL (Audio, 4-way Category) Validation Metrics ===")
        print(f"Multi-class accuracy: {acc_multi:.4f}")
        print("Multi-class confusion matrix:\n", cm_multi)
        print("Multi-class classification report:\n", report_multi)

        # Derived binary risk metrics:
        #   risk = 0 if label_idx == normal_idx else 1
        ybin_val = (y_val != self.normal_idx).astype(int)
        ybin_pred = (y_pred != self.normal_idx).astype(int)

        acc_bin = accuracy_score(ybin_val, ybin_pred)
        prec_bin = precision_score(ybin_val, ybin_pred, zero_division=0)
        rec_bin = recall_score(ybin_val, ybin_pred, zero_division=0)
        f1_bin = f1_score(ybin_val, ybin_pred, zero_division=0)
        cm_bin = confusion_matrix(ybin_val, ybin_pred)
        report_bin = classification_report(ybin_val, ybin_pred, zero_division=0)

        print("\n=== Client3FL (Audio, Derived Binary Risk) Validation Metrics ===")
        print(f"Accuracy : {acc_bin:.4f}")
        print(f"Precision: {prec_bin:.4f}")
        print(f"Recall   : {rec_bin:.4f}")
        print(f"F1-score : {f1_bin:.4f}")
        print("Confusion Matrix:\n", cm_bin)
        print("Classification Report:\n", report_bin)

        metrics = {
            "multi_accuracy": acc_multi,
            "multi_confusion_matrix": cm_multi.tolist(),
            "multi_classification_report": report_multi,
            "bin_accuracy": acc_bin,
            "bin_precision": prec_bin,
            "bin_recall": rec_bin,
            "bin_f1": f1_bin,
            "bin_confusion_matrix": cm_bin.tolist(),
            "bin_classification_report": report_bin,
        }

        # 7) Save pipeline (server treats 'model' as the audio model)
        self.pipeline = {
            'model': model,                 # 4-way model
            'scaler': self.scaler,
            'selected_indices': self.selected_indices.tolist()
            if self.selected_indices is not None else None,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
        }

        print("Client3FL: Local training done.")
        return self.pipeline, metrics

    def send_encrypted_pipeline(self):
        return self.cryptobox.encrypt_obj(self.pipeline)

    def receive_encrypted_pipeline(self, token):
        self.pipeline = self.cryptobox.decrypt_obj(token)
        return self.pipeline
