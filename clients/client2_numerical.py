# clients/client2_numerical.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import PyIFS
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


from utils.crypto_utils import CryptoBox


class Client2:
    """
    Numerical patient data client.
    Assumes:
        - CSV has column 'DEATH_EVENT' as binary label (0=Normal, 1=At-risk).
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.label_column = 'DEATH_EVENT'
        self.scaler = None

        self.X_scaled = None
        self.X_processed = None
        self.y = None
        self.feature_weights = None
        self.variances = None
        self.selected_indices = None
        self.X_selected = None

    def preprocess(self):
        df = pd.read_csv(self.csv_path)

        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        X_raw = df.drop(columns=[self.label_column])
        y = df[self.label_column].astype(int)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        self.X_scaled = X_scaled
        self.X_processed = pd.DataFrame(X_scaled, columns=X_raw.columns)
        self.y = y.values
        print(f"Client2: Data shape after scaling: {self.X_processed.shape}")

    def select_features(self):
        print("Client2: Running InfFS + fuzzy selection on numerical data...")
        inf = PyIFS.InfFS()
        alpha = 0.6
        RANKED, WEIGHT = inf.infFS(self.X_scaled, self.y, alpha, 1, 0)
        self.feature_weights = WEIGHT
        self.variances = np.var(self.X_scaled, axis=0)

        fw_norm = (self.feature_weights - np.min(self.feature_weights)) / (
            np.max(self.feature_weights) - np.min(self.feature_weights) + 1e-6
        )
        fv_norm = (self.variances - np.min(self.variances)) / (
            np.max(self.variances) - np.min(self.variances) + 1e-6
        )

        fw = ctrl.Antecedent(np.linspace(0, 1, 100), 'feature_weight')
        fv = ctrl.Antecedent(np.linspace(0, 1, 100), 'feature_variance')
        ss = ctrl.Consequent(np.linspace(0, 1, 100), 'selection_score')

        for var in [fw, fv, ss]:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.4])
            var['medium'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
            var['high'] = fuzz.trimf(var.universe, [0.6, 1, 1])

        rules = [
            ctrl.Rule(fw['low'] & fv['low'], ss['low']),
            ctrl.Rule(fw['medium'] & fv['low'], ss['medium']),
            ctrl.Rule(fw['high'] & fv['low'], ss['high']),
            ctrl.Rule(fw['low'] & fv['medium'], ss['low']),
            ctrl.Rule(fw['medium'] & fv['medium'], ss['medium']),
            ctrl.Rule(fw['high'] & fv['medium'], ss['high']),
            ctrl.Rule(fw['low'] & fv['high'], ss['low']),
            ctrl.Rule(fw['medium'] & fv['high'], ss['medium']),
            ctrl.Rule(fw['high'] & fv['high'], ss['high']),
        ]

        control_system = ctrl.ControlSystem(rules)
        simulation = ctrl.ControlSystemSimulation(control_system)

        selection_scores = []
        for w, v in zip(fw_norm, fv_norm):
            simulation.input['feature_weight'] = w
            simulation.input['feature_variance'] = v
            simulation.compute()
            selection_scores.append(simulation.output['selection_score'])

        threshold = np.percentile(selection_scores, 70)
        self.selected_indices = [i for i, score in enumerate(selection_scores) if score >= threshold]
        self.X_selected = self.X_scaled[:, self.selected_indices]

        columns = self.df.drop(columns=[self.label_column]).columns
        print(f"Client2: Selected {len(self.selected_indices)} features from numerical data")
        print("Client2: Selected Features:", [columns[i] for i in self.selected_indices])

    def get_selected_features(self):
        return self.X_selected, self.y


class Client2FL(Client2):
    """
    Federated Learning wrapper for Client2 (numerical patient data).
    """

    def __init__(self, csv_path: str, cryptobox: CryptoBox, prev_model=None):
        super().__init__(csv_path)
        self.cryptobox = cryptobox
        self.prev_model = prev_model
        self.pipeline = None

    def local_train(self, test_size=0.2, random_state=42):
        print("Client2FL: Starting local training (numerical)...")
        self.preprocess()
        self.select_features()
        X, y = self.get_selected_features()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        base_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42, 
            n_estimators=50
        )
        
        if self.prev_model is not None:
            print("Client2FL: Warm-starting from previous global numerical model.")
            base_model.fit(X_train, y_train, xgb_model=self.prev_model)
        else:
            print("Client2FL: Training numerical model from scratch.")
            base_model.fit(X_train, y_train)
        
        model = base_model

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(y_val, y_pred, zero_division=0)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }

        print("\n=== Client2FL (Numerical) Validation Metrics ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)

        self.pipeline = {
            'model': model,
            'scaler': self.scaler,
            'selected_indices': self.selected_indices
        }
        print("Client2FL: Local training done.")
        return self.pipeline, metrics

    def send_encrypted_pipeline(self):
        return self.cryptobox.encrypt_obj(self.pipeline)

    def receive_encrypted_pipeline(self, token):
        self.pipeline = self.cryptobox.decrypt_obj(token)
        return self.pipeline
