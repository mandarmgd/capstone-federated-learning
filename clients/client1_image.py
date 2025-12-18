# clients/client1_image.py

import numpy as np
import pandas as pd
import cv2
import os
from os import listdir
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
from keras._tf_keras.keras.applications.resnet50 import ResNet50
import skimage
from scipy.stats import kurtosis
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import PyIFS

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from utils.crypto_utils import CryptoBox
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

class Client1:
    """
    ECG image client (Option B).

    Multi-class labels:
        0 -> Normal Person
        1 -> Abnormal heartbeat
        2 -> History of MI

    Derived binary risk:
        0 if Normal Person
        1 otherwise
    """

    def __init__(self, image_folder: str):
        self.image_folder = image_folder
        self.image_data = None
        self.image_table = None
        self.extracted = None
        self.feature_weights = None
        self.selected_indices = None
        self.fused_features = None
        self.labels = None
        self.keep_cols = None

        self.label_to_idx = {
            'Normal Person': 0,
            'Abnormal heartbeat': 1,
            'History of MI': 2
        }
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.normal_idx = 0

    # --------------------- Data Loading --------------------- #

    def loadData(self):
        root_dir = listdir(self.image_folder)
        image_list, label_list = [], []

        for directory in root_dir:
            if directory not in self.label_to_idx:
                continue

            dir_path = os.path.join(self.image_folder, directory)
            if not os.path.isdir(dir_path):
                continue

            for file in listdir(dir_path):
                img_path = os.path.join(dir_path, file)
                image = self.convert_image_to_array(img_path)
                if image is not None and image.size != 0:
                    image_list.append(image)
                    label_list.append(self.label_to_idx[directory])

        image_list = np.array(image_list) / 255.0
        label_list = np.array(label_list, dtype=int)
        return image_list, label_list

    def convert_image_to_array(self, image_dir):
        try:
            image = cv2.imread(image_dir)
            if image is not None:
                image = cv2.resize(image, (75, 75))
                return img_to_array(image)
            return None
        except Exception as e:
            print(f"Error reading image {image_dir}: {e}")
            return None

    # -------------------- Feature Extraction ------------------- #

    def getWeights(self, images, labels):
        base_model_1 = InceptionV3(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
        features1 = base_model_1.predict(images, verbose=0).reshape(images.shape[0], -1)

        base_model_2 = ResNet50(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
        features2 = base_model_2.predict(images, verbose=0).reshape(images.shape[0], -1)

        features2 = features2[:, :features1.shape[1]]
        average_features = (features1 + features2) / 2.0

        handcrafted_features = self.feature_extraction(images)
        return average_features, handcrafted_features, labels

    def feature_extraction(self, df):
        data = {k: [] for k in [
            'Mean_R', 'Mean_G', 'Mean_B', 'Mean_RGB', 'StdDev_RGB',
            'Variance_RGB', 'Median_RGB', 'Entropy', 'Skewness_RGB',
            'Kurtosis_RGB', 'Brightness', 'Contrast',
            'GLCM_Contrast', 'GLCM_Energy', 'GLCM_Homogeneity',
            'GLCM_Correlation',
            'HuMoment_1', 'HuMoment_2', 'HuMoment_3',
            'HuMoment_4', 'HuMoment_5', 'HuMoment_6', 'HuMoment_7'
        ]}

        for image in df:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            L, _, _ = cv2.split(lab)
            L = L / (np.max(L) + 1e-6)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            humoments = cv2.HuMoments(cv2.moments(gray)).flatten()
            gray_u8 = skimage.img_as_ubyte(gray)
            gray_q = (gray_u8 // 32).astype(np.uint8)
            g = skimage.feature.graycomatrix(gray_q, [1], [0], levels=8,
                                             symmetric=False, normed=True)

            data['Mean_R'].append(image[:, :, 0].mean())
            data['Mean_G'].append(image[:, :, 1].mean())
            data['Mean_B'].append(image[:, :, 2].mean())
            data['Mean_RGB'].append(image.mean())
            data['StdDev_RGB'].append(np.std(image))
            data['Variance_RGB'].append(np.var(image))
            data['Median_RGB'].append(np.median(image))
            data['Entropy'].append(skimage.measure.shannon_entropy(image))
            data['Skewness_RGB'].append(3 * (np.mean(image) - np.median(image)) / (np.std(image) + 1e-6))
            data['Kurtosis_RGB'].append(kurtosis(image.flatten()))
            data['Brightness'].append(np.mean(L))
            data['Contrast'].append((np.max(L) - np.min(L)) / (np.max(L) + np.min(L) + 1e-6))
            data['GLCM_Contrast'].append(skimage.feature.graycoprops(g, 'contrast')[0][0])
            data['GLCM_Energy'].append(skimage.feature.graycoprops(g, 'energy')[0][0])
            data['GLCM_Homogeneity'].append(skimage.feature.graycoprops(g, 'homogeneity')[0][0])
            data['GLCM_Correlation'].append(skimage.feature.graycoprops(g, 'correlation')[0][0])

            for k in range(1, 8):
                data[f'HuMoment_{k}'].append(humoments[k - 1])

        df_features = pd.DataFrame(data)
        self.image_table = df_features
        return df_features

    # -------------------- Feature Selection -------------------- #

    def select_features(self):
        cleaned_data = np.nan_to_num(self.image_data, nan=0.0, posinf=0.0, neginf=0.0)
        variances = np.var(cleaned_data, axis=0)
        self.keep_cols = variances > 1e-8
        cleaned_data = cleaned_data[:, self.keep_cols]

        inf = PyIFS.InfFS()
        RANKED, WEIGHT = inf.infFS(
            cleaned_data,
            self.labels,
            alpha=0.6,
            verbose=0,
            supervision=1
        )

        self.image_data = cleaned_data
        variances = np.var(self.image_data, axis=0)

        feature_weight = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'feature_weight')
        feature_variance = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'feature_variance')
        selection_score = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'selection_score')

        for var in [feature_weight, feature_variance, selection_score]:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
            var['medium'] = fuzz.trimf(var.universe, [0, 0.5, 1])
            var['high'] = fuzz.trimf(var.universe, [0.5, 1, 1])

        rules = [
            ctrl.Rule(feature_weight['low'] & feature_variance['low'], selection_score['low']),
            ctrl.Rule(feature_weight['medium'] & feature_variance['medium'], selection_score['medium']),
            ctrl.Rule(feature_weight['high'] & feature_variance['high'], selection_score['high']),
        ]

        ctrl_system = ctrl.ControlSystem(rules)

        selection_scores = []
        for w, v in zip(WEIGHT, variances):
            sim = ctrl.ControlSystemSimulation(ctrl_system)
            sim.input['feature_weight'] = float(w)
            sim.input['feature_variance'] = float(v)
            sim.compute()
            selection_scores.append(sim.output.get('selection_score', 0.0))

        mean_score = np.mean(selection_scores)
        self.selected_indices = [i for i, s in enumerate(selection_scores) if s > mean_score][:100]

        X_selected = self.image_data[:, self.selected_indices]
        self.fused_features = np.concatenate(
            (X_selected, self.extracted.values), axis=1
        )

    # -------------------- Orchestration -------------------- #

    def load_and_extract(self):
        X, y = self.loadData()
        avg, ext, lab = self.getWeights(X, y)
        self.image_data = np.nan_to_num(avg, nan=0.0, posinf=0.0, neginf=0.0)
        self.extracted = ext.fillna(0)
        self.labels = lab

    def get_selected_features(self):
        return self.fused_features, self.labels


class Client1FL(Client1):
    """
    FL wrapper for ECG (Option B).
    """

    def __init__(self, image_folder: str, cryptobox: CryptoBox, prev_model=None):
        super().__init__(image_folder)
        self.cryptobox = cryptobox
        self.prev_model = prev_model
        self.pipeline = None

    def local_train(self, test_size=0.2, random_state=42):
        print("Client1FL: Starting local training (ECG)...")

        self.load_and_extract()
        self.select_features()
        X, y = self.get_selected_features()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=80
        )

        if self.prev_model is not None:
            model.fit(X_train, y_train, xgb_model=self.prev_model)
        else:
            model.fit(X_train, y_train)

        # ---- Multi-class evaluation (3-way) ----
        y_pred = model.predict(X_val)

        acc_multi = accuracy_score(y_val, y_pred)
        cm_multi = confusion_matrix(y_val, y_pred)
        report_multi = classification_report(
            y_val,
            y_pred,
            target_names=[self.idx_to_label[i] for i in range(3)],
            zero_division=0
        )

        print("\n=== Client1FL (ECG, 3-way Category) Validation Metrics ===")
        print(f"Multi-class accuracy: {acc_multi:.4f}")
        print("Multi-class confusion matrix:\n", cm_multi)
        print("Multi-class classification report:\n", report_multi)

        # ---- Derived binary risk evaluation ----
        ybin_val = (y_val != self.normal_idx).astype(int)
        ybin_pred = (y_pred != self.normal_idx).astype(int)

        acc_bin = accuracy_score(ybin_val, ybin_pred)
        prec_bin = precision_score(ybin_val, ybin_pred, zero_division=0)
        rec_bin = recall_score(ybin_val, ybin_pred, zero_division=0)
        f1_bin = f1_score(ybin_val, ybin_pred, zero_division=0)
        cm_bin = confusion_matrix(ybin_val, ybin_pred)
        report_bin = classification_report(ybin_val, ybin_pred, zero_division=0)

        print("\n=== Client1FL (ECG, Derived Binary Risk) Validation Metrics ===")
        print(f"Accuracy : {acc_bin:.4f}")
        print(f"Precision: {prec_bin:.4f}")
        print(f"Recall   : {rec_bin:.4f}")
        print(f"F1-score : {f1_bin:.4f}")
        print("Confusion Matrix:\n", cm_bin)
        print("Classification Report:\n", report_bin)

        self.pipeline = {
            "model": model,                   
            "selected_indices": self.selected_indices,
            "keep_cols": self.keep_cols,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
        }

        return self.pipeline, {}

    def send_encrypted_pipeline(self):
        return self.cryptobox.encrypt_obj(self.pipeline)

    def receive_encrypted_pipeline(self, token):
        self.pipeline = self.cryptobox.decrypt_obj(token)
        return self.pipeline
