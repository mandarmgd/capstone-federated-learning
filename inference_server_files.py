# inference_server_files.py

import os
import joblib
import numpy as np
import cv2
import librosa
import skimage
from scipy.stats import kurtosis

from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
from keras._tf_keras.keras.applications.resnet50 import ResNet50
from sklearn.preprocessing import StandardScaler

# 🔹 Explainability + LLM (ADDED)
from explainability.ecg_explainer import explain_ecg
from explainability.audio_explainer import explain_audio
from explainability.meta_explainer import explain_meta
from llm.personalization_engine import PersonalizationEngine

PIPELINES_PATH = "federated_multitask_pipelines_distributed.pkl"

# Lazy-load global pipelines and CNN backbones
_pipelines_cache = None
_inception = None
_resnet = None

# 🔹 Lazy-load LLM engine (ADDED)
_llm_engine = None


def get_llm_engine():
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = PersonalizationEngine(
            model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        )
    return _llm_engine


def load_pipelines():
    global _pipelines_cache
    if _pipelines_cache is None:
        _pipelines_cache = joblib.load(PIPELINES_PATH)
        print(f"[inference] Loaded pipelines from {PIPELINES_PATH}")
    return _pipelines_cache


def _load_backbones():
    global _inception, _resnet
    if _inception is None:
        _inception = InceptionV3(include_top=False, weights="imagenet", input_shape=(75, 75, 3))
    if _resnet is None:
        _resnet = ResNet50(include_top=False, weights="imagenet", input_shape=(75, 75, 3))
    return _inception, _resnet


# ---------------------------------------------------------------------
# ECG IMAGE → fused feature vector → risk probability (+ reason)
# ---------------------------------------------------------------------

def _convert_image_to_array(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (75, 75))
    return img_to_array(img)


def _ecg_handcrafted_features(image_array: np.ndarray) -> np.ndarray:
    data = {
        'Mean_R': [], 'Mean_G': [], 'Mean_B': [], 'Mean_RGB': [],
        'StdDev_RGB': [], 'Variance_RGB': [], 'Median_RGB': [],
        'Entropy': [], 'Skewness_RGB': [], 'Kurtosis_RGB': [],
        'Brightness': [], 'Contrast': [],
        'GLCM_Contrast': [], 'GLCM_Energy': [],
        'GLCM_Homogeneity': [], 'GLCM_Correlation': [],
        'HuMoment_1': [], 'HuMoment_2': [], 'HuMoment_3' : [],
        'HuMoment_4': [], 'HuMoment_5': [],
        'HuMoment_6': [], 'HuMoment_7': []
    }

    img = image_array.astype(np.uint8)
    L, _, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    L_norm = L / (np.max(L) + 1e-6)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    humoments = list(cv2.HuMoments(cv2.moments(gray)).flatten())
    gray8 = skimage.img_as_ubyte(gray)
    gray_q = gray8 // 32
    g = skimage.feature.graycomatrix(gray_q, [1], [0], levels=8, symmetric=False, normed=True)

    data['Mean_R'].append(img[:, :, 0].mean())
    data['Mean_G'].append(img[:, :, 1].mean())
    data['Mean_B'].append(img[:, :, 2].mean())
    data['Mean_RGB'].append(img.mean())
    data['StdDev_RGB'].append(np.std(img))
    data['Variance_RGB'].append(np.var(img))
    data['Median_RGB'].append(np.median(img))
    data['Entropy'].append(skimage.measure.shannon_entropy(img))
    data['Skewness_RGB'].append(3 * (np.mean(img) - np.median(img)) / (np.std(img) + 1e-6))
    data['Kurtosis_RGB'].append(kurtosis(img.flatten()))
    data['Brightness'].append(np.mean(L_norm))
    data['Contrast'].append((np.max(L_norm) - np.min(L_norm)) /
                            (np.max(L_norm) + np.min(L_norm) + 1e-6))
    data['GLCM_Contrast'].append(skimage.feature.graycoprops(g, 'contrast')[0][0])
    data['GLCM_Energy'].append(skimage.feature.graycoprops(g, 'energy')[0][0])
    data['GLCM_Homogeneity'].append(skimage.feature.graycoprops(g, 'homogeneity')[0][0])
    data['GLCM_Correlation'].append(skimage.feature.graycoprops(g, 'correlation')[0][0])

    for k in range(1, 8):
        data[f'HuMoment_{k}'].append(humoments[k - 1])

    feats = np.array([v[0] for v in data.values()], dtype=float).reshape(1, -1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _ecg_fused_features_from_image(image_path: str, selected_indices: np.ndarray) -> np.ndarray:
    inception, resnet = _load_backbones()

    img_arr = _convert_image_to_array(image_path)
    img_norm = np.array([img_arr]) / 255.0

    f1 = inception.predict(img_norm, verbose=0).reshape(1, -1)
    f2 = resnet.predict(img_norm, verbose=0).reshape(1, -1)
    f2 = f2[:, :f1.shape[1]]

    avg_features = (f1 + f2) / 2.0
    avg_features = np.nan_to_num(avg_features, nan=0.0, posinf=0.0, neginf=0.0)

    hc = _ecg_handcrafted_features(img_arr)
    X_sel = avg_features[:, selected_indices]
    return np.concatenate([X_sel, hc], axis=1)


def predict_ecg_from_image(image_path: str):
    pipes = load_pipelines()
    ecg_pipe = pipes.get("ecg")
    if ecg_pipe is None:
        raise RuntimeError("No ECG pipeline found")

    model = ecg_pipe["model"]
    selected_indices = np.array(ecg_pipe["selected_indices"])
    idx_to_label = ecg_pipe.get("idx_to_label")

    X = _ecg_fused_features_from_image(image_path, selected_indices)
    probs = model.predict_proba(X)[0]

    risk_prob = float(probs[1] + probs[2])
    risk_label = int(risk_prob >= 0.5)
    class_idx = int(np.argmax(probs))
    reason_label = idx_to_label.get(class_idx, str(class_idx))

    return {
        "risk_prob": risk_prob,
        "risk_label": risk_label,
        "reason_label": reason_label,
        "raw_probs": probs.tolist()
    }


# ---------------------------------------------------------------------
# AUDIO + META (UNCHANGED)
# ---------------------------------------------------------------------

def _extract_audio_features(file_path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        feats = np.hstack([mfccs, chroma, zcr, centroid])
        return feats.reshape(1, -1)
    except Exception:
        return np.zeros((1, 27), dtype=float)


def _prepare_audio_features(wav_path: str):
    pipes = load_pipelines()
    audio_pipe = pipes.get("audio")

    scaler = audio_pipe.get("scaler")
    selected_indices = np.array(audio_pipe["selected_indices"])

    X_raw = _extract_audio_features(wav_path)
    X_scaled = scaler.transform(X_raw) if scaler else X_raw
    X_final = X_scaled[:, selected_indices] if selected_indices is not None else X_scaled

    return X_final, audio_pipe


def predict_audio_from_wav(wav_path: str):
    X_final, audio_pipe = _prepare_audio_features(wav_path)
    model = audio_pipe["model"]
    proba = model.predict_proba(X_final)[0, 1]
    return float(proba), int(proba >= 0.5)


def predict_meta_from_features(meta_vector: np.ndarray):
    pipes = load_pipelines()
    meta_pipe = pipes.get("meta")

    model = meta_pipe["model"]
    scaler = meta_pipe.get("scaler")
    selected_indices = np.array(meta_pipe["selected_indices"])

    X = np.array(meta_vector, dtype=float).reshape(1, -1)
    X = scaler.transform(X) if scaler else X
    X = X[:, selected_indices] if selected_indices is not None else X

    proba = model.predict_proba(X)[0, 1]
    return float(proba), int(proba >= 0.5)


# ---------------------------------------------------------------------
# Combined prediction + explainability + personalization
# ---------------------------------------------------------------------

def combined_prediction(option: int,
                        ecg_image_path: str | None = None,
                        meta_features: np.ndarray | None = None,
                        audio_wav_path: str | None = None):

    probs = []
    explanations = []
    modalities = []

    pipes = load_pipelines()

    if option in (1, 4, 5, 7) and ecg_image_path is not None:
        ecg_out = predict_ecg_from_image(ecg_image_path)
        probs.append(ecg_out["risk_prob"])
        modalities.append("ecg")
        explanations.append(
            explain_ecg(ecg_image_path, pipes["ecg"], ecg_out)
        )

    if option in (2, 4, 6, 7) and meta_features is not None:
        p_meta, _ = predict_meta_from_features(meta_features)
        probs.append(p_meta)
        modalities.append("meta")
        explanations.append(
            explain_meta(meta_features, pipes["meta"])
        )

    if option in (3, 5, 6, 7) and audio_wav_path is not None:
        p_audio, _ = predict_audio_from_wav(audio_wav_path)
        probs.append(p_audio)
        modalities.append("audio")
        explanations.append(
            explain_audio(audio_wav_path, pipes["audio"])
        )

    if not probs:
        raise RuntimeError("No modalities provided")

    avg_prob = float(sum(probs) / len(probs))
    risk_label = int(avg_prob >= 0.5)

    # =====================================================
    # 🔒 CRITICAL FIX: TRIM EXPLANATIONS BEFORE LLM
    # =====================================================
    MAX_EXPLANATIONS = 3
    MAX_CHARS_PER_EXPLANATION = 800

    trimmed_explanations = []
    for e in explanations[:MAX_EXPLANATIONS]:
        if isinstance(e, str):
            trimmed_explanations.append(e[:MAX_CHARS_PER_EXPLANATION])
        else:
            trimmed_explanations.append(str(e)[:MAX_CHARS_PER_EXPLANATION])

    llm = get_llm_engine()
    context = {
        "final_risk_probability": avg_prob,
        "final_risk_label": "At-risk" if risk_label == 1 else "Normal",
        "modalities_used": modalities,
        "explanations": explanations
    }
    personalized_text = llm.generate(context)

    return {
        "risk_prob": avg_prob,
        "risk_label": risk_label,
        "modalities_used": modalities,
        "modal_explanations": trimmed_explanations,
        "personalized_explanation": personalized_text
    }

