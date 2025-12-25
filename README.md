

# Distributed Multi-Task Federated Learning for Cardiovascular Risk Assessment (for the elderly)

## Overview

A distributed, multi-task, multi-modal federated learning system designed to assess cardiovascular disease risk in older populations while strictly preserving data privacy. The system enables collaborative learning across multiple independent clients without sharing raw data, intermediate representations, or patient identifiers.

The framework consists of three modality-specific clients—ECG images, structured patient metadata, and heartbeat (PCG) audio—coordinated by a central federated server. Each client trains locally on its private dataset and exchanges only encrypted model pipelines with the server.

In addition to prediction, the system includes a privacy-preserving feedback and personalization layer. This layer converts federated model outputs and explainability signals into patient-level preventive and monitoring-oriented feedback using a locally deployed large language model. Raw data and feature values are never exposed to the LLM.

---

## System Architecture

The system follows a distributed client–server architecture:

* Independent federated clients for ECG images, patient metadata, and heartbeat audio
* A central federated server that orchestrates training rounds and coordinates encrypted pipeline exchange
* A local inference and feedback engine that performs decision-level aggregation, explainability synthesis, and personalized feedback generation

All components are designed to operate across heterogeneous devices and environments.

---

## Modalities and Learning Tasks

### ECG Image Client

* Input: ECG images
* Feature extraction:

  * Deep features using InceptionV3 and ResNet50
  * Handcrafted statistical, texture, and morphological features
* Feature selection:

  * Infinite Feature Selection (InfFS)
  * Fuzzy logic–based scoring
* Model:

  * XGBoost multi-class classifier
* Classes:

  * Normal Person
  * Abnormal Heartbeat
  * History of Myocardial Infarction
* Derived binary risk:

  * At-risk if predicted class is not Normal
* Explainability:

  * Grad-CAM applied to CNN feature extractors

---

### Patient Metadata Client

* Input: Structured numerical clinical data
* Preprocessing:

  * Standardization using StandardScaler
* Feature selection:

  * InfFS combined with fuzzy inference
* Model:

  * XGBoost binary classifier
* Output:

  * Probability of cardiovascular risk
* Explainability:

  * Feature-importance–based summaries derived from the trained model

---

### Heartbeat (PCG Audio) Client

* Input: Heart sound recordings in WAV format
* Feature extraction:

  * MFCCs
  * Chroma features
  * Zero-crossing rate
  * Spectral centroid
* Feature selection:

  * Mutual information–based ranking
* Model:

  * XGBoost multi-class classifier
* Classes:

  * Normal
  * Murmur
  * Extra heart sounds
  * Extrasystole
* Derived binary risk:

  * At-risk if predicted class is not Normal
* Class balancing:

  * SMOTE applied during local training
* Explainability:

  * Feature-importance–based summaries

---

## Federated Learning Strategy

The system implements a multi-task federated learning strategy where clients do not share patients, samples, or feature spaces. Each client independently trains on its local dataset and participates in federated rounds coordinated by a central server.

### Shared Model Paradigm

To retain a core characteristic of traditional federated learning, the same underlying model family is used across all clients. All three clients employ XGBoost-based models, ensuring algorithmic consistency while allowing modality-specific feature representations.

Key aspects of this design include:

* A shared learning algorithm across all clients
* Consistent model architecture and optimization paradigm
* Modality-specific feature extraction and feature selection
* Warm-starting of models across federated rounds using previously trained pipelines

Although the system does not perform parameter averaging or assume shared data distributions, maintaining a shared model paradigm preserves a fundamental federated learning characteristic while enabling heterogeneous, multi-modal collaboration.

---

## Security and Privacy

### Encrypted Client–Server Communication

All communication between clients and the federated server is secured using symmetric-key encryption based on the Fernet scheme.

Encrypted objects include:

* Trained model objects
* Feature selectors and indices
* Scalers and preprocessing components

Raw data, intermediate features, and patient identifiers never leave the client environment.

---

### Feedback and Personalization System

The system includes a feedback and personalization layer built on a locally deployed large language model.

This feedback system:

* Consumes only model-derived outputs:

  * Aggregated risk probabilities
  * Final risk labels
  * Abstracted explainability summaries
* Never accesses:

  * Raw ECG images
  * Raw heartbeat audio
  * Numerical feature values
  * Patient identifiers

The LLM generates structured, preventive, and monitoring-oriented feedback intended for clinical oversight or decision-support systems. It does not provide diagnoses, speculative reasoning, or raw-signal interpretation.

Strict prompt constraints are used to prevent hallucination and ensure the feedback remains grounded in federated model outputs.

---

## Explainability Framework

Explainability is integrated at the modality level:

* ECG images:

  * Grad-CAM–based qualitative interpretation of CNN feature activations
* Metadata and audio:

  * Feature-importance–based summaries derived from XGBoost models

Explainability outputs are used both for transparency and as structured inputs to the feedback system.

---

## Inference and Multi-Modal Aggregation

At inference time, the system supports flexible modality combinations:

* ECG only
* Metadata only
* Audio only
* Any pairwise combination
* All three modalities together

Risk probabilities from selected modalities are aggregated at the decision level using averaging. The final risk label is derived from the combined probability.

---

## Repository Structure

```
CVD_new3/
├── clients/
│   ├── __init__.py
│   ├── client1_image.py
│   ├── client2_numerical.py
│   ├── client3_audio.py
│   ├── client1_node.py
│   ├── client2_node.py
│   └── client3_node.py
│
├── datasets/
│   ├── ECG Dataset/
│   │   ├── Abnormal Heartbeat/
│   │   ├── History of MI/
│   │   └── Normal Person/
│   ├── Heartbeats/
│   │   ├── combined.csv
│   │   ├── set_a/
│   │   └── set_b/
│   └── heart_failure.csv
│
├── explainability/
│   ├── __init__.py
│   ├── gradcam_ecg.py
│   ├── ecg_explainer.py
│   ├── meta_explainer.py
│   └── audio_explainer.py
│
├── llm/
│   ├── __init__.py
│   ├── personalization_engine.py
│   └── prompts.py
│
├── models/
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
│
├── utils/
│   └── crypto_utils.py
│
├── federated_server_distributed.py
├── inference_server_files.py
├── run_demo_files.py
└── README.md
```

---

## Running the System

### Start Client Nodes

Each client runs as an independent FastAPI service.

```
python -m clients.client1_node
python -m clients.client2_node
python -m clients.client3_node
```

---

### Run Federated Training

```
python federated_server_distributed.py
```

This executes multiple federated rounds and saves encrypted pipelines and evaluation metrics.

---

### Run Inference and Feedback Demo

```
python run_demo_files.py
```

The demo allows selection of different modality combinations and outputs the final risk prediction along with personalized feedback.

---

## Intended Use and Scope

This project is a research-oriented prototype intended for:

* Privacy-preserving clinical AI research
* Multi-task federated learning with heterogeneous data
* Explainability-driven decision support
* Personalized feedback generation under strict privacy constraints

The system is not intended for clinical diagnosis or direct medical decision-making.

---

## License

For academic and research use only.

---

* or produce a viva/interview-ready explanation aligned exactly to this README
