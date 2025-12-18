# run_demo_files.py

import os
import numpy as np

from inference_server_files import (
    combined_prediction,
)

def choose_option():
    print("\nWhich clients do you want to use?")
    print(" 1) ECG images only")
    print(" 2) Patient data only")
    print(" 3) Heartbeat audio only")
    print(" 4) ECG + Patient data")
    print(" 5) ECG + Audio")
    print(" 6) Heartbeat + Patient data")
    print(" 7) ECG + Patient data + Audio")

    while True:
        choice = input("Enter option number (1-7): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= 7:
            return int(choice)
        print("Please enter a valid option 1–7.")


def parse_meta_vector(prompt):
    raw = input(prompt).strip()
    if raw == "":
        return None
    try:
        vals = [float(x) for x in raw.split(",") if x.strip() != ""]
        arr = np.array(vals, dtype=float).reshape(1, -1)
        return arr
    except Exception as e:
        print(f"Could not parse numeric features: {e}")
        return None


def ask_for_ecg_path():
    path = input("Enter ECG image path (e.g. datasets/ECG Dataset/Normal Person/xx.png): ").strip()
    if path == "":
        return None
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    return path


def ask_for_audio_path():
    path = input("Enter heart sound .wav path (e.g. datasets/Heartbeats/set_a/xxx.wav): ").strip()
    if path == "":
        return None
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    return path


def main():
    print("\n=== File-based Multi-Modal Demo (Local Inference) ===")

    option = choose_option()

    ecg_path = None
    meta_vec = None
    audio_path = None

    # ECG?
    if option in (1, 4, 5, 7):
        print("\nECG modality selected.")
        ecg_path = ask_for_ecg_path()
        if ecg_path is None:
            print("No valid ECG image provided; aborting.")
            return

    # Meta?
    if option in (2, 4, 6, 7):
        print("\nPatient metadata modality selected.")
        meta_vec = parse_meta_vector(
            "Enter patient numeric features (same order as training CSV, excluding DEATH_EVENT), comma-separated: "
        )
        if meta_vec is None:
            print("No valid metadata provided; aborting.")
            return

    # Audio?
    if option in (3, 5, 6, 7):
        print("\nAudio modality selected.")
        audio_path = ask_for_audio_path()
        if audio_path is None:
            print("No valid audio file provided; aborting.")
            return

    print("\nComputing combined prediction...")
    result = combined_prediction(
        option=option,
        ecg_image_path=ecg_path,
        meta_features=meta_vec,
        audio_wav_path=audio_path,
    )

    avg_prob = result["risk_prob"]
    label = result["risk_label"]

    print("\n=== Result ===")
    print(f"Average probability of At-risk: {avg_prob:.4f}")
    print(f"Predicted class: {'At-risk' if label == 1 else 'Normal'}")

    print("\n=== Personalized Explanation ===")
    print(result["personalized_explanation"])

if __name__ == "__main__":
    main()
