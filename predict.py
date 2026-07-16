#!/usr/bin/env python3
"""predict.py — score new email text with the trained phishing model."""
import sys
import joblib

MODEL = "model.joblib"


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 predict.py "email text to score"'); sys.exit(2)
    text = " ".join(sys.argv[1:])
    try:
        pipe = joblib.load(MODEL)
    except FileNotFoundError:
        print("Model not found. Run: python3 train.py"); sys.exit(1)
    prob = pipe.predict_proba([text])[0][1]
    label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
    print(f"Prediction: {label}")
    print(f"Phishing probability: {prob:.1%}")


if __name__ == "__main__":
    main()
