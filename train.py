#!/usr/bin/env python3
"""
train.py — train and evaluate a phishing-email classifier.

Pipeline: TF-IDF (word + bigram) -> Logistic Regression.
Prints accuracy, a full classification report, a confusion matrix, and the
terms most indicative of phishing. Saves the fitted pipeline to model.joblib.
"""
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA = "data/emails.csv"
MODEL = "model.joblib"


def load(path):
    X, y = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            X.append(r["text"]); y.append(int(r["label"]))
    return X, y


def main():
    X, y = load(DATA)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    print(f"Samples: {len(X)} ({sum(y)} phishing / {len(y)-sum(y)} legit)")
    print(f"Test accuracy: {accuracy_score(yte, pred):.3f}\n")
    print("Classification report (class 1 = phishing):")
    print(classification_report(yte, pred, target_names=["legit", "phishing"]))
    print("Confusion matrix [rows=true, cols=pred] (order: legit, phishing):")
    print(confusion_matrix(yte, pred))

    # top phishing-indicative terms
    vec = pipe.named_steps["tfidf"]; clf = pipe.named_steps["clf"]
    names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top = sorted(zip(coefs, names), reverse=True)[:12]
    print("\nTop phishing-indicative terms:")
    print("  " + ", ".join(f"{t}" for _, t in top))

    joblib.dump(pipe, MODEL)
    print(f"\nSaved model -> {MODEL}")


if __name__ == "__main__":
    main()
