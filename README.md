# AI Phishing Email Detector

A machine-learning classifier that flags phishing emails from their text — the AI + security
project in this portfolio. It demonstrates the full applied-ML workflow: labeled data,
feature engineering (TF-IDF), model training and evaluation, and inference on new emails.

## Why this project

Phishing is the entry point for a large share of breaches (see the incident-response case
study in this portfolio, where phishing awareness was a core recommendation). This project
takes that defensive need and implements a concrete, measurable control: an ML model that
scores an email's phishing likelihood, the kind of component that sits inside a secure email
gateway.

## How it works

1. **Data** — `data/emails.csv`, a labeled set of `text,label` rows (`1` = phishing,
   `0` = legitimate). Synthetic, built to include common phishing signals (urgency, credential
   requests, spoofed links, prize/lottery lures) and realistic legitimate mail.
2. **Features** — TF-IDF vectorization over word/bigram tokens (`scikit-learn`).
3. **Model** — Logistic Regression (interpretable, strong text baseline).
4. **Evaluation** — train/test split with accuracy, precision, recall, F1, and a confusion
   matrix. In security, **recall on the phishing class matters most** (missing a phish is worse
   than a false alarm) — the report calls this out.
5. **Inference** — `predict.py` scores arbitrary new email text.

## Setup

```bash
pip install -r requirements.txt
```

## Train and evaluate

```bash
python3 train.py
```

This trains the model, prints a full classification report and confusion matrix, shows the
top phishing-indicative terms the model learned, and saves the model to `model.joblib`.

## Score a new email

```bash
python3 predict.py "Urgent: your account is locked. Verify your password at http://secure-login.example to avoid suspension."
```

Output: predicted label + phishing probability.

## Sample results

See [`sample_output.txt`](./sample_output.txt) for a full run. Note: the model scores near-perfect
on this dataset because the synthetic templates are cleanly separable — that is expected for
demonstration data, **not** a claim of real-world performance. See limitations below.

## What this demonstrates

Applied ML for security · text feature engineering (TF-IDF) · model training & evaluation ·
choosing the right metric for the threat model (recall-weighted) · model interpretability
(top learned features) · packaging a model for inference. A natural bridge between the
cybersecurity work and AI/ML.

## Honest limitations

Trained on a small synthetic dataset for demonstration — not production-ready. A real
deployment would use large labeled corpora, header/URL/attachment features, adversarial
testing, and continuous retraining as attacker tactics shift.
