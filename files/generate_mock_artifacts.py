"""
PhishGuard AI — Mock Artifact Generator
==========================================
Generates placeholder model.pkl and scaler.pkl so M5 (Frontend) can build
and test the Streamlit app before M3 hands off the real trained model.

The mock model is a LogisticRegression trained on synthetically-biased data
that mimics expected phishing patterns. Its accuracy is not representative —
it exists only to provide a valid sklearn API surface for the app.

Usage
-----
    python3 generate_mock_artifacts.py

Outputs
-------
    models/model.pkl   — pickled LogisticRegression
    models/scaler.pkl  — pickled StandardScaler (fitted on same synthetic data)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

# ── Feature column order (must match utils.FEATURE_COLUMNS exactly) ─────────
FEATURE_COLUMNS = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_slashes",
    "has_at_symbol",
    "has_https",
    "has_ip_address",
    "num_subdomains",
    "digit_ratio",
    "url_entropy",
    "domain_length",
]

N_FEATURES = len(FEATURE_COLUMNS)


def _generate_synthetic_data(n_samples: int = 2000, seed: int = 42):
    """
    Create synthetic URL feature data with realistic phishing/legitimate splits.

    Phishing URLs tend to have:
        - longer URLs (url_length ~120)
        - more hyphens
        - @ symbols (has_at_symbol = 1)
        - no HTTPS (has_https = 0)
        - IP addresses (has_ip_address = 1 occasionally)
        - more subdomains
        - higher digit ratio
        - higher entropy

    Legitimate URLs tend to have:
        - shorter URLs (~40 chars)
        - fewer hyphens
        - HTTPS (has_https = 1)
        - 1 subdomain (www)
        - lower entropy
    """
    rng = np.random.default_rng(seed)
    half = n_samples // 2

    # ----- Legitimate URLs (label = 0) -----
    legit = np.column_stack([
        rng.normal(45, 15, half).clip(10, 150),     # url_length
        rng.poisson(2, half).clip(0, 8),             # num_dots
        rng.poisson(0.3, half).clip(0, 5),           # num_hyphens
        rng.poisson(3, half).clip(1, 10),            # num_slashes
        rng.binomial(1, 0.01, half),                 # has_at_symbol
        rng.binomial(1, 0.92, half),                 # has_https
        rng.binomial(1, 0.005, half),                # has_ip_address
        rng.poisson(1, half).clip(0, 3),             # num_subdomains
        rng.beta(1, 10, half),                       # digit_ratio
        rng.normal(3.5, 0.4, half).clip(2.5, 4.5),  # url_entropy
        rng.normal(12, 4, half).clip(4, 30),         # domain_length
    ]).astype(np.float64)
    legit_labels = np.zeros(half, dtype=int)

    # ----- Phishing URLs (label = 1) -----
    phish = np.column_stack([
        rng.normal(110, 30, half).clip(30, 300),     # url_length
        rng.poisson(5, half).clip(0, 15),            # num_dots
        rng.poisson(3, half).clip(0, 12),            # num_hyphens
        rng.poisson(5, half).clip(1, 20),            # num_slashes
        rng.binomial(1, 0.25, half),                 # has_at_symbol
        rng.binomial(1, 0.30, half),                 # has_https
        rng.binomial(1, 0.15, half),                 # has_ip_address
        rng.poisson(4, half).clip(0, 10),            # num_subdomains
        rng.beta(3, 5, half),                        # digit_ratio
        rng.normal(4.4, 0.5, half).clip(3.0, 5.5),  # url_entropy
        rng.normal(25, 8, half).clip(8, 60),         # domain_length
    ]).astype(np.float64)
    phish_labels = np.ones(half, dtype=int)

    X = np.vstack([legit, phish])
    y = np.concatenate([legit_labels, phish_labels])

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def generate_mock_artifacts(output_dir: Path = MODELS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic training data …")
    X, y = _generate_synthetic_data(n_samples=2000)

    print("Fitting StandardScaler …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training mock LogisticRegression …")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        random_state=42,
        C=1.0,
    )
    model.fit(X_scaled, y)

    # Quick sanity check
    preds = model.predict(X_scaled)
    acc = (preds == y).mean()
    print(f"Mock model training accuracy: {acc:.2%}  (synthetic data only — not real performance)")

    model_path = output_dir / "model.pkl"
    scaler_path = output_dir / "scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n✓  model.pkl  → {model_path}")
    print(f"✓  scaler.pkl → {scaler_path}")
    print("\nNOTE: Replace these files with M3's trained artifacts before final deployment.")


if __name__ == "__main__":
    generate_mock_artifacts()
