"""
PhishGuard AI — Prediction Pipeline
=====================================
Member 4 (Backend) deliverable.

Provides three public interfaces consumed by the Streamlit app (app.py):
    - extract_features(url: str) -> dict
    - load_artifacts()            -> (model, scaler)
    - predict(url: str)           -> PredictionResult

Design principles
-----------------
* Feature extraction is an exact mirror of M3's training implementation.
  Any deviation here causes silent data-skew at inference time.
* Artifacts (model + scaler) are loaded once at import time via a module-level
  cache, not on every prediction call.
* All public functions carry full type hints and docstrings.
* Errors surface as typed exceptions so the app can give the user meaningful
  feedback instead of a raw traceback.
"""

from __future__ import annotations

import logging
import math
import re
import socket
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phishguard.utils")

# ---------------------------------------------------------------------------
# Paths  (relative so they work on any machine and on Streamlit Cloud)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
MODELS_DIR = _ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Feature column order must be identical to M3's training pipeline.
FEATURE_COLUMNS: list[str] = [
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

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class PhishGuardError(Exception):
    """Base exception for all PhishGuard backend errors."""


class ArtifactNotFoundError(PhishGuardError):
    """Raised when model.pkl or scaler.pkl cannot be located."""


class InvalidURLError(PhishGuardError):
    """Raised when the input string cannot be parsed as a URL."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionResult:
    """
    Immutable result object returned by :func:`predict`.

    Attributes
    ----------
    label : str
        ``"Phishing"`` or ``"Legitimate"``.
    is_phishing : bool
        True when the model predicts phishing.
    confidence : float
        Probability (0–1) for the predicted class, rounded to 4 dp.
    features : dict[str, float]
        All 11 extracted feature values, useful for the explainability panel.
    url : str
        The original URL that was analysed.
    """

    label: str
    is_phishing: bool
    confidence: float
    features: dict[str, float] = field(default_factory=dict)
    url: str = ""

    def __str__(self) -> str:
        return (
            f"PredictionResult(label={self.label!r}, "
            f"confidence={self.confidence:.2%}, url={self.url!r})"
        )


# ---------------------------------------------------------------------------
# Artifact cache  (module-level singleton, loaded once)
# ---------------------------------------------------------------------------
_CACHE: dict[str, object] = {}


def load_artifacts(
    model_path: Path | str = MODEL_PATH,
    scaler_path: Path | str = SCALER_PATH,
    *,
    force_reload: bool = False,
) -> tuple[object, object]:
    """
    Load model and scaler from disk, caching the result in memory.

    The cache is keyed by resolved file paths so that calls with different
    paths still work correctly (useful in tests).

    Parameters
    ----------
    model_path  : path to model.pkl
    scaler_path : path to scaler.pkl
    force_reload: bypass cache and reload from disk

    Returns
    -------
    (model, scaler) tuple

    Raises
    ------
    ArtifactNotFoundError  if either file does not exist.
    PhishGuardError         if joblib fails to deserialise either file.
    """
    model_path = Path(model_path).resolve()
    scaler_path = Path(scaler_path).resolve()
    cache_key = f"{model_path}::{scaler_path}"

    if not force_reload and cache_key in _CACHE:
        logger.debug("Artifacts served from cache.")
        return _CACHE[cache_key]  # type: ignore[return-value]

    if not model_path.exists():
        raise ArtifactNotFoundError(
            f"Model file not found: {model_path}\n"
            "Ask M3 (AI Engineer) to save model.pkl to the models/ folder."
        )
    if not scaler_path.exists():
        raise ArtifactNotFoundError(
            f"Scaler file not found: {scaler_path}\n"
            "Ask M3 (AI Engineer) to save scaler.pkl to the models/ folder."
        )

    try:
        model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
    except Exception as exc:
        raise PhishGuardError(f"Failed to load model: {exc}") from exc

    try:
        scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded from %s", scaler_path)
    except Exception as exc:
        raise PhishGuardError(f"Failed to load scaler: {exc}") from exc

    _CACHE[cache_key] = (model, scaler)
    return model, scaler


# ---------------------------------------------------------------------------
# Feature engineering  (must mirror M3's implementation exactly)
# ---------------------------------------------------------------------------

# Regex compiled once at import time for efficiency.
_IP_PATTERN = re.compile(
    r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
)


def _shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.

    H = -Σ p(c) * log2(p(c))

    Higher entropy → more random character distribution → stronger phishing signal.
    Returns 0.0 for an empty string.
    """
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in freq.values())


def _count_subdomains(hostname: str) -> int:
    """
    Count the number of subdomain levels for a given hostname.

    We subtract 2 to account for the registered domain and TLD, so
    ``www.paypal.com`` has 1 subdomain (``www``),
    ``secure.paypal.verify.attacker.com`` has 3.
    Returns 0 when the count would be negative (bare domains / IPs).
    """
    if not hostname:
        return 0
    parts = hostname.split(".")
    return max(0, len(parts) - 2)


def _normalise_url(url: str) -> str:
    """
    Ensure the URL has a scheme so urlparse can split it correctly.
    If no scheme is present, ``http://`` is prepended.
    """
    url = url.strip()
    if not url:
        return url
    if "://" not in url:
        url = "http://" + url
    return url


def extract_features(url: str) -> dict[str, float]:
    """
    Extract the 11 URL-based features used during model training.

    This function is the single source of truth for feature extraction.
    M4 (utils.py) and M3 (training notebooks) must be kept in sync.

    Parameters
    ----------
    url : str
        Raw URL string as entered by the user.

    Returns
    -------
    dict[str, float]
        Mapping of feature name → numeric value.
        Keys are always present and follow the order in ``FEATURE_COLUMNS``.

    Raises
    ------
    InvalidURLError  if the URL is empty or cannot be parsed at all.
    """
    if not url or not url.strip():
        raise InvalidURLError("URL must not be empty.")

    url = url.strip()
    normalised = _normalise_url(url)

    try:
        parsed = urlparse(normalised)
    except ValueError as exc:
        raise InvalidURLError(f"Could not parse URL '{url}': {exc}") from exc

    hostname: str = parsed.hostname or ""
    full_url: str = normalised  # use normalised form for consistent length counts

    # ------------------------------------------------------------------ #
    # 1. url_length — total character count of the original URL
    # ------------------------------------------------------------------ #
    url_length = float(len(url))

    # ------------------------------------------------------------------ #
    # 2. num_dots — count of '.' characters
    # ------------------------------------------------------------------ #
    num_dots = float(url.count("."))

    # ------------------------------------------------------------------ #
    # 3. num_hyphens — count of '-' characters
    # ------------------------------------------------------------------ #
    num_hyphens = float(url.count("-"))

    # ------------------------------------------------------------------ #
    # 4. num_slashes — count of '/' characters (path separators)
    # ------------------------------------------------------------------ #
    num_slashes = float(url.count("/"))

    # ------------------------------------------------------------------ #
    # 5. has_at_symbol — '@' redirects browsers; strong phishing signal
    # ------------------------------------------------------------------ #
    has_at_symbol = 1.0 if "@" in url else 0.0

    # ------------------------------------------------------------------ #
    # 6. has_https — HTTPS absence is a weak-but-supporting phishing signal
    # ------------------------------------------------------------------ #
    has_https = 1.0 if parsed.scheme.lower() == "https" else 0.0

    # ------------------------------------------------------------------ #
    # 7. has_ip_address — IP addresses in URLs are almost always malicious
    # ------------------------------------------------------------------ #
    has_ip_address = 1.0 if _IP_PATTERN.search(hostname) else 0.0

    # ------------------------------------------------------------------ #
    # 8. num_subdomains — excessive subdomains mimic legitimacy
    # ------------------------------------------------------------------ #
    num_subdomains = float(_count_subdomains(hostname))

    # ------------------------------------------------------------------ #
    # 9. digit_ratio — auto-generated phishing URLs contain more random digits
    # ------------------------------------------------------------------ #
    digit_count = sum(1 for ch in url if ch.isdigit())
    digit_ratio = digit_count / len(url) if len(url) > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 10. url_entropy — Shannon entropy; higher = more random = more phishing
    # ------------------------------------------------------------------ #
    url_entropy = _shannon_entropy(url)

    # ------------------------------------------------------------------ #
    # 11. domain_length — unusually long domains are a common phishing pattern
    # ------------------------------------------------------------------ #
    domain_length = float(len(hostname))

    features: dict[str, float] = {
        "url_length": url_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_slashes": num_slashes,
        "has_at_symbol": has_at_symbol,
        "has_https": has_https,
        "has_ip_address": has_ip_address,
        "num_subdomains": num_subdomains,
        "digit_ratio": round(digit_ratio, 6),
        "url_entropy": round(url_entropy, 6),
        "domain_length": domain_length,
    }

    logger.debug("Features extracted for '%s': %s", url, features)
    return features


def _features_to_array(features: dict[str, float]) -> np.ndarray:
    """
    Convert a feature dict to a 2-D numpy array in column order.

    The column order must match FEATURE_COLUMNS exactly, which is the order
    used when training the scaler and model.
    """
    row = [features[col] for col in FEATURE_COLUMNS]
    return np.array(row, dtype=np.float64).reshape(1, -1)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    url: str,
    model=None,
    scaler=None,
) -> PredictionResult:
    """
    End-to-end prediction for a single URL.

    1. Extract features from the raw URL string.
    2. Scale the feature vector with the pre-fitted scaler.
    3. Run the model to get a class prediction and probability.
    4. Return a :class:`PredictionResult`.

    Parameters
    ----------
    url    : Raw URL string entered by the user.
    model  : Optional pre-loaded model (skips disk I/O, useful in tests).
    scaler : Optional pre-loaded scaler (skips disk I/O, useful in tests).

    Returns
    -------
    PredictionResult

    Raises
    ------
    InvalidURLError        on malformed/empty URL.
    ArtifactNotFoundError  if model/scaler files are missing.
    PhishGuardError        on any other backend error.
    """
    # Load artifacts if not provided (uses module-level cache)
    if model is None or scaler is None:
        model, scaler = load_artifacts()

    # Feature extraction
    features = extract_features(url)

    # Build the feature matrix and scale
    X_raw = _features_to_array(features)
    try:
        X_scaled = scaler.transform(X_raw)
    except Exception as exc:
        raise PhishGuardError(f"Scaler transform failed: {exc}") from exc

    # Predict
    try:
        prediction: int = int(model.predict(X_scaled)[0])
        probabilities: np.ndarray = model.predict_proba(X_scaled)[0]
    except Exception as exc:
        raise PhishGuardError(f"Model inference failed: {exc}") from exc

    # prediction: 1 = phishing, 0 = legitimate  (PhiUSIIL dataset convention)
    is_phishing = bool(prediction == 1)
    label = "Phishing" if is_phishing else "Legitimate"
    confidence = round(float(probabilities[prediction]), 4)

    result = PredictionResult(
        label=label,
        is_phishing=is_phishing,
        confidence=confidence,
        features=features,
        url=url,
    )

    logger.info(
        "Prediction → %s (confidence=%.2f%%) for URL: %s",
        label,
        confidence * 100,
        url,
    )
    return result


# ---------------------------------------------------------------------------
# Convenience helpers exposed to the app
# ---------------------------------------------------------------------------

def feature_display_names() -> dict[str, str]:
    """
    Human-readable labels for each feature, used in the Streamlit UI.
    """
    return {
        "url_length": "URL Length (chars)",
        "num_dots": "Number of Dots",
        "num_hyphens": "Number of Hyphens",
        "num_slashes": "Number of Slashes",
        "has_at_symbol": "Contains '@' Symbol",
        "has_https": "Uses HTTPS",
        "has_ip_address": "Contains IP Address",
        "num_subdomains": "Number of Subdomains",
        "digit_ratio": "Digit Ratio",
        "url_entropy": "URL Entropy (Shannon)",
        "domain_length": "Domain Length (chars)",
    }


def feature_security_notes() -> dict[str, str]:
    """
    One-sentence security interpretation for each feature.
    Displayed in the expandable 'Why?' panel in the app.
    """
    return {
        "url_length": "Phishing URLs are significantly longer on average to obscure the real destination.",
        "num_dots": "More dots indicate more subdomains — a common tactic to mimic trusted brands.",
        "num_hyphens": "Hyphens are used to mimic trusted brands, e.g. paypal-secure-login.com.",
        "num_slashes": "Deep URL paths can obscure a malicious destination behind a legitimate-looking prefix.",
        "has_at_symbol": "The '@' character causes browsers to ignore everything before it — a strong phishing signal.",
        "has_https": "Absence of HTTPS is a weak but supporting phishing indicator.",
        "has_ip_address": "Legitimate services almost never use raw IP addresses in their URLs.",
        "num_subdomains": "Attackers stack subdomains to mimic legitimacy, e.g. secure.paypal.verify.attacker.com.",
        "digit_ratio": "Auto-generated phishing URLs contain more random digits relative to their length.",
        "url_entropy": "Higher Shannon entropy means a more random character distribution — a phishing hallmark.",
        "domain_length": "Unusually long domain names are a common characteristic of phishing domains.",
    }


# ---------------------------------------------------------------------------
# CLI  (python utils.py <url>  — quick sanity check without the app)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utils.py <url>")
        sys.exit(1)

    test_url = sys.argv[1]
    print(f"\nAnalysing: {test_url}\n")

    try:
        feats = extract_features(test_url)
        print("Extracted features:")
        names = feature_display_names()
        for col in FEATURE_COLUMNS:
            print(f"  {names[col]:<30} {feats[col]}")

        result = predict(test_url)
        print(f"\nResult : {result.label}")
        print(f"Confidence : {result.confidence:.2%}")

    except ArtifactNotFoundError as e:
        print(f"\n[ArtifactNotFoundError] {e}")
        print("Running feature extraction only (model not available).")
        feats = extract_features(test_url)
        names = feature_display_names()
        print("\nExtracted features:")
        for col in FEATURE_COLUMNS:
            print(f"  {names[col]:<30} {feats[col]}")

    except InvalidURLError as e:
        print(f"\n[InvalidURLError] {e}")
        sys.exit(1)
